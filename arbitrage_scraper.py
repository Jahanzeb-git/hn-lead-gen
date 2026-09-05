"""
arbitrage_scraper.py — Worldwide Remote Job Arbitrage Agent

Runs as a separate GitHub Actions job, targeting companies that
explicitly hire worldwide (open to Pakistan) and match Jahanzeb's
backend / AI infrastructure skillset.

Sources (all free, no API key required):
  - Remotive API:  https://remotive.com/api/remote-jobs
  - Jobicy API:    https://jobicy.com/api/v2/remote-jobs

Pipeline:
  1. Pull jobs from both sources.
  2. Hard-filter: candidate_required_location must be Worldwide/Anywhere/Global
     or unspecified (i.e. company never restricted it — best case for us).
  3. Hard-filter: category must be software/backend/devops/AI — removes
     sales, design, marketing noise before any LLM call.
  4. Dedup against B2 state (same pattern as lead_scraper.py).
  5. Batch-score survivors with DeepSeek — same model, new system prompt
     tuned for full-time/long-term remote fit, not just contract/freelance.
  6. Qualifying leads (score >= 6) get stored to B2 and a Discord alert fires.

Discord message format uses 🌍 prefix to distinguish from 🎯 HN leads.

Required env vars (reuse existing GitHub Actions secrets):
  DEEPSEEK_API_KEY
  DEEPSEEK_API_URL    (default: dashscope-intl aliyun endpoint)
  DEEPSEEK_MODEL
  DISCORD_WEBHOOK_URL
  B2_KEY_ID
  B2_APPLICATION_KEY
  B2_BUCKET_NAME
  B2_ENDPOINT
  ARBITRAGE_OBJECT_KEY  (default: arbitrage_pipeline.json — separate file from HN)
"""

import datetime
import json
import logging
import os
import re
import sys
import time

import boto3
import requests
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("arbitrage_scraper")

# ── Config ──────────────────────────────────────────────────────────────────
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "hnscraper")
ARBITRAGE_OBJECT_KEY = os.environ.get("ARBITRAGE_OBJECT_KEY", "arbitrage_pipeline.json")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro-0813")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None

MIN_QUALIFY_SCORE = int(os.environ.get("MIN_QUALIFY_SCORE", 6))

_session = requests.Session()
_session.headers.update({"User-Agent": "Arbitrage Lead Agent (contact: jahanzeb-git)"})

# ── Location allow-list ──────────────────────────────────────────────────────
# If any of these substrings appear in the location field, the job is open
# to Pakistan-based engineers. Empty location also passes (no restriction stated).
GLOBAL_LOCATION_SIGNALS = [
    "worldwide", "anywhere", "global", "remote", "international",
    "work from anywhere", "all countries", "asia", "pakistan",
    "",  # blank = no restriction stated — passes
]

# ── Category allow-list (Remotive category slugs + Jobicy tags) ──────────────
ALLOWED_CATEGORIES = {
    "software-dev", "devops-sysadmin", "data", "product", "backend",
    "engineering", "ai", "machine-learning", "python", "developer",
    "software", "tech",
}

# ── Cheap regex pre-filter ───────────────────────────────────────────────────
_TECH_RE = re.compile(
    r"\b(?:python|fastapi|django|flask|backend|api|docker|kubernetes|k8s"
    r"|postgres|redis|celery|async|asyncio|aiohttp|llm|gpt|ai|ml|agent"
    r"|langchain|langgraph|openai|anthropic|vector|embedding|rag"
    r"|cloud|aws|gcp|azure|fly\.io|linux|devops|data\s*pipeline"
    r"|pydantic|sqlalchemy|microservice|grpc|kafka|airflow)\b",
    re.IGNORECASE,
)
MIN_TECH_HITS = 2


def passes_regex(text: str) -> bool:
    if not text:
        return False
    hits = set(m.lower() for m in _TECH_RE.findall(text))
    return len(hits) >= MIN_TECH_HITS


def location_is_global(location_str: str) -> bool:
    loc = (location_str or "").lower().strip()
    return any(sig in loc for sig in GLOBAL_LOCATION_SIGNALS)


def category_is_relevant(cat: str) -> bool:
    return any(kw in (cat or "").lower() for kw in ALLOWED_CATEGORIES)


# ── Source A: Remotive ───────────────────────────────────────────────────────
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
# All software-dev categories we care about — one request each because
# Remotive's API filters by a single category slug per call.
REMOTIVE_CATEGORIES = [
    "software-dev",
    "devops-sysadmin",
    "data",
    "product",
]


def fetch_remotive_jobs() -> list:
    jobs = []
    for cat in REMOTIVE_CATEGORIES:
        try:
            resp = _session.get(REMOTIVE_URL, params={"category": cat}, timeout=20)
            resp.raise_for_status()
            batch = resp.json().get("jobs", [])
            logger.info("Remotive [%s]: %d jobs returned", cat, len(batch))
            jobs.extend(batch)
        except Exception as e:
            logger.warning("Remotive fetch failed for category %s: %s", cat, e)
    return jobs


def normalize_remotive(job: dict) -> dict:
    """Normalize a Remotive job dict to our internal schema."""
    return {
        "id": f"remotive_{job['id']}",
        "source": "remotive",
        "title": job.get("title", ""),
        "company": job.get("company_name", ""),
        "location": job.get("candidate_required_location", ""),
        "salary": job.get("salary", ""),
        "url": job.get("url", ""),
        "category": job.get("category", ""),
        "description": re.sub(r"<[^>]+>", " ", job.get("description", "")),  # strip HTML
        "posted_at": job.get("publication_date", ""),
    }


# ── Source B: Jobicy ─────────────────────────────────────────────────────────
JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"
# Jobicy supports tag-based search and returns up to 50 results per call.
# We query multiple tags to maximise volume.
JOBICY_TAGS = ["python", "backend", "ai", "devops", "data-engineering"]


def fetch_jobicy_jobs() -> list:
    jobs = []
    for tag in JOBICY_TAGS:
        try:
            resp = _session.get(
                JOBICY_URL,
                params={"count": 50, "tag": tag},
                timeout=20,
            )
            resp.raise_for_status()
            batch = resp.json().get("jobs", [])
            logger.info("Jobicy [%s]: %d jobs returned", tag, len(batch))
            jobs.extend(batch)
        except Exception as e:
            logger.warning("Jobicy fetch failed for tag %s: %s", tag, e)
    return jobs


def normalize_jobicy(job: dict) -> dict:
    region = job.get("jobGeo", "") or ""
    description = re.sub(r"<[^>]+>", " ", job.get("jobDescription", ""))
    return {
        "id": f"jobicy_{job['id']}",
        "source": "jobicy",
        "title": job.get("jobTitle", ""),
        "company": job.get("companyName", ""),
        "location": region,
        "salary": job.get("annualSalaryMin", ""),
        "url": job.get("url", ""),
        "category": job.get("jobType", ""),
        "description": description,
        "posted_at": job.get("pubDate", ""),
    }


# ── Aggregation + pre-filtering ──────────────────────────────────────────────
def collect_all_jobs(seen_ids: set) -> list:
    raw_remotive = fetch_remotive_jobs()
    raw_jobicy = fetch_jobicy_jobs()

    candidates = []

    for job in raw_remotive:
        norm = normalize_remotive(job)
        if norm["id"] in seen_ids:
            continue
        if not location_is_global(norm["location"]):
            seen_ids.add(norm["id"])  # permanently skip — location blocked
            continue
        full_text = f"{norm['title']} {norm['description']}"
        if not passes_regex(full_text):
            seen_ids.add(norm["id"])
            continue
        candidates.append(norm)

    for job in raw_jobicy:
        norm = normalize_jobicy(job)
        if norm["id"] in seen_ids:
            continue
        if not location_is_global(norm["location"]):
            seen_ids.add(norm["id"])
            continue
        full_text = f"{norm['title']} {norm['description']}"
        if not passes_regex(full_text):
            seen_ids.add(norm["id"])
            continue
        candidates.append(norm)

    # Deduplicate across sources by (company + title) to avoid same job
    # appearing on both Remotive and Jobicy.
    seen_pairs = set()
    deduped = []
    for c in candidates:
        key = (c["company"].lower().strip(), c["title"].lower().strip())
        if key in seen_pairs:
            seen_ids.add(c["id"])
            continue
        seen_pairs.add(key)
        deduped.append(c)

    logger.info(
        "%d candidates passed location + regex + cross-source dedup.",
        len(deduped),
    )
    return deduped


# ── LLM scoring ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a job-fit scoring agent working exclusively for one specific engineer:
Jahanzeb Ahmed, based in Karachi, Pakistan.

## Who Jahanzeb Is
Self-taught. No formal CS degree. No prior corporate employment history.
He has built and shipped production-grade, complex systems:

- Author of `codepilot-ai` — an autonomous agent runtime (Python, Docker,
  Firecracker MicroVMs, FastAPI, AsyncIO, custom LLM tool-call protocols,
  NDJSON over Unix sockets). Not a tutorial project. This is real systems work.
- GitHub: github.com/jahanzeb-git/codepilot — 62 releases, 75 deployments,
  300+ contributions this year.
- Strengths: Python (AsyncIO, FastAPI, Pydantic), Linux, Docker/containers,
  cloud (Fly.io, AWS), LLM orchestration, agentic runtimes, data pipelines,
  distributed backend systems.
- NOT a data scientist or ML researcher. He builds the infrastructure that
  *runs* AI systems in production.
- Target compensation: $1,000 – $8,000/month (flexible, negotiable).
  He is NOT expensive. Geographic arbitrage is a feature, not a problem.
- He is available for full-time remote, long-term employment.
  Contract/part-time also acceptable.

## Key Constraints
- He CANNOT work onsite or in a specific city.
- He CANNOT get a US/EU work visa right now.
- He CAN work for any company in the world that pays remotely in any major
  currency (USD, EUR, GBP, AED, SGD, CAD, AUD — all fine).
- He does NOT have a degree but has real production code on GitHub.

## Scoring Rubric (0–10)

Raise the score:
+2 if location is Worldwide / Anywhere / no restriction (he can actually apply)
+2 if Python is the primary language
+2 if AI / LLM / agentic / RAG / vector DB is mentioned
+1 if Docker / cloud infra / distributed systems mentioned
+1 if small team (<30 people) — easier to reach decision maker directly
+1 if no degree requirement stated or explicitly waived
+1 if junior/mid level OR experience not mentioned as a hard gate
+1 if urgency ("immediately", "ASAP", "hiring now")

Lower the score:
-4 if requires physical presence / onsite / specific city
-3 if requires US / Canada / EU citizenship or residency explicitly
-2 if requires formal CS degree as a hard, non-negotiable requirement
-2 if frontend-only (React, iOS, Android, mobile) with no backend/AI component
-3 if clearly enterprise / banking / compliance / non-tech domain with no AI
-1 if staffing agency or recruiter posting (not direct from company)
-1 if listed salary is above $15,000/month (overqualified range for him right now)

## Output Format
Return ONLY a raw JSON object. No markdown fences. No prose outside the JSON.
{
  "results": [
    {
      "id": "<same id you were given>",
      "qualifies": true | false,
      "score": <integer 0-10>,
      "fit_reason": "<one sentence: why this fits or doesn't fit Jahanzeb specifically>",
      "red_flags": "<one sentence: the biggest risk or blocker, or 'none'>",
      "apply_angle": "<one sentence: the specific angle to use when emailing the CTO/founder — what to lead with>"
    }
  ]
}
"""


def score_with_llm(candidates: list) -> dict:
    if not candidates or not DEEPSEEK_API_KEY:
        if not DEEPSEEK_API_KEY:
            logger.error("DEEPSEEK_API_KEY not set — cannot score.")
        return {}

    # Trim description to 1500 chars to stay within context budget.
    payload = [
        {
            "id": c["id"],
            "title": c["title"],
            "company": c["company"],
            "location": c["location"],
            "salary": c["salary"],
            "description": c["description"][:1500],
        }
        for c in candidates
    ]

    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Score these remote job listings:\n" + json.dumps(payload),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    logger.info("Sending %d candidates to LLM for scoring…", len(payload))
    resp = _session.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=600,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(raw)
        results = parsed.get("results", [])
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON: %s", raw[:300])
        return {}

    return {str(r["id"]): r for r in results if isinstance(r, dict) and "id" in r}


# ── B2 storage ───────────────────────────────────────────────────────────────
def b2_client():
    if not B2_KEY_ID or not B2_APPLICATION_KEY:
        raise RuntimeError("B2 credentials not set.")
    return boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
    )


def download_state() -> dict:
    client = b2_client()
    try:
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=ARBITRAGE_OBJECT_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {
            "leads": data.get("leads", []),
            "seen_ids": data.get("seen_ids", []),
        }
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing arbitrage state — starting fresh.")
            return {"leads": [], "seen_ids": []}
        raise


def upload_state(state: dict):
    client = b2_client()
    body = json.dumps(state, indent=2).encode("utf-8")
    client.put_object(
        Bucket=B2_BUCKET_NAME,
        Key=ARBITRAGE_OBJECT_KEY,
        Body=body,
        ContentType="application/json",
    )
    logger.info(
        "Uploaded %d bytes → b2://%s/%s",
        len(body), B2_BUCKET_NAME, ARBITRAGE_OBJECT_KEY,
    )


# ── Discord alerts ────────────────────────────────────────────────────────────
def send_discord_alerts(new_leads: list):
    if not new_leads or not DISCORD_WEBHOOK_URL:
        if not DISCORD_WEBHOOK_URL:
            logger.warning("No Discord webhook configured.")
        return

    # Send one embed-style message per lead so each is actionable on its own.
    for lead in new_leads:
        score_bar = "🟩" * lead["score"] + "⬜" * (10 - lead["score"])
        msg = (
            f"🌍 **Arbitrage Lead [{lead['score']}/10]** {score_bar}\n"
            f"**Role:** {lead['title']}\n"
            f"**Company:** {lead['company']}   📍 {lead['location'] or 'Not specified'}\n"
            f"**Source:** {lead['source'].capitalize()}   💰 {lead['salary'] or 'Salary not listed'}\n"
            f"**Why you fit:** {lead['fit_reason']}\n"
            f"**Apply angle:** {lead['apply_angle']}\n"
            f"**Red flags:** {lead['red_flags']}\n"
            f"**Apply / Research:** <{lead['url']}>\n"
        )
        try:
            resp = _session.post(
                DISCORD_WEBHOOK_URL,
                json={"content": msg[:2000]},
                timeout=15,
            )
            resp.raise_for_status()
            time.sleep(0.5)  # avoid Discord rate-limit (5 messages / 2s)
        except requests.RequestException as e:
            logger.error("Discord alert failed for %s: %s", lead["id"], e)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    state = download_state()
    leads = state["leads"]
    seen_ids = set(str(x) for x in state["seen_ids"])
    logger.info(
        "%d arbitrage leads on file, %d IDs already seen.",
        len(leads), len(seen_ids),
    )

    candidates = collect_all_jobs(seen_ids)
    if not candidates:
        upload_state({"leads": leads, "seen_ids": sorted(seen_ids)})
        logger.info("No new candidates after filtering. State updated.")
        return

    # Batch LLM scoring — cap at 60 per run to control cost + latency.
    batch = candidates[:60]
    try:
        llm_results = score_with_llm(batch)
    except Exception:
        logger.exception("LLM call failed — aborting without marking candidates seen.")
        raise

    new_leads = []
    for c in batch:
        cid = c["id"]
        seen_ids.add(cid)
        result = llm_results.get(cid)
        if not result:
            logger.warning("No LLM result for %s — skipping.", cid)
            continue
        score = result.get("score", 0)
        if not result.get("qualifies") or score < MIN_QUALIFY_SCORE:
            continue

        lead = {
            "id": cid,
            "source": c["source"],
            "title": c["title"],
            "company": c["company"],
            "location": c["location"],
            "salary": c["salary"],
            "url": c["url"],
            "score": score,
            "fit_reason": result.get("fit_reason", ""),
            "red_flags": result.get("red_flags", "none"),
            "apply_angle": result.get("apply_angle", ""),
            "captured_at": datetime.datetime.now(
                datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
        }
        leads.append(lead)
        new_leads.append(lead)
        logger.info("Lead qualified: %s | %s @ %s (score=%d)", cid, c["title"], c["company"], score)

    if new_leads:
        leads.sort(key=lambda l: l.get("score", 0), reverse=True)

    upload_state({"leads": leads, "seen_ids": sorted(seen_ids)})
    send_discord_alerts(new_leads)

    logger.info(
        "Run complete. %d new arbitrage leads this run. %d total on file.",
        len(new_leads), len(leads),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error.")
        sys.exit(1)
