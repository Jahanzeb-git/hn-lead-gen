"""
lead_scraper.py — HN lead-gen V2 for AI/agentic contract & job leads.

Single-shot script, run by GitHub Actions cron (no daemon/while-True —
runners are ephemeral). Each run:

  1. Downloads leads_pipeline.json (dedup state + accumulated leads) from B2.
  2. Pulls current-month threads via the Algolia HN Search API:
       - "Ask HN: Who is hiring?"        (all new top-level comments)
       - "Ask HN: Freelancer? Seeking freelancer?" (all new top-level comments)
       - Show HN                          (new stories, last N hours)
       - Launch HN                        (new stories, last N hours)
  3. Cheap regex pre-filter (free, no network) knocks out obvious non-matches.
  4. Survivors go to a single batched LLM call (DeepSeek) which:
       - for Freelancer-thread comments: classifies SEEKING_WORK vs
         SEEKING_FREELANCER first, and only scores the latter
       - for everything else: scores AI/agentic contract-work relevance
  5. New qualifying leads get appended, dedup state updated, uploaded to B2.
  6. One batched Telegram message summarizing this run's new leads (if any).

Why Algolia instead of the HN Firebase API:
  - /items/:id returns a full nested comment tree in ONE call, instead of
    needing one HTTP request per comment ID (Firebase's kids-array
    approach). For a 400-comment hiring thread that's 1 call vs 400+.
  - Native tags (show_hn, ask_hn) and numericFilters (created_at_i, points)
    let us find Show/Launch HN and locate the current month's threads by
    title search, which Firebase has no equivalent for.

Dedup model:
  - `known_ids`: every comment/story ID ever scored (qualifying or not).
    This fixes a real V1 bug: V1 only remembered qualifying leads, so
    every non-matching comment was re-fetched and re-scored on every run
    for the thread's entire month-long life. We now remember rejections
    too, so nothing is ever reprocessed by either the regex or LLM stage.

Time-of-month self-throttle:
  - GitHub Actions cron fires every 15 minutes, all month. The script
    decides on each tick whether to actually do work, based on day-of-
    month, so day 1-2 get full 15-min granularity (when ~40-50% of a
    month's Who's Hiring comments land) while later days taper to hourly,
    then every 3h, then every 6h — see should_run_this_tick().

Required environment variables (GitHub Actions secrets):
    DEEPSEEK_API_KEY
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    B2_KEY_ID, B2_APPLICATION_KEY
    B2_BUCKET_NAME     (default "hnscraper")
    B2_ENDPOINT        (default the us-west-004 S3 endpoint)
    B2_OBJECT_KEY      (default "leads_pipeline.json")
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
logger = logging.getLogger("lead_scraper")

# --- Config (all from env, nothing sensitive hardcoded) ---
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "hnscraper")
B2_OBJECT_KEY = os.environ.get("B2_OBJECT_KEY", "leads_pipeline.json")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None

MAX_ITEMS_PER_RUN = int(os.environ.get("MAX_ITEMS_PER_RUN", 500))
SHOW_HN_LOOKBACK_HOURS = int(os.environ.get("SHOW_HN_LOOKBACK_HOURS", 6))
LAUNCH_HN_LOOKBACK_HOURS = int(os.environ.get("LAUNCH_HN_LOOKBACK_HOURS", 48))

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"

_session = requests.Session()
_session.headers.update({"User-Agent": "HN Lead Agent (contact: jahanzeb-git)"})

# --- Regex pre-filter (free, no network — runs before any LLM call) ---
# Kept from V1: cheap enough to run on every candidate, cuts volume before
# the paid stage. MIN_TECH_MATCHES is higher for Show HN because post
# bodies are much longer than a hiring-thread comment and would otherwise
# over-match on unrelated tech mentions.
TECH_SIGNALS = [
    r"\bagentic\b", r"\bagents?\b", r"\bllms?\b", r"\bai\b",
    r"\blanggraph\b", r"\blangchain\b", r"\bcrewai\b",
    r"\breact\s+agents?\b", r"\bmcp\b", r"\bmodel context protocol\b",
    r"\btool[\s-]?calling\b", r"\bfunction[\s-]?calling\b",
    r"\brag\b", r"\bretrieval[\s-]?augmented\b",
    r"\bvector\s?db\b", r"\bvector\s?database\b", r"\bembeddings?\b",
    r"\bllm\s?orchestration\b", r"\borchestration\b",
    r"\bpython\b", r"\bfastapi\b", r"\bbackend\b",
]
ENGAGEMENT_SIGNALS = [
    r"\bcontract\b", r"\bcontractor\b", r"\bcontract[\s-]to[\s-]hire\b",
    r"\bfreelance\b", r"\bfractional\b", r"\bfounding engineer\b",
    r"\bpart[\s-]time\b", r"\bconsult(?:ant|ing)?\b",
]
_TECH_PATTERN = re.compile("|".join(TECH_SIGNALS), re.IGNORECASE)
_ENGAGEMENT_PATTERN = re.compile("|".join(ENGAGEMENT_SIGNALS), re.IGNORECASE)
MIN_TECH_MATCHES_COMMENT = 2
MIN_TECH_MATCHES_SHOWHN = 3


def regex_prefilter(text: str, min_tech_matches: int = MIN_TECH_MATCHES_COMMENT):
    """Cheap first pass. Returns None if it can't possibly qualify, else
    match info that's cheap to compute and useful context for the LLM."""
    if not text:
        return None
    tech_matches = sorted(set(m.lower() for m in _TECH_PATTERN.findall(text)))
    if len(tech_matches) < min_tech_matches:
        return None
    engagement_matches = sorted(set(m.lower() for m in _ENGAGEMENT_PATTERN.findall(text)))
    return {"tech_matches": tech_matches, "engagement_matches": engagement_matches}


def get_json_with_backoff(url: str, params: dict = None, max_retries: int = 5, base_delay: float = 1.5):
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Request failed (attempt %d/%d) for %s: %s — retrying in %.1fs",
                attempt + 1, max_retries, url, e, delay,
            )
            time.sleep(delay)
    logger.error("Giving up on %s after %d attempts", url, max_retries)
    raise last_exc


def find_thread_by_title_prefix(title_prefix: str, tags: str = "story"):
    """Search Algolia for a story whose title starts with the given prefix,
    most-recent first. Used to locate the current month's Who's Hiring /
    Freelancer threads without needing the whoishiring bot's submission
    list (Firebase-only) — Algolia search_by_date does this directly."""
    data = get_json_with_backoff(
        f"{ALGOLIA_BASE}/search_by_date",
        params={"query": f'"{title_prefix}"', "tags": tags, "hitsPerPage": 10},
    )
    for hit in data.get("hits", []):
        if (hit.get("title") or "").startswith(title_prefix):
            return hit.get("objectID"), hit.get("title")
    return None, None


def fetch_item_tree(item_id):
    """Algolia's big win over Firebase: the ENTIRE nested comment tree for
    one story comes back in a single HTTP call, children included."""
    return get_json_with_backoff(f"{ALGOLIA_BASE}/items/{item_id}")


def flatten_top_level_comments(item_tree: dict):
    """We only score top-level comments (direct replies to the thread),
    same as V1 — nested reply-to-a-reply chatter isn't a hiring/freelance
    post, it's discussion about one."""
    children = item_tree.get("children", []) if item_tree else []
    return [c for c in children if c and not c.get("deleted") and not c.get("dead")]


def find_recent_stories(tag: str, lookback_hours: int, max_hits: int = 200):
    """Show HN / Launch HN: pull recent stories with the given Algolia tag,
    newest first, within the lookback window."""
    since = int(time.time()) - lookback_hours * 3600
    data = get_json_with_backoff(
        f"{ALGOLIA_BASE}/search_by_date",
        params={
            "tags": tag,
            "numericFilters": f"created_at_i>{since}",
            "hitsPerPage": max_hits,
        },
    )
    return data.get("hits", [])


# --- LLM scoring stage (DeepSeek) ---
# Batched: one call scores a whole list of candidates at once, not one
# call per candidate. This is the real cost lever — 40 candidates in one
# request shares a single system prompt + JSON-schema instruction instead
# of paying for it 40 times, and DeepSeek's prompt caching makes repeat
# runs with the same system prompt even cheaper on the input side.
_SYSTEM_PROMPT = """You are a lead-scoring assistant for a self-taught AI/agentic \
systems engineer (Python, LLMs, RAG, MCP, FastAPI) looking for remote contract \
or freelance work, bypassing traditional HR/degree-gated hiring.

You will receive a JSON array of HN posts/comments. For EACH one, return an \
object with:
  - "id": the same id you were given
  - "source_intent": for freelancer-thread items only, classify as \
"SEEKING_WORK" (someone advertising themselves for hire) or \
"SEEKING_FREELANCER" (someone looking to hire/pay a freelancer). For all \
other item types, use "N/A".
  - "qualifies": true/false — true ONLY if this is someone (a founder, \
company, or hiring manager) who might realistically pay for AI/agentic/LLM/\
Python backend engineering work, AND (for freelancer-thread items) \
source_intent is SEEKING_FREELANCER. A freelancer/job-seeker advertising \
themselves never qualifies, regardless of thread.
  - "score": integer 0-10, how strong a lead this is (funding/urgency/\
explicit tech-stack match/contract-friendly language all raise it)
  - "reasoning": one short sentence, why

Return ONLY a JSON object of the form {"results": [...]}, no prose, no \
markdown fences."""


def score_candidates_with_llm(candidates: list) -> dict:
    """candidates: list of {id, item_type, text}. Returns {id: result_dict}."""
    if not candidates:
        return {}
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    payload_items = [
        {"id": c["id"], "item_type": c["item_type"], "text": c["text"][:2000]}
        for c in candidates
    ]

    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Score these items as JSON:\n" + json.dumps(payload_items)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    resp = _session.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=500,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(raw)
        results = parsed.get("results", parsed if isinstance(parsed, list) else [])
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON output, skipping this batch: %s", raw[:300])
        return {}

    return {str(r["id"]): r for r in results if isinstance(r, dict) and "id" in r}


# --- B2 storage: state is one JSON blob, no separate DB ---
def b2_client():
    if not B2_KEY_ID or not B2_APPLICATION_KEY:
        raise RuntimeError("B2_KEY_ID / B2_APPLICATION_KEY are not set.")
    return boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
    )


def download_state_from_b2() -> dict:
    """State shape: {"leads": [...], "seen_ids": [...]}.
    `seen_ids` includes IDs that were scored and REJECTED — this is what
    fixes V1's forever-rescan bug."""
    client = b2_client()
    try:
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=B2_OBJECT_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        if isinstance(data, list):
            # backward-compat with V1's bare-array format
            return {"leads": data, "seen_ids": [l["id"] for l in data]}
        return {"leads": data.get("leads", []), "seen_ids": data.get("seen_ids", [])}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing %s in bucket yet — starting fresh.", B2_OBJECT_KEY)
            return {"leads": [], "seen_ids": []}
        raise


def upload_state_to_b2(state: dict):
    client = b2_client()
    body = json.dumps(state, indent=2).encode("utf-8")
    client.put_object(
        Bucket=B2_BUCKET_NAME, Key=B2_OBJECT_KEY, Body=body,
        ContentType="application/json",
    )
    logger.info("Uploaded %d bytes to b2://%s/%s", len(body), B2_BUCKET_NAME, B2_OBJECT_KEY)


# --- Discord: one batched message per run, not one per lead ---
def send_discord_summary(new_leads: list):
    if not new_leads:
        return
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord not configured — skipping notification for %d leads.", len(new_leads))
        return

    lines = [f"🎯 **{len(new_leads)} new HN lead(s):**\n"]
    for lead in new_leads[:15]:  # Discord has a 2000-char message cap
        lines.append(
            f"• [{lead['score']}/10] **{lead['source']}** — {lead.get('author', '?')}\n"
            f"  *{lead['reasoning']}*\n"
            f"  <{lead['url']}>"
        )
    text = "\n".join(lines)
    if len(new_leads) > 15:
        text += f"\n\n...and {len(new_leads) - 15} more (see leads_pipeline.json)."

    try:
        # truncate just in case to fit discord's hard limit
        resp = _session.post(DISCORD_WEBHOOK_URL, json={"content": text[:2000]}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Discord notification failed: %s", e)


# --- Day-of-month self-throttle ---
# GitHub Actions cron fires every 15 minutes, ALL month (see scrape.yml) —
# Actions cron syntax can't express "every 15 min but only on day 1".
# So instead we let the cheapest possible cron fire constantly, and this
# function decides on EVERY tick whether to actually do work, based on
# comment-arrival patterns: ~40-50% of a month's Who's Hiring comments
# land in the first 48h, so day 1-2 get full 15-min granularity; later
# days taper off since a wasted tick only costs ~1 Algolia search call
# (cheap) if should_run_this_tick says no and we exit before any LLM call.
def should_run_this_tick(now: datetime.datetime = None) -> bool:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    day, minute = now.day, now.minute

    if day <= 2:
        return True  # every 15-min tick does work — peak arrival window
    if 3 <= day <= 5:
        return minute == 0  # hourly
    if 6 <= day <= 14:
        return minute == 0 and now.hour % 3 == 0  # every 3 hours
    return minute == 0 and now.hour % 6 == 0  # every 6 hours, long tail


# --- Candidate gathering per thread type ---
def gather_hiring_candidates(seen_ids: set) -> list:
    thread_id, title = find_thread_by_title_prefix("Ask HN: Who is hiring?")
    if not thread_id:
        logger.warning("No active 'Who is hiring?' thread found.")
        return []
    logger.info("Who's Hiring thread: %s (id=%s)", title, thread_id)
    tree = fetch_item_tree(thread_id)
    candidates = []
    for c in flatten_top_level_comments(tree):
        cid = str(c["id"])
        if cid in seen_ids:
            continue
        text = c.get("text", "")
        if not regex_prefilter(text, MIN_TECH_MATCHES_COMMENT):
            seen_ids.add(cid)  # regex-rejected — never re-check this ID again
            continue
        candidates.append({
            "id": cid, "item_type": "hiring", "text": text,
            "author": c.get("author"), "thread_id": thread_id,
            "url": f"https://news.ycombinator.com/item?id={cid}",
        })
    return candidates


def gather_freelancer_candidates(seen_ids: set) -> list:
    """Both SEEKING_WORK and SEEKING_FREELANCER posts land in the same
    thread with no separate tag — that split can only be made by the LLM
    reading the text (regex can't reliably tell "I'm available" from
    "we're hiring"). So the regex pre-filter here just needs tech-relevance,
    same as hiring — the SEEKING_WORK/SEEKING_FREELANCER split and the
    "never surface SEEKING_WORK" rule are enforced entirely in the LLM
    stage via source_intent + qualifies (see _SYSTEM_PROMPT)."""
    thread_id, title = find_thread_by_title_prefix("Ask HN: Freelancer?")
    if not thread_id:
        logger.warning("No active 'Freelancer?' thread found.")
        return []
    logger.info("Freelancer thread: %s (id=%s)", title, thread_id)
    tree = fetch_item_tree(thread_id)
    candidates = []
    for c in flatten_top_level_comments(tree):
        cid = str(c["id"])
        if cid in seen_ids:
            continue
        text = c.get("text", "")
        if not regex_prefilter(text, MIN_TECH_MATCHES_COMMENT):
            seen_ids.add(cid)
            continue
        candidates.append({
            "id": cid, "item_type": "freelancer", "text": text,
            "author": c.get("author"), "thread_id": thread_id,
            "url": f"https://news.ycombinator.com/item?id={cid}",
        })
    return candidates


def gather_showhn_candidates(seen_ids: set) -> list:
    hits = find_recent_stories("show_hn", SHOW_HN_LOOKBACK_HOURS)
    candidates = []
    for hit in hits:
        sid = str(hit["objectID"])
        if sid in seen_ids:
            continue
        text = f"{hit.get('title', '')} {hit.get('story_text') or ''}"
        if not regex_prefilter(text, MIN_TECH_MATCHES_SHOWHN):
            seen_ids.add(sid)
            continue
        candidates.append({
            "id": sid, "item_type": "show_hn", "text": text,
            "author": hit.get("author"), "thread_id": None,
            "url": f"https://news.ycombinator.com/item?id={sid}",
        })
    return candidates


def gather_launchhn_candidates(seen_ids: set) -> list:
    # Launch HN posts are tagged "story" with title prefix "Launch HN:" —
    # Algolia has no dedicated launch_hn tag, so we search by title text.
    hits = find_recent_stories("story", LAUNCH_HN_LOOKBACK_HOURS)
    candidates = []
    for hit in hits:
        title = hit.get("title") or ""
        if not title.startswith("Launch HN:"):
            continue
        sid = str(hit["objectID"])
        if sid in seen_ids:
            continue
        text = f"{title} {hit.get('story_text') or ''}"
        if not regex_prefilter(text, MIN_TECH_MATCHES_COMMENT):
            seen_ids.add(sid)
            continue
        candidates.append({
            "id": sid, "item_type": "launch_hn", "text": text,
            "author": hit.get("author"), "thread_id": None,
            "url": f"https://news.ycombinator.com/item?id={sid}",
        })
    return candidates


def main():
    if not should_run_this_tick():
        logger.info("Throttle: skipping this tick per day-of-month schedule.")
        return

    state = download_state_from_b2()
    leads = state["leads"]
    seen_ids = set(str(x) for x in state["seen_ids"])
    logger.info("%d leads on file, %d IDs already seen (qualifying or not).", len(leads), len(seen_ids))

    candidates = []
    candidates += gather_hiring_candidates(seen_ids)
    candidates += gather_freelancer_candidates(seen_ids)
    candidates += gather_showhn_candidates(seen_ids)
    candidates += gather_launchhn_candidates(seen_ids)
    candidates = candidates[:MAX_ITEMS_PER_RUN]

    logger.info("%d candidates survived the regex pre-filter this run.", len(candidates))
    if not candidates:
        # Even with zero LLM-bound candidates, regex rejections were added
        # to seen_ids above — persist that so we don't re-fetch them forever.
        upload_state_to_b2({"leads": leads, "seen_ids": sorted(seen_ids)})
        logger.info("Run complete. No candidates this run.")
        return

    try:
        llm_results = score_candidates_with_llm(candidates)
    except Exception:
        logger.exception("LLM scoring call failed — aborting run without marking candidates seen, so they're retried next tick.")
        raise

    new_leads = []
    for c in candidates:
        cid = c["id"]
        seen_ids.add(cid)  # scored either way — never re-send to the LLM again
        result = llm_results.get(cid)
        if not result:
            logger.warning("No LLM result returned for candidate %s — treating as non-qualifying.", cid)
            continue
        if not result.get("qualifies"):
            continue

        lead = {
            "id": cid,
            "source": c["item_type"],
            "author": c.get("author"),
            "url": c["url"],
            "thread_id": c.get("thread_id"),
            "source_intent": result.get("source_intent", "N/A"),
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "text_snippet": c["text"][:500],
            "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        leads.append(lead)
        new_leads.append(lead)
        logger.info("Lead qualified: %s (%s, score=%s)", cid, c["item_type"], lead["score"])

    if new_leads:
        leads.sort(key=lambda l: l.get("score", 0), reverse=True)

    upload_state_to_b2({"leads": leads, "seen_ids": sorted(seen_ids)})
    send_discord_summary(new_leads)

    logger.info(
        "Run complete. %d new leads this run. %d total leads on file.",
        len(new_leads), len(leads),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error — failing the run so GitHub Actions flags it.")
        sys.exit(1)
