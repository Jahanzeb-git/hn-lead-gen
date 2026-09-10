"""
yc_proactive_targets.py — Proactive Open Source AI Target Hunter

Runs as a separate GitHub Actions job. Instead of finding active job postings,
it actively hunts for Seed/Series A "Open Source AI" startups from Y Combinator's
database. 

Goal: Find highly-funded, small (<50 team size) startups matching Jahanzeb's 
Backend/AI/Agentic skill profile, so he can execute the "Side Door" playbook:
Submit a PR -> DM the Founder.

Data Source: Y Combinator API
"""

import datetime
import json
import logging
import os
import re
import sys
import time
import urllib.parse

import boto3
import requests
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("yc_proactive_targets")

# ── Config ──────────────────────────────────────────────────────────────────
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "hnscraper")
# Separate state file for YC targets so it doesn't overlap with arbitrage jobs
YC_STATE_OBJECT_KEY = os.environ.get("YC_STATE_OBJECT_KEY", "yc_targets_state.json")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None

_session = requests.Session()
_session.headers.update({"User-Agent": "YC Target Agent (contact: jahanzeb-git)"})

# ── The Rich Tech Regex (Reused from arbitrage_scraper) ──────────────────────
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

def is_open_source(company: dict) -> bool:
    # Check if "Open Source" is in tags or explicitly mentioned in the description
    tags = [t.lower() for t in company.get("tags", [])]
    if "open source" in tags:
        return True
    
    desc = company.get("longDescription", "") or ""
    one_liner = company.get("oneLiner", "") or ""
    full_text = (desc + " " + one_liner).lower()
    
    if "open source" in full_text or "open-source" in full_text:
        return True
    return False

# ── YC Fetching & Filtering ──────────────────────────────────────────────────
def fetch_yc_companies() -> list:
    url = "https://api.ycombinator.com/v0.1/companies"
    companies = []
    try:
        logger.info("Fetching companies from YC API...")
        page = 1
        while True:
            resp = _session.get(url, params={"tags": "Open Source", "page": page}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("companies", [])
            companies.extend(batch)
            
            total_pages = data.get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.2) # Be nice to the API
            
        logger.info("Fetched %d companies in total across %d pages.", len(companies), page)
        return companies
    except Exception as e:
        logger.error("Failed to fetch YC companies: %s", e)
        return companies

def filter_companies(companies: list, seen_ids: set) -> list:
    targets = []
    
    for c in companies:
        cid = str(c.get("id"))
        
        # 1. Skip if we already alerted on this
        if cid in seen_ids:
            continue
            
        # 2. Only Active startups
        if c.get("status") != "Active":
            continue
            
        # 3. Size Filter: Small enough to reach CTO directly
        team_size = c.get("teamSize") or 0
        if team_size > 50:
            continue
            
        # 4. Open Source Filter
        if not is_open_source(c):
            continue
            
        # 5. Rich Tech/AI Regex Filter
        full_text = (
            f"{c.get('name', '')} {c.get('oneLiner', '')} "
            f"{c.get('longDescription', '')} {' '.join(c.get('tags', []))}"
        )
        if not passes_regex(full_text):
            continue
            
        targets.append(c)
        
    logger.info("Filtered down to %d highly relevant proactive targets.", len(targets))
    return targets

def format_github_search_url(company_name: str, website: str) -> str:
    # If they have a website, extract domain without www for precise searching
    if website:
        domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        query = f'"{domain}"'
    else:
        query = f'"{company_name}"'
    
    # URL encode the search query
    encoded_query = urllib.parse.quote_plus(query)
    return f"https://github.com/search?q={encoded_query}&type=Users"

# ── B2 Storage ───────────────────────────────────────────────────────────────
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
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=YC_STATE_OBJECT_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {
            "targets": data.get("targets", []),
            "seen_ids": data.get("seen_ids", []),
        }
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing YC state — starting fresh.")
            return {"targets": [], "seen_ids": []}
        raise

def upload_state(state: dict):
    client = b2_client()
    body = json.dumps(state, indent=2).encode("utf-8")
    client.put_object(
        Bucket=B2_BUCKET_NAME,
        Key=YC_STATE_OBJECT_KEY,
        Body=body,
        ContentType="application/json",
    )
    logger.info(
        "Uploaded %d bytes → b2://%s/%s",
        len(body), B2_BUCKET_NAME, YC_STATE_OBJECT_KEY,
    )

# ── Discord Alerts ────────────────────────────────────────────────────────────
def send_discord_alerts(new_targets: list):
    if not new_targets or not DISCORD_WEBHOOK_URL:
        if not DISCORD_WEBHOOK_URL:
            logger.warning("No Discord webhook configured for YC Scraper.")
        return

    for c in new_targets:
        cid = str(c.get("id"))
        name = c.get("name", "Unknown")
        website = c.get("website", "")
        one_liner = c.get("oneLiner", "")
        team_size = c.get("teamSize", "Unknown")
        batch = c.get("batch", "Unknown")
        url = c.get("url", "")
        
        gh_url = format_github_search_url(name, website)
        
        msg = (
            f"🎯 **Proactive Target: {name}** (Batch {batch})\n"
            f"**One Liner:** {one_liner}\n"
            f"**Team Size:** {team_size} (Perfect Side-Door Size)\n"
            f"**Tags:** {', '.join(c.get('tags', []))[:100]}...\n"
            f"**Website:** {website}\n"
            f"**YC Profile:** <{url}>\n"
            f"**Hunt on GitHub:** <{gh_url}>\n"
            f"*(Action: Find repo -> Fix bug/submit PR -> DM Founder)*\n"
        )
        try:
            resp = _session.post(
                DISCORD_WEBHOOK_URL,
                json={"content": msg[:2000]},
                timeout=15,
            )
            resp.raise_for_status()
            time.sleep(0.5)
        except requests.RequestException as e:
            logger.error("Discord alert failed for YC %s: %s", cid, e)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    state = download_state()
    targets_state = state["targets"]
    seen_ids = set(str(x) for x in state["seen_ids"])
    logger.info(
        "%d YC targets on file, %d IDs already seen.",
        len(targets_state), len(seen_ids),
    )

    all_companies = fetch_yc_companies()
    new_targets = filter_companies(all_companies, seen_ids)

    if not new_targets:
        logger.info("No new matching YC targets found. State remains unchanged.")
        # Optional: you can upload state anyway to confirm heartbeat, 
        # but skipping is fine if no new targets exist.
        return

    for c in new_targets:
        cid = str(c["id"])
        seen_ids.add(cid)
        
        target_record = {
            "id": cid,
            "name": c.get("name"),
            "oneLiner": c.get("oneLiner"),
            "website": c.get("website"),
            "teamSize": c.get("teamSize"),
            "batch": c.get("batch"),
            "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        targets_state.append(target_record)

    upload_state({"targets": targets_state, "seen_ids": sorted(seen_ids)})
    send_discord_alerts(new_targets)

    logger.info(
        "Run complete. %d new targets found. %d total targets in state.",
        len(new_targets), len(targets_state),
    )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in yc_proactive_targets.")
        sys.exit(1)
