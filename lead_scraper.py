"""
lead_scraper.py — single-file, one-shot HN lead-gen for B2B AI Infra Contracting.

Designed to be run once per invocation by a GitHub Actions cron job
(not a persistent daemon — GitHub runners are ephemeral, so there's
no `while True` here). Each run:

  1. Downloads the current leads_pipeline.json from B2 (if it exists).
  2. Builds a set of comment IDs already captured — this file IS the
     dedup state, no separate database needed.
  3. Finds the current "Ask HN: Who is hiring?" thread and pulls its
     top-level comment IDs.
  4. Skips any ID already in the file. Scores the rest against B2B
     AI-infra keyword + engagement signals.
  5. Appends new qualifying leads, re-sorts by priority/score, and
     uploads the same file back to B2.

Trade-off, on purpose for simplicity: non-qualifying comments are NOT
remembered anywhere, so they get re-scanned every run for as long as
that month's thread stays active. That's cheap — the HN API has no
real rate limit and a hiring thread only has a few hundred top-level
comments — so it's a fine price for not needing a second state file.

Required environment variables (set as GitHub Actions secrets):
    B2_KEY_ID              — B2 applicationKeyId
    B2_APPLICATION_KEY     — B2 applicationKey (secret)
    B2_BUCKET_NAME         — defaults to "hnscraper"
    B2_ENDPOINT            — defaults to the us-west-004 S3 endpoint
    B2_OBJECT_KEY          — defaults to "leads_pipeline.json"
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
MAX_COMMENTS_PER_RUN = int(os.environ.get("MAX_COMMENTS_PER_RUN", 500))

HN_API_BASE = "https://hacker-news.firebaseio.com/v0"

# --- Keyword / intent signals ---
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
MIN_TECH_MATCHES = 2

_session = requests.Session()
_session.headers.update({"User-Agent": "CodePilot Lead Gen (contact: jahanzeb-git)"})


def score_text(text: str):
    """Returns None if the text doesn't qualify, else a dict of match info."""
    if not text:
        return None
    tech_matches = sorted(set(m.lower() for m in _TECH_PATTERN.findall(text)))
    if len(tech_matches) < MIN_TECH_MATCHES:
        return None
    engagement_matches = sorted(set(m.lower() for m in _ENGAGEMENT_PATTERN.findall(text)))
    priority = len(engagement_matches) > 0
    score = len(tech_matches) + (3 * len(engagement_matches) if priority else 0)
    return {
        "matched_keywords": tech_matches,
        "engagement_signals": engagement_matches,
        "priority": priority,
        "score": score,
    }


def get_json_with_backoff(url: str, max_retries: int = 5, base_delay: float = 1.5):
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, timeout=15)
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


def find_latest_hiring_thread():
    data = get_json_with_backoff(f"{HN_API_BASE}/user/whoishiring.json")
    submissions = data.get("submitted", []) if data else []
    for post_id in submissions[:10]:
        post = get_json_with_backoff(f"{HN_API_BASE}/item/{post_id}.json")
        if post and post.get("title", "").startswith("Ask HN: Who is hiring?"):
            return post_id, post.get("title")
    return None, None


def b2_client():
    if not B2_KEY_ID or not B2_APPLICATION_KEY:
        raise RuntimeError("B2_KEY_ID / B2_APPLICATION_KEY are not set.")
    return boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
    )


def download_leads_from_b2() -> list:
    client = b2_client()
    try:
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=B2_OBJECT_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing %s in bucket yet — starting fresh.", B2_OBJECT_KEY)
            return []
        raise


def upload_leads_to_b2(leads: list):
    client = b2_client()
    body = json.dumps(leads, indent=2).encode("utf-8")
    client.put_object(
        Bucket=B2_BUCKET_NAME, Key=B2_OBJECT_KEY, Body=body,
        ContentType="application/json",
    )
    logger.info("Uploaded %d bytes to b2://%s/%s", len(body), B2_BUCKET_NAME, B2_OBJECT_KEY)


def main():
    leads = download_leads_from_b2()
    known_ids = {lead["id"] for lead in leads}
    logger.info("%d leads already captured on file.", len(leads))

    thread_id, title = find_latest_hiring_thread()
    if not thread_id:
        logger.warning("No 'Who is hiring?' thread found this run. Exiting.")
        return
    logger.info("Active thread: %s (id=%s)", title, thread_id)

    thread = get_json_with_backoff(f"{HN_API_BASE}/item/{thread_id}.json")
    comment_ids = thread.get("kids", []) if thread else []
    unseen = [c for c in comment_ids if c not in known_ids][:MAX_COMMENTS_PER_RUN]
    logger.info("%d total comments, %d not yet captured — scanning those.", len(comment_ids), len(unseen))

    new_lead_count = 0
    for i, comment_id in enumerate(unseen):
        try:
            comment = get_json_with_backoff(f"{HN_API_BASE}/item/{comment_id}.json")
        except Exception:
            logger.warning("Skipping comment %s after repeated failures.", comment_id)
            continue

        if not comment or comment.get("deleted") or comment.get("dead"):
            continue

        text = comment.get("text", "")
        result = score_text(text)
        if not result:
            continue

        lead = {
            "id": comment_id,
            "author": comment.get("by"),
            "time": datetime.datetime.fromtimestamp(
                comment.get("time", 0), tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"https://news.ycombinator.com/item?id={comment_id}",
            "thread_id": thread_id,
            "matched_keywords": result["matched_keywords"],
            "engagement_signals": result["engagement_signals"],
            "priority": result["priority"],
            "score": result["score"],
            "text_snippet": text[:500],
        }
        leads.append(lead)
        new_lead_count += 1
        logger.info(
            "Lead found: comment %s by %s (priority=%s, score=%d)",
            comment_id, comment.get("by"), result["priority"], result["score"],
        )

        if i % 25 == 0 and i > 0:
            logger.info("Scanned %d/%d comments this run...", i, len(unseen))

    if new_lead_count:
        leads.sort(key=lambda l: (l.get("priority", False), l.get("score", 0)), reverse=True)
        upload_leads_to_b2(leads)

    logger.info(
        "Run complete. %d new leads this run. %d total leads on file.",
        new_lead_count, len(leads),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error — failing the run so GitHub Actions flags it.")
        sys.exit(1)