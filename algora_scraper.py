"""
algora_scraper.py — Algora Open-Source Bounty Agent

Monitors Algora.io for Python / AI / backend bounties and fires a Discord
alert for each new qualifying bounty. No LLM scoring — bounties are
self-describing (amount + repo language + issue title). Rule-based filter
is fast, free, and accurate enough.

Why Algora?
  Unlike job applications (async, slow), bounties have a claim → PR → pay
  flow that can complete inside 24 hours. Fix a real bug, get paid, and
  the maintainer remembers you. The relationship is the actual prize.

Pipeline per run:
  1. Fetch all open bounties from Algora public API (no auth required).
  2. Filter: amount >= MIN_BOUNTY_USD (default $75).
  3. Filter: repo language / topics / title must match Python/AI keyword set.
  4. Dedup against B2 state.
  5. Fire one Discord message per new qualifying bounty.
  6. Upload updated state to B2.

Required env vars:
  DISCORD_WEBHOOK_URL
  B2_KEY_ID, B2_APPLICATION_KEY
  B2_BUCKET_NAME    (default: hnscraper)
  B2_ENDPOINT       (default: us-west-004 backblaze endpoint)
  MIN_BOUNTY_USD    (default: 75)
  ALGORA_OBJECT_KEY (default: algora_pipeline.json)
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
logger = logging.getLogger("algora_scraper")

# ── Config ────────────────────────────────────────────────────────────────────
B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "hnscraper")
ALGORA_OBJECT_KEY = os.environ.get("ALGORA_OBJECT_KEY", "algora_pipeline.json")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None
MIN_BOUNTY_USD = int(os.environ.get("MIN_BOUNTY_USD", 75))

_session = requests.Session()
_session.headers.update({
    "User-Agent": "BountyHunterBot/1.0 (github.com/Jahanzeb-git)",
    "Accept": "application/json",
})

# ── Algora API endpoints ──────────────────────────────────────────────────────
# Public REST feed — no auth required.
ALGORA_REST_URL = "https://algora.io/api/bounties"
ALGORA_GRAPHQL_URL = "https://console.algora.io/api/graphql"

# ── Relevance filter ──────────────────────────────────────────────────────────
RELEVANT_TAGS = re.compile(
    r"\b(?:python|fastapi|django|flask|backend|api|llm|ai|ml|agent|agentic"
    r"|openai|anthropic|langchain|langgraph|rag|vector|embedding|async"
    r"|asyncio|docker|kubernetes|k8s|cli|sdk|framework|inference|pipeline"
    r"|pydantic|sqlalchemy|postgres|redis|cloud|devops|data|nlp|gpt"
    r"|mistral|ollama|transformers|huggingface|mcp|tool.?call)\b",
    re.IGNORECASE,
)

EXCLUDE_LANGUAGES = {"swift", "kotlin", "objective-c", "dart", "elm", "ruby", "php"}


def is_relevant(bounty: dict) -> bool:
    lang = (bounty.get("language") or "").lower().strip()
    if lang in EXCLUDE_LANGUAGES:
        return False
    text = " ".join(filter(None, [
        bounty.get("repo_name", ""),
        bounty.get("repo_description", ""),
        bounty.get("issue_title", ""),
        bounty.get("issue_body", "")[:500],
        " ".join(bounty.get("topics", [])),
        lang,
    ]))
    return lang == "python" or RELEVANT_TAGS.search(text) is not None


# ── Algora fetch ──────────────────────────────────────────────────────────────

def normalize_bounty(raw: dict):
    bid = str(raw.get("id") or raw.get("objectId") or "")
    if not bid:
        return None

    amount_raw = raw.get("reward") or raw.get("amount") or raw.get("reward_amount") or 0
    try:
        amount = float(amount_raw)
        if amount > 100_000:
            amount = amount / 100  # convert cents to dollars
    except (TypeError, ValueError):
        amount = 0.0

    amount_fmt = raw.get("reward_formatted") or raw.get("reward_text") or f"${amount:.0f}"

    issue = raw.get("issue") or {}
    issue_title = raw.get("issue_title") or issue.get("title") or raw.get("title") or ""
    issue_url = (
        raw.get("issue_url") or issue.get("html_url") or issue.get("url") or raw.get("url") or ""
    )

    repo = raw.get("repository") or raw.get("repo") or issue.get("repository") or {}
    repo_name = raw.get("repo_name") or raw.get("repo_full_name") or repo.get("full_name") or repo.get("name") or ""
    repo_desc = raw.get("repo_description") or repo.get("description") or ""
    language = (raw.get("language") or repo.get("language") or "").lower()
    topics = raw.get("topics") or repo.get("topics") or []

    algora_url = (
        raw.get("algora_url") or raw.get("bounty_url") or f"https://algora.io/bounties/{bid}"
    )

    return {
        "id": f"algora_{bid}",
        "amount": amount,
        "amount_formatted": amount_fmt,
        "issue_title": issue_title,
        "issue_url": issue_url,
        "issue_body": (raw.get("issue_body") or issue.get("body") or "")[:500],
        "repo_name": repo_name,
        "repo_description": repo_desc,
        "language": language,
        "topics": topics,
        "algora_url": algora_url,
        "created_at": raw.get("created_at") or raw.get("inserted_at") or "",
    }


def fetch_bounties_rest() -> list:
    try:
        resp = _session.get(ALGORA_REST_URL, params={"status": "open", "limit": 100}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("bounties", []))
        return []
    except Exception as e:
        logger.warning("Algora REST fetch failed: %s", e)
        return []


def fetch_bounties_graphql() -> list:
    query = """
    query OpenBounties {
      bounties(where: {status: {_eq: "open"}}, limit: 100, order_by: {created_at: desc}) {
        id
        reward_formatted
        reward_amount
        created_at
        issue {
          title
          body
          url
          repository {
            full_name
            description
            language
            topics
          }
        }
      }
    }
    """
    try:
        resp = _session.post(ALGORA_GRAPHQL_URL, json={"query": query}, timeout=30)
        resp.raise_for_status()
        raw = resp.json().get("data", {}).get("bounties", [])
        normalized = []
        for b in raw:
            issue = b.get("issue") or {}
            repo = issue.get("repository") or {}
            bid = str(b.get("id", ""))
            normalized.append({
                "id": f"algora_{bid}",
                "amount": float(b.get("reward_amount") or 0),
                "amount_formatted": b.get("reward_formatted", "?"),
                "issue_title": issue.get("title", ""),
                "issue_url": issue.get("url", ""),
                "issue_body": (issue.get("body") or "")[:500],
                "repo_name": repo.get("full_name", ""),
                "repo_description": repo.get("description", ""),
                "language": (repo.get("language") or "").lower(),
                "topics": repo.get("topics") or [],
                "algora_url": f"https://algora.io/bounties/{bid}",
                "created_at": b.get("created_at", ""),
            })
        return normalized
    except Exception as e:
        logger.warning("Algora GraphQL fetch failed: %s", e)
        return []


def fetch_bounties_github() -> list:
    """Fallback fetch via GitHub Search API for open bounties and Algora tagged issues."""
    queries = [
        'is:issue is:open "algora.io"',
        'is:issue is:open label:bounty language:python',
        'is:issue is:open "/bounty" language:python',
    ]
    gh_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "application/vnd.github.v3+json",
    }
    bounties = []
    seen = set()

    for q in queries:
        try:
            url = f"https://api.github.com/search/issues?q={requests.utils.quote(q)}&sort=created&order=desc"
            r = requests.get(url, headers=gh_headers, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
            for item in data.get("items", []):
                bid = item.get("id")
                if not bid or bid in seen:
                    continue
                seen.add(bid)

                title = item.get("title", "")
                body = item.get("body", "") or ""
                html_url = item.get("html_url", "")
                repo_parts = html_url.split("/")
                repo_name = "/".join(repo_parts[3:5]) if len(repo_parts) >= 5 else ""

                # Extract dollar amount if present
                amt_m = re.search(r'\$(\d+(?:\.\d+)?)\b|\b(\d+)\s*USD\b', title + " " + body, re.IGNORECASE)
                amount = 0.0
                if amt_m:
                    val = amt_m.group(1) or amt_m.group(2)
                    try:
                        amount = float(val)
                    except ValueError:
                        amount = 0.0

                bounties.append({
                    "id": f"gh_{bid}",
                    "amount": amount,
                    "amount_formatted": f"${amount:.0f}" if amount > 0 else "Bounty",
                    "issue_title": title,
                    "issue_url": html_url,
                    "issue_body": body[:500],
                    "repo_name": repo_name,
                    "repo_description": "",
                    "language": "python",
                    "topics": ["bounty", "open-source"],
                    "algora_url": html_url,
                    "created_at": item.get("created_at", ""),
                })
        except Exception as e:
            logger.warning("GitHub bounty search error for '%s': %s", q, e)

    return bounties


def fetch_all_bounties() -> list:
    all_bounties = []
    seen_ids = set()

    logger.info("Fetching Algora bounties via REST…")
    raw_list = fetch_bounties_rest()
    if raw_list:
        logger.info("REST returned %d raw bounties.", len(raw_list))
        for raw in raw_list:
            if (n := normalize_bounty(raw)) and n["id"] not in seen_ids:
                seen_ids.add(n["id"])
                all_bounties.append(n)

    logger.info("Fetching Algora bounties via GraphQL…")
    gql = fetch_bounties_graphql()
    if gql:
        logger.info("GraphQL returned %d bounties.", len(gql))
        for item in gql:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_bounties.append(item)

    logger.info("Fetching open bounties via GitHub Search API…")
    gh_bounties = fetch_bounties_github()
    if gh_bounties:
        logger.info("GitHub Search API returned %d open bounties.", len(gh_bounties))
        for item in gh_bounties:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_bounties.append(item)

    return all_bounties


# ── B2 storage ────────────────────────────────────────────────────────────────

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
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=ALGORA_OBJECT_KEY)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return {"bounties": data.get("bounties", []), "seen_ids": data.get("seen_ids", [])}
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No existing Algora state — starting fresh.")
            return {"bounties": [], "seen_ids": []}
        raise


def upload_state(state: dict):
    client = b2_client()
    body = json.dumps(state, indent=2).encode("utf-8")
    client.put_object(
        Bucket=B2_BUCKET_NAME, Key=ALGORA_OBJECT_KEY, Body=body,
        ContentType="application/json",
    )
    logger.info("Uploaded %d bytes → b2://%s/%s", len(body), B2_BUCKET_NAME, ALGORA_OBJECT_KEY)


# ── Discord alerts ─────────────────────────────────────────────────────────────

def send_discord_alert(bounty: dict):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("No Discord webhook — skipping alert.")
        return

    amount = bounty["amount"]
    urgency = "🔥 HIGH VALUE" if amount >= 500 else ("💰 GOOD" if amount >= 200 else "💵 SMALL")
    lang = bounty["language"].capitalize() if bounty["language"] else "?"
    topics_str = ", ".join(bounty["topics"][:5]) if bounty["topics"] else "none"

    msg = (
        f"💰 **Algora Bounty — {urgency}**\n"
        f"**Repo:** `{bounty['repo_name'] or 'unknown'}`   🌐 {lang}\n"
        f"**Issue:** {bounty['issue_title'] or 'No title'}\n"
        f"**Bounty:** **{bounty['amount_formatted']}** USD\n"
        f"**Topics:** {topics_str}\n"
        f"**Claim:** <{bounty['algora_url']}>\n"
        f"**Issue:** <{bounty['issue_url']}>\n"
        f"⚡ Comment `/attempt` on the issue to claim it before others."
    )

    try:
        resp = _session.post(DISCORD_WEBHOOK_URL, json={"content": msg[:2000]}, timeout=15)
        resp.raise_for_status()
        time.sleep(0.5)
    except requests.RequestException as e:
        logger.error("Discord alert failed for %s: %s", bounty["id"], e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    state = download_state()
    seen_ids = set(state["seen_ids"])
    stored = state["bounties"]
    logger.info("%d bounties on file, %d IDs already seen.", len(stored), len(seen_ids))

    all_bounties = fetch_all_bounties()
    if not all_bounties:
        logger.info("No bounties fetched this run.")
        return

    logger.info("Fetched %d total open bounties.", len(all_bounties))

    new_bounties = []
    for bounty in all_bounties:
        bid = bounty["id"]
        if bid in seen_ids:
            continue
        seen_ids.add(bid)

        if bounty["amount"] < MIN_BOUNTY_USD:
            continue
        if not is_relevant(bounty):
            continue

        bounty["captured_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        stored.append(bounty)
        new_bounties.append(bounty)
        logger.info(
            "Bounty qualified: %s | %.0f USD | %s @ %s",
            bid, bounty["amount"], bounty["issue_title"][:60], bounty["repo_name"],
        )

    if new_bounties:
        stored.sort(key=lambda b: b.get("amount", 0), reverse=True)

    upload_state({"bounties": stored, "seen_ids": sorted(seen_ids)})

    for bounty in new_bounties:
        send_discord_alert(bounty)

    logger.info(
        "Run complete. %d new bounties this run. %d total on file.",
        len(new_bounties), len(stored),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error.")
        sys.exit(1)
