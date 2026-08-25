"""
download_leads.py — standalone script to pull the current
leads_pipeline.json from B2 onto your local machine, and report how
many leads have been captured.

This is a MIRROR, not a merge: the local file is fully overwritten
with whatever's currently in B2 every time you run this. B2 is the
source of truth; the local copy is just for you to open/inspect/feed
into something else.

Usage:
    python download_leads.py                 # download + save + print summary
    python download_leads.py --count-only     # just print counts, don't save a file
    python download_leads.py --path /some/other/file.json

Required environment variables:
    B2_KEY_ID, B2_APPLICATION_KEY   — B2 credentials
    B2_BUCKET_NAME                  — defaults to "hnscraper"
    B2_ENDPOINT                     — defaults to the us-west-004 S3 endpoint
    B2_OBJECT_KEY                   — defaults to "leads_pipeline.json"
"""
import argparse
import json
import os

import boto3
from botocore.exceptions import ClientError

B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APPLICATION_KEY = os.environ.get("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_NAME", "hnscraper")
B2_OBJECT_KEY = os.environ.get("B2_OBJECT_KEY", "leads_pipeline.json")

DEFAULT_LOCAL_PATH = os.path.expanduser("~/python/leads_pipeline.json")


def fetch_leads_from_b2() -> list:
    if not B2_KEY_ID or not B2_APPLICATION_KEY:
        raise RuntimeError("B2_KEY_ID / B2_APPLICATION_KEY are not set.")
    client = boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APPLICATION_KEY,
    )
    try:
        obj = client.get_object(Bucket=B2_BUCKET_NAME, Key=B2_OBJECT_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


def count_leads(leads: list) -> dict:
    """Returns a small breakdown: total, priority-only, and per-keyword frequency."""
    total = len(leads)
    priority = sum(1 for l in leads if l.get("priority"))

    keyword_counts = {}
    for lead in leads:
        for kw in lead.get("matched_keywords", []):
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

    top_keywords = sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "total_leads": total,
        "priority_leads": priority,
        "non_priority_leads": total - priority,
        "top_keywords": top_keywords,
    }


def print_summary(summary: dict):
    print(f"Total leads captured:     {summary['total_leads']}")
    print(f"  Priority (contract/fractional/etc mentioned): {summary['priority_leads']}")
    print(f"  Non-priority:            {summary['non_priority_leads']}")
    if summary["top_keywords"]:
        print("  Top matched keywords:")
        for kw, count in summary["top_keywords"]:
            print(f"    {kw}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Download and summarize HN leads from B2.")
    parser.add_argument(
        "--path", default=DEFAULT_LOCAL_PATH,
        help=f"Local path to save the JSON file (default: {DEFAULT_LOCAL_PATH})",
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Just fetch and print counts — don't write a local file.",
    )
    args = parser.parse_args()

    leads = fetch_leads_from_b2()
    summary = count_leads(leads)

    if not args.count_only:
        os.makedirs(os.path.dirname(args.path), exist_ok=True)
        
        # --- NEW: Deduplication Logic ---
        processed_file = os.path.join(os.path.dirname(args.path), "processed_ids.json")
        processed_ids = []
        if os.path.exists(processed_file):
            try:
                with open(processed_file, "r") as f:
                    processed_ids = json.load(f)
            except Exception:
                pass
                
        fresh_leads = [lead for lead in leads if lead["id"] not in processed_ids]
        
        with open(args.path, "w") as f:
            json.dump(fresh_leads, f, indent=2)
        print(f"Total in B2: {len(leads)}. Filtered out {len(leads) - len(fresh_leads)} processed leads.")
        print(f"Saved {len(fresh_leads)} FRESH leads to {args.path}\n")

    print_summary(summary)


if __name__ == "__main__":
    main()