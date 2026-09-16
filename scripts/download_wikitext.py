#!/usr/bin/env python3
"""
Download WikiText-103 test split and save as JSON for S3 upload.

This avoids runtime dependency on the HuggingFace datasets library to fetch
wikitext (which fails behind Artifactory proxies).

Usage:
    uv run python scripts/download_wikitext.py
    aws s3 cp data/dsets/wikitext_103_test.json s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/dsets/

Output:
    data/dsets/wikitext_103_test.json  (~5MB, list of text strings)
"""

import json
from pathlib import Path

from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_DIR / "data" / "dsets"
DEST_FILE = DEST_DIR / "wikitext_103_test.json"


def main():
    if DEST_FILE.exists():
        print(f"Already exists: {DEST_FILE}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading WikiText-103 test split...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    except (ValueError, FileNotFoundError):
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")

    texts = [item["text"] for item in ds]
    print(f"  {len(texts)} entries")

    with open(DEST_FILE, "w") as f:
        json.dump(texts, f)

    size_mb = DEST_FILE.stat().st_size / 1024 / 1024
    print(f"  Saved: {DEST_FILE} ({size_mb:.1f} MB)")
    print()
    print("Upload to S3 with:")
    print(f"  aws s3 cp {DEST_FILE} s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/dsets/")


if __name__ == "__main__":
    main()
