#!/usr/bin/env python3
"""
Download MMLU subset (4 categories) and save as JSON for S3 upload.

This avoids runtime dependency on the HuggingFace datasets library to fetch
MMLU (which fails behind Artifactory proxies and requires trust_remote_code).

Usage:
    uv run python scripts/download_mmlu.py
    aws s3 cp data/dsets/mmlu_subset.json s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/dsets/

Output:
    data/dsets/mmlu_subset.json  (~0.5MB, dict of category -> {test: [...], validation: [...]})
"""

import json
from pathlib import Path

from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_DIR / "data" / "dsets"
DEST_FILE = DEST_DIR / "mmlu_subset.json"

# Same categories used in capability_probe.py
CATEGORIES = [
    "abstract_algebra",
    "world_religions",
    "us_foreign_policy",
    "college_biology",
]


def download_category(category: str) -> dict | None:
    """Download one MMLU category, returns {test: [...], validation: [...]} or None."""
    # Try cais/mmlu first (newer), then lukaemon/mmlu (older)
    for repo in ["cais/mmlu", "lukaemon/mmlu"]:
        try:
            if repo == "cais/mmlu":
                test_split = "test"
                val_split = "validation"
            else:
                test_split = "test"
                val_split = "train"  # lukaemon uses train for few-shot

            ds = load_dataset(repo, category, split=test_split, trust_remote_code=True)
            val_ds = load_dataset(repo, category, split=val_split, trust_remote_code=True)

            # Normalize to list of dicts
            test_items = [dict(item) for item in ds]
            val_items = [dict(item) for item in val_ds]

            print(f"  {category}: {len(test_items)} test, {len(val_items)} validation (from {repo})")
            return {"test": test_items, "validation": val_items}

        except Exception as e:
            print(f"  {category}: failed with {repo} ({e}), trying next...")
            continue

    print(f"  {category}: FAILED (all sources)")
    return None


def main():
    if DEST_FILE.exists():
        print(f"Already exists: {DEST_FILE}")
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MMLU subset...")
    data = {}
    for category in CATEGORIES:
        result = download_category(category)
        if result is not None:
            data[category] = result

    with open(DEST_FILE, "w") as f:
        json.dump(data, f)

    size_mb = DEST_FILE.stat().st_size / 1024 / 1024
    print(f"\nSaved: {DEST_FILE} ({size_mb:.1f} MB)")
    print(f"Categories: {list(data.keys())}")
    print()
    print("Upload to S3 with:")
    print(f"  aws s3 cp {DEST_FILE} s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/dsets/")


if __name__ == "__main__":
    main()
