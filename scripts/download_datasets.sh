#!/usr/bin/env bash
set -euo pipefail

# Download all AlphaEdit datasets locally for upload to S3.
# These files cannot be downloaded from the K8s cluster (network restricted).
#
# Usage:
#   bash scripts/download_datasets.sh
#   aws s3 sync data/dsets/ s3://grainger-mlops-pimmachinelearning-prod/continual-learning/alphaedit/dsets/
#
# Files downloaded:
#   data/dsets/counterfact.json          (~48MB)
#   data/dsets/multi_counterfact.json    (~92MB)
#   data/dsets/zsre_mend_eval.json       (~25MB)
#   data/dsets/known_1000.json           (~0.5MB)
#   data/dsets/attribute_snippets.json   (~13MB)
#   data/dsets/idf.npy                   (~1MB)
#   data/dsets/tfidf_vocab.json          (~3MB)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEST_DIR="$PROJECT_DIR/data/dsets"

BASE_URL="https://memit.baulab.info/data/dsets"

FILES=(
    counterfact.json
    multi_counterfact.json
    zsre_mend_eval.json
    known_1000.json
    attribute_snippets.json
    idf.npy
    tfidf_vocab.json
)

mkdir -p "$DEST_DIR"

echo "=== Downloading AlphaEdit Datasets ==="
echo "  Destination: $DEST_DIR"
echo ""

for file in "${FILES[@]}"; do
    dest="$DEST_DIR/$file"
    if [[ -f "$dest" ]]; then
        echo "  Already exists: $file"
    else
        echo "  Downloading: $file"
        curl -L --fail -o "$dest" "$BASE_URL/$file"
    fi
done

echo ""
echo "=== Download complete ==="
echo ""
echo "Upload to S3 with:"
echo "  aws s3 sync $DEST_DIR s3://grainger-mlops-pimmachinelearning-prod/continual-learning/alphaedit/dsets/"
