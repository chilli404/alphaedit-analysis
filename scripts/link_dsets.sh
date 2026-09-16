#!/usr/bin/env bash
set -euo pipefail

# Links AlphaEdit datasets from S3 mount (or local fallback) into vendor/AlphaEdit/data/.
# On SkyPilot clusters, reads from the S3 mount. Locally, reads from data/dsets/.
#
# Usage: bash scripts/link_dsets.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_DIR="/s3-data"

S3_DSETS_SRC="$S3_DIR/continual-learning/alphaedit/dsets"
PROJECT_DSETS_SRC="$PROJECT_DIR/data/dsets"

DSETS_SRC="$PROJECT_DSETS_SRC"
[[ -d "$S3_DSETS_SRC" ]] && DSETS_SRC="$S3_DSETS_SRC"

DSETS_DST="$PROJECT_DIR/vendor/AlphaEdit/data"

FILES=(
    counterfact.json
    multi_counterfact.json
    zsre_mend_eval.json
    known_1000.json
    attribute_snippets.json
    idf.npy
    tfidf_vocab.json
    wikitext_103_test.json
    mmlu_subset.json
)

if [[ ! -d "$DSETS_SRC" ]]; then
    echo "ERROR: Dataset source directory not found."
    echo "Checked:"
    echo "  S3:      $S3_DSETS_SRC"
    echo "  Project: $PROJECT_DSETS_SRC"
    echo ""
    echo "Download datasets first: bash scripts/download_datasets.sh"
    exit 1
fi

echo "=== Linking AlphaEdit Datasets ==="
echo "  Source: $DSETS_SRC"
echo "  Target: $DSETS_DST"

mkdir -p "$DSETS_DST"

COUNT=0
for file in "${FILES[@]}"; do
    src="$DSETS_SRC/$file"
    if [[ -f "$src" ]]; then
        ln -sf "$src" "$DSETS_DST/$file"
        COUNT=$((COUNT + 1))
    else
        echo "  WARNING: missing $file"
    fi
done

echo "  Linked $COUNT / ${#FILES[@]} dataset files."

# Also link into baselines/EvoEdit/data/ if it exists
BASELINES_DST="$PROJECT_DIR/baselines/EvoEdit/data"
if [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]]; then
    mkdir -p "$BASELINES_DST"
    BL_COUNT=0
    for file in "${FILES[@]}"; do
        src="$DSETS_SRC/$file"
        if [[ -f "$src" ]] && [[ ! -f "$BASELINES_DST/$file" ]]; then
            ln -sf "$src" "$BASELINES_DST/$file"
            BL_COUNT=$((BL_COUNT + 1))
        fi
    done
    echo "  baselines/EvoEdit: linked $BL_COUNT dataset files"
fi

echo "=== Done ==="
