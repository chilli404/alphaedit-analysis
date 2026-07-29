#!/usr/bin/env bash
set -euo pipefail

# Build anonymized submission zip for supplementary material.
#
# Includes: src/, scripts/ (scrubbed), analysis/, configs/, tests/,
#           pyproject.toml, uv.lock, README, .gitmodules, vendor/AlphaEdit
#
# Excludes: .git/, .env, .venv/, CLAUDE.md, sky/, results/, data/, logs/,
#           all gitignored files, all Grainger-specific infra scripts
#
# Usage:
#   bash scripts/build_submission.sh
#   # Output: submission/alphaedit-reproducibility-code.zip

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/submission/alphaedit-reproducibility-code"
ZIP_OUTPUT="$PROJECT_DIR/submission/alphaedit-reproducibility-code.zip"

echo "=== Building Anonymized Submission Package ==="
echo "  Source: $PROJECT_DIR"
echo "  Output: $ZIP_OUTPUT"
echo ""

# Clean previous build
rm -rf "$PROJECT_DIR/submission"
mkdir -p "$BUILD_DIR"

# ─── Copy included directories ────────────────────────────────────────────────

echo "Copying source files..."

# Core code
cp -r "$PROJECT_DIR/src" "$BUILD_DIR/src"
cp -r "$PROJECT_DIR/scripts" "$BUILD_DIR/scripts"
cp -r "$PROJECT_DIR/analysis" "$BUILD_DIR/analysis"
cp -r "$PROJECT_DIR/configs" "$BUILD_DIR/configs"
cp -r "$PROJECT_DIR/tests" "$BUILD_DIR/tests"

# Top-level files
cp "$PROJECT_DIR/pyproject.toml" "$BUILD_DIR/"
cp "$PROJECT_DIR/uv.lock" "$BUILD_DIR/"
cp "$PROJECT_DIR/.python-version" "$BUILD_DIR/"
cp "$PROJECT_DIR/.gitignore" "$BUILD_DIR/"
cp "$PROJECT_DIR/.gitmodules" "$BUILD_DIR/"
cp "$PROJECT_DIR/README.md" "$BUILD_DIR/"

# Vendor submodule — include the code but not its .git or data/stats
if [[ -d "$PROJECT_DIR/vendor/AlphaEdit" ]]; then
    echo "Copying vendor submodule (without .git, data/stats, results)..."
    mkdir -p "$BUILD_DIR/vendor"
    rsync -a \
        --exclude='.git' \
        --exclude='data/stats/' \
        --exclude='data/multi_counterfact.json' \
        --exclude='data/counterfact.json' \
        --exclude='data/zsre_mend_eval.json' \
        --exclude='data/MQuAKE-CF-3k-v2.json' \
        --exclude='data/attribute_snippets.json' \
        --exclude='data/idf.npy' \
        --exclude='data/tfidf_vocab.json' \
        --exclude='results/' \
        --exclude='null_space_project.pt' \
        --exclude='*.bin' \
        --exclude='*.safetensors' \
        "$PROJECT_DIR/vendor/AlphaEdit/" "$BUILD_DIR/vendor/AlphaEdit/"
fi

# ─── Remove excluded files ────────────────────────────────────────────────────

echo "Removing excluded files..."

# Remove all __pycache__
find "$BUILD_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$BUILD_DIR" -name "*.pyo" -delete 2>/dev/null || true

# Remove .DS_Store
find "$BUILD_DIR" -name ".DS_Store" -delete 2>/dev/null || true

# Remove this build script from the copy
rm -f "$BUILD_DIR/scripts/build_submission.sh"

# Remove .env / CLAUDE.md if accidentally copied
rm -f "$BUILD_DIR/.env"
rm -f "$BUILD_DIR/CLAUDE.md"

# ─── Remove all SkyPilot / Grainger infra ─────────────────────────────────────

echo "Removing SkyPilot and Grainger-specific infrastructure..."

# SkyPilot (entire directory — cloud orchestration is infra, not science)
rm -rf "$BUILD_DIR/sky" 2>/dev/null || true

# Grainger-specific scripts (Artifactory, internal S3, remote cluster setup)
rm -f "$BUILD_DIR/scripts/remote_setup.sh"
rm -f "$BUILD_DIR/scripts/link_dsets.sh"
rm -f "$BUILD_DIR/scripts/link_stats.sh"
rm -f "$BUILD_DIR/scripts/download_datasets.sh"
rm -f "$BUILD_DIR/scripts/download_wikitext.py"
rm -f "$BUILD_DIR/scripts/download_mmlu.py"

# Model download utility with Artifactory fallback
rm -f "$BUILD_DIR/src/util/model_download.py"
rm -f "$BUILD_DIR/src/model_download.py"

# ─── Scrub any remaining identifying info in kept files ───────────────────────

echo "Scrubbing remaining identifying references..."

# Artifactory / Grainger URLs
find "$BUILD_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.toml" \) -exec \
    sed -i '' 's|graingerinc\.jfrog\.io[^ "]*||g' {} + 2>/dev/null || true

# S3 bucket names
find "$BUILD_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" \) -exec \
    sed -i '' 's|grainger-mlops-pimmachinelearning-[a-z]*|BUCKET_PLACEHOLDER|g' {} + 2>/dev/null || true

# Generic "grainger" references
find "$BUILD_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.toml" -o -name "*.md" \) -exec \
    sed -i '' 's|[Gg]rainger[^ ]*||g' {} + 2>/dev/null || true

# Email addresses
find "$BUILD_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.toml" \) -exec \
    sed -i '' 's|[a-zA-Z0-9._%+-]*@grainger\.com||g' {} + 2>/dev/null || true

# Internal S3 mount paths
find "$BUILD_DIR" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" \) -exec \
    sed -i '' 's|/s3-data/continual-learning/alphaedit|/data/alphaedit|g' {} + 2>/dev/null || true

# ─── Remove .gitmodules sky reference (keep only vendor) ──────────────────────

# .gitmodules only references vendor/AlphaEdit, which is fine — no changes needed

# ─── Verify no identifying info remains ──────────────────────────────────────

echo ""
echo "Verification — searching for remaining identifying info..."
FOUND=0

if grep -rl "grainger" "$BUILD_DIR" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.toml" --include="*.md" 2>/dev/null; then
    echo "  WARNING: 'grainger' still found in above files"
    FOUND=1
fi

if grep -rl "kamal\|chilukuri\|xksc003" "$BUILD_DIR" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.toml" --include="*.md" 2>/dev/null; then
    echo "  WARNING: Personal identifiers still found"
    FOUND=1
fi

if grep -rl "jfrog" "$BUILD_DIR" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.toml" 2>/dev/null; then
    echo "  WARNING: 'jfrog' references still found"
    FOUND=1
fi

if [[ $FOUND -eq 0 ]]; then
    echo "  PASS: No identifying information detected."
fi

# ─── Create zip ───────────────────────────────────────────────────────────────

echo ""
echo "Creating zip archive..."
cd "$PROJECT_DIR/submission"
zip -r -q "alphaedit-reproducibility-code.zip" "alphaedit-reproducibility-code/"

# Summary
echo ""
echo "=== Done ==="
echo "  Output: $ZIP_OUTPUT"
echo "  Size:   $(du -sh "$ZIP_OUTPUT" | cut -f1)"
echo ""
echo "  Files included:"
find "$BUILD_DIR" -type f | wc -l | xargs echo "   "
echo ""
echo "  Directory structure:"
find "$BUILD_DIR" -type d -maxdepth 2 | sed "s|$BUILD_DIR|.|" | sort
echo ""
echo "Next steps:"
echo "  1. Inspect: unzip -l $ZIP_OUTPUT | head -80"
echo "  2. Verify:  grep -r 'grainger\|kamal\|jfrog\|xksc' submission/alphaedit-reproducibility-code/"
echo "  3. Upload to OpenReview supplementary materials"
echo "     OR host at: https://anonymous.4open.science"
