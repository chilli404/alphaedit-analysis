#!/usr/bin/env bash
# freeze_eval_commit.sh — Generate configs/eval_commit_freeze.json
#
# Records the exact state of the evaluation environment for provenance.
# Run this ONCE after finalizing evaluation code, before starting paper runs.
#
# Usage:
#   bash scripts/freeze_eval_commit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="$PROJECT_ROOT/configs/eval_commit_freeze.json"

echo "Generating evaluation commit freeze..."

# Check for uncommitted changes
if ! git -C "$PROJECT_ROOT" diff --quiet HEAD 2>/dev/null; then
    echo "WARNING: Uncommitted changes detected. Freeze will record dirty state."
    echo "  Consider committing all changes before freezing."
fi

# Get current HEAD commit
MAIN_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")

# Get vendor submodule commit
VENDOR_DIR="$PROJECT_ROOT/vendor/AlphaEdit"
if [ -d "$VENDOR_DIR/.git" ] || [ -f "$VENDOR_DIR/.git" ]; then
    VENDOR_COMMIT=$(git -C "$VENDOR_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
else
    VENDOR_COMMIT="submodule_not_initialized"
fi

# Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "unknown")

# Hash uv.lock
UV_LOCK="$PROJECT_ROOT/uv.lock"
if [ -f "$UV_LOCK" ]; then
    UV_LOCK_HASH=$(shasum -a 256 "$UV_LOCK" | awk '{print $1}')
else
    UV_LOCK_HASH="file_not_found"
fi

# Hash eval_config.yaml
EVAL_CONFIG="$PROJECT_ROOT/configs/eval_config.yaml"
if [ -f "$EVAL_CONFIG" ]; then
    EVAL_CONFIG_HASH=$(shasum -a 256 "$EVAL_CONFIG" | awk '{print $1}')
else
    EVAL_CONFIG_HASH="file_not_found"
fi

# Hash metric_registry.yaml
METRIC_REGISTRY="$PROJECT_ROOT/configs/metric_registry.yaml"
if [ -f "$METRIC_REGISTRY" ]; then
    METRIC_REGISTRY_HASH=$(shasum -a 256 "$METRIC_REGISTRY" | awk '{print $1}')
else
    METRIC_REGISTRY_HASH="file_not_found"
fi

# Get today's date
FROZEN_AT=$(date -u +"%Y-%m-%d")

# Write JSON
cat > "$OUTPUT" << EOF
{
  "frozen_at": "$FROZEN_AT",
  "main_repo_commit": "$MAIN_COMMIT",
  "vendor_submodule_commit": "$VENDOR_COMMIT",
  "python_version": "$PYTHON_VERSION",
  "uv_lock_hash": "$UV_LOCK_HASH",
  "eval_config_hash": "$EVAL_CONFIG_HASH",
  "metric_registry_hash": "$METRIC_REGISTRY_HASH"
}
EOF

echo "Freeze written to: $OUTPUT"
echo "  Main commit:    $MAIN_COMMIT"
echo "  Vendor commit:  $VENDOR_COMMIT"
echo "  Python:         $PYTHON_VERSION"
echo "  uv.lock hash:   ${UV_LOCK_HASH:0:16}..."
echo "  eval_config:    ${EVAL_CONFIG_HASH:0:16}..."
echo "  metric_reg:     ${METRIC_REGISTRY_HASH:0:16}..."
