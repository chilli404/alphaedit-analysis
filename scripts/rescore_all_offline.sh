#!/usr/bin/env bash
# ============================================================================
# Rescore ALL per-case files with prob-pref metrics (NO GPU needed)
# Parallelized — runs 12 jobs at once for ~10x speedup.
#
# Handles corrupt/empty JSON files gracefully (skips them).
# Searches all run_* directories (not just run_000).
# Idempotent — skips directories that already have probpref_summary.json.
#
# Usage: bash scripts/rescore_all_offline.sh
#        bash scripts/rescore_all_offline.sh --force   # re-rescore everything
# Time:  ~1-2 minutes on a 14-core machine
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

JOBS=${RESCORE_JOBS:-12}
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

echo "Discovering run directories with per-case files..."

WORKFILE=$(mktemp)
trap "rm -f $WORKFILE" EXIT

# Find ALL run_* directories (not just run_000) that have per-case files
find results -type d -name "run_*" \
    -not -path "*/_archive/*" \
    -not -path "*/_archive_dev/*" \
    | sort | while read -r run_dir; do
    # Skip already rescored unless --force
    if [ "$FORCE" -eq 0 ] && [ -f "${run_dir}/probpref_summary.json" ]; then
        continue
    fi
    # Check for per-case files
    if find "$run_dir" -maxdepth 1 -name "100_edits-case_*.json" -print -quit 2>/dev/null | grep -q .; then
        echo "$run_dir"
    fi
done > "$WORKFILE"

TOTAL=$(wc -l < "$WORKFILE" | tr -d ' ')
if [ "$TOTAL" -eq 0 ]; then
    DONE=$(find results -name "probpref_summary.json" -not -path "*/_archive/*" | wc -l | tr -d ' ')
    echo "Nothing to rescore. $DONE directories already have probpref_summary.json."
    echo "Use --force to re-rescore everything."
    exit 0
fi

echo "Found $TOTAL directories to rescore. Running $JOBS in parallel."
echo ""

FAILFILE=$(mktemp)
trap "rm -f $WORKFILE $FAILFILE" EXIT

cat "$WORKFILE" | xargs -P "$JOBS" -n 1 bash -c '
    dir="$1"
    failfile="$2"
    label=$(echo "$dir" | sed "s|results/||; s|/run_[0-9]*||")
    output=$(uv run python scripts/eval_prob_preference.py rescore \
        --result_dir "$dir" --output "$dir/probpref_summary.json" 2>&1)
    if [ $? -eq 0 ]; then
        echo "  OK: $label"
    else
        echo "  FAIL: $label"
        echo "$label" >> "$failfile"
    fi
' _ {} "$FAILFILE"

DONE=$(find results -name "probpref_summary.json" -not -path "*/_archive/*" | wc -l | tr -d ' ')
FAILED=$(wc -l < "$FAILFILE" 2>/dev/null | tr -d ' ')

echo ""
echo "============================================"
echo "DONE: $DONE total probpref_summary.json files"
if [ "$FAILED" -gt 0 ]; then
    echo "FAILED: $FAILED directories (see below)"
    echo "--------------------------------------------"
    cat "$FAILFILE"
fi
echo "============================================"
