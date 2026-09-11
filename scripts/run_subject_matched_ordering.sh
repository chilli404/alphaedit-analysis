#!/usr/bin/env bash
set -euo pipefail

# Subject-Matched Ordering Experiment
#
# Generates subject-matched orderings (if needed) and runs the matched ordering
# pipeline. These orderings hold per-batch relation distribution CONSTANT while
# varying key-geometry exposure, eliminating the subject-overlap confound.
#
# Usage:
#   bash scripts/run_subject_matched_ordering.sh [SEED] [ALG] [ORDERING]
#   bash scripts/run_subject_matched_ordering.sh 2024 AlphaEdit subj_matched_clustered
#   bash scripts/run_subject_matched_ordering.sh 2024 AlphaEdit subj_matched_dispersed
#   bash scripts/run_subject_matched_ordering.sh 2024   # Runs all 4 combinations
#
# Environment variables:
#   Same as run_matched_ordering.sh (CUDA_DEVICE, TARGET_EDITS, SAVE_INTERVAL, etc.)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-2024}"
ALG="${2:-${ALG_NAME:-all}}"
ORDERING="${3:-${ORDERING:-all}}"

cd "$PROJECT_DIR"

RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
STREAM_DIR="$RESULT_ROOT/matched_ordering/orderings"

# ── Step 1: Generate orderings if not present ────────────────────────────────

generate_if_needed() {
    local seed="$1"
    local clust_file="$STREAM_DIR/subj_matched_clustered_seed${seed}.json"
    local disp_file="$STREAM_DIR/subj_matched_dispersed_seed${seed}.json"

    if [[ -f "$clust_file" ]] && [[ -f "$disp_file" ]]; then
        echo "  Orderings already exist for seed $seed"
        return 0
    fi

    echo "  Generating subject-matched orderings for seed $seed..."
    uv run python src/datasets/generate_subject_matched_orderings.py \
        --seed "$seed" \
        --output_dir "$RESULT_ROOT/matched_ordering"
}

generate_if_needed "$SEED"

# ── Step 2: Run experiments ──────────────────────────────────────────────────

run_one() {
    local alg="$1"
    local ordering="$2"
    echo ""
    echo "=== Subject-Matched: $alg / $ordering / seed $SEED ==="
    bash "$SCRIPT_DIR/run_matched_ordering.sh" "$SEED" "$alg" "$ordering"
}

if [[ "$ALG" == "all" ]] && [[ "$ORDERING" == "all" ]]; then
    for alg in AlphaEdit MEMIT-Seq-lp1.0-ld0.0-cache0; do
        for ordering in subj_matched_clustered subj_matched_dispersed; do
            run_one "$alg" "$ordering"
        done
    done
elif [[ "$ORDERING" == "all" ]]; then
    for ordering in subj_matched_clustered subj_matched_dispersed; do
        run_one "$ALG" "$ordering"
    done
else
    run_one "$ALG" "$ORDERING"
fi

echo ""
echo "=== Subject-matched ordering experiment complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
