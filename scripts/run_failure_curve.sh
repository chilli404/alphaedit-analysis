#!/usr/bin/env bash
set -euo pipefail

# Failure Characterization Curve
# Measures preservation metrics as a function of total edit count.
# Runs both AlphaEdit and MEMIT at each edit count level.
#
# This produces the key figure for the paper: a plot showing how
# metrics degrade as more edits are applied, and where (if anywhere)
# AlphaEdit's advantage over MEMIT disappears.
#
# Features:
#   - Resumable: skips (algorithm, edit_count) pairs that already have results
#   - Extended range: tests up to 10,000 edits (matching the original paper)
#
# Usage:
#   bash scripts/run_failure_curve.sh [SEED] [ALG_NAME]
#   bash scripts/run_failure_curve.sh 42 AlphaEdit    # Just AlphaEdit
#   bash scripts/run_failure_curve.sh 42 MEMIT        # Just MEMIT
#   bash scripts/run_failure_curve.sh 42 both         # Both (default)
#
# The edit counts tested: 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

SEED="${1:-42}"
ALG="${2:-both}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

EDIT_COUNTS=(500 1000 1500 2000 3000 5000 7500 10000)

echo "=== Failure Characterization Curve ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Edit counts: ${EDIT_COUNTS[*]}"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Results directory for checking resumability
RESULTS_BASE="vendor/AlphaEdit/results"

check_already_complete() {
    local alg_name="$1"
    local edit_count="$2"
    local seed="$3"

    # Check if result JSON files exist for this (alg, edit_count) combination.
    # AlphaEdit writes results as: results/{alg}/run_{id}/{num_edits}-edits-case_N.json
    # Each case file corresponds to one edit evaluation, so case_count ≈ dataset_size_limit.
    # A run covers this edit count only if it has at least edit_count case files.
    local alg_dir="$RESULTS_BASE/$alg_name"
    if [[ ! -d "$alg_dir" ]]; then
        return 1  # Not complete
    fi

    for run_dir in "$alg_dir"/run_*/; do
        if [[ ! -d "$run_dir" ]]; then
            continue
        fi
        local case_count
        case_count=$(find "$run_dir" -name "*_edits-case_*.json" 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$case_count" -ge "$edit_count" ]]; then
            echo "  REUSE: $alg_name at $edit_count edits (covered by $run_dir with $case_count case files)"
            return 0  # Already covered
        fi
    done

    return 1  # Not complete
}

run_at_edit_count() {
    local alg_name="$1"
    local edit_count="$2"
    local seed="$3"

    # Check for existing results (resumability)
    if check_already_complete "$alg_name" "$edit_count" "$seed"; then
        return 0
    fi

    echo "--- $alg_name at $edit_count edits (seed=$seed) ---"

    uv run python src/seeded_runner.py \
        --seed "$seed" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name "$alg_name" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$edit_count" \
        --num_edits 100 \
        --downstream_eval_steps 20 \
        --conserve_memory

    echo "--- $alg_name at $edit_count edits: DONE ---"
    echo ""
}

FAILED=0

for count in "${EDIT_COUNTS[@]}"; do
    case "$ALG" in
        AlphaEdit)
            run_at_edit_count "AlphaEdit" "$count" "$SEED" || FAILED=$((FAILED + 1))
            ;;
        MEMIT)
            run_at_edit_count "MEMIT" "$count" "$SEED" || FAILED=$((FAILED + 1))
            ;;
        both)
            run_at_edit_count "AlphaEdit" "$count" "$SEED" || FAILED=$((FAILED + 1))
            run_at_edit_count "MEMIT" "$count" "$SEED" || FAILED=$((FAILED + 1))
            ;;
        *)
            echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
            exit 1
            ;;
    esac
done

echo "=== Failure curve complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: vendor/AlphaEdit/results/"
if [[ $FAILED -gt 0 ]]; then
    echo "  WARNING: $FAILED runs failed (partial results saved; re-run to resume)"
fi
echo ""
echo "Next: python analysis/aggregate.py && python analysis/plots.py"
