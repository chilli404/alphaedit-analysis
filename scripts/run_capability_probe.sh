#!/usr/bin/env bash
set -euo pipefail

# General Capability Probe
#
# Runs the editing pipeline with perplexity + MMLU measurement at each
# downstream evaluation step. Produces a timeseries showing whether
# general model capabilities degrade as edits accumulate.
#
# This addresses the reviewer concern: "editing may destroy general
# capabilities while still passing CounterFact-specific metrics."
#
# Outputs:
#   results/capability_probe/probe_seed{SEED}_{ALG}_{LIMIT}edits.jsonl
#   Each line = one measurement point (edit_count, perplexity, mmlu_accuracy)
#
# Usage:
#   bash scripts/run_capability_probe.sh [SEED] [ALG] [DATASET_SIZE_LIMIT]
#   bash scripts/run_capability_probe.sh 42 AlphaEdit 2000
#   bash scripts/run_capability_probe.sh 42 MEMIT 2000
#   bash scripts/run_capability_probe.sh 42 both 2000    # default

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
ALG="${2:-both}"
DATASET_SIZE_LIMIT="${3:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
# Set to "true" to include MMLU (slower but more informative)
INCLUDE_MMLU="${INCLUDE_MMLU:-true}"

echo "=== General Capability Probe ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Include MMLU: $INCLUDE_MMLU"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

run_probe() {
    local alg_name="$1"

    echo "--- Capability probe: $alg_name (seed=$SEED, limit=$DATASET_SIZE_LIMIT) ---"

    local mmlu_flag=""
    if [[ "$INCLUDE_MMLU" == "false" ]]; then
        mmlu_flag="--no_mmlu"
    fi

    uv run python src/capability_probe_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name "$alg_name" \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits 100 \
        --probe_interval 5 \
        $mmlu_flag

    echo "--- $alg_name capability probe: DONE ---"
    echo ""
}

case "$ALG" in
    AlphaEdit)
        run_probe "AlphaEdit"
        ;;
    MEMIT)
        run_probe "MEMIT"
        ;;
    both)
        run_probe "AlphaEdit"
        run_probe "MEMIT"
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
        exit 1
        ;;
esac

echo "=== Capability probe complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: results/capability_probe/"
echo ""
echo "Next: python analysis/plots.py  (generates capability degradation figure)"
