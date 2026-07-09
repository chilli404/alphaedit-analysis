#!/usr/bin/env bash
set -euo pipefail

# Second Model Experiment: Mistral-7B-Instruct-v0.3
# Tests whether AlphaEdit's advantage generalizes beyond Llama-3-8B.
#
# NOTE: First run on Mistral-7B will compute covariance stats from Wikipedia
# (approx 30 minutes extra on first run). Stats are cached for subsequent runs.
#
# Usage:
#   bash scripts/run_second_model.sh [SEED] [ALG_NAME]
#   bash scripts/run_second_model.sh 42 AlphaEdit    # AlphaEdit on Mistral
#   bash scripts/run_second_model.sh 42 MEMIT        # MEMIT on Mistral
#   bash scripts/run_second_model.sh 42 both         # Both (default)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
ALG="${2:-both}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Mistral-7B.json}"

echo "=== Second Model Experiment ==="
echo "  Model: $MODEL_NAME"
echo "  Hparams: $HPARAMS_FNAME"
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

run_second_model() {
    local alg_name="$1"

    echo "--- $alg_name on $MODEL_NAME (seed=$SEED) ---"

    uv run python src/seeded_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name "$alg_name" \
        --model_name "$MODEL_NAME" \
        --hparams_fname "$HPARAMS_FNAME" \
        --ds_name mcf \
        --dataset_size_limit 2000 \
        --num_edits 100 \
        --downstream_eval_steps 5 \
        --conserve_memory

    echo "--- $alg_name on $MODEL_NAME: DONE ---"
    echo ""
}

case "$ALG" in
    AlphaEdit)
        run_second_model "AlphaEdit"
        ;;
    MEMIT)
        run_second_model "MEMIT"
        ;;
    both)
        run_second_model "AlphaEdit"
        run_second_model "MEMIT"
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
        exit 1
        ;;
esac

echo "=== Second model experiment complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: vendor/AlphaEdit/results/"
