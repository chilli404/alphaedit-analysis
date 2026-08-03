#!/usr/bin/env bash
set -euo pipefail

# MVE1-GPT-J: AlphaEdit on MultiCounterFact (GPT-J-6B)
# Cross-architecture reproduction: tests if age-biased forgetting replicates.
#
# Usage: bash scripts/run_mve1_gptj_mcf.sh [SEED]
# Default seed: 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-EleutherAI/gpt-j-6b}"

SEED="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

echo "=== MVE1-GPT-J: AlphaEdit on MultiCounterFact (seed=$SEED) ==="
echo "  Model: $MODEL_NAME"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"

uv run python src/runners/seeded_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name AlphaEdit \
    --model_name "$MODEL_NAME" \
    --hparams_fname EleutherAI_gpt-j-6B.json \
    --ds_name mcf \
    --dataset_size_limit 2000 \
    --num_edits 100 \
    --downstream_eval_steps 20 \
    --conserve_memory

echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
