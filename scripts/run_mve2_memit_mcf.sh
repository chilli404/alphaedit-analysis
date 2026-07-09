#!/usr/bin/env bash
set -euo pipefail

# MVE2: MEMIT baseline on MultiCounterFact
# Matched baseline comparison for AlphaEdit.
#
# Usage: bash scripts/run_mve2_memit_mcf.sh [SEED]
# Default seed: 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

SEED="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

echo "=== MVE2: MEMIT on MultiCounterFact (seed=$SEED) ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"

uv run python src/seeded_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name MEMIT \
    --model_name "$MODEL_NAME" \
    --hparams_fname Llama3-8B.json \
    --ds_name mcf \
    --dataset_size_limit 2000 \
    --num_edits 100 \
    --downstream_eval_steps 5 \
    --conserve_memory

echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
