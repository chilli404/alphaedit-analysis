#!/usr/bin/env bash
set -euo pipefail

# MVE3: AlphaEdit on zsRE
# Confirms AlphaEdit advantage is not CounterFact-only.
#
# Usage: bash scripts/run_mve3_alphaedit_zsre.sh [SEED]
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

echo "=== MVE3: AlphaEdit on zsRE (seed=$SEED) ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"

uv run python src/runners/seeded_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name AlphaEdit \
    --model_name "$MODEL_NAME" \
    --hparams_fname Llama3-8B.json \
    --ds_name zsre \
    --dataset_size_limit 2000 \
    --num_edits 100 \
    --downstream_eval_steps 20 \
    --conserve_memory

echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
