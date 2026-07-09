#!/usr/bin/env bash
set -euo pipefail

# MVE4: Conflict Sequence Stress Test
# Tests AlphaEdit under conflicting sequential edits (same subject, different objects).
# Provides "added value" required for TMLR reproducibility certification.
#
# Usage: bash scripts/run_mve4_conflict_seq.sh [SEED]
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

echo "=== MVE4: Conflict Sequence Stress Test (seed=$SEED) ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Project: $PROJECT_DIR"

cd "$PROJECT_DIR"

# Step 1: Generate the conflict dataset
echo "[1/2] Generating conflict dataset..."
uv run python src/conflict_dataset.py \
    --seed "$SEED" \
    --output_dir vendor/AlphaEdit/data

# Step 2: Run AlphaEdit with sequential single edits on conflict data
echo "[2/2] Running conflict sequence experiment..."
uv run python src/seeded_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name AlphaEdit \
    --model_name "$MODEL_NAME" \
    --hparams_fname Llama3-8B.json \
    --ds_name mcf \
    --dataset_size_limit 200 \
    --num_edits 1 \
    --downstream_eval_steps 0 \
    --conserve_memory

echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
