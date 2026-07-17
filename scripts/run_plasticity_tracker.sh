#!/usr/bin/env bash
# Run online plasticity + projection tracker for AlphaEdit collapse mechanism study.
#
# Captures per-batch metrics inline during editing:
#   - Raw vs projected update norms and cosine alignment
#   - Projection removed fraction (||ΔW_raw - ΔW_proj||_F / ||ΔW_raw||_F)
#   - Solve condition number and residual
#   - Checkpoint model weights + cache_c for post-hoc analysis
#
# Usage:
#   bash scripts/run_plasticity_tracker.sh 42
#   bash scripts/run_plasticity_tracker.sh 42 10000    # 10K edits
#   bash scripts/run_plasticity_tracker.sh 2024 5000   # 5K edits, seed 2024
#
# On SkyPilot cluster (checkpoints saved to /s3-data/...):
#   bash scripts/run_plasticity_tracker.sh 42 10000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:?Usage: $0 <seed> [dataset_size_limit] [start_from_batch]}"
DATASET_SIZE_LIMIT="${2:-10000}"
START_FROM_BATCH="${3:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"

echo "=========================================="
echo "Plasticity + Projection Tracker"
echo "  Seed:           ${SEED}"
echo "  Dataset limit:  ${DATASET_SIZE_LIMIT}"
echo "  Batch size:     ${NUM_EDITS}"
echo "  Save interval:  ${SAVE_INTERVAL}"
echo "  Start batch:    ${START_FROM_BATCH}"
echo "  Model:          ${MODEL_NAME}"
echo "  CUDA device:    ${CUDA_DEVICE}"
echo "  Started:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

cd "$PROJECT_DIR"

uv run python src/mechanism/plasticity_tracker.py \
    --seed "${SEED}" \
    --cuda_device "${CUDA_DEVICE}" \
    --model_name "${MODEL_NAME}" \
    --hparams_fname "${HPARAMS_FNAME}" \
    --ds_name mcf \
    --dataset_size_limit "${DATASET_SIZE_LIMIT}" \
    --num_edits "${NUM_EDITS}" \
    --save_interval "${SAVE_INTERVAL}" \
    --start_from_batch "${START_FROM_BATCH}" \
    --conserve_memory

echo ""
echo "Plasticity tracking complete."
echo "  Results: results/plasticity_tracking/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
