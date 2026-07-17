#!/usr/bin/env bash
set -euo pipefail

# Run post-hoc mechanism analysis on saved AlphaEdit checkpoints.
#
# Loads checkpoint weights + cache_c at each batch boundary and computes:
#   - Weight-spectrum distortion (relative to base model)
#   - Cache geometry (effective rank, condition number, consumption)
#   - Key-space crowding indicators
#
# Usage:
#   bash scripts/run_mechanism_analysis.sh 42
#   bash scripts/run_mechanism_analysis.sh 2024

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config (.env has HF_TOKEN, etc.)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:?Usage: $0 <seed>}"
CHECKPOINT_BASE="${CHECKPOINT_DIR:-/s3-data/continual-learning/alphaedit/checkpoints}/AlphaEdit/seed${SEED}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SKIP_BASE_MODEL="${SKIP_BASE_MODEL:-false}"

# Output: write to S3 if available, else local
if [[ -d "/s3-data/continual-learning/alphaedit/results" ]]; then
    OUTPUT_DIR="/s3-data/continual-learning/alphaedit/results/mechanism_analysis/seed${SEED}"
else
    OUTPUT_DIR="$PROJECT_DIR/results/mechanism_analysis"
fi

# Batch indices: checkpoints are at batch_9, batch_19, ..., batch_99 (0-indexed)
BATCH_INDICES="${BATCH_INDICES:-9 19 29 39 49 59 69 79 89 99}"

echo "=========================================="
echo "Mechanism Analysis"
echo "  Seed:        ${SEED}"
echo "  Checkpoints: ${CHECKPOINT_BASE}"
echo "  Model:       ${MODEL_NAME}"
echo "  Batches:     ${BATCH_INDICES}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  CUDA device: ${CUDA_DEVICE}"
echo "  HF_TOKEN:    ${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
echo "  Started:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

# Check checkpoint directory exists
if [ ! -d "${CHECKPOINT_BASE}" ]; then
    echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_BASE}"
    echo "  Available S3 paths:"
    ls /s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/ 2>/dev/null || echo "  (S3 not mounted)"
    exit 1
fi

# List available checkpoints
echo ""
echo "Available checkpoints:"
ls -d "${CHECKPOINT_BASE}"/batch_* 2>/dev/null | while read d; do
    batch=$(basename "$d" | sed 's/batch_//')
    echo "  batch_${batch}"
done
echo ""

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

EXTRA_ARGS=""
if [[ "$SKIP_BASE_MODEL" == "true" ]]; then
    EXTRA_ARGS="--skip_base_model"
fi

uv run python src/mechanism/mechanism_analyzer.py \
    --seed "${SEED}" \
    --checkpoint_base "${CHECKPOINT_BASE}" \
    --model_name "${MODEL_NAME}" \
    --hparams_fname "${HPARAMS_FNAME}" \
    --batch_indices ${BATCH_INDICES} \
    --output_dir "${OUTPUT_DIR}" \
    $EXTRA_ARGS

echo ""
echo "Mechanism analysis complete."
echo "  Results: ${OUTPUT_DIR}/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
