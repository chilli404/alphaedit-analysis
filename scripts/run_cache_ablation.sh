#!/usr/bin/env bash
set -euo pipefail

# Cache Ablation Experiment
#
# Tests the over-regularization hypothesis by scaling the accumulated
# cache term γ ∈ {0, 0.1, 0.25, 0.5, 1.0} at a late checkpoint.
#
# Predictions:
#   - Reducing γ → larger ||ΔW|| (update norm increases)
#   - Reducing γ → better residual attainment (edits land more precisely)
#   - Reducing γ → higher efficacy (new edits succeed)
#   - Reducing γ → lower locality (old edits get damaged)
#
# Usage:
#   bash scripts/run_cache_ablation.sh 42
#   bash scripts/run_cache_ablation.sh 42 69    # checkpoint batch (default: 69 = 7K edits)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:?Usage: $0 <seed> [checkpoint_batch]}"
CHECKPOINT_BATCH="${2:-${CHECKPOINT_BATCH:-69}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/failure_curve}"

# Output
OUTPUT_DIR="$RESULT_ROOT/cache_ablation/seed${SEED}"

echo "=========================================="
echo "Cache Ablation Experiment"
echo "  Seed:             ${SEED}"
echo "  Checkpoint batch: ${CHECKPOINT_BATCH} ($((($CHECKPOINT_BATCH + 1) * 100)) edits)"
echo "  Gamma values:     0 0.1 0.25 0.5 1.0"
echo "  Model:            ${MODEL_NAME}"
echo "  CUDA device:      ${CUDA_DEVICE}"
echo "  Output:           ${OUTPUT_DIR}"
echo "  Started:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

# Set HF_ENDPOINT before Python starts so huggingface_hub picks it up at import time.
# (seeded_runner.py achieves this via subprocess env; we must do it in the shell.)
if [[ -z "${HF_ENDPOINT:-}" ]]; then
    if curl -s --head --connect-timeout 5 https://graingerinc.jfrog.io > /dev/null 2>&1; then
        export HF_ENDPOINT="https://graingerinc.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote"
        echo "  HF_ENDPOINT set to Artifactory (pre-Python)"
    fi
fi

uv run python src/mechanism/cache_ablation_runner.py \
    --seed "${SEED}" \
    --checkpoint_batch "${CHECKPOINT_BATCH}" \
    --gamma_values 0 0.1 0.25 0.5 1.0 \
    --model_name "${MODEL_NAME}" \
    --hparams_fname "${HPARAMS_FNAME}" \
    --cuda_device "${CUDA_DEVICE}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Cache ablation complete."
echo "  Results: ${OUTPUT_DIR}/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
