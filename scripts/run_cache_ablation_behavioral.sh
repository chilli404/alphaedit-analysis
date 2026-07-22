#!/usr/bin/env bash
set -euo pipefail

# Cache Ablation — Behavioral Evaluation
#
# Applies edits with different cache scaling γ and measures
# actual token-level efficacy + retention tradeoff.
#
# Establishes causal link: γ↓ → efficacy↑, retention↓
#
# Usage:
#   bash scripts/run_cache_ablation_behavioral.sh 42
#   bash scripts/run_cache_ablation_behavioral.sh 42 69    # checkpoint batch

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/s3-data/continual-learning/alphaedit/checkpoints}"

# Output
EDITS=$(( (CHECKPOINT_BATCH + 1) * 100 ))
if [[ -d "/s3-data/continual-learning/alphaedit/results" ]]; then
    OUTPUT_DIR="/s3-data/continual-learning/alphaedit/results/cache_ablation_behavioral/seed${SEED}/${EDITS}edits/AlphaEdit"
else
    OUTPUT_DIR="$PROJECT_DIR/results/cache_ablation_behavioral/seed${SEED}/${EDITS}edits/AlphaEdit"
fi

echo "=========================================="
echo "Cache Ablation — Behavioral Evaluation"
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
# This routes model downloads through Artifactory on corporate clusters.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
    if curl -s --head --connect-timeout 5 https://graingerinc.jfrog.io > /dev/null 2>&1; then
        export HF_ENDPOINT="https://graingerinc.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote"
        echo "  HF_ENDPOINT set to Artifactory (pre-Python)"
    fi
fi

uv run python src/mechanism/cache_ablation_behavioral.py \
    --seed "${SEED}" \
    --checkpoint_batch "${CHECKPOINT_BATCH}" \
    --gamma_values 0 0.1 0.25 0.5 1.0 \
    --model_name "${MODEL_NAME}" \
    --hparams_fname "${HPARAMS_FNAME}" \
    --cuda_device "${CUDA_DEVICE}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Behavioral evaluation complete."
echo "  Results: ${OUTPUT_DIR}/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
