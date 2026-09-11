#!/usr/bin/env bash
set -euo pipefail

# Multilayer Key Extraction for Survival Predictor
#
# Extracts key vectors from all edited layers (Llama-3: 4,5,6,7,8; GPT-J: 3,4,5,6,7,8)
# to test whether the layer-6 predictor generalizes across layers.
#
# Only forward passes — no editing, no gradient. Fast: ~30 min/layer on GPU.
# Only one seed needed (keys are base-model activations, deterministic).
#
# Usage:
#   bash scripts/run_multilayer_key_extraction.sh [SEED]
#   MODEL_NAME=EleutherAI/gpt-j-6b bash scripts/run_multilayer_key_extraction.sh 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Save MODEL_NAME if pre-set (before .env can override it)
_PRESET_MODEL="${MODEL_NAME:-}"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Restore pre-set MODEL_NAME (takes priority over .env)
if [[ -n "$_PRESET_MODEL" ]]; then
    MODEL_NAME="$_PRESET_MODEL"
fi

SEED="${1:-42}"
MODEL="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
OUTPUT_BASE="$RESULT_ROOT/key_vectors"

# Determine layers based on model
case "$MODEL" in
    *gpt-j*|*GPT-J*)
        LAYERS=(3 4 5 6 7 8)
        MODEL_TAG="gptj"
        ;;
    *Qwen*|*qwen*)
        LAYERS=(4 5 6 7 8)
        MODEL_TAG="qwen"
        ;;
    *)
        LAYERS=(4 5 6 7 8)
        MODEL_TAG="llama3"
        ;;
esac

echo "=== Multilayer Key Extraction ==="
echo "  Model: $MODEL ($MODEL_TAG)"
echo "  Seed: $SEED"
echo "  Layers: ${LAYERS[*]}"
echo "  Device: cuda:$DEVICE"
echo "  Output: $OUTPUT_BASE/layer{N}/"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

TOTAL_START=$(date +%s)

for LAYER in "${LAYERS[@]}"; do
    LAYER_START=$(date +%s)
    if [[ "$MODEL_TAG" == "llama3" ]]; then
        OUTPUT_DIR="$OUTPUT_BASE/layer${LAYER}"
    else
        OUTPUT_DIR="$OUTPUT_BASE/${MODEL_TAG}/layer${LAYER}"
    fi
    mkdir -p "$OUTPUT_DIR"

    echo "--- Layer $LAYER ---"
    echo "  Output: $OUTPUT_DIR/seed${SEED}/keys_seed${SEED}.npz"

    # Skip if already exists
    if [[ -f "$OUTPUT_DIR/seed${SEED}/keys_seed${SEED}.npz" ]]; then
        echo "  SKIP: already exists"
        continue
    fi

    uv run python -m src.mechanism.compute_keys \
        --seed "$SEED" \
        --layer "$LAYER" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL" \
        --device "cuda:$DEVICE"

    LAYER_END=$(date +%s)
    echo "  Layer $LAYER done in $((LAYER_END - LAYER_START))s"
    echo ""
done

TOTAL_END=$(date +%s)
echo "=== All layers complete ==="
echo "  Total time: $((TOTAL_END - TOTAL_START))s"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
