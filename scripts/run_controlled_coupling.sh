#!/usr/bin/env bash
set -euo pipefail

# Controlled Coupling Experiment
# Tests whether semantic structure of the edit stream determines
# effective editing capacity by comparing low-coupling vs high-coupling streams.
#
# Usage:
#   bash scripts/run_controlled_coupling.sh [SEED] [STREAM_LENGTH]
#   CUDA_DEVICE=1 bash scripts/run_controlled_coupling.sh 42 5000
#   STREAM=low bash scripts/run_controlled_coupling.sh 42 5000

SEED="${1:-42}"
STREAM_LENGTH="${2:-${STREAM_LENGTH:-5000}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STREAM="${STREAM:-both}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

echo "=== Controlled Coupling Experiment ==="
echo "  Seed: $SEED"
echo "  Stream length: $STREAM_LENGTH"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Stream: $STREAM"
echo "  Save interval: $SAVE_INTERVAL"
echo ""

EXTRA_ARGS=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-false}" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --eval_at_checkpoints_only"
    echo "  Eval mode: milestone (checkpoints only)"
fi

uv run python src/runners/controlled_coupling_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --stream_length "$STREAM_LENGTH" \
    --num_edits 100 \
    --save_interval "$SAVE_INTERVAL" \
    --stream "$STREAM" \
    --conserve_memory \
    $EXTRA_ARGS

echo ""
echo "=== Done (seed=$SEED) ==="
