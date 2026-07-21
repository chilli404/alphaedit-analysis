#!/usr/bin/env bash
set -euo pipefail

# Full-history MEMIT-seq on controlled coupling streams.
#
# Runs MEMIT+SeqReg (λ_prev=1, λ_delta=0, unlimited cache) on the
# low-coupling and/or high-coupling 5K edit streams, for direct
# comparison with AlphaEdit controlled coupling results.
#
# Usage:
#   bash scripts/run_memit_seq_coupling.sh [SEED] [STREAM]
#   bash scripts/run_memit_seq_coupling.sh 42 both   # Both streams (default)
#   bash scripts/run_memit_seq_coupling.sh 42 low    # Low coupling only
#   bash scripts/run_memit_seq_coupling.sh 42 high   # High coupling only
#
# Environment variables:
#   CUDA_DEVICE      - GPU device index (default: 0)
#   MODEL_NAME       - Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   TARGET_EDITS     - Stream length (default: 5000)
#   FAST_CHECKPOINT  - If "true", only evaluate edited batch (default: true)
#   SAVE_INTERVAL    - Checkpoint save interval in batches (default: 10)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
SEED="${1:-42}"
STREAM="${2:-both}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET_SIZE_LIMIT="${TARGET_EDITS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Fixed settings: full-history MEMIT-seq (best config from factorial)
LAMBDA_PREV="1.0"
LAMBDA_DELTA="0.0"
CACHE_STRATEGY="all"
CACHE_MAX="none"

# Stream data paths
RESULTS_DIR="$PROJECT_DIR/results/controlled_coupling"
LOW_STREAM="$RESULTS_DIR/low_coupling_seed${SEED}.json"
HIGH_STREAM="$RESULTS_DIR/high_coupling_seed${SEED}.json"

# Checkpoint base
CKPT_BASE="${HOME}/.cache/memit_seqreg_checkpoints"

echo "=== MEMIT-seq Coupling Experiment ==="
echo "  Seed:       $SEED"
echo "  Stream(s):  $STREAM"
echo "  Edits:      $DATASET_SIZE_LIMIT"
echo "  Config:     λ_prev=$LAMBDA_PREV, λ_delta=$LAMBDA_DELTA, cache=$CACHE_STRATEGY"
echo "  Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Build fast checkpoint flag (default: true for coupling runs)
FAST_FLAG=""
if [[ "${FAST_CHECKPOINT:-true}" == "true" ]]; then
    FAST_FLAG="--fast_checkpoint"
    echo "  FAST MODE: only evaluate edited batch"
fi

run_stream() {
    local stream_name="$1"
    local stream_path="$2"
    local ckpt_dir="$CKPT_BASE/coupling_${stream_name}_seed${SEED}"

    if [[ ! -f "$stream_path" ]]; then
        echo "ERROR: Stream file not found: $stream_path"
        echo "  Run controlled_coupling_runner.py first to generate streams."
        return 1
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running $stream_name stream..."
    echo "  Dataset:    $stream_path"
    echo "  Checkpoint: $ckpt_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv run python src/runners/memit_sequential_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits 100 \
        --downstream_eval_steps 10 \
        --conserve_memory \
        --lambda_prev "$LAMBDA_PREV" \
        --lambda_delta "$LAMBDA_DELTA" \
        --cache_strategy "$CACHE_STRATEGY" \
        --cache_max "$CACHE_MAX" \
        --save_interval "$SAVE_INTERVAL" \
        --checkpoint_dir "$ckpt_dir" \
        --dataset_override "$stream_path" \
        $FAST_FLAG
}

# Run requested stream(s)
if [[ "$STREAM" == "both" || "$STREAM" == "low" ]]; then
    run_stream "low" "$LOW_STREAM"
fi

if [[ "$STREAM" == "both" || "$STREAM" == "high" ]]; then
    run_stream "high" "$HIGH_STREAM"
fi

echo ""
echo "=== MEMIT-seq Coupling complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
