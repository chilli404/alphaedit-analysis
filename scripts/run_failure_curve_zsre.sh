#!/usr/bin/env bash
set -euo pipefail

# zsRE Failure Curve — wrapper around checkpoint_runner with zsRE dataset
#
# Usage:
#   bash scripts/run_failure_curve_zsre.sh [SEED] [ALG] [TARGET_EDITS]
#   bash scripts/run_failure_curve_zsre.sh 42 AlphaEdit 10000
#   EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_zsre.sh 42 AlphaEdit 10000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

_CALLER_MODEL_NAME="${MODEL_NAME:-}"
_CALLER_HPARAMS="${HPARAMS_FNAME:-}"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${_CALLER_MODEL_NAME:-${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}}"
HPARAMS_FNAME="${_CALLER_HPARAMS:-${HPARAMS_FNAME:-Llama3-8B.json}}"

SEED="${1:-42}"
ALG="${2:-${ALG_NAME:-AlphaEdit}}"
TARGET_EDITS="${3:-${TARGET_EDITS:-10000}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"

EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-false}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    EVAL_FLAG="--fast_checkpoint"
fi

CKPT_DIR="${CHECKPOINT_ROOT}/failure_curve_zsre/${ALG}/seed${SEED}"
CKPT_ARGS="--checkpoint_dir $CKPT_DIR"

echo "========================================"
echo "zsRE Failure Curve"
echo "  Seed:     $SEED"
echo "  Alg:      $ALG"
echo "  Target:   $TARGET_EDITS edits"
echo "  Model:    $MODEL_NAME"
echo "  Hparams:  $HPARAMS_FNAME"
echo "  Ckpt:     $CKPT_DIR"
echo "  Eval:     ${EVAL_FLAG:-normal}"
echo "========================================"

cd "$PROJECT_DIR"

uv run python src/runners/checkpoint_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name "$ALG" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name zsre \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits "$NUM_EDITS" \
    --save_interval "$SAVE_INTERVAL" \
    --downstream_eval_steps 10 \
    --conserve_memory \
    $EVAL_FLAG \
    $CKPT_ARGS

echo ""
echo "zsRE failure curve complete: $ALG, seed $SEED, $TARGET_EDITS edits"
