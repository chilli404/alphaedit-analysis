#!/usr/bin/env bash
set -euo pipefail

# Protected Editing Experiment
#
# Runs AlphaEdit with live cosine-based or random protection: after each batch,
# reapplies K most vulnerable previously-installed edits.
#
# Usage:
#   bash scripts/run_protected_editing.sh SEED ORDERING PROTECTION [K]
#   bash scripts/run_protected_editing.sh 42 key_clustered cosine 10
#   bash scripts/run_protected_editing.sh 42 key_clustered random 10

SEED="${1:?Usage: $0 SEED ORDERING PROTECTION [K]}"
ORDERING="${2:?Usage: $0 SEED ORDERING PROTECTION [K]}"
PROTECTION="${3:-cosine}"
PROTECT_K="${4:-${PROTECT_K:-10}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
KEYS_PATH="${KEYS_PATH:-$RESULT_ROOT/key_vectors/full_mcf/keys_seed42_layer6.npz}"
STREAM_LENGTH="${TARGET_EDITS:-10000}"

echo "========================================"
echo "Protected Editing Experiment"
echo "  Seed:       $SEED"
echo "  Ordering:   $ORDERING"
echo "  Protection: $PROTECTION (K=$PROTECT_K)"
echo "  Model:      $MODEL_NAME"
echo "  Stream:     $STREAM_LENGTH edits"
echo "========================================"

cd "$PROJECT_DIR"

uv run python src/runners/protected_editing_runner.py \
    --seed "$SEED" \
    --ordering "$ORDERING" \
    --protection "$PROTECTION" \
    --protect_k "$PROTECT_K" \
    --stream_length "$STREAM_LENGTH" \
    --keys_path "$KEYS_PATH" \
    --model_name "$MODEL_NAME"
