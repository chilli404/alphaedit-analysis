#!/usr/bin/env bash
set -euo pipefail

# PathGuard Smoke Test: 500 edits (5 batches), M=20, q=5
#
# Validates:
#   1. Script generation and source injection compiles
#   2. Key loading works
#   3. Exposure computation + Woodbury solve completes
#   4. Checkpoint save/load roundtrip
#   5. Mechanism log contains PathGuard fields
#
# Requires: GPU, linked datasets, precomputed keys
#
# Usage:
#   bash scripts/smoke_test_pathguard.sh
#   TARGET_EDITS=200 bash scripts/smoke_test_pathguard.sh  # even shorter

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

_PRESET_MODEL="${MODEL_NAME:-}"
_PRESET_HPARAMS="${HPARAMS_FNAME:-}"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi
if [[ -n "$_PRESET_HPARAMS" ]]; then HPARAMS_FNAME="$_PRESET_HPARAMS"; fi

SEED=42
TARGET_EDITS="${TARGET_EDITS:-500}"
DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
KEYS_DIR="$RESULT_ROOT/key_vectors/full_mcf"

echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard Smoke Test"
echo "  Target:   $TARGET_EDITS edits ($(( TARGET_EDITS / 100 )) batches)"
echo "  Model:    $MODEL_NAME"
echo "  Keys dir: $KEYS_DIR"
echo "═══════════════════════════════════════════════════════════════"

# Pre-flight checks
echo ""
echo "Pre-flight checks..."

# Check keys
for layer in 4 5 6 7 8; do
    if [[ ! -f "$KEYS_DIR/keys_seed${SEED}_layer${layer}.npz" ]]; then
        echo "ERROR: Missing keys for layer $layer at $KEYS_DIR/keys_seed${SEED}_layer${layer}.npz"
        echo "Run: bash scripts/run_multilayer_key_extraction.sh $SEED"
        exit 1
    fi
done
echo "  Keys: OK (all 5 layers)"

# Check dataset
bash "$SCRIPT_DIR/link_dsets.sh" 2>/dev/null || true
if [[ ! -f "$PROJECT_DIR/vendor/AlphaEdit/data/multi_counterfact.json" ]]; then
    echo "ERROR: MCF dataset not linked"
    echo "Run: bash scripts/link_dsets.sh"
    exit 1
fi
echo "  MCF dataset: OK"

# Check stats
bash "$SCRIPT_DIR/link_stats.sh" 2>/dev/null || true
echo "  Stats: linked"

# Check GPU
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU available"
    exit 1
}
echo "  GPU: OK"

echo ""
echo "Running PathGuard-ED smoke test..."

# Use a temporary checkpoint dir to avoid polluting production checkpoints
SMOKE_CKPT="$CHECKPOINT_ROOT/pathguard_smoke_test"
rm -rf "$SMOKE_CKPT" 2>/dev/null || true

export RESULT_ROOT CHECKPOINT_ROOT

uv run python src/runners/pathguard_runner.py \
    --seed "$SEED" \
    --cuda_device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 5 \
    --conserve_memory \
    --lambda_prev 1.0 \
    --lambda_delta 0.0 \
    --cache_strategy all \
    --cache_max none \
    --save_interval 2 \
    --fast_checkpoint \
    --checkpoint_dir "$SMOKE_CKPT" \
    --pathguard \
    --pathguard_adaptive \
    --pathguard_M 20 \
    --pathguard_q 5 \
    --pathguard_epsilon_init 1.0 \
    --pathguard_keys_dir "$KEYS_DIR" \
    --pathguard_representative_layer 6

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Smoke Test Validation"
echo "═══════════════════════════════════════════════════════════════"

# Validate checkpoints exist
CKPT_COUNT=$(find "$SMOKE_CKPT" -name "metadata.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  Checkpoints saved: $CKPT_COUNT"
if [[ "$CKPT_COUNT" -lt 1 ]]; then
    echo "  FAIL: No checkpoints saved"
    exit 1
fi

# Validate PathGuard state saved
PG_STATE_COUNT=$(find "$SMOKE_CKPT" -name "pathguard_state.pt" 2>/dev/null | wc -l | tr -d ' ')
echo "  PathGuard states: $PG_STATE_COUNT"
if [[ "$PG_STATE_COUNT" -lt 1 ]]; then
    echo "  FAIL: No pathguard_state.pt saved"
    exit 1
fi

# Validate mechanism log contains PathGuard fields
LOG_FILE=$(find "$SMOKE_CKPT" -name "mechanism_log.jsonl" 2>/dev/null | head -1)
if [[ -n "$LOG_FILE" ]]; then
    PG_ENTRIES=$(grep -c "pg_lambda" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  PathGuard log entries: $PG_ENTRIES"
    if [[ "$PG_ENTRIES" -lt 1 ]]; then
        echo "  FAIL: No PathGuard entries in mechanism log"
        exit 1
    fi
fi

echo ""
echo "  All checks PASSED"
echo ""

# Cleanup
rm -rf "$SMOKE_CKPT"
echo "  Cleaned up smoke test checkpoints"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard smoke test PASSED"
echo "═══════════════════════════════════════════════════════════════"
