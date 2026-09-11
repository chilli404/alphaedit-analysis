#!/usr/bin/env bash
set -euo pipefail

# GPT-J Intermediate Checkpoint Evaluation
#
# Evaluates all 10K MCF facts at each failure-curve checkpoint to build
# a multi-timepoint panel for GPT-J survival model (Criticism #14).
#
# Checkpoints: batch_9 (1K edits), batch_29 (3K), batch_49 (5K), batch_99 (10K)
# Each checkpoint is loaded and all facts are evaluated for efficacy/paraphrase/neighborhood.
#
# Usage:
#   bash scripts/run_gptj_intermediate_eval.sh [SEED]
#   CHECKPOINT_ROOT=/path/to/checkpoints bash scripts/run_gptj_intermediate_eval.sh 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
MODEL_NAME="${MODEL_NAME:-EleutherAI/gpt-j-6b}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"

# GPT-J checkpoints are under a specific algorithm variant
ALG_DIR="AlphaEdit-C0-15000.0-t0.005"
CKPT_BASE="$CHECKPOINT_ROOT/failure_curve/gpt-j-6b/$ALG_DIR/seed${SEED}"

# Map batch numbers to edit counts
declare -A BATCH_TO_EDITS=(
    [9]=1000
    [29]=3000
    [49]=5000
    [99]=10000
)

echo "=== GPT-J Intermediate Checkpoint Evaluation ==="
echo "  Model: $MODEL_NAME"
echo "  Seed: $SEED"
echo "  Checkpoint base: $CKPT_BASE"
echo "  Output: $RESULT_ROOT/failure_curve_gptj/seed${SEED}/"
echo "  Batches: ${!BATCH_TO_EDITS[@]}"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Check which checkpoints exist
AVAILABLE_BATCHES=""
for batch in 9 29 49 99; do
    ckpt="$CKPT_BASE/batch_${batch}"
    if [[ -d "$ckpt" ]] && [[ -f "$ckpt/model_weights.pt" ]]; then
        echo "  ✓ batch_${batch} (${BATCH_TO_EDITS[$batch]} edits)"
        AVAILABLE_BATCHES="$AVAILABLE_BATCHES $batch"
    else
        echo "  ✗ batch_${batch} — NOT FOUND at $ckpt"
    fi
done

if [[ -z "$AVAILABLE_BATCHES" ]]; then
    echo "ERROR: No checkpoints found at $CKPT_BASE"
    echo "  Expected: batch_9/, batch_29/, batch_49/, batch_99/ with model_weights.pt"
    exit 1
fi

# Run evaluation for each available checkpoint
for batch in $AVAILABLE_BATCHES; do
    edits=${BATCH_TO_EDITS[$batch]}
    OUTPUT_DIR="$RESULT_ROOT/failure_curve_gptj/seed${SEED}/${edits}edits/AlphaEdit/run_000"
    CKPT_PATH="$CKPT_BASE/batch_${batch}"

    # Skip if already evaluated
    mkdir -p "$OUTPUT_DIR"
    n_existing=$(find "$OUTPUT_DIR" -name "*.json" -not -name "edit_ordering.json" 2>/dev/null | wc -l || echo "0")
    if [[ $n_existing -gt 5000 ]]; then
        echo "  Skipping batch_${batch}: $n_existing files already at $OUTPUT_DIR"
        continue
    fi

    echo ""
    echo "--- Evaluating batch_${batch} (${edits} edits) ---"
    echo "  Checkpoint: $CKPT_PATH"
    echo "  Output: $OUTPUT_DIR"

    uv run python scripts/eval_checkpoint_allcases.py \
        --checkpoint_path "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --num_edits "$edits"

    echo "  Batch $batch done."
done

echo ""
echo "=== All evaluations complete ==="
echo "  Results at: $RESULT_ROOT/failure_curve_gptj/seed${SEED}/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
