#!/usr/bin/env bash
set -euo pipefail

# Offline Capability Probe
#
# Runs perplexity + MMLU probes on existing failure-curve checkpoints.
# Does NOT re-edit the model — loads base model once, iterates checkpoints.
#
# Requires: failure_curve checkpoints exist for the given seed/algorithm
# at CHECKPOINT_ROOT/failure_curve/{ALG}/seed{SEED}/batch_*/model_weights.pt
#
# Outputs:
#   $RESULT_ROOT/capability_probe/seed{SEED}/{EDITS}edits/{ALG}/offline_probe_*.jsonl
#
# Usage:
#   bash scripts/run_capability_probe_offline.sh [SEED] [ALG]
#   bash scripts/run_capability_probe_offline.sh 42
#   bash scripts/run_capability_probe_offline.sh 42 AlphaEdit
#   bash scripts/run_capability_probe_offline.sh 42 both

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"

SEED="${1:-${SEED:-42}}"
ALG="${2:-${ALG_NAME:-AlphaEdit}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
INCLUDE_MMLU="${INCLUDE_MMLU:-true}"

echo "=== Offline Capability Probe ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Checkpoint root: $CHECKPOINT_ROOT"
echo "  Result root: $RESULT_ROOT"
echo "  Include MMLU: $INCLUDE_MMLU"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

run_offline() {
    local alg_name="$1"
    local ckpt_dir="$CHECKPOINT_ROOT/failure_curve/${alg_name}/seed${SEED}"

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "ERROR: No checkpoints found at: $ckpt_dir"
        echo "  Run failure curve first:"
        echo "    EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_checkpointed.sh $SEED $alg_name 10000"
        exit 1
    fi

    local num_ckpts
    num_ckpts=$(ls -d "$ckpt_dir"/batch_*/model_weights.pt 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$num_ckpts" -eq 0 ]]; then
        echo "ERROR: Checkpoint dir exists but contains no batch_*/model_weights.pt: $ckpt_dir"
        exit 1
    fi

    echo "--- Offline probe: $alg_name (seed=$SEED, $num_ckpts checkpoints) ---"

    local mmlu_flag=""
    if [[ "$INCLUDE_MMLU" == "false" ]]; then
        mmlu_flag="--no_mmlu"
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run python src/mechanism/capability_probe_offline.py \
        --seed "$SEED" \
        --alg_name "$alg_name" \
        --checkpoint_dir "$ckpt_dir" \
        --model_name "$MODEL_NAME" \
        $mmlu_flag

    echo "--- $alg_name: DONE ---"
    echo ""
}

case "$ALG" in
    AlphaEdit)
        run_offline "AlphaEdit"
        ;;
    MEMIT)
        run_offline "MEMIT"
        ;;
    both)
        run_offline "AlphaEdit"
        run_offline "MEMIT"
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
        exit 1
        ;;
esac

echo "=== Offline capability probe complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: $RESULT_ROOT/capability_probe/"
