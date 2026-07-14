#!/usr/bin/env bash
set -euo pipefail

# General Capability Probe
#
# Runs the editing pipeline with perplexity + MMLU measurement at each
# downstream evaluation step. Produces a timeseries showing whether
# general model capabilities degrade as edits accumulate.
#
# This addresses the reviewer concern: "editing may destroy general
# capabilities while still passing CounterFact-specific metrics."
#
# Checkpoint reuse: If failure curve checkpoints exist for this seed/algorithm,
# the offline probe is used (loads model once, applies checkpoint weights,
# no re-editing). Set FORCE_ONLINE=true to skip checkpoint detection.
#
# Outputs:
#   results/capability_probe/probe_seed{SEED}_{ALG}_{LIMIT}edits.jsonl
#   Each line = one measurement point (edit_count, perplexity, mmlu_accuracy)
#
# Usage:
#   bash scripts/run_capability_probe.sh [SEED] [ALG] [DATASET_SIZE_LIMIT]
#   bash scripts/run_capability_probe.sh 42 AlphaEdit 2000
#   bash scripts/run_capability_probe.sh 42 MEMIT 2000
#   bash scripts/run_capability_probe.sh 42 both 2000    # default
#   FORCE_ONLINE=true bash scripts/run_capability_probe.sh 42  # Skip checkpoints

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

SEED="${1:-42}"
ALG="${2:-both}"
DATASET_SIZE_LIMIT="${3:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
# Set to "true" to include MMLU (slower but more informative)
INCLUDE_MMLU="${INCLUDE_MMLU:-true}"
FORCE_ONLINE="${FORCE_ONLINE:-false}"

echo "=== General Capability Probe ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Include MMLU: $INCLUDE_MMLU"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# --- Checkpoint detection helper ---
# Returns the checkpoint directory for a given algorithm if valid checkpoints exist.
find_checkpoints() {
    local alg_name="$1"

    # Priority 1: Explicit override
    if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
        local candidate="${CHECKPOINT_DIR}/${alg_name}/seed${SEED}"
        if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/model_weights.pt &>/dev/null; then
            echo "$candidate"
            return 0
        fi
    fi

    # Priority 2: S3 mount (SkyPilot clusters)
    local candidate="/s3-data/continual-learning/alphaedit/checkpoints/${alg_name}/seed${SEED}"
    if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/model_weights.pt &>/dev/null; then
        echo "$candidate"
        return 0
    fi

    # Priority 3: Local cache
    candidate="$HOME/.cache/alphaedit_checkpoints/${alg_name}/seed${SEED}"
    if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/model_weights.pt &>/dev/null; then
        echo "$candidate"
        return 0
    fi

    return 1
}

# --- Offline probe: reuse failure curve checkpoints ---
run_probe_offline() {
    local alg_name="$1"
    local ckpt_dir="$2"

    local num_ckpts
    num_ckpts=$(ls -d "$ckpt_dir"/batch_*/model_weights.pt 2>/dev/null | wc -l | tr -d ' ')
    echo "--- Capability probe (OFFLINE): $alg_name (seed=$SEED, $num_ckpts checkpoints) ---"
    echo "  Checkpoint dir: $ckpt_dir"

    local mmlu_flag=""
    if [[ "$INCLUDE_MMLU" == "false" ]]; then
        mmlu_flag="--no_mmlu"
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" uv run python src/capability_probe_offline.py \
        --seed "$SEED" \
        --alg_name "$alg_name" \
        --checkpoint_dir "$ckpt_dir" \
        --model_name "$MODEL_NAME" \
        $mmlu_flag

    echo "--- $alg_name capability probe (OFFLINE): DONE ---"
    echo ""
}

# --- Online probe: full editing with probing ---
run_probe_online() {
    local alg_name="$1"

    echo "--- Capability probe (ONLINE): $alg_name (seed=$SEED, limit=$DATASET_SIZE_LIMIT) ---"
    echo "  No checkpoints found, editing from scratch"

    local mmlu_flag=""
    if [[ "$INCLUDE_MMLU" == "false" ]]; then
        mmlu_flag="--no_mmlu"
    fi

    uv run python src/capability_probe_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name "$alg_name" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits 100 \
        --probe_interval 5 \
        $mmlu_flag

    echo "--- $alg_name capability probe (ONLINE): DONE ---"
    echo ""
}

# --- Dispatch: offline if checkpoints exist, online otherwise ---
run_probe() {
    local alg_name="$1"

    if [[ "$FORCE_ONLINE" != "true" ]]; then
        local ckpt_dir
        if ckpt_dir=$(find_checkpoints "$alg_name"); then
            run_probe_offline "$alg_name" "$ckpt_dir"
            return
        fi
    fi

    run_probe_online "$alg_name"
}

case "$ALG" in
    AlphaEdit)
        run_probe "AlphaEdit"
        ;;
    MEMIT)
        run_probe "MEMIT"
        ;;
    both)
        run_probe "AlphaEdit"
        run_probe "MEMIT"
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
        exit 1
        ;;
esac

echo "=== Capability probe complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: results/capability_probe/"
echo ""
echo "Next: python analysis/plots.py  (generates capability degradation figure)"
