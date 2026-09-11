#!/usr/bin/env bash
set -euo pipefail

# Joint Memory-Space × Update-Space Experiment (GPT-J-6B)
#
# 2×2 interaction study combining:
#   - Stream ordering (memory-space geometry): key_clustered vs key_dispersed
#   - Projector capacity (update-space): low r/d (~20.7%) vs high r/d (~56.1%)
#
# All cells run AlphaEdit + C₀ (INJECT_C0=true) to observe how memory-space
# geometry interacts with available update capacity.
#
# Scientific question: Does the history-geometry retention gap (clustered vs
# dispersed) change systematically with available update capacity?
#
# Expected results:
#   - Binding regime (low r/d): geometry effect is AMPLIFIED — limited capacity
#     means memory-interfering orderings cause disproportionate damage
#   - Permissive regime (high r/d): geometry effect is ATTENUATED — excess
#     capacity provides a buffer against same-cluster interference
#
# Threshold-to-r/d mapping (verified via SVD of normalized covariance, layer 6):
#   - threshold=0.0052 → r/d ≈ 20.7% (binding)
#   - threshold=0.0105 → r/d ≈ 56.1% (permissive)
#
# Usage:
#   bash scripts/run_joint_twospace_gptj.sh SEED ORDERING THRESHOLD
#   bash scripts/run_joint_twospace_gptj.sh 2024 key_clustered 0.05
#   bash scripts/run_joint_twospace_gptj.sh 2024 key_dispersed 0.005
#
# Environment variables:
#   TARGET_EDITS     - Stream length (default: 5000)
#   SAVE_INTERVAL    - Checkpoint interval in batches (default: 10)
#   EVAL_AT_CHECKPOINTS_ONLY - "true" for milestone eval (RECOMMENDED)
#   C0_WEIGHT        - Weight α for C₀ injection (default: 15000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Preserve caller overrides
_CALLER_MODEL_NAME="${MODEL_NAME:-}"
_CALLER_HPARAMS="${HPARAMS_FNAME:-}"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Force GPT-J model
export MODEL_NAME="${_CALLER_MODEL_NAME:-EleutherAI/gpt-j-6b}"
export HPARAMS_FNAME="${_CALLER_HPARAMS:-EleutherAI_gpt-j-6B.json}"

SEED="${1:-${SEED:-42}}"
ORDERING="${2:-${ORDERING:?ERROR: ORDERING not set (key_clustered or key_dispersed)}}"
THRESHOLD="${3:-${NULLSPACE_THRESHOLD:?ERROR: NULLSPACE_THRESHOLD not set (e.g., 0.05 or 0.005)}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TARGET_EDITS="${TARGET_EDITS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100
C0_WEIGHT="${C0_WEIGHT:-15000}"

# Resolve ordering stream file (GPT-J specific orderings)
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
STREAM_DIR="$RESULT_ROOT/matched_ordering_gptj/orderings"
STREAM_FILE="${ORDERING}_seed${SEED}.json"

if [[ ! -f "$STREAM_DIR/$STREAM_FILE" ]]; then
    # Auto-generate orderings if keys exist but streams don't
    KEYS_PATH="$RESULT_ROOT/key_vectors/gptj_full_mcf/keys_seed${SEED}_layer5.npz"
    if [[ -f "$KEYS_PATH" ]]; then
        echo "  Stream not found — generating GPT-J orderings from existing keys..."
        uv run python src/datasets/generate_orderings.py \
            --seed "$SEED" \
            --keys_path "$KEYS_PATH" \
            --output_dir "$RESULT_ROOT/matched_ordering_gptj"
    fi

    if [[ ! -f "$STREAM_DIR/$STREAM_FILE" ]]; then
        echo "ERROR: GPT-J ordering stream not found: $STREAM_DIR/$STREAM_FILE"
        echo ""
        echo "  This experiment requires GPT-J-specific key vectors and orderings."
        echo ""
        echo "  Generate GPT-J orderings first:"
        echo "    1. bash scripts/run_gptj_key_extraction.sh $SEED"
        echo "    2. uv run python src/datasets/generate_orderings.py --seed $SEED \\"
        echo "         --keys_path results/key_vectors/gptj_full_mcf/keys_seed${SEED}_layer5.npz \\"
        echo "         --output_dir results/matched_ordering_gptj"
        exit 1
    fi
fi
STREAM_PATH="$STREAM_DIR/$STREAM_FILE"

# Resolve checkpoint and results directories
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
CKPT_DIR="$CHECKPOINT_ROOT/joint_twospace/gpt-j-6b/t${THRESHOLD}/${ORDERING}/seed${SEED}"
mkdir -p "$CKPT_DIR"

RESULTS_DIR="$RESULT_ROOT/joint_twospace/gpt-j-6b/t${THRESHOLD}/${ORDERING}/seed${SEED}"
mkdir -p "$RESULTS_DIR"

echo "=== Joint Memory × Update Space (GPT-J-6B) ==="
echo "  Seed:        $SEED"
echo "  Ordering:    $ORDERING (memory-space geometry)"
echo "  Threshold:   $THRESHOLD (update-space capacity)"
echo "  C₀ weight:   $C0_WEIGHT"
echo "  Stream:      $STREAM_PATH"
echo "  Checkpoint:  $CKPT_DIR"
echo "  Results:     $RESULTS_DIR"
echo "  Edits:       $TARGET_EDITS"
echo "  Started:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-true}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    EVAL_FLAG="--fast_checkpoint"
fi

uv run python src/runners/checkpoint_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name AlphaEdit \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits "$NUM_EDITS" \
    --save_interval "$SAVE_INTERVAL" \
    --downstream_eval_steps 0 \
    --conserve_memory \
    --nullspace_threshold "$THRESHOLD" \
    --inject_c0 \
    --c0_weight "$C0_WEIGHT" \
    --checkpoint_dir "$CKPT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --dataset_override "$STREAM_PATH" \
    $EVAL_FLAG

echo ""
echo "=== Joint twospace cell complete ==="
echo "  Ordering:   $ORDERING"
echo "  Threshold:  $THRESHOLD"
echo "  Results:    $RESULTS_DIR"
echo "  Finished:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
