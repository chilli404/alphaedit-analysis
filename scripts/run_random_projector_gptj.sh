#!/usr/bin/env bash
set -euo pipefail

# Random Projector Comparison: GPT-J-6B
#
# Runs AlphaEdit with a random orthogonal projector of specified rank,
# instead of the eigenspace projector from covariance statistics.
#
# This separates the effect of RANK (how many dimensions are feasible)
# from ORIENTATION (which directions are feasible). If AlphaEdit's
# performance depends on the eigenspace orientation, the random projector
# should perform worse on preservation while matching on efficacy at
# high rank.
#
# Usage:
#   bash scripts/run_random_projector_gptj.sh [SEED] [RANK] [PROJECTOR_SEED]
#   bash scripts/run_random_projector_gptj.sh 42 3390 0
#   bash scripts/run_random_projector_gptj.sh 42 9200 0
#
# Environment variables:
#   TARGET_EDITS     - Total edits (default: 5000)
#   PROJECTOR_SEED   - Seed for random projector generation (default: 0)
#   EVAL_AT_CHECKPOINTS_ONLY - Recommended for long runs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

export MODEL_NAME="EleutherAI/gpt-j-6b"
export HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"

SEED="${1:-42}"
RANK="${2:-3390}"
PROJECTOR_SEED="${3:-${PROJECTOR_SEED:-0}}"
TARGET_EDITS="${TARGET_EDITS:-10000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100

echo "=== Random Projector Comparison (GPT-J-6B) ==="
echo "  Seed: $SEED"
echo "  Rank: $RANK (of 16384)"
echo "  r/d: $(python3 -c "print(f'{$RANK/16384*100:.1f}%')")"
echo "  Projector seed: $PROJECTOR_SEED"
echo "  Target edits: $TARGET_EDITS"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Step 1: Generate the random projector
STATS_DIR="data/stats/gpt-j-6b/wikipedia_stats"
PROJECTOR_FILE="$STATS_DIR/null_space_project_random_r${RANK}_s${PROJECTOR_SEED}.pt"

if [[ ! -f "$PROJECTOR_FILE" ]]; then
    echo "--- Generating random projector (rank=$RANK, seed=$PROJECTOR_SEED) ---"
    uv run python src/experiments/generate_random_projector.py \
        --rank "$RANK" \
        --seed "$PROJECTOR_SEED" \
        --output "$PROJECTOR_FILE"
    echo ""
fi

# Step 2: Symlink the random projector as the active null_space_project.pt
# We use a temporary symlink override: the checkpoint_runner's nullspace_threshold
# injection replaces the P file path. Instead, we use --projector_override.
#
# Since checkpoint_runner doesn't have --projector_override yet, we use the
# threshold injection mechanism with a specially-named file.
# The threshold injection replaces "null_space_project.pt" with
# "null_space_project_t{threshold}.pt" in the source. We use a sentinel
# threshold value and name the random projector accordingly.

SENTINEL_THRESHOLD="0.99${RANK}"
SENTINEL_FILE="$STATS_DIR/null_space_project_t${SENTINEL_THRESHOLD}.pt"
VENDOR_SENTINEL="$PROJECT_DIR/vendor/AlphaEdit/null_space_project_t${SENTINEL_THRESHOLD}.pt"

# Create symlink in data/stats/ (canonical location) and vendor/AlphaEdit/ (where evaluate.py loads from)
ln -sf "$(realpath "$PROJECTOR_FILE")" "$SENTINEL_FILE"
ln -sf "$(realpath "$PROJECTOR_FILE")" "$VENDOR_SENTINEL"
echo "  Linked: $SENTINEL_FILE -> $PROJECTOR_FILE"
echo "  Linked: $VENDOR_SENTINEL -> $PROJECTOR_FILE"

# Step 3: Run AlphaEdit with the sentinel threshold (loads our random P)
EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-true}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    EVAL_FLAG="--fast_checkpoint"
fi

echo ""
echo "--- Running AlphaEdit with random projector (rank=$RANK) ---"

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
    --nullspace_threshold "$SENTINEL_THRESHOLD" \
    $EVAL_FLAG

# Cleanup sentinel symlinks
rm -f "$SENTINEL_FILE"
rm -f "$VENDOR_SENTINEL"

echo ""
echo "=== Random Projector comparison complete ==="
echo "  Rank: $RANK, r/d = $(python3 -c "print(f'{$RANK/16384*100:.1f}%')")"
echo "  Checkpoint dir tag: AlphaEdit-t${SENTINEL_THRESHOLD}"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Compare against eigenspace projector:"
echo "  bash scripts/run_projection_sweep_gptj.sh $SEED <matching_threshold> AlphaEdit"
