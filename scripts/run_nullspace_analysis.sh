#!/usr/bin/env bash
set -euo pipefail

# Null-Space Rank Consumption Analysis
#
# Runs the instrumented AlphaEdit experiment that tracks how the null-space
# is consumed as sequential edits accumulate. This produces the mechanistic
# analysis figure showing WHY AlphaEdit eventually degrades.
#
# The tracker records per-edit-batch:
#   - Initial null-space rank per layer (from SVD threshold)
#   - Accumulated covariance cache_c rank (grows with edits)
#   - Consumption ratio (cache rank / null-space rank)
#   - Spectral properties (top singular values)
#
# Checkpoint reuse: If failure curve checkpoints exist for this seed,
# the offline analyzer is used (reads cache_c directly, no re-editing).
# Set FORCE_ONLINE=true to skip checkpoint detection and always re-edit.
#
# Usage:
#   bash scripts/run_nullspace_analysis.sh [SEED] [DATASET_SIZE_LIMIT]
#   bash scripts/run_nullspace_analysis.sh 42 2000   # Full run
#   bash scripts/run_nullspace_analysis.sh 42 500    # Quick run
#   FORCE_ONLINE=true bash scripts/run_nullspace_analysis.sh 42  # Skip checkpoints

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
DATASET_SIZE_LIMIT="${2:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
FORCE_ONLINE="${FORCE_ONLINE:-false}"

echo "=== Null-Space Rank Consumption Analysis ==="
echo "  Seed: $SEED"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Num edits per batch: $NUM_EDITS"
echo "  Model: $MODEL_NAME"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# --- Checkpoint detection ---
# Look for failure curve checkpoints (AlphaEdit only — MEMIT has no cache_c)
CKPT_DIR=""
if [[ "$FORCE_ONLINE" != "true" ]]; then
    # Priority 1: Explicit override
    if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
        candidate="${CHECKPOINT_DIR}/failure_curve/AlphaEdit/seed${SEED}"
        if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/cache_c.pt &>/dev/null; then
            CKPT_DIR="$candidate"
        fi
    fi

    # Priority 2: S3 mount (SkyPilot clusters)
    if [[ -z "$CKPT_DIR" ]]; then
        candidate="/s3-data/continual-learning/alphaedit/checkpoints/failure_curve/AlphaEdit/seed${SEED}"
        if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/cache_c.pt &>/dev/null; then
            CKPT_DIR="$candidate"
        fi
    fi

    # Priority 3: Local cache
    if [[ -z "$CKPT_DIR" ]]; then
        candidate="$HOME/.cache/alphaedit_checkpoints/failure_curve/AlphaEdit/seed${SEED}"
        if [[ -d "$candidate" ]] && ls "$candidate"/batch_*/cache_c.pt &>/dev/null; then
            CKPT_DIR="$candidate"
        fi
    fi
fi

if [[ -n "$CKPT_DIR" ]]; then
    # --- Offline path: reuse failure curve checkpoints ---
    num_ckpts=$(ls -d "$CKPT_DIR"/batch_*/cache_c.pt 2>/dev/null | wc -l | tr -d ' ')
    echo "  Mode: OFFLINE (reusing $num_ckpts failure curve checkpoints)"
    echo "  Checkpoint dir: $CKPT_DIR"
    echo ""

    uv run python src/mechanism/nullspace_offline_analyzer.py \
        --seed "$SEED" \
        --checkpoint_dir "$CKPT_DIR" \
        --device "cuda:${CUDA_DEVICE}"
else
    # --- Online path: run full editing with tracking ---
    echo "  Mode: ONLINE (no checkpoints found, editing from scratch)"
    echo ""

    uv run python src/mechanism/nullspace_tracker.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname "$HPARAMS_FNAME" \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --downstream_eval_steps 5 \
        --conserve_memory
fi

echo ""
echo "=== Null-space analysis complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: results/nullspace_tracking/"
echo ""
echo "Next: python analysis/plots.py  (generates mechanistic figure)"
