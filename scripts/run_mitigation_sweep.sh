#!/usr/bin/env bash
set -euo pipefail

# Cache Mitigation Hyperparameter Sweep
#
# Runs all 12 mitigation variants on a single seed. After analyzing results,
# run the top 2-3 variants across all 5 seeds using run_all_seeds.sh.
#
# Strategies:
#   svd_truncation:   K ∈ {5, 10}, retain_ratio ∈ {0.5, 0.75, 0.9} → 6 variants
#   exponential_decay: decay ∈ {0.90, 0.95, 0.99} → 3 variants
#   periodic_reset:   K ∈ {5, 10, 20} → 3 variants
#
# Usage:
#   bash scripts/run_mitigation_sweep.sh [SEED] [DATASET_SIZE_LIMIT]
#   bash scripts/run_mitigation_sweep.sh 42 2000   # Full sweep
#   bash scripts/run_mitigation_sweep.sh 42 500    # Quick sweep

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
DATASET_SIZE_LIMIT="${2:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"

echo "=== Cache Mitigation Hyperparameter Sweep ==="
echo "  Seed: $SEED"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Num edits per batch: $NUM_EDITS"
echo "  Model: $MODEL_NAME"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

COMPLETED=0
FAILED=0

run_variant() {
    local strategy="$1"
    shift
    echo ""
    echo "--- Running: $strategy $* ---"
    if uv run python src/cache_mitigation_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname "$HPARAMS_FNAME" \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --downstream_eval_steps 5 \
        --conserve_memory \
        --strategy "$strategy" \
        "$@"; then
        COMPLETED=$((COMPLETED + 1))
        echo "--- Completed: $strategy $* ---"
    else
        FAILED=$((FAILED + 1))
        echo "--- FAILED: $strategy $* ---"
    fi
}

# SVD Truncation: 6 variants
for interval in 5 10; do
    for ratio in 0.5 0.75 0.9; do
        run_variant svd_truncation \
            --truncation_interval "$interval" \
            --retain_ratio "$ratio"
    done
done

# Exponential Decay: 3 variants
for decay in 0.90 0.95 0.99; do
    run_variant exponential_decay \
        --decay_factor "$decay"
done

# Periodic Reset: 3 variants
for interval in 5 10 20; do
    run_variant periodic_reset \
        --reset_interval "$interval"
done

echo ""
echo "=== Mitigation sweep complete ==="
echo "  Completed: $COMPLETED / 12"
echo "  Failed: $FAILED / 12"
echo "  Results: results/mitigation/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
