#!/usr/bin/env bash
set -euo pipefail

# Same-Fact, Different-Key Logit-Damage Experiment
#
# Stage 1 (GPU ~30min): Extract prompt-variant keys
# Stage 2 (GPU ~3h): Run A/B intervention with variant prompts
#
# Usage:
#   bash scripts/run_same_fact_damage.sh SEED
#   bash scripts/run_same_fact_damage.sh 42 stage1   # Key extraction only
#   bash scripts/run_same_fact_damage.sh 42 stage2   # Intervention only (needs stage1)
#   bash scripts/run_same_fact_damage.sh 42 all       # Both stages

SEED="${1:?Usage: $0 SEED [STAGE]}"
STAGE="${2:-all}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
KEYS_PATH="${KEYS_PATH:-$RESULT_ROOT/key_vectors/full_mcf/keys_seed42_layer6.npz}"

VARIANT_KEYS="$RESULT_ROOT/prompt_variant_keys/prompt_variant_keys_seed${SEED}_layer6.npz"

echo "========================================"
echo "Same-Fact Damage Experiment"
echo "  Seed:     $SEED"
echo "  Stage:    $STAGE"
echo "========================================"

cd "$PROJECT_DIR"

if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
    if [[ -f "$VARIANT_KEYS" ]]; then
        echo "Variant keys already exist: $VARIANT_KEYS"
    else
        echo ""
        echo "=== Stage 1: Extracting prompt-variant keys ==="
        uv run python src/experiments/prompt_variant_keys.py \
            --seed "$SEED" \
            --n_records 5000 \
            --output_dir "$RESULT_ROOT/prompt_variant_keys"
    fi
fi

if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    echo ""
    echo "=== Stage 2: Same-fact A/B intervention ==="
    uv run python src/runners/same_fact_damage_runner.py \
        --seed "$SEED" \
        --keys_path "$KEYS_PATH" \
        --variant_keys_path "$VARIANT_KEYS" \
        --install_batches "${INSTALL_BATCHES:-10}" \
        --n_trials "${N_TRIALS:-10}"
fi
