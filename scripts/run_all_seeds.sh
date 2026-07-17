#!/usr/bin/env bash
set -euo pipefail

# Run experiments across seeds.
# This is the meta-runner for the full reproduction.
#
# Seed policy:
#   - Core reproduction (MVE1-4): 5 seeds (42, 137, 2024, 7, 99)
#   - Extensions (failure_curve, nullspace, coupling, etc.): 3 seeds (42, 137, 2024)
#
# Usage: bash scripts/run_all_seeds.sh [EXPERIMENT]
#   EXPERIMENT: mve1, mve2, mve3, mve4, mve, failure_curve, second_model,
#              nullspace, coupling_stress, order_sensitivity, capability, all
#
# Examples:
#   bash scripts/run_all_seeds.sh mve1             # MVE1 × 5 seeds
#   bash scripts/run_all_seeds.sh coupling_stress  # Coupling × 3 seeds
#   bash scripts/run_all_seeds.sh all              # Everything with appropriate seeds
#
# Environment variables:
#   CUDA_DEVICE: GPU index (default: 0)
#   MVE_SEEDS: Override MVE seed list (default: "42 137 2024 7 99")
#   EXT_SEEDS: Override extension seed list (default: "42 137 2024")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT="${1:-all}"

# Core reproduction uses 5 seeds; extensions use 3
MVE_SEEDS="${MVE_SEEDS:-42 137 2024 7 99}"
EXT_SEEDS="${EXT_SEEDS:-42 137 2024}"

echo "=== AlphaEdit Replication: Full Seed Sweep ==="
echo "Experiment: $EXPERIMENT"
echo "MVE seeds (5): $MVE_SEEDS"
echo "Extension seeds (3): $EXT_SEEDS"
echo "CUDA device: ${CUDA_DEVICE:-0}"
echo ""

run_experiment() {
    local exp_script="$1"
    local seed="$2"
    echo ""
    echo "--- Running $exp_script with seed=$seed ---"
    bash "$SCRIPT_DIR/$exp_script" "$seed"
    echo "--- Finished $exp_script seed=$seed ---"
    echo ""
}

run_with_seeds() {
    local seeds="$1"
    shift
    for seed in $seeds; do
        for script in "$@"; do
            run_experiment "$script" "$seed"
        done
    done
}

case "$EXPERIMENT" in
    mve1)
        run_with_seeds "$MVE_SEEDS" "run_mve1_alphaedit_mcf.sh"
        ;;
    mve2)
        run_with_seeds "$MVE_SEEDS" "run_mve2_memit_mcf.sh"
        ;;
    mve3)
        run_with_seeds "$MVE_SEEDS" "run_mve3_alphaedit_zsre.sh"
        ;;
    mve4)
        run_with_seeds "$MVE_SEEDS" "run_mve4_conflict_seq.sh"
        ;;
    mve)
        run_with_seeds "$MVE_SEEDS" \
            "run_mve1_alphaedit_mcf.sh" \
            "run_mve2_memit_mcf.sh" \
            "run_mve3_alphaedit_zsre.sh" \
            "run_mve4_conflict_seq.sh"
        ;;
    failure_curve)
        run_with_seeds "$EXT_SEEDS" "run_failure_curve_checkpointed.sh"
        ;;
    second_model)
        run_with_seeds "$EXT_SEEDS" "run_second_model.sh"
        ;;
    nullspace)
        run_with_seeds "$EXT_SEEDS" "run_nullspace_analysis.sh"
        ;;
    coupling_stress)
        run_with_seeds "$EXT_SEEDS" "run_coupling_stress.sh"
        ;;
    order_sensitivity)
        run_with_seeds "$EXT_SEEDS" "run_order_sensitivity.sh"
        ;;
    capability)
        run_with_seeds "$EXT_SEEDS" "run_capability_probe.sh"
        ;;
    all)
        # Core reproduction: 5 seeds
        run_with_seeds "$MVE_SEEDS" \
            "run_mve1_alphaedit_mcf.sh" \
            "run_mve2_memit_mcf.sh" \
            "run_mve3_alphaedit_zsre.sh" \
            "run_mve4_conflict_seq.sh"
        # Extensions: 3 seeds
        run_with_seeds "$EXT_SEEDS" \
            "run_failure_curve_checkpointed.sh" \
            "run_second_model.sh" \
            "run_nullspace_analysis.sh" \
            "run_capability_probe.sh" \
            "run_coupling_stress.sh" \
            "run_order_sensitivity.sh"
        ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo "Valid options: mve1, mve2, mve3, mve4, failure_curve, second_model, nullspace, coupling_stress, order_sensitivity, capability, mve, all"
        exit 1
        ;;
esac

echo ""
echo "=== All runs complete ==="
echo "Results: vendor/AlphaEdit/results/"
echo "Metadata: results/metadata/"
