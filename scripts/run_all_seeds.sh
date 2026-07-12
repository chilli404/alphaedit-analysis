#!/usr/bin/env bash
set -euo pipefail

# Run all MVE experiments across all seeds.
# This is the meta-runner for the full reproduction.
#
# Usage: bash scripts/run_all_seeds.sh [EXPERIMENT]
#   EXPERIMENT: mve1, mve2, mve3, mve4, or all (default: all)
#
# Environment variables:
#   CUDA_DEVICE: GPU index (default: 0)
#   SEEDS: Space-separated seed list (default: "42 137 2024")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT="${1:-all}"
SEEDS="${SEEDS:-42 137 2024 7 99}"

echo "=== AlphaEdit Replication: Full Seed Sweep ==="
echo "Experiment: $EXPERIMENT"
echo "Seeds: $SEEDS"
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

for seed in $SEEDS; do
    case "$EXPERIMENT" in
        mve1)
            run_experiment "run_mve1_alphaedit_mcf.sh" "$seed"
            ;;
        mve2)
            run_experiment "run_mve2_memit_mcf.sh" "$seed"
            ;;
        mve3)
            run_experiment "run_mve3_alphaedit_zsre.sh" "$seed"
            ;;
        mve4)
            run_experiment "run_mve4_conflict_seq.sh" "$seed"
            ;;
        failure_curve)
            run_experiment "run_failure_curve.sh" "$seed"
            ;;
        second_model)
            run_experiment "run_second_model.sh" "$seed"
            ;;
        mve)
            run_experiment "run_mve1_alphaedit_mcf.sh" "$seed"
            run_experiment "run_mve2_memit_mcf.sh" "$seed"
            run_experiment "run_mve3_alphaedit_zsre.sh" "$seed"
            run_experiment "run_mve4_conflict_seq.sh" "$seed"
            ;;
        nullspace)
            run_experiment "run_nullspace_analysis.sh" "$seed"
            ;;
        coupling_stress)
            run_experiment "run_coupling_stress.sh" "$seed"
            ;;
        order_sensitivity)
            run_experiment "run_order_sensitivity.sh" "$seed"
            ;;
        rome)
            run_experiment "run_rome_baseline.sh" "$seed"
            ;;
        capability)
            run_experiment "run_capability_probe.sh" "$seed"
            ;;
        all)
            run_experiment "run_mve1_alphaedit_mcf.sh" "$seed"
            run_experiment "run_mve2_memit_mcf.sh" "$seed"
            run_experiment "run_mve3_alphaedit_zsre.sh" "$seed"
            run_experiment "run_mve4_conflict_seq.sh" "$seed"
            run_experiment "run_rome_baseline.sh" "$seed"
            run_experiment "run_failure_curve.sh" "$seed"
            run_experiment "run_second_model.sh" "$seed"
            run_experiment "run_nullspace_analysis.sh" "$seed"
            run_experiment "run_capability_probe.sh" "$seed"
            run_experiment "run_coupling_stress.sh" "$seed"
            run_experiment "run_order_sensitivity.sh" "$seed"
            ;;
        *)
            echo "ERROR: Unknown experiment '$EXPERIMENT'"
            echo "Valid options: mve1, mve2, mve3, mve4, rome, failure_curve, second_model, nullspace, coupling_stress, order_sensitivity, capability, mve, all"
            exit 1
            ;;
    esac
done

echo ""
echo "=== All runs complete ==="
echo "Results: vendor/AlphaEdit/results/"
echo "Metadata: results/metadata/"
