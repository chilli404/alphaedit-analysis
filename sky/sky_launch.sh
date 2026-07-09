#!/usr/bin/env bash
set -euo pipefail

# Launch AlphaEdit experiments on SkyPilot.
#
# Usage:
#   bash sky/sky_launch.sh              # Launch all MVE experiments, all seeds
#   bash sky/sky_launch.sh mve1         # Launch only MVE1, all seeds
#   bash sky/sky_launch.sh mve1 42      # Launch only MVE1, seed 42
#
# Prerequisites:
#   - sky check (cloud credentials configured)
#   - HF_TOKEN environment variable set
#   - Covariance stats available for file_mounts (see below)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SKY_YAML="$SCRIPT_DIR/alphaedit_gpu.yaml"

EXPERIMENT="${1:-all}"
SINGLE_SEED="${2:-}"
SEEDS="${SEEDS:-42 137 2024 7 99}"

echo "=== SkyPilot AlphaEdit Launcher ==="
echo "  Config: $SKY_YAML"
echo "  Experiment: $EXPERIMENT"
echo "  Seeds: ${SINGLE_SEED:-$SEEDS}"
echo ""

launch_job() {
    local exp_name="$1"
    local seed="$2"
    local job_name="ae-${exp_name}-s${seed}"

    echo "Launching: $job_name"
    sky launch "$SKY_YAML" \
        --env "EXPERIMENT_NAME=$exp_name" \
        --env "SEED=$seed" \
        --name "$job_name" \
        --detach-run \
        --env-file "$PROJECT_DIR/.env" \
        -y
    echo "  Submitted: $job_name"
}

# Determine which experiments to run
case "$EXPERIMENT" in
    mve1) EXPERIMENTS=(mve1_alphaedit_mcf) ;;
    mve2) EXPERIMENTS=(mve2_memit_mcf) ;;
    mve3) EXPERIMENTS=(mve3_alphaedit_zsre) ;;
    mve4) EXPERIMENTS=(mve4_conflict_seq) ;;
    failure_curve) EXPERIMENTS=(failure_curve) ;;
    second_model) EXPERIMENTS=(second_model) ;;
    nullspace) EXPERIMENTS=(nullspace_analysis) ;;
    coupling_stress) EXPERIMENTS=(coupling_stress) ;;
    order_sensitivity) EXPERIMENTS=(order_sensitivity) ;;
    capability_probe) EXPERIMENTS=(capability_probe) ;;
    rome) EXPERIMENTS=(rome_baseline) ;;
    mve)  EXPERIMENTS=(mve1_alphaedit_mcf mve2_memit_mcf mve3_alphaedit_zsre mve4_conflict_seq) ;;
    all)  EXPERIMENTS=(mve1_alphaedit_mcf mve2_memit_mcf mve3_alphaedit_zsre mve4_conflict_seq rome_baseline failure_curve second_model nullspace_analysis coupling_stress order_sensitivity capability_probe) ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo "Valid: mve1, mve2, mve3, mve4, rome, failure_curve, second_model, nullspace, coupling_stress, order_sensitivity, capability_probe, mve, all"
        exit 1
        ;;
esac

# Launch jobs
for exp in "${EXPERIMENTS[@]}"; do
    if [[ -n "$SINGLE_SEED" ]]; then
        launch_job "$exp" "$SINGLE_SEED"
    else
        for seed in $SEEDS; do
            launch_job "$exp" "$seed"
        done
    fi
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: sky queue"
echo "View logs:    sky logs <job_name>"
