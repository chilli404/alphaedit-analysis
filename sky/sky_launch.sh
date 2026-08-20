#!/usr/bin/env bash
set -euo pipefail

# Launch AlphaEdit experiments on SkyPilot.
#
# Seed policy:
#   - Core reproduction (MVE1-4): 5 seeds (42, 137, 2024, 7, 99)
#   - Extensions (failure_curve, nullspace, coupling, etc.): 3 seeds (42, 137, 2024)
#
# Usage:
#   bash sky/sky_launch.sh              # Launch all experiments with appropriate seeds
#   bash sky/sky_launch.sh mve1         # Launch MVE1, 5 seeds
#   bash sky/sky_launch.sh mve1 42      # Launch MVE1, seed 42 only
#   bash sky/sky_launch.sh coupling_stress      # 3 seeds
#   bash sky/sky_launch.sh coupling_stress 42   # Single seed override
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

# Seed lists: MVE uses 5 seeds, extensions use 3
MVE_SEEDS="${MVE_SEEDS:-42 137 2024 7 99}"
EXT_SEEDS="${EXT_SEEDS:-42 137 2024}"

# MVE experiments (5 seeds)
MVE_EXPERIMENTS="mve1_alphaedit_mcf mve2_memit_mcf mve3_alphaedit_zsre"

seeds_for_experiment() {
    local exp="$1"
    for mve_exp in $MVE_EXPERIMENTS; do
        if [[ "$exp" == "$mve_exp" ]]; then
            echo "$MVE_SEEDS"
            return
        fi
    done
    echo "$EXT_SEEDS"
}

echo "=== SkyPilot AlphaEdit Launcher ==="
echo "  Config: $SKY_YAML"
echo "  Experiment: $EXPERIMENT"
echo "  MVE seeds (5): $MVE_SEEDS"
echo "  Extension seeds (3): $EXT_SEEDS"
if [[ -n "$SINGLE_SEED" ]]; then
    echo "  Override: seed $SINGLE_SEED only"
fi
echo ""

cluster_exists() {
    sky status 2>/dev/null | sed $'s/\033\[[0-9;]*m//g' | grep -q "^$1 "
}

launch_job() {
    local exp_name="$1"
    local seed="$2"

    # For failure_curve experiments with ALG_NAME=both, split into separate clusters
    if [[ "$exp_name" == "failure_curve"* ]] && [[ "${ALG_NAME:-both}" == "both" ]]; then
        echo "Splitting 'both' into separate AlphaEdit and MEMIT clusters"

        # Launch AlphaEdit cluster
        local saved_alg_name="$ALG_NAME"
        ALG_NAME="AlphaEdit" launch_single_job "$exp_name" "$seed"

        # Launch MEMIT cluster
        ALG_NAME="MEMIT" launch_single_job "$exp_name" "$seed"

        # Restore original ALG_NAME
        ALG_NAME="$saved_alg_name"
        return
    fi

    # For comparison_ordered: split into per-(algorithm, order) clusters
    if [[ "$exp_name" == "comparison_ordered" ]]; then
        local saved_alg="$ALG_NAME"
        local saved_order="${ORDER_ID:-}"

        # Determine algorithms
        local algs=()
        case "${ALG_NAME:-all}" in
            all)   algs=("AlphaEdit" "MEMIT" "MEMIT_seq") ;;
            both)  algs=("AlphaEdit" "MEMIT") ;;
            *)     algs=("$ALG_NAME") ;;
        esac

        # Determine orders
        local orders=()
        case "${ORDER_ID:-both}" in
            both) orders=(0 1) ;;
            *)    orders=("$ORDER_ID") ;;
        esac

        echo "Splitting comparison_ordered into ${#algs[@]} algs × ${#orders[@]} orders = $(( ${#algs[@]} * ${#orders[@]} )) clusters"
        for alg in "${algs[@]}"; do
            for order in "${orders[@]}"; do
                ALG_NAME="$alg" ORDER_ID="$order" launch_single_job "$exp_name" "$seed"
            done
        done

        ALG_NAME="$saved_alg"
        ORDER_ID="$saved_order"
        return
    fi

    # For interference_experiment phase2: split into 2 clusters by ordering
    if [[ "$exp_name" == "interference_experiment" ]] && [[ "${PHASE:-}" == "phase2" ]] && [[ -z "${ORDERING:-}" ]]; then
        echo "Splitting interference phase2 into 2 clusters (key_clustered, key_dispersed)"
        local saved_ordering="${ORDERING:-}"
        for ordering in key_clustered key_dispersed; do
            ORDERING="$ordering" launch_single_job "$exp_name" "$seed"
        done
        ORDERING="$saved_ordering"
        return
    fi

    # For matched_ordering: split into all 4 combinations {MEMIT-Seq,AlphaEdit} × {key_clustered,key_dispersed}
    if [[ "$exp_name" == "matched_ordering" ]] && [[ "${ALG_NAME:-all}" == "all" ]]; then
        echo "Splitting matched_ordering into 4 clusters (2 algs × 2 orderings)"
        local saved_alg="${ALG_NAME:-all}"
        local saved_ordering="${ORDERING:-}"
        for alg in MEMIT-Seq-lp1.0-ld0.0-cache0 AlphaEdit; do
            for ordering in key_clustered key_dispersed; do
                ALG_NAME="$alg" ORDERING="$ordering" launch_single_job "$exp_name" "$seed"
            done
        done
        ALG_NAME="$saved_alg"
        ORDERING="$saved_ordering"
        return
    fi

    launch_single_job "$exp_name" "$seed"
}

launch_single_job() {
    local exp_name="$1"
    local seed="$2"

    # Include algorithm name and edit count in cluster name for failure_curve experiments
    local job_name="ae-${exp_name}-s${seed}"
    if [[ "$exp_name" == "failure_curve"* ]]; then
        if [[ -n "${ALG_NAME:-}" ]] && [[ "${ALG_NAME}" != "both" ]]; then
            local alg_tag="$(echo ${ALG_NAME} | tr '[:upper:]' '[:lower:]')"
            if [[ "${INJECT_C0:-}" == "true" ]]; then
                alg_tag="${alg_tag}-c0"
            fi
            job_name="ae-${exp_name}-${alg_tag}-s${seed}"
        fi
        if [[ -n "${TARGET_EDITS:-}" ]]; then
            job_name="${job_name}-${TARGET_EDITS}e"
        fi
    fi
    # Include algorithm and order in cluster name for comparison_ordered
    if [[ "$exp_name" == "comparison_ordered" ]]; then
        local alg_lower="$(echo ${ALG_NAME:-unknown} | tr '[:upper:]_' '[:lower:]-')"
        job_name="ae-cmpord-${alg_lower}-o${ORDER_ID:-0}-s${seed}"
        if [[ -n "${TARGET_EDITS:-}" ]]; then
            job_name="${job_name}-${TARGET_EDITS}e"
        fi
    fi
    # Include phase and ordering in cluster name for interference_experiment
    if [[ "$exp_name" == "interference_experiment" ]]; then
        local phase_short="${PHASE:-all}"
        if [[ -n "${ORDERING:-}" ]]; then
            local ord_short="${ORDERING}"
            ord_short="${ord_short/key_clustered/kclust}"
            ord_short="${ord_short/key_dispersed/kdisp}"
            job_name="ae-interf-${phase_short}-${ord_short}-s${seed}"
        else
            job_name="ae-interf-${phase_short}-s${seed}"
        fi
    fi
    # Polykernel editor GPT-J: include C0 in cluster name
    if [[ "$exp_name" == "polykernel_editor_gptj" ]]; then
        if [[ "${INJECT_C0:-}" == "true" ]]; then
            job_name="ae-poly2-gptj-c0-s${seed}"
        else
            job_name="ae-poly2-gptj-s${seed}"
        fi
    fi
    # Include threshold and cell in cluster name for projection_sweep_gptj
    if [[ "$exp_name" == "projection_sweep_gptj" ]]; then
        local t_short="${NULLSPACE_THRESHOLD:-0.02}"
        local cell_short="${SWEEP_CELL:-both}"
        cell_short="$(echo $cell_short | tr '[:upper:]' '[:lower:]')"
        job_name="ae-psweep-t${t_short}-${cell_short}-s${seed}"
    fi
    # Include algorithm and ordering in cluster name for matched_ordering
    if [[ "$exp_name" == "matched_ordering" ]]; then
        local alg_short="$(echo ${ALG_NAME:-unknown} | tr '[:upper:]_' '[:lower:]-')"
        # Shorten ordering: key_clustered→kclust, key_dispersed→kdisp
        local ord_short="${ORDERING:-unknown}"
        ord_short="${ord_short/key_clustered/kclust}"
        ord_short="${ord_short/key_dispersed/kdisp}"
        job_name="ae-mo-${alg_short}-${ord_short}-s${seed}"
    fi

    # Extra env vars for checkpointed experiments
    local extra_envs=""
    if [[ -n "${TARGET_EDITS:-}" ]]; then
        extra_envs="$extra_envs --env TARGET_EDITS=$TARGET_EDITS"
    fi
    if [[ -n "${ALG_NAME:-}" ]]; then
        extra_envs="$extra_envs --env ALG_NAME=$ALG_NAME"
    fi
    if [[ -n "${SAVE_INTERVAL:-}" ]]; then
        extra_envs="$extra_envs --env SAVE_INTERVAL=$SAVE_INTERVAL"
    fi
    if [[ -n "${LAMBDA_PREV:-}" ]]; then
        extra_envs="$extra_envs --env LAMBDA_PREV=$LAMBDA_PREV"
    fi
    if [[ -n "${LAMBDA_DELTA:-}" ]]; then
        extra_envs="$extra_envs --env LAMBDA_DELTA=$LAMBDA_DELTA"
    fi
    if [[ -n "${MOM2_OVERRIDE:-}" ]]; then
        extra_envs="$extra_envs --env MOM2_OVERRIDE=$MOM2_OVERRIDE"
    fi
    if [[ "${INJECT_C0:-}" == "true" ]]; then
        extra_envs="$extra_envs --env INJECT_C0=true"
    fi
    if [[ -n "${C0_WEIGHT:-}" ]]; then
        extra_envs="$extra_envs --env C0_WEIGHT=$C0_WEIGHT"
    fi
    if [[ -n "${FAST_CHECKPOINT:-}" ]]; then
        extra_envs="$extra_envs --env FAST_CHECKPOINT=$FAST_CHECKPOINT"
    fi
    if [[ -n "${EVAL_AT_CHECKPOINTS_ONLY:-}" ]]; then
        extra_envs="$extra_envs --env EVAL_AT_CHECKPOINTS_ONLY=$EVAL_AT_CHECKPOINTS_ONLY"
    fi
    if [[ -n "${CHECKPOINT_BATCH:-}" ]]; then
        extra_envs="$extra_envs --env CHECKPOINT_BATCH=$CHECKPOINT_BATCH"
    fi
    if [[ -n "${MODEL_NAME:-}" ]]; then
        extra_envs="$extra_envs --env MODEL_NAME=$MODEL_NAME"
    fi
    if [[ -n "${HPARAMS_FNAME:-}" ]]; then
        extra_envs="$extra_envs --env HPARAMS_FNAME=$HPARAMS_FNAME"
    fi
    if [[ -n "${KERNEL_TYPE:-}" ]]; then
        extra_envs="$extra_envs --env KERNEL_TYPE=$KERNEL_TYPE"
    fi
    if [[ -n "${KERNEL_SIGMA:-}" ]]; then
        extra_envs="$extra_envs --env KERNEL_SIGMA=$KERNEL_SIGMA"
    fi
    if [[ -n "${KERNEL_DEGREE:-}" ]]; then
        extra_envs="$extra_envs --env KERNEL_DEGREE=$KERNEL_DEGREE"
    fi
    if [[ -n "${DATASET_SIZE_LIMIT:-}" ]]; then
        extra_envs="$extra_envs --env DATASET_SIZE_LIMIT=$DATASET_SIZE_LIMIT"
    fi
    if [[ -n "${EDIT_ONLY:-}" ]]; then
        extra_envs="$extra_envs --env EDIT_ONLY=$EDIT_ONLY"
    fi
    if [[ -n "${EVAL_ONLY:-}" ]]; then
        extra_envs="$extra_envs --env EVAL_ONLY=$EVAL_ONLY"
    fi
    if [[ -n "${LOAD_CHECKPOINT:-}" ]]; then
        extra_envs="$extra_envs --env LOAD_CHECKPOINT=$LOAD_CHECKPOINT"
    fi
    if [[ -n "${SAVE_INTERVAL:-}" ]]; then
        extra_envs="$extra_envs --env SAVE_INTERVAL=$SAVE_INTERVAL"
    fi
    if [[ -n "${ORDER_ID:-}" ]]; then
        extra_envs="$extra_envs --env ORDER_ID=$ORDER_ID"
    fi
    if [[ -n "${STREAM:-}" ]]; then
        extra_envs="$extra_envs --env STREAM=$STREAM"
    fi
    if [[ -n "${ORDERING:-}" ]]; then
        extra_envs="$extra_envs --env ORDERING=$ORDERING"
    fi
    if [[ -n "${PHASE:-}" ]]; then
        extra_envs="$extra_envs --env PHASE=$PHASE"
    fi
    if [[ -n "${CONTINUE_FROM_RUN:-}" ]]; then
        extra_envs="$extra_envs --env CONTINUE_FROM_RUN=$CONTINUE_FROM_RUN"
    fi
    if [[ -n "${NULLSPACE_THRESHOLD:-}" ]]; then
        extra_envs="$extra_envs --env NULLSPACE_THRESHOLD=$NULLSPACE_THRESHOLD"
    fi
    if [[ -n "${SWEEP_CELL:-}" ]]; then
        extra_envs="$extra_envs --env SWEEP_CELL=$SWEEP_CELL"
    fi

    # GPU override (e.g., GPU_TYPE="A100:1" for memory-intensive jobs)
    local gpu_flag=""
    if [[ -n "${GPU_TYPE:-}" ]]; then
        gpu_flag="--gpus $GPU_TYPE"
    fi

    if cluster_exists "$job_name"; then
        echo "Exec on existing cluster: $job_name"
        sky exec "$job_name" \
            --env-file "$PROJECT_DIR/.env" \
            --env "EXPERIMENT_NAME=$exp_name" \
            --env "SEED=$seed" \
            $extra_envs \
            --detach-run \
            "$SKY_YAML"
    else
        echo "Creating new cluster: $job_name"
        sky launch "$SKY_YAML" \
            --env-file "$PROJECT_DIR/.env" \
            --env "EXPERIMENT_NAME=$exp_name" \
            --env "SEED=$seed" \
            $extra_envs \
            $gpu_flag \
            --cluster "$job_name" \
            --detach-run \
            -y
    fi
    echo "  Submitted: $job_name"
}

# Determine which experiments to run
case "$EXPERIMENT" in
    mve1) EXPERIMENTS=(mve1_alphaedit_mcf) ;;
    mve2) EXPERIMENTS=(mve2_memit_mcf) ;;
    mve3) EXPERIMENTS=(mve3_alphaedit_zsre) ;;
    failure_curve|failure_curve_ckpt) EXPERIMENTS=(failure_curve_checkpointed) ;;
    capability_probe) EXPERIMENTS=(capability_probe) ;;
    capability_probe_offline) EXPERIMENTS=(capability_probe_offline) ;;
    memit_sequential) EXPERIMENTS=(memit_sequential) ;;
    polykernel) EXPERIMENTS=(polykernel_diagnostic) ;;
    polykernel_editor) EXPERIMENTS=(polykernel_editor) ;;
    mechanism_analysis) EXPERIMENTS=(mechanism_analysis) ;;
    comparison_ordered) EXPERIMENTS=(comparison_ordered) ;;
    matched_ordering) EXPERIMENTS=(matched_ordering) ;;
    interference) EXPERIMENTS=(interference_experiment) ;;
    poly2_diagnostic) EXPERIMENTS=(poly2_diagnostic) ;;
    polykernel_seqreg) EXPERIMENTS=(polykernel_seqreg) ;;
    # Cross-model experiments (Qwen2.5-7B-Instruct)
    mve1_qwen) EXPERIMENTS=(mve1_qwen_mcf) ;;
    mve2_qwen) EXPERIMENTS=(mve2_qwen_memit_mcf) ;;
    failure_curve_qwen) EXPERIMENTS=(failure_curve_qwen) ;;
    memit_sequential_qwen) EXPERIMENTS=(memit_sequential_qwen) ;;
    # Cross-model experiments (GPT-J-6B)
    mve1_gptj) EXPERIMENTS=(mve1_gptj_mcf) ;;
    mve2_gptj) EXPERIMENTS=(mve2_gptj_memit_mcf) ;;
    failure_curve_gptj) EXPERIMENTS=(failure_curve_gptj) ;;
    polykernel_editor_gptj) EXPERIMENTS=(polykernel_editor_gptj) ;;
    projection_sweep_gptj)
        # "both" launches two separate clusters for parallelism
        if [[ "${SWEEP_CELL:-both}" == "both" ]]; then
            EXPERIMENTS=(projection_sweep_gptj projection_sweep_gptj)
            _SWEEP_CELLS=(AlphaEdit AlphaEdit-C0)
        else
            EXPERIMENTS=(projection_sweep_gptj)
        fi
        ;;
    memit_sequential_gptj) EXPERIMENTS=(memit_sequential_gptj) ;;
    # Cross-model bundles
    cross_model_qwen) EXPERIMENTS=(mve1_qwen_mcf mve2_qwen_memit_mcf) ;;
    cross_model_gptj) EXPERIMENTS=(mve1_gptj_mcf mve2_gptj_memit_mcf) ;;
    cross_model) EXPERIMENTS=(mve1_qwen_mcf mve2_qwen_memit_mcf mve1_gptj_mcf mve2_gptj_memit_mcf) ;;
    mve)  EXPERIMENTS=(mve1_alphaedit_mcf mve2_memit_mcf mve3_alphaedit_zsre) ;;
    all)  EXPERIMENTS=(mve1_alphaedit_mcf mve2_memit_mcf mve3_alphaedit_zsre failure_curve_checkpointed capability_probe_offline) ;;
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo "Valid: mve1, mve2, mve3, failure_curve, failure_curve_ckpt, capability_probe, capability_probe_offline, memit_sequential, polykernel, polykernel_editor, polykernel_seqreg, mechanism_analysis, comparison_ordered, matched_ordering, interference, poly2_diagnostic, mve, all"
        echo "Cross-model: mve1_qwen, mve2_qwen, failure_curve_qwen, memit_sequential_qwen, mve1_gptj, mve2_gptj, failure_curve_gptj, memit_sequential_gptj, cross_model_qwen, cross_model_gptj, cross_model"
        exit 1
        ;;
esac

# Launch jobs
_exp_idx=0
for exp in "${EXPERIMENTS[@]}"; do
    # For projection_sweep_gptj "both": override SWEEP_CELL per iteration
    if [[ -n "${_SWEEP_CELLS+x}" ]] && [[ $_exp_idx -lt ${#_SWEEP_CELLS[@]} ]]; then
        export SWEEP_CELL="${_SWEEP_CELLS[$_exp_idx]}"
    fi
    _exp_idx=$((_exp_idx + 1))

    if [[ -n "$SINGLE_SEED" ]]; then
        launch_job "$exp" "$SINGLE_SEED"
    else
        local_seeds=$(seeds_for_experiment "$exp")
        for seed in $local_seeds; do
            launch_job "$exp" "$seed"
        done
    fi
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: sky queue"
echo "View logs:    sky logs <job_name>"
