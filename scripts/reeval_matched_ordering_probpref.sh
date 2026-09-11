#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate matched ordering checkpoints with dual metrics (prob-pref + argmax).
#
# Launches SkyPilot jobs in parallel (5 at a time) to re-evaluate checkpoints
# using eval_matched_ordering.py v2, which computes both probability-preference
# (official published metric) and argmax metrics.
#
# Usage:
#   bash scripts/reeval_matched_ordering_probpref.sh                           # Dry run (print all)
#   bash scripts/reeval_matched_ordering_probpref.sh --p0 --10k-only --execute # Fast: 9 jobs, ~45 min
#   bash scripts/reeval_matched_ordering_probpref.sh --p0 --p1 --10k-only --execute  # Core + cross-alg
#   bash scripts/reeval_matched_ordering_probpref.sh --all --sparse --execute  # Everything, 5 checkpoints
#
# Checkpoint subsets:
#   --10k-only   Only batch_99 (10K endpoint). ~40 min/job.
#   --sparse     1K, 3K, 5K, 7K, 10K only. ~1.7h/job.
#   (default)    All available checkpoints. ~3.5h/job.
#
# Priority tiers:
#   --p0   AlphaEdit fb_high/low/random0 × 3 seeds                    (9 jobs)
#   --p1   MEMIT-Seq + PathGuard + EvoEdit on fb orderings, seed 42   (11 jobs)
#   --p2   key_clustered / key_dispersed                               (12 jobs)
#   --p3   fb_random1/2, schedulers, suffix rescue, greedy             (32 jobs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SKY_YAML="$PROJECT_DIR/sky/eval_probpref.yaml"

EXECUTE=false
CHECKPOINTS_OVERRIDE=""
RUN_P0=false
RUN_P1=false
RUN_P2=false
RUN_P3=false
ANY_PRIORITY=false

for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=true ;;
        --p0) RUN_P0=true; ANY_PRIORITY=true ;;
        --p1) RUN_P1=true; ANY_PRIORITY=true ;;
        --p2) RUN_P2=true; ANY_PRIORITY=true ;;
        --p3) RUN_P3=true; ANY_PRIORITY=true ;;
        --all) RUN_P0=true; RUN_P1=true; RUN_P2=true; RUN_P3=true; ANY_PRIORITY=true ;;
        --10k-only) CHECKPOINTS_OVERRIDE="99" ;;
        --sparse) CHECKPOINTS_OVERRIDE="9 29 49 69 99" ;;
        --full5) CHECKPOINTS_OVERRIDE="9 29 49 69 99" ;;
    esac
done

if [[ "$ANY_PRIORITY" == "false" ]]; then
    RUN_P0=true; RUN_P1=true; RUN_P2=true; RUN_P3=true
fi

JOB_COUNT=0
RUNNING=0
PARALLEL=${PARALLEL:-5}

launch_eval() {
    local alg="$1"
    local ordering="$2"
    local seed="$3"
    local priority_tag="$4"
    local extra_env="${5:-}"

    # Shorten names for cluster ID (63-char limit)
    local alg_short="$(echo "$alg" | tr '[:upper:]_' '[:lower:]-')"
    alg_short="${alg_short/memit-seq-lp1.0-ld0.0-cache0/mseq}"
    alg_short="${alg_short/memit-seq-poly2-hybrid-lp1.0-ld0.0-cache0/mspoly}"
    alg_short="${alg_short/pathguard-ed-m200-e0.1/pged}"
    alg_short="${alg_short/pathguard-ed-poly2-hybrid-m200-e0.1/pgpoly}"
    alg_short="${alg_short/alphaedit/ae}"
    alg_short="${alg_short/evoedit/evo}"

    local ord_short="$ordering"
    ord_short="${ord_short/fb_high_exposure/fbhi}"
    ord_short="${ord_short/fb_low_exposure/fblo}"
    ord_short="${ord_short/fb_random/fbrnd}"
    ord_short="${ord_short/sched_balanced/schbal}"
    ord_short="${ord_short/sched_conditioning_only/schcnd}"
    ord_short="${ord_short/sched_exposure_only/schexp}"
    ord_short="${ord_short/suffix_random_late/sfrl}"
    ord_short="${ord_short/suffix_random/sfrnd}"
    ord_short="${ord_short/suffix_spread_late/sfsl}"
    ord_short="${ord_short/suffix_spread/sfsprd}"
    ord_short="${ord_short/key_clustered/kclust}"
    ord_short="${ord_short/key_dispersed/kdisp}"
    ord_short="${ord_short/greedy_minmax/greedy}"
    ord_short="${ord_short/cluster_topo/ctopo}"

    local cluster_name="ppv2-${alg_short}-${ord_short}-s${seed}"
    cluster_name="${cluster_name:0:63}"

    JOB_COUNT=$((JOB_COUNT + 1))

    if [[ "$EXECUTE" == "true" ]]; then
        echo "  [$priority_tag #$JOB_COUNT] Launching: $cluster_name"
        local -a cmd=(sky launch "$SKY_YAML"
            --env-file "$PROJECT_DIR/.env"
            --env "SEED=$seed"
            --env "ALG_NAME=$alg"
            --env "ORDERING=$ordering")
        if [[ -n "$CHECKPOINTS_OVERRIDE" ]]; then
            cmd+=(--env "CHECKPOINTS=$CHECKPOINTS_OVERRIDE")
        fi
        if [[ -n "${extra_env:-}" ]]; then
            cmd+=($extra_env)
        fi
        cmd+=(--cluster "$cluster_name" --detach-run -y)
        "${cmd[@]}" &

        RUNNING=$((RUNNING + 1))
        if [[ "$RUNNING" -ge "$PARALLEL" ]]; then
            echo "    [batch of $RUNNING provisioning — waiting for them to finish...]"
            wait
            RUNNING=0
        fi
    else
        echo "  [$priority_tag #$JOB_COUNT] $cluster_name"
        local ckpt_str=""
        if [[ -n "$CHECKPOINTS_OVERRIDE" ]]; then
            ckpt_str=" --env 'CHECKPOINTS=$CHECKPOINTS_OVERRIDE'"
        fi
        echo "    sky launch $SKY_YAML --env SEED=$seed --env ALG_NAME=$alg --env ORDERING=$ordering$ckpt_str --cluster $cluster_name --detach-run -y"
        echo ""
    fi
}

echo "=== Matched Ordering Prob-Pref Re-evaluation ==="
echo "  YAML:        $SKY_YAML"
echo "  Execute:     $EXECUTE"
echo "  Priorities:  p0=$RUN_P0 p1=$RUN_P1 p2=$RUN_P2 p3=$RUN_P3"
echo "  Checkpoints: ${CHECKPOINTS_OVERRIDE:-auto-detect all}"
echo "  Parallelism: $PARALLEL concurrent clusters"
echo ""

# ============================================================================
# P0: AlphaEdit fixed-batch dose-response (9 jobs)
#     fb_high/low/random0 × 3 seeds — core Table 2 / Figure 3-4
#     These have per-case files already rescored, but GPU re-eval gives
#     the v2 full_eval format with cohort breakdowns at every checkpoint.
# ============================================================================

if [[ "$RUN_P0" == "true" ]]; then
    echo "--- P0: AlphaEdit fixed-batch dose-response (9 jobs) ---"
    for seed in 42 137 2024; do
        for ordering in fb_high_exposure fb_low_exposure fb_random0; do
            launch_eval "AlphaEdit" "$ordering" "$seed" "P0"
        done
    done
    echo ""
fi

# ============================================================================
# P1: Cross-algorithm comparison (11 jobs)
#     Only algorithms × orderings × seeds that have checkpoints on S3.
#     Verified checkpoint existence 2026-09-08.
# ============================================================================

if [[ "$RUN_P1" == "true" ]]; then
    echo "--- P1: Cross-algorithm comparison (11 jobs) ---"

    # MEMIT-Seq: checkpoints exist for fb_high/low/random0 × seed42 only
    for ordering in fb_high_exposure fb_low_exposure fb_random0; do
        launch_eval "MEMIT-Seq-lp1.0-ld0.0-cache0" "$ordering" "42" "P1"
    done

    # PathGuard-ED: checkpoints exist for fb_high/low/random0 × seed42 only
    for ordering in fb_high_exposure fb_low_exposure fb_random0; do
        launch_eval "PathGuard-ED-M200-e0.1" "$ordering" "42" "P1"
    done

    # PathGuard-ED-poly2-hybrid: wider coverage
    #   seed42: fb_low, fb_random0
    #   seed2024: fb_high, fb_low, fb_random0
    #   seed137: fb_low, fb_random0
    # (some of these already have argmax full_eval; all need prob-pref)
    launch_eval "PathGuard-ED-poly2-hybrid-M200-e0.1" "fb_low_exposure" "42" "P1"
    launch_eval "PathGuard-ED-poly2-hybrid-M200-e0.1" "fb_random0" "42" "P1"

    # EvoEdit: checkpoints at results/evoedit/{ordering}/seed42/10000edits/EvoEdit/run_000/checkpoints/
    #   Uses edits_NNNNNN naming (not batch_N). Need CHECKPOINT_DIR override.
    #   fb_random0/seed42 already has v2 on S3 — skip it.
    echo "  [NOTE] EvoEdit uses edits_NNNNNN checkpoint format — needs custom CHECKPOINT_DIR"
    echo "  [NOTE] fb_random0/seed42 already has v2 on S3 — skipping"
    # These will need the YAML to handle the different checkpoint path.
    # For now, launch with the standard path — the YAML auto-detects and will
    # fall back to the evoedit results path if the standard matched_ordering path fails.
    launch_eval "EvoEdit" "fb_high_exposure" "42" "P1"
    launch_eval "EvoEdit" "fb_low_exposure" "42" "P1"

    echo ""
fi

# ============================================================================
# P2: Key geometry orderings (12 jobs)
#     key_clustered / key_dispersed for AlphaEdit + MEMIT-Seq
# ============================================================================

if [[ "$RUN_P2" == "true" ]]; then
    echo "--- P2: Key geometry orderings ---"
    # AlphaEdit: all seeds have checkpoints
    for seed in 42 137 2024; do
        for ordering in key_clustered key_dispersed; do
            launch_eval "AlphaEdit" "$ordering" "$seed" "P2"
        done
    done
    # MEMIT-Seq: only seed42 has checkpoints on S3
    for ordering in key_clustered key_dispersed; do
        launch_eval "MEMIT-Seq-lp1.0-ld0.0-cache0" "$ordering" "42" "P2"
    done
    echo ""
fi

# ============================================================================
# P3: Remaining AlphaEdit orderings (32 jobs)
#     fb_random1/2, schedulers, suffix rescue, greedy — appendix
# ============================================================================

if [[ "$RUN_P3" == "true" ]]; then
    echo "--- P3: Remaining orderings — appendix (32 jobs) ---"

    # fb_random1/2 (complete the 5-path dose-response, Table 2)
    for seed in 42 137 2024; do
        for ordering in fb_random1 fb_random2; do
            launch_eval "AlphaEdit" "$ordering" "$seed" "P3"
        done
    done

    # Schedulers (appendix Table 5)
    for seed in 42 137 2024; do
        for ordering in sched_balanced sched_exposure_only sched_conditioning_only; do
            launch_eval "AlphaEdit" "$ordering" "$seed" "P3"
        done
    done

    # Suffix rescue (Table 3, Figure 5)
    for seed in 42 137 2024; do
        for ordering in suffix_random suffix_spread suffix_random_late suffix_spread_late; do
            launch_eval "AlphaEdit" "$ordering" "$seed" "P3"
        done
    done

    # Greedy / random / misc (AlphaEdit only — MEMIT-Seq has no greedy checkpoints)
    for seed in 42 2024; do
        launch_eval "AlphaEdit" "greedy_minmax" "$seed" "P3"
    done
    launch_eval "AlphaEdit" "random" "42" "P3"
    launch_eval "AlphaEdit" "cluster_topo" "42" "P3"

    echo ""
fi

# Wait for any remaining provisioning
if [[ "$EXECUTE" == "true" ]] && [[ "$RUNNING" -gt 0 ]]; then
    echo "  [waiting for final $RUNNING to finish provisioning...]"
    wait
fi

echo "=== Summary ==="
echo "  Total jobs: $JOB_COUNT"
if [[ "$EXECUTE" == "false" ]]; then
    echo ""
    echo "  DRY RUN — no jobs launched. Add --execute to run."
    echo ""
    echo "  Recommended:"
    echo "    # Fast: just 10K endpoints for core tables (~45 min at 8 GPUs)"
    echo "    bash scripts/reeval_matched_ordering_probpref.sh --p0 --p1 --10k-only --execute"
    echo ""
    echo "    # Full: all checkpoints for temporal trajectory figures"
    echo "    bash scripts/reeval_matched_ordering_probpref.sh --p0 --sparse --execute"
    echo ""
    echo "    # Everything for submission"
    echo "    bash scripts/reeval_matched_ordering_probpref.sh --all --10k-only --execute"
fi
echo ""
echo "  Monitor: sky queue"
echo "  Logs:    sky logs <cluster-name>"
echo "  Teardown: sky down -a"
