#!/usr/bin/env bash
set -uo pipefail
# ============================================================================
# GPU Smoke Test: 1 batch (100 edits) of EVERY algorithm × runner combination.
#
# Each algorithm gets a 10-minute timeout. Verbose logging throughout.
#
# REQUIRES: GPU (L40S or equivalent, ~48GB VRAM)
# TIME: ~45-90 min total
#
# Usage:
#   bash tests/test_smoke_all_algorithms.sh              # local
#   bash tests/test_smoke_all_algorithms.sh --skypilot    # on SkyPilot cluster
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SKYPILOT=0
[[ "${1:-}" == "--skypilot" ]] && SKYPILOT=1

if [ "$SKYPILOT" -eq 1 ]; then
    export RESULT_ROOT="/s3-data/continual-learning/alphaedit/results/_smoke_test"
    export CHECKPOINT_ROOT="/s3-data/continual-learning/alphaedit/checkpoints/_smoke_test"
else
    TMPDIR=$(mktemp -d)
    export RESULT_ROOT="$TMPDIR/results"
    export CHECKPOINT_ROOT="$TMPDIR/checkpoints"
    mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT"
    trap "rm -rf $TMPDIR" EXIT
fi

SEED=42
EDITS=10
DATASET_LIMIT=20  # 2 batches of 10 — tests editing, checkpoint save, and eval
TIMEOUT=600  # 10 minutes per algorithm (editing only, no eval)
PASS=0
FAIL=0
SKIP=0
ERRORS=""
START_ALL=$(date +%s)

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_and_check() {
    local label="$1"
    local expected_ckpt_prefix="$2"
    local ckpt_search_dir="$3"
    shift 3

    log "───────────────────────────────────────────"
    log "START [$((PASS + FAIL + 1))/11]: $label"
    log "  Config: $*" | head -c 200
    echo ""
    local t0=$(date +%s)

    # Run with timeout — unbuffered output for live streaming via sky logs
    PYTHONUNBUFFERED=1 timeout "$TIMEOUT" "$@" 2>&1
    local exit_code=$?
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))

    if [ "$exit_code" -eq 124 ]; then
        log "❌ $label: TIMEOUT after ${TIMEOUT}s"
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: TIMEOUT (${TIMEOUT}s)"
        return
    elif [ "$exit_code" -ne 0 ]; then
        log "❌ $label: FAILED (exit $exit_code, ${elapsed}s)"
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: exit code $exit_code"
        return
    fi

    log "  Ran in ${elapsed}s"

    # Check checkpoint path prefix
    if [ -n "$ckpt_search_dir" ] && [ -n "$expected_ckpt_prefix" ]; then
        ckpt_found=$(find "$ckpt_search_dir" -name "model_weights.pt" 2>/dev/null | head -1)
        if [ -n "$ckpt_found" ]; then
            if [[ "$ckpt_found" == *"$expected_ckpt_prefix"* ]]; then
                log "  ✓ Checkpoint path contains '$expected_ckpt_prefix'"
            else
                log "  ❌ Expected '$expected_ckpt_prefix' in: $ckpt_found"
                FAIL=$((FAIL+1))
                ERRORS="$ERRORS\n  $label: wrong checkpoint path"
                return
            fi
        else
            log "  ⚠ No checkpoint found (may be expected for this runner)"
        fi
    fi

    PASS=$((PASS+1))
    log "✓ $label PASSED (${elapsed}s)"
}

log "============================================"
log "  SMOKE TEST: All Algorithms"
log "  Results:     $RESULT_ROOT"
log "  Checkpoints: $CHECKPOINT_ROOT"
log "  Timeout:     ${TIMEOUT}s per algorithm"
log "  Batch size:  $EDITS edits"
log "  Dataset:     $DATASET_LIMIT records ($(($DATASET_LIMIT / $EDITS)) batches)"
log "  Algorithms:  8 vendor + 3 baselines = 11 total"
log "============================================"
log ""

# -----------------------------------------------------------------------
# GROUP 1: checkpoint_runner (AlphaEdit, MEMIT)
# -----------------------------------------------------------------------

run_and_check "AlphaEdit (checkpoint_runner)" "AlphaEdit" "$CHECKPOINT_ROOT/failure_curve" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name AlphaEdit --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 1

run_and_check "MEMIT (checkpoint_runner)" "MEMIT" "$CHECKPOINT_ROOT/failure_curve" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name MEMIT --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 1

# -----------------------------------------------------------------------
# GROUP 2: polykernel_seqreg_runner (MEMIT-Seq, REVIVE+X)
# -----------------------------------------------------------------------

run_and_check "MEMIT-Seq" "MEMIT-Seq" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+MEMIT" "MEMIT-Seq-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+AlphaEdit" "AlphaEdit-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg AlphaEdit --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+NSE" "NSE-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg NSE --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+RECT" "MEMIT_rect-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

# -----------------------------------------------------------------------
# GROUP 3: pathguard_runner
# -----------------------------------------------------------------------

run_and_check "PathGuard" "PathGuard" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/runners/pathguard_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 2 --cache_strategy all --cache_max none \
    --save_interval 1 --pathguard --pathguard_M 200 --pathguard_adaptive \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

# -----------------------------------------------------------------------
# GROUP 4: Baselines (EvoEdit, NSE, RECT via shell scripts)
# -----------------------------------------------------------------------

run_baseline() {
    local label="$1"
    local script="$2"
    local logfile=$(mktemp)

    log "───────────────────────────────────────────"
    log "START [$((PASS + FAIL + 1))/11]: $label"
    log "  Script: $script, TARGET_EDITS=$DATASET_LIMIT, NUM_EDITS=$EDITS, SEED=$SEED"
    local t0=$(date +%s)

    PYTHONUNBUFFERED=1 timeout "$TIMEOUT" bash -c "TARGET_EDITS=$DATASET_LIMIT NUM_EDITS=$EDITS bash $script $SEED" 2>&1
    local exit_code=$?
    local elapsed=$(( $(date +%s) - t0 ))

    if [ "$exit_code" -eq 0 ]; then
        PASS=$((PASS+1))
        log "✓ $label PASSED (${elapsed}s)"
    elif [ "$exit_code" -eq 124 ]; then
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: TIMEOUT (${TIMEOUT}s)"
        log "❌ $label: TIMEOUT after ${TIMEOUT}s"
    else
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: exit $exit_code"
        log "❌ $label: FAILED (exit $exit_code, ${elapsed}s)"
    fi
}

run_baseline "EvoEdit (baselines)" "scripts/run_evoedit_baseline.sh"
run_baseline "NSE (baselines)" "scripts/run_nse_baseline.sh"
run_baseline "RECT-Aligned (baselines)" "scripts/run_rect_aligned_paper_replication.sh"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))

log ""
log "============================================"
log "  RESULTS: $PASS passed, $FAIL failed (of 11 algorithms)"
log "  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
log "============================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failures:$ERRORS"
    exit 1
fi
log "  All smoke tests passed."
