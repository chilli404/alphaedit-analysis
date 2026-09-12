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
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --skypilot) SKYPILOT=1 ;;
        *) FILTER="$FILTER $arg" ;;
    esac
done
FILTER="${FILTER# }"  # trim leading space

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

should_run() {
    local label="$1"
    if [ -z "$FILTER" ]; then return 0; fi
    for f in $FILTER; do
        [[ "$label" == *"$f"* ]] && return 0
    done
    SKIP=$((SKIP+1))
    return 1
}

run_and_check() {
    local label="$1"
    local expected_ckpt_path="$2"  # exact path to model_weights.pt (empty = skip check)
    local _unused="$3"  # kept for call-site compat
    shift 3

    should_run "$label" || return

    log "───────────────────────────────────────────"
    log "START [$((PASS + FAIL + SKIP + 1))/11]: $label"
    log "  Config: $*" | head -c 200
    echo ""
    local t0=$(date +%s)

    # Run with timeout — full output to log, only errors to stdout
    local logfile="$RESULT_ROOT/_smoke_${label// /_}.log"
    PYTHONUNBUFFERED=1 timeout "$TIMEOUT" "$@" > "$logfile" 2>&1
    local exit_code=$?
    # Show errors if any
    if [ "$exit_code" -ne 0 ]; then
        log "  Last 20 lines of output:"
        tail -20 "$logfile" | sed 's/^/    /'
    fi
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

    log "  Ran in ${elapsed}s (exit=$exit_code)"

    # Also check log for errors even if exit code is 0
    if grep -qiE "error:|Traceback|KeyError|TypeError|unrecognized arguments|OutOfMemory|CUDA out of memory" "$logfile" 2>/dev/null; then
        log "  ❌ $label: errors found in output despite exit code $exit_code"
        log "  Error lines:"
        grep -iE "error:|Traceback|KeyError|TypeError|unrecognized arguments|OutOfMemory|CUDA out of memory" "$logfile" | tail -5 | sed 's/^/    /'
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: errors in output"
        return
    fi

    # Check exact checkpoint path
    if [ -n "$expected_ckpt_path" ]; then
        if [ -f "$expected_ckpt_path" ]; then
            log "  ✓ Checkpoint exists: $expected_ckpt_path"
            # Also verify metadata.json exists alongside it
            local meta_dir=$(dirname "$expected_ckpt_path")
            if [ -f "$meta_dir/metadata.json" ]; then
                log "  ✓ metadata.json present"
            else
                log "  ⚠ metadata.json missing at $meta_dir"
            fi
        else
            log "  ❌ Checkpoint NOT found at expected path:"
            log "     $expected_ckpt_path"
            # Show what checkpoints DO exist for debugging
            local search_base=$(echo "$expected_ckpt_path" | sed 's|/batch_[0-9]*/.*||')
            if [ -d "$search_base" ]; then
                log "  Existing checkpoints:"
                find "$search_base" -name "model_weights.pt" 2>/dev/null | sed 's/^/     /'
            fi
            FAIL=$((FAIL+1))
            ERRORS="$ERRORS\n  $label: checkpoint not at expected path"
            return
        fi
    fi

    # Check for OOM in log
    if grep -qi "CUDA out of memory\|OutOfMemoryError\|OOM" "$logfile" 2>/dev/null; then
        log "  ❌ $label: GPU OOM detected"
        grep -i "CUDA out of memory\|OutOfMemoryError" "$logfile" | tail -2 | sed 's/^/    /'
        FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: GPU OOM"; return
    fi

    # Algorithm-specific log validation
    validate_log "$label" "$logfile" || return

    PASS=$((PASS+1))
    log "✓ $label PASSED (${elapsed}s)"
}

validate_log() {
    local label="$1"
    local logfile="$2"

    case "$label" in
        *REVIVE*)
            # REVIVE must compute SVD and report split_rank per layer
            if ! grep -q "\[REVIVE\] layer=" "$logfile"; then
                log "  ❌ REVIVE SVD not computed — filter not running"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: REVIVE SVD missing"; return 1
            fi
            if ! grep -q "full_matrices=True" "$logfile"; then
                log "  ❌ REVIVE not using full_matrices=True"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: wrong SVD mode"; return 1
            fi
            if ! grep -q "svd_device=cuda" "$logfile"; then
                log "  ❌ REVIVE SVD not on GPU (should be cuda)"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: SVD on CPU"; return 1
            fi
            if ! grep -q "Dynamic SVD" "$logfile"; then
                log "  ❌ REVIVE not using dynamic SVD (current weight)"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: static SVD"; return 1
            fi
            local svd_count=$(grep -c "\[REVIVE\] layer=" "$logfile")
            log "  ✓ REVIVE: SVD computed ${svd_count}x, full_matrices=True, device=cuda, dynamic"
            ;;
        *AlphaEdit*checkpoint*)
            # AlphaEdit checkpoint_runner must resolve model name correctly
            if ! grep -q "llama3-8b-instruct\|Llama3-8B" "$logfile"; then
                log "  ❌ Model name not canonical"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: bad model name"; return 1
            fi
            # Must save cache_c
            if ! grep -q "cache_c" "$logfile"; then
                log "  ❌ No cache_c in checkpoint"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: missing cache_c"; return 1
            fi
            log "  ✓ AlphaEdit: canonical model name, cache_c saved"
            ;;
        *MEMIT*checkpoint*)
            # MEMIT must load covariance stats from correct path
            if ! grep -q "Loading cached data/stats/llama3-8b-instruct" "$logfile"; then
                log "  ❌ MEMIT not loading stats from canonical path"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: wrong stats path"; return 1
            fi
            log "  ✓ MEMIT: stats loaded from canonical path"
            ;;
        *PathGuard*)
            if ! grep -q "PathGuard\|pathguard" "$logfile"; then
                log "  ❌ PathGuard mechanism not active"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: PathGuard inactive"; return 1
            fi
            log "  ✓ PathGuard: mechanism active"
            ;;
        *MEMIT-Seq*)
            if ! grep -q "\[SeqReg+Kernel\]" "$logfile"; then
                log "  ❌ SeqReg+Kernel patch not applied"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: kernel patch missing"; return 1
            fi
            log "  ✓ MEMIT-Seq: kernel patch applied"
            ;;
        *EvoEdit*)
            if ! grep -q "EvoEdit" "$logfile"; then
                log "  ❌ EvoEdit algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: EvoEdit not invoked"; return 1
            fi
            log "  ✓ EvoEdit: algorithm invoked"
            ;;
        *NSE*baselines*)
            if ! grep -q "NSE\|nse" "$logfile"; then
                log "  ❌ NSE algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: NSE not invoked"; return 1
            fi
            log "  ✓ NSE: algorithm invoked"
            ;;
        *RECT*)
            if ! grep -q "MEMIT_seq_rect\|rect" "$logfile"; then
                log "  ❌ RECT algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: RECT not invoked"; return 1
            fi
            log "  ✓ RECT: algorithm invoked"
            ;;
    esac

    # Universal checks — all algorithms must show these
    if ! grep -q "LAYER [4-8]" "$logfile"; then
        log "  ❌ No LAYER editing output"
        FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: no editing"; return 1
    fi
    if ! grep -q "Deltas successfully computed\|New weights successfully inserted" "$logfile"; then
        log "  ❌ Editing did not complete"
        FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: incomplete"; return 1
    fi
    local batch_count=$(grep -c "_edit==" "$logfile" 2>/dev/null || echo 0)
    log "  ✓ $batch_count batch(es) completed with layer editing"

    return 0
}

# Clean up previous smoke test data so runners don't resume from stale checkpoints
log "Cleaning previous smoke test data..."
# S3 FUSE doesn't support rm -rf well on nested dirs; delete contents explicitly
find "$RESULT_ROOT" -type f -delete 2>/dev/null || true
find "$RESULT_ROOT" -type d -empty -delete 2>/dev/null || true
find "$CHECKPOINT_ROOT" -type f -delete 2>/dev/null || true
find "$CHECKPOINT_ROOT" -type d -empty -delete 2>/dev/null || true
mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT"
# Verify cleanup worked
stale=$(find "$CHECKPOINT_ROOT" -name "model_weights.pt" 2>/dev/null | wc -l)
if [ "$stale" -gt 0 ]; then
    log "  WARNING: $stale stale checkpoints remain — runners may resume instead of starting fresh"
else
    log "  ✓ Cleaned: $CHECKPOINT_ROOT"
fi
log "  ✓ Cleaned: $RESULT_ROOT"
log ""

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

run_and_check "AlphaEdit (checkpoint_runner)" "$CHECKPOINT_ROOT/failure_curve/AlphaEdit/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name AlphaEdit --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 0

run_and_check "MEMIT (checkpoint_runner)" "$CHECKPOINT_ROOT/failure_curve/MEMIT/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name MEMIT --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 0

# -----------------------------------------------------------------------
# GROUP 2: polykernel_seqreg_runner (MEMIT-Seq, REVIVE+X)
# -----------------------------------------------------------------------

run_and_check "MEMIT-Seq" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-lp1.0-ld0.0-cache0/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+MEMIT" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+AlphaEdit" "$CHECKPOINT_ROOT/polykernel_seqreg/AlphaEdit-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg AlphaEdit --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+NSE" "$CHECKPOINT_ROOT/polykernel_seqreg/NSE-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg NSE --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+RECT" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT_rect-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_0/model_weights.pt" "" \
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

run_and_check "PathGuard-poly2-hybrid" "$CHECKPOINT_ROOT/pathguard/PathGuard-ED-poly2-hybrid-M200-e0.1/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/runners/pathguard_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --cache_strategy all --cache_max none \
    --save_interval 1 --pathguard --pathguard_M 200 --pathguard_adaptive \
    --pathguard_kernel_degree 2 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

# -----------------------------------------------------------------------
# GROUP 4: Baselines (EvoEdit, NSE, RECT via shell scripts)
# -----------------------------------------------------------------------

run_baseline() {
    local label="$1"
    local script="$2"

    should_run "$label" || return

    local logfile=$(mktemp)
    log "───────────────────────────────────────────"
    log "START [$((PASS + FAIL + SKIP + 1))/11]: $label"
    log "  Script: $script, TARGET_EDITS=$DATASET_LIMIT, NUM_EDITS=$EDITS, SEED=$SEED"
    local t0=$(date +%s)

    local logfile="$RESULT_ROOT/_smoke_${label// /_}.log"
    # Baselines need ordering streams from the real results dir, not _smoke_test
    local real_result_root="/s3-data/continual-learning/alphaedit/results"
    [ ! -d "$real_result_root/matched_ordering/orderings" ] && real_result_root="$PROJECT_DIR/results"
    PYTHONUNBUFFERED=1 timeout "$TIMEOUT" bash -c "TARGET_EDITS=$DATASET_LIMIT NUM_EDITS=$EDITS RESULT_ROOT=$real_result_root CHECKPOINT_ROOT=$CHECKPOINT_ROOT bash $script $SEED" > "$logfile" 2>&1
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        log "  Last 20 lines of output:"
        tail -20 "$logfile" | sed 's/^/    /'
    fi
    local elapsed=$(( $(date +%s) - t0 ))

    # Check log for errors even if exit code is 0
    if grep -qiE "^ERROR:|Traceback|KeyError|TypeError|unrecognized arguments" "$logfile" 2>/dev/null; then
        log "❌ $label: errors found in output"
        grep -iE "^ERROR:|Traceback|KeyError|TypeError|unrecognized arguments" "$logfile" | tail -5 | sed 's/^/    /'
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: errors in output"
    elif [ "$exit_code" -eq 0 ]; then
        validate_log "$label" "$logfile" && {
            PASS=$((PASS+1))
            log "✓ $label PASSED (${elapsed}s)"
        }
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
log "  RESULTS: $PASS passed, $FAIL failed, $SKIP skipped (of 11 algorithms)"
log "  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
log "============================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failures:$ERRORS"
    exit 1
fi
log "  All smoke tests passed."
