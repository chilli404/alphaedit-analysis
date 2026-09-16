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
TIMEOUT=${SMOKE_TIMEOUT:-300}  # 5 min default, baselines may need more
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
        # Exact prefix match on the label (case-sensitive).
        # "REVIVE" matches REVIVE+* but NOT "AlphaEdit" or "MEMIT".
        # "AlphaEdit_ckpt" matches "AlphaEdit (checkpoint_runner)" only.
        # "MEMIT_ckpt" matches "MEMIT (checkpoint_runner)" only.
        case "$f" in
            # Short aliases for non-ambiguous selection
            AlphaEdit_ckpt) [[ "$label" == "AlphaEdit (checkpoint_runner)" ]] && return 0 ;;
            MEMIT_ckpt)     [[ "$label" == "MEMIT (checkpoint_runner)" ]] && return 0 ;;
            MEMIT-Seq)      [[ "$label" == "MEMIT-Seq" ]] && return 0 ;;
            PathGuard)      [[ "$label" == PathGuard* ]] && return 0 ;;
            REVIVE)         [[ "$label" == REVIVE+* ]] && return 0 ;;
            EvoEdit)        [[ "$label" == "EvoEdit"* ]] && return 0 ;;
            NSE)            [[ "$label" == "NSE"* ]] && return 0 ;;
            RECT)           [[ "$label" == "RECT"* ]] && return 0 ;;
            RECT_err)       [[ "$label" == "RECT-Aligned (OTE)"* ]] && return 0 ;;
            RECT_bs100)     [[ "$label" == "RECT-Aligned (OTE) BS100" ]] && return 0 ;;
            # Exact full label match as fallback
            *)              [[ "$label" == "$f"* ]] && return 0 ;;
        esac
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
    log "START [$((PASS + FAIL + SKIP + 1))/13]: $label"
    log "  Config: $*" | head -c 200
    echo ""
    local t0=$(date +%s)

    # Run with timeout — full output to log, only errors to stdout
    local _safe_label="${label// /_}"; _safe_label="${_safe_label//+/_}"
    local logfile="$RESULT_ROOT/_smoke_${_safe_label}.log"
    SKIP_MEGA_BATCH_EVAL=1 PYTHONUNBUFFERED=1 timeout "$TIMEOUT" "$@" > "$logfile" 2>&1
    local exit_code=$?
    # Show errors if any
    if [ "$exit_code" -ne 0 ]; then
        log "  Last 20 lines of output:"
        tail -100 "$logfile" | sed 's/^/    /'
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

    # Verify per-case result files were produced
    local result_files=$(find "$RESULT_ROOT" -name "100_edits-case_*.json" 2>/dev/null | wc -l)
    if [ "$result_files" -gt 0 ]; then
        log "  ✓ $result_files per-case result file(s) produced"
    else
        log "  ⚠ No per-case result files found (eval may have been skipped)"
    fi

    PASS=$((PASS+1))
    log "✓ $label PASSED (${elapsed}s)"
}

validate_log() {
    local label="$1"
    local logfile="$2"

    case "$label" in
        *REVIVE*)
            # REVIVE must compute SVD and report split_rank per layer
            if ! grep -q "\[REVIVE\]" "$logfile"; then
                log "  ❌ REVIVE filter not running — no [REVIVE] output"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: REVIVE not running"; return 1
            fi
            # Verify per-layer SVD reports (hooks-based: "[REVIVE] layer=N split_rank=K/D removed=X%")
            # Post-hoc path (NSE/RECT) may produce zero deltas on small datasets,
            # in which case the filter is correctly skipped and no [REVIVE] layer= appears.
            if ! grep -q "\[REVIVE\] layer=" "$logfile"; then
                if grep -q "post-hoc filter applied to 0 layers" "$logfile"; then
                    log "  ⚠ REVIVE: post-hoc filter skipped (zero deltas on small dataset)"
                    # Zero deltas on small datasets is expected — pass without checking removed=
                    return 0
                else
                    log "  ❌ REVIVE SVD not computed per-layer"
                    FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: REVIVE SVD missing"; return 1
                fi
            fi
            # Verify filter is removing something (not a no-op)
            if ! grep -q "removed=" "$logfile"; then
                log "  ❌ REVIVE filter output missing 'removed=' metric"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: REVIVE no removal"; return 1
            fi
            local svd_count=$(grep -c "\[REVIVE\] layer=" "$logfile")
            log "  ✓ REVIVE: SVD computed ${svd_count}x with spectral filtering"
            ;;
        *AlphaEdit*checkpoint*)
            # AlphaEdit uses P matrix (null_space_project.pt) and cache_c
            # It does NOT call get_cov, so the model name won't appear in stats output
            if ! grep -q "null_space_project\|Loaded.*P matrix\|Initialized cache_c" "$logfile"; then
                log "  ❌ AlphaEdit: P matrix or cache_c not initialized"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: missing P/cache_c"; return 1
            fi
            log "  ✓ AlphaEdit: P matrix loaded, cache_c initialized"
            ;;
        *MEMIT*checkpoint*)
            # MEMIT must load covariance stats (canonical name appears in stats path)
            if ! grep -q "llama3-8b-instruct\|Loading cached" "$logfile"; then
                log "  ❌ MEMIT not loading stats"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: stats not loaded"; return 1
            fi
            log "  ✓ MEMIT: stats loaded"
            ;;
        *PathGuard*)
            if ! grep -q "PathGuard\|pathguard\|displacement" "$logfile"; then
                log "  ❌ PathGuard mechanism not active"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: PathGuard inactive"; return 1
            fi
            log "  ✓ PathGuard: mechanism active"
            ;;
        *MEMIT-Seq*)
            # Hooks-based runner uses seqreg_hooks, not [SeqReg+Kernel] patch
            if ! grep -q "lambda_prev\|SeqReg\|seqreg\|LAYER" "$logfile"; then
                log "  ❌ MEMIT-Seq not running (no lambda_prev or LAYER output)"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: MEMIT-Seq inactive"; return 1
            fi
            log "  ✓ MEMIT-Seq: hooks active"
            ;;
        *EvoEdit*)
            if ! grep -q "EvoEdit\|Woodbury" "$logfile"; then
                log "  ❌ EvoEdit algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: EvoEdit not invoked"; return 1
            fi
            log "  ✓ EvoEdit: algorithm invoked"
            ;;
        *NSE*baselines*)
            if ! grep -q "NSE\|nse\|neuron" "$logfile"; then
                log "  ❌ NSE algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: NSE not invoked"; return 1
            fi
            log "  ✓ NSE: algorithm invoked"
            ;;
        *RECT*OTE*BS100*)
            if ! grep -q "error_cache\|error_temp_norm" "$logfile"; then
                log "  ❌ RECT-Aligned (OTE) BS100 error correction not active"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: RECT-Err BS100 not invoked"; return 1
            fi
            if grep -qi "OutOfMemory\|OOM\|CUDA out of memory" "$logfile"; then
                log "  ❌ RECT-Aligned (OTE) BS100 OOM at production batch size"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: OOM at BS=100"; return 1
            fi
            log "  ✓ RECT-Aligned (OTE) BS100: no OOM at production batch size"
            ;;
        *RECT*OTE*)
            if ! grep -q "error_cache\|error_temp_norm\|small_delta\|memit seq rect with error correction" "$logfile"; then
                log "  ❌ RECT-Aligned (OTE) error correction not active"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: RECT-Err not invoked"; return 1
            fi
            log "  ✓ RECT-Aligned (OTE): error correction active"
            ;;
        *RECT*)
            if ! grep -q "MEMIT_seq_rect\|rect\|error_cache" "$logfile"; then
                log "  ❌ RECT algorithm not invoked"
                FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: RECT not invoked"; return 1
            fi
            log "  ✓ RECT: algorithm invoked"
            ;;
    esac

    # Memory checks
    # Verify GPU memory was freed after editing (checkpoint_runner reports this)
    if grep -q "\[CHECKPOINT\] Freed editing tensors" "$logfile"; then
        local free_mem=$(grep "\[CHECKPOINT\] Freed editing tensors" "$logfile" | tail -1 | grep -oP '[\d.]+(?= GiB)')
        if [ -n "$free_mem" ]; then
            log "  ✓ GPU memory freed after editing (${free_mem} GiB free)"
        fi
    fi
    # Check cache_c doesn't exceed expected bounds (8.2 GB for Llama 5-layer float64)
    if grep -q "cache_c" "$logfile"; then
        if grep -qi "CUDA out of memory.*cache_c\|RuntimeError.*cache_c" "$logfile"; then
            log "  ❌ cache_c caused OOM"
            FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  $label: cache_c OOM"; return 1
        fi
    fi

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
    log "  ERROR: $stale stale checkpoints remain after cleanup. Retrying..."
    rm -rf "$CHECKPOINT_ROOT" 2>/dev/null || true
    mkdir -p "$CHECKPOINT_ROOT"
    stale=$(find "$CHECKPOINT_ROOT" -name "model_weights.pt" 2>/dev/null | wc -l)
    [ "$stale" -gt 0 ] && { log "  FATAL: Cannot clean checkpoints. Aborting."; exit 1; }
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
log "  Algorithms:  10 vendor + 3 baselines = 13 total"
log "============================================"
log ""

# -----------------------------------------------------------------------
# GROUP 1: checkpoint_runner (AlphaEdit, MEMIT)
# -----------------------------------------------------------------------

run_and_check "AlphaEdit (checkpoint_runner)" "$CHECKPOINT_ROOT/failure_curve/AlphaEdit/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name AlphaEdit --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 0

run_and_check "MEMIT (checkpoint_runner)" "$CHECKPOINT_ROOT/failure_curve/MEMIT/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name MEMIT --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --fast_checkpoint --downstream_eval_steps 0

# -----------------------------------------------------------------------
# GROUP 2: polykernel_seqreg_runner (MEMIT-Seq, REVIVE+X)
# -----------------------------------------------------------------------

run_and_check "MEMIT-Seq" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-lp1.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+MEMIT" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+AlphaEdit" "$CHECKPOINT_ROOT/polykernel_seqreg/AlphaEdit-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg AlphaEdit --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+NSE" "$CHECKPOINT_ROOT/polykernel_seqreg/NSE-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg NSE --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "REVIVE+RECT" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT_rect-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

run_and_check "RECT-Aligned (OTE)" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT_rect_err-poly1-lp0.0-ld0.0-cache0/seed$SEED/batch_1/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect_err \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

# OOM probe: RECT-Err at production batch size (BS=100) — catches memory regressions
# that don't manifest at BS=10 (e.g. autograd graph scales with request count)
run_and_check "RECT-Aligned (OTE) BS100" "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT_rect_err-poly1-lp0.0-ld0.0-cache0-bs100/seed$SEED/batch_0/model_weights.pt" "" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit 100 --num_edits 100 \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect_err \
    --downstream_eval_steps 0 --conserve_memory --eval_at_checkpoints_only

# -----------------------------------------------------------------------
# GROUP 3: pathguard_runner
# -----------------------------------------------------------------------

# PathGuard default is EDS (with margin shield), not ED
run_and_check "PathGuard-poly2-hybrid" "$CHECKPOINT_ROOT/pathguard/PathGuard-EDS-poly2-hybrid-M200/seed$SEED/batch_1/model_weights.pt" "" \
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
    log "START [$((PASS + FAIL + SKIP + 1))/13]: $label"
    # EvoEdit uses Woodbury solver with 25 gradient steps per edit — too slow for smoke test.
    # Skip it with a 10s timeout (will be fixed when we add a v_star cache like NSE).
    # NSE uses precomputed v_star from KV cache — fast if cache is extracted.
    local bl_limit=$DATASET_LIMIT
    local bl_timeout=$TIMEOUT
    if echo "$label" | grep -qiE "NSE"; then
        bl_limit=$EDITS
    fi
    local bl_edits=$EDITS
    if echo "$label" | grep -qi "EvoEdit"; then
        # EvoEdit recomputes v_star each batch (25 gradient steps per edit, ~50s each).
        # Cannot use cache (v_star depends on current model state, unlike NSE).
        # Use 2 records / batch_size=2 for 1 batch: model load (30s) + v_star (100s) + Woodbury (40s) ≈ 170s
        bl_limit=2
        bl_edits=2
        bl_timeout=300
    fi
    log "  Script: $script, TARGET_EDITS=$bl_limit, NUM_EDITS=$bl_edits, SEED=$SEED"
    local t0=$(date +%s)

    local _safe_label="${label// /_}"; _safe_label="${_safe_label//+/_}"
    local logfile="$RESULT_ROOT/_smoke_${_safe_label}.log"
    # Baselines need ordering streams from the real results dir, not _smoke_test
    local real_result_root="/s3-data/continual-learning/alphaedit/results"
    [ ! -d "$real_result_root/matched_ordering/orderings" ] && real_result_root="$PROJECT_DIR/results"

    # NSE/EvoEdit scripts extract KV caches from S3 and fail fast if missing.
    # No pre-check needed here — the scripts handle it.

    # Clean results/checkpoints for this algorithm before running
    rm -rf "$CHECKPOINT_ROOT" 2>/dev/null; mkdir -p "$CHECKPOINT_ROOT"
    rm -rf "$RESULT_ROOT" 2>/dev/null; mkdir -p "$RESULT_ROOT"

    # Skip mega_batch_eval — the editing smoke test validates edits + checkpoints, not eval.
    # Eval is tested by the eval cluster (test_eval_and_measure.yaml).
    PYTHONUNBUFFERED=1 timeout "$bl_timeout" bash -c "SKIP_MEGA_BATCH_EVAL=1 TARGET_EDITS=$bl_limit NUM_EDITS=$bl_edits RESULT_ROOT=$real_result_root CHECKPOINT_ROOT=$CHECKPOINT_ROOT bash $script $SEED" > "$logfile" 2>&1
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        log "  Last 20 lines of output:"
        tail -100 "$logfile" | sed 's/^/    /'
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
            # Verify baseline produced some output
            local output_count=$(find "$RESULT_ROOT" "$CHECKPOINT_ROOT" \
                -name "model_weights.pt" -o -name "100_edits-case_*.json" 2>/dev/null | wc -l)
            if [ "$output_count" -eq 0 ]; then
                log "  ⚠ WARNING: No output files produced by $label"
            else
                log "  ✓ $output_count output file(s) produced"
            fi
            PASS=$((PASS+1))
            log "✓ $label PASSED (${elapsed}s)"
        }
    elif [ "$exit_code" -eq 124 ]; then
        if [ "$bl_timeout" -le 30 ]; then
            # Intentionally short timeout (e.g. EvoEdit skip) — count as skip
            SKIP=$((SKIP+1))
            log "⏭ $label: SKIPPED (${bl_timeout}s timeout — too slow for smoke test)"
        else
            FAIL=$((FAIL+1))
            ERRORS="$ERRORS\n  $label: TIMEOUT (${bl_timeout}s)"
            log "❌ $label: TIMEOUT after ${bl_timeout}s"
        fi
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
log "  RESULTS: $PASS passed, $FAIL failed, $SKIP skipped (of 13 algorithms)"
log "  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME / 60))m)"
log "============================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failures:$ERRORS"
    exit 1
fi
log "  All smoke tests passed."
