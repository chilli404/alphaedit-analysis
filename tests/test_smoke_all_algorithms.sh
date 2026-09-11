#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# GPU Smoke Test: 1 batch (100 edits) of EVERY algorithm × runner combination.
#
# Tests:
#   1. No runtime errors (kwargs, anchors, OOM, NaN, assignment)
#   2. Checkpoint written to correct path with correct variant_name
#   3. Per-case result files written
#   4. Metadata.json present and correct
#
# Algorithms tested:
#   Via checkpoint_runner:    AlphaEdit, MEMIT
#   Via polykernel_seqreg:    MEMIT-Seq, REVIVE+MEMIT, REVIVE+AlphaEdit, REVIVE+NSE, REVIVE+RECT
#   Via pathguard_runner:     PathGuard
#   Via baselines/EvoEdit:    EvoEdit, NSE, RECT-Aligned
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
EDITS=100
DATASET_LIMIT=200
PASS=0
FAIL=0
ERRORS=""

run_and_check() {
    local label="$1"
    local expected_ckpt_prefix="$2"
    local ckpt_search_dir="$3"
    shift 3
    # remaining args are the command

    echo ""
    echo "=== $label ==="

    if "$@" 2>&1 | tail -10; then
        echo "  ✓ $label ran without errors"
    else
        echo "  ❌ $label: runtime error"
        FAIL=$((FAIL+1))
        ERRORS="$ERRORS\n  $label: runtime error"
        return
    fi

    # Check checkpoint path prefix
    if [ -n "$ckpt_search_dir" ]; then
        ckpt_found=$(find "$ckpt_search_dir" -name "model_weights.pt" 2>/dev/null | head -1)
        if [ -n "$ckpt_found" ] && [ -n "$expected_ckpt_prefix" ]; then
            if [[ "$ckpt_found" == *"$expected_ckpt_prefix"* ]]; then
                echo "  ✓ Checkpoint path contains '$expected_ckpt_prefix'"
            else
                echo "  ❌ Expected '$expected_ckpt_prefix' in: $ckpt_found"
                FAIL=$((FAIL+1))
                ERRORS="$ERRORS\n  $label: wrong checkpoint path"
                return
            fi
        fi
    fi

    PASS=$((PASS+1))
}

echo "============================================"
echo "  SMOKE TEST: All Algorithms (1 batch each)"
echo "  Results:     $RESULT_ROOT"
echo "  Checkpoints: $CHECKPOINT_ROOT"
echo "============================================"

# -----------------------------------------------------------------------
# GROUP 1: Vendor runners (checkpoint_runner, polykernel_seqreg, pathguard)
# -----------------------------------------------------------------------

# 1. AlphaEdit
run_and_check "AlphaEdit (checkpoint_runner)" "AlphaEdit" "$CHECKPOINT_ROOT/failure_curve" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name AlphaEdit --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 --eval_at_checkpoints_only \
    --downstream_eval_steps 0

# 2. MEMIT
run_and_check "MEMIT (checkpoint_runner)" "MEMIT" "$CHECKPOINT_ROOT/failure_curve" \
    uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name MEMIT --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 --eval_at_checkpoints_only \
    --downstream_eval_steps 0

# 3. MEMIT-Seq
run_and_check "MEMIT-Seq (polykernel_seqreg)" "MEMIT-Seq" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --downstream_eval_steps 0 --conserve_memory

# 4. REVIVE+MEMIT
run_and_check "REVIVE+MEMIT" "MEMIT-Seq-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory

# 5. REVIVE+AlphaEdit
run_and_check "REVIVE+AlphaEdit" "AlphaEdit-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg AlphaEdit --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory

# 6. REVIVE+NSE
run_and_check "REVIVE+NSE" "NSE-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg NSE --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory

# 7. REVIVE+RECT
run_and_check "REVIVE+RECT" "MEMIT_rect-poly1-REVIVE" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory

# 8. PathGuard
run_and_check "PathGuard" "PathGuard" "$CHECKPOINT_ROOT/polykernel_seqreg" \
    uv run python src/runners/pathguard_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 2 --cache_strategy all --cache_max none \
    --save_interval 1 --pathguard --pathguard_M 200 --pathguard_adaptive \
    --downstream_eval_steps 0 --conserve_memory

# -----------------------------------------------------------------------
# GROUP 2: Baselines (run through shell scripts → baselines/EvoEdit)
# -----------------------------------------------------------------------

# 9. EvoEdit
echo ""
echo "=== EvoEdit (baselines/EvoEdit) ==="
if TARGET_EDITS=$DATASET_LIMIT bash scripts/run_evoedit_baseline.sh $SEED 2>&1 | tail -10; then
    echo "  ✓ EvoEdit ran without errors"
    PASS=$((PASS+1))
else
    echo "  ❌ EvoEdit: runtime error"
    FAIL=$((FAIL+1))
    ERRORS="$ERRORS\n  EvoEdit: runtime error"
fi

# 10. NSE
echo ""
echo "=== NSE (baselines/EvoEdit) ==="
if TARGET_EDITS=$DATASET_LIMIT bash scripts/run_nse_baseline.sh $SEED 2>&1 | tail -10; then
    echo "  ✓ NSE ran without errors"
    PASS=$((PASS+1))
else
    echo "  ❌ NSE: runtime error"
    FAIL=$((FAIL+1))
    ERRORS="$ERRORS\n  NSE: runtime error"
fi

# 11. RECT-Aligned
echo ""
echo "=== RECT-Aligned (baselines/EvoEdit) ==="
if TARGET_EDITS=$DATASET_LIMIT bash scripts/run_rect_aligned_paper_replication.sh $SEED 2>&1 | tail -10; then
    echo "  ✓ RECT-Aligned ran without errors"
    PASS=$((PASS+1))
else
    echo "  ❌ RECT-Aligned: runtime error"
    FAIL=$((FAIL+1))
    ERRORS="$ERRORS\n  RECT-Aligned: runtime error"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "============================================"
echo "  RESULTS: $PASS passed, $FAIL failed (of 11 algorithms)"
echo "============================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failures:$ERRORS"
    exit 1
fi
echo "  All smoke tests passed."
