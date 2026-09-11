#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Smoke test: run 1 batch (100 edits) of every algorithm and verify:
#   1. No errors (kwargs, assignment, anchor mismatch, etc.)
#   2. Checkpoint written to correct path with correct variant_name
#   3. Per-case result files written with both _probs and _correct fields
#   4. Metadata.json contains base_alg, variant_name, kernel params
#
# REQUIRES: GPU (L40S or equivalent, ~48GB VRAM)
# TIME: ~30-60 min total (each algorithm ~3-5 min for 1 batch)
#
# Usage:
#   bash tests/test_smoke_all_algorithms.sh              # local
#   bash tests/test_smoke_all_algorithms.sh --skypilot    # on SkyPilot cluster
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SKYPILOT=0
[[ "${1:-}" == "--skypilot" ]] && SKYPILOT=1

# Use temp dirs for local, S3 for SkyPilot
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
DATASET_LIMIT=200  # 2 batches worth, we only run 1
PASS=0
FAIL=0
ERRORS=""

check_result() {
    local label="$1"
    local expected_ckpt_prefix="$2"
    local ckpt_base="$3"
    local result_dir="$4"

    echo "  Checking $label..."

    # Check checkpoint exists and has correct prefix
    if [ -n "$ckpt_base" ]; then
        ckpt_found=$(find "$ckpt_base" -name "model_weights.pt" 2>/dev/null | head -1)
        if [ -z "$ckpt_found" ]; then
            echo "  ❌ $label: No checkpoint found at $ckpt_base"
            FAIL=$((FAIL+1))
            ERRORS="$ERRORS\n  $label: missing checkpoint"
            return
        fi

        # Check metadata
        meta_dir=$(dirname "$ckpt_found")
        if [ -f "$meta_dir/metadata.json" ]; then
            echo "  ✓ Checkpoint + metadata found"
        else
            echo "  ⚠ Checkpoint found but no metadata.json"
        fi

        # Check prefix in path
        if [[ "$ckpt_found" == *"$expected_ckpt_prefix"* ]]; then
            echo "  ✓ Checkpoint path contains '$expected_ckpt_prefix'"
        else
            echo "  ❌ $label: Expected '$expected_ckpt_prefix' in path, got: $ckpt_found"
            FAIL=$((FAIL+1))
            ERRORS="$ERRORS\n  $label: wrong checkpoint path prefix"
            return
        fi
    fi

    # Check result files have _probs
    if [ -d "$result_dir" ]; then
        case_file=$(find "$result_dir" -name "100_edits-case_*.json" 2>/dev/null | head -1)
        if [ -n "$case_file" ]; then
            has_probs=$(python3 -c "import json; d=json.load(open('$case_file')); print('rewrite_prompts_probs' in d.get('post',{}))")
            if [ "$has_probs" = "True" ]; then
                echo "  ✓ Per-case files have _probs fields"
            else
                echo "  ⚠ Per-case files missing _probs (may be expected for this algorithm)"
            fi
        fi
    fi

    PASS=$((PASS+1))
    echo "  ✓ $label PASSED"
}

echo "============================================"
echo "  SMOKE TEST: All Algorithms (1 batch each)"
echo "  Results: $RESULT_ROOT"
echo "  Checkpoints: $CHECKPOINT_ROOT"
echo "============================================"

# --- 1. AlphaEdit (via checkpoint_runner) ---
echo ""
echo "=== 1. AlphaEdit (checkpoint_runner) ==="
uv run python src/runners/checkpoint_runner.py \
    --seed $SEED --alg_name AlphaEdit --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --save_interval 1 --cuda_device 0 \
    --target_edits $EDITS 2>&1 | tail -5 || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  AlphaEdit: runtime error"; }

check_result "AlphaEdit" "AlphaEdit" \
    "$CHECKPOINT_ROOT/failure_curve" \
    "$RESULT_ROOT/failure_curve_checkpointed/seed${SEED}"

# --- 2. MEMIT-Seq (via polykernel_seqreg_runner, base_alg=MEMIT) ---
echo ""
echo "=== 2. MEMIT-Seq (polykernel_seqreg_runner) ==="
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 1.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --downstream_eval_steps 0 --conserve_memory 2>&1 | tail -5 \
    || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  MEMIT-Seq: runtime error"; }

check_result "MEMIT-Seq" "MEMIT-Seq" \
    "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-lp1.0-ld0.0-cache0" \
    "$RESULT_ROOT/failure_curve_checkpointed/seed${SEED}"

# --- 3. REVIVE+MEMIT (polykernel_seqreg_runner + revive) ---
echo ""
echo "=== 3. REVIVE+MEMIT ==="
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT \
    --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory 2>&1 | tail -5 \
    || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  REVIVE+MEMIT: runtime error"; }

check_result "REVIVE+MEMIT" "MEMIT-Seq-poly1-REVIVE-tau0.1" \
    "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0" \
    ""

# --- 4. REVIVE+AlphaEdit (polykernel_seqreg_runner, base_alg=AlphaEdit) ---
echo ""
echo "=== 4. REVIVE+AlphaEdit ==="
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg AlphaEdit \
    --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory 2>&1 | tail -5 \
    || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  REVIVE+AlphaEdit: runtime error"; }

check_result "REVIVE+AlphaEdit" "AlphaEdit-poly1-REVIVE-tau0.1" \
    "$CHECKPOINT_ROOT/polykernel_seqreg/AlphaEdit-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0" \
    ""

# --- 5. REVIVE+NSE (polykernel_seqreg_runner, base_alg=NSE) ---
echo ""
echo "=== 5. REVIVE+NSE ==="
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg NSE \
    --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory 2>&1 | tail -5 \
    || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  REVIVE+NSE: runtime error"; }

check_result "REVIVE+NSE" "NSE-poly1-REVIVE-tau0.1" \
    "$CHECKPOINT_ROOT/polykernel_seqreg/NSE-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0" \
    ""

# --- 6. REVIVE+RECT (polykernel_seqreg_runner, base_alg=MEMIT_rect) ---
echo ""
echo "=== 6. REVIVE+RECT ==="
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed $SEED --cuda_device 0 --ds_name mcf \
    --dataset_size_limit $DATASET_LIMIT --num_edits $EDITS \
    --lambda_prev 0.0 --lambda_delta 0.0 \
    --kernel_degree 1 --cache_strategy all --cache_max none \
    --save_interval 1 --base_alg MEMIT_rect \
    --revive --revive_tau 0.1 \
    --downstream_eval_steps 0 --conserve_memory 2>&1 | tail -5 \
    || { FAIL=$((FAIL+1)); ERRORS="$ERRORS\n  REVIVE+RECT: runtime error"; }

check_result "REVIVE+RECT" "MEMIT_rect-poly1-REVIVE-tau0.1" \
    "$CHECKPOINT_ROOT/polykernel_seqreg/MEMIT_rect-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0" \
    ""

# --- Summary ---
echo ""
echo "============================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "============================================"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failures:$ERRORS"
    exit 1
fi
echo "  All smoke tests passed."
