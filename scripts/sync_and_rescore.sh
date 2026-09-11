#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Phase 1: Sync S3 per-case files + rescore everything with prob-pref metrics
#
# Run from the repo root:
#   bash scripts/sync_and_rescore.sh
#
# This does NOT require a GPU. Estimated time: ~15-20 minutes total.
# ============================================================================

export AWS_REGION=us-east-2
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

S3_BUCKET="s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit"
S3_RESULTS="${S3_BUCKET}/results"
LOCAL_RESULTS="$PROJECT_DIR/results"

# ---------------------------------------------------------------------------
# Use aws s3 sync with high parallelism. It prints transfer progress natively.
# Crank up concurrent requests for many small files.
# ---------------------------------------------------------------------------
export AWS_MAX_CONCURRENT_REQUESTS=100
aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB

s3_sync() {
    local src="$1" dst="$2" label="${3:-Syncing}"
    echo "  ▶ ${label}: ${src}"
    mkdir -p "$dst"
    aws s3 sync "$src" "$dst" --region us-east-2 --only-show-errors --no-progress &
    local pid=$!

    # Simple spinner + file count while syncing
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        local count
        count=$(find "$dst" -type f -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        printf "\r    %s  %d files synced..." "${spin:i%10:1}" "$count"
        i=$((i+1))
        sleep 0.5
    done
    wait "$pid"
    local final
    final=$(find "$dst" -type f -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    printf "\r    ✓ %d files                          \n" "$final"
}

echo "=========================================="
echo "STEP 1: Sync per-case JSONs from S3"
echo "=========================================="
echo "These contain _probs fields needed for prob-pref rescoring."
echo "~50K files, ~150 MB. Using s5cmd for parallel download."
echo ""

# AlphaEdit matched_ordering per-case files
S3_MO="${S3_RESULTS}/matched_ordering/AlphaEdit"
LOCAL_MO="${LOCAL_RESULTS}/matched_ordering/AlphaEdit"

for ordering in fb_high_exposure fb_low_exposure key_clustered key_dispersed; do
    for seed in 42 2024 137; do
        src="${S3_MO}/${ordering}/seed${seed}/AlphaEdit/"
        dst="${LOCAL_MO}/${ordering}/seed${seed}/AlphaEdit/"
        s3_sync "$src" "$dst" "${ordering}/s${seed}"
    done
done

echo ""
echo "=========================================="
echo "STEP 2: Sync Qwen failure curve (new cross-model data)"
echo "=========================================="
echo "~2300 files, ~13 MB"

s3_sync "${S3_RESULTS}/failure_curve_qwen" "$LOCAL_RESULTS/failure_curve_qwen" "qwen_failure_curve"

echo ""
echo "=========================================="
echo "STEP 3: Sync GPT-J matched ordering orderings"
echo "=========================================="

s3_sync "${S3_RESULTS}/matched_ordering_gptj" "$LOCAL_RESULTS/matched_ordering_gptj" "gptj_orderings"

echo ""
echo "=========================================="
echo "STEP 4: Rescore failure curve per-case files (CPU, ~2 min)"
echo "=========================================="
echo "494K per-case files already local. Rescoring with prob-pref metrics."

uv run python scripts/eval_prob_preference.py \
    --mode rescore \
    --results-dir "$LOCAL_RESULTS/failure_curve_checkpointed" \
    --output-dir "$LOCAL_RESULTS/failure_curve_checkpointed_probpref"

echo ""
echo "=========================================="
echo "STEP 5: Rescore GPT-J failure curve (CPU, ~30 sec)"
echo "=========================================="
echo "Note: only efficacy _probs available for GPT-J (not paraphrase/neighborhood)"

uv run python scripts/eval_prob_preference.py \
    --mode rescore \
    --results-dir "$LOCAL_RESULTS/failure_curve_gptj" \
    --output-dir "$LOCAL_RESULTS/failure_curve_gptj_probpref"

echo ""
echo "=========================================="
echo "STEP 6: Rescore matched_ordering per-case files (CPU, ~1 min)"
echo "=========================================="
echo "Only the orderings with per-case files synced from S3."

for ordering in fb_high_exposure fb_low_exposure key_clustered key_dispersed; do
    for seed in 42 2024 137; do
        percase_dir="${LOCAL_MO}/${ordering}/seed${seed}/AlphaEdit"
        if [ -d "$percase_dir/run_000" ] && [ "$(ls "$percase_dir/run_000/"*.json 2>/dev/null | wc -l)" -gt 100 ]; then
            echo "  Rescoring ${ordering}/seed${seed}..."
            uv run python scripts/eval_prob_preference.py \
                --mode rescore \
                --results-dir "$percase_dir" \
                --output "$LOCAL_MO/${ordering}/seed${seed}/full_eval_seed${seed}_probpref.json"
        fi
    done
done

echo ""
echo "=========================================="
echo "STEP 7: Regenerate analysis (CPU, ~15 min)"
echo "=========================================="
echo "Re-run the full analysis pipeline with prob-pref as default."

cd "$PROJECT_DIR"
make -C analysis tables
make -C analysis all

echo ""
echo "=========================================="
echo "DONE. All submission-blocking numbers updated."
echo "=========================================="
echo ""
echo "Next steps (require GPU):"
echo "  1. Complete fb_high/low 10K evals for s2024/s137:"
echo "     bash scripts/reeval_matched_ordering_probpref.sh --p0 --execute"
echo "  2. Re-eval remaining matched_ordering with dual metrics:"
echo "     bash scripts/reeval_matched_ordering_probpref.sh --p1 --execute"
