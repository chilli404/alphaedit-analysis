#!/usr/bin/env bash
# sync_results_from_s3.sh — Download all experiment results from S3.
# Uses --size-only so existing local files are NEVER overwritten
# (only files missing locally or with a different size are fetched).
#
# Prerequisites:
#   aws configure set default.s3.max_concurrent_requests 64
#   aws configure set default.s3.multipart_threshold 64MB
#   aws configure set default.s3.multipart_chunksize 64MB
#   aws configure set default.s3.max_queue_size 10000

set -euo pipefail

S3_BASE="s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/results"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_BASE="$PROJECT_DIR/results"

mkdir -p "$LOCAL_BASE"

# ── Directories that need syncing (S3 has files missing locally) ──────────
# Listed in priority order: paper-critical first, then supplementary.

SYNC_DIRS=(
    # Large gaps — paper-critical
    matched_ordering           # +222K files (biggest gap)
    matched_ordering_gptj      # +5 files (new)
    polykernel_editor_gpt-j-6b # +12K files
    polykernel_editor          # +3K files
    failure_curve_qwen         # +2.3K files (new, cross-model)

    # Small gaps — paper-critical
    logit_damage               # +2 files (intervention results)
    logit_damage_memit_seq     # +3 files (MEMIT-Seq attenuation)
    same_fact_damage           # +3 files (same-fact intervention)
    key_vectors                # +6 files
    interference               # +2 files
    capability_probe           # +1 file
    polykernel_diagnostic      # +2 files

    # New directories not yet local
    evoedit                    # +63 files
    pathguard                  # +35 files
    metadata                   # +29 files
    retained_energy_analysis   # +1 file
    prompt_variant_keys        # +3 files

    # Already-complete but sync anyway to be safe (--size-only = fast no-op)
    failure_curve_checkpointed # local has more; sync catches any S3-only files
    failure_curve_gptj         # local has more
    failure_curve_zsre         # off by 1
    mve1_alphaedit_mcf         # exact match
    mve2_memit_mcf             # exact match
    mve3_alphaedit_zsre        # exact match
    comparison_ordered         # local has 1 extra
    matched_ordering_zsre      # exact match
    projection_sweep_gptj     # exact match
    projection_sweep_gptj_t0.0052
    projection_sweep_gptj_t0.0105
    joint_twospace             # exact match
    mechanism_analysis         # exact match
    zsre_ordering_experiment   # exact match
    figures                    # local has extras
    polykernel_seqreg          # local-only data
)

TOTAL=${#SYNC_DIRS[@]}
IDX=0

for dir in "${SYNC_DIRS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "[$IDX/$TOTAL] Syncing $dir ..."
    aws s3 sync \
        "$S3_BASE/$dir/" \
        "$LOCAL_BASE/$dir/" \
        --size-only \
        --only-show-errors
    echo "  done: $dir"
done

echo ""
echo "=== Sync complete ==="
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "=== LOCAL RESULTS SUMMARY ==="
for d in "$LOCAL_BASE"/*/; do
    dirname=$(basename "$d")
    n_json=$(find "$d" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  $dirname: $n_json JSON files, $sz"
done
