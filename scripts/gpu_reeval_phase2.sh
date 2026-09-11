#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Phase 2: GPU re-evaluation of matched_ordering checkpoints with dual metrics
#
# REQUIRES: GPU (L40S/A100/A10G), SkyPilot configured, S3 access
#
# This script provides the commands — run them manually or via SkyPilot.
#
# Total: ~72 GPU evals
#   P0 (submission-blocking): 15 evals, ~2h at 8 parallel GPUs
#   P1 (cross-algorithm):     19 evals, ~2.5h at 8 parallel GPUs
#   P2 (appendix):            38 evals, ~4.5h at 8 parallel GPUs
# ============================================================================

export AWS_REGION=us-east-2
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "GPU Phase 2: Re-evaluate with dual metrics"
echo "============================================"
echo ""
echo "Option A: Use the existing SkyPilot launcher (recommended):"
echo ""
echo "  # P0 only (submission-blocking, ~2h with 8 GPUs):"
echo "  bash scripts/reeval_matched_ordering_probpref.sh --p0 --execute"
echo ""
echo "  # P0 + P1 (required, ~4.5h with 8 GPUs):"
echo "  bash scripts/reeval_matched_ordering_probpref.sh --p0 --p1 --execute"
echo ""
echo "  # Everything (~9h with 8 GPUs):"
echo "  bash scripts/reeval_matched_ordering_probpref.sh --execute"
echo ""
echo "Option B: Complete the 4 missing 10K evals (fb_high/low × s2024/s137):"
echo ""
echo "  These have batch_99 checkpoints on S3 but eval stopped at 9K."
echo "  The reeval script above handles these as part of P0."
echo "  Alternatively, run manually on a local GPU:"
echo ""

for ordering in fb_high_exposure fb_low_exposure; do
    for seed in 2024 137; do
        echo "  uv run python scripts/eval_matched_ordering.py \\"
        echo "    --seed $seed --alg_name AlphaEdit --ordering $ordering \\"
        echo "    --checkpoints 99 --num_edits 100"
        echo ""
    done
done

echo ""
echo "Option C: Evaluate remaining orderings that lack per-case files:"
echo ""
echo "  # These orderings had no per-case files on S3, so they need GPU re-eval:"
echo "  # fb_random1, fb_random2, sched_*, suffix_*, greedy_minmax, random, etc."
echo "  # All handled by the --p2 flag in the reeval script."
echo ""
echo "  bash scripts/reeval_matched_ordering_probpref.sh --p2 --execute"
echo ""
echo "============================================"
echo "After GPU evals complete, run:"
echo "  make -C analysis tables"
echo "  make -C analysis all"
echo "============================================"
