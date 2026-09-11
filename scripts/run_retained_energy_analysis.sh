#!/usr/bin/env bash
set -euo pipefail

# Post-hoc retained-energy analysis for projection capacity sweep.
#
# For each threshold, computes:
#   ||P @ K_edit||_F² / ||K_edit||_F²    (retained edit energy)
#   ||ΔW @ P||_F² / ||ΔW||_F²           (retained update energy)
#
# No GPU needed — runs on CPU using stored checkpoints and P matrices.
#
# Usage:
#   bash scripts/run_retained_energy_analysis.sh [SEED] [BATCH]
#   bash scripts/run_retained_energy_analysis.sh 42 49
#   bash scripts/run_retained_energy_analysis.sh 42 --all

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
BATCH="${2:-49}"

# The 7 thresholds used in the projection capacity sweep
THRESHOLDS="${THRESHOLDS:-0.005 0.01 0.02 0.05 0.1 0.2 0.5}"

echo "=== Retained Energy Analysis ==="
echo "  Seed: $SEED"
echo "  Batch: $BATCH"
echo "  Thresholds: $THRESHOLDS"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

BATCH_ARG="--batch $BATCH"
if [[ "$BATCH" == "--all" ]] || [[ "$BATCH" == "all" ]]; then
    BATCH_ARG="--all_batches"
fi

uv run python src/experiments/retained_energy_analysis.py \
    --seed "$SEED" \
    $BATCH_ARG \
    --thresholds $THRESHOLDS \
    --model_name "EleutherAI/gpt-j-6b"

echo ""
echo "=== Retained Energy Analysis complete ==="
echo "  Results: results/retained_energy_analysis/seed${SEED}/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
