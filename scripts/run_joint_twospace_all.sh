#!/usr/bin/env bash
set -euo pipefail

# Joint Memory-Space × Update-Space: Launch All 4 Cells
#
# Runs the full 2×2 interaction study on GPT-J-6B:
#   Cell 1: key_clustered  × low capacity  (threshold=0.01052, r/d≈20.7%)
#   Cell 2: key_clustered  × high capacity (threshold=0.0105, r/d≈56.1%)
#   Cell 3: key_dispersed  × low capacity  (threshold=0.01052, r/d≈20.7%)
#   Cell 4: key_dispersed  × high capacity (threshold=0.0105, r/d≈56.1%)
#
# All cells: AlphaEdit + C₀, GPT-J-6B, milestone evaluation.
#
# Usage:
#   bash scripts/run_joint_twospace_all.sh [SEED]
#   bash scripts/run_joint_twospace_all.sh 2024
#
# Override thresholds:
#   THRESHOLD_LOW=0.05 THRESHOLD_HIGH=0.005 bash scripts/run_joint_twospace_all.sh 2024

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-2024}"

# Threshold mapping: determined from GPT-J projector diagnostics
# Low capacity (binding regime): ~20.7% of dimensions retained
THRESHOLD_LOW="${THRESHOLD_LOW:-0.0052}"
# High capacity (permissive regime): ~56.1% of dimensions retained
THRESHOLD_HIGH="${THRESHOLD_HIGH:-0.0105}"

echo "=== Joint Memory × Update Space: Full 2×2 Study ==="
echo "  Seed:           $SEED"
echo "  Low capacity:   threshold=$THRESHOLD_LOW (r/d ≈ 20.7%)"
echo "  High capacity:  threshold=$THRESHOLD_HIGH (r/d ≈ 56.1%)"
echo "  Orderings:      key_clustered, key_dispersed"
echo "  Model:          GPT-J-6B + C₀"
echo ""
echo "  Expected: 4 cells × ~10-12h each"
echo "  On SkyPilot, use sky_launch.sh to run in parallel instead."
echo ""

FAILED=0

for ordering in key_clustered key_dispersed; do
    for threshold in "$THRESHOLD_LOW" "$THRESHOLD_HIGH"; do
        echo "──────────────────────────────────────────────────────────────────────"
        echo "  Cell: $ordering × threshold=$threshold"
        echo "──────────────────────────────────────────────────────────────────────"
        bash "$SCRIPT_DIR/run_joint_twospace_gptj.sh" "$SEED" "$ordering" "$threshold" || {
            echo "  FAILED: $ordering × $threshold"
            FAILED=$((FAILED + 1))
        }
        echo ""
    done
done

echo "=== All cells complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $FAILED -gt 0 ]]; then
    echo "  WARNING: $FAILED cells failed (checkpoints preserved for resume)"
fi
echo ""
echo "Analyze with:"
echo "  uv run python analysis/joint_twospace_analysis.py --seed $SEED"
