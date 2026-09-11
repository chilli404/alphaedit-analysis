#!/usr/bin/env bash
set -euo pipefail

# Capacity Sign Reversal Replication (Seed 2024)
#
# Replicates the critical capacity result from the seed-42 projection sweep:
# at low r/d (binding regime), C₀ HURTS efficacy; at high r/d, C₀ HELPS.
#
# Runs exactly 4 cells on GPT-J-6B to confirm the sign reversal:
#   1. Threshold for r/d ≈ 20.7%, P-only (AlphaEdit)
#   2. Threshold for r/d ≈ 20.7%, P+C₀ (AlphaEdit-C0)
#   3. Threshold for r/d ≈ 56.1%, P-only (AlphaEdit)
#   4. Threshold for r/d ≈ 56.1%, P+C₀ (AlphaEdit-C0)
#
# IMPORTANT: The threshold values below are ESTIMATES based on the seed-42
# sweep. Before launching, verify with:
#   uv run python scripts/check_threshold_ranks.py
# Then update THRESHOLD_BINDING and THRESHOLD_PERMISSIVE if needed.
#
# Expected result (from seed-42):
#   Binding  (r/d≈20.7%): Δ(C₀|P) = -14.0 pp (C₀ hurts)
#   Permissive (r/d≈56.1%): Δ(C₀|P) = +5.8 pp (C₀ helps)
#   If the SIGN REVERSAL replicates, the capacity finding is confirmed.
#
# Usage:
#   bash scripts/run_capacity_replication_seed2024.sh          # All 4 cells sequentially
#   bash scripts/run_capacity_replication_seed2024.sh 2024     # Explicit seed
#   CELL=binding bash scripts/run_capacity_replication_seed2024.sh  # Only binding regime
#   CELL=permissive bash scripts/run_capacity_replication_seed2024.sh  # Only permissive regime
#
# On SkyPilot (4 parallel clusters):
#   bash sky/sky_launch.sh capacity_replication 2024

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-2024}"
CELL="${CELL:-all}"

# Threshold values producing target r/d ratios on GPT-J-6B (layer 6).
# Verified via SVD of normalized covariance (mom2/count):
#   threshold=0.0052 → r/d=20.7%, threshold=0.0105 → r/d=56.1%
THRESHOLD_BINDING="${THRESHOLD_BINDING:-0.0052}"     # r/d ≈ 20.7% (binding regime)
THRESHOLD_PERMISSIVE="${THRESHOLD_PERMISSIVE:-0.0105}" # r/d ≈ 56.1% (permissive regime)

echo "=== Capacity Sign Reversal Replication ==="
echo "  Seed: $SEED"
echo "  Cell: $CELL"
echo "  Binding threshold:    $THRESHOLD_BINDING (target r/d ≈ 20.7%)"
echo "  Permissive threshold: $THRESHOLD_PERMISSIVE (target r/d ≈ 56.1%)"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "  NOTE: Run 'uv run python scripts/check_threshold_ranks.py' to verify"
echo "        that these thresholds produce the correct r/d values."
echo ""

export EVAL_AT_CHECKPOINTS_ONLY=true
export TARGET_EDITS="${TARGET_EDITS:-10000}"

FAILED=0

run_binding() {
    echo "=== BINDING REGIME (r/d ≈ 20.7%, threshold=$THRESHOLD_BINDING) ==="
    echo ""

    echo "--- P-only (AlphaEdit) ---"
    bash "$SCRIPT_DIR/run_projection_sweep_gptj.sh" "$SEED" "$THRESHOLD_BINDING" "AlphaEdit" || FAILED=$((FAILED + 1))

    echo "--- P+C₀ (AlphaEdit-C0) ---"
    INJECT_C0=true bash "$SCRIPT_DIR/run_projection_sweep_gptj.sh" "$SEED" "$THRESHOLD_BINDING" "AlphaEdit-C0" || FAILED=$((FAILED + 1))
}

run_permissive() {
    echo "=== PERMISSIVE REGIME (r/d ≈ 56.1%, threshold=$THRESHOLD_PERMISSIVE) ==="
    echo ""

    echo "--- P-only (AlphaEdit) ---"
    bash "$SCRIPT_DIR/run_projection_sweep_gptj.sh" "$SEED" "$THRESHOLD_PERMISSIVE" "AlphaEdit" || FAILED=$((FAILED + 1))

    echo "--- P+C₀ (AlphaEdit-C0) ---"
    INJECT_C0=true bash "$SCRIPT_DIR/run_projection_sweep_gptj.sh" "$SEED" "$THRESHOLD_PERMISSIVE" "AlphaEdit-C0" || FAILED=$((FAILED + 1))
}

case "$CELL" in
    binding)
        run_binding
        ;;
    permissive)
        run_permissive
        ;;
    all)
        run_binding
        run_permissive
        ;;
    *)
        echo "ERROR: Unknown cell '$CELL'. Use: binding, permissive, or all"
        exit 1
        ;;
esac

echo ""
echo "=== Capacity replication complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $FAILED -gt 0 ]]; then
    echo "  WARNING: $FAILED cells failed"
    exit 1
fi
echo ""
echo "  Next steps:"
echo "    1. Compare P-only vs P+C₀ efficacy at each threshold"
echo "    2. Confirm sign(Δ(C₀|P)) reverses between binding and permissive"
echo "    3. Run: uv run python src/experiments/retained_energy_analysis.py --seed $SEED"
