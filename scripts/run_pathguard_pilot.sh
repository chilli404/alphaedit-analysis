#!/usr/bin/env bash
set -euo pipefail

# PathGuard Pilot: run the full ablation matrix for the decisive pilot.
#
# Matrix: 6 variants × 3 orderings × N seeds
# Tune on seed 42 / fb_random0 first, then freeze and run all seeds.
#
# Usage:
#   bash scripts/run_pathguard_pilot.sh [SEED...]
#   bash scripts/run_pathguard_pilot.sh 42           # Single seed (tuning)
#   bash scripts/run_pathguard_pilot.sh 42 137 2024  # Full pilot

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEEDS="${@:-42}"

ORDERINGS="fb_high_exposure fb_random0 fb_low_exposure"
VARIANTS="PathGuard-E PathGuard-ED PathGuard-EDS PathGuard-ED-random"

TOTAL=0
for SEED in $SEEDS; do
    for VARIANT in $VARIANTS; do
        for ORDERING in $ORDERINGS; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard Pilot"
echo "  Seeds:      $SEEDS"
echo "  Variants:   $VARIANTS"
echo "  Orderings:  $ORDERINGS"
echo "  Total runs: $TOTAL"
echo "═══════════════════════════════════════════════════════════════"
echo ""

COMPLETED=0
for SEED in $SEEDS; do
    for VARIANT in $VARIANTS; do
        for ORDERING in $ORDERINGS; do
            COMPLETED=$((COMPLETED + 1))
            echo "[$COMPLETED/$TOTAL] $VARIANT seed $SEED ordering $ORDERING"
            bash "$SCRIPT_DIR/run_pathguard.sh" "$SEED" "$VARIANT" "$ORDERING" || {
                echo "  FAILED: $VARIANT seed $SEED ordering $ORDERING"
                continue
            }
            echo ""
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard Pilot Complete: $COMPLETED/$TOTAL runs"
echo "═══════════════════════════════════════════════════════════════"
