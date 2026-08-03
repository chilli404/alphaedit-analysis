#!/usr/bin/env bash
set -euo pipefail

# Run interference-aware scheduling experiment.
#
# Thin wrapper around run_matched_ordering.sh that:
#   - Sets TARGET_EDITS=10000 (scheduling uses full 10K stream)
#   - Validates the ordering file exists before launching
#   - Delegates all editing logic to the existing pipeline
#
# Usage:
#   bash scripts/run_scheduling_experiment.sh SEED ALG METHOD
#   bash scripts/run_scheduling_experiment.sh 42 AlphaEdit greedy_minmax
#   bash scripts/run_scheduling_experiment.sh 42 AlphaEdit random
#   bash scripts/run_scheduling_experiment.sh 2024 AlphaEdit greedy_minmax
#   bash scripts/run_scheduling_experiment.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 greedy_minmax
#   bash scripts/run_scheduling_experiment.sh 42 AlphaEdit cluster_topo
#
# SkyPilot:
#   ALG_NAME=AlphaEdit ORDERING=greedy_minmax TARGET_EDITS=10000 \
#     bash sky/sky_launch.sh matched_ordering 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SEED="${1:-42}"
ALG="${2:-AlphaEdit}"
METHOD="${3:-greedy_minmax}"

# Scheduling experiments use the full 10K stream
export TARGET_EDITS="${TARGET_EDITS:-10000}"

# Resolve ordering file path
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
ORDERING_FILE="$RESULT_ROOT/matched_ordering/orderings/${METHOD}_seed${SEED}.json"

# Validate ordering file exists
if [ ! -f "$ORDERING_FILE" ]; then
    echo "ERROR: Ordering file not found: $ORDERING_FILE"
    echo ""
    echo "Generate it first with:"
    echo "  uv run python scheduling/generate_scheduling_orderings.py --seed $SEED --methods $METHOD"
    echo ""
    echo "Then validate geometry:"
    echo "  uv run python scheduling/validate_ordering.py --seed $SEED"
    exit 1
fi

N_RECORDS=$(uv run python -c "import json; print(len(json.load(open('$ORDERING_FILE'))))")
echo "=== Scheduling Experiment ==="
echo "  Seed:     $SEED"
echo "  Alg:      $ALG"
echo "  Method:   $METHOD"
echo "  Records:  $N_RECORDS"
echo "  Target:   $TARGET_EDITS edits"
echo "  Ordering: $ORDERING_FILE"
echo "==========================="

# Delegate to existing matched ordering pipeline
exec bash "$SCRIPT_DIR/run_matched_ordering.sh" "$SEED" "$ALG" "$METHOD"
