#!/usr/bin/env bash
# run_poly2_diagnostic.sh — P7: Polykernel (degree-2) diagnostic comparison
#
# Runs polykernel editor with degree-2 kernel as a diagnostic comparison
# against the factorial ablation cells. Measures locality, capability,
# and weight damage alongside behavioral gains.
#
# Usage:
#   bash scripts/run_poly2_diagnostic.sh SEED [TARGET_EDITS]
#
# Examples:
#   bash scripts/run_poly2_diagnostic.sh 42           # 3K edits default
#   bash scripts/run_poly2_diagnostic.sh 42 5000      # 5K edits
#
# Environment variables:
#   FAST_CHECKPOINT=true           (recommended)
#   ORDER_ID=0                     (default, matches factorial cell D)
#   KERNEL_DEGREE=2                (default)

set -euo pipefail

SEED="${1:?Usage: $0 SEED [TARGET_EDITS]}"
TARGET_EDITS="${2:-${TARGET_EDITS:-3000}}"
ORDER_ID="${ORDER_ID:-0}"
KERNEL_DEGREE="${KERNEL_DEGREE:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "P7: Poly2 Diagnostic"
echo "  Seed:          $SEED"
echo "  Target edits:  $TARGET_EDITS"
echo "  Order ID:      $ORDER_ID"
echo "  Kernel degree: $KERNEL_DEGREE"
echo "========================================================================"

# Run polykernel editor with capability probes
POLYKERNEL_ARGS=(
    --seed "$SEED"
    --ds_name mcf
    --dataset_size_limit "$TARGET_EDITS"
    --num_edits 100
    --kernel_degree "$KERNEL_DEGREE"
    --order_id "$ORDER_ID"
    --capability_probe_interval 10
)

if [ "${FAST_CHECKPOINT:-}" = "true" ]; then
    POLYKERNEL_ARGS+=(--fast_checkpoint)
fi

uv run python "$PROJECT_ROOT/src/polykernel/polykernel_editor_runner.py" \
    "${POLYKERNEL_ARGS[@]}"

echo "========================================================================"
echo "Poly2 diagnostic complete."
echo "========================================================================"
