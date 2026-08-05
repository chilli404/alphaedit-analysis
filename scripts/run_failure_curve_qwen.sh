#!/usr/bin/env bash
set -euo pipefail

# Failure Curve: Qwen2.5-7B-Instruct
# Wrapper around the checkpointed failure curve runner with Qwen-specific defaults.
#
# Usage:
#   bash scripts/run_failure_curve_qwen.sh [SEED] [ALG] [TARGET_EDITS]
#   bash scripts/run_failure_curve_qwen.sh 42 AlphaEdit 10000
#   bash scripts/run_failure_curve_qwen.sh 42 both 5000
#
# Environment variables: same as run_failure_curve_checkpointed.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HPARAMS_FNAME="Qwen2.5-7B.json"

echo "=== Failure Curve (Qwen2.5-7B-Instruct) ==="
echo "  Model: $MODEL_NAME"

exec bash "$SCRIPT_DIR/run_failure_curve_checkpointed.sh" "${@}"
