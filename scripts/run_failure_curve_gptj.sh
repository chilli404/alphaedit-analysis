#!/usr/bin/env bash
set -euo pipefail

# Failure Curve: GPT-J-6B
# Wrapper around the checkpointed failure curve runner with GPT-J-specific defaults.
#
# Usage:
#   bash scripts/run_failure_curve_gptj.sh [SEED] [ALG] [TARGET_EDITS]
#   bash scripts/run_failure_curve_gptj.sh 42 AlphaEdit 5000
#   bash scripts/run_failure_curve_gptj.sh 42 both 5000
#
# Environment variables: same as run_failure_curve_checkpointed.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export MODEL_NAME="EleutherAI/gpt-j-6b"
export HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"

echo "=== Failure Curve (GPT-J-6B) ==="
echo "  Model: $MODEL_NAME"

exec bash "$SCRIPT_DIR/run_failure_curve_checkpointed.sh" "${@}"
