#!/usr/bin/env bash
set -euo pipefail

# Polykernel+SeqReg: GPT-J-6B
# Wrapper that forces GPT-J model and passes through to run_polykernel_seqreg.sh
#
# Usage:
#   bash scripts/run_polykernel_seqreg_gptj.sh [SEED] [LAMBDA_PREV] [LAMBDA_DELTA]
#   TARGET_EDITS=10000 bash scripts/run_polykernel_seqreg_gptj.sh 42 1.0 0.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config FIRST, then override with GPT-J specifics
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Force GPT-J model
export MODEL_NAME="EleutherAI/gpt-j-6b"
export HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"

# Default to 10K edits
export TARGET_EDITS="${TARGET_EDITS:-10000}"

echo "=== Polykernel+SeqReg (GPT-J-6B) ==="
echo "  Model: $MODEL_NAME"

exec bash "$SCRIPT_DIR/run_polykernel_seqreg.sh" "${@}"
