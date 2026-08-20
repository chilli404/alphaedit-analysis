#!/usr/bin/env bash
set -euo pipefail

# Polykernel Editor: GPT-J-6B
# Wrapper that forces GPT-J model and passes through to run_polykernel_editor.sh
#
# Usage:
#   bash scripts/run_polykernel_editor_gptj.sh [SEED]
#   INJECT_C0=true bash scripts/run_polykernel_editor_gptj.sh 42
#   DATASET_SIZE_LIMIT=10000 INJECT_C0=true bash scripts/run_polykernel_editor_gptj.sh 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config FIRST, then override with GPT-J specifics
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Force GPT-J model
export MODEL_NAME="EleutherAI/gpt-j-6b"
export HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"

# Default to 10K edits for GPT-J polykernel
export DATASET_SIZE_LIMIT="${DATASET_SIZE_LIMIT:-10000}"

echo "=== Polykernel Editor (GPT-J-6B) ==="
echo "  Model: $MODEL_NAME"

exec bash "$SCRIPT_DIR/run_polykernel_editor.sh" "${@}"
