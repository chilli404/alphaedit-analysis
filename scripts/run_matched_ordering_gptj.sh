#!/usr/bin/env bash
# GPT-J matched ordering wrapper — delegates to run_matched_ordering.sh
# with GPT-J model and GPT-J-specific ordering files.
#
# Key differences from Llama runs:
# - Uses GPT-J-clustered orderings (matched_ordering_gptj/orderings/)
# - Uses GPT-J-specific checkpoint path (prevents cross-model contamination)
# - Forces FORCE_EDIT=true to prevent auto-detecting Llama checkpoints
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-EleutherAI/gpt-j-6b}"
export HPARAMS_FNAME="${HPARAMS_FNAME:-EleutherAI_gpt-j-6B.json}"

# Use GPT-J-specific ordering files and checkpoint paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"

# Override stream dir to use GPT-J orderings
export STREAM_DIR_OVERRIDE="$RESULT_ROOT/matched_ordering_gptj/orderings"

# Override checkpoint path to GPT-J-specific location
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
export CKPT_DIR_OVERRIDE="$CHECKPOINT_ROOT/matched_ordering/gpt-j-6b"

# Force editing (no auto-detect of Llama checkpoints)
export FORCE_EDIT=true

exec bash "$SCRIPT_DIR/run_matched_ordering.sh" "$@"
