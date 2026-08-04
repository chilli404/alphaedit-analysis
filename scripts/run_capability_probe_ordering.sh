#!/usr/bin/env bash
set -euo pipefail

# Run offline capability probes (WikiText perplexity + MMLU) on matched
# ordering checkpoints. Loads each checkpoint's model weights and measures
# general language capability to detect global collapse.
#
# Prerequisites:
#   - GPU with >= 16GB VRAM (Llama-3-8B in fp16)
#   - Checkpoints available at $CHECKPOINT_ROOT/matched_ordering/{ALG}/{ORDERING}/seed{SEED}/
#
# Usage:
#   bash scripts/run_capability_probe_ordering.sh 42 AlphaEdit greedy_minmax
#   bash scripts/run_capability_probe_ordering.sh 42 AlphaEdit cluster_topo
#   bash scripts/run_capability_probe_ordering.sh 42 AlphaEdit key_clustered
#
# On SkyPilot (checkpoints on S3 FUSE mount):
#   CHECKPOINT_ROOT=/s3-data/continual-learning/alphaedit/checkpoints \
#     bash scripts/run_capability_probe_ordering.sh 42 AlphaEdit greedy_minmax

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-${SEED:-42}}"
ALG="${2:-${ALG_NAME:?Set ALG_NAME or pass as arg 2}}"
ORDERING="${3:-${ORDERING:?Set ORDERING or pass as arg 3}}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"

CKPT_DIR="$CHECKPOINT_ROOT/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
OUT_DIR="$RESULT_ROOT/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
OUT_FILE="$OUT_DIR/capability_probe_seed${SEED}.jsonl"

echo "=== Capability Probe: Matched Ordering ==="
echo "  Seed:       $SEED"
echo "  Algorithm:  $ALG"
echo "  Ordering:   $ORDERING"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output:     $OUT_FILE"
echo ""

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    echo "  On SkyPilot, set CHECKPOINT_ROOT=/s3-data/continual-learning/alphaedit/checkpoints"
    exit 1
fi

# Count available checkpoints
N_CKPTS=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "batch_*" | wc -l | tr -d ' ')
if [[ "$N_CKPTS" -eq 0 ]]; then
    echo "ERROR: No batch_* checkpoint directories found in $CKPT_DIR"
    exit 1
fi
echo "  Found $N_CKPTS checkpoints"

mkdir -p "$OUT_DIR"

cd "$PROJECT_DIR"

uv run python src/mechanism/capability_probe_offline.py \
    --seed "$SEED" \
    --alg_name "$ALG" \
    --checkpoint_dir "$CKPT_DIR" \
    --output "$OUT_FILE"

echo ""
echo "=== Capability Probe Complete ==="
echo "  Output: $OUT_FILE"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
