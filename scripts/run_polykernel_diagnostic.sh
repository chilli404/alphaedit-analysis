#!/usr/bin/env bash
set -euo pipefail

# Polynomial-Kernel Memory Diagnostic
#
# Tests whether AlphaEdit/MEMIT failure modes are consistent with a linear
# key-space capacity bottleneck by comparing Gram matrix geometry under
# linear vs degree-2 polynomial kernels.
#
# Two stages:
#   Stage 1 (GPU): Extract raw edit keys during editing
#   Stage 2 (CPU): Gram matrix analysis, metrics, interpretation
#
# Modes:
#   Batch mode (default): --num_edits 100 --dataset_size_limit 2000
#   Coupling mode (COUPLING_MODE=true): --num_edits 1, uses coupling dataset
#
# ALG_NAME can be "AlphaEdit", "MEMIT", or "both" (runs both sequentially).
#
# Usage:
#   bash scripts/run_polykernel_diagnostic.sh [SEED] [ALG_NAME] [DATASET_SIZE_LIMIT] [NUM_EDITS]
#   bash scripts/run_polykernel_diagnostic.sh 42 AlphaEdit 2000 100
#   bash scripts/run_polykernel_diagnostic.sh 42 MEMIT 2000 100
#   bash scripts/run_polykernel_diagnostic.sh 42 both
#   COUPLING_MODE=true bash scripts/run_polykernel_diagnostic.sh 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
ALG_NAME="${2:-${ALG_NAME:-AlphaEdit}}"
DATASET_SIZE_LIMIT="${3:-${DATASET_SIZE_LIMIT:-2000}}"
NUM_EDITS="${4:-${NUM_EDITS:-100}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
COUPLING_MODE="${COUPLING_MODE:-false}"
RANK_THRESHOLD="${RANK_THRESHOLD:-1e-5}"
WINDOW_SIZE="${WINDOW_SIZE:-5}"

# Handle "both" by running each algorithm sequentially
if [[ "$ALG_NAME" == "both" ]]; then
    echo "=== ALG_NAME=both: running AlphaEdit then MEMIT ==="
    echo ""
    ALG_NAME=AlphaEdit bash "$0" "$SEED" AlphaEdit "$DATASET_SIZE_LIMIT" "$NUM_EDITS"
    ALG_NAME=MEMIT bash "$0" "$SEED" MEMIT "$DATASET_SIZE_LIMIT" "$NUM_EDITS"
    echo "=== Both algorithms complete ==="
    exit 0
fi

echo "=== Polynomial-Kernel Memory Diagnostic ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG_NAME"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Coupling mode: $COUPLING_MODE"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

RESULTS_DIR="results/polykernel_diagnostic"
KEYS_FILE="$RESULTS_DIR/keys_${ALG_NAME}_seed${SEED}.pt"

# --- Stage 1: Key Extraction (GPU) ---
echo "=== Stage 1: Key Extraction (GPU) — $ALG_NAME ==="

EXTRA_ARGS=""

if [[ "$COUPLING_MODE" == "true" ]]; then
    # Coupling mode: use coupling dataset with num_edits=1
    COUPLING_DATASET="results/coupling_stress/coupling_dataset_seed${SEED}.json"
    if [[ ! -f "$COUPLING_DATASET" ]]; then
        echo "  Coupling dataset not found at $COUPLING_DATASET"
        echo "  Generating via coupling_dataset module..."
        uv run python -c "
import sys; sys.path.insert(0, 'src/datasets')
from coupling_dataset import generate_coupling_dataset
from pathlib import Path
import json

data_dir = Path('vendor/AlphaEdit/data')
sequence = generate_coupling_dataset(data_dir=data_dir, seed=${SEED}, max_pairs_per_type=60, warmup_count=20)
out_path = Path('${COUPLING_DATASET}')
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(sequence, f)
print(f'Generated {len(sequence)} records -> {out_path}')
" || true
        if [[ ! -f "$COUPLING_DATASET" ]]; then
            echo "ERROR: Failed to generate coupling dataset"
            exit 1
        fi
    fi
    # Count records in coupling dataset
    COUPLING_SIZE=$(uv run python -c "import json; print(len(json.load(open('${COUPLING_DATASET}'))))")
    echo "  Coupling dataset: $COUPLING_DATASET ($COUPLING_SIZE records)"
    EXTRA_ARGS="--coupling_dataset $COUPLING_DATASET --num_edits 1 --dataset_size_limit $COUPLING_SIZE"
else
    echo "  Batch mode: $NUM_EDITS edits/batch, $DATASET_SIZE_LIMIT total"
    EXTRA_ARGS="--num_edits $NUM_EDITS --dataset_size_limit $DATASET_SIZE_LIMIT"
fi

uv run python src/polykernel/polykernel_key_extractor.py \
    --seed "$SEED" \
    --alg_name "$ALG_NAME" \
    --cuda_device "$CUDA_DEVICE" \
    --conserve_memory \
    $EXTRA_ARGS

echo ""

# --- Stage 2: Gram Matrix Analysis (CPU) ---
echo "=== Stage 2: Gram Matrix Analysis (CPU) — $ALG_NAME ==="

if [[ ! -f "$KEYS_FILE" ]]; then
    echo "ERROR: Keys file not found at $KEYS_FILE"
    echo "  Stage 1 may have failed."
    exit 1
fi

uv run python src/polykernel/polykernel_diagnostic.py \
    --keys_file "$KEYS_FILE" \
    --rank_threshold "$RANK_THRESHOLD" \
    --window_size "$WINDOW_SIZE"

echo ""
echo "=== Polynomial-kernel diagnostic complete ($ALG_NAME) ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Keys: $KEYS_FILE"
echo "  Analysis: $RESULTS_DIR/analysis_${ALG_NAME}_seed${SEED}.json"
echo ""
