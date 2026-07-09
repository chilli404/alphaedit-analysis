#!/usr/bin/env bash
set -euo pipefail

# Smoke test: validates the entire pipeline with minimal compute.
# Runs 2 edits on 10 dataset samples with no GLUE eval.
# Should complete in ~5 minutes on GPU (mainly model load time).
#
# Usage: bash scripts/smoke_test.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_DEVICE="${CUDA_DEVICE:-0}"

echo "=== SMOKE TEST ==="
echo "  2 edits, 10 samples, no GLUE eval"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Verify stats are linked
STATS_DIR="vendor/AlphaEdit/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats"
if [[ ! -d "$STATS_DIR" ]] || [[ -z "$(ls -A "$STATS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: Covariance stats not found at $STATS_DIR"
    echo "Run: bash scripts/link_stats.sh"
    exit 1
fi
echo "  Stats directory: OK ($(ls "$STATS_DIR"/*.npz 2>/dev/null | wc -l) files)"

# Verify submodule
if [[ ! -f "vendor/AlphaEdit/experiments/evaluate.py" ]]; then
    echo "ERROR: AlphaEdit submodule not initialized"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi
echo "  AlphaEdit submodule: OK"

# Run minimal experiment
echo ""
echo "  Running minimal AlphaEdit experiment..."
uv run python src/seeded_runner.py \
    --seed 42 \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name AlphaEdit \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname Llama3-8B.json \
    --ds_name mcf \
    --dataset_size_limit 10 \
    --num_edits 2 \
    --downstream_eval_steps 0 \
    --skip_generation_tests \
    --conserve_memory

# Verify output exists
RESULTS_DIR="vendor/AlphaEdit/results/AlphaEdit"
if ls "$RESULTS_DIR"/run_*/2_edits-case_*.json 1>/dev/null 2>&1; then
    echo ""
    echo "=== SMOKE TEST PASSED ==="
    echo "Result files:"
    ls "$RESULTS_DIR"/run_*/2_edits-case_*.json
else
    echo ""
    echo "=== SMOKE TEST FAILED ==="
    echo "No result files found in $RESULTS_DIR"
    exit 1
fi
