#!/usr/bin/env bash
set -euo pipefail

# Update-Level Interference Experiment
#
# Execution order:
#   Phase 0: Verification checks (mostly CPU, ~10min)
#   Phase 1: Coarse checkpoint-difference interference (CPU, ~30min)
#   eval: Per-case behavioral evaluation (GPU, ~30min)
#   stopnogo: Stop/no-go analysis (CPU)
#   extract_base: Extract base model weight for directional analysis (GPU, one-time ~30s)
#   directional: Directional alignment analysis (CPU, ~5min, requires extract_base + eval)
#   install_eval: Evaluate first-1K at 1K checkpoint with margin (GPU, ~30min)
#   install_analyze: Key geometry features + logistic model (CPU, ~5min)
#   eval_all: All GPU work (5K eval + 1K eval + base weight extraction)
#   analyze: All CPU analysis (stopnogo + directional + installation strength)
#   multilayer_keys: Extract keys for layers 4-8 (GPU, ~15min)
#   multilayer_extract_base: Extract base weights for layers 4-8 (GPU, ~30s)
#   multilayer_gpu: All multi-layer GPU work (keys + base weights)
#   multilayer_analyze: Run directional + installation analysis for layers 4-8 (CPU, ~25min)
#   Phase 2: Fine-grained per-batch delta capture (GPU, ~2h per ordering)
#
# Usage (local):
#   bash scripts/run_interference_experiment.sh 42 phase0
#   bash scripts/run_interference_experiment.sh 42 phase1
#   bash scripts/run_interference_experiment.sh 42 eval
#   bash scripts/run_interference_experiment.sh 42 stopnogo
#   bash scripts/run_interference_experiment.sh 42 extract_base
#   bash scripts/run_interference_experiment.sh 42 directional
#   bash scripts/run_interference_experiment.sh 42 install_eval
#   bash scripts/run_interference_experiment.sh 42 install_analyze
#   bash scripts/run_interference_experiment.sh 42 phase2
#   bash scripts/run_interference_experiment.sh 42 all
#
# Usage (SkyPilot — PHASE from env):
#   PHASE=phase1 bash scripts/run_interference_experiment.sh 42
#
# Environment variables:
#   PHASE            - Which phase to run (default: all)
#   ORDERING         - For phase2: key_clustered or key_dispersed (default: both)
#   CUDA_DEVICE      - GPU device index (default: 0)
#   CHECKPOINT_BASE  - Override checkpoint directory
#   MODEL_NAME       - Model name (default: meta-llama/Meta-Llama-3-8B-Instruct)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
PHASE="${2:-${PHASE:-analyze}}"
ORDERING="${3:-${ORDERING:-}}"  # For phase2: key_clustered or key_dispersed
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

# Resolve checkpoint base
if [[ -n "${CHECKPOINT_BASE:-}" ]]; then
    CKPT_BASE="$CHECKPOINT_BASE"
elif [[ -d "/s3-data/continual-learning/alphaedit/checkpoints" ]]; then
    CKPT_BASE="/s3-data/continual-learning/alphaedit/checkpoints"
else
    CKPT_BASE="$HOME/.cache/alphaedit_checkpoints"
fi


echo "=== Update-Level Interference Experiment ==="
echo "  Phase:      $PHASE"
echo "  Seed:       $SEED"
echo "  Checkpoint: $CKPT_BASE"
echo "  Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

case "$PHASE" in
    phase0|0)
        echo "--- Phase 0: Verification Checks ---"
        uv run python src/experiments/interference_from_checkpoints.py \
            --phase 0 \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"
        ;;

    phase1|1)
        echo "--- Phase 1: Coarse Checkpoint-Difference Interference ---"
        uv run python src/experiments/interference_from_checkpoints.py \
            --phase 1 \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"
        ;;

    phase2|2)
        echo "--- Phase 2: Fine-Grained Delta Capture (GPU) ---"
        if [[ -z "$ORDERING" ]]; then
            # Run both orderings sequentially
            echo "  Running key_clustered..."
            uv run python src/runners/update_interference_runner.py \
                --seed "$SEED" \
                --ordering key_clustered \
                --cuda_device "$CUDA_DEVICE" \
                --model_name "$MODEL_NAME" \
                --checkpoint_base "$CKPT_BASE/matched_ordering/AlphaEdit/key_clustered/seed${SEED}"

            echo ""
            echo "  Running key_dispersed..."
            uv run python src/runners/update_interference_runner.py \
                --seed "$SEED" \
                --ordering key_dispersed \
                --cuda_device "$CUDA_DEVICE" \
                --model_name "$MODEL_NAME" \
                --checkpoint_base "$CKPT_BASE/matched_ordering/AlphaEdit/key_dispersed/seed${SEED}"
        else
            echo "  Running $ORDERING..."
            uv run python src/runners/update_interference_runner.py \
                --seed "$SEED" \
                --ordering "$ORDERING" \
                --cuda_device "$CUDA_DEVICE" \
                --model_name "$MODEL_NAME" \
                --checkpoint_base "$CKPT_BASE/matched_ordering/AlphaEdit/${ORDERING}/seed${SEED}"
        fi
        ;;

    eval)
        echo "--- Per-Case Behavioral Evaluation (GPU) ---"
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --eval_behavioral \
            --model_name "$MODEL_NAME"
        ;;

    stopnogo)
        echo "--- Stop/No-Go Analysis ---"
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --stop_nogo
        ;;

    extract_base)
        echo "--- Extract Base Model Weight (one-time, GPU) ---"
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --extract_base_weight \
            --extract_only \
            --model_name "$MODEL_NAME"
        ;;

    directional)
        echo "--- Directional Alignment Analysis (CPU) ---"
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"
        ;;

    install_eval)
        echo "--- Installation Strength: Evaluate at 1K (GPU) ---"
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --eval_at_1k \
            --checkpoint_base "$CKPT_BASE" \
            --model_name "$MODEL_NAME"
        ;;

    install_analyze)
        echo "--- Installation Strength: Features + Model (CPU) ---"
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --analyze \
            --checkpoint_base "$CKPT_BASE"
        ;;

    multilayer_keys)
        echo "--- Extract Keys for Layers 4-8 (GPU) ---"
        # Force FUSE mount to list orderings directory
        STREAM_DIR="${RESULT_ROOT:-$(uv run python -c 'import sys; sys.path.insert(0,"src/util"); from paths import get_result_root; print(get_result_root())')}/matched_ordering"
        echo "  Stream dir: $STREAM_DIR"
        ls "$STREAM_DIR/orderings/" || true
        for LAYER in 4 5 6 7 8; do
            echo ""
            echo "=== Layer $LAYER ==="
            uv run python analysis/matched_ordering_key_geometry.py \
                --seed "$SEED" \
                --layer "$LAYER" \
                --stream_dir "$STREAM_DIR"
        done
        ;;

    multilayer_extract_base)
        echo "--- Extract Base Weights for Layers 4-8 (GPU) ---"
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --extract_base_weight \
            --extract_only \
            --layers "4,5,6,7,8" \
            --model_name "$MODEL_NAME"
        ;;

    multilayer_gpu)
        echo "--- All Multi-Layer GPU Work (keys + base weights) ---"
        echo ""

        # Force FUSE mount to list orderings directory
        STREAM_DIR="${RESULT_ROOT:-$(uv run python -c 'import sys; sys.path.insert(0,"src/util"); from paths import get_result_root; print(get_result_root())')}/matched_ordering"
        echo "  Stream dir: $STREAM_DIR"
        ls "$STREAM_DIR/orderings/" || true

        echo "=== Extract Keys for Layers 4-8 ==="
        for LAYER in 4 5 6 7 8; do
            echo ""
            echo "  Layer $LAYER..."
            uv run python analysis/matched_ordering_key_geometry.py \
                --seed "$SEED" \
                --layer "$LAYER" \
                --stream_dir "$STREAM_DIR"
        done

        echo ""
        echo "=== Extract Base Weights for Layers 4-8 ==="
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --extract_base_weight \
            --extract_only \
            --layers "4,5,6,7,8" \
            --model_name "$MODEL_NAME"
        ;;

    multilayer_analyze)
        echo "--- Multi-Layer Analysis: Layers 4-8 (CPU) ---"
        for LAYER in 4 5 6 7 8; do
            echo ""
            echo "========================================"
            echo "=== Layer $LAYER ==="
            echo "========================================"

            echo ""
            echo "--- Directional Alignment (layer $LAYER) ---"
            uv run python src/experiments/directional_alignment.py \
                --seed "$SEED" \
                --layer "$LAYER" \
                --checkpoint_base "$CKPT_BASE"

            echo ""
            echo "--- Installation Strength Features + Model (layer $LAYER) ---"
            uv run python src/experiments/installation_strength.py \
                --seed "$SEED" \
                --layer "$LAYER" \
                --analyze \
                --checkpoint_base "$CKPT_BASE"
        done
        ;;

    eval_all)
        echo "--- All GPU Evaluations (5K eval + 1K eval + base weight) ---"
        echo ""

        echo "=== Per-Case Eval at 5K ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --eval_behavioral \
            --model_name "$MODEL_NAME"

        echo ""
        echo "=== Installation Strength: Eval at 1K ==="
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --eval_at_1k \
            --checkpoint_base "$CKPT_BASE" \
            --model_name "$MODEL_NAME"

        echo ""
        echo "=== Extract Base Weight ==="
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --extract_base_weight \
            --extract_only \
            --model_name "$MODEL_NAME"
        ;;

    analyze)
        echo "--- All CPU Analysis (stopnogo + directional + installation strength) ---"
        echo ""

        echo "=== Stop/No-Go ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --stop_nogo

        echo ""
        echo "=== Directional Alignment ==="
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"

        echo ""
        echo "=== Installation Strength: Features + Model ==="
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --analyze \
            --checkpoint_base "$CKPT_BASE"
        ;;

    all)
        echo "--- Full Pipeline ---"
        echo ""

        echo "=== Phase 0: Verification ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --phase 0 \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"

        echo ""
        echo "=== Phase 1: Coarse Interference ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --phase 1 \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"

        echo ""
        echo "=== GPU: Per-Case Eval at 5K ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --eval_behavioral \
            --model_name "$MODEL_NAME"

        echo ""
        echo "=== GPU: Installation Strength Eval at 1K ==="
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --eval_at_1k \
            --checkpoint_base "$CKPT_BASE" \
            --model_name "$MODEL_NAME"

        echo ""
        echo "=== GPU: Extract Base Weight ==="
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --extract_base_weight \
            --extract_only \
            --model_name "$MODEL_NAME"

        echo ""
        echo "=== CPU: Stop/No-Go ==="
        uv run python src/experiments/interference_from_checkpoints.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE" \
            --stop_nogo

        echo ""
        echo "=== CPU: Directional Alignment ==="
        uv run python src/experiments/directional_alignment.py \
            --seed "$SEED" \
            --checkpoint_base "$CKPT_BASE"

        echo ""
        echo "=== CPU: Installation Strength Analysis ==="
        uv run python src/experiments/installation_strength.py \
            --seed "$SEED" \
            --analyze \
            --checkpoint_base "$CKPT_BASE"
        ;;

    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 SEED PHASE [ORDERING]"
        echo "Phases: phase0 phase1 eval install_eval extract_base eval_all | stopnogo directional install_analyze analyze | multilayer_keys multilayer_extract_base multilayer_gpu multilayer_analyze | phase2 | all"
        exit 1
        ;;
esac

echo ""
echo "=== Done ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
