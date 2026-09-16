#!/usr/bin/env bash
#
# Direct Logit-Damage A/B Intervention Experiment
#
# Installs N batches of edits, then tests whether a HIGH-cosine future batch
# causes more logit damage to focal edits than a LOW-cosine future batch.
#
# Prerequisites:
#   - Fixed-batch orderings generated:
#     uv run python src/datasets/generate_orderings.py --seed SEED --fixed_batch
#   - Precomputed key vectors at results/key_vectors/full_mcf/keys_seed42_layer6.npz
#
# Usage:
#   bash scripts/run_logit_damage_experiment.sh SEED
#   bash scripts/run_logit_damage_experiment.sh 42 --n_trials 5
#
set -euo pipefail

SEED="${1:?Usage: $0 SEED [extra-args...]}"
shift

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
KEYS_PATH="${KEYS_PATH:-$RESULT_ROOT/key_vectors/full_mcf/keys_seed42_layer6.npz}"

# Ensure fixed-batch orderings exist
BA_PATH="$RESULT_ROOT/matched_ordering/diagnostics/fixed_batch_assignment_seed${SEED}.json"
if [[ ! -f "$BA_PATH" ]]; then
    echo "Generating fixed-batch orderings for seed $SEED..."
    cd "$PROJECT_DIR"
    if [[ -d "/s3-data/continual-learning/alphaedit/dsets" ]]; then
        DATA_DIR="/s3-data/continual-learning/alphaedit/dsets"
    else
        DATA_DIR="$PROJECT_DIR/data/dsets"
    fi
    uv run python src/datasets/generate_orderings.py \
        --seed "$SEED" --fixed_batch --keys_path "$KEYS_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$RESULT_ROOT/matched_ordering"
fi

echo "========================================"
echo "Logit Damage A/B Intervention"
echo "  Seed:       $SEED"
echo "  Keys:       $KEYS_PATH"
echo "  Results:    $RESULT_ROOT/logit_damage/seed${SEED}/"
echo "========================================"

cd "$PROJECT_DIR"
uv run python src/runners/logit_damage_runner.py \
    --seed "$SEED" \
    --keys_path "$KEYS_PATH" \
    --install_batches "${INSTALL_BATCHES:-10}" \
    --n_trials "${N_TRIALS:-10}" \
    "$@"
