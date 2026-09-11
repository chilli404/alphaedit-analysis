#!/usr/bin/env bash
set -euo pipefail

# MEMIT-Seq Logit-Damage A/B Intervention
# Cross-editor replication: does interference attenuate under history-aware editing?
#
# Usage:
#   bash scripts/run_logit_damage_memit.sh SEED
#   bash scripts/run_logit_damage_memit.sh 42 --lambda_prev 1.0 --lambda_delta 1.0

SEED="${1:?Usage: $0 SEED [extra-args...]}"
shift

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
KEYS_PATH="${KEYS_PATH:-$RESULT_ROOT/key_vectors/full_mcf/keys_seed42_layer6.npz}"

cd "$PROJECT_DIR"
uv run python src/runners/logit_damage_memit_runner.py \
    --seed "$SEED" \
    --keys_path "$KEYS_PATH" \
    --install_batches "${INSTALL_BATCHES:-10}" \
    --n_trials "${N_TRIALS:-10}" \
    --lambda_prev "${LAMBDA_PREV:-1.0}" \
    --lambda_delta "${LAMBDA_DELTA:-1.0}" \
    "$@"
