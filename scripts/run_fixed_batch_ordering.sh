#!/usr/bin/env bash
#
# Fixed-Batch Ordering Experiment
#
# Holds batch MEMBERSHIP constant while permuting temporal SEQUENCE of batches.
# Isolates cross-batch future-key exposure from within-batch Gram matrix effects.
#
# Usage:
#   bash scripts/run_fixed_batch_ordering.sh SEED [ALG] [PHASE]
#
# Examples:
#   bash scripts/run_fixed_batch_ordering.sh 2024                    # Generate + run all orderings
#   bash scripts/run_fixed_batch_ordering.sh 2024 AlphaEdit          # Generate + run AlphaEdit only
#   bash scripts/run_fixed_batch_ordering.sh 2024 AlphaEdit generate # Generate orderings only
#   bash scripts/run_fixed_batch_ordering.sh 2024 AlphaEdit edit     # Run editing only (orderings must exist)
#
set -euo pipefail

SEED="${1:?Usage: $0 SEED [ALG] [PHASE]}"
ALG="${2:-AlphaEdit}"
PHASE="${3:-all}"  # generate | edit | all

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
N_RANDOM="${N_RANDOM_PERMS:-3}"

# Resolve keys path: prefer RESULT_ROOT (works on both S3 and local)
KEYS_PATH="${KEYS_PATH:-$RESULT_ROOT/key_vectors/full_mcf/keys_seed42_layer6.npz}"

echo "========================================"
echo "Fixed-Batch Ordering Experiment"
echo "  Seed:     $SEED"
echo "  Alg:      $ALG"
echo "  Phase:    $PHASE"
echo "  Results:  $RESULT_ROOT"
echo "  N random: $N_RANDOM"
echo "========================================"

# ── Phase 1: Generate fixed-batch orderings ──────────────────────────────

if [[ "$PHASE" == "generate" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== Generating fixed-batch orderings ==="

    ORD_DIR="$RESULT_ROOT/matched_ordering/orderings"
    HI_PATH="$ORD_DIR/fb_high_exposure_seed${SEED}.json"

    if [[ -f "$HI_PATH" ]]; then
        echo "  Orderings already exist at $ORD_DIR"
        echo "  Delete to regenerate."
    else
        cd "$PROJECT_DIR"
        # Resolve data_dir for MCF dataset (S3 or local)
        if [[ -n "${DSETS_ROOT:-}" ]] && [[ -d "$DSETS_ROOT" ]]; then
            DATA_DIR="$DSETS_ROOT"
        else
            DATA_DIR="$PROJECT_DIR/data/dsets"
        fi
        uv run python src/datasets/generate_orderings.py \
            --seed "$SEED" \
            --fixed_batch \
            --n_random_perms "$N_RANDOM" \
            --keys_path "$KEYS_PATH" \
            --data_dir "$DATA_DIR" \
            --output_dir "$RESULT_ROOT/matched_ordering"
    fi
fi

# ── Phase 2: Run editing experiments ─────────────────────────────────────

if [[ "$PHASE" == "edit" || "$PHASE" == "all" ]]; then
    echo ""
    echo "=== Running fixed-batch ordering experiments ==="

    ORDERINGS=("fb_high_exposure" "fb_low_exposure")
    for i in $(seq 0 $((N_RANDOM - 1))); do
        ORDERINGS+=("fb_random${i}")
    done

    for ORDERING in "${ORDERINGS[@]}"; do
        STREAM="$RESULT_ROOT/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
        if [[ ! -f "$STREAM" ]]; then
            echo "  ERROR: Stream not found: $STREAM"
            echo "  Run with PHASE=generate first."
            exit 1
        fi

        echo ""
        echo "--- Running: $ALG / $ORDERING / seed $SEED ---"
        bash "$PROJECT_DIR/scripts/run_matched_ordering.sh" "$SEED" "$ALG" "$ORDERING"
    done
fi

echo ""
echo "========================================"
echo "Fixed-batch ordering experiment complete."
echo "  Results: $RESULT_ROOT/matched_ordering/$ALG/fb_*/seed${SEED}/"
echo "========================================"
