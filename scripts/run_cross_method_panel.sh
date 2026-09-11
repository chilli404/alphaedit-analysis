#!/usr/bin/env bash
set -euo pipefail

# Launch the full cross-method temporal-path stress test panel for ICLR.
#
# Methods:
#   AlphaEdit     — already complete (3 seeds × 5 orderings)
#   EvoEdit       — seed 42 done; seeds 2024+137 launching or running
#   MEMIT-Seq     — seed 42 done; seeds 2024+137 launching or running
#   NSE           — NEW: needs full editing runs
#   REVIVE        — has existing data but checkpoint format issues
#
# Methods NOT available (no public code):
#   QueueEDIT     — no public repo found
#   AlphaEdit+    — no public repo found (only original AlphaEdit repo)
#   SpecEdit      — no public repo found
#   DipEdit       — no public repo found
#
# Minimum experiment: HIGH + RANDOM0, seeds 42 + 2024, 10K edits
# Full experiment: HIGH + LOW + RANDOM0, seeds 42 + 2024 + 137, 10K edits
#
# Usage:
#   bash scripts/run_cross_method_panel.sh              # Dry run
#   bash scripts/run_cross_method_panel.sh --execute     # Launch all
#   bash scripts/run_cross_method_panel.sh --nse-only    # Just NSE
#   bash scripts/run_cross_method_panel.sh --gptj        # GPT-J variants

echo "═══════════════════════════════════════════════════════════════"
echo "Cross-Method Temporal-Path Panel for ICLR"
echo "═══════════════════════════════════════════════════════════════"

EXECUTE=false
NSE_ONLY=false
GPTJ=false
for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=true ;;
        --nse-only) NSE_ONLY=true ;;
        --gptj) GPTJ=true ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKY_YAML="$PROJECT_DIR/sky/alphaedit_gpu.yaml"

launch() {
    local cluster="$1"; shift
    local desc="$1"; shift
    # remaining args are --env pairs
    if [[ "$EXECUTE" == "true" ]]; then
        echo "  LAUNCH: $cluster ($desc)"
        sky launch "$SKY_YAML" --env-file "$PROJECT_DIR/.env" "$@" \
            --cluster "$cluster" --detach-run -y
    else
        echo "  [DRY] $cluster ($desc)"
        echo "    sky launch $SKY_YAML --env-file .env $@ --cluster $cluster --detach-run -y"
    fi
    echo ""
}

# ============================================================
# NSE: Neuron-Level Sequential Editing (ACL 2025)
# Repo: baselines/EvoEdit/nse/ (already integrated)
# Runner: scripts/run_nse_baseline.sh
# ============================================================
echo ""
echo "--- NSE (Neuron-Level Sequential Editing) ---"
echo "  4 minimum runs + 2 optional = 6 total"
echo "  ~10h each on L40S"
echo ""

# Minimum: HIGH + RANDOM0, seeds 42 + 2024
launch "nse-fbhi-s42" "NSE HIGH seed42" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=42 --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000

launch "nse-fbrnd0-s42" "NSE RANDOM0 seed42" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=42 --env ORDERING=fb_random0 --env TARGET_EDITS=10000

launch "nse-fbhi-s2024" "NSE HIGH seed2024" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=2024 --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000

launch "nse-fbrnd0-s2024" "NSE RANDOM0 seed2024" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=2024 --env ORDERING=fb_random0 --env TARGET_EDITS=10000

# Optional: LOW on both seeds
launch "nse-fblo-s42" "NSE LOW seed42 (optional)" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=42 --env ORDERING=fb_low_exposure --env TARGET_EDITS=10000

launch "nse-fblo-s2024" "NSE LOW seed2024 (optional)" \
    --env EXPERIMENT_NAME=nse_baseline --env SEED=2024 --env ORDERING=fb_low_exposure --env TARGET_EDITS=10000

if [[ "$NSE_ONLY" == "true" ]]; then
    echo "=== NSE-only mode. Done. ==="
    exit 0
fi

# ============================================================
# GPT-J variants (EvoEdit + REVIVE on GPT-J)
# ============================================================
if [[ "$GPTJ" == "true" ]]; then
    echo ""
    echo "--- GPT-J Cross-Method Runs ---"
    echo "  EvoEdit + NSE on GPT-J, HIGH + RANDOM0, seed 42"
    echo ""

    launch "evo-gptj-fbhi-s42" "EvoEdit GPT-J HIGH seed42" \
        --env EXPERIMENT_NAME=evoedit_baseline --env SEED=42 \
        --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
        --env MODEL_NAME=EleutherAI/gpt-j-6b

    launch "evo-gptj-fbrnd0-s42" "EvoEdit GPT-J RANDOM0 seed42" \
        --env EXPERIMENT_NAME=evoedit_baseline --env SEED=42 \
        --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
        --env MODEL_NAME=EleutherAI/gpt-j-6b

    launch "nse-gptj-fbhi-s42" "NSE GPT-J HIGH seed42" \
        --env EXPERIMENT_NAME=nse_baseline --env SEED=42 \
        --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
        --env MODEL_NAME=EleutherAI/gpt-j-6b

    launch "nse-gptj-fbrnd0-s42" "NSE GPT-J RANDOM0 seed42" \
        --env EXPERIMENT_NAME=nse_baseline --env SEED=42 \
        --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
        --env MODEL_NAME=EleutherAI/gpt-j-6b
fi

echo "═══════════════════════════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Methods with public code and integration ready:"
echo "  ✓ AlphaEdit     — complete (3 seeds × 5 orderings)"
echo "  ✓ EvoEdit       — seed 42 done, seeds 2024+137 running"
echo "  ✓ MEMIT-Seq     — seed 42 done, seeds 2024+137 running"
echo "  ✓ NSE           — runner written, ready to launch"
echo "  ~ REVIVE        — existing runs have checkpoint format issues"
echo ""
echo "Methods WITHOUT public repos (cannot run):"
echo "  ✗ QueueEDIT     — no public code found"
echo "  ✗ AlphaEdit+    — no public code found"
echo "  ✗ SpecEdit      — no public code found"
echo "  ✗ DipEdit       — no public code found"
echo ""
echo "Achievable ICLR panel: AlphaEdit + EvoEdit + MEMIT-Seq + NSE (+ REVIVE if fixed)"
echo ""
if [[ "$EXECUTE" == "false" ]]; then
    echo "DRY RUN. Add --execute to launch."
    echo "  bash scripts/run_cross_method_panel.sh --execute          # All NSE + panel"
    echo "  bash scripts/run_cross_method_panel.sh --nse-only --execute  # Just NSE"
    echo "  bash scripts/run_cross_method_panel.sh --gptj --execute      # + GPT-J variants"
fi
