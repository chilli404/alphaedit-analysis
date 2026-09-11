#!/usr/bin/env bash
set -euo pipefail
#
# Reproduce EvoEdit anchor result to verify our evaluation protocol.
#
# Published EvoEdit numbers (10K edits, BS=100, LLaMA-3-8B, MCF):
#   Efficacy: 98.29%   Generalization: 91.21%   Specificity: 63.91%
#
# This script has TWO modes:
#
#   MODE 1 — Evaluate existing checkpoint (fast, ~2-4 hours on 1 GPU)
#     Downloads the EvoEdit fb_random0/seed42 checkpoint from S3 and evaluates
#     with both probability-preference (paper metric) and argmax (our metric).
#
#   MODE 2 — Fresh run with default MCF ordering (slow, ~12+ hours on 1 GPU)
#     Runs EvoEdit from scratch with the default sequential MCF ordering to
#     exactly match the paper's setup, then evaluates.
#
# Usage:
#   # Mode 1: Evaluate existing checkpoint (recommended first)
#   bash scripts/reproduce_evoedit_anchor.sh eval
#   bash scripts/reproduce_evoedit_anchor.sh eval 2000     # quick: just 2K edits
#   bash scripts/reproduce_evoedit_anchor.sh eval 10000    # full 10K anchor
#
#   # Mode 2: Fresh run (for exact paper reproduction)
#   bash scripts/reproduce_evoedit_anchor.sh run
#
#   # On SkyPilot (fresh run on GPU cluster)
#   EXPERIMENT_NAME=evoedit_anchor bash sky/sky_launch.sh
#
# Environment variables:
#   CUDA_DEVICE     GPU index (default: 0)
#   MODEL_NAME      Model to use (default: NousResearch/Meta-Llama-3-8B-Instruct)
#   RESULT_ROOT     Where to write results (default: ./results)
#   TARGET_EDITS    For mode 2: how many edits to run (default: 10000)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source .env if present (for HF_TOKEN etc.)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODE="${1:-eval}"
EDITS="${2:-10000}"
SEED="${3:-42}"
ORDERING="${4:-fb_random0}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
TARGET_EDITS="${TARGET_EDITS:-10000}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "================================================================"
echo "EvoEdit Anchor Reproduction"
echo "  Mode:       $MODE"
echo "  Edits:      $EDITS"
echo "  Seed:       $SEED"
echo "  Model:      $MODEL_NAME"
echo "  GPU:        $DEVICE"
echo "================================================================"

case "$MODE" in
  eval)
    # ------------------------------------------------------------------
    # MODE 1: Evaluate existing checkpoint with dual metrics
    # ------------------------------------------------------------------
    echo ""
    echo "Evaluating EvoEdit checkpoint with both metrics..."
    echo "  Checkpoint: S3 evoedit/${ORDERING}/seed${SEED}/10000edits (edits_$(printf '%06d' $EDITS))"
    echo "  Stream:     ${RESULT_ROOT}/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
    echo ""

    STREAM_PATH="${RESULT_ROOT}/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"

    # Build eval command
    EVAL_ARGS=(
        "$PROJECT_DIR/scripts/eval_anchor_evoedit.py"
        --edits "$EDITS"
        --seed "$SEED"
        --ordering "$ORDERING"
        --model_name "$MODEL_NAME"
    )

    # Use stream if available (matches the exact facts that were edited)
    if [[ -f "$STREAM_PATH" ]]; then
        EVAL_ARGS+=(--stream_path "$STREAM_PATH")
        echo "  Using stream-ordered dataset (matches edited facts)"
    else
        echo "  Using default MCF dataset ordering"
        echo "  WARNING: Results may differ slightly from checkpoint's actual edit order"
    fi

    uv run python "${EVAL_ARGS[@]}"
    ;;

  run)
    # ------------------------------------------------------------------
    # MODE 2: Fresh EvoEdit run with default MCF ordering
    # ------------------------------------------------------------------
    echo ""
    echo "Running EvoEdit fresh with default MCF ordering..."
    echo "  This matches the paper's exact setup: BS=100, sequential case_id order"
    echo "  Checkpoints saved every 1000 edits"
    echo ""

    EVOEDIT_DIR="$PROJECT_DIR/baselines/EvoEdit"
    cd "$EVOEDIT_DIR"

    # --- Data setup ---
    mkdir -p data/stats
    LOCAL_DSETS="$PROJECT_DIR/data/dsets"
    S3_DSETS="/s3-data/continual-learning/alphaedit/dsets"
    DSET_SRC="$LOCAL_DSETS"
    [[ -d "$S3_DSETS" ]] && DSET_SRC="$S3_DSETS"

    for f in multi_counterfact.json counterfact.json zsre_mend_eval.json \
             attribute_snippets.json tfidf_vocab.json idf.npy \
             known_1000.json mmlu_subset.json wikitext_103_test.json; do
        if [[ -f "$DSET_SRC/$f" ]] && [[ ! -f "data/$f" ]]; then
            ln -sf "$DSET_SRC/$f" "data/$f"
        fi
    done
    echo "  Datasets linked from: $DSET_SRC"

    # Link covariance stats
    S3_STATS="/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct"
    LOCAL_STATS="$PROJECT_DIR/data/stats/llama3-8b-instruct/wikipedia_stats"
    VENDOR_STATS="$PROJECT_DIR/vendor/AlphaEdit/data/stats/Llama3-8B/wikipedia_stats"
    STATS_SRC=""
    for candidate in "$S3_STATS" "$VENDOR_STATS" "$LOCAL_STATS"; do
        if [[ -d "$candidate" ]] && ls "$candidate"/*.npz >/dev/null 2>&1; then
            STATS_SRC="$candidate"
            break
        fi
    done
    if [[ -n "$STATS_SRC" ]]; then
        mkdir -p "data/stats/Llama3-8B/wikipedia_stats"
        for f in "$STATS_SRC"/*.npz; do
            [[ -f "$f" ]] && ln -sf "$f" "data/stats/Llama3-8B/wikipedia_stats/$(basename "$f")"
        done
        echo "  Stats linked from: $STATS_SRC"
    else
        echo "  WARNING: No covariance stats found — will compute from scratch"
    fi

    # Model name variant symlinks for stats
    for variant in Meta-Llama-3-8B Meta-Llama-3-8B-Instruct \
        NousResearch_Meta-Llama-3-8B-Instruct NousResearch-Meta-Llama-3-8B-Instruct; do
        ln -sf Llama3-8B "data/stats/${variant}" 2>/dev/null || true
    done

    # Link null_space_project.pt if available
    for p in "$STATS_SRC/null_space_project.pt" \
             "$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt"; do
        if [[ -f "$p" ]]; then
            ln -sf "$p" "null_space_project.pt"
            break
        fi
    done

    # Runtime patches
    sed -i.bak 's/"meta-llama-3-8b-instruct": 8192,/"meta-llama-3-8b-instruct": 8192, "nousresearch--meta-llama-3-8b-instruct": 8192, "meta-llama-3-8b": 4096,/' glue_eval/useful_functions.py 2>/dev/null || true
    sed -i.bak 's/    z_error_threshold = 1e-2,/    z_error_threshold = 1e-2, **_kwargs,/' EvoEdit/EvoEdit_main.py 2>/dev/null || true
    sed -i.bak 's/20200501.en/20220301.en/' rome/layer_stats.py 2>/dev/null || true

    # NLTK data
    python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ('punkt', 'punkt_tab')]" 2>/dev/null || true

    export PYTHONHASHSEED="$SEED"

    ANCHOR_RESULT_DIR="$RESULT_ROOT/anchor_eval/evoedit_default_order/seed${SEED}"
    mkdir -p "$ANCHOR_RESULT_DIR"

    # Run EvoEdit with default MCF ordering (NO stream override)
    uv run python -c "
import os, sys, random, json
import numpy as np
import torch

seed = $SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.argv = [
    'experiments.evaluate',
    '--alg_name=EvoEdit',
    '--model_name=$MODEL_NAME',
    '--hparams_fname=Llama3-8B.json',
    '--ds_name=mcf',
    '--dataset_size_limit=$TARGET_EDITS',
    '--num_edits=100',
    '--downstream_eval_steps=0',
    '--save_every=1000',
    '--conserve_memory',
    '--forgetting_eval_interval=0',
]

with open('experiments/evaluate.py', 'r') as f:
    source = f.read()

# Patch 1: Model loading — resolve to local/S3/HF cache
_model_load_original = '        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()\n        tok = AutoTokenizer.from_pretrained(model_name)'
_model_load_patched = '        import sys as _sys; _sys.path.insert(0, \"$PROJECT_DIR/src/util\")\n        from model_resolve import resolve_model_path as _resolve_model\n        _resolved = _resolve_model(model_name)\n        print(f\"  [MODEL] Resolved: {model_name} -> {_resolved}\")\n        model = AutoModelForCausalLM.from_pretrained(_resolved).cuda()\n        tok = AutoTokenizer.from_pretrained(_resolved)'
assert _model_load_original in source, 'Model load anchor not found'
source = source.replace(_model_load_original, _model_load_patched, 1)

# Patch 2: CUDA line
source = source.replace('#os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"', '# CUDA managed by runner')
source = source.replace('os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"', '# CUDA managed by runner')

# Patch 3: P-cache
_p_compute_original = '''    if alg_name == \"AlphaEdit\" or alg_name == \"EvoEdit\":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, \"null_space_project.pt\")'''
_p_compute_patched = '''    if alg_name == \"AlphaEdit\" or alg_name == \"EvoEdit\":
        import os as _pos
        if _pos.path.exists(\"null_space_project.pt\"):
            P = torch.load(\"null_space_project.pt\", map_location=\"cpu\")
            print(f\"  [P-CACHE] Loaded null_space_project.pt: {P.shape}\")
        else:
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, \"null_space_project.pt\")
            print(\"  [P-CACHE] Computed and saved null_space_project.pt\")'''
if _p_compute_original in source:
    source = source.replace(_p_compute_original, _p_compute_patched, 1)

# Patch 4: Override RESULTS_DIR
_globals_import = 'from util.globals import *'
if _globals_import in source:
    source = source.replace(
        _globals_import,
        _globals_import + '\nRESULTS_DIR = Path(\"$ANCHOR_RESULT_DIR\")\n',
        1,
    )

# Patch 5: Lightweight checkpoint save
_save_original = '''                try:
                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size=\"5GB\"
                    )
                    tok.save_pretrained(ckpt_dir)'''
_save_patched = '''                try:
                    _layer_weights = {}
                    _all_params = dict(model.named_parameters())
                    for _layer_idx in hparams.layers:
                        _rkey = hparams.rewrite_module_tmp.format(_layer_idx) + \".weight\"
                        if _rkey in _all_params:
                            _layer_weights[_rkey] = _all_params[_rkey].data.cpu()
                    torch.save(_layer_weights, str(ckpt_dir / \"model_weights.pt\"))
                    print(f\"  Saved {len(_layer_weights)} layer weights to {ckpt_dir / 'model_weights.pt'}\")'''
if _save_original in source:
    source = source.replace(_save_original, _save_patched, 1)

# NO dataset override — use default MCF sequential ordering (matches paper)
print('[Anchor] Running with default MCF sequential ordering (case_id 0,1,2,...)')
print('[Anchor] evaluate.py patched: model resolve, results dir, P-cache, checkpoints')

exec(compile(source, 'experiments/evaluate.py', 'exec'),
     {'__name__': '__main__', '__file__': 'experiments/evaluate.py'})
"

    echo ""
    echo "EvoEdit run complete. Now evaluating checkpoints with dual metrics..."
    echo ""

    # Find the run directory and evaluate the 10K checkpoint
    RUN_DIR=$(find "$ANCHOR_RESULT_DIR/EvoEdit" -maxdepth 1 -name "run_*" | sort | tail -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "ERROR: No run directory found in $ANCHOR_RESULT_DIR/EvoEdit"
        exit 1
    fi

    CKPT_10K="$RUN_DIR/checkpoints/edits_$(printf '%06d' $TARGET_EDITS)"
    if [[ ! -f "$CKPT_10K/model_weights.pt" ]]; then
        echo "ERROR: No checkpoint at $CKPT_10K"
        echo "  Available:"
        ls "$RUN_DIR/checkpoints/" 2>/dev/null || echo "  (none)"
        exit 1
    fi

    cd "$PROJECT_DIR"
    uv run python scripts/eval_anchor_evoedit.py \
        --edits "$TARGET_EDITS" \
        --seed "$SEED" \
        --ordering "default_mcf" \
        --checkpoint_dir "$CKPT_10K" \
        --model_name "$MODEL_NAME"

    echo ""
    echo "================================================================"
    echo "Anchor reproduction complete."
    echo "  Run results:  $ANCHOR_RESULT_DIR"
    echo "  Eval results: $RESULT_ROOT/anchor_eval/"
    echo "================================================================"
    ;;

  compare)
    # ------------------------------------------------------------------
    # MODE 3: Compare all existing checkpoints at multiple edit counts
    # ------------------------------------------------------------------
    echo ""
    echo "Comparing metrics across edit counts..."
    echo ""

    for N in 1000 2000 5000 10000; do
        if (( N > EDITS )); then
            continue
        fi
        echo "--- $N edits ---"
        STREAM_PATH="${RESULT_ROOT}/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"

        COMPARE_ARGS=(
            "$PROJECT_DIR/scripts/compare_eval_metrics.py"
            --edits "$N"
            --model_name "$MODEL_NAME"
            --label "EvoEdit@${N}"
        )

        if [[ -f "$STREAM_PATH" ]]; then
            COMPARE_ARGS+=(--stream_path "$STREAM_PATH")
        fi

        # Try to find checkpoint
        CKPT_DIR="$HOME/.cache/evoedit_anchor/${ORDERING}_seed${SEED}/edits_$(printf '%06d' $N)"
        if [[ -f "$CKPT_DIR/model_weights.pt" ]]; then
            COMPARE_ARGS+=(--checkpoint_dir "$CKPT_DIR")
        else
            echo "  Downloading checkpoint for $N edits..."
            S3_CKPT="s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/results/evoedit/${ORDERING}/seed${SEED}/10000edits/EvoEdit/run_000/checkpoints/edits_$(printf '%06d' $N)/"
            mkdir -p "$CKPT_DIR"
            aws s3 cp "${S3_CKPT}model_weights.pt" "$CKPT_DIR/model_weights.pt" 2>/dev/null || true
            aws s3 cp "${S3_CKPT}meta.json" "$CKPT_DIR/meta.json" 2>/dev/null || true

            if [[ -f "$CKPT_DIR/model_weights.pt" ]]; then
                COMPARE_ARGS+=(--checkpoint_dir "$CKPT_DIR")
            else
                echo "  WARNING: Could not download checkpoint for $N edits, skipping"
                continue
            fi
        fi

        OUT_FILE="$RESULT_ROOT/anchor_eval/compare_${ORDERING}_s${SEED}_${N}edits.json"
        mkdir -p "$(dirname "$OUT_FILE")"
        COMPARE_ARGS+=(--output "$OUT_FILE")

        uv run python "${COMPARE_ARGS[@]}"
        echo ""
    done
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {eval|run|compare} [EDITS] [SEED] [ORDERING]"
    exit 1
    ;;
esac
