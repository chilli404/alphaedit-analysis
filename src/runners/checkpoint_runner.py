#!/usr/bin/env python3
"""
Checkpoint-Based Failure Curve Runner for AlphaEdit / MEMIT.

Enables long-running failure curve experiments (2000→10000 edits) to survive
8-hour SkyPilot cluster limits by saving model state at milestones and
resuming from checkpoints in subsequent cluster runs.

Implementation approach:
  Source-injection pattern — injects checkpoint save/load/skip logic into
  evaluate.py at known anchor points (pinned commit b84624f).

Injection points:
  1. BEFORE the main edit loop: load checkpoint (restore model weights + cache_c)
  2. BEFORE the per-batch edit call: skip guard (skip already-processed batches)
  3. AFTER the per-batch edit call: save checkpoint at interval boundaries

Checkpoint contents:
  checkpoints/{alg_name}/seed{seed}/batch_{N}/
      metadata.json          — batch index, total edits, timestamp
      model_weights.pt       — state dict for edited layers only (~560MB)
      cache_c.pt             — covariance cache (AlphaEdit only, ~320MB)

Checkpoint dir resolution:
  1. --checkpoint_dir if provided
  2. /s3-data/continual-learning/alphaedit/checkpoints/ if exists
  3. ~/.cache/alphaedit_checkpoints/

Usage:
    python src/checkpoint_runner.py \\
        --seed 42 \\
        --alg_name AlphaEdit \\
        --ds_name mcf \\
        --dataset_size_limit 5000 \\
        --num_edits 100 \\
        --start_from_batch 0 \\
        --save_interval 10 \\
        --downstream_eval_steps 10 \\
        --conserve_memory
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Add src/util to path for shared utilities
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from model_download import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file, build_order_shuffle_injection, SHUFFLE_ANCHOR
from dataset_fingerprint import build_fingerprint_injection
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root


# --- Source anchors from evaluate.py at commit b84624f ---

# Anchor for the main edit loop (inject checkpoint load BEFORE this)
LOOP_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'

# Anchor for per-batch edit timing (inject skip guard BEFORE this)
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'

# Anchor for after the entire if/elif/else edit chain (inject checkpoint save BEFORE this)
POST_EDIT_ANCHOR = '        exec_time = time() - start'

# CUDA patch target
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def resolve_checkpoint_dir(explicit_dir: str | None, alg_name: str, seed: int, order_id: int = 0) -> Path:
    """Resolve the checkpoint directory in priority order.

    If explicit_dir is provided, use it as-is (the caller sets the full path).
    Otherwise, auto-resolve using CHECKPOINT_ROOT env var:
        Ordering runs (order_id > 0):  {CHECKPOINT_ROOT}/comparison_ordered/{alg}/seed{N}/order{M}/
        Standard failure curve:        {CHECKPOINT_ROOT}/failure_curve/{alg}/seed{N}/
    """
    if explicit_dir:
        return Path(explicit_dir)

    base = get_checkpoint_root()

    if order_id > 0:
        return base / "comparison_ordered" / alg_name / f"seed{seed}" / f"order{order_id}"
    else:
        return base / "failure_curve" / alg_name / f"seed{seed}"


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Find the latest checkpoint batch in the directory.

    Returns (batch_idx, batch_dir) or None if no checkpoints exist.
    """
    if not ckpt_dir.exists():
        return None

    batch_dirs = sorted(
        [d for d in ckpt_dir.glob("batch_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    if not batch_dirs:
        return None

    # Find the highest batch number with a valid metadata.json
    for batch_dir in reversed(batch_dirs):
        metadata_file = batch_dir / "metadata.json"
        if metadata_file.exists():
            try:
                batch_idx = int(batch_dir.name.split("_")[1])
                return (batch_idx, batch_dir)
            except (ValueError, IndexError):
                continue

    return None


def _resolve_results_dir(args: argparse.Namespace) -> Path | None:
    """Build the results directory path for evaluate.py output.

    Structure: {project_root}/results/{experiment}/seed{seed}/{edits}edits[/order{N}]

    evaluate.py appends {AlgName}/run_000/ to RESULTS_DIR, so final path is:
        results/{experiment}/seed{seed}/{edits}edits[/order{N}]/{AlgName}/run_000/
    """
    if args.results_dir:
        return Path(args.results_dir)

    # Auto-construct from experiment name env var + args
    experiment = os.environ.get("EXPERIMENT_NAME", "")
    if not experiment:
        return None  # Fall back to vendor/AlphaEdit/results/ (legacy behavior)

    results_base = get_result_root() / experiment / f"seed{args.seed}"
    results_base = results_base / f"{args.dataset_size_limit}edits"

    # Include order subdirectory for ordering experiments
    if "ordered" in experiment or "order" in experiment:
        results_base = results_base / f"order{args.order_id}"

    return results_base


def build_checkpoint_script(
    seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    start_from_batch: int,
    save_interval: int,
    checkpoint_dir: str,
    fast_checkpoint: bool = False,
    eval_at_checkpoints_only: bool = False,
    order_id: int = 0,
    results_dir: str | None = None,
    result_root: str | None = None,
    dir_name: str | None = None,
) -> str:
    """
    Build an inline Python script that:
    1. Seeds all RNGs
    2. Injects checkpoint save/load/skip into evaluate.py
    3. Executes the patched evaluate.py as __main__

    Evaluation modes:
      - Normal: Evaluate all facts after every batch (slow, complete)
      - fast_checkpoint: Evaluate only edited batch after every batch (fast, partial)
      - eval_at_checkpoints_only: Evaluate all facts only at checkpoint boundaries (balanced)
    """
    argv_parts = [
        "experiments.evaluate",
        f"--alg_name={alg_name}",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
        f"--downstream_eval_steps={downstream_eval_steps}",
        "--generation_test_interval=1",
        "--skip_generation_tests",
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)

    # Build optional RESULTS_DIR override injection
    if results_dir:
        results_dir_injection = (
            f'\n_globals_import = \'from util.globals import *\'\n'
            f'assert _globals_import in source, "globals import not found in evaluate.py"\n'
            f'source = source.replace(\n'
            f'    _globals_import,\n'
            f'    _globals_import + \'\\nRESULTS_DIR = Path("{results_dir}")\\n\',\n'
            f'    1,\n'
            f')\n'
            f'print(f"  [RESULTS_DIR] Overridden to: {results_dir}")\n'
        )
    else:
        results_dir_injection = ""

    # Build optional dir_name override (for MEMIT-Seq: passes MEMIT to ALG_DICT but uses variant as dir_name)
    if dir_name:
        results_dir_injection += (
            f'\nsource = source.replace(\n'
            f'    \'dir_name=args.alg_name,\',\n'
            f'    \'dir_name="{dir_name}",\',\n'
            f'    1,\n'
            f')\n'
            f'print(f"  [DIR_NAME] Overridden to: {dir_name}")\n'
        )

    # Mega-batch eval injection (outside f-string to avoid Python 3.10 nested-quote issues)
    mega_batch_eval_injection = '''    # === MEGA-BATCH EVAL: batched multi-token scoring (injected by checkpoint_runner) ===
    def _mega_batch_eval(model, tok, records, case_result_template, num_edits, case_ids, exec_time, batch_size=8):
        # Evaluate records with batched forward passes using full multi-token scoring.
        # Produces IDENTICAL results to per-record compute_rewrite_quality_counterfact
        # (same log-prob scoring, same argmax correctness) but batches the expensive
        # model forward pass across multiple records.
        import torch as _mbe_torch
        import numpy as _mbe_np
        import json as _mbe_json
        from time import time as _mbe_time
        from pathlib import Path as _mbe_Path

        _is_llama = 'llama' in model.config._name_or_path.lower()
        _mbe_start = _mbe_time()
        _mbe_total = len(records)
        _mbe_done = 0
        _mbe_skipped = 0

        for batch_start in range(0, _mbe_total, batch_size):
            batch_records = records[batch_start:batch_start + batch_size]

            # --- Phase 1: Collect all sequences for this batch ---
            all_sequences = []
            record_meta = []

            for record in batch_records:
                out_file = _mbe_Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    record_meta.append(None)
                    _mbe_skipped += 1
                    continue

                subject = record["requested_rewrite"]["subject"]
                target_new = record["requested_rewrite"]["target_new"]["str"]
                target_true = record["requested_rewrite"]["target_true"]["str"]

                rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
                paraphrase_prompts = record["paraphrase_prompts"]
                neighborhood_prompts = record["neighborhood_prompts"]

                prefixes = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
                which_correct = (
                    [0] * len(rewrite_prompts)
                    + [0] * len(paraphrase_prompts)
                    + [1] * len(neighborhood_prompts)
                )

                a_tok = tok(f" {target_new}")["input_ids"]
                b_tok = tok(f" {target_true}")["input_ids"]
                if _is_llama:
                    a_tok = a_tok[1:]
                    b_tok = b_tok[1:]

                prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
                if _is_llama:
                    prefix_lens = [l - 1 for l in prefix_lens]

                seqs = [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]]
                seq_start_idx = len(all_sequences)
                all_sequences.extend(seqs)

                record_meta.append({
                    "record": record,
                    "out_file": out_file,
                    "a_tok": a_tok,
                    "b_tok": b_tok,
                    "prefix_lens": prefix_lens,
                    "which_correct": which_correct,
                    "n_prefixes": len(prefixes),
                    "n_rewrite": len(rewrite_prompts),
                    "n_paraphrase": len(paraphrase_prompts),
                    "n_neighborhood": len(neighborhood_prompts),
                    "seq_start_idx": seq_start_idx,
                    "n_seqs": len(seqs),
                })

            # --- Phase 2: Forward pass ---
            if not all_sequences:
                _mbe_done += len(batch_records)
                continue

            prompt_tok = tok(all_sequences, padding=True, return_tensors="pt").to("cuda")
            with _mbe_torch.no_grad():
                logits = model(**prompt_tok).logits

            if _is_llama:
                logits = logits[:, 1:, :]

            # --- Phase 3: Score each record using exact vendor logic ---
            for meta in record_meta:
                if meta is None:
                    continue

                record = meta["record"]
                start_idx = meta["seq_start_idx"]
                n_seqs = meta["n_seqs"]
                a_tok = meta["a_tok"]
                b_tok = meta["b_tok"]
                prefix_lens = meta["prefix_lens"]
                which_correct = meta["which_correct"]
                choice_a_len = len(a_tok)
                choice_b_len = len(b_tok)

                rec_logits = logits[start_idx:start_idx + n_seqs]

                probs = _mbe_np.zeros((n_seqs,), dtype=_mbe_np.float32)
                targets_correct = []

                for i in range(n_seqs):
                    cur_len = choice_a_len if i % 2 == 0 else choice_b_len
                    for j in range(cur_len):
                        cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                        probs[i] += -_mbe_torch.nn.functional.log_softmax(
                            rec_logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
                        )[cur_tok].item()
                    probs[i] /= cur_len

                    if (which_correct[i // 2] == 0 and i % 2 == 0) or (
                        which_correct[i // 2] == 1 and i % 2 == 1
                    ):
                        correct = True
                        for j in range(cur_len):
                            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                            if rec_logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                                correct = False
                                break
                        targets_correct.append(correct)

                ret_probs = [
                    {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
                    for i in range(0, n_seqs, 2)
                ]
                n_rw = meta["n_rewrite"]
                n_para = meta["n_paraphrase"]
                n_neigh = meta["n_neighborhood"]
                cutoffs = [0, n_rw, n_rw + n_para, n_rw + n_para + n_neigh]
                ret_corrects_flat = targets_correct
                ret_corrects = [
                    ret_corrects_flat[cutoffs[i]:cutoffs[i+1]]
                    for i in range(3)
                ]

                post = {
                    "rewrite_prompts_probs": ret_probs[:n_rw],
                    "rewrite_prompts_correct": ret_corrects[0],
                    "paraphrase_prompts_probs": ret_probs[n_rw:n_rw + n_para],
                    "paraphrase_prompts_correct": ret_corrects[1],
                    "neighborhood_prompts_probs": ret_probs[n_rw + n_para:],
                    "neighborhood_prompts_correct": ret_corrects[2],
                }

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": post,
                }
                with open(meta["out_file"], "w") as f:
                    _mbe_json.dump(metrics, f, indent=1)

            del prompt_tok, logits
            _mbe_torch.cuda.empty_cache()

            _mbe_done += len(batch_records)
            _elapsed = _mbe_time() - _mbe_start
            _rate = _mbe_done / _elapsed if _elapsed > 0 else 0
            if _mbe_done % (batch_size * 4) < batch_size or _mbe_done >= _mbe_total:
                print(f"  [MEGA-BATCH EVAL] {_mbe_done}/{_mbe_total} records "
                      f"({_mbe_skipped} skipped, {_rate:.1f} rec/s, "
                      f"{_elapsed:.0f}s elapsed)")

        print(f"  [MEGA-BATCH EVAL] Complete: {_mbe_total} records in {_mbe_time() - _mbe_start:.1f}s")

    # --- Call mega-batch eval or fall through to original loop ---
    if _do_final_eval:
        _records_to_eval = list(ds)
        # === CHECKPOINT: fast mode - filter to batch records only (injected) ===
        if _ckpt_fast_mode:
            _records_to_eval = [r for r in ds if r["case_id"] in case_ids]
        _mega_batch_eval(edited_model, tok, _records_to_eval, case_result_template, num_edits, case_ids, exec_time)
    # === END mega-batch eval ===
    _eval_skipped = 0
    for record in ds:
        break  # Mega-batch handles all eval above; skip vendor fallback loop
        # === CHECKPOINT: skip entire evaluation if _do_final_eval is False (injected) ===
        if not _do_final_eval:
            break
        # === CHECKPOINT: skip cases already evaluated (resume-safe) ===
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            _eval_skipped += 1
            continue
        if _eval_skipped > 0:
            print(f"  [CHECKPOINT] Skipped {_eval_skipped} already-evaluated cases, resuming from case {record['case_id']}")
            _eval_skipped = 0
        # === END eval resume guard ==='''

    script = textwrap.dedent(f"""\
import os, sys, random, json
import numpy as np
import torch

# 1. Seed all sources of randomness
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# 2. Set sys.argv
sys.argv = {argv_str}

# 3. Checkpoint parameters
_ckpt_start_batch = {start_from_batch}
_ckpt_save_interval = {save_interval}
_ckpt_dir = "{checkpoint_dir}"
_ckpt_alg_name = "{alg_name}"
_ckpt_seed = {seed}
_ckpt_num_edits = {num_edits}
_ckpt_fast_mode = {fast_checkpoint}
_ckpt_eval_at_checkpoints_only = {eval_at_checkpoints_only}
_ckpt_dataset_size_limit = {dataset_size_limit}

def _ckpt_save(cnt, model, cache_c, hparams, alg_name):
    \"\"\"Save model weights and cache_c at checkpoint boundary.\"\"\"
    import json, shutil
    from pathlib import Path
    from datetime import datetime, timezone

    batch_dir = Path(_ckpt_dir) / f"batch_{{cnt}}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save only the edited layer weights (much smaller than full model)
    layer_weights = {{}}
    for layer_idx in hparams.layers:
        for key in ["mlp.down_proj.weight", "mlp.up_proj.weight"]:
            param_name = f"model.layers.{{layer_idx}}.{{key}}"
            param = dict(model.named_parameters()).get(param_name)
            if param is not None:
                layer_weights[param_name] = param.data.cpu()

    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    # Save cache_c (AlphaEdit only)
    if alg_name == "AlphaEdit" and cache_c is not None:
        torch.save(cache_c.cpu(), str(batch_dir / "cache_c.pt"))

    # Save metadata
    metadata = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _ckpt_num_edits,
        "alg_name": alg_name,
        "seed": _ckpt_seed,
        "num_edits_per_batch": _ckpt_num_edits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }}
    with open(str(batch_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [CHECKPOINT] Saved batch {{cnt}} ({{(cnt+1) * _ckpt_num_edits}} total edits) -> {{batch_dir}}")

def _ckpt_load(model, hparams, alg_name):
    \"\"\"Load model weights (and cache_c for AlphaEdit) from the start_from_batch checkpoint.\"\"\"
    from pathlib import Path

    if _ckpt_start_batch <= 0:
        return None  # No checkpoint to load

    batch_dir = Path(_ckpt_dir) / f"batch_{{_ckpt_start_batch - 1}}"
    if not batch_dir.exists():
        print(f"  [CHECKPOINT] WARNING: Expected checkpoint at {{batch_dir}} not found. Starting from scratch.")
        return None

    # Load model weights
    weights_file = batch_dir / "model_weights.pt"
    if weights_file.exists():
        layer_weights = torch.load(str(weights_file), map_location="cuda")
        param_dict = dict(model.named_parameters())
        loaded_count = 0
        for param_name, param_data in layer_weights.items():
            if param_name in param_dict:
                param_dict[param_name].data.copy_(param_data)
                loaded_count += 1
        print(f"  [CHECKPOINT] Loaded {{loaded_count}} parameter tensors from {{weights_file}}")
    else:
        print(f"  [CHECKPOINT] WARNING: No model_weights.pt in {{batch_dir}}")

    # Load cache_c (AlphaEdit only)
    cache_c_loaded = None
    if alg_name == "AlphaEdit":
        cache_file = batch_dir / "cache_c.pt"
        if cache_file.exists():
            cache_c_loaded = torch.load(str(cache_file), map_location="cpu")
            print(f"  [CHECKPOINT] Loaded cache_c from {{cache_file}} (shape: {{cache_c_loaded.shape}})")
        else:
            print(f"  [CHECKPOINT] WARNING: No cache_c.pt in {{batch_dir}} (AlphaEdit will start with zero cache)")

    print(f"  [CHECKPOINT] Resuming from batch {{_ckpt_start_batch}} ({{_ckpt_start_batch * _ckpt_num_edits}} edits already applied)")
    return cache_c_loaded

def _ckpt_should_skip(cnt):
    \"\"\"Return True if this batch was already processed (before start_from_batch).\"\"\"
    return cnt < _ckpt_start_batch

def _ckpt_should_save(cnt):
    \"\"\"Return True if we should save a checkpoint at this batch.\"\"\"
    return (cnt + 1) % _ckpt_save_interval == 0

# 4. Read evaluate.py source
with open("experiments/evaluate.py", "r") as f:
    source = f.read()

# 5. Patch CUDA_VISIBLE_DEVICES
cuda_patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
assert cuda_patch_target in source, (
    "CUDA_VISIBLE_DEVICES patch target not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
source = source.replace(
    cuda_patch_target,
    '# CUDA_VISIBLE_DEVICES managed by checkpoint_runner',
)

# 5a. Override RESULTS_DIR to write to project results/ directory
{results_dir_injection}
# 5b. Patch P/cache_c initialization to not depend on hardcoded model name whitelist.
# The upstream code only initializes P for model names in a fixed list. If hparams.model_name
# doesn't match, P and cache_c are never created and the edit call crashes with
# TypeError: 'NoneType' object is not subscriptable.
# Fix: add an else branch that infers dimensions from W_out generically.
_p_init_anchor = '''        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out'''
assert _p_init_anchor in source, (
    "P initialization anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)
_p_init_patched = '''        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5"]:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        else:
            # Fallback: infer dimensions from W_out (handles any model not in whitelist)
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            if alg_name == "AlphaEdit":
                P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            print(f"  [CHECKPOINT] WARNING: model_name '{{hparams.model_name}}' not in upstream whitelist, using fallback P init (dim={{W_out.shape[1]}})")
        del W_out'''
source = source.replace(_p_init_anchor, _p_init_patched, 1)

# 5c. Inject eval results pre-sync: copy partial results from S3/previous run into the new run_dir
# so that (a) the already_finished check in the edit loop skips fully-evaluated batches and
# (b) the eval resume guard at the bottom skips individual cases.
_presync_anchor = '    print(f"Results will be stored at {{run_dir}}")'
assert _presync_anchor in source, (
    "Pre-sync anchor (run_dir print) not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)
_presync_injection = '''    print(f"Results will be stored at {{run_dir}}")
    # === CHECKPOINT: pre-sync partial eval results from previous runs (injected) ===
    if _ckpt_start_batch > 0:
        import glob as _glob_mod
        _s3_results_base = Path("{result_root}") / "failure_curve_checkpointed"
        _s3_eval_dir = _s3_results_base / f"seed{{_ckpt_seed}}" / f"{{_ckpt_dataset_size_limit}}edits" / dir_name / "run_000"
        if _s3_eval_dir.exists():
            _existing_evals = list(_s3_eval_dir.glob("*_edits-case_*.json"))
            if _existing_evals:
                import shutil as _shutil_mod
                _synced = 0
                for _ef in _existing_evals:
                    _dest = run_dir / _ef.name
                    if not _dest.exists():
                        _shutil_mod.copy2(str(_ef), str(_dest))
                        _synced += 1
                print(f"  [CHECKPOINT] Pre-synced {{_synced}} eval results from S3 ({{len(_existing_evals)}} total in source)")
        else:
            print(f"  [CHECKPOINT] No S3 eval results found at {{_s3_eval_dir}}")
    # === END pre-sync ==='''
source = source.replace(_presync_anchor, _presync_injection, 1)

# 5d. Inject order shuffle + dataset fingerprint before loop
loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert loop_anchor in source, (
    "Loop anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

_order_id = {order_id}
if _order_id > 0:
    _shuffle_code = (
        f'    # === ORDER SHUFFLE: shuffle dataset with order_id={{_order_id}} (injected) ===\\n'
        f'    import random as _order_rng_module\\n'
        f'    _order_rng = _order_rng_module.Random({{_order_id}})\\n'
        f'    _shuffled_indices = list(range(len(ds)))\\n'
        f'    _order_rng.shuffle(_shuffled_indices)\\n'
        f'    ds.data = [ds.data[i] for i in _shuffled_indices]\\n'
        f'    print("ORDER SHUFFLE: shuffled " + str(len(ds)) + " records with order_id={{_order_id}}")\\n'
        f'    # === END order shuffle ===\\n'
    )
    source = source.replace(loop_anchor, _shuffle_code + loop_anchor, 1)

# Inject fingerprint
_fp_code = '''    # === FINGERPRINT: compute dataset fingerprint (injected) ===
    import hashlib as _fp_hashlib
    import json as _fp_json
    _fp_case_ids = [r["case_id"] for r in ds]
    _fp_id_bytes = _fp_json.dumps(_fp_case_ids, separators=(",", ":")).encode("utf-8")
    _fp_sha256 = _fp_hashlib.sha256(_fp_id_bytes).hexdigest()
    print(f"  [FINGERPRINT] Dataset: {{len(ds)}} records, SHA-256: {{_fp_sha256[:16]}}...")
    print(f"  [FINGERPRINT] Order ID: ''' + str(_order_id) + ''', first 5 IDs: {{_fp_case_ids[:5]}}")
    _fp_ordering_path = run_dir / "edit_ordering.json"
    _fp_ordering = {{
        "case_ids_ordered": _fp_case_ids,
        "n_records": len(ds),
        "order_id": ''' + str(_order_id) + ''',
        "fingerprint": {{
            "sha256": _fp_sha256,
            "n_records": len(ds),
            "first_5_ids": _fp_case_ids[:5],
            "last_5_ids": _fp_case_ids[-5:] if len(_fp_case_ids) >= 5 else _fp_case_ids,
        }},
    }}
    with open(str(_fp_ordering_path), "w") as _fp_f:
        _fp_json.dump(_fp_ordering, _fp_f, indent=2)
    # === END fingerprint ===
'''
source = source.replace(loop_anchor, _fp_code + loop_anchor, 1)

# 6. Inject checkpoint LOAD before the main edit loop
load_injection = '''    # === CHECKPOINT: load state from previous run (injected) ===
    exec_time = 0
    edited_model = model
    _ckpt_cache_c_loaded = None
    if _ckpt_start_batch > 0 and '_ckpt_load' in globals():
        _ckpt_cache_c_loaded = _ckpt_load(model, hparams, alg_name)
        if _ckpt_cache_c_loaded is not None and alg_name == "AlphaEdit":
            cache_c = _ckpt_cache_c_loaded
    # === END checkpoint load ===
'''
source = source.replace(
    loop_anchor,
    load_injection + loop_anchor,
    1,
)

# 7. Inject SKIP guard before the per-batch edit call
pre_anchor = '        start = time()\\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
assert pre_anchor in source, (
    "Pre-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

skip_injection = '''        # === CHECKPOINT: skip already-processed batches (injected) ===
        if '_ckpt_should_skip' in globals() and _ckpt_should_skip(cnt):
            cnt += 1
            continue
        # === END checkpoint skip ===
'''
source = source.replace(
    pre_anchor,
    skip_injection + pre_anchor,
    1,
)

# 8. Inject checkpoint SAVE after the entire if/elif/else edit chain
post_anchor = '        exec_time = time() - start'
assert post_anchor in source, (
    "Post-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

save_injection = '''        # === CHECKPOINT: save at interval boundaries (injected) ===
        if '_ckpt_should_save' in globals() and _ckpt_should_save(cnt):
            _ckpt_save(cnt, model, cache_c if alg_name == "AlphaEdit" else None, hparams, alg_name)
        # === END checkpoint save ===
'''
source = source.replace(
    post_anchor,
    save_injection + post_anchor,
    1,
)

# 9. Inject CHECKPOINT-ONLY EVAL guard (skip entire evaluation for non-checkpoint batches)
# The evaluation section in evaluate.py is OUTSIDE the edit loop (runs once after all edits).
# We inject a flag check that prevents evaluation when the final batch is not a checkpoint boundary.
eval_start_anchor = '    # torch.save(hs, "post_edit_hs_memit.pt")\\n    start = time()'
assert eval_start_anchor in source, (
    "Evaluation start anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

checkpoint_eval_skip = '''    # torch.save(hs, "post_edit_hs_memit.pt")
    # === CHECKPOINT: skip evaluation if last batch is not a checkpoint boundary (injected) ===
    _do_final_eval = True
    if _ckpt_eval_at_checkpoints_only and not _ckpt_should_save(cnt - 1):
        _do_final_eval = False
        print(f"  [CHECKPOINT] Skipping final evaluation (batch {{cnt-1}} not at checkpoint boundary)")
        print(f"  [CHECKPOINT] Evaluation will run when resumed and a checkpoint boundary is reached.")
    # === END checkpoint eval skip ===
    start = time()'''
source = source.replace(
    eval_start_anchor,
    checkpoint_eval_skip,
    1,
)

# 10. Inject MEGA-BATCH evaluation to replace per-record eval loop.
# Replicates exact multi-token scoring from test_batch_prediction but batches
# multiple records per forward pass for ~10-30x speedup.
eval_anchor = '    for record in ds:\\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
assert eval_anchor in source, (
    "Evaluation loop anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

mega_batch_eval_injection = {repr(mega_batch_eval_injection)}
source = source.replace(
    eval_anchor,
    mega_batch_eval_injection,
    1,
)

# 11. Verify all injections succeeded
assert "CHECKPOINT: load state" in source, "Load injection failed"
assert "CHECKPOINT: skip already-processed" in source, "Skip injection failed"
assert "CHECKPOINT: save at interval" in source, "Save injection failed"
assert "CHECKPOINT: skip evaluation if last batch is not" in source, "Checkpoint-only eval injection failed"
assert "CHECKPOINT: skip entire evaluation if _do_final_eval" in source, "Final eval guard injection failed"
assert "CHECKPOINT: fast mode" in source, "Fast eval injection failed"
assert "CHECKPOINT: skip cases already evaluated" in source, "Eval resume injection failed"

# 12. Execute
exec(compile(source, "experiments/evaluate.py", "exec"),
     {{
         "__name__": "__main__",
         "__file__": "experiments/evaluate.py",
         "_ckpt_start_batch": _ckpt_start_batch,
         "_ckpt_save_interval": _ckpt_save_interval,
         "_ckpt_dir": _ckpt_dir,
         "_ckpt_alg_name": _ckpt_alg_name,
         "_ckpt_seed": _ckpt_seed,
         "_ckpt_num_edits": _ckpt_num_edits,
         "_ckpt_fast_mode": _ckpt_fast_mode,
         "_ckpt_eval_at_checkpoints_only": _ckpt_eval_at_checkpoints_only,
         "_ckpt_dataset_size_limit": _ckpt_dataset_size_limit,
         "_ckpt_save": _ckpt_save,
         "_ckpt_load": _ckpt_load,
         "_ckpt_should_skip": _ckpt_should_skip,
         "_ckpt_should_save": _ckpt_should_save,
     }})

# 13. Final summary
print(f"\\n=== Checkpoint runner complete ===")
print(f"  Algorithm: {{_ckpt_alg_name}}")
print(f"  Resumed from batch: {{_ckpt_start_batch}}")
print(f"  Save interval: every {{_ckpt_save_interval}} batches")
print(f"  Fast checkpoint: {{_ckpt_fast_mode}}")
print(f"  Eval at checkpoints only: {{_ckpt_eval_at_checkpoints_only}}")
print(f"  Checkpoint dir: {{_ckpt_dir}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the checkpointed experiment."""
    alphaedit_root = get_alphaedit_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    # Resolve model path
    model_name = resolve_model_path(args.model_name)

    # Validate anchors exist in the source before launching
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for anchor_name, anchor_str in [
        ("LOOP_ANCHOR", LOOP_ANCHOR),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
    ]:
        if anchor_str not in eval_source:
            print(f"ERROR: {anchor_name} not found in evaluate.py.")
            print("  The upstream code has diverged from pinned commit b84624f.")
            sys.exit(1)

    # Resolve checkpoint directory
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.alg_name, args.seed, args.order_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve results directory (where evaluate.py writes per-case JSONs)
    results_dir_override = _resolve_results_dir(args)

    # Determine start_from_batch
    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        # Auto-detect from latest checkpoint
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = latest[0] + 1
            # Cap at total_batches (checkpoint dir may have checkpoints from longer runs)
            if start_from_batch > total_batches:
                start_from_batch = total_batches
                print(f"  Auto-detected: checkpoint at batch {latest[0]} exceeds target ({total_batches} batches). Will run eval only.")
            else:
                print(f"  Auto-detected: resume from batch {start_from_batch} (checkpoint at batch {latest[0]})")
        else:
            start_from_batch = 0
            print("  No existing checkpoints found. Starting from batch 0.")

    # For MEMIT-Seq variants, pass MEMIT as the algorithm (for ALG_DICT lookup)
    # but keep the full variant name for dir_name (results directory naming)
    eval_alg_name = args.alg_name
    dir_name_override = None
    if args.alg_name.startswith("MEMIT-Seq"):
        eval_alg_name = "MEMIT"
        dir_name_override = args.alg_name

    script = build_checkpoint_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name=eval_alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        start_from_batch=start_from_batch,
        save_interval=args.save_interval,
        checkpoint_dir=str(ckpt_dir),
        fast_checkpoint=args.fast_checkpoint,
        eval_at_checkpoints_only=args.eval_at_checkpoints_only,
        order_id=args.order_id,
        results_dir=str(results_dir_override) if results_dir_override else None,
        result_root=str(get_result_root()),
        dir_name=dir_name_override,
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Determine evaluation mode description
    if args.eval_at_checkpoints_only:
        eval_mode = f"Milestone only (every {args.save_interval} batches)"
    elif args.fast_checkpoint:
        eval_mode = "Fast (edited batch only)"
    else:
        eval_mode = "Full (all facts every batch)"

    print(f"{'=' * 70}")
    print("Checkpoint-Based Failure Curve Runner")
    print(f"  Algorithm:       {args.alg_name}")
    print(f"  Dataset:         {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits/batch: {args.num_edits}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Resume from:     batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:   every {args.save_interval} batches")
    print(f"  Evaluation:      {eval_mode}")
    print(f"  Seed:            {args.seed}")
    print(f"  CUDA:            device {args.cuda_device}")
    print(f"  Model:           {args.model_name}")
    print(f"  Checkpoint dir:  {ckpt_dir}")
    if results_dir_override:
        print(f"  Results dir:     {results_dir_override}")
    else:
        print(f"  Results dir:     {alphaedit_root / 'results'} (legacy)")
    print(f"  Started:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Checkpoint run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Detect run_dir and run_id created by evaluate.py
    from seeded_runner import find_latest_run_dir
    project_root = get_project_root()
    run_dir_rel, run_id = find_latest_run_dir(args.alg_name)

    # Find latest checkpoint to determine where this segment ended
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    ended_at_batch = latest_ckpt[0] if latest_ckpt else total_batches - 1

    # Record metadata as JSONL (append mode — one line per segment/resume)
    # This way multiple resumes don't overwrite each other.
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "order_id": args.order_id,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "alphaedit_commit": "b84624f",
        "eval_config_hash": hash_eval_config(),
        "cuda_device": args.cuda_device,
        "experiment": "failure_curve_ckpt",
        "algorithm": args.alg_name,
        "dataset": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "run_dir": run_dir_rel,
        "run_id": run_id,
        "checkpoint_dir": str(ckpt_dir),
        "resumed_from_batch": start_from_batch if start_from_batch > 0 else None,
        "ended_at_batch": ended_at_batch,
        "params": {
            "model_name": args.model_name,
            "hparams_fname": args.hparams_fname,
            "save_interval": args.save_interval,
            "fast_checkpoint": args.fast_checkpoint,
            "eval_at_checkpoints_only": args.eval_at_checkpoints_only,
            "downstream_eval_steps": args.downstream_eval_steps,
        },
    }

    results_dir = get_result_root()
    metadata_dir = results_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    # JSONL: append one line per segment so multi-resume runs build a history
    metadata_file = metadata_dir / f"run_seed{args.seed}_{args.alg_name}_ckpt_{args.dataset_size_limit}.jsonl"
    with open(metadata_file, "a") as f:
        f.write(json.dumps(metadata) + "\n")
    print(f"Metadata appended to: {metadata_file}")

    print(f"\n{'=' * 70}")
    print("Checkpoint run completed.")
    print(f"  Finished:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Results:     {alphaedit_root / 'results' / args.alg_name}")
    if run_id:
        print(f"  Run ID:      {run_id}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Segment:     batch {start_from_batch} → {ended_at_batch}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint-based failure curve runner for AlphaEdit/MEMIT"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    # Experiment parameters
    parser.add_argument("--alg_name", required=True,
                        help="Algorithm name: AlphaEdit, MEMIT, or MEMIT-Seq-lp{X}-ld{Y}-cache{Z}")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # Checkpoint parameters
    parser.add_argument("--start_from_batch", type=int, default=-1,
                        help="Batch to resume from (-1 = auto-detect from latest checkpoint)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N batches (default: 10 = every 1000 edits)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: S3 mount or ~/.cache)")

    # Evaluation mode (mutually exclusive)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true",
                        help="Fast mode: only evaluate edited batch after each edit (partial preservation measurement)")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true",
                        help="Milestone mode: evaluate full dataset only at checkpoint boundaries (RECOMMENDED for conferences)")

    # Order sensitivity
    parser.add_argument("--order_id", type=int, default=0,
                        help="Edit ordering ID (0=canonical, >0=shuffle with Random(order_id))")

    # Output directory
    parser.add_argument("--results_dir", default=None,
                        help="Override RESULTS_DIR so evaluate.py writes directly to project results/ "
                             "(default: auto-construct from experiment name, seed, edits, order)")

    # Retention probes (for mechanism figure)
    parser.add_argument("--retention_probe_batches", type=str, default=None,
                        help="Comma-separated list of historical batch indices to re-evaluate at each checkpoint "
                             "(e.g., '0,5,10,15,20'). Enables retention-by-age metric for mechanism figure.")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
