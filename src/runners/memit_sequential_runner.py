#!/usr/bin/env python3
"""
MEMIT+SeqReg: Non-projected analogue of AlphaEdit's sequential regularization.

Scientific Question:
    Does MEMIT with AlphaEdit-like sequential regularization (Eq. 12) close
    the performance gap to AlphaEdit, or is the null-space projection P
    still necessary?

AlphaEdit Eq. 12 (projected):
    minimize ||ΔPK - R||² + λ_prev ||ΔPK_prev||² + λ_delta ||ΔP||²
    where P is the null-space projection matrix

MEMIT+SeqReg (non-projected analogue):
    minimize ||ΔK - R||² + λ_prev ||ΔK_prev||² + λ_delta ||Δ||²
    Implemented via LHS augmentation:
        lhs = α·C₀ + K_new@K_new^T + λ_prev·K_prev@K_prev^T + λ_delta·I

Key insight:
    λ_prev protects previous edits (preservation in previous-key directions)
    λ_delta minimizes overall update size (Frobenius norm)
    Both match AlphaEdit's objective structure but without projection

Setting λ_prev=0 and λ_delta=0 recovers exact original MEMIT.

Calibration settings:
    A: λ_prev=1, λ_delta=1        # Direct Eq. 12 coefficient analogue
    B: λ_prev=1, λ_delta=1e-4     # Weak ridge
    C: λ_prev=10, λ_delta=1       # Strong prev-key protection
    D: λ_prev=100, λ_delta=1      # Very strong prev-key protection

Implementation: Dual source injection (following alphaedit_stream_runner.py):
  1. Read memit_main.py, inject LHS augmentation + cache storage + norm logging
  2. Compile/exec patched memit → extract apply_memit_to_model
  3. Read evaluate.py, replace MEMIT import, inject batch counter
  4. Exec evaluate.py with patched function

Usage:
    python src/memit_sequential_runner.py \\
        --seed 42 --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \\
        --lambda_prev 1.0 --lambda_delta 1.0 \\
        --cache_strategy recent --cache_max 20 \\
        --downstream_eval_steps 10 --conserve_memory
"""

import argparse
import json
import os
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
from source_patches import patch_evaluate_file
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root


def resolve_checkpoint_dir(explicit_dir: str | None, seed: int, lambda_prev: float, lambda_delta: float, cache_max: int | None = None, ordering: str | None = None) -> Path:
    """Resolve checkpoint directory for MEMIT+SeqReg.

    Convention:
        Standard:         {CHECKPOINT_ROOT}/failure_curve/MEMIT-Seq-lp{lp}-ld{ld}-cache{cm}/seed{N}/
        Matched ordering: {CHECKPOINT_ROOT}/matched_ordering/MEMIT-Seq-lp{lp}-ld{ld}-cache{cm}/{ordering}/seed{N}/
    """
    if explicit_dir:
        return Path(explicit_dir)

    cache_max_str = str(cache_max) if cache_max is not None else "0"
    variant_name = f"MEMIT-Seq-lp{lambda_prev}-ld{lambda_delta}-cache{cache_max_str}"

    if ordering:
        return get_checkpoint_root() / "matched_ordering" / variant_name / ordering / f"seed{seed}"

    return get_checkpoint_root() / "failure_curve" / variant_name / f"seed{seed}"


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Find the latest checkpoint batch in the directory."""
    if not ckpt_dir.exists():
        return None

    batch_dirs = sorted(
        [d for d in ckpt_dir.glob("batch_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    if not batch_dirs:
        return None

    for batch_dir in reversed(batch_dirs):
        metadata_file = batch_dir / "metadata.json"
        if metadata_file.exists():
            try:
                batch_idx = int(batch_dir.name.split("_")[1])
                return (batch_idx, batch_dir)
            except (ValueError, IndexError):
                continue

    return None


# --- Source anchors (commit b84624f) ---

# evaluate.py
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
# Inject skip guard BEFORE the per-batch edit timing
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
# Inject batch increment AFTER exec_time (i.e. after the entire if/elif/else edit chain)
# so we don't break the if/elif/else structure that assigns edited_model.
POST_EDIT_ANCHOR = '        exec_time = time() - start'
MEMIT_IMPORT_ANCHOR = 'from memit.memit_main import apply_memit_to_model, get_context_templates'

# memit_main.py
SOLVE_ANCHOR = '        adj_k = torch.linalg.solve(\n            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,\n            layer_ks,\n        )'
DELTAS_ANCHOR = '            deltas[weight_name] = ('


def build_sequential_script(
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
    lambda_prev: float,
    lambda_delta: float,
    cache_strategy: str,
    cache_max: int | None,
    output_jsonl: str,
    debug_freeze_batch: int | None,
    fast_checkpoint: bool = False,
    eval_at_checkpoints_only: bool = False,
    order_id: int = 0,
    save_interval: int = 10,
    checkpoint_dir: str = "",
    start_from_batch: int = 0,
    dataset_override: str | None = None,
    eval_results_dir: str = "",
    variant_name: str = "",
) -> str:
    """
    Build inline Python script for MEMIT+SeqReg.
    Uses dual source injection: patches memit_main.py and evaluate.py.
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
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)
    cache_max_repr = repr(cache_max)

    # Injection code for memit_main.py: replaces the solve line
    solve_replacement = r'''        # === MEMIT+SeqReg: augmented solve (injected) ===
        # Compute _K_prev from cache BEFORE appending current keys
        _K_prev = None
        _kpkp_norm = 0.0
        if _memit_lambda_prev > 0 and layer in _memit_prev_cache and len(_memit_prev_cache[layer]) > 0:
            _K_prev = torch.cat(_memit_prev_cache[layer], dim=1).to(layer_ks.device).double()
            _kpkp_mat = _K_prev @ _K_prev.T
            _kpkp_norm = torch.linalg.norm(_kpkp_mat, ord='fro').item()

        # Base LHS (before augmentation)
        _lhs_base = hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T
        _base_lhs_norm = torch.linalg.norm(_lhs_base, ord='fro').item()

        # Augmented LHS
        _lhs = _lhs_base
        if _K_prev is not None:
            _lhs = _lhs + _memit_lambda_prev * (_K_prev @ _K_prev.T)
        if _memit_lambda_delta > 0:
            _lhs = _lhs + _memit_lambda_delta * torch.eye(_lhs.shape[0], device=_lhs.device, dtype=_lhs.dtype)

        # Store LHS norms for logging
        _memit_lhs_norms = {
            "base_lhs_norm": _base_lhs_norm,
            "kpkp_norm": _kpkp_norm,
            "identity_dim": _lhs.shape[0],
        }

        adj_k = torch.linalg.solve(_lhs, layer_ks)
        # === END augmented solve ==='''

    # Injection code for memit_main.py: before deltas storage
    log_and_cache_code = r'''            # === MEMIT+SeqReg: log + store keys (injected) ===
            # Log update norm and ||ΔW K_prev|| BEFORE appending current keys
            _upd_norm = torch.linalg.norm(upd_matrix).item()
            _dw_kprev_norm = 0.0
            _cache_batches = len(_memit_prev_cache.get(layer, []))
            _cache_keys = sum(k.shape[1] for k in _memit_prev_cache.get(layer, []))
            if _K_prev is not None:
                _dw_kprev_norm = torch.linalg.norm(upd_matrix.double() @ _K_prev).item()

            # Build log entry with LHS term norms
            _log_entry = {
                "batch": _memit_batch_idx[0], "layer": int(layer),
                "upd_norm": _upd_norm, "dw_kprev_norm": _dw_kprev_norm,
                "cache_batches": _cache_batches, "cache_keys": _cache_keys,
            }
            if '_memit_lhs_norms' in locals():
                _log_entry.update(_memit_lhs_norms)
            _memit_log.append(_log_entry)

            # Now append current keys to cache
            if _memit_lambda_prev > 0 or _memit_cache_strategy == "all":
                if layer not in _memit_prev_cache:
                    _memit_prev_cache[layer] = []
                _memit_prev_cache[layer].append(layer_ks.detach().cpu())
                if _memit_cache_max is not None and len(_memit_prev_cache[layer]) > _memit_cache_max:
                    if _memit_cache_strategy == "recent":
                        _memit_prev_cache[layer] = _memit_prev_cache[layer][-_memit_cache_max:]
            del _K_prev
            # === END log + store keys ==='''

    # Injection into evaluate.py: increment batch counter after each edit
    # NOTE: _memit_batch_idx is a list [int] (mutable container) so that both
    # the memit_main.py and evaluate.py exec namespaces share the same object,
    # and [0] += 1 avoids rebinding (which would cause UnboundLocalError in main()).
    batch_increment_hook = r'''        # === MEMIT+SeqReg: increment batch (injected) ===
        if '_memit_batch_idx' in globals():
            _memit_batch_idx[0] += 1
        # === END batch increment ===
'''

    # Debug freeze mode code (injected into evaluate.py before POST_EDIT_ANCHOR)
    debug_freeze_code = ""
    if debug_freeze_batch is not None:
        debug_freeze_code = f'''        # === MEMIT+SeqReg: debug freeze mode (injected) ===
        if '_memit_batch_idx' in globals() and _memit_batch_idx[0] == {debug_freeze_batch + 1}:
            import copy as _copy_mod
            print("\\n=== DEBUG FREEZE: same-state comparison at batch {debug_freeze_batch} ===")
            _frozen_cache = _copy_mod.deepcopy(_memit_prev_cache)
            _frozen_weights = {{k: v.detach().clone() for k, v in dict(model.named_parameters()).items()}}
            for _test_lp in [0.0, 0.1, 1.0, 10.0]:
                # Temporarily set lambda_prev and rerun
                _orig_lp = _memit_lambda_prev
                _memit_lambda_prev = _test_lp
                _memit_prev_cache = _copy_mod.deepcopy(_frozen_cache)
                # Apply same edit again
                _debug_model_copy, _ = apply_memit_to_model(
                    model, tok,
                    [
                        {{"case_id": record["case_id"], **rewrite_dict}}
                        for record in record_chunks
                        for rewrite_dict in (
                            record["requested_rewrite"]
                            if isinstance(record["requested_rewrite"], list)
                            else [record["requested_rewrite"]]
                        )
                    ],
                    hparams,
                    return_orig_weights=False,
                )
                # Check last logged entry
                _last_entries = [e for e in _memit_log if e["batch"] == _memit_batch_idx[0]]
                _avg_dw_kprev = sum(e["dw_kprev_norm"] for e in _last_entries) / max(len(_last_entries), 1)
                _avg_upd = sum(e["upd_norm"] for e in _last_entries) / max(len(_last_entries), 1)
                print(f"  lambda_prev={{_test_lp:6.2f}} -> avg ||ΔW||={{_avg_upd:.4f}}, avg ||ΔW@K_prev||={{_avg_dw_kprev:.4f}}")
                # Restore model weights
                with torch.no_grad():
                    for _pn, _pv in _frozen_weights.items():
                        dict(model.named_parameters())[_pn].data.copy_(_pv)
                _memit_lambda_prev = _orig_lp
                # Remove debug log entries
                _memit_log[:] = [e for e in _memit_log if e["batch"] != _memit_batch_idx[0]]
            _memit_prev_cache = _copy_mod.deepcopy(_frozen_cache)
            print("=== END DEBUG FREEZE ===\\n")
            del _frozen_cache, _frozen_weights
        # === END debug freeze ===
'''

    # Mega-batch eval injection (outside f-string to avoid Python 3.10 nested-quote issues)
    mega_batch_eval_injection = '''    # === MEGA-BATCH EVAL: batched multi-token scoring (injected by memit_sequential_runner) ===
    def _mega_batch_eval(model, tok, records, case_result_template, num_edits, case_ids, exec_time, batch_size=8):
        # Evaluate records with batched forward passes using full multi-token scoring.
        # Produces IDENTICAL results to per-record compute_rewrite_quality_counterfact
        # (same log-prob scoring, same argmax correctness) but batches the expensive
        # model forward pass across multiple records.
        import torch as _mbe_torch
        import numpy as _mbe_np
        import json as _mbe_json
        from itertools import chain as _mbe_chain
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
            # Each record produces: len(prefixes) * 2 sequences (each prefix paired with target_new and target_true)
            all_sequences = []
            record_meta = []  # per-record metadata for scoring

            for record in batch_records:
                out_file = _mbe_Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    record_meta.append(None)  # skip marker
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

                # Tokenize targets
                a_tok = tok(f" {target_new}")["input_ids"]
                b_tok = tok(f" {target_true}")["input_ids"]
                if _is_llama:
                    a_tok = a_tok[1:]
                    b_tok = b_tok[1:]

                # Compute prefix lengths
                prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
                if _is_llama:
                    prefix_lens = [l - 1 for l in prefix_lens]

                # Build sequences: each prefix paired with [target_new, target_true]
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

                # Extract this record's logits
                rec_logits = logits[start_idx:start_idx + n_seqs]

                # Exact replica of test_batch_prediction scoring
                probs = _mbe_np.zeros((n_seqs,), dtype=_mbe_np.float32)
                targets_correct = []

                for i in range(n_seqs):
                    cur_len = choice_a_len if i % 2 == 0 else choice_b_len
                    # Multi-token log-prob
                    for j in range(cur_len):
                        cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                        probs[i] += -_mbe_torch.nn.functional.log_softmax(
                            rec_logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
                        )[cur_tok].item()
                    probs[i] /= cur_len

                    # Argmax correctness (only for the "correct" target)
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

                # Build output in identical format to compute_rewrite_quality_counterfact
                ret_probs = [
                    {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
                    for i in range(0, n_seqs, 2)
                ]
                n_rw = meta["n_rewrite"]
                n_para = meta["n_paraphrase"]
                n_neigh = meta["n_neighborhood"]
                cutoffs = [0, n_rw, n_rw + n_para, n_rw + n_para + n_neigh]
                ret_corrects_flat = targets_correct
                # Split corrects by prompt group
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

            # Free GPU memory
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

    # --- Call mega-batch eval or skip ---
    if _do_final_eval:
        _records_to_eval = list(ds)
        if _memit_fast_mode:
            _records_to_eval = [r for r in ds if r["case_id"] in case_ids]
        _mega_batch_eval(edited_model, tok, _records_to_eval, case_result_template, num_edits, case_ids, exec_time)
    # === END mega-batch eval ===
    for record in ds:
        # Guard: skip if eval already done or disabled
        if not _do_final_eval:
            break
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'''

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

# 3. MEMIT+SeqReg parameters (shared state)
_memit_lambda_prev = {lambda_prev}
_memit_lambda_delta = {lambda_delta}
_memit_prev_cache = {{}}
_memit_cache_max = {cache_max_repr}
_memit_cache_strategy = "{cache_strategy}"
_memit_batch_idx = [0]  # mutable container: shared across exec namespaces, avoids UnboundLocalError
_memit_log = []
_memit_output_jsonl = "{output_jsonl}"
_memit_fast_mode = {fast_checkpoint}

# 3b. Checkpoint parameters
_ckpt_save_interval = {save_interval}
_ckpt_dir = "{checkpoint_dir}"
_ckpt_start_batch = {start_from_batch}
_ckpt_num_edits = {num_edits}
_ckpt_eval_at_checkpoints_only = {eval_at_checkpoints_only}

def _ckpt_save(cnt, model, hparams):
    \"\"\"Save model weights, prev_cache, batch_idx, and log at checkpoint boundary.\"\"\"
    from pathlib import Path
    from datetime import datetime, timezone

    batch_dir = Path(_ckpt_dir) / f"batch_{{cnt}}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save edited layer weights only
    layer_weights = {{}}
    for layer_idx in hparams.layers:
        for key in ["mlp.down_proj.weight", "mlp.up_proj.weight"]:
            param_name = f"model.layers.{{layer_idx}}.{{key}}"
            param = dict(model.named_parameters()).get(param_name)
            if param is not None:
                layer_weights[param_name] = param.data.cpu()
    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    # Save prev_cache (dict of layer -> list of key tensors)
    torch.save(_memit_prev_cache, str(batch_dir / "prev_cache.pt"))

    # Save log entries so far
    with open(str(batch_dir / "mechanism_log.jsonl"), "w") as f:
        for entry in _memit_log:
            f.write(json.dumps(entry) + "\\n")

    # Save metadata
    metadata = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _ckpt_num_edits,
        "batch_idx_counter": _memit_batch_idx[0],
        "lambda_prev": _memit_lambda_prev,
        "lambda_delta": _memit_lambda_delta,
        "cache_strategy": _memit_cache_strategy,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }}
    with open(str(batch_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [CHECKPOINT] Saved batch {{cnt}} ({{(cnt+1) * _ckpt_num_edits}} edits) -> {{batch_dir}}")

def _ckpt_load(model, hparams):
    \"\"\"Load checkpoint state: model weights, prev_cache, log.\"\"\"
    from pathlib import Path

    if _ckpt_start_batch <= 0:
        return False

    batch_dir = Path(_ckpt_dir) / f"batch_{{_ckpt_start_batch - 1}}"
    if not batch_dir.exists():
        print(f"  [CHECKPOINT] WARNING: Expected checkpoint at {{batch_dir}} not found. Starting from scratch.")
        return False

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
        print(f"  [CHECKPOINT] Loaded {{loaded_count}} parameter tensors")

    # Load prev_cache
    cache_file = batch_dir / "prev_cache.pt"
    if cache_file.exists():
        global _memit_prev_cache
        _memit_prev_cache = torch.load(str(cache_file), map_location="cpu")
        total_keys = sum(sum(k.shape[1] for k in v) for v in _memit_prev_cache.values())
        print(f"  [CHECKPOINT] Loaded prev_cache ({{len(_memit_prev_cache)}} layers, {{total_keys}} total keys)")

    # Load log entries
    log_file = batch_dir / "mechanism_log.jsonl"
    if log_file.exists():
        global _memit_log
        _memit_log.clear()
        with open(str(log_file)) as f:
            for line in f:
                _memit_log.append(json.loads(line))
        print(f"  [CHECKPOINT] Loaded {{len(_memit_log)}} log entries")

    # Restore batch counter
    _memit_batch_idx[0] = _ckpt_start_batch
    print(f"  [CHECKPOINT] Resuming from batch {{_ckpt_start_batch}} ({{_ckpt_start_batch * _ckpt_num_edits}} edits already applied)")
    return True

def _ckpt_should_skip(cnt):
    \"\"\"Return True if this batch was already processed.\"\"\"
    return cnt < _ckpt_start_batch

def _ckpt_should_save(cnt):
    \"\"\"Return True if we should save a checkpoint at this batch.\"\"\"
    return _ckpt_dir and (cnt + 1) % _ckpt_save_interval == 0

# 4. Read and patch memit_main.py
with open("memit/memit_main.py", "r") as f:
    _memit_source = f.read()

# Fix relative imports for standalone exec
_memit_source = _memit_source.replace("from .compute_ks", "from memit.compute_ks")
_memit_source = _memit_source.replace("from .compute_z", "from memit.compute_z")
_memit_source = _memit_source.replace("from .memit_hparams", "from memit.memit_hparams")

# Inject augmented solve (replace original solve)
_solve_anchor = {repr(SOLVE_ANCHOR)}
assert _solve_anchor in _memit_source, (
    "SOLVE_ANCHOR not found in memit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_solve_replacement = {repr(solve_replacement)}
_memit_source = _memit_source.replace(_solve_anchor, _solve_replacement, 1)

# Inject log + cache storage before deltas assignment
_deltas_anchor = {repr(DELTAS_ANCHOR)}
assert _deltas_anchor in _memit_source, (
    "DELTAS_ANCHOR not found in memit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_log_cache_code = {repr(log_and_cache_code)}
_memit_source = _memit_source.replace(_deltas_anchor, _log_cache_code + "\\n" + _deltas_anchor, 1)

# Verify injections
assert "MEMIT+SeqReg: augmented solve" in _memit_source, "Solve injection failed"
assert "MEMIT+SeqReg: log + store keys" in _memit_source, "Log/cache injection failed"

# 5. Compile and exec patched memit
_memit_ns = {{
    "__name__": "memit.memit_main",
    "__file__": "memit/memit_main.py",
    "_memit_lambda_prev": _memit_lambda_prev,
    "_memit_lambda_delta": _memit_lambda_delta,
    "_memit_prev_cache": _memit_prev_cache,
    "_memit_cache_max": _memit_cache_max,
    "_memit_cache_strategy": _memit_cache_strategy,
    "_memit_batch_idx": _memit_batch_idx,
    "_memit_log": _memit_log,
}}
exec(compile(_memit_source, "memit/memit_main.py", "exec"), _memit_ns)
_patched_apply_memit = _memit_ns["apply_memit_to_model"]
_patched_get_context_templates = _memit_ns["get_context_templates"]

print("[SeqReg] memit_main.py patched successfully")
print(f"  lambda_prev={{_memit_lambda_prev}}, lambda_delta={{_memit_lambda_delta}}")
print(f"  cache_strategy={{_memit_cache_strategy}}, cache_max={{_memit_cache_max}}")

# 6. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    _eval_source = f.read()

# Replace MEMIT import (we provide it via exec globals)
_import_anchor = {repr(MEMIT_IMPORT_ANCHOR)}
assert _import_anchor in _eval_source, (
    "MEMIT_IMPORT_ANCHOR not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_eval_source = _eval_source.replace(
    _import_anchor,
    "# apply_memit_to_model patched by memit_sequential_runner",
)

# Patch CUDA
_cuda_target = {repr(CUDA_PATCH_TARGET)}
assert _cuda_target in _eval_source, "CUDA patch target not found in evaluate.py."
_eval_source = _eval_source.replace(
    _cuda_target,
    "# CUDA_VISIBLE_DEVICES managed by memit_sequential_runner",
)

# Override RESULTS_DIR
_globals_import = 'from util.globals import *'
assert _globals_import in _eval_source, "globals import not found in evaluate.py"
_eval_source = _eval_source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)
print(f"  [RESULTS_DIR] Overridden to: {eval_results_dir}")

# Override dir_name to use variant
_eval_source = _eval_source.replace(
    'dir_name=args.alg_name,',
    'dir_name="{variant_name}",',
    1,
)

# Inject order shuffle + fingerprint before loop
_loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert _loop_anchor in _eval_source, "Loop anchor not found in evaluate.py."

_seqreg_order_id = {order_id}
if _seqreg_order_id > 0:
    _shuffle_code = (
        f'    # === ORDER SHUFFLE: shuffle dataset with order_id={{_seqreg_order_id}} (injected) ===\\n'
        f'    import random as _order_rng_module\\n'
        f'    _order_rng = _order_rng_module.Random({{_seqreg_order_id}})\\n'
        f'    _shuffled_indices = list(range(len(ds)))\\n'
        f'    _order_rng.shuffle(_shuffled_indices)\\n'
        f'    ds.data = [ds.data[i] for i in _shuffled_indices]\\n'
        f'    print("ORDER SHUFFLE: shuffled " + str(len(ds)) + " records with order_id={{_seqreg_order_id}}")\\n'
        f'    # === END order shuffle ===\\n'
    )
    _eval_source = _eval_source.replace(_loop_anchor, _shuffle_code + _loop_anchor, 1)

# Inject fingerprint
_fp_code = '''    # === FINGERPRINT: compute dataset fingerprint (injected) ===
    import hashlib as _fp_hashlib
    import json as _fp_json
    _fp_case_ids = [r["case_id"] for r in ds]
    _fp_id_bytes = _fp_json.dumps(_fp_case_ids, separators=(",", ":")).encode("utf-8")
    _fp_sha256 = _fp_hashlib.sha256(_fp_id_bytes).hexdigest()
    print(f"  [FINGERPRINT] Dataset: {{len(ds)}} records, SHA-256: {{_fp_sha256[:16]}}...")
    print(f"  [FINGERPRINT] Order ID: ''' + str(_seqreg_order_id) + ''', first 5 IDs: {{_fp_case_ids[:5]}}")
    _fp_ordering_path = run_dir / "edit_ordering.json"
    _fp_ordering = {{
        "case_ids_ordered": _fp_case_ids,
        "n_records": len(ds),
        "order_id": ''' + str(_seqreg_order_id) + ''',
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
_eval_source = _eval_source.replace(_loop_anchor, _fp_code + _loop_anchor, 1)

# Inject dataset override (for coupling streams, etc.)
_ds_override_path = {repr(dataset_override) if dataset_override else 'None'}
if _ds_override_path:
    _ds_override_code = '''    # === DATASET OVERRIDE: replace ds.data with external file (injected) ===
    import json as _dsov_json
    with open("{dataset_override}", "r") as _dsov_f:
        ds.data = _dsov_json.load(_dsov_f)
    print(f"  [OVERRIDE] Loaded {{len(ds)}} records from {dataset_override}")
    # === END dataset override ===
'''
    _eval_source = _eval_source.replace(_loop_anchor, _ds_override_code + _loop_anchor, 1)

# Inject checkpoint LOAD before the loop
_ckpt_load_injection = '''    # === CHECKPOINT: load state from previous run (injected) ===
    exec_time = 0  # Default: prevents UnboundLocalError if all batches skipped
    edited_model = model  # Default: if all batches skipped, model IS the edited model (weights restored from checkpoint)
    if _ckpt_start_batch > 0 and '_ckpt_load' in globals():
        _ckpt_load(model, hparams)
    # === END checkpoint load ===
'''
_eval_source = _eval_source.replace(_loop_anchor, _ckpt_load_injection + _loop_anchor, 1)

# Inject SKIP guard before per-batch edit call
_pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert _pre_anchor in _eval_source, "PRE_EDIT_ANCHOR not found in evaluate.py."
_skip_injection = '''        # === CHECKPOINT: skip already-processed batches (injected) ===
        if '_ckpt_should_skip' in globals() and _ckpt_should_skip(cnt):
            cnt += 1
            continue
        # === END checkpoint skip ===
'''
_eval_source = _eval_source.replace(_pre_anchor, _skip_injection + _pre_anchor, 1)

# Inject batch increment + checkpoint save + debug freeze AFTER POST_EDIT_ANCHOR (exec_time line)
# This preserves the if/elif/else chain that assigns edited_model.
_post_anchor = {repr(POST_EDIT_ANCHOR)}
assert _post_anchor in _eval_source, "POST_EDIT_ANCHOR not found in evaluate.py."
_batch_hook = {repr(batch_increment_hook)}
_ckpt_save_hook = '''        # === CHECKPOINT: save at interval boundaries (injected) ===
        if '_ckpt_should_save' in globals() and _ckpt_should_save(cnt):
            _ckpt_save(cnt, model, hparams)
        # === END checkpoint save ===
'''
_debug_code = {repr(debug_freeze_code)}
_eval_source = _eval_source.replace(_post_anchor, _post_anchor + "\\n" + _batch_hook + _ckpt_save_hook + _debug_code, 1)

# Inject CHECKPOINT-ONLY EVAL guard (skip evaluation for non-checkpoint batches)
_eval_start_anchor = '    # torch.save(hs, "post_edit_hs_memit.pt")\\n    start = time()'
assert _eval_start_anchor in _eval_source, (
    "Evaluation start anchor not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_checkpoint_eval_skip = '''    # torch.save(hs, "post_edit_hs_memit.pt")
    # === MEMIT+SeqReg: skip evaluation if last batch is not a checkpoint boundary (injected) ===
    _do_final_eval = True
    if _ckpt_eval_at_checkpoints_only and not _ckpt_should_save(cnt - 1):
        _do_final_eval = False
        print(f"  [CHECKPOINT] Skipping final evaluation (batch {{cnt-1}} not at checkpoint boundary)")
    # === END checkpoint eval skip ===
    start = time()'''
_eval_source = _eval_source.replace(_eval_start_anchor, _checkpoint_eval_skip, 1)

# Inject MEGA-BATCH evaluation to replace the per-record eval loop.
# This replicates the exact multi-token scoring from test_batch_prediction
# but batches multiple records per forward pass for ~10-30x speedup.
_eval_anchor = '    for record in ds:\\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
assert _eval_anchor in _eval_source, (
    "Evaluation loop anchor not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)

# The full replacement: defines _mega_batch_eval inline, then calls it instead of the per-record loop.
_mega_batch_eval_injection = {repr(mega_batch_eval_injection)}
_eval_source = _eval_source.replace(_eval_anchor, _mega_batch_eval_injection, 1)

print("[SeqReg] evaluate.py patched successfully")
if {fast_checkpoint}:
    print("  Fast checkpoint mode: ENABLED (only evaluate edited batch)")
if {eval_at_checkpoints_only}:
    print("  Eval at checkpoints only: ENABLED (milestone mode)")

# 7. Execute patched evaluate.py
exec(compile(_eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_memit_to_model": _patched_apply_memit,
    "get_context_templates": _patched_get_context_templates,
    "_memit_lambda_prev": _memit_lambda_prev,
    "_memit_lambda_delta": _memit_lambda_delta,
    "_memit_prev_cache": _memit_prev_cache,
    "_memit_cache_max": _memit_cache_max,
    "_memit_cache_strategy": _memit_cache_strategy,
    "_memit_batch_idx": _memit_batch_idx,
    "_memit_log": _memit_log,
    "_memit_fast_mode": _memit_fast_mode,
    "_ckpt_start_batch": _ckpt_start_batch,
    "_ckpt_save_interval": _ckpt_save_interval,
    "_ckpt_dir": _ckpt_dir,
    "_ckpt_eval_at_checkpoints_only": _ckpt_eval_at_checkpoints_only,
    "_ckpt_save": _ckpt_save,
    "_ckpt_load": _ckpt_load,
    "_ckpt_should_skip": _ckpt_should_skip,
    "_ckpt_should_save": _ckpt_should_save,
}})

# 8. Write log to JSONL
with open(_memit_output_jsonl, "w") as f:
    for entry in _memit_log:
        f.write(json.dumps(entry) + "\\n")

print(f"\\n[SeqReg] Log written: {{_memit_output_jsonl}} ({{len(_memit_log)}} entries)")
""")
    return script


def validate_anchors() -> None:
    """Verify all source anchors exist in the pinned code."""
    alphaedit_root = get_alphaedit_root()

    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("MEMIT_IMPORT_ANCHOR", MEMIT_IMPORT_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    memit_source = (alphaedit_root / "memit" / "memit_main.py").read_text()
    for name, anchor in [
        ("SOLVE_ANCHOR", SOLVE_ANCHOR),
        ("DELTAS_ANCHOR", DELTAS_ANCHOR),
    ]:
        assert anchor in memit_source, f"{name} not found in memit_main.py"

    print("  All source anchors validated.")


def run(args: argparse.Namespace) -> None:
    """Launch MEMIT+SeqReg experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    model_name = resolve_model_path(args.model_name)

    print("Validating source anchors...")
    validate_anchors()

    # Parse cache_max
    cache_max = None if args.cache_max == "none" else int(args.cache_max)

    # Output directory
    # evaluate.py appends {dir_name}/run_000/ to RESULTS_DIR, so we set results_dir
    # to the level ABOVE where variant_name gets appended.
    cache_max_str = str(cache_max) if cache_max is not None else "0"
    variant_name = f"MEMIT-Seq-lp{args.lambda_prev}-ld{args.lambda_delta}-cache{cache_max_str}"
    ordering = getattr(args, 'ordering', None)
    if ordering:
        # Matched ordering: evaluate.py writes to {results_dir}/{variant}/run_000/
        # Final: results/matched_ordering/{ordering}/seed{N}/{edits}edits/{variant}/run_000/
        results_dir = (
            get_result_root() / "matched_ordering" / ordering
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    else:
        # Standard failure curve: evaluate.py writes to {results_dir}/{variant}/run_000/
        # Final: results/failure_curve_checkpointed/seed{N}/{edits}edits/{variant}/run_000/
        results_dir = (
            get_result_root() / "failure_curve_checkpointed"
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = results_dir / f"log_seed{args.seed}_lp{args.lambda_prev}_ld{args.lambda_delta}_{timestamp}.jsonl"

    # Resolve checkpoint directory and auto-detect resume point
    ordering = getattr(args, 'ordering', None)
    ckpt_dir = resolve_checkpoint_dir(
        args.checkpoint_dir, args.seed, args.lambda_prev, args.lambda_delta, cache_max, ordering=ordering
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        # Auto-detect from latest checkpoint
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = latest[0] + 1
            if start_from_batch >= total_batches:
                start_from_batch = total_batches
                print(f"  Auto-detected: checkpoint at batch {latest[0]} covers all {total_batches} batches. Will run eval only.")
            else:
                print(f"  Auto-detected: resume from batch {start_from_batch} (checkpoint at batch {latest[0]})")
        else:
            start_from_batch = 0
            print("  No existing checkpoints found. Starting from batch 0.")

    script = build_sequential_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name="MEMIT",
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        lambda_prev=args.lambda_prev,
        lambda_delta=args.lambda_delta,
        cache_strategy=args.cache_strategy,
        cache_max=cache_max,
        output_jsonl=str(output_jsonl),
        debug_freeze_batch=args.debug_freeze_batch,
        fast_checkpoint=args.fast_checkpoint,
        eval_at_checkpoints_only=args.eval_at_checkpoints_only,
        order_id=args.order_id,
        save_interval=args.save_interval,
        checkpoint_dir=str(ckpt_dir),
        start_from_batch=start_from_batch,
        dataset_override=args.dataset_override,
        eval_results_dir=str(results_dir),
        variant_name=variant_name,
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\n{'=' * 70}")
    print("MEMIT+SeqReg Runner")
    print(f"  Seed:           {args.seed}")
    print(f"  λ_prev:         {args.lambda_prev}")
    print(f"  λ_delta:        {args.lambda_delta}")
    print(f"  Cache strategy: {args.cache_strategy}")
    print(f"  Cache max:      {cache_max}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:      {args.num_edits}")
    print(f"  Total batches:  {total_batches}")
    print(f"  Resume from:    batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:  every {args.save_interval} batches")
    print(f"  Checkpoint dir: {ckpt_dir}")
    if args.eval_at_checkpoints_only:
        eval_mode = f"Milestone only (every {args.save_interval} batches)"
    elif args.fast_checkpoint:
        eval_mode = "Fast (edited batch only)"
    else:
        eval_mode = "Full (all facts every batch)"
    print(f"  Evaluation:     {eval_mode}")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    if args.dataset_override:
        print(f"  Dataset override: {args.dataset_override}")
    if args.debug_freeze_batch is not None:
        print(f"  DEBUG FREEZE:   batch {args.debug_freeze_batch}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "experiment": "memit_seqreg_ridge",
        "seed": args.seed,
        "order_id": args.order_id,
        "lambda_prev": args.lambda_prev,
        "lambda_delta": args.lambda_delta,
        "cache_strategy": args.cache_strategy,
        "cache_max": cache_max,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "ds_name": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "downstream_eval_steps": args.downstream_eval_steps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "eval_config_hash": hash_eval_config(),
        "output_jsonl": str(output_jsonl),
    }
    meta_path = results_dir / f"metadata_seed{args.seed}_lp{args.lambda_prev}_ld{args.lambda_delta}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Launch
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(alphaedit_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Experiment failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("MEMIT+SeqReg completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Log:       {output_jsonl}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="MEMIT+SeqReg: control baseline for sequential editing"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    # Model and data
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # SeqReg parameters (non-projected analogue of AlphaEdit Eq. 12)
    parser.add_argument("--lambda_prev", type=float, default=0.0,
                        help="Previous-key protection: λ_prev ||ΔK_prev||² (AlphaEdit Eq. 12 uses λ=1)")
    parser.add_argument("--lambda_delta", type=float, default=0.0,
                        help="Update size minimization: λ_delta ||Δ||² (AlphaEdit Eq. 12 uses λ=1)")
    parser.add_argument("--cache_strategy", default="recent", choices=["recent", "all"],
                        help="Cache management strategy (default: recent)")
    parser.add_argument("--cache_max", default="20",
                        help="Max batches in cache (default: 20, use 'none' for unlimited)")

    # Checkpoint and resume
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N batches (default: 10)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Explicit checkpoint directory (default: ~/.cache/memit_seqreg_checkpoints/...)")
    parser.add_argument("--start_from_batch", type=int, default=-1,
                        help="Resume from this batch (-1 = auto-detect from latest checkpoint)")

    # Debug and performance
    parser.add_argument("--debug_freeze_batch", type=int, default=None,
                        help="Run same-state diagnostic at this batch (tests λ_prev effect)")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true",
                        help="Fast checkpoint mode: only evaluate edited batch, not entire dataset (much faster)")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true",
                        help="Milestone mode: evaluate full dataset only at checkpoint boundaries (RECOMMENDED for papers)")

    # Dataset override (for coupling streams, etc.)
    parser.add_argument("--dataset_override", type=str, default=None,
                        help="Path to JSON file to replace dataset (e.g., coupling stream)")

    # Order sensitivity
    parser.add_argument("--order_id", type=int, default=0,
                        help="Edit ordering ID (0=canonical, >0=shuffle with Random(order_id))")

    # Matched ordering
    parser.add_argument("--ordering", type=str, default=None,
                        help="Ordering type (e.g. key_clustered, key_dispersed) — routes checkpoints to matched_ordering/")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
