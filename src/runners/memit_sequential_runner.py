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

Implementation: Dual source injection (following coupling_stress_runner.py):
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


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    return get_project_root() / "vendor" / "AlphaEdit"


# --- Source anchors (commit b84624f) ---

# evaluate.py
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'
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
                "batch": _memit_batch_idx, "layer": int(layer),
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
    batch_increment_hook = r'''        # === MEMIT+SeqReg: increment batch (injected) ===
        if '_memit_batch_idx' in globals():
            _memit_batch_idx += 1
        # === END batch increment ===
'''

    # Debug freeze mode code (injected into evaluate.py before POST_EDIT_ANCHOR)
    debug_freeze_code = ""
    if debug_freeze_batch is not None:
        debug_freeze_code = f'''        # === MEMIT+SeqReg: debug freeze mode (injected) ===
        if '_memit_batch_idx' in globals() and _memit_batch_idx == {debug_freeze_batch + 1}:
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
                _last_entries = [e for e in _memit_log if e["batch"] == _memit_batch_idx]
                _avg_dw_kprev = sum(e["dw_kprev_norm"] for e in _last_entries) / max(len(_last_entries), 1)
                _avg_upd = sum(e["upd_norm"] for e in _last_entries) / max(len(_last_entries), 1)
                print(f"  lambda_prev={{_test_lp:6.2f}} -> avg ||ΔW||={{_avg_upd:.4f}}, avg ||ΔW@K_prev||={{_avg_dw_kprev:.4f}}")
                # Restore model weights
                with torch.no_grad():
                    for _pn, _pv in _frozen_weights.items():
                        dict(model.named_parameters())[_pn].data.copy_(_pv)
                _memit_lambda_prev = _orig_lp
                # Remove debug log entries
                _memit_log[:] = [e for e in _memit_log if e["batch"] != _memit_batch_idx]
            _memit_prev_cache = _copy_mod.deepcopy(_frozen_cache)
            print("=== END DEBUG FREEZE ===\\n")
            del _frozen_cache, _frozen_weights
        # === END debug freeze ===
'''

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
_memit_batch_idx = 0
_memit_log = []
_memit_output_jsonl = "{output_jsonl}"
_memit_fast_mode = {fast_checkpoint}

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

# Inject batch increment + debug freeze before POST_EDIT_ANCHOR
_post_anchor = {repr(POST_EDIT_ANCHOR)}
assert _post_anchor in _eval_source, "POST_EDIT_ANCHOR not found in evaluate.py."
_batch_hook = {repr(batch_increment_hook)}
_debug_code = {repr(debug_freeze_code)}
_eval_source = _eval_source.replace(_post_anchor, _batch_hook + _debug_code + _post_anchor, 1)

# Inject FAST EVAL guard in evaluation loop (skip records not in current batch)
_eval_anchor = '    for record in ds:\\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
assert _eval_anchor in _eval_source, (
    "Evaluation loop anchor not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_fast_eval_injection = '''    for record in ds:
        # === MEMIT+SeqReg: fast mode - skip evaluating non-batch records (injected) ===
        if _memit_fast_mode and record["case_id"] not in case_ids:
            continue
        # === END fast mode guard ===
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'''
_eval_source = _eval_source.replace(_eval_anchor, _fast_eval_injection, 1)

print("[SeqReg] evaluate.py patched successfully")
if {fast_checkpoint}:
    print("  Fast checkpoint mode: ENABLED (only evaluate edited batch)")

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

    # Output directory
    results_dir = project_root / "results" / "memit_seqreg"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = results_dir / f"log_seed{args.seed}_lp{args.lambda_prev}_ld{args.lambda_delta}_{timestamp}.jsonl"

    # Parse cache_max
    cache_max = None if args.cache_max == "none" else int(args.cache_max)

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
    print(f"  Fast checkpoint: {'YES - only evaluate edited batch' if args.fast_checkpoint else 'NO - full dataset evaluation'}")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    if args.debug_freeze_batch is not None:
        print(f"  DEBUG FREEZE:   batch {args.debug_freeze_batch}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "experiment": "memit_seqreg_ridge",
        "seed": args.seed,
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

    # Debug and performance
    parser.add_argument("--debug_freeze_batch", type=int, default=None,
                        help="Run same-state diagnostic at this batch (tests λ_prev effect)")
    parser.add_argument("--fast_checkpoint", action="store_true",
                        help="Fast checkpoint mode: only evaluate edited batch, not entire dataset (much faster)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
