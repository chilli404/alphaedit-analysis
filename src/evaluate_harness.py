"""Maintained evaluate harness — replaces exec(compile(patched_evaluate.py)).

This module provides run_experiment(), which replaces the source-injection
pattern used by all runners. Instead of reading vendor evaluate.py as text,
patching it with string replacements, and exec'ing it, runners now:

  1. Load model, tokenizer, hparams, and dataset (using helpers from this module)
  2. Provide an apply_fn (the algorithm's editing function)
  3. Provide ExperimentHooks for algorithm-specific behavior
  4. Call run_experiment()

The vendor's evaluate.py had 570 lines + ~200 lines of injected patches per runner.
This harness is ~200 lines with no string manipulation.

What still uses exec(compile()):
  - memit_main.py kernel solve replacement (polykernel_seqreg_runner)
  - AlphaEdit_main.py C₀ injection (checkpoint_runner)
  These are contained: they compile the ALGORITHM file, extract the apply function,
  and pass it to this harness as apply_fn.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, Optional

import numpy as np
import torch


@dataclass
class ExperimentHooks:
    """Callbacks for algorithm-specific behavior at each stage of the edit loop.

    All hooks are optional. Unset hooks use sensible defaults (do nothing, eval every batch).
    """
    before_edit: Optional[Callable] = None
    # (batch_idx, model, records, hparams) -> None
    # Called before apply_fn. Use for: skip guard, pre-edit logging.

    after_edit: Optional[Callable] = None
    # (batch_idx, model, records, hparams, edit_result, exec_time) -> None
    # Called after apply_fn. Use for: checkpoint save, mechanism logging.

    should_eval: Optional[Callable] = None
    # (batch_idx) -> bool
    # Whether to evaluate after this batch. Default: every batch.

    eval_fn: Optional[Callable] = None
    # (model, tok, all_records_so_far, case_result_template, num_edits, case_ids, exec_time) -> None
    # Overrides default per-record evaluation. Use for mega_batch_eval.

    extra_apply_kwargs: Optional[Callable] = None
    # (batch_idx) -> dict
    # Additional kwargs passed to apply_fn (e.g. cache_c, P for AlphaEdit).


CANONICAL_MODEL_NAMES = {
    "llama": "llama3-8b-instruct",
    "gpt-j": "gpt-j-6b",
    "qwen": "qwen2.5-7b-instruct",
}


def _canonical_name_or_path(model_name: str) -> str:
    """Map any model name variant to ONE canonical name.

    This name is used for:
      1. model.config._name_or_path → vendor covariance lookup (.replace("/","_"))
      2. GLUE context length map (.lower().split("/")[-1])
      3. Stats directory name (link_stats.sh symlinks)

    All three transforms must produce the same result. Using the stats
    directory name directly (lowercase, no slashes) satisfies all three.
    """
    mn = model_name.lower()
    for key, canonical in CANONICAL_MODEL_NAMES.items():
        if key in mn:
            return canonical
    return model_name.split("/")[-1].lower()


def load_model_and_tok(model_name: str, device: str = "cuda", dtype=None, token: str = None):
    """Load model and tokenizer, resolving paths via model_resolve.

    Sets model.config._name_or_path to a canonical value so the vendor
    stats-loading code finds covariance files under one directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent / "util"))
    from model_resolve import resolve_model_path

    model_path = resolve_model_path(model_name)
    load_kwargs = {"token": token or os.environ.get("HF_TOKEN")}
    if dtype:
        load_kwargs["torch_dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).to(device)
    tok = AutoTokenizer.from_pretrained(model_path, token=load_kwargs["token"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Normalize _name_or_path so vendor code finds stats under ONE directory
    model.config._name_or_path = _canonical_name_or_path(model_name)

    return model, tok


def load_dataset(ds_name: str, size_limit: int, alphaedit_root: Path = None,
                 dataset_override: str = None) -> list[dict]:
    """Load MCF or zsRE dataset records.

    If dataset_override is provided, load from that JSON file (for ordering experiments).
    Otherwise, load from vendor data directory.
    """
    if dataset_override:
        with open(dataset_override) as f:
            records = json.load(f)
        if size_limit and len(records) > size_limit:
            records = records[:size_limit]
        return records

    if alphaedit_root is None:
        from paths import get_alphaedit_root
        alphaedit_root = get_alphaedit_root()

    # Pre-populate vendor globals module BEFORE adding to sys.path
    # (vendor's dsets imports util.globals which reads globals.yml from CWD)
    _ensure_vendor_globals(alphaedit_root)

    sys.path.insert(0, str(alphaedit_root))

    if ds_name in ("mcf", "multi_counterfact"):
        from dsets import MultiCounterFactDataset
        ds = MultiCounterFactDataset(str(alphaedit_root / "data"), size=size_limit)
    elif ds_name in ("zsre", "zsre_mend_eval"):
        from dsets import MENDQADataset
        ds = MENDQADataset(str(alphaedit_root / "data"), size=size_limit)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    return list(ds)


def _ensure_vendor_globals(alphaedit_root: Path):
    """Pre-populate vendor util.globals module so it doesn't need globals.yml from CWD.

    The vendor code does `from util.globals import *` which reads globals.yml
    relative to CWD. Instead of chdir'ing, we create the module with the right
    values directly. Must be called BEFORE adding alphaedit_root to sys.path.
    """
    import types

    # Create the 'util' package if it doesn't exist as a vendor module yet
    if "util" not in sys.modules or not hasattr(sys.modules.get("util"), "__path__"):
        util_mod = types.ModuleType("util")
        util_mod.__path__ = [str(alphaedit_root / "util")]
        sys.modules["util"] = util_mod

    # Create util.globals with the correct paths
    if "util.globals" not in sys.modules:
        mod = types.ModuleType("util.globals")
        mod.RESULTS_DIR = alphaedit_root / "results"
        mod.DATA_DIR = alphaedit_root / "data"
        mod.STATS_DIR = alphaedit_root / "data" / "stats"
        mod.HPARAMS_DIR = alphaedit_root / "hparams"
        mod.KV_DIR = alphaedit_root / "share" / "projects" / "rewriting-knowledge" / "kvs"
        mod.REMOTE_ROOT_URL = "https://memit.baulab.info"
        sys.modules["util.globals"] = mod


def run_experiment(
    model,
    tok,
    hparams,
    dataset: list[dict],
    apply_fn: Callable,
    *,
    alg_name: str,
    num_edits: int = 100,
    results_dir: Path,
    ds_name: str = "mcf",
    conserve_memory: bool = True,
    hooks: ExperimentHooks = None,
    max_batches: int = None,
) -> dict:
    """Run the full edit-and-evaluate loop.

    This replaces the 570-line vendor evaluate.py + ~200 lines of patches per runner.

    Args:
        model: The model to edit.
        tok: The tokenizer.
        hparams: Algorithm hyperparameters.
        dataset: Pre-loaded and ordered dataset records.
        apply_fn: Algorithm's apply function (e.g. apply_memit_to_model).
        alg_name: Algorithm name for dispatch logic.
        num_edits: Batch size.
        results_dir: Where per-case JSONs are written.
        ds_name: Dataset name (for choosing eval function).
        conserve_memory: Back up weights on CPU to save GPU memory.
        hooks: Algorithm-specific callbacks.
        max_batches: Stop after this many batches (for testing).

    Returns:
        Dict with summary metrics.
    """
    if hooks is None:
        hooks = ExperimentHooks()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine run directory (run_000, run_001, etc.)
    existing_runs = sorted(results_dir.glob("run_*"))
    run_id = len(existing_runs)
    run_dir = results_dir / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    case_result_template = str(run_dir / f"{num_edits}_edits-case_{{}}.json")

    total_batches = len(dataset) // num_edits
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    conserve_args = {}
    if conserve_memory:
        conserve_args["return_orig_weights_device"] = "cpu"

    all_case_ids = []
    summary = {"batches_run": 0, "total_edits": 0}

    for batch_idx in range(total_batches):
        start_idx = batch_idx * num_edits
        end_idx = start_idx + num_edits
        records = dataset[start_idx:end_idx]
        case_ids = [r["case_id"] for r in records]
        all_case_ids.extend(case_ids)

        # Pre-edit hook
        if hooks.before_edit:
            hooks.before_edit(batch_idx, model, records, hparams)

        # Get extra kwargs from hook
        extra_kwargs = {}
        if hooks.extra_apply_kwargs:
            extra_kwargs = hooks.extra_apply_kwargs(batch_idx)

        # Build requests in vendor format: flatten requested_rewrite to top level
        requests = [
            {"case_id": r["case_id"], **r["requested_rewrite"]}
            if "requested_rewrite" in r else r
            for r in records
        ]

        # Apply the edit
        start_time = time()
        edit_result = apply_fn(
            model, tok, requests, hparams,
            return_orig_weights=False,
            **conserve_args,
            **extra_kwargs,
        )
        exec_time = time() - start_time

        # Unpack result — some algorithms return (model, weights_copy), others (model, cache_c)
        if isinstance(edit_result, tuple):
            edited_model = edit_result[0]
            edit_extra = edit_result[1] if len(edit_result) > 1 else None
        else:
            edited_model = edit_result
            edit_extra = None

        # Post-edit hook
        if hooks.after_edit:
            hooks.after_edit(batch_idx, model, records, hparams, edit_extra, exec_time)

        # Evaluate
        do_eval = True
        if hooks.should_eval:
            do_eval = hooks.should_eval(batch_idx)

        if do_eval and hooks.eval_fn:
            hooks.eval_fn(model, tok, dataset[:end_idx], case_result_template,
                         num_edits, all_case_ids, exec_time)

        summary["batches_run"] = batch_idx + 1
        summary["total_edits"] = end_idx

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{total_batches} ({end_idx} edits)", flush=True)

    return summary
