#!/usr/bin/env python3
"""
Seeded reproducibility wrapper for AlphaEdit / MEMIT / ROME experiments.

Uses evaluate_harness.run_experiment() instead of exec(compile(evaluate.py)).
No string patching, no reading vendor source as text.

Usage:
    python src/runners/seeded_runner.py \
        --seed 42 --cuda_device 0 \
        --alg_name AlphaEdit --ds_name mcf \
        --dataset_size_limit 2000 --num_edits 100
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))
sys.path.insert(0, str(_SRC_DIR))

from model_registry import DEFAULT_MODEL
from model_resolve import resolve_model_path
from setup_hparams import link_hparams
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root
from evaluate_harness import (
    run_experiment, load_model_and_tok, load_dataset,
    ExperimentHooks, _canonical_name_or_path,
)
from mega_batch_eval import get_mega_batch_eval_source

# Algorithm dispatch — maps alg_name to (HyperParamsClass, apply_fn)
# Imported lazily after vendor is on sys.path
ALG_DICT = None


def _init_alg_dict():
    """Import algorithm functions from vendor tree (requires vendor on sys.path)."""
    global ALG_DICT
    if ALG_DICT is not None:
        return

    alphaedit_root = get_alphaedit_root()
    sys.path.insert(0, str(alphaedit_root))

    from AlphaEdit import AlphaEditHyperParams
    from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
    from memit import MEMITHyperParams
    from memit.memit_main import apply_memit_to_model
    from rome import ROMEHyperParams
    from rome.rome_main import apply_rome_to_model

    ALG_DICT = {
        "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
        "MEMIT": (MEMITHyperParams, apply_memit_to_model),
        "ROME": (ROMEHyperParams, apply_rome_to_model),
    }


def _seed_everything(seed: int):
    """Seed all RNG sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    """Resolve results directory from experiment name or args."""
    experiment = os.environ.get("EXPERIMENT_NAME", "")
    if not experiment:
        experiment = f"{args.alg_name.lower()}_{args.ds_name}"

    results_base = get_result_root() / experiment / f"seed{args.seed}"
    results_base = results_base / f"{args.dataset_size_limit}edits"

    order_id = getattr(args, "order_id", 0)
    if "ordered" in experiment or "order" in experiment:
        results_base = results_base / f"order{order_id}"

    return results_base


def _make_eval_fn():
    """Instantiate mega_batch_eval function from the shared module source."""
    ns = {}
    exec(get_mega_batch_eval_source(), ns)
    return ns["_mega_batch_eval"]


def record_metadata(seed, args, results_dir, run_id="run_000"):
    """Save run metadata for reproducibility."""
    alphaedit_root = get_alphaedit_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(alphaedit_root), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "order_id": getattr(args, "order_id", 0),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "alphaedit_commit": commit,
        "eval_config_hash": hash_eval_config(),
        "cuda_device": args.cuda_device,
        "experiment": os.environ.get("EXPERIMENT_NAME", f"{args.alg_name.lower()}_{args.ds_name}"),
        "algorithm": args.alg_name,
        "dataset": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "run_id": run_id,
        "runner": "seeded_runner (harness-based)",
        "params": {
            "model_name": args.model_name,
            "hparams_fname": args.hparams_fname,
            "downstream_eval_steps": args.downstream_eval_steps,
            "conserve_memory": args.conserve_memory,
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    meta_path = results_dir / f"run_seed{seed}_{args.alg_name}_{args.ds_name}_{args.dataset_size_limit}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    return meta_path


def run(args: argparse.Namespace) -> None:
    """Run the experiment using evaluate_harness."""
    alphaedit_root = get_alphaedit_root()
    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    # Setup
    link_hparams()
    _seed_everything(args.seed)

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Resolve paths
    results_dir = _resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("Seeded Runner (harness-based)")
    print(f"  Algorithm:  {args.alg_name}")
    print(f"  Dataset:    {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:  {args.num_edits}")
    print(f"  Seed:       {args.seed}")
    print(f"  CUDA:       device {args.cuda_device}")
    print(f"  Model:      {args.model_name}")
    print(f"  Results:    {results_dir}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Load model + tokenizer
    model, tok = load_model_and_tok(args.model_name)

    # Load dataset
    dataset = load_dataset(args.ds_name, args.dataset_size_limit, alphaedit_root)

    # Shuffle dataset if order_id > 0
    if args.order_id > 0:
        import random as _r
        _r2 = _r.Random(args.order_id)
        _r2.shuffle(dataset)
        print(f"  Dataset shuffled with order_id={args.order_id}")

    # Import algorithm
    _init_alg_dict()
    if args.alg_name not in ALG_DICT:
        print(f"ERROR: Unknown algorithm '{args.alg_name}'. Available: {list(ALG_DICT.keys())}")
        sys.exit(1)

    params_class, apply_fn = ALG_DICT[args.alg_name]

    # Load hparams
    hparams_dir = alphaedit_root / "hparams" / args.alg_name
    hparams_path = hparams_dir / args.hparams_fname
    if not hparams_path.exists():
        for alg_dir in ["MEMIT", "AlphaEdit", "ROME"]:
            alt = alphaedit_root / "hparams" / alg_dir / args.hparams_fname
            if alt.exists():
                hparams_path = alt
                break
    hparams = params_class.from_json(hparams_path)

    # Build eval function from shared mega_batch_eval
    mbe_fn = _make_eval_fn()

    # Set up hooks
    hooks = ExperimentHooks(
        eval_fn=mbe_fn,
    )

    # For AlphaEdit: need to pass cache_c and P via extra_apply_kwargs
    cache_c = None
    P = None
    if args.alg_name == "AlphaEdit":
        # AlphaEdit manages cache_c internally (returned from apply function)
        # and P is loaded from the stats directory
        pass  # cache_c/P handled by the apply function itself

    # Run the experiment
    alg_results_dir = results_dir / args.alg_name
    summary = run_experiment(
        model=model,
        tok=tok,
        hparams=hparams,
        dataset=dataset,
        apply_fn=apply_fn,
        alg_name=args.alg_name,
        num_edits=args.num_edits,
        results_dir=alg_results_dir,
        ds_name=args.ds_name,
        conserve_memory=args.conserve_memory,
        hooks=hooks,
    )

    # Record metadata
    record_metadata(args.seed, args, results_dir)

    print(f"\n{'=' * 70}")
    print("Experiment completed.")
    print(f"  Batches: {summary['batches_run']}")
    print(f"  Edits:   {summary['total_edits']}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Seeded reproducibility wrapper (harness-based)"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", required=True, choices=["AlphaEdit", "MEMIT", "ROME"])
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", required=True, choices=["mcf", "cf", "zsre", "mquake"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=5)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--generation_test_interval", type=int, default=1)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--order_id", type=int, default=0)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
