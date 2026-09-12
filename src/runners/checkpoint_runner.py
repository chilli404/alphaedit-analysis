#!/usr/bin/env python3
"""
Checkpoint-Based Failure Curve Runner for AlphaEdit / MEMIT.

Uses evaluate_harness.run_experiment() instead of exec(compile(evaluate.py)).
Checkpoint save/load/resume via shared checkpoint_io module.

Still uses exec(compile()) for:
  - AlphaEdit C₀ injection (--inject_c0): patches AlphaEdit_main.py internals

Usage:
    python src/runners/checkpoint_runner.py \
        --seed 42 --alg_name AlphaEdit --ds_name mcf \
        --dataset_size_limit 5000 --num_edits 100 \
        --save_interval 10 --conserve_memory
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
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root
from evaluate_harness import (
    run_experiment, load_model_and_tok, load_dataset,
    ExperimentHooks,
)
from checkpoint_io import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    should_save, should_skip,
)
from mega_batch_eval import get_mega_batch_eval_source
from experiment_config import ExperimentConfig


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    experiment = os.environ.get("EXPERIMENT_NAME", "failure_curve_checkpointed")
    results_base = get_result_root() / experiment / f"seed{args.seed}"
    results_base = results_base / f"{args.dataset_size_limit}edits"
    if args.order_id > 0:
        results_base = results_base / f"order{args.order_id}"
    return results_base


def _make_eval_fn(fast_mode: bool = False):
    """Create mega_batch_eval function from the shared module."""
    ns = {}
    exec(get_mega_batch_eval_source(), ns)
    mbe_fn = ns["_mega_batch_eval"]

    if fast_mode:
        def fast_eval(model, tok, records, template, num_edits, case_ids, exec_time):
            batch_records = [r for r in records if r["case_id"] in case_ids[-num_edits:]]
            mbe_fn(model, tok, batch_records, template, num_edits, case_ids, exec_time, batch_size=2)
        return fast_eval
    return mbe_fn


def _load_c0_patched_apply(alphaedit_root: Path, c0_weight: float):
    """Load AlphaEdit with C₀ injection via exec(compile(AlphaEdit_main.py)).

    This patches the solve line to include α·C₀ in the LHS.
    Returns the patched (apply_fn, get_cov_fn).
    """
    ae_path = alphaedit_root / "AlphaEdit" / "AlphaEdit_main.py"
    ae_source = ae_path.read_text()

    ae_source = ae_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
    ae_source = ae_source.replace("from .compute_z", "from AlphaEdit.compute_z")
    ae_source = ae_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")

    solve_anchor = (
        '        upd_matrix = torch.linalg.solve(\n'
        '                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T\n'
        '        )'
    )
    assert solve_anchor in ae_source, "AlphaEdit solve anchor not found"

    solve_replacement = f"""        # === AlphaEdit+C₀: augmented solve with covariance ===
        _cov_for_layer = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer), hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype)
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ ({c0_weight} * _cov_for_layer.float() + layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )"""

    ae_source = ae_source.replace(solve_anchor, solve_replacement, 1)

    ae_ns = {
        "__name__": "AlphaEdit.AlphaEdit_main",
        "__file__": str(ae_path),
    }
    exec(compile(ae_source, str(ae_path), "exec"), ae_ns)
    print(f"[AlphaEdit+C0] Patched: c0_weight={c0_weight}")
    return ae_ns["apply_AlphaEdit_to_model"], ae_ns["get_cov"]


def run(args: argparse.Namespace) -> None:
    alphaedit_root = get_alphaedit_root()
    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    _seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Resolve checkpoint directory
    ckpt_alg_name = args.alg_name
    if args.inject_c0:
        ckpt_alg_name = f"{args.alg_name}-C0-{args.c0_weight}"

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        _exp_type = "comparison_ordered" if args.order_id > 0 else "failure_curve"
        _exp_config = ExperimentConfig(
            base_alg=ckpt_alg_name, seed=args.seed,
            model_name=args.model_name,
            experiment_type=_exp_type, order_id=args.order_id,
        )
        ckpt_dir = _exp_config.checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve results directory
    results_dir = _resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine start batch (auto-detect from checkpoint)
    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
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

    # Print config
    eval_mode = "Milestone" if args.eval_at_checkpoints_only else ("Fast" if args.fast_checkpoint else "Full")
    print(f"{'=' * 70}")
    print("Checkpoint-Based Failure Curve Runner (harness-based)")
    print(f"  Algorithm:       {args.alg_name}")
    if args.inject_c0:
        print(f"  C₀ injection:    ENABLED (α={args.c0_weight})")
    print(f"  Num edits/batch: {args.num_edits}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Resume from:     batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:   every {args.save_interval} batches")
    print(f"  Evaluation:      {eval_mode}")
    print(f"  Checkpoint dir:  {ckpt_dir}")
    print(f"  Started:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Load model + tokenizer
    model, tok = load_model_and_tok(args.model_name)

    # Load dataset
    dataset = load_dataset(
        args.ds_name, args.dataset_size_limit, alphaedit_root,
        dataset_override=args.dataset_override,
    )

    # Shuffle if order_id > 0
    if args.order_id > 0:
        _r = random.Random(args.order_id)
        _r.shuffle(dataset)
        print(f"  Dataset shuffled with order_id={args.order_id}")

    # Import algorithm
    sys.path.insert(0, str(alphaedit_root))
    if args.inject_c0:
        apply_fn, _ = _load_c0_patched_apply(alphaedit_root, args.c0_weight)
        from AlphaEdit import AlphaEditHyperParams as HParams
    elif args.alg_name == "AlphaEdit" or args.alg_name.startswith("AlphaEdit"):
        from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model as apply_fn
        from AlphaEdit import AlphaEditHyperParams as HParams
    else:
        from memit.memit_main import apply_memit_to_model as apply_fn
        from memit import MEMITHyperParams as HParams

    # Load hparams
    alg_for_hparams = "AlphaEdit" if "AlphaEdit" in args.alg_name else "MEMIT"
    hparams_path = alphaedit_root / "hparams" / alg_for_hparams / args.hparams_fname
    hparams = HParams.from_json(hparams_path)

    # Load null-space projection P for AlphaEdit
    P = None
    if "AlphaEdit" in args.alg_name:
        p_path = alphaedit_root / "null_space_project.pt"
        if p_path.exists():
            P = torch.load(str(p_path), map_location="cpu")
            print(f"  Loaded null-space projection P from {p_path}")
        else:
            raise FileNotFoundError(
                f"P matrix not found at {p_path}. Run link_stats.sh first."
            )

    # Load checkpoint if resuming
    cache_c = None
    if start_from_batch > 0:
        ckpt_result = load_checkpoint(
            model, hparams, str(ckpt_dir), start_from_batch - 1,
            extra_state_keys=["cache_c.pt"] if "AlphaEdit" in args.alg_name else None,
        )
        if ckpt_result.get("cache_c.pt") is not None:
            cache_c = ckpt_result["cache_c.pt"]
            print(f"  [CHECKPOINT] Loaded cache_c (shape: {cache_c.shape})")

    # Build eval function
    mbe_fn = _make_eval_fn(fast_mode=args.fast_checkpoint)

    # Build hooks
    def before_edit(batch_idx, model, records, hparams):
        if should_skip(batch_idx, start_from_batch):
            return  # harness will still call apply_fn, we need a different mechanism

    def after_edit(batch_idx, model, records, hparams, edit_extra, exec_time):
        nonlocal cache_c
        # AlphaEdit returns cache_c as edit_extra
        if edit_extra is not None and "AlphaEdit" in args.alg_name:
            cache_c = edit_extra
        if should_save(batch_idx, args.save_interval):
            extra_state = {}
            if cache_c is not None and "AlphaEdit" in args.alg_name:
                extra_state["cache_c.pt"] = cache_c.cpu()
            save_checkpoint(
                batch_idx, model, hparams, str(ckpt_dir), args.num_edits,
                extra_state=extra_state,
                metadata={"alg_name": args.alg_name, "seed": args.seed},
            )
        print(f"Execution took {exec_time}", flush=True)

    def should_eval_fn(batch_idx):
        if args.eval_at_checkpoints_only:
            return should_save(batch_idx, args.save_interval)
        return True

    def extra_kwargs(batch_idx):
        kw = {}
        if "AlphaEdit" in args.alg_name:
            if P is not None:
                kw["P"] = P
            if cache_c is not None:
                kw["cache_c"] = cache_c
        return kw

    hooks = ExperimentHooks(
        after_edit=after_edit,
        should_eval=should_eval_fn,
        eval_fn=mbe_fn,
        extra_apply_kwargs=extra_kwargs,
    )

    # Determine dir_name for results
    dir_name = args.alg_name
    if args.inject_c0:
        dir_name = ckpt_alg_name

    alg_results_dir = results_dir / dir_name

    # Run
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

    # Free editing tensors before final eval summary
    import gc
    if cache_c is not None:
        del cache_c
    gc.collect()
    torch.cuda.empty_cache()
    _mem_free = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
    print(f"  [CHECKPOINT] Freed editing tensors before eval ({_mem_free:.1f} GiB free)")

    # Metadata
    results_meta_dir = get_result_root() / "metadata"
    results_meta_dir.mkdir(parents=True, exist_ok=True)
    _ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "algorithm": args.alg_name,
        "dataset_size_limit": args.dataset_size_limit,
        "checkpoint_dir": str(ckpt_dir),
        "runner": "checkpoint_runner (harness-based)",
    }
    meta_path = results_meta_dir / f"run_seed{args.seed}_{args.alg_name}_ckpt_{args.dataset_size_limit}_{_ts}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to: {meta_path}")

    print(f"\n{'=' * 70}")
    print("Checkpoint run completed.")
    print(f"  Finished:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Batches:     {summary['batches_run']}")
    print(f"  Edits:       {summary['total_edits']}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint-based failure curve runner (harness-based)"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", required=True)
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--start_from_batch", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=None)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true")
    parser.add_argument("--order_id", type=int, default=0)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--inject_c0", action="store_true")
    parser.add_argument("--c0_weight", type=float, default=15000.0)
    parser.add_argument("--nullspace_threshold", type=float, default=None)
    parser.add_argument("--dataset_override", type=str, default=None)
    parser.add_argument("--retention_probe_batches", type=str, default=None)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
