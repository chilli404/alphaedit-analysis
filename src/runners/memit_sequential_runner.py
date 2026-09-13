#!/usr/bin/env python3
"""
MEMIT+SeqReg: Non-projected analogue of AlphaEdit's sequential regularization.

Uses evaluate_harness.run_experiment() + seqreg_hooks() instead of
exec(compile(evaluate.py)) + exec(compile(memit_main.py)).

Scientific Question:
    Does MEMIT with AlphaEdit-like sequential regularization (Eq. 12) close
    the performance gap, or is the null-space projection P still necessary?

Usage:
    python src/runners/memit_sequential_runner.py \
        --seed 42 --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \
        --lambda_prev 1.0 --lambda_delta 1.0 \
        --cache_strategy recent --cache_max 20 \
        --downstream_eval_steps 10 --conserve_memory
"""

import argparse
import json
import os
import random
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
from algorithms.hooks import AlgorithmHooks, compose_hooks
from algorithms.hook_presets import seqreg_hooks
from algorithms.memit_with_hooks import apply_memit_with_hooks


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_eval_fn(fast_mode: bool = False):
    ns = {}
    exec(get_mega_batch_eval_source(), ns)
    mbe_fn = ns["_mega_batch_eval"]

    if fast_mode:
        def fast_eval(model, tok, records, template, num_edits, case_ids, exec_time):
            batch_records = [r for r in records if r["case_id"] in case_ids[-num_edits:]]
            mbe_fn(model, tok, batch_records, template, num_edits, case_ids, exec_time, batch_size=2)
        return fast_eval
    return mbe_fn


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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parse cache_max
    cache_max = None if args.cache_max == "none" else int(args.cache_max)
    mom2_override = args.mom2_override
    ordering = getattr(args, 'ordering', None)

    # Build variant name and paths via ExperimentConfig
    exp_config = ExperimentConfig(
        base_alg="MEMIT", seed=args.seed, ordering=ordering,
        model_name=args.model_name,
        lambda_prev=args.lambda_prev, lambda_delta=args.lambda_delta,
        cache_max=cache_max, mom2_override=mom2_override,
        experiment_type="failure_curve",
    )
    variant_name = exp_config.memit_seq_variant_name

    # Checkpoint directory
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = exp_config.checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Results directory
    if ordering:
        results_dir = (
            get_result_root() / "matched_ordering" / ordering
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    else:
        results_dir = (
            get_result_root() / "failure_curve_checkpointed"
            / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect resume point
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
    print(f"\n{'=' * 70}")
    print("MEMIT+SeqReg Runner (harness-based)")
    print(f"  Seed:           {args.seed}")
    print(f"  lambda_prev:    {args.lambda_prev}")
    print(f"  lambda_delta:   {args.lambda_delta}")
    print(f"  alpha (mom2):   {mom2_override if mom2_override is not None else 'hparams default (15000)'}")
    print(f"  Cache strategy: {args.cache_strategy}")
    print(f"  Cache max:      {cache_max}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:      {args.num_edits}")
    print(f"  Total batches:  {total_batches}")
    print(f"  Resume from:    batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:  every {args.save_interval} batches")
    print(f"  Evaluation:     {eval_mode}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Model:          {args.model_name}")
    if args.dataset_override:
        print(f"  Dataset override: {args.dataset_override}")
    if args.debug_freeze_batch is not None:
        print(f"  DEBUG FREEZE:   batch {args.debug_freeze_batch}")
    print(f"  Variant:        {variant_name}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
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

    # Load hparams
    sys.path.insert(0, str(alphaedit_root))
    from memit import MEMITHyperParams
    hparams_path = alphaedit_root / "hparams" / "MEMIT" / args.hparams_fname
    hparams = MEMITHyperParams.from_json(hparams_path)

    # Build algorithm hooks — SeqReg augmentation replaces memit_main.py exec(compile())
    algo_hooks = seqreg_hooks(
        lambda_prev=args.lambda_prev,
        lambda_delta=args.lambda_delta,
        cache_strategy=args.cache_strategy,
        cache_max=cache_max,
    )
    algo_state = algo_hooks.get_state()

    # Load checkpoint if resuming
    if start_from_batch > 0:
        ckpt_result = load_checkpoint(
            model, hparams, str(ckpt_dir), start_from_batch - 1,
            extra_state_keys=["prev_cache.pt", "mechanism_log.jsonl"],
        )
        if ckpt_result.get("prev_cache.pt") is not None:
            algo_state["prev_cache"] = ckpt_result["prev_cache.pt"]
            total_keys = sum(sum(k.shape[1] for k in v) for v in algo_state["prev_cache"].values())
            print(f"  [CHECKPOINT] Loaded prev_cache ({len(algo_state['prev_cache'])} layers, {total_keys} total keys)")
        if ckpt_result.get("mechanism_log.jsonl") is not None:
            algo_state["mechanism_log"] = ckpt_result["mechanism_log.jsonl"]
            print(f"  [CHECKPOINT] Loaded {len(algo_state['mechanism_log'])} log entries")
        algo_state["batch_idx"] = [start_from_batch]

    # Build eval function
    mbe_fn = _make_eval_fn(fast_mode=args.fast_checkpoint)

    # Wrap apply_fn to pass hooks and state
    def apply_fn(model, tok, requests, hparams, **kwargs):
        return apply_memit_with_hooks(
            model, tok, requests, hparams,
            hooks=algo_hooks, state=algo_state,
            **kwargs,
        )

    # Build experiment hooks
    def after_edit(batch_idx, model, records, hparams, edit_extra, exec_time):
        algo_state["batch_idx"] = [batch_idx + 1]

        if should_save(batch_idx, args.save_interval):
            save_checkpoint(
                batch_idx, model, hparams, str(ckpt_dir), args.num_edits,
                extra_state={
                    "prev_cache.pt": algo_state.get("prev_cache", {}),
                    "mechanism_log.jsonl": algo_state.get("mechanism_log", []),
                },
                metadata={
                    "lambda_prev": args.lambda_prev,
                    "lambda_delta": args.lambda_delta,
                    "mom2_override": mom2_override,
                    "cache_strategy": args.cache_strategy,
                    "batch_idx_counter": algo_state["batch_idx"][0],
                },
            )
        print(f"Execution took {exec_time}", flush=True)

    def should_eval_fn(batch_idx):
        if args.eval_at_checkpoints_only:
            return should_save(batch_idx, args.save_interval)
        return True

    hooks = ExperimentHooks(
        after_edit=after_edit,
        should_eval=should_eval_fn,
        eval_fn=mbe_fn,
    )

    alg_results_dir = results_dir / variant_name

    # Run
    summary = run_experiment(
        model=model,
        tok=tok,
        hparams=hparams,
        dataset=dataset,
        apply_fn=apply_fn,
        alg_name="MEMIT",
        num_edits=args.num_edits,
        results_dir=alg_results_dir,
        ds_name=args.ds_name,
        conserve_memory=args.conserve_memory,
        hooks=hooks,
    )

    # Write mechanism log (re-ensure parent dir exists — S3 FUSE may drop it)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = results_dir / f"log_seed{args.seed}_lp{args.lambda_prev}_ld{args.lambda_delta}_{timestamp}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log_entries = algo_state.get("mechanism_log", [])
    with open(output_jsonl, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"\n[SeqReg] Log written: {output_jsonl} ({len(log_entries)} entries)")

    # Save metadata
    metadata = {
        "experiment": "memit_seqreg_ridge",
        "seed": args.seed,
        "order_id": args.order_id,
        "lambda_prev": args.lambda_prev,
        "lambda_delta": args.lambda_delta,
        "mom2_override": mom2_override,
        "cache_strategy": args.cache_strategy,
        "cache_max": cache_max,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "ds_name": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "eval_config_hash": hash_eval_config(),
        "runner": "memit_sequential_runner (harness-based)",
        "variant_name": variant_name,
    }
    meta_path = results_dir / f"metadata_seed{args.seed}_lp{args.lambda_prev}_ld{args.lambda_delta}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("MEMIT+SeqReg completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Batches:   {summary['batches_run']}")
    print(f"  Edits:     {summary['total_edits']}")
    print(f"  Log:       {output_jsonl}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="MEMIT+SeqReg: control baseline for sequential editing"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--lambda_prev", type=float, default=0.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--mom2_override", type=float, default=None)
    parser.add_argument("--cache_strategy", default="all", choices=["recent", "all"])
    parser.add_argument("--cache_max", default="none")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--start_from_batch", type=int, default=-1)
    parser.add_argument("--debug_freeze_batch", type=int, default=None)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true")
    parser.add_argument("--dataset_override", type=str, default=None)
    parser.add_argument("--order_id", type=int, default=0)
    parser.add_argument("--ordering", type=str, default=None)
    parser.add_argument("--continue_from_run", type=str, default=None)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
