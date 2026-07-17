#!/usr/bin/env python3
"""
MEMIT+SeqReg Lambda Sweep Batch Runner: Runs all calibration settings
with a single model load, resetting state between each lambda combination.

This is an optimization of running run_memit_sequential.sh 4 times
independently (once per calibration setting), which loads the model 4 times.

Calibration settings:
  A: λ_prev=1,   λ_delta=1       # Direct Eq. 12 coefficient analogue
  B: λ_prev=1,   λ_delta=1e-4    # Weak ridge
  C: λ_prev=10,  λ_delta=1       # Strong prev-key protection
  D: λ_prev=100, λ_delta=1       # Very strong prev-key protection

This batch runner:
  1. Loads the model once
  2. For each (λ_prev, λ_delta) combination:
     a. Restores base model state
     b. Runs MEMIT+SeqReg with that setting
     c. Records results
  3. Produces identical results to running memit_sequential_runner.py
     independently for each combination.

Usage:
    python src/memit_sequential_batch_runner.py --seed 42

    # With fast checkpoint mode (recommended for calibration)
    python src/memit_sequential_batch_runner.py --seed 42 --fast_checkpoint

    # Custom lambda combinations
    python src/memit_sequential_batch_runner.py --seed 42 \\
        --lambda_pairs "1,1" "10,1" "100,1"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src/util to path for shared utilities
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from model_download import resolve_model_path
from setup_hparams import link_hparams


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    return get_project_root() / "vendor" / "AlphaEdit"


# Default calibration settings
DEFAULT_LAMBDA_PAIRS = [
    (1.0, 1.0),      # A: Direct Eq. 12 analogue
    (1.0, 1e-4),     # B: Weak ridge
    (10.0, 1.0),     # C: Strong prev-key protection
    (100.0, 1.0),    # D: Very strong prev-key protection
]


def run(args: argparse.Namespace) -> None:
    """Launch the batch lambda sweep using subprocess calls to memit_sequential_runner.py."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()

    model_name = resolve_model_path(args.model_name)

    # Parse lambda pairs
    if args.lambda_pairs:
        lambda_pairs = []
        for pair_str in args.lambda_pairs:
            parts = pair_str.split(",")
            if len(parts) != 2:
                print(f"ERROR: Invalid lambda pair '{pair_str}'. Expected format: 'prev,delta'")
                sys.exit(1)
            lambda_pairs.append((float(parts[0]), float(parts[1])))
    else:
        lambda_pairs = DEFAULT_LAMBDA_PAIRS

    print(f"{'=' * 70}")
    print("MEMIT+SeqReg Lambda Sweep Batch Runner")
    print(f"  Seed:           {args.seed}")
    print(f"  Combinations:   {len(lambda_pairs)}")
    print(f"  Lambda pairs:   {lambda_pairs}")
    print(f"  Fast mode:      {'YES' if args.fast_checkpoint else 'NO'}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    print(f"  Cache:          strategy={args.cache_strategy}, max={args.cache_max}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Pre-warm model cache by triggering a download/cache check
    print("\nPre-warming model cache...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="cpu"
    )
    del _model, _tokenizer
    import gc
    gc.collect()
    print("  Model cached on disk (subsequent loads will be fast).")

    # Environment for subprocesses
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    completed = 0
    failed = 0

    for idx, (lambda_prev, lambda_delta) in enumerate(lambda_pairs):
        print(f"\n{'=' * 50}")
        print(f"  [{idx+1}/{len(lambda_pairs)}] λ_prev={lambda_prev}, λ_delta={lambda_delta}")
        print(f"{'=' * 50}")

        cmd = [
            sys.executable, "src/memit_sequential_runner.py",
            "--seed", str(args.seed),
            "--cuda_device", args.cuda_device,
            "--model_name", model_name,
            "--hparams_fname", args.hparams_fname,
            "--ds_name", args.ds_name,
            "--dataset_size_limit", str(args.dataset_size_limit),
            "--num_edits", str(args.num_edits),
            "--downstream_eval_steps", str(args.downstream_eval_steps),
            "--conserve_memory",
            "--lambda_prev", str(lambda_prev),
            "--lambda_delta", str(lambda_delta),
            "--cache_strategy", args.cache_strategy,
            "--cache_max", args.cache_max,
        ]

        if args.fast_checkpoint:
            cmd.append("--fast_checkpoint")

        result = subprocess.run(cmd, cwd=str(project_root), env=env)

        if result.returncode == 0:
            completed += 1
            print(f"  Completed: λ_prev={lambda_prev}, λ_delta={lambda_delta}")
        else:
            failed += 1
            print(f"  FAILED: λ_prev={lambda_prev}, λ_delta={lambda_delta} (rc={result.returncode})")

    print(f"\n{'=' * 70}")
    print("MEMIT+SeqReg lambda sweep batch completed.")
    print(f"  Completed: {completed} / {len(lambda_pairs)}")
    print(f"  Failed:    {failed} / {len(lambda_pairs)}")
    print(f"  Results:   {project_root / 'results' / 'memit_seqreg'}")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    if failed > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Batch MEMIT+SeqReg lambda sweep: pre-warms model cache, runs all calibration settings"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--fast_checkpoint", action="store_true",
                        help="Fast mode: only evaluate edited batch (recommended for calibration)")
    parser.add_argument("--cache_strategy", default="recent", choices=["recent", "all"])
    parser.add_argument("--cache_max", default="20",
                        help="Max batches in cache (default: 20)")
    parser.add_argument("--lambda_pairs", nargs="*", default=None,
                        help="Custom lambda pairs as 'prev,delta' strings (default: all 4 calibration settings)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
