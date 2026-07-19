#!/usr/bin/env python3
"""
4-Cell Factorial Ablation Runner.

Runs 4 editing configurations in sequence on the same dataset to isolate
the contributions of history tracking, ridge regularization, and null-space
projection to AlphaEdit's performance.

Design (from advisor):
  Cell A: MEMIT            — No history, No ridge, No projection
                             lhs = α·C₀ + K_new@K_new^T
  Cell B: MEMIT-seq        — History (cache_c), No ridge, No projection
                             lhs = α·C₀ + C_hist + K_new@K_new^T
  Cell C: MEMIT-seq+ridge  — History, Ridge (λI), No projection
                             lhs = α·C₀ + C_hist + K_new@K_new^T + L2·I
  Cell D: AlphaEdit        — History, Ridge (L2), Projection (P)
                             Full AlphaEdit with P-projected solve

Key distinction (from advisor):
  - α·C₀ (mom2_update_weight × covariance) ≠ λI (isotropic ridge)
  - α scales original-distribution covariance (preserves pretrained directions)
  - λI stabilizes the solve and penalizes update magnitude in all directions
  - For cells B and C: keep MEMIT-seq α unchanged; cell C adds +L2·I using
    AlphaEdit's released L2 value

Implementation: Sequential single-model-load pattern with state_dict restore.
Each cell runs the full edit sequence, restoring base weights between cells.
Uses checkpoint_runner backbone for long runs.

Usage:
    python src/runners/factorial_ablation_runner.py \\
        --seed 42 --dataset_size_limit 3000 --num_edits 100 \\
        --save_interval 10 --order_id 0

    # Run specific cells only:
    python src/runners/factorial_ablation_runner.py \\
        --seed 42 --cells A,D --dataset_size_limit 2000

Environment:
    FAST_CHECKPOINT=true    Fast evaluation mode
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
from source_patches import patch_evaluate_file
from eval_config import hash_eval_config


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    return get_project_root() / "vendor" / "AlphaEdit"


# Cell definitions
CELLS = {
    "A": {
        "name": "MEMIT (vanilla)",
        "alg_name": "MEMIT",
        "lambda_prev": 0.0,
        "lambda_delta": 0.0,
        "description": "No history, no ridge, no projection",
    },
    "B": {
        "name": "MEMIT-seq",
        "alg_name": "MEMIT",  # Uses memit_sequential_runner
        "lambda_prev": 1.0,   # mom2_update_weight default as coefficient
        "lambda_delta": 0.0,
        "description": "History (cache_c accumulated), no ridge, no projection",
    },
    "C": {
        "name": "MEMIT-seq+ridge",
        "alg_name": "MEMIT",  # Uses memit_sequential_runner
        "lambda_prev": 1.0,
        "lambda_delta": 10.0,  # AlphaEdit's released L2 value
        "description": "History + ridge (λ=L2), no projection",
    },
    "D": {
        "name": "AlphaEdit",
        "alg_name": "AlphaEdit",
        "lambda_prev": None,  # N/A — uses projection
        "lambda_delta": None,
        "description": "Full AlphaEdit: history + ridge + null-space projection",
    },
}


def run_cell(
    cell_id: str,
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    save_interval: int,
    downstream_eval_steps: int,
    order_id: int,
    fast_checkpoint: bool,
    eval_at_checkpoints_only: bool,
    project_root: Path,
) -> int:
    """Run a single factorial cell. Returns exit code."""
    cell = CELLS[cell_id]
    print(f"\n{'=' * 70}")
    print(f"FACTORIAL CELL {cell_id}: {cell['name']}")
    print(f"  {cell['description']}")
    print(f"{'=' * 70}")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if cell_id == "D":
        # Cell D: use checkpoint_runner with AlphaEdit
        # Force --start_from_batch 0 to avoid resuming stale failure curve checkpoints
        cmd = [
            sys.executable, str(project_root / "src" / "runners" / "checkpoint_runner.py"),
            "--seed", str(seed),
            "--alg_name", "AlphaEdit",
            "--model_name", model_name,
            "--hparams_fname", hparams_fname,
            "--ds_name", ds_name,
            "--dataset_size_limit", str(dataset_size_limit),
            "--num_edits", str(num_edits),
            "--save_interval", str(save_interval),
            "--downstream_eval_steps", str(downstream_eval_steps),
            "--order_id", str(order_id),
            "--start_from_batch", "0",
        ]
        if fast_checkpoint:
            cmd.append("--fast_checkpoint")
        elif eval_at_checkpoints_only:
            cmd.append("--eval_at_checkpoints_only")

    elif cell_id == "A":
        # Cell A: use checkpoint_runner with MEMIT (no sequential reg)
        # Force --start_from_batch 0 to avoid resuming stale failure curve checkpoints
        cmd = [
            sys.executable, str(project_root / "src" / "runners" / "checkpoint_runner.py"),
            "--seed", str(seed),
            "--alg_name", "MEMIT",
            "--model_name", model_name,
            "--hparams_fname", hparams_fname,
            "--ds_name", ds_name,
            "--dataset_size_limit", str(dataset_size_limit),
            "--num_edits", str(num_edits),
            "--save_interval", str(save_interval),
            "--downstream_eval_steps", str(downstream_eval_steps),
            "--order_id", str(order_id),
            "--start_from_batch", "0",
        ]
        if fast_checkpoint:
            cmd.append("--fast_checkpoint")
        elif eval_at_checkpoints_only:
            cmd.append("--eval_at_checkpoints_only")

    else:
        # Cells B, C: use memit_sequential_runner
        cmd = [
            sys.executable, str(project_root / "src" / "runners" / "memit_sequential_runner.py"),
            "--seed", str(seed),
            "--model_name", model_name,
            "--hparams_fname", hparams_fname,
            "--ds_name", ds_name,
            "--dataset_size_limit", str(dataset_size_limit),
            "--num_edits", str(num_edits),
            "--lambda_prev", str(cell["lambda_prev"]),
            "--lambda_delta", str(cell["lambda_delta"]),
            "--downstream_eval_steps", str(downstream_eval_steps),
            "--order_id", str(order_id),
        ]
        if fast_checkpoint:
            cmd.append("--fast_checkpoint")

    result = subprocess.run(cmd, env=env)
    return result.returncode


def run(args: argparse.Namespace) -> None:
    """Run factorial ablation across selected cells."""
    project_root = get_project_root()
    alphaedit_root = get_alphaedit_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    model_name = resolve_model_path(args.model_name)

    # Parse cells
    cells_to_run = [c.strip() for c in args.cells.split(",")]
    for c in cells_to_run:
        if c not in CELLS:
            print(f"ERROR: Unknown cell '{c}'. Valid: {list(CELLS.keys())}")
            sys.exit(1)

    fast_checkpoint = args.fast_checkpoint or os.environ.get("FAST_CHECKPOINT") == "true"
    eval_at_checkpoints_only = args.eval_at_checkpoints_only

    print(f"{'=' * 70}")
    print("Factorial Ablation Runner")
    print(f"  Cells:       {cells_to_run}")
    print(f"  Seed:        {args.seed}")
    print(f"  Order ID:    {args.order_id}")
    print(f"  Dataset:     {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:   {args.num_edits}")
    print(f"  Fast mode:   {fast_checkpoint}")
    print(f"  Model:       {args.model_name}")
    print(f"  Started:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    results = {}
    for cell_id in cells_to_run:
        rc = run_cell(
            cell_id=cell_id,
            seed=args.seed,
            cuda_device=args.cuda_device,
            model_name=model_name,
            hparams_fname=args.hparams_fname,
            ds_name=args.ds_name,
            dataset_size_limit=args.dataset_size_limit,
            num_edits=args.num_edits,
            save_interval=args.save_interval,
            downstream_eval_steps=args.downstream_eval_steps,
            order_id=args.order_id,
            fast_checkpoint=fast_checkpoint,
            eval_at_checkpoints_only=eval_at_checkpoints_only,
            project_root=project_root,
        )
        results[cell_id] = "success" if rc == 0 else f"failed (rc={rc})"
        if rc != 0:
            print(f"\nWARNING: Cell {cell_id} failed with return code {rc}. Continuing...")

    # Save ablation metadata
    results_dir = project_root / "results" / "factorial_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": "factorial_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "order_id": args.order_id,
        "cells_run": cells_to_run,
        "cell_results": results,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "model_name": args.model_name,
        "eval_config_hash": hash_eval_config(),
        "cell_definitions": {k: v for k, v in CELLS.items() if k in cells_to_run},
    }
    meta_path = results_dir / f"metadata_seed{args.seed}_order{args.order_id}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Factorial Ablation Complete")
    print(f"  Results: {results}")
    print(f"  Metadata: {meta_path}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Exit with error if any cell failed
    if any("failed" in v for v in results.values()):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="4-cell factorial ablation: history × ridge × projection"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get(
        "MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=3000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--order_id", type=int, default=0)
    parser.add_argument("--cells", default="A,B,C,D",
                        help="Comma-separated cell IDs to run (default: A,B,C,D)")
    parser.add_argument("--fast_checkpoint", action="store_true")
    parser.add_argument("--eval_at_checkpoints_only", action="store_true")
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
