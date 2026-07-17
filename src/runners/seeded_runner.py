#!/usr/bin/env python3
"""
Seeded reproducibility wrapper for AlphaEdit experiments.

Handles three critical issues with the upstream evaluate.py:
1. No seed argument exposed — we inject seeds for random, numpy, torch
2. Hardcoded CUDA_VISIBLE_DEVICES="1" at line 2 — we patch it out
3. Module-level `args` variable referenced inside main() — we use exec
   with __name__="__main__" to trigger argparse naturally

Usage (standalone):
    python src/seeded_runner.py \\
        --seed 42 \\
        --cuda_device 0 \\
        --alg_name AlphaEdit \\
        --ds_name mcf \\
        --dataset_size_limit 2000 \\
        --num_edits 100 \\
        --downstream_eval_steps 5

Usage (from shell scripts):
    Called by scripts/run_mveN_*.sh with appropriate arguments.
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
from source_patches import patch_evaluate_file


def get_project_root() -> Path:
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


def build_runner_script(
    seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    use_cache: bool,
) -> str:
    """
    Build an inline Python script that:
    1. Seeds all RNGs
    2. Patches CUDA_VISIBLE_DEVICES override in evaluate.py
    3. Executes evaluate.py as __main__ with correct sys.argv
    """
    # Build sys.argv for argparse inside evaluate.py
    argv_parts = [
        "experiments.evaluate",
        f"--alg_name={alg_name}",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
        f"--downstream_eval_steps={downstream_eval_steps}",
        f"--generation_test_interval={generation_test_interval}",
    ]
    if skip_generation_tests:
        argv_parts.append("--skip_generation_tests")
    if conserve_memory:
        argv_parts.append("--conserve_memory")
    if use_cache:
        argv_parts.append("--use_cache")

    argv_str = repr(argv_parts)

    script = textwrap.dedent(f"""\
        import os, sys, random
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

        # 2. Set sys.argv for argparse in evaluate.py
        sys.argv = {argv_str}

        # 3. Read evaluate.py and patch the hardcoded CUDA line
        with open("experiments/evaluate.py", "r") as f:
            source = f.read()

        patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
        if patch_target not in source:
            print("WARNING: CUDA_VISIBLE_DEVICES patch target not found in evaluate.py")
            print("  The upstream code may have changed. Proceeding without patch.")
        else:
            source = source.replace(
                patch_target,
                '# CUDA_VISIBLE_DEVICES managed by seeded_runner',
            )

        # 4. Execute as __main__ (triggers argparse + main() call)
        exec(compile(source, "experiments/evaluate.py", "exec"),
             {{"__name__": "__main__", "__file__": "experiments/evaluate.py"}})
    """)
    return script


def find_latest_run_dir(alg_name: str) -> tuple[str | None, str | None]:
    """Find the most recently created run_NNN directory for this algorithm.

    Returns (run_dir_relative, run_id) or (None, None) if not found.
    run_dir_relative is relative to the project root (e.g. "vendor/AlphaEdit/results/AlphaEdit/run_001").
    """
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()
    alg_results = alphaedit_root / "results" / alg_name

    if not alg_results.exists():
        return None, None

    run_dirs = sorted(
        [d for d in alg_results.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not run_dirs:
        return None, None

    latest = run_dirs[-1]
    run_id = latest.name
    run_dir_relative = str(latest.relative_to(project_root))
    return run_dir_relative, run_id


def record_metadata(
    seed: int,
    args: argparse.Namespace,
    results_dir: Path,
    experiment_name: str | None = None,
    run_dir: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Save run metadata for reproducibility record.

    Args:
        seed: Random seed used for this run.
        args: Parsed command-line arguments.
        results_dir: Project-level results directory.
        experiment_name: Short experiment identifier (e.g. "mve1", "failure_curve").
        run_dir: Relative path to the vendor results directory for this run.
        run_id: The run_NNN identifier (e.g. "run_001").

    Returns:
        Path to the written metadata file.
    """
    alphaedit_root = get_alphaedit_root()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(alphaedit_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "alphaedit_commit": commit,
        "cuda_device": args.cuda_device,
        "experiment": experiment_name or f"{args.alg_name.lower()}_{args.ds_name}",
        "algorithm": args.alg_name,
        "dataset": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "run_dir": run_dir,
        "run_id": run_id,
        "params": {
            "model_name": args.model_name,
            "hparams_fname": args.hparams_fname,
            "downstream_eval_steps": args.downstream_eval_steps,
            "skip_generation_tests": args.skip_generation_tests,
            "generation_test_interval": args.generation_test_interval,
            "conserve_memory": args.conserve_memory,
            "use_cache": args.use_cache,
        },
    }

    metadata_dir = results_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_dir / f"run_seed{seed}_{args.alg_name}_{args.ds_name}_{args.dataset_size_limit}.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_file}")
    return metadata_file



def validate_environment(args: argparse.Namespace) -> None:
    """Pre-flight checks before launching an expensive experiment."""
    alphaedit_root = get_alphaedit_root()

    # Check evaluate.py exists and has expected patch target
    eval_path = alphaedit_root / "experiments" / "evaluate.py"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found")
        sys.exit(1)

    source = eval_path.read_text()
    if 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' not in source:
        print("WARNING: CUDA_VISIBLE_DEVICES patch target not found in evaluate.py")
        print("  The upstream code may have changed format.")

    # Check covariance stats exist (for AlphaEdit)
    if args.alg_name == "AlphaEdit":
        # Canonical path used by link_stats.sh and SkyPilot setup
        model_short = args.model_name.split("/")[-1]
        stats_dir = alphaedit_root / "data" / "stats" / model_short / "wikipedia_stats"
        if stats_dir.exists() and any(stats_dir.glob("*.npz")):
            n_files = len(list(stats_dir.glob("*.npz")))
            print(f"  Stats:    {n_files} NPZ files in {stats_dir.relative_to(alphaedit_root)}")
        else:
            print(f"WARNING: Covariance stats not found at: {stats_dir}")
            print("  Run scripts/link_stats.sh first.")


def run(args: argparse.Namespace) -> None:
    """Launch the experiment as a subprocess with full seed control."""
    alphaedit_root = get_alphaedit_root()
    results_dir = get_project_root() / "results"

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    # Link project hparams into submodule
    link_hparams()
    patch_evaluate_file(alphaedit_root)

    # Run pre-flight validation
    validate_environment(args)

    # Resolve model path (falls back to Artifactory mirror if HF access fails)
    model_name = resolve_model_path(args.model_name)

    # Build the inline runner script
    script = build_runner_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name=args.alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        use_cache=args.use_cache,
    )

    # Set up environment for subprocess
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"  # Prevent non-deterministic tokenizer threading

    if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
        print("WARNING: HF_TOKEN not set. Model download may fail.")

    print(f"{'=' * 70}")
    print("AlphaEdit Seeded Runner")
    print(f"  Algorithm:  {args.alg_name}")
    print(f"  Dataset:    {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:  {args.num_edits}")
    print(f"  Seed:       {args.seed}")
    print(f"  CUDA:       device {args.cuda_device}")
    print(f"  Model:      {args.model_name}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Find the uv executable to use the project's venv
    cmd = [
        sys.executable, "-c", script
    ]

    result = subprocess.run(
        cmd,
        cwd=str(alphaedit_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Experiment failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Detect run_dir and run_id created by evaluate.py
    run_dir_rel, run_id = find_latest_run_dir(args.alg_name)

    # Record metadata with run_dir and run_id
    record_metadata(
        args.seed, args, results_dir,
        run_dir=run_dir_rel,
        run_id=run_id,
    )

    print(f"\n{'=' * 70}")
    print("Experiment completed successfully.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Results:   {alphaedit_root / 'results' / args.alg_name}")
    if run_id:
        print(f"  Run ID:    {run_id}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Seeded reproducibility wrapper for AlphaEdit experiments"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True, help="Random seed for full reproducibility")
    parser.add_argument("--cuda_device", default="0", help="CUDA device index (default: 0)")

    # Experiment parameters (mirror evaluate.py's interface)
    parser.add_argument("--alg_name", required=True, choices=["AlphaEdit", "MEMIT", "ROME"])
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", required=True, choices=["mcf", "cf", "zsre", "mquake"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=5)
    parser.add_argument("--skip_generation_tests", action="store_true")
    parser.add_argument("--generation_test_interval", type=int, default=1)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--use_cache", action="store_true")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
