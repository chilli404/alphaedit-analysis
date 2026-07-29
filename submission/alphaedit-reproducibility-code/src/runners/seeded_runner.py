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
from source_patches import patch_evaluate_file, build_order_shuffle_injection, SHUFFLE_ANCHOR
from dataset_fingerprint import build_fingerprint_injection
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root


def _resolve_results_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the project-level results directory for this experiment.

    Uses EXPERIMENT_NAME env var if set (SkyPilot), otherwise derives from args.
    Returns the base dir that evaluate.py will use as RESULTS_DIR
    (evaluate.py then appends {Alg}/run_000/ internally).
    """
    experiment = os.environ.get("EXPERIMENT_NAME", "")
    if not experiment:
        # Derive from algorithm + dataset (e.g. "alphaedit_mcf")
        experiment = f"{args.alg_name.lower()}_{args.ds_name}"

    results_base = get_result_root() / experiment / f"seed{args.seed}"
    results_base = results_base / f"{args.dataset_size_limit}edits"

    # For order-sensitivity experiments, add order subdirectory
    order_id = getattr(args, "order_id", 0)
    if "ordered" in experiment or "order" in experiment:
        results_base = results_base / f"order{order_id}"

    return results_base


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
    order_id: int = 0,
    results_dir: str | None = None,
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

    # Build RESULTS_DIR injection (outside f-string to avoid escaping issues)
    if results_dir:
        results_dir_injection = (
            f'\n# 3a. Override RESULTS_DIR to project-level results\n'
            f'_globals_import = \'from util.globals import *\'\n'
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
{results_dir_injection}
# 4. Inject order shuffle (if order_id > 0)
shuffle_anchor = '    for record_chunks in chunks(ds, num_edits):'
if shuffle_anchor in source:
    shuffle_code = {repr(build_order_shuffle_injection(order_id))}
    if shuffle_code:
        source = source.replace(shuffle_anchor, shuffle_code + shuffle_anchor, 1)

# 5. Inject dataset fingerprint
if shuffle_anchor in source:
    fingerprint_code = {repr(build_fingerprint_injection(order_id))}
    source = source.replace(shuffle_anchor, fingerprint_code + shuffle_anchor, 1)

# 6. Execute as __main__ (triggers argparse + main() call)
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

    # Order ID (0 = canonical, >0 = shuffled)
    order_id = getattr(args, "order_id", 0)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "order_id": order_id,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "alphaedit_commit": commit,
        "eval_config_hash": hash_eval_config(),
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

    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = results_dir / f"run_seed{seed}_{args.alg_name}_{args.ds_name}_{args.dataset_size_limit}.json"

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

    # Resolve results directory
    results_dir_override = _resolve_results_dir(args)

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
        order_id=args.order_id,
        results_dir=str(results_dir_override) if results_dir_override else None,
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
    print(f"  Results:    {results_dir_override}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

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

    # Record metadata in the results directory
    results_dir = results_dir_override or get_result_root()
    run_dir = results_dir / args.alg_name / "run_000" if results_dir_override else None
    run_dir_rel = str(run_dir.relative_to(get_project_root())) if run_dir and run_dir.exists() else None

    record_metadata(
        args.seed, args, results_dir,
        experiment_name=os.environ.get("EXPERIMENT_NAME", ""),
        run_dir=run_dir_rel,
        run_id="run_000",
    )

    print(f"\n{'=' * 70}")
    print("Experiment completed successfully.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Results:   {results_dir / args.alg_name / 'run_000'}")
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
    parser.add_argument("--order_id", type=int, default=0,
                        help="Edit ordering ID (0=canonical, >0=shuffle with Random(order_id))")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
