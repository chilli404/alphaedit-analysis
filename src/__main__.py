"""Unified CLI for alphaedit-analysis experiments.

Usage:
    uv run python -m src run --method alphaedit --seed 42
    uv run python -m src run --method revive+memit --seed 42 --ordering fb_high
    uv run python -m src list-methods
    uv run python -m src patch
    uv run python -m src smoke-test --skypilot
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_runner_args(method_def, cli_args) -> list[str]:
    """Build the argument list for a Python runner from method defaults + CLI overrides."""
    args = []

    # Merge defaults with CLI overrides
    params = dict(method_def.defaults)

    # CLI overrides
    if cli_args.seed is not None:
        params["seed"] = cli_args.seed
    if cli_args.ordering:
        params["ordering"] = cli_args.ordering
    if cli_args.edits:
        params["dataset_size_limit"] = cli_args.edits
    if cli_args.num_edits:
        params["num_edits"] = cli_args.num_edits
    if cli_args.ds:
        params["ds_name"] = cli_args.ds
    if cli_args.save_interval:
        params["save_interval"] = cli_args.save_interval
    if cli_args.lambda_prev is not None:
        params["lambda_prev"] = cli_args.lambda_prev
    if cli_args.lambda_delta is not None:
        params["lambda_delta"] = cli_args.lambda_delta
    if cli_args.revive_tau is not None:
        params["revive_tau"] = cli_args.revive_tau
    if cli_args.fast_checkpoint:
        params["fast_checkpoint"] = True
    if cli_args.eval_at_checkpoints:
        params["eval_at_checkpoints_only"] = True
    if cli_args.cuda_device is not None:
        params["cuda_device"] = cli_args.cuda_device

    # Set common defaults
    params.setdefault("seed", 42)
    params.setdefault("num_edits", 100)
    params.setdefault("dataset_size_limit", 10000)
    params.setdefault("ds_name", "mcf")
    params.setdefault("cuda_device", "0")
    params.setdefault("downstream_eval_steps", 0)
    params.setdefault("conserve_memory", True)

    # Build argument list
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    return args


def build_shell_env(method_def, cli_args) -> dict:
    """Build environment variables for a shell-based baseline runner."""
    env = dict(os.environ)
    if cli_args.edits:
        env["TARGET_EDITS"] = str(cli_args.edits)
    if cli_args.num_edits:
        env["NUM_EDITS"] = str(cli_args.num_edits)
    if cli_args.ordering:
        env["ORDERING"] = cli_args.ordering
    return env


def cmd_run(args):
    """Run an editing experiment."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from method_registry import get_method

    method = get_method(args.method)
    seed = str(args.seed or 42)

    if method.shell:
        script = PROJECT_ROOT / method.runner
        env = build_shell_env(method, args)
        cmd = ["bash", str(script), seed]
        print(f"[CLI] Running: {' '.join(cmd)}")
        print(f"[CLI] Method: {method.name} ({method.description})")
        result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        sys.exit(result.returncode)
    else:
        runner_path = PROJECT_ROOT / "src" / method.runner
        runner_args = build_runner_args(method, args)
        cmd = ["uv", "run", "python", str(runner_path)] + runner_args
        print(f"[CLI] Running: {' '.join(cmd[:6])}...")
        print(f"[CLI] Method: {method.name} ({method.description})")
        print(f"[CLI] Full args: {' '.join(runner_args)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        sys.exit(result.returncode)


def cmd_list_methods(args):
    """List all registered methods."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from method_registry import list_methods

    methods = list_methods()
    print(f"{'Method':<25s} {'Runner':<45s} Description")
    print(f"{'-'*25} {'-'*45} {'-'*40}")
    for m in methods:
        runner = m.runner
        if m.shell:
            runner += " (shell)"
        print(f"{m.name:<25s} {runner:<45s} {m.description}")


def cmd_patch(args):
    """Apply all vendor and baseline patches."""
    cmd = ["uv", "run", "python", str(PROJECT_ROOT / "scripts" / "patches" / "apply_all.py")]
    if args.vendor_only:
        cmd.append("--vendor-only")
    elif args.baselines_only:
        cmd.append("--baselines-only")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


def cmd_smoke_test(args):
    """Run the GPU smoke test."""
    script = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
    cmd = ["bash", str(script)]
    if args.skypilot:
        cmd.append("--skypilot")
    if args.filter:
        cmd.extend(args.filter)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Unified CLI for alphaedit-analysis experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an editing experiment")
    run_parser.add_argument("--method", "-m", required=True, help="Method name (use 'list-methods' to see all)")
    run_parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed (default: 42)")
    run_parser.add_argument("--ordering", "-o", default=None, help="Ordering stream name (e.g. fb_high_exposure)")
    run_parser.add_argument("--edits", "-e", type=int, default=None, help="Total edits / dataset size limit")
    run_parser.add_argument("--num-edits", type=int, default=None, help="Edits per batch (default: 100)")
    run_parser.add_argument("--ds", default=None, help="Dataset name (mcf, zsre)")
    run_parser.add_argument("--save-interval", type=int, default=None, help="Checkpoint save interval")
    run_parser.add_argument("--lambda-prev", type=float, default=None)
    run_parser.add_argument("--lambda-delta", type=float, default=None)
    run_parser.add_argument("--revive-tau", type=float, default=None)
    run_parser.add_argument("--fast-checkpoint", action="store_true")
    run_parser.add_argument("--eval-at-checkpoints", action="store_true")
    run_parser.add_argument("--cuda-device", default=None)
    run_parser.set_defaults(func=cmd_run)

    # --- list-methods ---
    list_parser = subparsers.add_parser("list-methods", help="List all registered methods")
    list_parser.set_defaults(func=cmd_list_methods)

    # --- patch ---
    patch_parser = subparsers.add_parser("patch", help="Apply all vendor and baseline patches")
    patch_parser.add_argument("--vendor-only", action="store_true")
    patch_parser.add_argument("--baselines-only", action="store_true")
    patch_parser.set_defaults(func=cmd_patch)

    # --- smoke-test ---
    smoke_parser = subparsers.add_parser("smoke-test", help="Run GPU smoke test")
    smoke_parser.add_argument("--skypilot", action="store_true")
    smoke_parser.add_argument("filter", nargs="*", help="Algorithm name filter(s)")
    smoke_parser.set_defaults(func=cmd_smoke_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
