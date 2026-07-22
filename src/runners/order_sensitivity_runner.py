#!/usr/bin/env python3
"""
Edit Order Sensitivity Runner for AlphaEdit.

Tests whether the ORDER of sequential edits affects final performance.
AlphaEdit's null-space projection P is static, so early edits "claim" the
best null-space directions. This experiment shuffles the dataset with
different random seeds and measures variance in final metrics.

Hypothesis: AlphaEdit is significantly more order-sensitive than MEMIT
because its sequential null-space constraint creates path-dependence.

Implementation approach:
  Source injection (same pattern as nullspace_tracker.py). Injects a dataset
  shuffle BEFORE the edit loop begins. The shuffle seed is independent of
  the model/RNG seed, so the same model initialization sees different
  edit orderings.

Output: Standard AlphaEdit/MEMIT result JSONs (same format as MVE),
        tagged with order_seed in metadata JSONL.

Usage:
    python src/order_sensitivity_runner.py \\
        --seed 42 \\
        --order_seed 0 \\
        --alg_name AlphaEdit \\
        --ds_name mcf \\
        --dataset_size_limit 2000 \\
        --num_edits 100 \\
        --downstream_eval_steps 5
"""

import argparse
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
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


# Source anchor from evaluate.py at commit b84624f.
# Shuffle code is injected BEFORE this line (after dataset is loaded, before loop).
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'

# CUDA patch target
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def build_order_script(
    seed: int,
    order_seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    metadata_jsonl: str,
    eval_results_dir: str = "",
) -> str:
    """
    Build an inline Python script that:
    1. Seeds model RNGs with --seed
    2. Injects dataset shuffle with --order_seed before the edit loop
    3. Executes the patched evaluate.py as __main__
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

    script = textwrap.dedent(f"""\
import os, sys, random, json
import numpy as np
import torch

# 1. Seed all sources of randomness (model/compute seed)
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

# 3. Order sensitivity parameters
_order_seed = {order_seed}
_metadata_jsonl = "{metadata_jsonl}"

# 4. Read evaluate.py source
with open("experiments/evaluate.py", "r") as f:
    source = f.read()

# 5. Patch CUDA_VISIBLE_DEVICES
cuda_patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
assert cuda_patch_target in source, (
    "CUDA_VISIBLE_DEVICES patch target not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
source = source.replace(
    cuda_patch_target,
    '# CUDA_VISIBLE_DEVICES managed by order_sensitivity_runner',
)

# 5a. Override RESULTS_DIR
_globals_import = 'from util.globals import *'
assert _globals_import in source, "globals import not found in evaluate.py"
source = source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)
print(f"  [RESULTS_DIR] Overridden to: {eval_results_dir}")

# 6. Inject dataset shuffle BEFORE the edit loop
shuffle_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert shuffle_anchor in source, (
    "Shuffle anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

shuffle_injection = '    # === ORDER SENSITIVITY: shuffle dataset (injected) ===\\n'
shuffle_injection += '    import random as _order_rng_module\\n'
shuffle_injection += '    _order_rng = _order_rng_module.Random(' + str(_order_seed) + ')\\n'
shuffle_injection += '    _shuffled_indices = list(range(len(ds)))\\n'
shuffle_injection += '    _order_rng.shuffle(_shuffled_indices)\\n'
shuffle_injection += '    ds.data = [ds.data[i] for i in _shuffled_indices]\\n'
shuffle_injection += '    print("ORDER SENSITIVITY: shuffled " + str(len(ds)) + " records with order_seed=' + str(_order_seed) + '")\\n'
shuffle_injection += '    # === END order sensitivity ===\\n'
source = source.replace(
    shuffle_anchor,
    shuffle_injection + shuffle_anchor,
    1,
)

# 7. Verify injection succeeded
assert "ORDER SENSITIVITY: shuffle dataset" in source, "Shuffle injection failed"

# 8. Execute
exec(compile(source, "experiments/evaluate.py", "exec"),
     {{
         "__name__": "__main__",
         "__file__": "experiments/evaluate.py",
         "_order_seed": _order_seed,
     }})

# 9. Write metadata
metadata = {{
    "order_seed": _order_seed,
    "model_seed": seed,
    "alg_name": "{alg_name}",
    "dataset_size_limit": {dataset_size_limit},
    "num_edits": {num_edits},
}}
with open(_metadata_jsonl, "a") as f:
    f.write(json.dumps(metadata) + "\\n")

# 10. Final summary
print(f"\\n=== Order sensitivity run complete ===")
print(f"  Algorithm: {alg_name}")
print(f"  Order seed: {{_order_seed}}")
print(f"  Model seed: {{seed}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the order sensitivity experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    # Resolve model path (falls back to Artifactory mirror if HF access fails)
    model_name = resolve_model_path(args.model_name)

    # Validate anchors exist
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    if SHUFFLE_ANCHOR not in eval_source:
        print("ERROR: Shuffle anchor not found in evaluate.py.")
        print("  The upstream code has diverged from pinned commit b84624f.")
        print(f"  Expected: {SHUFFLE_ANCHOR}")
        sys.exit(1)
    if CUDA_PATCH_TARGET not in eval_source:
        print("ERROR: CUDA patch target not found in evaluate.py.")
        sys.exit(1)

    # Output directory
    output_dir = (
        project_root / "results" / "order_sensitivity"
        / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
        / f"order{args.order_seed}" / args.alg_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metadata_jsonl = (
        output_dir
        / f"order_{args.alg_name}_seed{args.seed}_order{args.order_seed}_{timestamp}.jsonl"
    )

    script = build_order_script(
        seed=args.seed,
        order_seed=args.order_seed,
        cuda_device=args.cuda_device,
        alg_name=args.alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        metadata_jsonl=str(metadata_jsonl),
        eval_results_dir=str(output_dir.parent),
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'=' * 70}")
    print("Edit Order Sensitivity Experiment")
    print(f"  Algorithm:   {args.alg_name}")
    print(f"  Order seed:  {args.order_seed}")
    print(f"  Model seed:  {args.seed}")
    print(f"  Dataset:     {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:   {args.num_edits}")
    print(f"  CUDA:        device {args.cuda_device}")
    print(f"  Model:       {args.model_name}")
    print(f"  Metadata:    {metadata_jsonl}")
    print(f"  Started:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Order sensitivity run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Order sensitivity experiment completed.")
    print(f"  Algorithm: {args.alg_name}, order_seed={args.order_seed}")
    print(f"  Metadata: {metadata_jsonl}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Test edit order sensitivity for AlphaEdit vs MEMIT"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Model/RNG seed (controls model initialization)")
    parser.add_argument("--order_seed", type=int, required=True,
                        help="Shuffle seed (controls dataset ordering)")
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", required=True,
                        choices=["AlphaEdit", "MEMIT"],
                        help="Algorithm to test")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=5)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
