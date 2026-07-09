#!/usr/bin/env python3
"""
Cache Mitigation Runner for AlphaEdit.

Tests whether simple cache_c management strategies can extend AlphaEdit's
operational range beyond its natural null-space saturation point.

Three mitigation strategies:
  1. SVD truncation: Every K batches, keep only top-r singular vectors of cache_c
  2. Exponential decay: Before each batch, apply cache_c *= decay_factor
  3. Periodic reset: Every K batches, zero out cache_c entirely

Implementation approach:
  Same source-injection pattern as nullspace_tracker.py — injects mitigation
  code into evaluate.py at the POST_EDIT_ANCHOR (after cache_c is updated,
  before the next batch begins). Pinned to commit b84624f.

Output: Standard AlphaEdit result JSONs + mitigation metadata JSONL.

Usage:
    python src/cache_mitigation_runner.py \\
        --seed 42 \\
        --strategy svd_truncation \\
        --truncation_interval 5 \\
        --retain_ratio 0.75 \\
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

from model_download import resolve_model_path
from setup_hparams import link_hparams


def get_project_root() -> Path:
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


# Source anchor from evaluate.py at commit b84624f.
# Mitigation code is injected BEFORE this line (after cache_c has been updated).
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'

# CUDA patch target
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def build_mitigation_script(
    seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    strategy: str,
    truncation_interval: int,
    retain_ratio: float,
    decay_factor: float,
    reset_interval: int,
    metadata_jsonl: str,
) -> str:
    """
    Build an inline Python script that:
    1. Seeds all RNGs
    2. Injects cache mitigation code into evaluate.py at the post-edit anchor
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

# 1. Seed all sources of randomness
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

# 3. Mitigation parameters (available in exec namespace)
_mitigation_strategy = "{strategy}"
_truncation_interval = {truncation_interval}
_retain_ratio = {retain_ratio}
_decay_factor = {decay_factor}
_reset_interval = {reset_interval}
_metadata_jsonl = "{metadata_jsonl}"
_mitigation_log = []

def _apply_mitigation(cnt, cache_c):
    \"\"\"Apply cache_c mitigation after each edit batch.\"\"\"
    applied = False

    if _mitigation_strategy == "svd_truncation":
        if (cnt + 1) % _truncation_interval == 0:
            for i in range(cache_c.shape[0]):
                layer_cache = cache_c[i].float()
                if layer_cache.abs().max() > 0:
                    U, S, Vt = torch.linalg.svd(layer_cache, full_matrices=False)
                    r = max(1, int(len(S) * _retain_ratio))
                    cache_c[i] = (U[:, :r] * S[:r]) @ Vt[:r, :]
            applied = True

    elif _mitigation_strategy == "exponential_decay":
        cache_c *= _decay_factor
        applied = True

    elif _mitigation_strategy == "periodic_reset":
        if (cnt + 1) % _reset_interval == 0:
            cache_c.zero_()
            applied = True

    if applied:
        _mitigation_log.append({{
            "batch_idx": cnt,
            "strategy": _mitigation_strategy,
            "applied": True,
        }})

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
    '# CUDA_VISIBLE_DEVICES managed by cache_mitigation_runner',
)

# 6. Inject mitigation code BEFORE POST_EDIT_ANCHOR
post_anchor = '        elif alg_name == "MEMIT_prune":'
assert post_anchor in source, (
    "Post-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

mitigation_injection = '''        # === CACHE MITIGATION: apply strategy (injected) ===
        if alg_name == "AlphaEdit" and '_apply_mitigation' in dir():
            _apply_mitigation(cnt, cache_c)
        # === END cache mitigation ===
'''
source = source.replace(
    post_anchor,
    mitigation_injection + post_anchor,
    1,
)

# 7. Verify injection succeeded
assert "CACHE MITIGATION: apply strategy" in source, "Mitigation injection failed"

# 8. Execute
exec(compile(source, "experiments/evaluate.py", "exec"),
     {{
         "__name__": "__main__",
         "__file__": "experiments/evaluate.py",
         "_mitigation_strategy": _mitigation_strategy,
         "_truncation_interval": _truncation_interval,
         "_retain_ratio": _retain_ratio,
         "_decay_factor": _decay_factor,
         "_reset_interval": _reset_interval,
         "_apply_mitigation": _apply_mitigation,
         "_mitigation_log": _mitigation_log,
     }})

# 9. Write metadata
metadata = {{
    "strategy": _mitigation_strategy,
    "truncation_interval": _truncation_interval,
    "retain_ratio": _retain_ratio,
    "decay_factor": _decay_factor,
    "reset_interval": _reset_interval,
    "seed": seed,
    "total_mitigations_applied": len(_mitigation_log),
}}
with open(_metadata_jsonl, "a") as f:
    f.write(json.dumps(metadata) + "\\n")

# 10. Final summary
print(f"\\n=== Cache mitigation complete ===")
print(f"  Strategy: {{_mitigation_strategy}}")
print(f"  Mitigations applied: {{len(_mitigation_log)}}")
print(f"  Metadata: {{_metadata_jsonl}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the mitigation experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()

    # Resolve model path (falls back to Artifactory mirror if HF access fails)
    model_name = resolve_model_path(args.model_name)

    # Validate anchor exists
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    if POST_EDIT_ANCHOR not in eval_source:
        print("ERROR: Post-edit anchor not found in evaluate.py.")
        print("  The upstream code has diverged from pinned commit b84624f.")
        sys.exit(1)
    if CUDA_PATCH_TARGET not in eval_source:
        print("ERROR: CUDA patch target not found in evaluate.py.")
        sys.exit(1)

    # Output directory
    output_dir = project_root / "results" / "mitigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metadata_jsonl = output_dir / f"mitigation_{args.strategy}_seed{args.seed}_{timestamp}.jsonl"

    # Build strategy description for logging
    if args.strategy == "svd_truncation":
        strategy_desc = f"SVD truncation (K={args.truncation_interval}, retain={args.retain_ratio})"
    elif args.strategy == "exponential_decay":
        strategy_desc = f"Exponential decay (factor={args.decay_factor})"
    elif args.strategy == "periodic_reset":
        strategy_desc = f"Periodic reset (K={args.reset_interval})"
    else:
        strategy_desc = args.strategy

    script = build_mitigation_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name="AlphaEdit",
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        strategy=args.strategy,
        truncation_interval=args.truncation_interval,
        retain_ratio=args.retain_ratio,
        decay_factor=args.decay_factor,
        reset_interval=args.reset_interval,
        metadata_jsonl=str(metadata_jsonl),
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'=' * 70}")
    print("Cache Mitigation Experiment")
    print(f"  Strategy:   {strategy_desc}")
    print("  Algorithm:  AlphaEdit (with cache mitigation)")
    print(f"  Dataset:    {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:  {args.num_edits}")
    print(f"  Seed:       {args.seed}")
    print(f"  CUDA:       device {args.cuda_device}")
    print(f"  Model:      {args.model_name}")
    print(f"  Metadata:   {metadata_jsonl}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Mitigation run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Cache mitigation experiment completed.")
    print(f"  Strategy: {strategy_desc}")
    print(f"  Metadata: {metadata_jsonl}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run AlphaEdit with cache_c mitigation strategies"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=5)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # Mitigation strategy arguments
    parser.add_argument(
        "--strategy", required=True,
        choices=["svd_truncation", "exponential_decay", "periodic_reset"],
        help="Cache mitigation strategy to apply"
    )
    parser.add_argument(
        "--truncation_interval", type=int, default=5,
        help="For svd_truncation: apply every K batches"
    )
    parser.add_argument(
        "--retain_ratio", type=float, default=0.75,
        help="For svd_truncation: fraction of singular values to keep"
    )
    parser.add_argument(
        "--decay_factor", type=float, default=0.95,
        help="For exponential_decay: multiply cache_c by this each batch"
    )
    parser.add_argument(
        "--reset_interval", type=int, default=10,
        help="For periodic_reset: zero cache_c every K batches"
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
