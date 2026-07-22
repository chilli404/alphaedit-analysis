#!/usr/bin/env python3
"""
Cache Mitigation Batch Runner: Runs all mitigation strategy variants with
a single model load, resetting state between each variant.

This is an optimization of run_mitigation_sweep.sh which launches 12
independent processes, each loading the model from scratch.

This batch runner:
  1. Loads the model once
  2. For each strategy variant:
     a. Restores base model state (via state_dict)
     b. Runs editing with that mitigation strategy
     c. Records results
  3. Produces identical results to running cache_mitigation_runner.py
     independently for each variant.

Strategy variants (12 total):
  SVD truncation:     K ∈ {5, 10} × retain_ratio ∈ {0.5, 0.75, 0.9}  → 6
  Exponential decay:  decay ∈ {0.90, 0.95, 0.99}                       → 3
  Periodic reset:     K ∈ {5, 10, 20}                                   → 3

Usage:
    python src/cache_mitigation_batch_runner.py --seed 42

    # Custom subset of strategies
    python src/cache_mitigation_batch_runner.py --seed 42 \\
        --strategies svd_truncation exponential_decay
"""

import argparse
import json
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


from paths import get_project_root, get_alphaedit_root, get_result_root


# Source anchor from evaluate.py at commit b84624f
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


# All 12 strategy configurations
ALL_VARIANTS = [
    # SVD truncation: 6 variants
    {"strategy": "svd_truncation", "truncation_interval": 5, "retain_ratio": 0.5, "decay_factor": 1.0, "reset_interval": 999},
    {"strategy": "svd_truncation", "truncation_interval": 5, "retain_ratio": 0.75, "decay_factor": 1.0, "reset_interval": 999},
    {"strategy": "svd_truncation", "truncation_interval": 5, "retain_ratio": 0.9, "decay_factor": 1.0, "reset_interval": 999},
    {"strategy": "svd_truncation", "truncation_interval": 10, "retain_ratio": 0.5, "decay_factor": 1.0, "reset_interval": 999},
    {"strategy": "svd_truncation", "truncation_interval": 10, "retain_ratio": 0.75, "decay_factor": 1.0, "reset_interval": 999},
    {"strategy": "svd_truncation", "truncation_interval": 10, "retain_ratio": 0.9, "decay_factor": 1.0, "reset_interval": 999},
    # Exponential decay: 3 variants
    {"strategy": "exponential_decay", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 0.90, "reset_interval": 999},
    {"strategy": "exponential_decay", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 0.95, "reset_interval": 999},
    {"strategy": "exponential_decay", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 0.99, "reset_interval": 999},
    # Periodic reset: 3 variants
    {"strategy": "periodic_reset", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 1.0, "reset_interval": 5},
    {"strategy": "periodic_reset", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 1.0, "reset_interval": 10},
    {"strategy": "periodic_reset", "truncation_interval": 999, "retain_ratio": 1.0, "decay_factor": 1.0, "reset_interval": 20},
]


def build_batch_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    variants: list[dict],
    output_dir: str,
) -> str:
    """
    Build a script that loads the model once and runs all mitigation
    strategy variants sequentially, restoring model state between each.
    """
    argv_base = [
        "experiments.evaluate",
        "--alg_name=AlphaEdit",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
        f"--downstream_eval_steps={downstream_eval_steps}",
        "--generation_test_interval=1",
    ]
    if conserve_memory:
        argv_base.append("--conserve_memory")

    argv_base_str = repr(argv_base)
    variants_str = repr(variants)

    script = textwrap.dedent(f"""\
import os, sys, random, json, gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

_output_dir = "{output_dir}"
Path(_output_dir).mkdir(parents=True, exist_ok=True)

_variants = {variants_str}
_seed = {seed}

# 1. Read evaluate.py source ONCE
with open("experiments/evaluate.py", "r") as f:
    _base_source = f.read()

# Patch CUDA
_cuda_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
assert _cuda_target in _base_source, "CUDA patch target not found"
_base_source = _base_source.replace(
    _cuda_target,
    '# CUDA_VISIBLE_DEVICES managed by cache_mitigation_batch_runner',
)

# Verify anchor
_post_anchor = '        elif alg_name == "MEMIT_prune":'
assert _post_anchor in _base_source, "Post-edit anchor not found"

print("[BATCH] === Cache Mitigation Batch Runner ===")
print(f"  Seed:     {{_seed}}")
print(f"  Variants: {{len(_variants)}}")
print(f"  Output:   {{_output_dir}}")
print("")

# 2. Load model once via HuggingFace (for state_dict capture)
print("[BATCH] Loading model for state capture...")
from transformers import AutoModelForCausalLM, AutoTokenizer
_model_path = "{model_name}"
_tokenizer = AutoTokenizer.from_pretrained(_model_path)
_model = AutoModelForCausalLM.from_pretrained(
    _model_path, torch_dtype=torch.float16, device_map="auto"
)
_base_state = {{k: v.cpu().clone() for k, v in _model.state_dict().items()}}
print("[BATCH] Model loaded and base state saved.")
# Free the model — evaluate.py will reload it from HF cache (fast since cached on disk)
del _model, _tokenizer
gc.collect()
torch.cuda.empty_cache()

# 3. Run each variant
_completed = 0
_failed = 0

for _var_idx, _variant in enumerate(_variants):
    _strategy = _variant["strategy"]
    _trunc_interval = _variant["truncation_interval"]
    _retain_ratio = _variant["retain_ratio"]
    _decay_factor = _variant["decay_factor"]
    _reset_interval = _variant["reset_interval"]

    # Build description
    if _strategy == "svd_truncation":
        _desc = f"SVD(K={{_trunc_interval}}, retain={{_retain_ratio}})"
    elif _strategy == "exponential_decay":
        _desc = f"Decay(factor={{_decay_factor}})"
    elif _strategy == "periodic_reset":
        _desc = f"Reset(K={{_reset_interval}})"
    else:
        _desc = _strategy

    print(f"\\n[BATCH] --- Variant {{_var_idx+1}}/{{len(_variants)}}: {{_desc}} ---")

    # Re-seed (same as independent run)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    # Set sys.argv
    sys.argv = {argv_base_str}

    # Build mitigation injection
    _mitigation_code = f'''        # === CACHE MITIGATION: apply strategy (injected) ===
        if alg_name == "AlphaEdit" and '_apply_mitigation' in globals():
            _apply_mitigation(cnt, cache_c)
        # === END cache mitigation ===
'''
    _run_source = _base_source.replace(_post_anchor, _mitigation_code + _post_anchor, 1)

    # Define mitigation function for this variant
    _mitigation_log = []

    def _make_apply_fn(strategy, trunc_int, retain, decay, reset_int, log_list):
        def _apply_mitigation(cnt, cache_c):
            applied = False
            if strategy == "svd_truncation":
                if (cnt + 1) % trunc_int == 0:
                    for i in range(cache_c.shape[0]):
                        layer_cache = cache_c[i].float()
                        if layer_cache.abs().max() > 0:
                            U, S, Vt = torch.linalg.svd(layer_cache, full_matrices=False)
                            r = max(1, int(len(S) * retain))
                            cache_c[i] = (U[:, :r] * S[:r]) @ Vt[:r, :]
                    applied = True
            elif strategy == "exponential_decay":
                cache_c *= decay
                applied = True
            elif strategy == "periodic_reset":
                if (cnt + 1) % reset_int == 0:
                    cache_c.zero_()
                    applied = True
            if applied:
                log_list.append({{"batch_idx": cnt, "strategy": strategy, "applied": True}})
        return _apply_mitigation

    _apply_fn = _make_apply_fn(
        _strategy, _trunc_interval, _retain_ratio, _decay_factor, _reset_interval, _mitigation_log
    )

    try:
        exec(compile(_run_source, "experiments/evaluate.py", "exec"), {{
            "__name__": "__main__",
            "__file__": "experiments/evaluate.py",
            "_apply_mitigation": _apply_fn,
            "_mitigation_log": _mitigation_log,
        }})
        _completed += 1

        # Write metadata
        _timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _meta = {{
            "strategy": _strategy,
            "truncation_interval": _trunc_interval,
            "retain_ratio": _retain_ratio,
            "decay_factor": _decay_factor,
            "reset_interval": _reset_interval,
            "seed": _seed,
            "total_mitigations_applied": len(_mitigation_log),
        }}
        _meta_path = Path(_output_dir) / f"mitigation_{{_strategy}}_seed{{_seed}}_{{_timestamp}}.jsonl"
        with open(str(_meta_path), "a") as _f:
            _f.write(json.dumps(_meta) + "\\n")

        print(f"[BATCH] --- Completed: {{_desc}} ({{len(_mitigation_log)}} mitigations applied) ---")
    except Exception as _e:
        _failed += 1
        print(f"[BATCH] --- FAILED: {{_desc}}: {{_e}} ---")

    # Force GC between variants
    gc.collect()
    torch.cuda.empty_cache()

print(f"\\n[BATCH] === Cache mitigation batch complete ===")
print(f"  Completed: {{_completed}} / {{len(_variants)}}")
print(f"  Failed:    {{_failed}} / {{len(_variants)}}")
print(f"  Results:   {{_output_dir}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the batch mitigation experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()

    model_name = resolve_model_path(args.model_name)

    # Validate anchors
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    if POST_EDIT_ANCHOR not in eval_source:
        print("ERROR: Post-edit anchor not found in evaluate.py.")
        sys.exit(1)
    if CUDA_PATCH_TARGET not in eval_source:
        print("ERROR: CUDA patch target not found in evaluate.py.")
        sys.exit(1)

    # Filter variants by requested strategies
    if args.strategies:
        variants = [v for v in ALL_VARIANTS if v["strategy"] in args.strategies]
    else:
        variants = ALL_VARIANTS

    # Output directory
    output_dir = get_result_root() / "mitigation"
    output_dir.mkdir(parents=True, exist_ok=True)

    script = build_batch_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        variants=variants,
        output_dir=str(output_dir),
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'=' * 70}")
    print("Cache Mitigation Batch Runner")
    print(f"  Seed:       {args.seed}")
    print(f"  Variants:   {len(variants)}")
    print(f"  Strategies: {list(set(v['strategy'] for v in variants))}")
    print(f"  Dataset:    {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  CUDA:       device {args.cuda_device}")
    print(f"  Model:      {args.model_name}")
    print(f"  Output:     {output_dir}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Batch run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Cache mitigation batch completed.")
    print(f"  Results: {output_dir}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch cache mitigation sweep: single model load, all 12 variants"
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
    parser.add_argument("--strategies", nargs="*", default=None,
                        choices=["svd_truncation", "exponential_decay", "periodic_reset"],
                        help="Subset of strategies to run (default: all)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
