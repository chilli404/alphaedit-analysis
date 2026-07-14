#!/usr/bin/env python3
"""
Checkpoint-Based Failure Curve Runner for AlphaEdit / MEMIT.

Enables long-running failure curve experiments (2000→10000 edits) to survive
8-hour SkyPilot cluster limits by saving model state at milestones and
resuming from checkpoints in subsequent cluster runs.

Implementation approach:
  Same source-injection pattern as nullspace_tracker.py — injects checkpoint
  save/load/skip logic into evaluate.py at known anchor points (pinned commit
  b84624f).

Injection points:
  1. BEFORE the main edit loop: load checkpoint (restore model weights + cache_c)
  2. BEFORE the per-batch edit call: skip guard (skip already-processed batches)
  3. AFTER the per-batch edit call: save checkpoint at interval boundaries

Checkpoint contents:
  checkpoints/{alg_name}/seed{seed}/batch_{N}/
      metadata.json          — batch index, total edits, timestamp
      model_weights.pt       — state dict for edited layers only (~560MB)
      cache_c.pt             — covariance cache (AlphaEdit only, ~320MB)

Checkpoint dir resolution:
  1. --checkpoint_dir if provided
  2. /s3-data/continual-learning/alphaedit/checkpoints/ if exists
  3. ~/.cache/alphaedit_checkpoints/

Usage:
    python src/checkpoint_runner.py \\
        --seed 42 \\
        --alg_name AlphaEdit \\
        --ds_name mcf \\
        --dataset_size_limit 5000 \\
        --num_edits 100 \\
        --start_from_batch 0 \\
        --save_interval 10 \\
        --downstream_eval_steps 10 \\
        --conserve_memory
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

from model_download import resolve_model_path
from setup_hparams import link_hparams


def get_project_root() -> Path:
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


# --- Source anchors from evaluate.py at commit b84624f ---

# Anchor for the main edit loop (inject checkpoint load BEFORE this)
LOOP_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'

# Anchor for per-batch edit timing (inject skip guard BEFORE this)
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'

# Anchor for the boundary after primary algorithms (inject checkpoint save BEFORE this)
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'

# CUDA patch target
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def resolve_checkpoint_dir(explicit_dir: str | None, alg_name: str, seed: int) -> Path:
    """Resolve the checkpoint directory in priority order."""
    if explicit_dir:
        base = Path(explicit_dir)
    elif Path("/s3-data/continual-learning/alphaedit/checkpoints").exists():
        base = Path("/s3-data/continual-learning/alphaedit/checkpoints")
    else:
        base = Path.home() / ".cache" / "alphaedit_checkpoints"

    return base / alg_name / f"seed{seed}"


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    """Find the latest checkpoint batch in the directory.

    Returns (batch_idx, batch_dir) or None if no checkpoints exist.
    """
    if not ckpt_dir.exists():
        return None

    batch_dirs = sorted(
        [d for d in ckpt_dir.glob("batch_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    if not batch_dirs:
        return None

    # Find the highest batch number with a valid metadata.json
    for batch_dir in reversed(batch_dirs):
        metadata_file = batch_dir / "metadata.json"
        if metadata_file.exists():
            try:
                batch_idx = int(batch_dir.name.split("_")[1])
                return (batch_idx, batch_dir)
            except (ValueError, IndexError):
                continue

    return None


def build_checkpoint_script(
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
    start_from_batch: int,
    save_interval: int,
    checkpoint_dir: str,
    fast_checkpoint: bool = False,
    eval_at_checkpoints_only: bool = False,
) -> str:
    """
    Build an inline Python script that:
    1. Seeds all RNGs
    2. Injects checkpoint save/load/skip into evaluate.py
    3. Executes the patched evaluate.py as __main__

    Evaluation modes:
      - Normal: Evaluate all facts after every batch (slow, complete)
      - fast_checkpoint: Evaluate only edited batch after every batch (fast, partial)
      - eval_at_checkpoints_only: Evaluate all facts only at checkpoint boundaries (balanced)
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

# 3. Checkpoint parameters
_ckpt_start_batch = {start_from_batch}
_ckpt_save_interval = {save_interval}
_ckpt_dir = "{checkpoint_dir}"
_ckpt_alg_name = "{alg_name}"
_ckpt_seed = {seed}
_ckpt_num_edits = {num_edits}
_ckpt_fast_mode = {fast_checkpoint}
_ckpt_eval_at_checkpoints_only = {eval_at_checkpoints_only}

def _ckpt_save(cnt, model, cache_c, hparams, alg_name):
    \"\"\"Save model weights and cache_c at checkpoint boundary.\"\"\"
    import json, shutil
    from pathlib import Path
    from datetime import datetime, timezone

    batch_dir = Path(_ckpt_dir) / f"batch_{{cnt}}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save only the edited layer weights (much smaller than full model)
    layer_weights = {{}}
    for layer_idx in hparams.layers:
        for key in ["mlp.down_proj.weight", "mlp.up_proj.weight"]:
            param_name = f"model.layers.{{layer_idx}}.{{key}}"
            param = dict(model.named_parameters()).get(param_name)
            if param is not None:
                layer_weights[param_name] = param.data.cpu()

    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    # Save cache_c (AlphaEdit only)
    if alg_name == "AlphaEdit" and cache_c is not None:
        torch.save(cache_c.cpu(), str(batch_dir / "cache_c.pt"))

    # Save metadata
    metadata = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _ckpt_num_edits,
        "alg_name": alg_name,
        "seed": _ckpt_seed,
        "num_edits_per_batch": _ckpt_num_edits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }}
    with open(str(batch_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [CHECKPOINT] Saved batch {{cnt}} ({{(cnt+1) * _ckpt_num_edits}} total edits) -> {{batch_dir}}")

def _ckpt_load(model, hparams, alg_name):
    \"\"\"Load model weights (and cache_c for AlphaEdit) from the start_from_batch checkpoint.\"\"\"
    from pathlib import Path

    if _ckpt_start_batch <= 0:
        return None  # No checkpoint to load

    batch_dir = Path(_ckpt_dir) / f"batch_{{_ckpt_start_batch - 1}}"
    if not batch_dir.exists():
        print(f"  [CHECKPOINT] WARNING: Expected checkpoint at {{batch_dir}} not found. Starting from scratch.")
        return None

    # Load model weights
    weights_file = batch_dir / "model_weights.pt"
    if weights_file.exists():
        layer_weights = torch.load(str(weights_file), map_location="cuda")
        param_dict = dict(model.named_parameters())
        loaded_count = 0
        for param_name, param_data in layer_weights.items():
            if param_name in param_dict:
                param_dict[param_name].data.copy_(param_data)
                loaded_count += 1
        print(f"  [CHECKPOINT] Loaded {{loaded_count}} parameter tensors from {{weights_file}}")
    else:
        print(f"  [CHECKPOINT] WARNING: No model_weights.pt in {{batch_dir}}")

    # Load cache_c (AlphaEdit only)
    cache_c_loaded = None
    if alg_name == "AlphaEdit":
        cache_file = batch_dir / "cache_c.pt"
        if cache_file.exists():
            cache_c_loaded = torch.load(str(cache_file), map_location="cuda")
            print(f"  [CHECKPOINT] Loaded cache_c from {{cache_file}} (shape: {{cache_c_loaded.shape}})")
        else:
            print(f"  [CHECKPOINT] WARNING: No cache_c.pt in {{batch_dir}} (AlphaEdit will start with zero cache)")

    print(f"  [CHECKPOINT] Resuming from batch {{_ckpt_start_batch}} ({{_ckpt_start_batch * _ckpt_num_edits}} edits already applied)")
    return cache_c_loaded

def _ckpt_should_skip(cnt):
    \"\"\"Return True if this batch was already processed (before start_from_batch).\"\"\"
    return cnt < _ckpt_start_batch

def _ckpt_should_save(cnt):
    \"\"\"Return True if we should save a checkpoint at this batch.\"\"\"
    return (cnt + 1) % _ckpt_save_interval == 0

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
    '# CUDA_VISIBLE_DEVICES managed by checkpoint_runner',
)

# 6. Inject checkpoint LOAD before the main edit loop
loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert loop_anchor in source, (
    "Loop anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

load_injection = '''    # === CHECKPOINT: load state from previous run (injected) ===
    _ckpt_cache_c_loaded = None
    if _ckpt_start_batch > 0 and '_ckpt_load' in globals():
        _ckpt_cache_c_loaded = _ckpt_load(model, hparams, alg_name)
        if _ckpt_cache_c_loaded is not None and alg_name == "AlphaEdit":
            cache_c = _ckpt_cache_c_loaded
    # === END checkpoint load ===
'''
source = source.replace(
    loop_anchor,
    load_injection + loop_anchor,
    1,
)

# 7. Inject SKIP guard before the per-batch edit call
pre_anchor = '        start = time()\\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
assert pre_anchor in source, (
    "Pre-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

skip_injection = '''        # === CHECKPOINT: skip already-processed batches (injected) ===
        if '_ckpt_should_skip' in globals() and _ckpt_should_skip(cnt):
            cnt += 1
            continue
        # === END checkpoint skip ===
'''
source = source.replace(
    pre_anchor,
    skip_injection + pre_anchor,
    1,
)

# 8. Inject checkpoint SAVE after edit (before MEMIT_prune branch)
post_anchor = '        elif alg_name == "MEMIT_prune":'
assert post_anchor in source, (
    "Post-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

save_injection = '''        # === CHECKPOINT: save at interval boundaries (injected) ===
        if '_ckpt_should_save' in globals() and _ckpt_should_save(cnt):
            _ckpt_save(cnt, model, cache_c if alg_name == "AlphaEdit" else None, hparams, alg_name)
        # === END checkpoint save ===
'''
source = source.replace(
    post_anchor,
    save_injection + post_anchor,
    1,
)

# 9. Inject CHECKPOINT-ONLY EVAL guard (skip entire evaluation for non-checkpoint batches)
# The evaluation section in evaluate.py is OUTSIDE the edit loop (runs once after all edits).
# We inject a flag check that prevents evaluation when the final batch is not a checkpoint boundary.
eval_start_anchor = '    # torch.save(hs, "post_edit_hs_memit.pt")\\n    start = time()'
assert eval_start_anchor in source, (
    "Evaluation start anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

checkpoint_eval_skip = '''    # torch.save(hs, "post_edit_hs_memit.pt")
    # === CHECKPOINT: skip evaluation if last batch is not a checkpoint boundary (injected) ===
    _do_final_eval = True
    if _ckpt_eval_at_checkpoints_only and not _ckpt_should_save(cnt - 1):
        _do_final_eval = False
        print(f"  [CHECKPOINT] Skipping final evaluation (batch {{cnt-1}} not at checkpoint boundary)")
        print(f"  [CHECKPOINT] Evaluation will run when resumed and a checkpoint boundary is reached.")
    # === END checkpoint eval skip ===
    start = time()'''
source = source.replace(
    eval_start_anchor,
    checkpoint_eval_skip,
    1,
)

# 10. Inject FAST EVAL guard in evaluation loop (skip records not in current batch)
eval_anchor = '    for record in ds:\\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
assert eval_anchor in source, (
    "Evaluation loop anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

fast_eval_injection = '''    for record in ds:
        # === CHECKPOINT: skip entire evaluation if _do_final_eval is False (injected) ===
        if not _do_final_eval:
            break
        # === CHECKPOINT: fast mode - skip evaluating non-batch records (injected) ===
        if _ckpt_fast_mode and record["case_id"] not in case_ids:
            continue
        # === END fast mode guard ===
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'''
source = source.replace(
    eval_anchor,
    fast_eval_injection,
    1,
)

# 11. Verify all injections succeeded
assert "CHECKPOINT: load state" in source, "Load injection failed"
assert "CHECKPOINT: skip already-processed" in source, "Skip injection failed"
assert "CHECKPOINT: save at interval" in source, "Save injection failed"
assert "CHECKPOINT: skip evaluation if last batch is not" in source, "Checkpoint-only eval injection failed"
assert "CHECKPOINT: skip entire evaluation if _do_final_eval" in source, "Final eval guard injection failed"
assert "CHECKPOINT: fast mode" in source, "Fast eval injection failed"

# 12. Execute
exec(compile(source, "experiments/evaluate.py", "exec"),
     {{
         "__name__": "__main__",
         "__file__": "experiments/evaluate.py",
         "_ckpt_start_batch": _ckpt_start_batch,
         "_ckpt_save_interval": _ckpt_save_interval,
         "_ckpt_dir": _ckpt_dir,
         "_ckpt_alg_name": _ckpt_alg_name,
         "_ckpt_seed": _ckpt_seed,
         "_ckpt_num_edits": _ckpt_num_edits,
         "_ckpt_fast_mode": _ckpt_fast_mode,
         "_ckpt_eval_at_checkpoints_only": _ckpt_eval_at_checkpoints_only,
         "_ckpt_save": _ckpt_save,
         "_ckpt_load": _ckpt_load,
         "_ckpt_should_skip": _ckpt_should_skip,
         "_ckpt_should_save": _ckpt_should_save,
     }})

# 13. Final summary
print(f"\\n=== Checkpoint runner complete ===")
print(f"  Algorithm: {{_ckpt_alg_name}}")
print(f"  Resumed from batch: {{_ckpt_start_batch}}")
print(f"  Save interval: every {{_ckpt_save_interval}} batches")
print(f"  Fast checkpoint: {{_ckpt_fast_mode}}")
print(f"  Eval at checkpoints only: {{_ckpt_eval_at_checkpoints_only}}")
print(f"  Checkpoint dir: {{_ckpt_dir}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the checkpointed experiment."""
    alphaedit_root = get_alphaedit_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()

    # Resolve model path
    model_name = resolve_model_path(args.model_name)

    # Validate anchors exist in the source before launching
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for anchor_name, anchor_str in [
        ("LOOP_ANCHOR", LOOP_ANCHOR),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
    ]:
        if anchor_str not in eval_source:
            print(f"ERROR: {anchor_name} not found in evaluate.py.")
            print("  The upstream code has diverged from pinned commit b84624f.")
            sys.exit(1)

    # Resolve checkpoint directory
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.alg_name, args.seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Determine start_from_batch
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        # Auto-detect from latest checkpoint
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = latest[0] + 1
            print(f"  Auto-detected: resume from batch {start_from_batch} (checkpoint at batch {latest[0]})")
        else:
            start_from_batch = 0
            print("  No existing checkpoints found. Starting from batch 0.")

    script = build_checkpoint_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name=args.alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        start_from_batch=start_from_batch,
        save_interval=args.save_interval,
        checkpoint_dir=str(ckpt_dir),
        fast_checkpoint=args.fast_checkpoint,
        eval_at_checkpoints_only=args.eval_at_checkpoints_only,
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    total_batches = args.dataset_size_limit // args.num_edits

    # Determine evaluation mode description
    if args.eval_at_checkpoints_only:
        eval_mode = f"Milestone only (every {args.save_interval} batches)"
    elif args.fast_checkpoint:
        eval_mode = "Fast (edited batch only)"
    else:
        eval_mode = "Full (all facts every batch)"

    print(f"{'=' * 70}")
    print("Checkpoint-Based Failure Curve Runner")
    print(f"  Algorithm:       {args.alg_name}")
    print(f"  Dataset:         {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits/batch: {args.num_edits}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Resume from:     batch {start_from_batch} ({start_from_batch * args.num_edits} edits)")
    print(f"  Save interval:   every {args.save_interval} batches")
    print(f"  Evaluation:      {eval_mode}")
    print(f"  Seed:            {args.seed}")
    print(f"  CUDA:            device {args.cuda_device}")
    print(f"  Model:           {args.model_name}")
    print(f"  Checkpoint dir:  {ckpt_dir}")
    print(f"  Started:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Checkpoint run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Detect run_dir and run_id created by evaluate.py
    from seeded_runner import find_latest_run_dir
    project_root = get_project_root()
    run_dir_rel, run_id = find_latest_run_dir(args.alg_name)

    # Find latest checkpoint to determine where this segment ended
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    ended_at_batch = latest_ckpt[0] if latest_ckpt else total_batches - 1

    # Record metadata as JSONL (append mode — one line per segment/resume)
    # This way multiple resumes don't overwrite each other.
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "alphaedit_commit": "b84624f",
        "cuda_device": args.cuda_device,
        "experiment": "failure_curve_ckpt",
        "algorithm": args.alg_name,
        "dataset": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "run_dir": run_dir_rel,
        "run_id": run_id,
        "checkpoint_dir": str(ckpt_dir),
        "resumed_from_batch": start_from_batch if start_from_batch > 0 else None,
        "ended_at_batch": ended_at_batch,
        "params": {
            "model_name": args.model_name,
            "hparams_fname": args.hparams_fname,
            "save_interval": args.save_interval,
            "fast_checkpoint": args.fast_checkpoint,
            "eval_at_checkpoints_only": args.eval_at_checkpoints_only,
            "downstream_eval_steps": args.downstream_eval_steps,
        },
    }

    results_dir = project_root / "results"
    metadata_dir = results_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    # JSONL: append one line per segment so multi-resume runs build a history
    metadata_file = metadata_dir / f"run_seed{args.seed}_{args.alg_name}_ckpt_{args.dataset_size_limit}.jsonl"
    with open(metadata_file, "a") as f:
        f.write(json.dumps(metadata) + "\n")
    print(f"Metadata appended to: {metadata_file}")

    print(f"\n{'=' * 70}")
    print("Checkpoint run completed.")
    print(f"  Finished:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Results:     {alphaedit_root / 'results' / args.alg_name}")
    if run_id:
        print(f"  Run ID:      {run_id}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Segment:     batch {start_from_batch} → {ended_at_batch}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint-based failure curve runner for AlphaEdit/MEMIT"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    # Experiment parameters
    parser.add_argument("--alg_name", required=True, choices=["AlphaEdit", "MEMIT"])
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # Checkpoint parameters
    parser.add_argument("--start_from_batch", type=int, default=-1,
                        help="Batch to resume from (-1 = auto-detect from latest checkpoint)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N batches (default: 10 = every 1000 edits)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint directory (default: S3 mount or ~/.cache)")

    # Evaluation mode (mutually exclusive)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true",
                        help="Fast mode: only evaluate edited batch after each edit (partial preservation measurement)")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true",
                        help="Milestone mode: evaluate full dataset only at checkpoint boundaries (RECOMMENDED for conferences)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
