#!/usr/bin/env python3
"""
Online plasticity + projection tracker for AlphaEdit collapse mechanism study.

Extends checkpoint_runner with per-batch inline measurements:
  - Immediate efficacy (right after insertion)
  - Projection removed fraction: ||ΔW_raw - ΔW_proj||_F / ||ΔW_raw||_F
  - Update norms (raw and projected)
  - Solve residual and condition number
  - Weight-spectrum delta from previous batch

This script patches BOTH AlphaEdit_main.py (to capture raw vs projected updates)
AND evaluate.py (for immediate evaluation and checkpoint management).

Uses dual source injection pattern (like coupling_stress_runner.py).

Output: JSONL with one record per batch containing all plasticity + mechanism metrics.

Usage:
    python src/plasticity_tracker.py \\
        --seed 42 \\
        --cuda_device 0 \\
        --model_name NousResearch/Meta-Llama-3-8B-Instruct \\
        --hparams_fname Llama3-8B.json \\
        --ds_name mcf \\
        --dataset_size_limit 10000 \\
        --num_edits 100 \\
        --checkpoint_dir /s3-data/.../checkpoints/AlphaEdit/seed42 \\
        --save_interval 10
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALPHAEDIT_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))


# ─── Source Anchors ──────────────────────────────────────────────────────────

# In AlphaEdit_main.py: after the constrained solve, before returning
# This is where we measure raw vs projected update
ALPHAEDIT_SOLVE_ANCHOR = "upd_matrix = (right_vector @ (left_pseudo @ P_).T).T"

# In evaluate.py: before and after edit application
EVAL_PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
EVAL_POST_EDIT_ANCHOR = '        exec_time = time() - start'
EVAL_LOOP_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
EVAL_CUDA_PATCH = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def build_plasticity_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    save_interval: int,
    checkpoint_dir: str,
    output_jsonl: str,
    conserve_memory: bool = True,
    start_from_batch: int = 0,
    downstream_eval_steps: int = 999,
) -> str:
    """
    Build inline script that injects plasticity/projection measurement into
    both AlphaEdit_main.py and evaluate.py.
    """
    argv_parts = [
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
        argv_parts.append("--conserve_memory")

    script = textwrap.dedent(f"""\
import os, sys, random, json, math, time
import numpy as np
import torch
from pathlib import Path

# ─── 1. Seed ────────────────────────────────────────────────────────
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.argv = {repr(argv_parts)}

# ─── 2. Tracking state ──────────────────────────────────────────────
_plast_output = "{output_jsonl}"
_plast_checkpoint_dir = Path("{checkpoint_dir}")
_plast_save_interval = {save_interval}
_plast_start_batch = {start_from_batch}
_plast_batch_records = []

# Per-batch projection tracking (populated by patched AlphaEdit_main)
_plast_current_batch_projections = []  # list of per-layer dicts

def _plast_log_projection(layer_idx, raw_update, projected_update, right_vector, condition_number, solve_residual):
    \"\"\"Called from patched AlphaEdit_main.py after each layer's solve.\"\"\"
    raw_norm = raw_update.norm().item()
    proj_norm = projected_update.norm().item()
    removed = (raw_update - projected_update).norm().item()
    removed_fraction = removed / raw_norm if raw_norm > 0 else 0.0

    # Cosine alignment between raw and projected
    cosine = 0.0
    if raw_norm > 0 and proj_norm > 0:
        cosine = (raw_update.flatten() @ projected_update.flatten()).item() / (raw_norm * proj_norm)

    _plast_current_batch_projections.append({{
        "layer_idx": layer_idx,
        "raw_update_norm": round(raw_norm, 6),
        "projected_update_norm": round(proj_norm, 6),
        "removed_norm": round(removed, 6),
        "removed_fraction": round(removed_fraction, 6),
        "cosine_alignment": round(cosine, 6),
        "right_vector_norm": round(right_vector.norm().item(), 6),
        "condition_number": round(condition_number, 4) if not math.isinf(condition_number) else "inf",
        "solve_residual": round(solve_residual, 6),
    }})

def _plast_record_batch(cnt, model, hparams, exec_time):
    \"\"\"Called after each edit batch. Records aggregate metrics.\"\"\"
    record = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * {num_edits},
        "exec_time_s": round(exec_time, 2),
        "projection_metrics": list(_plast_current_batch_projections),
    }}

    # Aggregate projection stats
    if _plast_current_batch_projections:
        fracs = [p["removed_fraction"] for p in _plast_current_batch_projections]
        record["mean_removed_fraction"] = round(np.mean(fracs), 6)
        record["max_removed_fraction"] = round(max(fracs), 6)
        norms = [p["projected_update_norm"] for p in _plast_current_batch_projections]
        record["mean_update_norm"] = round(np.mean(norms), 6)
        cosines = [p["cosine_alignment"] for p in _plast_current_batch_projections]
        record["mean_cosine_alignment"] = round(np.mean(cosines), 6)

    _plast_batch_records.append(record)
    _plast_current_batch_projections.clear()

    # Write incrementally
    with open(_plast_output, "a") as f:
        f.write(json.dumps(record) + "\\n")

    return record

def _plast_should_skip(cnt):
    return cnt < _plast_start_batch

def _plast_should_save(cnt):
    return (cnt + 1) % _plast_save_interval == 0

def _plast_save_checkpoint(cnt, model, cache_c, hparams):
    ckpt_dir = _plast_checkpoint_dir / f"batch_{{cnt}}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save edited layer weights only
    param_dict = dict(model.named_parameters())
    edited_weights = {{}}
    for layer in hparams.layers:
        for name, param in param_dict.items():
            if f"layers.{{layer}}" in name and "down_proj" in name:
                edited_weights[name] = param.detach().cpu()
                break

    torch.save(edited_weights, ckpt_dir / "model_weights.pt")
    if cache_c is not None:
        torch.save(cache_c.cpu(), ckpt_dir / "cache_c.pt")

    metadata = {{"batch_idx": cnt, "total_edits": (cnt + 1) * {num_edits},
                 "timestamp": datetime.now(timezone.utc).isoformat()}}
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"  [PLASTICITY] Checkpoint saved: {{ckpt_dir}}")

from datetime import datetime, timezone

# ─── 3. Patch AlphaEdit_main.py ─────────────────────────────────────
# Read and inject projection logging into the solve

alphaedit_source_path = Path("AlphaEdit/AlphaEdit_main.py")
ae_source = alphaedit_source_path.read_text()

# Inject after the constrained solve to capture raw vs projected
solve_anchor = "{ALPHAEDIT_SOLVE_ANCHOR}"
assert solve_anchor in ae_source, f"AlphaEdit solve anchor not found"

# The raw update (before projection P) is: right_vector @ left_pseudo^T
# The projected update is: right_vector @ (left_pseudo @ P_)^T = upd_matrix^T
# Actually upd_matrix = (right_vector @ (left_pseudo @ P_).T).T
# So projected = upd_matrix (the line itself IS the projected result)
# Raw would be: right_vector @ left_pseudo.T without P_

projection_injection = '''
    # === PLASTICITY TRACKER: log projection metrics ===
    try:
        _raw_update = (right_vector @ left_pseudo.T).T
        _proj_update = upd_matrix
        # Condition number of the LHS
        _lhs_for_cond = hparams.mom2_update_weight * cache_c[cache_id].to(left_vector.device) + left_vector @ left_vector.T
        _svs = torch.linalg.svdvals(_lhs_for_cond.float())
        _svs_pos = _svs[_svs > 1e-10]
        _cond = (_svs_pos[0] / _svs_pos[-1]).item() if len(_svs_pos) >= 2 else float("inf")
        # Solve residual
        _residual = (upd_matrix @ left_vector - right_vector).norm().item()
        if '_plast_log_projection' in dir():
            _plast_log_projection(layer, _raw_update, _proj_update, right_vector, _cond, _residual)
        elif '_plast_log_projection' in globals():
            _plast_log_projection(layer, _raw_update, _proj_update, right_vector, _cond, _residual)
    except Exception as _e:
        print(f"  [PLASTICITY] Projection logging error: {{_e}}")
    # === END plasticity tracking ===
'''
ae_source = ae_source.replace(solve_anchor, solve_anchor + projection_injection, 1)

# Add **_kwargs to handle extra arguments from evaluate.py
if "def execute_alphaedit(" in ae_source and "**_kwargs" not in ae_source.split("def execute_alphaedit(")[1].split(")")[0]:
    ae_source = ae_source.replace(
        "return_orig_weights_device=None,\\n)",
        "return_orig_weights_device=None,\\n    **_kwargs,\\n)",
        1
    )

# Compile patched AlphaEdit main
ae_code = compile(ae_source, "AlphaEdit/AlphaEdit_main.py", "exec")
ae_namespace = {{"__name__": "AlphaEdit_main_patched", "__file__": "AlphaEdit/AlphaEdit_main.py"}}
ae_namespace["_plast_log_projection"] = _plast_log_projection
ae_namespace["_plast_current_batch_projections"] = _plast_current_batch_projections
exec(ae_code, ae_namespace)
_patched_execute_alphaedit = ae_namespace["execute_alphaedit"]
print("[PLASTICITY] Patched AlphaEdit_main.py with projection logging")

# ─── 4. Patch evaluate.py ───────────────────────────────────────────
with open("experiments/evaluate.py", "r") as f:
    eval_source = f.read()

# CUDA patch
eval_source = eval_source.replace(
    '{EVAL_CUDA_PATCH}',
    '# CUDA managed by plasticity_tracker',
)

# Inject batch skip (for checkpoint resume)
pre_anchor = '{EVAL_PRE_EDIT_ANCHOR}'
assert pre_anchor in eval_source, "Pre-edit anchor not found"
skip_injection = '''        # === PLASTICITY: skip already-processed batches ===
        if '_plast_should_skip' in globals() and _plast_should_skip(cnt):
            cnt += 1
            continue
        # === END skip ===
'''
eval_source = eval_source.replace(pre_anchor, skip_injection + pre_anchor, 1)

# Inject post-edit recording + checkpointing
post_anchor = '{EVAL_POST_EDIT_ANCHOR}'
assert post_anchor in eval_source, "Post-edit anchor not found"
post_injection = '''        # === PLASTICITY: record batch metrics + checkpoint ===
        if '_plast_record_batch' in globals():
            _plast_record_batch(cnt, edited_model, hparams, time() - start)
        if '_plast_should_save' in globals() and _plast_should_save(cnt):
            _plast_save_checkpoint(cnt, edited_model,
                                   cache_c if 'cache_c' in dir() else None, hparams)
        # === END plasticity post-edit ===
'''
eval_source = eval_source.replace(post_anchor, post_injection + "        " + post_anchor, 1)

# Replace the AlphaEdit import with our patched version
# The evaluate.py imports: from AlphaEdit.AlphaEdit_main import execute_alphaedit
eval_source = eval_source.replace(
    "from AlphaEdit.AlphaEdit_main import execute_alphaedit",
    "# execute_alphaedit imported from patched namespace (plasticity_tracker)",
)

# ─── 5. Execute ─────────────────────────────────────────────────────
exec_namespace = {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "_plast_output": _plast_output,
    "_plast_batch_records": _plast_batch_records,
    "_plast_current_batch_projections": _plast_current_batch_projections,
    "_plast_log_projection": _plast_log_projection,
    "_plast_record_batch": _plast_record_batch,
    "_plast_should_skip": _plast_should_skip,
    "_plast_should_save": _plast_should_save,
    "_plast_save_checkpoint": _plast_save_checkpoint,
    "execute_alphaedit": _patched_execute_alphaedit,
}}

print("[PLASTICITY] Executing patched evaluate.py...")
exec(compile(eval_source, "experiments/evaluate.py", "exec"), exec_namespace)

# ─── 6. Summary ─────────────────────────────────────────────────────
print(f"\\n=== Plasticity tracking complete ===")
print(f"  Recorded {{len(_plast_batch_records)}} batches")
print(f"  Output: {{_plast_output}}")
if _plast_batch_records:
    last = _plast_batch_records[-1]
    print(f"  Last batch {{last['batch_idx']}}: removed_frac={{last.get('mean_removed_fraction', '?')}}")
""")
    return script


def main():
    parser = argparse.ArgumentParser(description="Online plasticity + projection tracker")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf")
    parser.add_argument("--dataset_size_limit", type=int, default=10000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--start_from_batch", type=int, default=0)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--downstream_eval_steps", type=int, default=999)
    args = parser.parse_args()

    # Resolve checkpoint dir
    if args.checkpoint_dir is None:
        s3_path = Path(f"/s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed{args.seed}")
        if s3_path.exists():
            args.checkpoint_dir = str(s3_path)
        else:
            args.checkpoint_dir = str(Path.home() / ".cache" / "alphaedit_checkpoints" / f"AlphaEdit/seed{args.seed}")

    # Output
    output_dir = PROJECT_ROOT / "results" / "plasticity_tracking"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = output_dir / f"plasticity_seed{args.seed}_{args.dataset_size_limit}edits_{timestamp}.jsonl"

    print("=" * 70)
    print("Online Plasticity + Projection Tracker")
    print(f"  Seed:           {args.seed}")
    print(f"  Model:          {args.model_name}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Batch size:     {args.num_edits}")
    print(f"  Save interval:  {args.save_interval}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Start batch:    {args.start_from_batch}")
    print(f"  Output:         {output_jsonl}")
    print("=" * 70)

    from model_download import resolve_model_path
    from setup_hparams import link_hparams
    from source_patches import patch_evaluate_file

    link_hparams()
    patch_evaluate_file(ALPHAEDIT_ROOT)
    model_name = resolve_model_path(args.model_name)

    script = build_plasticity_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        output_jsonl=str(output_jsonl),
        conserve_memory=args.conserve_memory,
        start_from_batch=args.start_from_batch,
        downstream_eval_steps=args.downstream_eval_steps,
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ALPHAEDIT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Run failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nPlasticity tracking complete. Output: {output_jsonl}")


if __name__ == "__main__":
    main()
