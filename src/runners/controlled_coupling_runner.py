#!/usr/bin/env python3
"""
Controlled Coupling Runner: Tests whether semantic structure of the edit stream
determines effective editing capacity.

Runs AlphaEdit on TWO matched streams (low-coupling vs high-coupling) with
inline mechanism measurement, particularly cache eigenspectrum tracking after
each cache update.

Architecture (dual source injection, following plasticity_tracker.py):
  1. Generate/load controlled coupling streams
  2. Patch AlphaEdit_main.py: inject cache eigenspectrum measurement after
     cache_c update + projection measurement after solve
  3. Patch evaluate.py: dataset override, batch recording, milestone eval,
     checkpoint save/load
  4. Execute as subprocess for each stream

Key hypothesis: High-coupling stream collapses 2K-3K edits earlier than
low-coupling stream because subject clustering causes faster key-space
concentration (cache dominance growth).

Output: JSONL per stream with per-batch mechanism + evaluation metrics.

Usage:
    python src/runners/controlled_coupling_runner.py \\
        --seed 42 --cuda_device 0 --stream_length 5000 \\
        --num_edits 100 --save_interval 10 --eval_at_checkpoints_only

    # Quick smoke test
    python src/runners/controlled_coupling_runner.py \\
        --seed 42 --stream_length 200 --num_edits 100 --save_interval 1
"""

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALPHAEDIT_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
SRC_DIR = PROJECT_ROOT / "src"

# NOTE: Do NOT add full SRC_DIR — src/datasets/ shadows HuggingFace 'datasets'
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))
if str(SRC_DIR / "datasets") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "datasets"))


# ─── Source Anchors (commit b84624f) ─────────────────────────────────────────

CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        exec_time = time() - start'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'

# Checkpoint loading anchor (injected after P initialization)
CHECKPOINT_LOAD_ANCHOR = '        torch.save(P, "null_space_project.pt")'

# AlphaEdit_main.py anchors
RESID_ANCHOR = '        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers'
UPD_ANCHOR = '        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)'

# Cache update anchor: where cache_c gets updated (after model edit, second layer loop)
CACHE_UPDATE_ANCHOR = '        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T'


def build_controlled_coupling_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_path: str,
    stream_name: str,
    stream_length: int,
    num_edits: int,
    save_interval: int,
    eval_at_checkpoints_only: bool,
    checkpoint_dir: str,
    output_jsonl: str,
    conserve_memory: bool = True,
    start_from_batch: int = 0,
) -> str:
    """
    Build inline script that patches both AlphaEdit_main.py and evaluate.py
    with inline cache eigenspectrum measurement + mechanism recording.
    """
    argv_parts = [
        "experiments.evaluate",
        "--alg_name=AlphaEdit",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={stream_length}",
        f"--num_edits={num_edits}",
        "--downstream_eval_steps=999",
        "--generation_test_interval=1",
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)

    # ─── Injection code for AlphaEdit_main.py ─────────────────────────────

    # After cache_c update: compute eigenspectrum snapshot
    # NOTE: At this point in the code, cache_c[i] is on CPU, layer variable is `layer`,
    # and the loop variable is `i` (enumerating hparams.layers)
    cache_eigenspectrum_injection = r'''
        # === CONTROLLED_COUPLING: cache eigenspectrum (injected) ===
        if '_cc_cache_metrics' in globals():
            import torch as _torch_cc
            import math as _math_cc
            try:
                _cc_C = cache_c[i].float().cuda()
                _cc_eigvals = _torch_cc.linalg.eigvalsh(_cc_C)
                _cc_svs = _cc_eigvals.flip(0).clamp(min=0).sqrt()
                _cc_nr = int((_cc_svs > 1e-5).sum().item())
                _cc_svs_pos = _cc_svs[_cc_svs > 1e-10]
                if len(_cc_svs_pos) > 1:
                    _cc_p = _cc_svs_pos / _cc_svs_pos.sum()
                    _cc_H = -(_cc_p * _torch_cc.log(_cc_p)).sum().item()
                    _cc_er = _math_cc.exp(_cc_H)
                else:
                    _cc_er = float(len(_cc_svs_pos))
                _cc_top_share = (_cc_svs[0] / _cc_svs.sum()).item() if _cc_svs.sum() > 0 else 0.0
                _cc_cond = (_cc_svs_pos[0] / _cc_svs_pos[-1]).item() if len(_cc_svs_pos) >= 2 else float("inf")
                _cc_cache_metrics.append({
                    "layer": layer,
                    "layer_position": i,
                    "numerical_rank": _cc_nr,
                    "effective_rank": round(_cc_er, 2),
                    "top_sv_share": round(_cc_top_share, 6),
                    "condition": round(_cc_cond, 2) if not _math_cc.isinf(_cc_cond) else "inf",
                })
                del _cc_C, _cc_eigvals, _cc_svs, _cc_svs_pos
                _torch_cc.cuda.empty_cache()
            except Exception as _cc_e:
                print(f"  [CC] Cache metric error: {_cc_e}")
        # === END cache eigenspectrum ===
'''

    # After upd_matrix_match_shape: measure projection constraint effect.
    # At this point: upd_matrix is the final update (after shape matching),
    # layer_ks, resid, P[i,:,:] are in scope.
    # We measure projection_loss = 1 - ||P @ K @ resid^T|| / ||K @ resid^T||
    # (same metric as coupling_stress_runner.py — what fraction of the desired
    # direction is removed by null-space projection P)
    projection_injection = r'''
        # === CONTROLLED_COUPLING: projection measurement (injected) ===
        if '_cc_projection_metrics' in globals():
            import torch as _torch_cc2
            try:
                _cc_rhs = layer_ks @ resid.T
                _cc_proj_rhs = P[i,:,:].cuda() @ _cc_rhs
                _cc_rhs_norm = _torch_cc2.linalg.norm(_cc_rhs).item()
                _cc_proj_norm = _torch_cc2.linalg.norm(_cc_proj_rhs).item()
                _cc_projection_loss = 1.0 - (_cc_proj_norm / max(_cc_rhs_norm, 1e-10))
                _cc_upd_norm = _torch_cc2.linalg.norm(upd_matrix).item()
                _cc_projection_metrics.append({
                    "layer": layer,
                    "rhs_norm": round(_cc_rhs_norm, 6),
                    "projected_rhs_norm": round(_cc_proj_norm, 6),
                    "removed_fraction": round(max(0.0, _cc_projection_loss), 6),
                    "update_norm": round(_cc_upd_norm, 6),
                })
                del _cc_rhs, _cc_proj_rhs
            except Exception as _cc_pe:
                print(f"  [CC] Projection metric error: {_cc_pe}")
        # === END projection measurement ===
'''

    # ─── Injection code for evaluate.py ───────────────────────────────────

    dataset_override = f'''
    # === CONTROLLED_COUPLING: dataset override (injected) ===
    import json as _json_cc
    with open("{dataset_path}", "r") as _ccf:
        _cc_stream_data = _json_cc.load(_ccf)
    ds.data = _cc_stream_data
    print(f"  [CC] Loaded {{len(_cc_stream_data)}} records ({stream_name} stream)")
    # === END dataset override ===
'''

    pre_batch_hook = f'''        # === CONTROLLED_COUPLING: pre-batch (injected) ===
        if '_cc_cache_metrics' in globals():
            _cc_cache_metrics.clear()
            _cc_projection_metrics.clear()
        if '_cc_should_skip' in globals() and _cc_should_skip(cnt):
            cnt += 1
            continue
        # === END pre-batch ===
'''

    post_batch_hook = f'''        # === CONTROLLED_COUPLING: post-batch (injected) ===
        if '_cc_record_batch' in globals():
            _cc_record_batch(cnt, edited_model, hparams, exec_time,
                             record_chunks, cache_c if 'cache_c' in dir() else None)
        if '_cc_should_save' in globals() and _cc_should_save(cnt):
            _cc_save_checkpoint(cnt, edited_model,
                                cache_c if 'cache_c' in dir() else None, hparams)
        # === END post-batch ===
'''

    # Use UPD_ANCHOR (after upd_matrix_match_shape) for projection measurement
    upd_anchor = UPD_ANCHOR

    script = textwrap.dedent(f"""\
import os, sys, random, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

# ─── 1. Seed ────────────────────────────────────────────────────────
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.argv = {argv_str}

# ─── 2. Tracking state ──────────────────────────────────────────────
_cc_output = "{output_jsonl}"
_cc_stream_name = "{stream_name}"
_cc_checkpoint_dir = Path("{checkpoint_dir}")
_cc_save_interval = {save_interval}
_cc_start_batch = {start_from_batch}
_cc_eval_at_checkpoints_only = {eval_at_checkpoints_only}
_cc_num_edits = {num_edits}
_cc_batch_records = []

# Per-batch mechanism tracking (populated by patched AlphaEdit_main)
_cc_cache_metrics = []
_cc_projection_metrics = []

def _cc_record_batch(cnt, model, hparams, exec_time, record_chunks, cache_c):
    \"\"\"Called after each edit batch. Records mechanism metrics.\"\"\"
    record = {{
        "stream": _cc_stream_name,
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _cc_num_edits,
        "seed": seed,
        "exec_time_s": round(exec_time, 2),
    }}

    # Aggregate cache metrics
    if _cc_cache_metrics:
        ers = [m["effective_rank"] for m in _cc_cache_metrics]
        nrs = [m["numerical_rank"] for m in _cc_cache_metrics]
        tops = [m["top_sv_share"] for m in _cc_cache_metrics]
        conds = [m["condition"] for m in _cc_cache_metrics if m["condition"] != "inf"]
        record["mechanism"] = {{
            "layers": list(_cc_cache_metrics),
            "aggregate": {{
                "mean_cache_effective_rank": round(np.mean(ers), 2),
                "mean_cache_numerical_rank": round(np.mean(nrs), 2),
                "mean_cache_top_sv_share": round(np.mean(tops), 6),
                "mean_cache_condition": round(np.mean(conds), 2) if conds else "inf",
            }},
        }}

    # Aggregate projection metrics
    if _cc_projection_metrics:
        fracs = [m["removed_fraction"] for m in _cc_projection_metrics]
        norms = [m["update_norm"] for m in _cc_projection_metrics]
        record["mechanism"] = record.get("mechanism", {{}})
        record["mechanism"]["aggregate"] = record["mechanism"].get("aggregate", {{}})
        record["mechanism"]["aggregate"]["mean_removed_fraction"] = round(np.mean(fracs), 6)
        record["mechanism"]["aggregate"]["mean_update_norm"] = round(np.mean(norms), 6)
        record["mechanism"]["projection_layers"] = list(_cc_projection_metrics)

    # Evaluation placeholder (filled by milestone eval if applicable)
    record["evaluation"] = {{
        "evaluated_at_this_batch": False,
        "overall_efficacy": None,
        "early_cohort_retention": None,
    }}

    _cc_batch_records.append(record)
    _cc_cache_metrics.clear()
    _cc_projection_metrics.clear()

    # Write full file each time (S3-FUSE doesn't support append mode)
    with open(_cc_output, "w") as f:
        for rec in _cc_batch_records:
            f.write(json.dumps(rec) + "\\n")

    if (cnt + 1) % 5 == 0:
        agg = record.get("mechanism", {{}}).get("aggregate", {{}})
        print(f"  [CC] Batch {{cnt}}: edits={{(cnt+1)*_cc_num_edits}}, "
              f"eff_rank={{agg.get('mean_cache_effective_rank', '?')}}, "
              f"removed_frac={{agg.get('mean_removed_fraction', '?')}}")

def _cc_should_skip(cnt):
    return cnt < _cc_start_batch

def _cc_should_save(cnt):
    return (cnt + 1) % _cc_save_interval == 0

def _cc_save_checkpoint(cnt, model, cache_c, hparams):
    ckpt_dir = _cc_checkpoint_dir / f"batch_{{cnt}}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    metadata = {{"batch_idx": cnt, "total_edits": (cnt + 1) * _cc_num_edits,
                 "stream": _cc_stream_name, "seed": seed,
                 "timestamp": datetime.now(timezone.utc).isoformat()}}
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"  [CC] Checkpoint saved: {{ckpt_dir}}")

# ─── 3. Patch AlphaEdit_main.py ─────────────────────────────────────
alphaedit_source_path = Path("AlphaEdit/AlphaEdit_main.py")
ae_source = alphaedit_source_path.read_text()

# Fix relative imports
ae_source = ae_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
ae_source = ae_source.replace("from .compute_z", "from AlphaEdit.compute_z")
ae_source = ae_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")

# Inject cache eigenspectrum after cache update
cache_anchor = {repr(CACHE_UPDATE_ANCHOR)}
assert cache_anchor in ae_source, (
    "CACHE_UPDATE_ANCHOR not found in AlphaEdit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
cache_injection = {repr(cache_eigenspectrum_injection)}
ae_source = ae_source.replace(cache_anchor, cache_anchor + "\\n" + cache_injection, 1)

# Inject projection measurement after upd_matrix_match_shape
_upd_anchor = {repr(upd_anchor)}
assert _upd_anchor in ae_source, (
    "UPD_ANCHOR not found in AlphaEdit_main.py."
)
proj_injection = {repr(projection_injection)}
ae_source = ae_source.replace(_upd_anchor, _upd_anchor + "\\n" + proj_injection, 1)

# Compile patched AlphaEdit main
ae_code = compile(ae_source, "AlphaEdit/AlphaEdit_main.py", "exec")
ae_namespace = {{
    "__name__": "AlphaEdit.AlphaEdit_main",
    "__file__": "AlphaEdit/AlphaEdit_main.py",
    "_cc_cache_metrics": _cc_cache_metrics,
    "_cc_projection_metrics": _cc_projection_metrics,
}}
exec(ae_code, ae_namespace)
_patched_apply = ae_namespace["apply_AlphaEdit_to_model"]
_patched_get_cov = ae_namespace["get_cov"]
print("[CC] AlphaEdit_main.py patched with cache eigenspectrum + projection measurement")

# ─── 4. Patch evaluate.py ───────────────────────────────────────────
with open("experiments/evaluate.py", "r") as f:
    eval_source = f.read()

# Replace AlphaEdit import
import_anchor = {repr(ALGO_IMPORT_ANCHOR)}
assert import_anchor in eval_source, "ALGO_IMPORT_ANCHOR not found in evaluate.py."
eval_source = eval_source.replace(
    import_anchor,
    "# apply_AlphaEdit_to_model patched by controlled_coupling_runner",
)

# Patch CUDA
cuda_target = {repr(CUDA_PATCH_TARGET)}
assert cuda_target in eval_source, "CUDA patch target not found in evaluate.py."
eval_source = eval_source.replace(cuda_target, "# CUDA managed by controlled_coupling_runner")

# Inject dataset override before for loop
shuffle_anchor = {repr(SHUFFLE_ANCHOR)}
assert shuffle_anchor in eval_source, "SHUFFLE_ANCHOR not found in evaluate.py."
dataset_override = {repr(dataset_override)}
eval_source = eval_source.replace(shuffle_anchor, dataset_override + "\\n" + shuffle_anchor, 1)

# Inject pre-batch hook
pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert pre_anchor in eval_source, "PRE_EDIT_ANCHOR not found in evaluate.py."
pre_hook = {repr(pre_batch_hook)}
eval_source = eval_source.replace(pre_anchor, pre_hook + pre_anchor, 1)

# Inject post-batch hook (AFTER exec_time line, preserving if/elif/else chain)
post_anchor = {repr(POST_EDIT_ANCHOR)}
assert post_anchor in eval_source, "POST_EDIT_ANCHOR not found in evaluate.py."
post_hook = {repr(post_batch_hook)}
eval_source = eval_source.replace(post_anchor, post_anchor + "\\n" + post_hook, 1)


# Inject checkpoint loading after P initialization
ckpt_load_anchor = {repr(CHECKPOINT_LOAD_ANCHOR)}
if _cc_start_batch > 0:
    assert ckpt_load_anchor in eval_source, "CHECKPOINT_LOAD_ANCHOR not found in evaluate.py."
    _ckpt_injection = '''
    # === CONTROLLED_COUPLING: checkpoint resumption (injected) ===
    _ckpt_dir = Path("{checkpoint_dir}") / f"batch_{{_cc_start_batch - 1}}"
    if (_ckpt_dir / "model_weights.pt").exists():
        print(f"  [CC] Loading checkpoint from {{_ckpt_dir}}")
        _ckpt_weights = torch.load(_ckpt_dir / "model_weights.pt", map_location="cuda")
        _param_dict = dict(model.named_parameters())
        _loaded = 0
        for _wname, _wtensor in _ckpt_weights.items():
            if _wname in _param_dict:
                _param_dict[_wname].data.copy_(_wtensor.cuda())
                _loaded += 1
        del _ckpt_weights
        print(f"    Model weights restored ({{_loaded}} params)")
        if (_ckpt_dir / "cache_c.pt").exists():
            cache_c = torch.load(_ckpt_dir / "cache_c.pt", map_location="cpu")
            print(f"    cache_c restored: shape={{cache_c.shape}}")
        else:
            print("    WARNING: no cache_c.pt in checkpoint, starting fresh cache")
        torch.cuda.empty_cache()
        print(f"  [CC] Resumed from batch {{_cc_start_batch - 1}} ({{_cc_start_batch * {num_edits}}} edits)")
    else:
        print(f"  [CC] WARNING: checkpoint dir {{_ckpt_dir}} not found, starting from scratch")
    # === END checkpoint resumption ===
'''
    eval_source = eval_source.replace(ckpt_load_anchor, ckpt_load_anchor + "\\n" + _ckpt_injection, 1)
    print(f"[CC] Checkpoint resumption injected (start_from_batch={{_cc_start_batch}})")

print("[CC] evaluate.py patched with dataset override + mechanism hooks")

# ─── 5. Execute ─────────────────────────────────────────────────────
exec(compile(eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_AlphaEdit_to_model": _patched_apply,
    "get_cov": _patched_get_cov,
    "_cc_output": _cc_output,
    "_cc_stream_name": _cc_stream_name,
    "_cc_cache_metrics": _cc_cache_metrics,
    "_cc_projection_metrics": _cc_projection_metrics,
    "_cc_record_batch": _cc_record_batch,
    "_cc_should_skip": _cc_should_skip,
    "_cc_should_save": _cc_should_save,
    "_cc_save_checkpoint": _cc_save_checkpoint,
    "_cc_batch_records": _cc_batch_records,
    "_cc_checkpoint_dir": _cc_checkpoint_dir,
    "_cc_save_interval": _cc_save_interval,
    "_cc_start_batch": _cc_start_batch,
    "_cc_eval_at_checkpoints_only": _cc_eval_at_checkpoints_only,
    "_cc_num_edits": _cc_num_edits,
}})

# ─── 6. Summary ─────────────────────────────────────────────────────
print(f"\\n=== Controlled Coupling ({{_cc_stream_name}}) complete ===")
print(f"  Recorded {{len(_cc_batch_records)}} batches")
print(f"  Output: {{_cc_output}}")
if _cc_batch_records:
    last = _cc_batch_records[-1]
    agg = last.get("mechanism", {{}}).get("aggregate", {{}})
    print(f"  Last batch {{last['batch_idx']}}: "
          f"eff_rank={{agg.get('mean_cache_effective_rank', '?')}}, "
          f"removed_frac={{agg.get('mean_removed_fraction', '?')}}")
""")
    return script


def validate_anchors() -> None:
    """Verify that all source anchors exist in the pinned code."""
    eval_path = ALPHAEDIT_ROOT / "experiments" / "evaluate.py"
    eval_source = eval_path.read_text()

    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("SHUFFLE_ANCHOR", SHUFFLE_ANCHOR),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("ALGO_IMPORT_ANCHOR", ALGO_IMPORT_ANCHOR),
        ("CHECKPOINT_LOAD_ANCHOR", CHECKPOINT_LOAD_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    algo_path = ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py"
    algo_source = algo_path.read_text()

    for name, anchor in [
        ("RESID_ANCHOR", RESID_ANCHOR),
        ("UPD_ANCHOR", UPD_ANCHOR),
        ("CACHE_UPDATE_ANCHOR", CACHE_UPDATE_ANCHOR),
    ]:
        assert anchor in algo_source, f"{name} not found in AlphaEdit_main.py"

    print("  All source anchors validated.")


def run_stream(
    stream_name: str,
    dataset_path: str,
    args: argparse.Namespace,
    model_name: str,
    output_jsonl: Path,
    checkpoint_dir: Path,
) -> int:
    """Run one stream (low or high coupling). Returns subprocess return code."""
    script = build_controlled_coupling_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name="mcf",
        dataset_path=dataset_path,
        stream_name=stream_name,
        stream_length=args.stream_length,
        num_edits=args.num_edits,
        save_interval=args.save_interval,
        eval_at_checkpoints_only=args.eval_at_checkpoints_only,
        checkpoint_dir=str(checkpoint_dir),
        output_jsonl=str(output_jsonl),
        conserve_memory=args.conserve_memory,
        start_from_batch=args.start_from_batch,
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\n{'─' * 70}")
    print(f"Running {stream_name} stream...")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output:  {output_jsonl}")
    print(f"{'─' * 70}")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ALPHAEDIT_ROOT),
        env=env,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Controlled coupling experiment: tests semantic structure vs editing capacity"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get(
        "MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_at_checkpoints_only", action="store_true", default=False)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--start_from_batch", type=int, default=0)
    parser.add_argument("--stream", choices=["both", "low", "high"], default="both",
                        help="Which stream(s) to run")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    from model_download import resolve_model_path
    from setup_hparams import link_hparams
    from source_patches import patch_evaluate_file

    link_hparams()
    patch_evaluate_file(ALPHAEDIT_ROOT)
    model_name = resolve_model_path(args.model_name)

    # Validate anchors
    print("Validating source anchors...")
    validate_anchors()

    # Results directory
    results_dir = PROJECT_ROOT / "results" / "controlled_coupling"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load streams
    data_dir = Path(args.data_dir) if args.data_dir else ALPHAEDIT_ROOT / "data"
    low_path = results_dir / f"low_coupling_seed{args.seed}.json"
    high_path = results_dir / f"high_coupling_seed{args.seed}.json"

    if low_path.exists() and high_path.exists():
        print(f"\nLoading cached streams...")
        print(f"  Low:  {low_path}")
        print(f"  High: {high_path}")
    else:
        print("\nGenerating controlled coupling streams...")
        from controlled_coupling_dataset import generate_controlled_streams, validate_stream_properties

        low_stream, high_stream = generate_controlled_streams(
            data_dir=data_dir,
            seed=args.seed,
            stream_length=args.stream_length,
            batch_size=args.num_edits,
        )

        props = validate_stream_properties(low_stream, high_stream, args.num_edits)
        print(f"  Coupling differential: {props['coupling_differential']:.2f}")

        with open(low_path, "w") as f:
            json.dump(low_stream, f)
        with open(high_path, "w") as f:
            json.dump(high_stream, f)
        with open(results_dir / f"stream_properties_seed{args.seed}.json", "w") as f:
            json.dump(props, f, indent=2)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 70}")
    print("Controlled Coupling Experiment")
    print(f"  Seed:          {args.seed}")
    print(f"  Model:         {model_name}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.num_edits}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Eval mode:     {'milestone' if args.eval_at_checkpoints_only else 'every batch'}")
    print(f"  Stream(s):     {args.stream}")
    print(f"  Started:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "seed": args.seed,
        "model_name": model_name,
        "stream_length": args.stream_length,
        "num_edits": args.num_edits,
        "save_interval": args.save_interval,
        "eval_at_checkpoints_only": args.eval_at_checkpoints_only,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
    }
    with open(results_dir / f"metadata_seed{args.seed}_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Run streams
    streams_to_run = []
    if args.stream in ("both", "low"):
        streams_to_run.append(("low_coupling", str(low_path)))
    if args.stream in ("both", "high"):
        streams_to_run.append(("high_coupling", str(high_path)))

    # Resolve checkpoint base: prefer S3-mounted path for crash resilience
    s3_ckpt_base = Path("/s3-data/continual-learning/alphaedit/checkpoints/controlled_coupling")
    if s3_ckpt_base.parent.parent.exists():
        ckpt_base = s3_ckpt_base
        print(f"  Checkpoints: {ckpt_base} (S3-mounted, crash-resilient)")
    else:
        ckpt_base = results_dir / "checkpoints"
        print(f"  Checkpoints: {ckpt_base} (local)")

    for stream_name, dataset_path in streams_to_run:
        # Write JSONL to checkpoint dir (S3-mounted = crash-resilient)
        ckpt_dir = ckpt_base / f"{stream_name}" / f"seed{args.seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = ckpt_dir / f"{stream_name}_seed{args.seed}_{timestamp}.jsonl"

        # Auto-detect checkpoint for resumption
        start_batch = args.start_from_batch
        if start_batch == 0:
            existing_ckpts = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
                key=lambda p: int(p.name.split("_")[1]),
            ) if ckpt_dir.exists() else []
            if existing_ckpts:
                last_ckpt = existing_ckpts[-1]
                last_batch = int(last_ckpt.name.split("_")[1])
                # Only resume if the checkpoint has model weights
                if (last_ckpt / "model_weights.pt").exists():
                    start_batch = last_batch + 1
                    print(f"  [CC] Auto-resume: found checkpoint at batch_{last_batch}, starting from batch {start_batch}")

        # Override start_from_batch for this stream
        args.start_from_batch = start_batch

        rc = run_stream(
            stream_name=stream_name,
            dataset_path=dataset_path,
            args=args,
            model_name=model_name,
            output_jsonl=output_jsonl,
            checkpoint_dir=ckpt_dir,
        )

        if rc != 0:
            print(f"\nERROR: {stream_name} stream failed with code {rc}")
            if args.stream == "both":
                print("Continuing with next stream...")
            else:
                sys.exit(rc)

    print(f"\n{'=' * 70}")
    print("Controlled coupling experiment complete.")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Results:  {results_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
