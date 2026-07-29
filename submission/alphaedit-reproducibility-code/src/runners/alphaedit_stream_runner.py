#!/usr/bin/env python3
"""
AlphaEdit Stream Runner: Runs AlphaEdit on a pre-generated edit stream with
inline mechanism measurement (cache eigenspectrum, projection loss) and
checkpointing.

Architecture (dual source injection):
  1. Patch AlphaEdit_main.py: inject cache eigenspectrum measurement after
     cache_c update + projection measurement after solve
  2. Patch evaluate.py: dataset override, batch recording, milestone eval,
     checkpoint save/load
  3. Execute as subprocess

Output: JSONL per stream with per-batch mechanism + evaluation metrics.

Usage:
    python src/runners/alphaedit_stream_runner.py \\
        --seed 42 --cuda_device 0 --stream_length 5000 \\
        --stream clustered --stream_path /path/to/stream.json \\
        --checkpoint_base /path/to/checkpoints \\
        --num_edits 100 --save_interval 10 --eval_at_checkpoints_only
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
SRC_DIR = PROJECT_ROOT / "src"

# NOTE: Do NOT add full SRC_DIR — src/datasets/ shadows HuggingFace 'datasets'
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))

from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root

ALPHAEDIT_ROOT = get_alphaedit_root()


# ─── Source Anchors (commit b84624f) ─────────────────────────────────────────

CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        exec_time = time() - start'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'

# Checkpoint loading anchor: inject BEFORE glue_save_location (after P + cache_c init)
# This anchor is stable regardless of whether P-cache patch has been applied.
CHECKPOINT_LOAD_ANCHOR = "    glue_save_location = str(run_dir) + '/' + 'glue_eval/'"

# AlphaEdit_main.py anchors
RESID_ANCHOR = '        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers'
UPD_ANCHOR = '        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)'

# Functional projection loss anchor: inject BEFORE shape matching (upd_matrix is [d_in, d_out])
FUNCTIONAL_LOSS_ANCHOR = '        # Adjust update matrix shape'

# Cache update anchor: where cache_c gets updated (after model edit, second layer loop)
CACHE_UPDATE_ANCHOR = '        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T'


def build_alphaedit_stream_script(
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
        # === STREAM_RUNNER: cache eigenspectrum (injected) ===
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

    # Before upd_matrix_match_shape: compute functional projection loss.
    # At this point: upd_matrix is [d_in, d_out] (pre-match), and all solve
    # variables are in scope: layer_ks [d_in, n], resid [d_out, n], P[i,:,:],
    # cache_c[i,:,:], hparams.L2.
    #
    # Key metrics:
    #   q_t = ||ΔW_proj.T @ K_new|| / ||ΔW_raw.T @ K_new||
    #     - ratio of edit signal preserved after projection
    #   fit_quality_proj = 1 - ||resid - ΔW_proj.T @ K|| / ||resid||
    #     - how well projected update achieves edit target
    #   fit_quality_raw = 1 - ||resid - ΔW_raw.T @ K|| / ||resid||
    #     - how well unconstrained update achieves edit target
    #   removed_fraction = 1 - ||P @ K @ resid^T|| / ||K @ resid^T||
    #     - RHS-level signal removal (fast proxy)
    functional_projection_injection = r'''
        # === STREAM_RUNNER: functional projection loss (injected) ===
        if '_cc_projection_metrics' in globals():
            import torch as _torch_fpl
            try:
                # --- RHS-level removed fraction (fast, backward compat) ---
                _fpl_rhs = layer_ks @ resid.T
                _fpl_proj_rhs = P[i,:,:].cuda() @ _fpl_rhs
                _fpl_rhs_norm = _torch_fpl.linalg.norm(_fpl_rhs).item()
                _fpl_proj_rhs_norm = _torch_fpl.linalg.norm(_fpl_proj_rhs).item()
                _fpl_removed_fraction = max(0.0, 1.0 - (_fpl_proj_rhs_norm / max(_fpl_rhs_norm, 1e-10)))

                # --- Functional q_t: raw vs projected solve ---
                # upd_matrix is the PROJECTED solve result [d_in, d_out]
                # Compute raw (unconstrained) solve: (K@K^T + C + λI) X = K @ resid^T
                _fpl_lhs_raw = (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda() +
                                hparams.L2 * _torch_fpl.eye(layer_ks.shape[0], device="cuda", dtype=_torch_fpl.float))
                _fpl_upd_raw = _torch_fpl.linalg.solve(_fpl_lhs_raw, _fpl_rhs)

                # Effect on current edit keys (output-space change)
                # upd_matrix shape: [d_in, d_out], so .T is [d_out, d_in]
                # layer_ks shape: [d_in, n]
                # effect shape: [d_out, n] — matches resid
                _fpl_effect_proj = upd_matrix.T @ layer_ks   # [d_out, n]
                _fpl_effect_raw = _fpl_upd_raw.T @ layer_ks  # [d_out, n]

                _fpl_effect_proj_norm = _torch_fpl.linalg.norm(_fpl_effect_proj).item()
                _fpl_effect_raw_norm = _torch_fpl.linalg.norm(_fpl_effect_raw).item()

                # q_t: functional signal preservation ratio
                _fpl_q_t = _fpl_effect_proj_norm / max(_fpl_effect_raw_norm, 1e-10)

                # Fit quality: how well each solve achieves the target residual
                _fpl_resid_norm = _torch_fpl.linalg.norm(resid).item()
                _fpl_fit_proj = 1.0 - (_torch_fpl.linalg.norm(resid - _fpl_effect_proj).item() /
                                       max(_fpl_resid_norm, 1e-10))
                _fpl_fit_raw = 1.0 - (_torch_fpl.linalg.norm(resid - _fpl_effect_raw).item() /
                                      max(_fpl_resid_norm, 1e-10))

                _cc_projection_metrics.append({
                    "layer": layer,
                    "rhs_norm": round(_fpl_rhs_norm, 6),
                    "projected_rhs_norm": round(_fpl_proj_rhs_norm, 6),
                    "removed_fraction": round(_fpl_removed_fraction, 6),
                    "update_norm": round(_torch_fpl.linalg.norm(upd_matrix).item(), 6),
                    "q_t": round(_fpl_q_t, 6),
                    "fit_quality_projected": round(_fpl_fit_proj, 6),
                    "fit_quality_raw": round(_fpl_fit_raw, 6),
                    "effect_norm_projected": round(_fpl_effect_proj_norm, 6),
                    "effect_norm_raw": round(_fpl_effect_raw_norm, 6),
                    "target_norm": round(_fpl_resid_norm, 6),
                })
                del _fpl_rhs, _fpl_proj_rhs, _fpl_lhs_raw, _fpl_upd_raw
                del _fpl_effect_proj, _fpl_effect_raw
                _torch_fpl.cuda.empty_cache()
            except Exception as _fpl_e:
                print(f"  [CC] Functional projection loss error: {_fpl_e}")
        # === END functional projection loss ===
'''

    # ─── Injection code for evaluate.py ───────────────────────────────────

    dataset_override = f'''
    # === STREAM_RUNNER: dataset override (injected) ===
    import json as _json_cc
    with open("{dataset_path}", "r") as _ccf:
        _cc_stream_data = _json_cc.load(_ccf)
    ds.data = _cc_stream_data
    print(f"  [CC] Loaded {{len(_cc_stream_data)}} records ({stream_name} stream)")
    # === END dataset override ===
'''

    pre_batch_hook = f'''        # === STREAM_RUNNER: pre-batch (injected) ===
        if '_cc_cache_metrics' in globals():
            _cc_cache_metrics.clear()
            _cc_projection_metrics.clear()
        if '_cc_should_skip' in globals() and _cc_should_skip(cnt):
            cnt += 1
            continue
        # === END pre-batch ===
'''

    post_batch_hook = f'''        # === STREAM_RUNNER: post-batch (injected) ===
        if '_cc_record_batch' in globals():
            _cc_record_batch(cnt, edited_model, hparams, exec_time,
                             record_chunks, cache_c if 'cache_c' in dir() else None)
        if '_cc_should_save' in globals() and _cc_should_save(cnt):
            _cc_save_checkpoint(cnt, edited_model,
                                cache_c if 'cache_c' in dir() else None, hparams)
        # === END post-batch ===
'''

    # Use FUNCTIONAL_LOSS_ANCHOR (before upd_matrix_match_shape) for projection measurement
    functional_anchor = FUNCTIONAL_LOSS_ANCHOR

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

    # Aggregate projection + functional metrics
    if _cc_projection_metrics:
        fracs = [m["removed_fraction"] for m in _cc_projection_metrics]
        norms = [m["update_norm"] for m in _cc_projection_metrics]
        q_ts = [m["q_t"] for m in _cc_projection_metrics if "q_t" in m]
        fits_proj = [m["fit_quality_projected"] for m in _cc_projection_metrics if "fit_quality_projected" in m]
        fits_raw = [m["fit_quality_raw"] for m in _cc_projection_metrics if "fit_quality_raw" in m]
        record["mechanism"] = record.get("mechanism", {{}})
        record["mechanism"]["aggregate"] = record["mechanism"].get("aggregate", {{}})
        record["mechanism"]["aggregate"]["mean_removed_fraction"] = round(np.mean(fracs), 6)
        record["mechanism"]["aggregate"]["mean_update_norm"] = round(np.mean(norms), 6)
        if q_ts:
            record["mechanism"]["aggregate"]["mean_q_t"] = round(np.mean(q_ts), 6)
            record["mechanism"]["aggregate"]["min_q_t"] = round(min(q_ts), 6)
        if fits_proj:
            record["mechanism"]["aggregate"]["mean_fit_quality_projected"] = round(np.mean(fits_proj), 6)
            record["mechanism"]["aggregate"]["mean_fit_quality_raw"] = round(np.mean(fits_raw), 6)
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
              f"removed_frac={{agg.get('mean_removed_fraction', '?')}}, "
              f"q_t={{agg.get('mean_q_t', '?')}}")

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

# Inject functional projection loss BEFORE shape matching (upd_matrix is pre-match)
_func_anchor = {repr(functional_anchor)}
assert _func_anchor in ae_source, (
    "FUNCTIONAL_LOSS_ANCHOR not found in AlphaEdit_main.py."
)
func_injection = {repr(functional_projection_injection)}
ae_source = ae_source.replace(_func_anchor, func_injection + _func_anchor, 1)

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
    "# apply_AlphaEdit_to_model patched by alphaedit_stream_runner",
)

# Patch CUDA
cuda_target = {repr(CUDA_PATCH_TARGET)}
assert cuda_target in eval_source, "CUDA patch target not found in evaluate.py."
eval_source = eval_source.replace(cuda_target, "# CUDA managed by alphaedit_stream_runner")

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


# Inject checkpoint loading BEFORE glue_save_location (after P + cache_c fully initialized)
ckpt_load_anchor = {repr(CHECKPOINT_LOAD_ANCHOR)}
if _cc_start_batch > 0:
    assert ckpt_load_anchor in eval_source, "CHECKPOINT_LOAD_ANCHOR not found in evaluate.py."
    _ckpt_injection = '''
    # === STREAM_RUNNER: checkpoint resumption (injected) ===
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
    eval_source = eval_source.replace(ckpt_load_anchor, _ckpt_injection + "\\n" + ckpt_load_anchor, 1)
    print(f"[CC] Checkpoint resumption injected (start_from_batch={{_cc_start_batch}})")

# Always inject defaults before the for loop to prevent UnboundLocalError
# when all batches are skipped (checkpoint resumption or already_finished).
# The CHECKPOINT_LOAD_ANCHOR is just before cnt=0 and the editing for-loop.
_exec_time_default = '''
    exec_time = 0  # Default: overwritten by each batch, prevents UnboundLocalError if all skipped
    edited_model = model  # Default: checkpoint loading already modified model in-place
'''
assert ckpt_load_anchor in eval_source, "CHECKPOINT_LOAD_ANCHOR not found for exec_time init."
eval_source = eval_source.replace(ckpt_load_anchor, ckpt_load_anchor + "\\n" + _exec_time_default, 1)

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
        ("FUNCTIONAL_LOSS_ANCHOR", FUNCTIONAL_LOSS_ANCHOR),
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
    script = build_alphaedit_stream_script(
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
        description="AlphaEdit stream runner: runs AlphaEdit on a pre-generated edit stream with mechanism measurement"
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
    parser.add_argument("--stream", choices=["clustered", "dispersed", "key_clustered", "key_dispersed", "custom"], default="clustered",
                        help="Stream type name (used in output paths)")
    parser.add_argument("--stream_path", type=str, required=True,
                        help="Path to stream JSON file")
    parser.add_argument("--stream_name", type=str, default=None,
                        help="Override stream name for output paths")
    parser.add_argument("--checkpoint_base", type=str, default=None,
                        help="Override checkpoint base directory")
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

    stream_name = args.stream_name or args.stream
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 70}")
    print("AlphaEdit Stream Runner")
    print(f"  Seed:          {args.seed}")
    print(f"  Model:         {model_name}")
    print(f"  Stream:        {stream_name}")
    print(f"  Stream path:   {args.stream_path}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.num_edits}")
    print(f"  Save interval: {args.save_interval}")
    print(f"  Eval mode:     {'milestone' if args.eval_at_checkpoints_only else 'every batch'}")
    print(f"  Started:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Resolve checkpoint base
    if args.checkpoint_base:
        ckpt_dir = Path(args.checkpoint_base)
        print(f"  Checkpoints: {ckpt_dir} (user-specified)")
    else:
        ckpt_dir = get_checkpoint_root() / "matched_ordering" / f"{stream_name}" / f"seed{args.seed}"
        print(f"  Checkpoints: {ckpt_dir}")
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
            if (last_ckpt / "model_weights.pt").exists():
                start_batch = last_batch + 1
                print(f"  Auto-resume: found checkpoint at batch_{last_batch}, starting from batch {start_batch}")

    args.start_from_batch = start_batch

    rc = run_stream(
        stream_name=stream_name,
        dataset_path=args.stream_path,
        args=args,
        model_name=model_name,
        output_jsonl=output_jsonl,
        checkpoint_dir=ckpt_dir,
    )

    if rc != 0:
        print(f"\nERROR: {stream_name} stream failed with code {rc}")
        sys.exit(rc)

    print(f"\n{'=' * 70}")
    print("AlphaEdit stream run complete.")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Output:   {output_jsonl}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
