#!/usr/bin/env python3
"""
Kernel Editor — Prototype kernel-augmented knowledge editor.

Replaces the linear K@K^T outer product in MEMIT/AlphaEdit's solve with a
kernel-weighted version that amplifies regularization in crowded key directions:

    Polynomial kernel (--kernel_type poly):
        G_poly = (1 + K^T @ K)^p             (element-wise, degree-p)
        scale  = trace(G_lin) / (G_poly * G_lin).sum()   (equalize KKT trace)
        KKT    = K @ (G_poly * scale) @ K^T

    RBF kernel (--kernel_type rbf):
        dist²[i,j] = ||k_i||² + ||k_j||² - 2·k_i^T·k_j
        σ² = median(dist²)  or user-specified
        G_rbf = exp(-dist² / (2σ²))
        scale = trace(G_lin) / (G_rbf * G_lin).sum()    (equalize KKT trace)
        KKT   = K @ (G_rbf * scale) @ K^T

This replaces K@K^T in both:
    MEMIT:     solve(α·C₀ + KKT_kernel, K)
    AlphaEdit: solve(P @ (KKT_kernel + cache_c) + L2·I, P @ K @ R^T)

For AlphaEdit, the cache_c accumulation also uses the kernel-weighted version
so cumulative regularization properly accounts for crowding.

Motivation: The polykernel diagnostic shows that linear key-space effective rank
saturates at ~500 (L8) for 2000 edits, while poly2 would provide 2.6× more
independent directions. The kernel editor amplifies regularization in crowded
directions without requiring an explicit feature-space lifting.

Implementation: Dual source injection (patches algo file + evaluate.py).

Usage:
    python src/polykernel_editor_runner.py \
        --seed 42 --alg_name AlphaEdit \
        --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \
        --kernel_type poly --kernel_degree 2 \
        --downstream_eval_steps 10 --conserve_memory

    python src/polykernel_editor_runner.py \
        --seed 42 --alg_name AlphaEdit \
        --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \
        --kernel_type rbf --kernel_sigma median \
        --downstream_eval_steps 10 --conserve_memory
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from model_download import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    return get_project_root() / "vendor" / "AlphaEdit"


# --- Source anchors (commit b84624f) ---

# evaluate.py
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'
MEMIT_IMPORT_ANCHOR = 'from memit.memit_main import apply_memit_to_model, get_context_templates'

# evaluate.py anchors for checkpoint/eval modes
LOOP_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
EXEC_TIME_ANCHOR = '        exec_time = time() - start'
EVAL_START_ANCHOR = '    # torch.save(hs, "post_edit_hs_memit.pt")\n    start = time()'
EVAL_LOOP_ANCHOR = '    for record in ds:\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'

# AlphaEdit_main.py anchors
ALPHAEDIT_SOLVE_ANCHOR = '        upd_matrix = torch.linalg.solve(\n                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T\n        )'
ALPHAEDIT_CACHE_ANCHOR = '        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T'

# memit_main.py anchors
MEMIT_SOLVE_ANCHOR = '        adj_k = torch.linalg.solve(\n            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,\n            layer_ks,\n        )'


def build_editor_script(
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
    kernel_degree: int,
    kernel_type: str,
    kernel_sigma: str,
    output_jsonl: str,
    edit_only: bool = False,
    save_interval: int = 10,
    checkpoint_dir: str = "",
    eval_only: bool = False,
    load_checkpoint: str = "",
) -> str:
    """
    Build inline Python script for the kernel editor.
    Uses dual source injection: patches algo file and evaluate.py.

    Modes:
      - Normal: edit + evaluate (default, for small experiments)
      - edit_only: apply all edits, checkpoint every save_interval batches, skip eval
      - eval_only: load checkpoint, skip editing, run full evaluation
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
        "--skip_generation_tests",
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)

    # --- Shared kernel Gram computation (used by all injection sites) ---
    # This snippet computes _G_kernel from _G_lin based on kernel_type.
    # Expects _G_lin to be defined. Sets _G_kernel and _scale.
    kernel_gram_code = r'''
        _trace_lin = _G_lin.trace().clamp(min=1e-12)
        if _polykernel_type == "poly":
            _G_kernel = (1.0 + _G_lin).pow(_polykernel_degree)
        else:  # rbf
            _diag = _G_lin.diag()
            _dist_sq = _diag.unsqueeze(0) + _diag.unsqueeze(1) - 2.0 * _G_lin
            _dist_sq = _dist_sq.clamp(min=0.0)
            if _polykernel_sigma == "median":
                # Median heuristic: use median of off-diagonal distances
                _n = _dist_sq.shape[0]
                _mask = ~torch.eye(_n, dtype=torch.bool, device=_dist_sq.device)
                _sigma_sq = _dist_sq[_mask].median().clamp(min=1e-12)
            else:
                _sigma_sq = torch.tensor(float(_polykernel_sigma) ** 2, device=_dist_sq.device, dtype=_dist_sq.dtype)
            _G_kernel = (-_dist_sq / (2.0 * _sigma_sq)).exp()
        _frob_inner = (_G_kernel * _G_lin).sum().clamp(min=1e-12)
        _scale = (_trace_lin / _frob_inner).item()'''

    # --- AlphaEdit injection code ---
    alphaedit_solve_replacement = r'''        # === KERNEL EDITOR: kernel-weighted solve (injected) ===
        _G_lin = layer_ks.T @ layer_ks  # (n, n)''' + kernel_gram_code + r'''
        _KKT_kernel = layer_ks @ (_G_kernel * _scale) @ layer_ks.T  # (d, d)
        # Log kernel metrics
        if '_polykernel_log' in globals():
            _polykernel_log.append({
                "batch": _polykernel_batch_idx, "layer": int(layer),
                "trace_ratio": _scale,
                "G_lin_rank": int((_G_lin.diag() > 1e-8).sum().item()),
                "kernel_type": _polykernel_type,
                "phase": "solve",
            })
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (_KKT_kernel + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        del _G_lin, _G_kernel, _KKT_kernel
        # === END kernel solve ==='''

    alphaedit_cache_replacement = r'''        # === KERNEL EDITOR: kernel-weighted cache accumulation (injected) ===
        _ks_cpu = layer_ks.cpu()
        _G_lin = _ks_cpu.T @ _ks_cpu''' + kernel_gram_code.replace('_dist_sq.device', '_G_lin.device') + r'''
        cache_c[i,:,:] += _ks_cpu @ (_G_kernel * _scale) @ _ks_cpu.T
        del _ks_cpu, _G_lin, _G_kernel
        # === END kernel cache ==='''

    # --- MEMIT injection code ---
    memit_solve_replacement = r'''        # === KERNEL EDITOR: kernel-weighted solve (injected) ===
        _G_lin = layer_ks.T @ layer_ks  # (n, n) in double''' + kernel_gram_code + r'''
        _KKT_kernel = layer_ks @ (_G_kernel * _scale) @ layer_ks.T  # (d, d)
        # Log kernel metrics
        if '_polykernel_log' in globals():
            _polykernel_log.append({
                "batch": _polykernel_batch_idx, "layer": int(layer),
                "trace_ratio": _scale,
                "G_lin_rank": int((_G_lin.diag() > 1e-8).sum().item()),
                "kernel_type": _polykernel_type,
                "phase": "solve",
            })
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + _KKT_kernel,
            layer_ks,
        )
        del _G_lin, _G_kernel, _KKT_kernel
        # === END kernel solve ==='''

    # --- evaluate.py injection: batch counter ---
    # NOTE: must use globals() dict access, not bare assignment, because
    # bare += would make Python's compiler treat the var as local (UnboundLocalError)
    batch_increment_hook = r'''        # === POLYKERNEL EDITOR: increment batch (injected) ===
        if '_polykernel_batch_idx' in globals():
            globals()['_polykernel_batch_idx'] += 1
        # === END batch increment ===
'''

    # Build script based on algorithm
    if alg_name == "AlphaEdit":
        algo_file = "AlphaEdit/AlphaEdit_main.py"
        algo_module_name = "AlphaEdit.AlphaEdit_main"
        import_fixes = """
_algo_source = _algo_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
_algo_source = _algo_source.replace("from .compute_z", "from AlphaEdit.compute_z")
_algo_source = _algo_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")
"""
        import_anchor = ALGO_IMPORT_ANCHOR
        import_replacement = "# apply_AlphaEdit_to_model patched by polykernel_editor_runner"
        apply_fn_name = "apply_AlphaEdit_to_model"
        extra_fn_name = "get_cov"
        extra_globals = '"get_cov": _patched_extra_fn,'
        solve_anchor_repr = repr(ALPHAEDIT_SOLVE_ANCHOR)
        solve_replacement_repr = repr(alphaedit_solve_replacement)
        cache_patch_code = f"""
# Patch cache_c accumulation
_cache_anchor = {repr(ALPHAEDIT_CACHE_ANCHOR)}
assert _cache_anchor in _algo_source, (
    "ALPHAEDIT_CACHE_ANCHOR not found in AlphaEdit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_cache_replacement = {repr(alphaedit_cache_replacement)}
_algo_source = _algo_source.replace(_cache_anchor, _cache_replacement, 1)
"""
    else:
        algo_file = "memit/memit_main.py"
        algo_module_name = "memit.memit_main"
        import_fixes = """
_algo_source = _algo_source.replace("from .compute_ks", "from memit.compute_ks")
_algo_source = _algo_source.replace("from .compute_z", "from memit.compute_z")
_algo_source = _algo_source.replace("from .memit_hparams", "from memit.memit_hparams")
"""
        import_anchor = MEMIT_IMPORT_ANCHOR
        import_replacement = "# apply_memit_to_model patched by polykernel_editor_runner"
        apply_fn_name = "apply_memit_to_model"
        extra_fn_name = "get_context_templates"
        extra_globals = '"get_context_templates": _patched_extra_fn,'
        solve_anchor_repr = repr(MEMIT_SOLVE_ANCHOR)
        solve_replacement_repr = repr(memit_solve_replacement)
        cache_patch_code = ""  # MEMIT has no cache_c

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

# 3. Kernel editor parameters (shared state)
_polykernel_degree = {kernel_degree}
_polykernel_type = "{kernel_type}"
_polykernel_sigma = "{kernel_sigma}"
_polykernel_batch_idx = 0
_polykernel_log = []
_polykernel_output_jsonl = "{output_jsonl}"
_pk_edit_only = {edit_only}
_pk_save_interval = {save_interval}
_pk_checkpoint_dir = "{checkpoint_dir}"
_pk_eval_only = {eval_only}
_pk_checkpoint_load_path = "{load_checkpoint}"

def _pk_save_checkpoint(cnt, model, cache_c, hparams, alg_name):
    \"\"\"Save model weights + cache_c at checkpoint boundary.\"\"\"
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime as _dt, timezone as _tz

    batch_dir = _Path(_pk_checkpoint_dir) / f"batch_{{cnt}}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save only edited layer weights
    layer_weights = {{}}
    for layer_idx in hparams.layers:
        param_name = f"model.layers.{{layer_idx}}.mlp.down_proj.weight"
        param = dict(model.named_parameters()).get(param_name)
        if param is not None:
            layer_weights[param_name] = param.data.cpu()

    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    # Save cache_c (AlphaEdit only)
    if alg_name == "AlphaEdit" and cache_c is not None:
        torch.save(cache_c.cpu(), str(batch_dir / "cache_c.pt"))

    metadata = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * {num_edits},
        "alg_name": alg_name,
        "seed": {seed},
        "kernel_type": _polykernel_type,
        "kernel_degree": _polykernel_degree,
        "timestamp_utc": _dt.now(_tz.utc).isoformat(),
    }}
    with open(str(batch_dir / "metadata.json"), "w") as f:
        _json.dump(metadata, f, indent=2)

    print(f"  [PK-CKPT] Saved batch {{cnt}} ({{(cnt+1) * {num_edits}}} edits) -> {{batch_dir}}")

def _pk_load_from_checkpoint(model, hparams, alg_name):
    \"\"\"Load model weights from checkpoint for eval-only mode.\"\"\"
    from pathlib import Path as _Path

    ckpt_path = _Path(_pk_checkpoint_load_path)
    if not ckpt_path.exists():
        print(f"  [PK-CKPT] ERROR: Checkpoint not found at {{ckpt_path}}")
        sys.exit(1)

    weights_file = ckpt_path / "model_weights.pt"
    if weights_file.exists():
        layer_weights = torch.load(str(weights_file), map_location="cuda")
        param_dict = dict(model.named_parameters())
        loaded = 0
        for pname, pdata in layer_weights.items():
            if pname in param_dict:
                param_dict[pname].data.copy_(pdata)
                loaded += 1
        print(f"  [PK-CKPT] Loaded {{loaded}} param tensors from {{weights_file}}")

    cache_c_loaded = None
    if alg_name == "AlphaEdit":
        cache_file = ckpt_path / "cache_c.pt"
        if cache_file.exists():
            cache_c_loaded = torch.load(str(cache_file), map_location="cpu")
            print(f"  [PK-CKPT] Loaded cache_c (shape: {{cache_c_loaded.shape}})")

    return cache_c_loaded

# 4. Read and patch {algo_file}
with open("{algo_file}", "r") as f:
    _algo_source = f.read()

# Fix relative imports for standalone exec
{import_fixes}

# Inject kernel-weighted solve (replace original solve)
_solve_anchor = {solve_anchor_repr}
assert _solve_anchor in _algo_source, (
    "SOLVE_ANCHOR not found in {algo_file}. "
    "Upstream code has changed from pinned commit b84624f."
)
_solve_replacement = {solve_replacement_repr}
_algo_source = _algo_source.replace(_solve_anchor, _solve_replacement, 1)
{cache_patch_code}
# Verify injection
assert "KERNEL EDITOR: kernel-weighted solve" in _algo_source, "Solve injection failed"

# 5. Compile and exec patched algo
_algo_ns = {{
    "__name__": "{algo_module_name}",
    "__file__": "{algo_file}",
    "_polykernel_degree": _polykernel_degree,
    "_polykernel_type": _polykernel_type,
    "_polykernel_sigma": _polykernel_sigma,
    "_polykernel_batch_idx": _polykernel_batch_idx,
    "_polykernel_log": _polykernel_log,
}}
exec(compile(_algo_source, "{algo_file}", "exec"), _algo_ns)
_patched_apply = _algo_ns["{apply_fn_name}"]
_patched_extra_fn = _algo_ns["{extra_fn_name}"]

print("[Kernel Editor] {algo_file} patched successfully")
print(f"  kernel_type={{_polykernel_type}}, degree={{_polykernel_degree}}, sigma={{_polykernel_sigma}}")

# 6. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    _eval_source = f.read()

# Replace algo import (we provide it via exec globals)
_import_anchor = {repr(import_anchor)}
assert _import_anchor in _eval_source, (
    "Import anchor not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_eval_source = _eval_source.replace(
    _import_anchor,
    "{import_replacement}",
)

# Patch CUDA
_cuda_target = {repr(CUDA_PATCH_TARGET)}
assert _cuda_target in _eval_source, "CUDA patch target not found in evaluate.py."
_eval_source = _eval_source.replace(
    _cuda_target,
    "# CUDA_VISIBLE_DEVICES managed by polykernel_editor_runner",
)

# Inject batch increment before POST_EDIT_ANCHOR
_post_anchor = {repr(POST_EDIT_ANCHOR)}
assert _post_anchor in _eval_source, "POST_EDIT_ANCHOR not found in evaluate.py."
_batch_hook = {repr(batch_increment_hook)}
_eval_source = _eval_source.replace(_post_anchor, _batch_hook + _post_anchor, 1)

# --- EDIT-ONLY MODE: inject checkpoint save + skip eval ---
if _pk_edit_only:
    # Inject checkpoint save after exec_time (after edit completes)
    _exec_anchor = {repr(EXEC_TIME_ANCHOR)}
    assert _exec_anchor in _eval_source, "EXEC_TIME_ANCHOR not found in evaluate.py."
    _save_hook = '''        # === PK-CKPT: save checkpoint at interval (injected) ===
        if _pk_edit_only and (cnt + 1) % _pk_save_interval == 0:
            _pk_save_checkpoint(cnt, model, cache_c if alg_name == "AlphaEdit" else None, hparams, alg_name)
        # === END PK-CKPT save ===
'''
    _eval_source = _eval_source.replace(_exec_anchor, _save_hook + _exec_anchor, 1)

    # Inject skip-eval: replace the evaluation loop to just break immediately
    _eval_start = {repr(EVAL_START_ANCHOR)}
    assert _eval_start in _eval_source, "EVAL_START_ANCHOR not found in evaluate.py."
    _skip_eval_hook = '''    # torch.save(hs, "post_edit_hs_memit.pt")
    # === PK-CKPT: skip ALL evaluation in edit-only mode (injected) ===
    if _pk_edit_only:
        print(f"  [PK-CKPT] Edit-only mode: skipping evaluation. All {{cnt}} batches edited.")
        print(f"  [PK-CKPT] Final checkpoint at: {{_pk_checkpoint_dir}}")
    # === END skip eval ===
    start = time()'''
    _eval_source = _eval_source.replace(_eval_start, _skip_eval_hook, 1)

    _eval_loop = {repr(EVAL_LOOP_ANCHOR)}
    assert _eval_loop in _eval_source, "EVAL_LOOP_ANCHOR not found in evaluate.py."
    _eval_loop_skip = '''    for record in ds:
        # === PK-CKPT: skip evaluation in edit-only mode (injected) ===
        if _pk_edit_only:
            break
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'''
    _eval_source = _eval_source.replace(_eval_loop, _eval_loop_skip, 1)

# --- EVAL-ONLY MODE: inject model load + skip editing ---
if _pk_eval_only:
    # Inject model load before the main loop
    _loop_anchor = {repr(LOOP_ANCHOR)}
    assert _loop_anchor in _eval_source, "LOOP_ANCHOR not found in evaluate.py."
    _load_hook = '''    # === PK-CKPT: load checkpoint for eval-only mode (injected) ===
    if _pk_eval_only:
        _loaded_cache = _pk_load_from_checkpoint(model, hparams, alg_name)
        if _loaded_cache is not None and alg_name == "AlphaEdit":
            cache_c = _loaded_cache
        print(f"  [PK-CKPT] Eval-only mode: model loaded from checkpoint, skipping all edits.")
    exec_time = 0  # Prevent UnboundLocalError in eval loop (no edits performed)
    edited_model = model  # Model already has edits from checkpoint
    # === END PK-CKPT load ===
'''
    _eval_source = _eval_source.replace(_loop_anchor, _load_hook + _loop_anchor, 1)

    # Inject skip guard before each batch's edit call
    _pre_anchor = {repr(PRE_EDIT_ANCHOR)}
    assert _pre_anchor in _eval_source, "PRE_EDIT_ANCHOR not found in evaluate.py."
    _skip_edit_hook = '''        # === PK-CKPT: skip editing in eval-only mode (injected) ===
        if _pk_eval_only:
            cnt += 1
            continue
        # === END PK-CKPT skip edit ===
'''
    _eval_source = _eval_source.replace(_pre_anchor, _skip_edit_hook + _pre_anchor, 1)

print("[Kernel Editor] evaluate.py patched successfully")

# 7. Execute patched evaluate.py
exec(compile(_eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "{apply_fn_name}": _patched_apply,
    {extra_globals}
    "_polykernel_degree": _polykernel_degree,
    "_polykernel_type": _polykernel_type,
    "_polykernel_sigma": _polykernel_sigma,
    "_polykernel_batch_idx": _polykernel_batch_idx,
    "_polykernel_log": _polykernel_log,
    "_pk_edit_only": _pk_edit_only,
    "_pk_save_interval": _pk_save_interval,
    "_pk_checkpoint_dir": _pk_checkpoint_dir,
    "_pk_eval_only": _pk_eval_only,
    "_pk_save_checkpoint": _pk_save_checkpoint,
    "_pk_load_from_checkpoint": _pk_load_from_checkpoint,
    "_pk_checkpoint_load_path": _pk_checkpoint_load_path,
}})

# 8. Write log to JSONL
with open(_polykernel_output_jsonl, "w") as f:
    for entry in _polykernel_log:
        f.write(json.dumps(entry) + "\\n")

print(f"\\n[PolyKernel Editor] Log written: {{_polykernel_output_jsonl}} ({{len(_polykernel_log)}} entries)")
""")
    return script


def validate_anchors(alg_name: str) -> None:
    """Verify all source anchors exist in the pinned code."""
    alphaedit_root = get_alphaedit_root()

    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    if alg_name == "AlphaEdit":
        assert ALGO_IMPORT_ANCHOR in eval_source, "ALGO_IMPORT_ANCHOR not found in evaluate.py"
        algo_source = (alphaedit_root / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert ALPHAEDIT_SOLVE_ANCHOR in algo_source, "ALPHAEDIT_SOLVE_ANCHOR not found in AlphaEdit_main.py"
        assert ALPHAEDIT_CACHE_ANCHOR in algo_source, "ALPHAEDIT_CACHE_ANCHOR not found in AlphaEdit_main.py"
    else:
        assert MEMIT_IMPORT_ANCHOR in eval_source, "MEMIT_IMPORT_ANCHOR not found in evaluate.py"
        algo_source = (alphaedit_root / "memit" / "memit_main.py").read_text()
        assert MEMIT_SOLVE_ANCHOR in algo_source, "MEMIT_SOLVE_ANCHOR not found in memit_main.py"

    print("  All source anchors validated.")


def run(args: argparse.Namespace) -> None:
    """Launch poly-kernel editor experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    model_name = resolve_model_path(args.model_name)

    print("Validating source anchors...")
    validate_anchors(args.alg_name)

    # Output directory
    results_dir = project_root / "results" / "polykernel_editor"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    kernel_tag = f"{args.kernel_type}{args.kernel_degree}" if args.kernel_type == "poly" else f"rbf_{args.kernel_sigma}"
    output_jsonl = results_dir / f"log_{args.alg_name}_seed{args.seed}_{kernel_tag}_{timestamp}.jsonl"

    # Resolve checkpoint dir for edit_only mode
    checkpoint_dir = ""
    if args.edit_only:
        if args.checkpoint_dir:
            checkpoint_dir = args.checkpoint_dir
        elif Path("/s3-data/continual-learning/alphaedit/checkpoints").exists():
            checkpoint_dir = str(Path("/s3-data/continual-learning/alphaedit/checkpoints") / f"{args.alg_name}_poly{args.kernel_degree}" / f"seed{args.seed}")
        else:
            checkpoint_dir = str(Path.home() / ".cache" / "alphaedit_checkpoints" / f"{args.alg_name}_poly{args.kernel_degree}" / f"seed{args.seed}")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    script = build_editor_script(
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
        kernel_degree=args.kernel_degree,
        kernel_type=args.kernel_type,
        kernel_sigma=args.kernel_sigma,
        output_jsonl=str(output_jsonl),
        edit_only=args.edit_only,
        save_interval=args.save_interval,
        checkpoint_dir=checkpoint_dir,
        eval_only=args.eval_only,
        load_checkpoint=args.load_checkpoint or "",
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
        print("WARNING: HF_TOKEN not set. Model download may fail.")

    mode = "edit_only" if args.edit_only else ("eval_only" if args.eval_only else "full")
    print(f"\n{'=' * 70}")
    print("Kernel Editor")
    print(f"  Mode:           {mode}")
    print(f"  Seed:           {args.seed}")
    print(f"  Algorithm:      {args.alg_name}")
    print(f"  Kernel type:    {args.kernel_type}")
    print(f"  Kernel degree:  {args.kernel_degree}" if args.kernel_type == "poly" else f"  Kernel sigma:   {args.kernel_sigma}")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:      {args.num_edits}")
    if args.edit_only:
        print(f"  Save interval:  every {args.save_interval} batches ({args.save_interval * args.num_edits} edits)")
        print(f"  Checkpoint dir: {checkpoint_dir}")
    elif args.eval_only:
        print(f"  Load from:      {args.load_checkpoint}")
    else:
        print(f"  Eval steps:     {args.downstream_eval_steps}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "experiment": "polykernel_editor",
        "seed": args.seed,
        "alg_name": args.alg_name,
        "kernel_type": args.kernel_type,
        "kernel_degree": args.kernel_degree,
        "kernel_sigma": args.kernel_sigma,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "ds_name": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "downstream_eval_steps": args.downstream_eval_steps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "output_jsonl": str(output_jsonl),
    }
    meta_path = results_dir / f"metadata_{args.alg_name}_seed{args.seed}_{kernel_tag}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Launch
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(alphaedit_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Experiment failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Kernel Editor completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Log:       {output_jsonl}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Kernel editor: kernel-augmented MEMIT/AlphaEdit (poly or RBF)"
    )

    # Seed and hardware
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    # Model and data
    parser.add_argument("--alg_name", choices=["AlphaEdit", "MEMIT"], default="AlphaEdit")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    # Kernel parameters
    parser.add_argument("--kernel_type", choices=["poly", "rbf"], default="poly",
                        help="Kernel type: 'poly' for polynomial, 'rbf' for Gaussian RBF.")
    parser.add_argument("--kernel_degree", type=int, default=2,
                        help="Polynomial kernel degree (default: 2). Only used with --kernel_type poly.")
    parser.add_argument("--kernel_sigma", default="median",
                        help="RBF bandwidth: 'median' for median heuristic, or a float value. Only used with --kernel_type rbf.")

    # Checkpoint / long-run modes
    parser.add_argument("--edit_only", action="store_true",
                        help="Edit-only mode: apply all edits, save checkpoints, skip evaluation.")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N batches (default: 10 = every 1000 edits).")
    parser.add_argument("--checkpoint_dir", default="",
                        help="Directory for saving/loading checkpoints. Auto-resolved if not specified.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Eval-only mode: load checkpoint, skip editing, run full evaluation.")
    parser.add_argument("--load_checkpoint", default="",
                        help="Path to checkpoint directory to load (e.g., .../batch_99). Required with --eval_only.")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
