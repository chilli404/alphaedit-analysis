#!/usr/bin/env python3
"""
Coupling Stress Runner: Measures projection residuals inside AlphaEdit's
null-space constrained edit loop.

This runner uses double source injection to hook INSIDE apply_AlphaEdit_to_model's
per-layer loop, measuring how much of each edit's desired direction is removed
by the null-space projection. This is correlated with semantic coupling type
to test whether the editability/preservation separability assumption breaks down
for related edits.

Key measurement: projection_loss = 1 - ||P @ K @ resid^T|| / ||K @ resid^T||

Architecture:
  1. Read AlphaEdit_main.py, fix relative imports, inject measurement code
  2. Compile/exec patched algo into namespace → extract apply_AlphaEdit_to_model
  3. Read evaluate.py, replace algo import, inject dataset override + per-batch hooks
  4. Exec evaluate.py with patched function in globals

Usage:
    python src/coupling_stress_runner.py \\
        --seed 42 --cuda_device 0 \\
        --max_pairs_per_type 60 --warmup_count 20 --conserve_memory
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
from source_patches import patch_evaluate_file


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    return get_project_root() / "vendor" / "AlphaEdit"


# --- Source anchors (commit b84624f) ---

# evaluate.py anchors (same as other runners)
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'

# AlphaEdit_main.py anchors
RESID_ANCHOR = '        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers'
UPD_ANCHOR = '        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)'


def build_coupling_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    coupling_dataset_path: str,
    output_jsonl: str,
) -> str:
    """
    Build the inline Python script that:
    1. Seeds RNGs
    2. Patches AlphaEdit_main.py with measurement code (in-memory)
    3. Patches evaluate.py with dataset override + per-batch hooks
    4. Executes the combined patched code
    """
    argv_parts = [
        "experiments.evaluate",
        "--alg_name=AlphaEdit",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        "--num_edits=1",
        f"--downstream_eval_steps={downstream_eval_steps}",
        "--generation_test_interval=1",
        "--skip_generation_tests",
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)

    # The measurement code injected into AlphaEdit_main.py AFTER resid computation
    pre_solve_measurement = r'''
        # === COUPLING MEASUREMENT: pre-solve (injected) ===
        if '_coupling_measure' in globals():
            import torch as _torch_m
            _rhs = layer_ks @ resid.T
            _projected_rhs = P[i,:,:].cuda() @ _rhs
            _rhs_norm = _torch_m.linalg.norm(_rhs).item()
            _proj_norm = _torch_m.linalg.norm(_projected_rhs).item()
            _coupling_layer_data[str(layer)] = {
                "resid_norm": _rhs_norm,
                "projected_rhs_norm": _proj_norm,
                "projection_loss": 1.0 - (_proj_norm / max(_rhs_norm, 1e-10)),
            }
            del _rhs, _projected_rhs
        # === END coupling pre-solve ===
'''

    # The measurement code injected AFTER upd_matrix_match_shape
    post_solve_measurement = r'''
        # === COUPLING MEASUREMENT: post-solve (injected) ===
        if '_coupling_measure' in globals():
            import torch as _torch_m
            _coupling_layer_data[str(layer)]["upd_matrix_norm"] = _torch_m.linalg.norm(upd_matrix).item()
        # === END coupling post-solve ===
'''

    # The per-batch measurement hook injected at PRE_EDIT_ANCHOR in evaluate.py
    pre_batch_hook = r'''
        # === COUPLING: per-batch pre-hook (injected) ===
        if '_coupling_output' in globals() and alg_name == "AlphaEdit":
            _coupling_layer_data.clear()
        # === END coupling pre-hook ===
'''

    # The per-batch measurement hook injected at POST_EDIT_ANCHOR in evaluate.py
    post_batch_hook = r'''
        # === COUPLING: per-batch post-hook (injected) ===
        if '_coupling_output' in globals() and alg_name == "AlphaEdit":
            _current_record = record_chunks[0] if isinstance(record_chunks, list) else record_chunks
            _case_id = _current_record.get("case_id", cnt)
            _meta = _coupling_metadata.get(_case_id, {})

            _layer_agg = list(_coupling_layer_data.values())
            _losses = [d.get("projection_loss", 0) for d in _layer_agg]
            _upd_norms = [d.get("upd_matrix_norm", 0) for d in _layer_agg]

            _record = {
                "edit_idx": cnt,
                "case_id": _case_id,
                "coupling_type": _meta.get("coupling_type", -1),
                "coupling_type_name": _meta.get("coupling_type_name", "unknown"),
                "role": _meta.get("role", "unknown"),
                "pair_id": _meta.get("pair_id"),
                "layers": dict(_coupling_layer_data),
                "aggregate": {
                    "mean_projection_loss": sum(_losses) / max(len(_losses), 1),
                    "max_projection_loss": max(_losses) if _losses else 0,
                    "total_upd_norm": sum(_upd_norms),
                },
            }
            with open(_coupling_output, "a") as _f:
                _f.write(json.dumps(_record) + "\n")
            _coupling_layer_data.clear()
        # === END coupling post-hook ===
'''

    # Dataset override code (replaces ds.data with coupling dataset)
    dataset_override = f'''
    # === COUPLING: dataset override (injected) ===
    import json as _json_loader
    with open("{coupling_dataset_path}", "r") as _cdf:
        _coupling_data = _json_loader.load(_cdf)
    ds.data = _coupling_data
    _coupling_metadata = {{r["case_id"]: r.get("coupling_metadata", {{}}) for r in _coupling_data}}
    print(f"  [Coupling] Loaded {{len(_coupling_data)}} records from coupling dataset")
    # === END dataset override ===
'''

    script = textwrap.dedent(f"""\
import os, sys, random, json, math
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

# 2. Set sys.argv for evaluate.py
sys.argv = {argv_str}

# 3. Output path
_coupling_output = "{output_jsonl}"
_coupling_layer_data = {{}}
_coupling_metadata = {{}}
_coupling_measure = True

# 4. Read and patch AlphaEdit_main.py
with open("AlphaEdit/AlphaEdit_main.py", "r") as f:
    _algo_source = f.read()

# Fix relative imports for standalone exec
_algo_source = _algo_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
_algo_source = _algo_source.replace("from .compute_z", "from AlphaEdit.compute_z")
_algo_source = _algo_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")

# Inject measurement after resid computation
_resid_anchor = {repr(RESID_ANCHOR)}
assert _resid_anchor in _algo_source, (
    "RESID_ANCHOR not found in AlphaEdit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_pre_solve_code = {repr(pre_solve_measurement)}
_algo_source = _algo_source.replace(_resid_anchor, _resid_anchor + "\\n" + _pre_solve_code, 1)

# Inject measurement after upd_matrix_match_shape
_upd_anchor = {repr(UPD_ANCHOR)}
assert _upd_anchor in _algo_source, (
    "UPD_ANCHOR not found in AlphaEdit_main.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_post_solve_code = {repr(post_solve_measurement)}
_algo_source = _algo_source.replace(_upd_anchor, _upd_anchor + "\\n" + _post_solve_code, 1)

# Compile and exec patched algo
_algo_ns = {{
    "__name__": "AlphaEdit.AlphaEdit_main",
    "__file__": "AlphaEdit/AlphaEdit_main.py",
    "_coupling_measure": True,
    "_coupling_layer_data": _coupling_layer_data,
}}
exec(compile(_algo_source, "AlphaEdit/AlphaEdit_main.py", "exec"), _algo_ns)
_patched_apply = _algo_ns["apply_AlphaEdit_to_model"]
_patched_get_cov = _algo_ns["get_cov"]

print("[Coupling] AlphaEdit_main.py patched successfully")

# 5. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    _eval_source = f.read()

# Replace AlphaEdit import (we provide it via exec globals)
_import_anchor = {repr(ALGO_IMPORT_ANCHOR)}
assert _import_anchor in _eval_source, (
    "ALGO_IMPORT_ANCHOR not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
_eval_source = _eval_source.replace(
    _import_anchor,
    "# apply_AlphaEdit_to_model patched by coupling_stress_runner",
)

# Patch CUDA
_cuda_target = {repr(CUDA_PATCH_TARGET)}
assert _cuda_target in _eval_source, (
    "CUDA patch target not found in evaluate.py."
)
_eval_source = _eval_source.replace(
    _cuda_target,
    "# CUDA_VISIBLE_DEVICES managed by coupling_stress_runner",
)

# Inject dataset override BEFORE the for loop
_shuffle_anchor = {repr(SHUFFLE_ANCHOR)}
assert _shuffle_anchor in _eval_source, (
    "SHUFFLE_ANCHOR not found in evaluate.py."
)
_dataset_override = {repr(dataset_override)}
_eval_source = _eval_source.replace(_shuffle_anchor, _dataset_override + "\\n" + _shuffle_anchor, 1)

# Inject pre-batch hook
_pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert _pre_anchor in _eval_source, (
    "PRE_EDIT_ANCHOR not found in evaluate.py."
)
_pre_hook = {repr(pre_batch_hook)}
_eval_source = _eval_source.replace(_pre_anchor, _pre_hook + _pre_anchor, 1)

# Inject post-batch hook
_post_anchor = {repr(POST_EDIT_ANCHOR)}
assert _post_anchor in _eval_source, (
    "POST_EDIT_ANCHOR not found in evaluate.py."
)
_post_hook = {repr(post_batch_hook)}
_eval_source = _eval_source.replace(_post_anchor, _post_hook + _post_anchor, 1)

print("[Coupling] evaluate.py patched successfully")

# 6. Execute patched evaluate.py
exec(compile(_eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_AlphaEdit_to_model": _patched_apply,
    "get_cov": _patched_get_cov,
    "_coupling_output": _coupling_output,
    "_coupling_layer_data": _coupling_layer_data,
    "_coupling_metadata": _coupling_metadata,
    "_coupling_measure": True,
}})

print(f"[Coupling] Results written to {{_coupling_output}}")
""")
    return script


def validate_anchors() -> None:
    """Verify that all source anchors exist in the pinned code."""
    alphaedit_root = get_alphaedit_root()

    # Check evaluate.py
    eval_path = alphaedit_root / "experiments" / "evaluate.py"
    eval_source = eval_path.read_text()

    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("SHUFFLE_ANCHOR", SHUFFLE_ANCHOR),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("ALGO_IMPORT_ANCHOR", ALGO_IMPORT_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    # Check AlphaEdit_main.py
    algo_path = alphaedit_root / "AlphaEdit" / "AlphaEdit_main.py"
    algo_source = algo_path.read_text()

    for name, anchor in [
        ("RESID_ANCHOR", RESID_ANCHOR),
        ("UPD_ANCHOR", UPD_ANCHOR),
    ]:
        assert anchor in algo_source, f"{name} not found in AlphaEdit_main.py"

    print("  All source anchors validated.")


def run(args: argparse.Namespace) -> None:
    """Generate coupling dataset and run instrumented experiment."""
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

    # Validate all anchors before committing to expensive GPU run
    print("Validating source anchors...")
    validate_anchors()

    # Generate coupling dataset (or load from cache if already generated)
    results_dir = project_root / "results" / "coupling_stress"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = results_dir / f"coupling_dataset_seed{args.seed}.json"

    if dataset_path.exists():
        print(f"\nLoading cached coupling dataset from {dataset_path}...")
        with open(dataset_path, "r") as f:
            sequence = json.load(f)
        print(f"  Loaded {len(sequence)} records (cached)")
    else:
        print("\nGenerating coupling dataset...")
        from coupling_dataset import generate_coupling_dataset

        data_dir = alphaedit_root / "data"
        sequence = generate_coupling_dataset(
            data_dir=data_dir,
            seed=args.seed,
            max_pairs_per_type=args.max_pairs_per_type,
            warmup_count=args.warmup_count,
        )
        with open(dataset_path, "w") as f:
            json.dump(sequence, f)
        print(f"  Generated {len(sequence)} records → {dataset_path}")

    # Output JSONL path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = results_dir / f"coupling_trace_seed{args.seed}_{timestamp}.jsonl"

    # Build script
    script = build_coupling_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name="mcf",
        dataset_size_limit=len(sequence),
        downstream_eval_steps=0,
        conserve_memory=args.conserve_memory,
        coupling_dataset_path=str(dataset_path),
        output_jsonl=str(output_jsonl),
    )

    # Set up environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
        print("WARNING: HF_TOKEN not set. Model download may fail.")

    print(f"\n{'=' * 70}")
    print("Coupling Stress Test Runner")
    print(f"  Seed:           {args.seed}")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    print(f"  Pairs/type:     {args.max_pairs_per_type}")
    print(f"  Warmup:         {args.warmup_count}")
    print(f"  Total edits:    {len(sequence)}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "seed": args.seed,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "total_edits": len(sequence),
        "max_pairs_per_type": args.max_pairs_per_type,
        "warmup_count": args.warmup_count,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "output_jsonl": str(output_jsonl),
        "dataset_path": str(dataset_path),
    }
    meta_path = results_dir / f"metadata_seed{args.seed}.json"
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
    print("Coupling stress test completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Trace:     {output_jsonl}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Coupling stress test: measures projection residuals by coupling type"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--max_pairs_per_type", type=int, default=60)
    parser.add_argument("--warmup_count", type=int, default=20)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
