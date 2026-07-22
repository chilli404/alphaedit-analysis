#!/usr/bin/env python3
"""
Polynomial-Kernel Memory Diagnostic — Stage 1: Key Extraction

Extracts the raw edit keys (layer activations at subject token positions) from
AlphaEdit or MEMIT during the editing process. Keys are saved as .pt files for
offline analysis by polykernel_diagnostic.py (Stage 2).

This tests whether AlphaEdit/MEMIT failure modes are consistent with a linear
key-space capacity bottleneck — i.e., whether a degree-2 polynomial kernel
would create more linearly independent key directions.

Architecture (dual source injection, following coupling_stress_runner.py):
  1. Read AlphaEdit_main.py (or memit_main.py), fix relative imports, inject
     key-save hook after compute_ks
  2. Compile/exec patched algo → extract apply_*_to_model
  3. Read evaluate.py, replace algo import, inject batch-metadata tracking
  4. Exec evaluate.py with patched function in globals

Usage:
    python src/polykernel_key_extractor.py \
        --seed 42 --alg_name AlphaEdit \
        --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \
        [--coupling_dataset path/to/coupling_dataset_seed42.json]
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
from paths import get_project_root, get_alphaedit_root, get_result_root


# --- Source anchors (commit b84624f) ---

# evaluate.py anchors
CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'
MEMIT_IMPORT_ANCHOR = 'from memit.memit_main import apply_memit_to_model, get_context_templates'

# AlphaEdit_main.py key anchor (line 111 — first compute_ks in the update loop)
ALPHAEDIT_KEYS_ANCHOR = '        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T'

# memit_main.py key anchor (line 156 — compute_ks in execute_memit)
MEMIT_KEYS_ANCHOR = '        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T'


def build_extraction_script(
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
    coupling_dataset_path: str | None,
    output_pt: str,
    eval_results_dir: str = "",
) -> str:
    """
    Build inline Python script for key extraction.
    Uses dual source injection: patches algo file and evaluate.py.
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

    # Key-save hook injected AFTER the layer_ks = compute_ks line in algo file
    key_save_hook = r'''
        # === POLYKERNEL: save keys (injected) ===
        if '_polykernel_keys' in globals():
            _polykernel_keys.setdefault(int(layer), []).append(
                layer_ks.detach().cpu().to(torch.float16)
            )
        # === END polykernel save ===
'''

    # Batch metadata hook injected before PRE_EDIT_ANCHOR in evaluate.py
    batch_metadata_hook = r'''        # === POLYKERNEL: batch metadata (injected) ===
        if '_polykernel_batch_tags' in globals():
            _polykernel_batch_tags.append({
                "batch_idx": cnt,
                "case_ids": [r["case_id"] for r in record_chunks],
                "coupling_meta": [r.get("coupling_metadata", {}) for r in record_chunks],
            })
        # === END polykernel batch metadata ===
'''

    # Dataset override for coupling mode
    dataset_override = ""
    if coupling_dataset_path:
        dataset_override = f'''
    # === POLYKERNEL: coupling dataset override (injected) ===
    import json as _json_loader
    with open("{coupling_dataset_path}", "r") as _cdf:
        _coupling_data = _json_loader.load(_cdf)
    ds.data = _coupling_data
    print(f"  [Polykernel] Loaded {{len(_coupling_data)}} records from coupling dataset")
    # === END coupling dataset override ===
'''

    # Build the script based on algorithm
    if alg_name == "AlphaEdit":
        algo_file = "AlphaEdit/AlphaEdit_main.py"
        algo_module_name = "AlphaEdit.AlphaEdit_main"
        import_fixes = """
_algo_source = _algo_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
_algo_source = _algo_source.replace("from .compute_z", "from AlphaEdit.compute_z")
_algo_source = _algo_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")
"""
        import_anchor = ALGO_IMPORT_ANCHOR
        import_replacement = "# apply_AlphaEdit_to_model patched by polykernel_key_extractor"
        apply_fn_name = "apply_AlphaEdit_to_model"
        extra_fn_name = "get_cov"
        extra_globals = '"get_cov": _patched_extra_fn,'
        algo_ns_extras = ""
    else:
        algo_file = "memit/memit_main.py"
        algo_module_name = "memit.memit_main"
        import_fixes = """
_algo_source = _algo_source.replace("from .compute_ks", "from memit.compute_ks")
_algo_source = _algo_source.replace("from .compute_z", "from memit.compute_z")
_algo_source = _algo_source.replace("from .memit_hparams", "from memit.memit_hparams")
"""
        import_anchor = MEMIT_IMPORT_ANCHOR
        import_replacement = "# apply_memit_to_model patched by polykernel_key_extractor"
        apply_fn_name = "apply_memit_to_model"
        extra_fn_name = "get_context_templates"
        extra_globals = '"get_context_templates": _patched_extra_fn,'
        algo_ns_extras = ""

    # The keys anchor is the same text in both files
    keys_anchor = ALPHAEDIT_KEYS_ANCHOR

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

# 3. Shared state for key extraction
_polykernel_keys = {{}}   # {{layer_int: [Tensor(d_model, n_keys), ...]}}
_polykernel_batch_tags = []

# 4. Read and patch {algo_file}
with open("{algo_file}", "r") as f:
    _algo_source = f.read()

# Fix relative imports
{import_fixes}

# Inject key-save hook after compute_ks
_keys_anchor = {repr(keys_anchor)}
assert _keys_anchor in _algo_source, (
    "KEYS_ANCHOR not found in {algo_file}. "
    "Upstream code has changed from pinned commit b84624f."
)
_key_hook = {repr(key_save_hook)}
_algo_source = _algo_source.replace(_keys_anchor, _keys_anchor + _key_hook, 1)

# Compile and exec patched algo
_algo_ns = {{
    "__name__": "{algo_module_name}",
    "__file__": "{algo_file}",
    "_polykernel_keys": _polykernel_keys,
}}
exec(compile(_algo_source, "{algo_file}", "exec"), _algo_ns)
_patched_apply = _algo_ns["{apply_fn_name}"]
_patched_extra_fn = _algo_ns["{extra_fn_name}"]

print("[Polykernel] {algo_file} patched successfully")

# 5. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    _eval_source = f.read()

# Replace algo import
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
    "# CUDA_VISIBLE_DEVICES managed by polykernel_key_extractor",
)

# Override RESULTS_DIR to project-level results
_globals_import = 'from util.globals import *'
assert _globals_import in _eval_source, "globals import not found in evaluate.py"
_eval_source = _eval_source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)
print(f"  [RESULTS_DIR] Overridden to: {eval_results_dir}")

# Inject coupling dataset override before for loop (if applicable)
_shuffle_anchor = {repr(SHUFFLE_ANCHOR)}
assert _shuffle_anchor in _eval_source, "SHUFFLE_ANCHOR not found in evaluate.py."
_dataset_override = {repr(dataset_override)}
if _dataset_override.strip():
    _eval_source = _eval_source.replace(_shuffle_anchor, _dataset_override + "\\n" + _shuffle_anchor, 1)

# Inject batch metadata hook before PRE_EDIT_ANCHOR
_pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert _pre_anchor in _eval_source, "PRE_EDIT_ANCHOR not found in evaluate.py."
_batch_hook = {repr(batch_metadata_hook)}
_eval_source = _eval_source.replace(_pre_anchor, _batch_hook + _pre_anchor, 1)

print("[Polykernel] evaluate.py patched successfully")

# 6. Execute patched evaluate.py
exec(compile(_eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "{apply_fn_name}": _patched_apply,
    {extra_globals}
    "_polykernel_keys": _polykernel_keys,
    "_polykernel_batch_tags": _polykernel_batch_tags,
}})

# 7. Save extracted keys
print(f"\\n[Polykernel] Saving keys...")
_output = {{
    "keys": {{}},
    "batch_tags": _polykernel_batch_tags,
    "metadata": {{
        "seed": {seed},
        "alg_name": "{alg_name}",
        "model_name": "{model_name}",
        "num_edits": {num_edits},
        "dataset_size_limit": {dataset_size_limit},
        "ds_name": "{ds_name}",
        "coupling_dataset": {repr(coupling_dataset_path)},
        "timestamp_utc": "{datetime.now(timezone.utc).isoformat()}",
        "alphaedit_commit": "b84624f",
    }},
}}

# Concatenate all key tensors per layer
for _layer, _key_list in _polykernel_keys.items():
    if _key_list:
        _output["keys"][_layer] = torch.cat(_key_list, dim=1)
        print(f"  Layer {{_layer}}: {{_output['keys'][_layer].shape[1]}} keys, shape {{_output['keys'][_layer].shape}}")

torch.save(_output, "{output_pt}")
print(f"[Polykernel] Saved to {output_pt}")
print(f"  Total layers: {{len(_output['keys'])}}")
print(f"  Total batches: {{len(_polykernel_batch_tags)}}")
""")
    return script


def validate_anchors(alg_name: str) -> None:
    """Verify that all source anchors exist in the pinned code."""
    alphaedit_root = get_alphaedit_root()

    # Check evaluate.py
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("SHUFFLE_ANCHOR", SHUFFLE_ANCHOR),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    if alg_name == "AlphaEdit":
        assert ALGO_IMPORT_ANCHOR in eval_source, "ALGO_IMPORT_ANCHOR not found in evaluate.py"
        algo_source = (alphaedit_root / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert ALPHAEDIT_KEYS_ANCHOR in algo_source, "ALPHAEDIT_KEYS_ANCHOR not found in AlphaEdit_main.py"
    else:
        assert MEMIT_IMPORT_ANCHOR in eval_source, "MEMIT_IMPORT_ANCHOR not found in evaluate.py"
        algo_source = (alphaedit_root / "memit" / "memit_main.py").read_text()
        assert MEMIT_KEYS_ANCHOR in algo_source, "MEMIT_KEYS_ANCHOR not found in memit_main.py"

    print("  All source anchors validated.")


def run(args: argparse.Namespace) -> None:
    """Run key extraction experiment."""
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
    results_dir = (
        get_result_root() / "polykernel_diagnostic"
        / f"seed{args.seed}" / f"{args.dataset_size_limit}edits" / args.alg_name
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    output_pt = results_dir / f"keys_{args.alg_name}_seed{args.seed}.pt"

    # Resolve coupling dataset path
    coupling_dataset_path = None
    if args.coupling_dataset:
        coupling_dataset_path = str(Path(args.coupling_dataset).resolve())
        if not Path(coupling_dataset_path).exists():
            print(f"ERROR: Coupling dataset not found: {coupling_dataset_path}")
            sys.exit(1)

    script = build_extraction_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name=args.alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=0,
        conserve_memory=args.conserve_memory,
        coupling_dataset_path=coupling_dataset_path,
        output_pt=str(output_pt),
        eval_results_dir=str(results_dir.parent),
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    if "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env:
        print("WARNING: HF_TOKEN not set. Model download may fail.")

    print(f"\n{'=' * 70}")
    print("Polynomial-Kernel Key Extractor")
    print(f"  Seed:           {args.seed}")
    print(f"  Algorithm:      {args.alg_name}")
    print(f"  CUDA:           device {args.cuda_device}")
    print(f"  Model:          {args.model_name}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:      {args.num_edits}")
    if coupling_dataset_path:
        print(f"  Coupling data:  {coupling_dataset_path}")
    print(f"  Output:         {output_pt}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Save metadata
    metadata = {
        "experiment": "polykernel_key_extraction",
        "seed": args.seed,
        "alg_name": args.alg_name,
        "model_name": args.model_name,
        "hparams_fname": args.hparams_fname,
        "ds_name": args.ds_name,
        "dataset_size_limit": args.dataset_size_limit,
        "num_edits": args.num_edits,
        "coupling_dataset": coupling_dataset_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alphaedit_commit": "b84624f",
        "output_pt": str(output_pt),
    }
    meta_path = results_dir / f"metadata_keys_{args.alg_name}_seed{args.seed}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Launch
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(alphaedit_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Key extraction failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Key extraction completed.")
    print(f"  Finished:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Keys:      {output_pt}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Polynomial-kernel diagnostic: extract edit keys from AlphaEdit/MEMIT"
    )

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", choices=["AlphaEdit", "MEMIT"], default="AlphaEdit")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--coupling_dataset", type=str, default=None,
                        help="Path to coupling dataset JSON (enables coupling mode)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
