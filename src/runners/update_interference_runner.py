#!/usr/bin/env python3
"""
Phase 2: Fine-Grained Update Interference Runner

Records W_after - W_before for layer 6 at each batch (batches 10-49),
then computes per-key interference on all tracked keys.

Architecture: Dual source injection (same as alphaedit_stream_runner.py).
Patches AlphaEdit_main.py (minimal, no mechanism measurement) and evaluate.py
(delta capture + interference computation at each batch).

Key differences from alphaedit_stream_runner.py:
  - No cache eigenspectrum or projection loss measurement
  - Captures actual applied weight delta (W_after - W_before)
  - Computes interference on ALL 5000 tracked keys at each batch
  - Accumulates path interference per key (only for post-installation batches)
  - Saves per-batch delta norms + accumulated results

Output: JSON with per-batch and cumulative interference metrics.

Usage:
    python src/runners/update_interference_runner.py \
        --seed 42 --ordering key_clustered \
        --start_from_batch 10 --end_batch 49

    # Resume from existing checkpoint:
    python src/runners/update_interference_runner.py \
        --seed 42 --ordering key_dispersed \
        --start_from_batch 10 --end_batch 49
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
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))

from paths import get_alphaedit_root, get_checkpoint_root, get_result_root

ALPHAEDIT_ROOT = get_alphaedit_root()

# ─── Source Anchors (commit b84624f) ─────────────────────────────────────────

CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        exec_time = time() - start'
ALGO_IMPORT_ANCHOR = 'from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov'
CHECKPOINT_LOAD_ANCHOR = "    glue_save_location = str(run_dir) + '/' + 'glue_eval/'"

# AlphaEdit_main.py anchors (minimal patching for this runner)
CACHE_UPDATE_ANCHOR = '        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T'


def build_interference_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    dataset_path: str,
    ordering_name: str,
    stream_length: int,
    num_edits: int,
    start_batch: int,
    end_batch: int,
    checkpoint_dir: str,
    keys_path: str,
    output_path: str,
) -> str:
    """Build inline script with delta capture + interference computation."""

    argv_parts = [
        "experiments.evaluate",
        "--alg_name=AlphaEdit",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        "--ds_name=mcf",
        f"--dataset_size_limit={stream_length}",
        f"--num_edits={num_edits}",
        "--downstream_eval_steps=999",
        "--generation_test_interval=1",
        "--conserve_memory",
    ]
    argv_str = repr(argv_parts)

    # ─── evaluate.py injections ───────────────────────────────────────

    dataset_override = f'''
    # === INTERFERENCE_RUNNER: dataset override (injected) ===
    import json as _json_ir
    with open("{dataset_path}", "r") as _irf:
        _ir_stream_data = _json_ir.load(_irf)
    ds.data = _ir_stream_data
    print(f"  [IR] Loaded {{len(_ir_stream_data)}} records ({ordering_name} stream)")
    # === END dataset override ===
'''

    pre_batch_hook = f'''        # === INTERFERENCE_RUNNER: pre-batch (injected) ===
        if _ir_should_skip(cnt):
            cnt += 1
            continue
        if cnt > _ir_end_batch:
            print(f"  [IR] Reached end batch {{_ir_end_batch}}, stopping")
            break
        # Capture W_before for layer 6
        _ir_capture_w_before(model)
        # === END pre-batch ===
'''

    post_batch_hook = f'''        # === INTERFERENCE_RUNNER: post-batch (injected) ===
        _ir_compute_delta(cnt, model)
        if (cnt + 1) % _ir_save_interval == 0:
            _ir_save_checkpoint(cnt, model, cache_c if 'cache_c' in dir() else None, hparams)
        # === END post-batch ===
'''

    script = textwrap.dedent(f"""\
import os, sys, random, json, time
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

# ─── 2. Interference tracking state ─────────────────────────────────
_ir_ordering_name = "{ordering_name}"
_ir_checkpoint_dir = Path("{checkpoint_dir}")
_ir_output_path = Path("{output_path}")
_ir_start_batch = {start_batch}
_ir_end_batch = {end_batch}
_ir_num_edits = {num_edits}
_ir_save_interval = 10
_ir_layer = 6
_ir_weight_key = "model.layers.6.mlp.down_proj.weight"

# Load tracked keys
print("[IR] Loading tracked keys...")
_ir_keys_data = np.load("{keys_path}")
_ir_keys = torch.from_numpy(_ir_keys_data["keys"]).float()  # [5000, 14336]
_ir_case_ids = _ir_keys_data["case_ids"]  # [5000]
print(f"  Keys: {{_ir_keys.shape}}, case_ids: {{_ir_case_ids.shape}}")

# Load ordering to determine installation batches
with open("{dataset_path}", "r") as _f:
    _ir_ordering_records = json.load(_f)
_ir_ordering_case_ids = [r["case_id"] for r in _ir_ordering_records]

# Map case_id -> key index
_ir_cid_to_kidx = {{int(cid): i for i, cid in enumerate(_ir_case_ids)}}

# Map key_index -> installation batch
_ir_installation_batch = np.full(5000, -1, dtype=np.int32)
for pos, cid in enumerate(_ir_ordering_case_ids):
    kidx = _ir_cid_to_kidx.get(cid, -1)
    if kidx >= 0:
        _ir_installation_batch[kidx] = pos // _ir_num_edits

# Accumulators
_ir_path_sum = np.zeros(5000, dtype=np.float64)  # Accumulated path interference
_ir_path_fro = np.zeros(5000, dtype=np.float64)  # Frobenius-normalized
_ir_cum_delta = torch.zeros(4096, 14336, dtype=torch.float32)  # Cumulative delta
_ir_W_before = None  # Captured before each batch
_ir_batch_results = []  # Per-batch records

def _ir_should_skip(cnt):
    return cnt < _ir_start_batch

def _ir_capture_w_before(model):
    global _ir_W_before
    param_dict = dict(model.named_parameters())
    _ir_W_before = param_dict[_ir_weight_key].detach().cpu().float().clone()

def _ir_compute_delta(cnt, model):
    global _ir_W_before, _ir_cum_delta
    param_dict = dict(model.named_parameters())
    W_after = param_dict[_ir_weight_key].detach().cpu().float()
    delta = W_after - _ir_W_before  # [4096, 14336]

    delta_fro = torch.linalg.norm(delta).item()

    # Compute interference on all keys: delta @ K.T -> [4096, 5000]
    effects = delta @ _ir_keys.T
    norms = torch.linalg.norm(effects, dim=0).numpy()  # [5000]

    # Key norms for Frobenius normalization
    key_norms = torch.linalg.norm(_ir_keys, dim=1).numpy()  # [5000]
    I_fro = norms / (delta_fro * key_norms + 1e-10)

    # Accumulate only for keys installed BEFORE this batch
    eligible = _ir_installation_batch < cnt  # boolean mask [5000]
    _ir_path_sum[eligible] += norms[eligible]
    _ir_path_fro[eligible] += I_fro[eligible]

    # Cumulative delta (for net displacement computation)
    _ir_cum_delta += delta

    # Record this batch
    n_eligible = int(eligible.sum())
    eligible_norms = norms[eligible]
    batch_record = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _ir_num_edits,
        "delta_fro": round(float(delta_fro), 8),
        "n_eligible_keys": n_eligible,
        "mean_interference": round(float(eligible_norms.mean()), 8) if n_eligible > 0 else 0.0,
        "median_interference": round(float(np.median(eligible_norms)), 8) if n_eligible > 0 else 0.0,
        "max_interference": round(float(eligible_norms.max()), 8) if n_eligible > 0 else 0.0,
        "mean_I_fro": round(float(I_fro[eligible].mean()), 8) if n_eligible > 0 else 0.0,
    }}
    _ir_batch_results.append(batch_record)

    if (cnt + 1) % 5 == 0 or cnt == _ir_end_batch:
        print(f"  [IR] Batch {{cnt}}: ||dW||_F={{delta_fro:.6f}}, "
              f"eligible={{n_eligible}}, mean_interf={{batch_record['mean_interference']:.6f}}")

    # Free memory
    del effects, W_after, delta
    _ir_W_before = None

def _ir_save_checkpoint(cnt, model, cache_c, hparams):
    ckpt_dir = _ir_checkpoint_dir / f"batch_{{cnt}}"
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
    metadata = {{"batch_idx": cnt, "total_edits": (cnt + 1) * _ir_num_edits,
                 "stream": _ir_ordering_name, "seed": seed,
                 "timestamp": datetime.now(timezone.utc).isoformat()}}
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"  [IR] Checkpoint saved: {{ckpt_dir}}")

# ─── 3. Patch AlphaEdit_main.py (minimal) ─────────────────────────
alphaedit_source_path = Path("AlphaEdit/AlphaEdit_main.py")
ae_source = alphaedit_source_path.read_text()

# Fix relative imports
ae_source = ae_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
ae_source = ae_source.replace("from .compute_z", "from AlphaEdit.compute_z")
ae_source = ae_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")

# Compile (no additional injections needed for interference runner)
ae_code = compile(ae_source, "AlphaEdit/AlphaEdit_main.py", "exec")
ae_namespace = {{
    "__name__": "AlphaEdit.AlphaEdit_main",
    "__file__": "AlphaEdit/AlphaEdit_main.py",
}}
exec(ae_code, ae_namespace)
_patched_apply = ae_namespace["apply_AlphaEdit_to_model"]
_patched_get_cov = ae_namespace["get_cov"]
print("[IR] AlphaEdit_main.py compiled (no injections)")

# ─── 4. Patch evaluate.py ───────────────────────────────────────────
with open("experiments/evaluate.py", "r") as f:
    eval_source = f.read()

# Replace AlphaEdit import
import_anchor = {repr(ALGO_IMPORT_ANCHOR)}
assert import_anchor in eval_source, "ALGO_IMPORT_ANCHOR not found"
eval_source = eval_source.replace(
    import_anchor,
    "# apply_AlphaEdit_to_model patched by update_interference_runner",
)

# Patch CUDA
cuda_target = {repr(CUDA_PATCH_TARGET)}
assert cuda_target in eval_source, "CUDA patch target not found"
eval_source = eval_source.replace(cuda_target, "# CUDA managed by update_interference_runner")

# Inject dataset override
shuffle_anchor = {repr(SHUFFLE_ANCHOR)}
assert shuffle_anchor in eval_source, "SHUFFLE_ANCHOR not found"
dataset_override = {repr(dataset_override)}
eval_source = eval_source.replace(shuffle_anchor, dataset_override + "\\n" + shuffle_anchor, 1)

# Inject pre-batch hook
pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert pre_anchor in eval_source, "PRE_EDIT_ANCHOR not found"
pre_hook = {repr(pre_batch_hook)}
eval_source = eval_source.replace(pre_anchor, pre_hook + pre_anchor, 1)

# Inject post-batch hook
post_anchor = {repr(POST_EDIT_ANCHOR)}
assert post_anchor in eval_source, "POST_EDIT_ANCHOR not found"
post_hook = {repr(post_batch_hook)}
eval_source = eval_source.replace(post_anchor, post_anchor + "\\n" + post_hook, 1)

# Inject checkpoint loading
ckpt_load_anchor = {repr(CHECKPOINT_LOAD_ANCHOR)}
if _ir_start_batch > 0:
    assert ckpt_load_anchor in eval_source, "CHECKPOINT_LOAD_ANCHOR not found"
    _ckpt_injection = '''
    # === INTERFERENCE_RUNNER: checkpoint resumption (injected) ===
    _ckpt_dir = Path("{checkpoint_dir}") / f"batch_{{_ir_start_batch - 1}}"
    if (_ckpt_dir / "model_weights.pt").exists():
        print(f"  [IR] Loading checkpoint from {{_ckpt_dir}}")
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
        torch.cuda.empty_cache()
        print(f"  [IR] Resumed from batch {{_ir_start_batch - 1}} ({{_ir_start_batch * {num_edits}}} edits)")
    else:
        print(f"  [IR] WARNING: checkpoint {{_ckpt_dir}} not found, starting from scratch")
    # === END checkpoint resumption ===
'''
    eval_source = eval_source.replace(ckpt_load_anchor, _ckpt_injection + "\\n" + ckpt_load_anchor, 1)

# Always inject defaults
_exec_time_default = '''
    exec_time = 0
    edited_model = model
'''
assert ckpt_load_anchor in eval_source
eval_source = eval_source.replace(ckpt_load_anchor, ckpt_load_anchor + "\\n" + _exec_time_default, 1)

print("[IR] evaluate.py patched with delta capture + interference hooks")

# ─── 5. Execute ─────────────────────────────────────────────────────
exec(compile(eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_AlphaEdit_to_model": _patched_apply,
    "get_cov": _patched_get_cov,
    "_ir_should_skip": _ir_should_skip,
    "_ir_capture_w_before": _ir_capture_w_before,
    "_ir_compute_delta": _ir_compute_delta,
    "_ir_save_checkpoint": _ir_save_checkpoint,
    "_ir_start_batch": _ir_start_batch,
    "_ir_end_batch": _ir_end_batch,
    "_ir_save_interval": _ir_save_interval,
    "_ir_num_edits": _ir_num_edits,
}})

# ─── 6. Final computation + save ────────────────────────────────────
print(f"\\n=== Interference Runner ({{_ir_ordering_name}}) complete ===")
print(f"  Recorded {{len(_ir_batch_results)}} batches ({{_ir_start_batch}}-{{_ir_end_batch}})")

# Compute net displacement for first-1K
first_1k_kidx = []
for pos in range(min(1000, len(_ir_ordering_case_ids))):
    cid = _ir_ordering_case_ids[pos]
    kidx = _ir_cid_to_kidx.get(cid, -1)
    if kidx >= 0:
        first_1k_kidx.append(kidx)
first_1k_kidx = np.array(first_1k_kidx)

K_first1K = _ir_keys[first_1k_kidx]  # [~1000, 14336]
net_effects = _ir_cum_delta @ K_first1K.T  # [4096, ~1000]
U_net_first1K = torch.linalg.norm(net_effects, dim=0).numpy()

# Baseline: need W_{1K} for relative displacement
# W_{1K} was the starting point (loaded from batch_9 checkpoint)
# Reconstruct from checkpoint
_w1k_path = _ir_checkpoint_dir / "batch_9" / "model_weights.pt"
if _w1k_path.exists():
    _w1k_weights = torch.load(str(_w1k_path), map_location="cpu")
    W_1K = _w1k_weights[_ir_weight_key].float()
    baseline_effects = W_1K @ K_first1K.T
    baseline_norms = torch.linalg.norm(baseline_effects, dim=0).numpy()
    d_rel_first1K = U_net_first1K / (baseline_norms + 1e-10)
    del W_1K, _w1k_weights
else:
    print("  WARNING: batch_9 checkpoint not found, cannot compute d_rel")
    baseline_norms = np.ones(len(first_1k_kidx))
    d_rel_first1K = U_net_first1K

# Reconstruction check: cum_delta should equal (W_end - W_start)
_w_end_path = _ir_checkpoint_dir / f"batch_{{_ir_end_batch}}" / "model_weights.pt"
_w_start_path = _ir_checkpoint_dir / f"batch_{{_ir_start_batch - 1}}" / "model_weights.pt"
if _w_end_path.exists() and _w_start_path.exists():
    _w_end = torch.load(str(_w_end_path), map_location="cpu")[_ir_weight_key].float()
    _w_start = torch.load(str(_w_start_path), map_location="cpu")[_ir_weight_key].float()
    _direct = _w_end - _w_start
    _recon_error = torch.linalg.norm(_ir_cum_delta - _direct).item()
    _direct_norm = torch.linalg.norm(_direct).item()
    _rel_recon_error = _recon_error / (_direct_norm + 1e-10)
    print(f"  Reconstruction check: rel_error={{_rel_recon_error:.2e}}")
    del _w_end, _w_start, _direct

# Save results
results = {{
    "metadata": {{
        "ordering": _ir_ordering_name,
        "seed": seed,
        "start_batch": _ir_start_batch,
        "end_batch": _ir_end_batch,
        "n_batches_recorded": len(_ir_batch_results),
        "layer": _ir_layer,
        "weight_key": _ir_weight_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }},
    "batch_results": _ir_batch_results,
    "first_1K": {{
        "case_ids": [int(_ir_ordering_case_ids[i]) for i in range(min(1000, len(_ir_ordering_case_ids)))],
        "key_indices": first_1k_kidx.tolist(),
        "U_path": _ir_path_sum[first_1k_kidx].tolist(),
        "U_net": U_net_first1K.tolist(),
        "d_rel": d_rel_first1K.tolist(),
        "I_fro_path": _ir_path_fro[first_1k_kidx].tolist(),
        "baseline_output_norm": baseline_norms.tolist(),
    }},
    "all_5K": {{
        "case_ids": [int(cid) for cid in _ir_ordering_case_ids],
        "U_path": _ir_path_sum.tolist(),
        "I_fro_path": _ir_path_fro.tolist(),
        "installation_batch": _ir_installation_batch.tolist(),
    }},
}}

_ir_output_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(_ir_output_path), "w") as f:
    json.dump(results, f, indent=2)
print(f"  Output: {{_ir_output_path}}")

# Summary stats for first-1K
print(f"\\n  First-1K interference summary:")
_f1k_path = _ir_path_sum[first_1k_kidx]
print(f"    U_path: mean={{_f1k_path.mean():.6f}}, median={{np.median(_f1k_path):.6f}}, std={{_f1k_path.std():.6f}}")
print(f"    U_net:  mean={{U_net_first1K.mean():.6f}}, median={{np.median(U_net_first1K):.6f}}")
print(f"    d_rel:  mean={{d_rel_first1K.mean():.6f}}, median={{np.median(d_rel_first1K):.6f}}")
print(f"    Path/Net ratio: {{_f1k_path.mean() / (U_net_first1K.mean() + 1e-10):.2f}} (>1 means cancellation)")
""")
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Fine-grained update interference runner"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ordering", required=True,
                        choices=["key_clustered", "key_dispersed"])
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get(
        "MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--start_from_batch", type=int, default=10,
                        help="First batch to record (default: 10 = after 1K edits)")
    parser.add_argument("--end_batch", type=int, default=49,
                        help="Last batch to record (default: 49 = 5K edits)")
    parser.add_argument("--checkpoint_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from model_download import resolve_model_path
    from setup_hparams import link_hparams
    from source_patches import patch_evaluate_file

    link_hparams()
    patch_evaluate_file(ALPHAEDIT_ROOT)
    model_name = resolve_model_path(args.model_name)

    # Resolve paths — shell launcher is responsible for S3 vs local resolution
    if args.checkpoint_base:
        ckpt_dir = Path(args.checkpoint_base)
    else:
        ckpt_dir = get_checkpoint_root() / "matched_ordering" / "AlphaEdit" / args.ordering / f"seed{args.seed}"

    # Verify batch_9 checkpoint exists (needed for resumption)
    batch9_path = ckpt_dir / "batch_9" / "model_weights.pt"
    if not batch9_path.exists():
        print(f"ERROR: batch_9 checkpoint not found at {batch9_path}")
        print("  Phase 2 requires existing batch_9 (1K edits) checkpoint.")
        sys.exit(1)

    # Resolve stream path (local results dir; shell launcher handles S3 symlinks)
    stream_path = get_result_root() / "matched_ordering" / "orderings" / f"{args.ordering}_seed{args.seed}.json"
    if not stream_path.exists():
        print(f"ERROR: Stream file not found at {stream_path}")
        print(f"  Generate with: uv run python src/datasets/generate_orderings.py --seed {args.seed}")
        sys.exit(1)

    # Resolve keys path
    keys_path = get_result_root() / "matched_ordering" / "key_geometry" / f"keys_seed{args.seed}_layer6.npz"
    if not keys_path.exists():
        print(f"ERROR: Keys not found at {keys_path}")
        sys.exit(1)

    # Output: results/interference/AlphaEdit/{ordering}/seed{seed}/fine_grained.json
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_result_root() / "interference" / "AlphaEdit" / args.ordering / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fine_grained.json"

    print(f"\n{'=' * 70}")
    print("Phase 2: Fine-Grained Update Interference Runner")
    print(f"  Seed:        {args.seed}")
    print(f"  Ordering:    {args.ordering}")
    print(f"  Model:       {model_name}")
    print(f"  Batches:     {args.start_from_batch} -> {args.end_batch}")
    print(f"  Checkpoint:  {ckpt_dir}")
    print(f"  Stream:      {stream_path}")
    print(f"  Keys:        {keys_path}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 70}")

    script = build_interference_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        dataset_path=str(stream_path),
        ordering_name=args.ordering,
        stream_length=args.stream_length,
        num_edits=args.num_edits,
        start_batch=args.start_from_batch,
        end_batch=args.end_batch,
        checkpoint_dir=str(ckpt_dir),
        keys_path=str(keys_path),
        output_path=str(output_path),
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
        print(f"\nERROR: Runner failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nPhase 2 complete. Results at: {output_path}")


if __name__ == "__main__":
    main()
