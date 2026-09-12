#!/usr/bin/env python3
"""
Protected Editing Runner: cosine-based live vulnerability protection.

After each editing batch, identifies the K most vulnerable previously-installed
edits (by cosine overlap with the just-applied batch's keys) and reapplies them
to restore their target representations.

Strategies:
  - cosine_protection: reapply K edits with highest max-cosine to the new batch
  - random_protection: reapply K randomly selected old edits
  - no_protection: standard AlphaEdit (baseline, use existing runs)

Architecture: dual source injection (same as alphaedit_stream_runner).
Patches AlphaEdit_main.py to extract per-batch keys, then patches evaluate.py
to add the protection reapplication step after each batch.

Usage:
    python src/runners/protected_editing_runner.py \
        --seed 42 --ordering key_clustered \
        --protection cosine --protect_k 10

    python src/runners/protected_editing_runner.py \
        --seed 42 --ordering key_clustered \
        --protection random --protect_k 10
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
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
CACHE_UPDATE_ANCHOR = '        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T'


def build_protected_editing_script(
    seed: int,
    cuda_device: str,
    model_name: str,
    hparams_fname: str,
    dataset_path: str,
    stream_name: str,
    stream_length: int,
    num_edits: int,
    save_interval: int,
    checkpoint_dir: str,
    output_jsonl: str,
    keys_path: str,
    protection: str,
    protect_k: int,
    start_from_batch: int = 0,
) -> str:

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

    # Key-capture injection for AlphaEdit_main.py: save the per-batch keys
    # after they're computed (line 111) so the protection step can use them
    key_capture_injection = r'''
        # === PROTECTED_RUNNER: capture batch keys (injected) ===
        if '_prot_batch_keys' in globals():
            _prot_batch_keys.setdefault(int(layer), layer_ks.detach().cpu().float())
        # === END key capture ===
'''

    # Dataset override for evaluate.py
    dataset_override = f'''
    # === PROTECTED_RUNNER: dataset override (injected) ===
    import json as _json_pr
    with open("{dataset_path}", "r") as _prf:
        _pr_stream_data = _json_pr.load(_prf)
    ds.data = _pr_stream_data
    print(f"  [PROT] Loaded {{len(_pr_stream_data)}} records ({stream_name} stream)")
    # === END dataset override ===
'''

    pre_batch_hook = f'''        # === PROTECTED_RUNNER: pre-batch (injected) ===
        if '_prot_should_skip' in globals() and _prot_should_skip(cnt):
            cnt += 1
            continue
        _prot_batch_keys.clear()
        # === END pre-batch ===
'''

    # The protection step happens in the post-batch hook
    post_batch_hook = f'''        # === PROTECTED_RUNNER: post-batch + protection (injected) ===
        _prot_record_and_protect(cnt, edited_model, hparams, exec_time,
                                 record_chunks, cache_c if 'cache_c' in dir() else None,
                                 P if 'P' in dir() else None, tok)
        if (cnt + 1) % {save_interval} == 0:
            _prot_save_checkpoint(cnt, edited_model,
                                  cache_c if 'cache_c' in dir() else None, hparams)
        # === END post-batch ===
'''

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

# ─── 2. Protection state ──────────────────────────────────────────
_prot_protection = "{protection}"
_prot_K = {protect_k}
_prot_stream_name = "{stream_name}"
_prot_checkpoint_dir = Path("{checkpoint_dir}")
_prot_output = "{output_jsonl}"
_prot_start_batch = {start_from_batch}
_prot_num_edits = {num_edits}
_prot_batch_records = []

# Per-batch key capture (populated by patched AlphaEdit_main)
_prot_batch_keys = {{}}

# History of all installed edits (for protection selection)
_prot_installed_edits = []  # list of (case_id, request_dict, batch_idx)

# Precomputed key vectors for cosine-based selection
_prot_keys_data = np.load("{keys_path}")
_prot_all_keys = torch.from_numpy(_prot_keys_data["keys"]).float()
_prot_all_cids = _prot_keys_data["case_ids"].tolist()
_prot_cid_to_kidx = {{int(cid): i for i, cid in enumerate(_prot_all_cids)}}

# L2-normalize keys for cosine computation
_prot_key_norms = torch.linalg.norm(_prot_all_keys, dim=1, keepdim=True)
_prot_keys_normed = _prot_all_keys / torch.clamp(_prot_key_norms, min=1e-8)

print(f"[PROT] Protection: {{_prot_protection}}, K={{_prot_K}}")
print(f"[PROT] Loaded {{len(_prot_all_cids)}} precomputed keys")

def _prot_should_skip(cnt):
    return cnt < _prot_start_batch

def _prot_select_vulnerable(batch_records, installed_edits, K):
    \"\"\"Select K most vulnerable installed edits based on cosine to new batch keys.\"\"\"
    if not installed_edits or K <= 0:
        return []

    # Get keys for new batch
    new_cids = [r["case_id"] for r in batch_records]
    new_kidxs = [_prot_cid_to_kidx.get(cid, -1) for cid in new_cids]
    new_kidxs = [k for k in new_kidxs if k >= 0]
    if not new_kidxs:
        return []
    new_keys_normed = _prot_keys_normed[new_kidxs]  # [n_new, d]

    # Get keys for installed edits
    old_cids = [e[0] for e in installed_edits]
    old_kidxs = [_prot_cid_to_kidx.get(cid, -1) for cid in old_cids]

    # Compute max cosine for each old edit to any new edit
    scores = []
    for i, (cid, req, batch_idx) in enumerate(installed_edits):
        kidx = old_kidxs[i]
        if kidx < 0:
            scores.append((i, 0.0))
            continue
        old_key = _prot_keys_normed[kidx:kidx+1]  # [1, d]
        cos = (old_key @ new_keys_normed.T).max().item()
        scores.append((i, cos))

    # Sort by cosine (highest = most vulnerable)
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [installed_edits[scores[j][0]] for j in range(min(K, len(scores)))]
    return selected

def _prot_select_random(installed_edits, K):
    \"\"\"Select K random installed edits for protection.\"\"\"
    if not installed_edits or K <= 0:
        return []
    K = min(K, len(installed_edits))
    return random.sample(installed_edits, K)

def _prot_record_and_protect(cnt, model, hparams, exec_time,
                              record_chunks, cache_c, P, tok):
    \"\"\"After each batch: record, select vulnerable edits, reapply them.\"\"\"

    # Track installed edits
    for r in record_chunks:
        req = {{
            "case_id": r["case_id"],
            "prompt": r["requested_rewrite"]["prompt"],
            "subject": r["requested_rewrite"]["subject"],
            "target_new": r["requested_rewrite"]["target_new"],
            "target_true": r["requested_rewrite"]["target_true"],
        }}
        _prot_installed_edits.append((r["case_id"], req, cnt))

    # Select edits to protect
    n_protected = 0
    protection_cosines = []

    if _prot_protection == "cosine" and cnt > 0:
        vulnerable = _prot_select_vulnerable(record_chunks, _prot_installed_edits[:-len(record_chunks)], _prot_K)
        if vulnerable:
            protect_requests = [e[1] for e in vulnerable]
            protection_cosines = [
                float((_prot_keys_normed[_prot_cid_to_kidx[e[0]]:_prot_cid_to_kidx[e[0]]+1] @
                       _prot_keys_normed[[_prot_cid_to_kidx.get(r["case_id"], 0) for r in record_chunks]].T).max())
                for e in vulnerable if e[0] in _prot_cid_to_kidx
            ]
            # Reapply these edits (cache_c is updated in place by the vendor code)
            _, cache_c_new = apply_AlphaEdit_to_model(model, tok, protect_requests,
                                                       hparams, cache_c=cache_c, P=P)
            cache_c[:] = cache_c_new
            n_protected = len(protect_requests)

    elif _prot_protection == "random" and cnt > 0:
        vulnerable = _prot_select_random(_prot_installed_edits[:-len(record_chunks)], _prot_K)
        if vulnerable:
            protect_requests = [e[1] for e in vulnerable]
            _, cache_c_new = apply_AlphaEdit_to_model(model, tok, protect_requests,
                                                       hparams, cache_c=cache_c, P=P)
            cache_c[:] = cache_c_new
            n_protected = len(protect_requests)

    # Record
    record = {{
        "stream": _prot_stream_name,
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _prot_num_edits,
        "protection": _prot_protection,
        "n_protected": n_protected,
        "protection_cosines": protection_cosines[:5],
        "n_installed": len(_prot_installed_edits),
        "exec_time_s": round(exec_time, 2),
        "seed": seed,
    }}

    _prot_batch_records.append(record)

    with open(_prot_output, "w") as f:
        for rec in _prot_batch_records:
            f.write(json.dumps(rec) + "\\n")

    if (cnt + 1) % 5 == 0 or n_protected > 0:
        print(f"  [PROT] Batch {{cnt}}: edits={{(cnt+1)*_prot_num_edits}}, "
              f"protected={{n_protected}}/{{_prot_K}}, "
              f"installed={{len(_prot_installed_edits)}}")

def _prot_save_checkpoint(cnt, model, cache_c, hparams):
    ckpt_dir = _prot_checkpoint_dir / f"batch_{{cnt}}"
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
    metadata = {{"batch_idx": cnt, "total_edits": (cnt + 1) * _prot_num_edits,
                 "stream": _prot_stream_name, "seed": seed,
                 "protection": _prot_protection, "protect_k": _prot_K,
                 "timestamp": datetime.now(timezone.utc).isoformat()}}
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"  [PROT] Checkpoint saved: {{ckpt_dir}}")

# ─── 3. Patch AlphaEdit_main.py ─────────────────────────────────────
ae_source = Path("AlphaEdit/AlphaEdit_main.py").read_text()
ae_source = ae_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
ae_source = ae_source.replace("from .compute_z", "from AlphaEdit.compute_z")
ae_source = ae_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")

# Inject key capture after cache update
_cache_anchor = {repr(CACHE_UPDATE_ANCHOR)}
assert _cache_anchor in ae_source, "CACHE_UPDATE_ANCHOR not found"
_key_capture = {repr(key_capture_injection)}
ae_source = ae_source.replace(_cache_anchor, _cache_anchor + "\\n" + _key_capture, 1)

ae_code = compile(ae_source, "AlphaEdit/AlphaEdit_main.py", "exec")
ae_namespace = {{
    "__name__": "AlphaEdit.AlphaEdit_main",
    "__file__": "AlphaEdit/AlphaEdit_main.py",
    "_prot_batch_keys": _prot_batch_keys,
}}
exec(ae_code, ae_namespace)
_patched_apply = ae_namespace["apply_AlphaEdit_to_model"]
_patched_get_cov = ae_namespace["get_cov"]
# Also make the unpatched version accessible for protection reapplication
apply_AlphaEdit_to_model = _patched_apply
print("[PROT] AlphaEdit_main.py patched")

# ─── 4. Patch evaluate.py ───────────────────────────────────────────
with open("experiments/evaluate.py", "r") as f:
    eval_source = f.read()

import_anchor = {repr(ALGO_IMPORT_ANCHOR)}
assert import_anchor in eval_source
eval_source = eval_source.replace(import_anchor, "# patched by protected_editing_runner")

cuda_target = {repr(CUDA_PATCH_TARGET)}
assert cuda_target in eval_source
eval_source = eval_source.replace(cuda_target, "# CUDA managed by protected_editing_runner")

shuffle_anchor = {repr(SHUFFLE_ANCHOR)}
assert shuffle_anchor in eval_source
dataset_override = {repr(dataset_override)}
eval_source = eval_source.replace(shuffle_anchor, dataset_override + "\\n" + shuffle_anchor, 1)

pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert pre_anchor in eval_source
pre_hook = {repr(pre_batch_hook)}
eval_source = eval_source.replace(pre_anchor, pre_hook + pre_anchor, 1)

post_anchor = {repr(POST_EDIT_ANCHOR)}
assert post_anchor in eval_source
post_hook = {repr(post_batch_hook)}
eval_source = eval_source.replace(post_anchor, post_anchor + "\\n" + post_hook, 1)

# Checkpoint loading for resumption
ckpt_load_anchor = {repr(CHECKPOINT_LOAD_ANCHOR)}
if _prot_start_batch > 0:
    assert ckpt_load_anchor in eval_source
    _ckpt_injection = '''
    _ckpt_dir = Path("{checkpoint_dir}") / f"batch_{{_prot_start_batch - 1}}"
    if (_ckpt_dir / "model_weights.pt").exists():
        print(f"  [PROT] Loading checkpoint from {{_ckpt_dir}}")
        _ckpt_w = torch.load(_ckpt_dir / "model_weights.pt", map_location="cuda")
        _pd = dict(model.named_parameters())
        for _wn, _wt in _ckpt_w.items():
            if _wn in _pd: _pd[_wn].data.copy_(_wt.cuda())
        del _ckpt_w
        if (_ckpt_dir / "cache_c.pt").exists():
            cache_c = torch.load(_ckpt_dir / "cache_c.pt", map_location="cpu")
        torch.cuda.empty_cache()
        print(f"  [PROT] Resumed from batch {{_prot_start_batch - 1}}")
    '''
    eval_source = eval_source.replace(ckpt_load_anchor, _ckpt_injection + "\\n" + ckpt_load_anchor, 1)

_exec_time_default = '''
    exec_time = 0
    edited_model = model
'''
assert ckpt_load_anchor in eval_source
eval_source = eval_source.replace(ckpt_load_anchor, ckpt_load_anchor + "\\n" + _exec_time_default, 1)

print("[PROT] evaluate.py patched")

# ─── 5. Execute ─────────────────────────────────────────────────────
exec(compile(eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_AlphaEdit_to_model": _patched_apply,
    "get_cov": _patched_get_cov,
    "_prot_batch_keys": _prot_batch_keys,
    "_prot_should_skip": _prot_should_skip,
    "_prot_record_and_protect": _prot_record_and_protect,
    "_prot_save_checkpoint": _prot_save_checkpoint,
    "_prot_start_batch": _prot_start_batch,
    "_prot_num_edits": _prot_num_edits,
}})

print(f"\\n[PROT] Protected editing complete. {{len(_prot_batch_records)}} batches recorded.")
print(f"  Protection: {{_prot_protection}}, K={{_prot_K}}")
print(f"  Output: {{_prot_output}}")
""")
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Protected editing runner with live vulnerability mitigation"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ordering", required=True)
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--protection", choices=["cosine", "random", "none"],
                        default="cosine")
    parser.add_argument("--protect_k", type=int, default=10,
                        help="Number of edits to reapply per batch")
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--checkpoint_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from model_registry import DEFAULT_MODEL
    from model_resolve import resolve_model_path
    from setup_hparams import link_hparams
    from source_patches import patch_evaluate_file, patch_glue_eval_file

    link_hparams()
    patch_evaluate_file(ALPHAEDIT_ROOT)
    patch_glue_eval_file(ALPHAEDIT_ROOT)
    model_name = resolve_model_path(args.model_name)

    result_root = get_result_root()
    ckpt_root = get_checkpoint_root()

    # Resolve stream path
    stream_path = result_root / "matched_ordering" / "orderings" / f"{args.ordering}_seed{args.seed}.json"
    assert stream_path.exists(), f"Stream not found: {stream_path}"

    # Resolve keys
    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path

    # Checkpoint and output dirs
    prot_tag = f"{args.protection}_k{args.protect_k}"
    if args.checkpoint_base:
        ckpt_dir = Path(args.checkpoint_base)
    else:
        ckpt_dir = ckpt_root / "protected_editing" / prot_tag / args.ordering / f"seed{args.seed}"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = result_root / "protected_editing" / prot_tag / args.ordering / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / f"mechanism_seed{args.seed}.jsonl"

    # Auto-resume
    start_batch = 0
    if ckpt_dir.exists():
        existing = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
            key=lambda d: int(d.name.split("_")[1])
        )
        if existing and (existing[-1] / "model_weights.pt").exists():
            start_batch = int(existing[-1].name.split("_")[1]) + 1
            print(f"Auto-resuming from batch {start_batch}")

    print(f"\n{'=' * 70}")
    print("Protected Editing Runner")
    print(f"  Seed:       {args.seed}")
    print(f"  Ordering:   {args.ordering}")
    print(f"  Protection: {args.protection} (K={args.protect_k})")
    print(f"  Stream:     {stream_path}")
    print(f"  Keys:       {keys_path}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Output:     {output_jsonl}")
    print(f"  Resume:     batch {start_batch}")
    print(f"{'=' * 70}\n")

    script = build_protected_editing_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        dataset_path=str(stream_path),
        stream_name=args.ordering,
        stream_length=args.stream_length,
        num_edits=args.num_edits,
        save_interval=args.save_interval,
        checkpoint_dir=str(ckpt_dir),
        output_jsonl=str(output_jsonl),
        keys_path=str(keys_path),
        protection=args.protection,
        protect_k=args.protect_k,
        start_from_batch=start_batch,
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
        print(f"\nERROR: Runner failed (rc={result.returncode})")
        sys.exit(result.returncode)

    print(f"\nDone. Results at: {output_jsonl}")


if __name__ == "__main__":
    main()
