#!/usr/bin/env python3
"""
PathGuard: Hazard-Gated Displacement-Constrained Sequential Editing.

Uses evaluate_harness.run_experiment() for the edit-eval loop.
Still uses exec(compile(memit_main.py)) for the PathGuard Woodbury solve,
which replaces the entire solve pipeline and needs access to vendor internals
(targets, resid, layer iteration state) that the hook API doesn't expose.

Variants (controlled by CLI flags):
    PathGuard-E:   --pathguard --pathguard_fixed_lambda X
    PathGuard-ED:  --pathguard --pathguard_adaptive
    PathGuard-EDS: --pathguard --pathguard_adaptive (with margin shield, default)

Usage:
    python src/runners/pathguard_runner.py \
        --seed 42 --pathguard --pathguard_adaptive \
        --lambda_prev 1.0 --lambda_delta 0.0 \
        --dataset_size_limit 10000 --num_edits 100 \
        --eval_at_checkpoints_only --save_interval 10 \
        --ordering fb_high_exposure \
        --dataset_override results/matched_ordering/orderings/fb_high_exposure_seed42.json
"""

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))
sys.path.insert(0, str(_SRC_DIR))

from model_registry import DEFAULT_MODEL
from model_resolve import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file, patch_glue_eval_file
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root
from evaluate_harness import (
    run_experiment, load_model_and_tok, load_dataset,
    ExperimentHooks,
)
from checkpoint_io import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    should_save, should_skip, validate_checkpoint_path,
)
from mega_batch_eval import get_mega_batch_eval_source


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_eval_fn(fast_mode: bool = False):
    ns = {}
    exec(get_mega_batch_eval_source(), ns)
    mbe_fn = ns["_mega_batch_eval"]
    if fast_mode:
        def fast_eval(model, tok, records, template, num_edits, case_ids, exec_time):
            batch_records = [r for r in records if r["case_id"] in case_ids[-num_edits:]]
            mbe_fn(model, tok, batch_records, template, num_edits, case_ids, exec_time, batch_size=2)
        return fast_eval
    return mbe_fn


def _derive_variant_name(args) -> str:
    if args.pathguard_adaptive:
        pg = "PathGuard-EDS" if not args.pathguard_no_margin_shield else "PathGuard-ED"
    elif args.pathguard_fixed_lambda is not None:
        pg = "PathGuard-E"
    else:
        pg = "PathGuard"
    if args.pathguard_random:
        pg += "-random"
    if args.pathguard_kernel_degree > 0:
        pg += f"-poly{args.pathguard_kernel_degree}-hybrid"
    eps = f"-e{args.pathguard_epsilon_init}" if args.pathguard_epsilon_init != 1.0 else ""
    return f"{pg}-M{args.pathguard_M}{eps}"


def _resolve_ckpt_dir(args, variant_name: str, ordering: str | None) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    mn = (args.model_name or "").lower()
    tag = ""
    if "gpt-j" in mn:
        tag = "gpt-j-6b"
    elif "qwen2.5-7b" in mn:
        tag = "qwen2.5-7b"
    base = get_checkpoint_root() / "matched_ordering" if ordering else get_checkpoint_root() / "pathguard"
    if tag:
        base = base / tag
    if ordering:
        return base / variant_name / ordering / f"seed{args.seed}"
    return base / variant_name / f"seed{args.seed}"


SOLVE_ANCHOR = '        adj_k = torch.linalg.solve(\n            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,\n            layer_ks,\n        )'
DELTAS_ANCHOR = '            deltas[weight_name] = ('


def _load_pathguard_apply(alphaedit_root: Path, args, pg_state: dict):
    """Build patched apply_memit_to_model with PathGuard Woodbury solve via exec(compile()).

    This is the ONE remaining exec(compile()) — the Woodbury solve replaces the
    entire MEMIT solve pipeline and needs access to vendor internals.
    """

    # Read and fix imports in memit_main.py
    memit_path = alphaedit_root / "memit" / "memit_main.py"
    src = memit_path.read_text()
    src = src.replace("from .compute_ks", "from memit.compute_ks")
    src = src.replace("from .compute_z", "from memit.compute_z")
    src = src.replace("from .memit_hparams", "from memit.memit_hparams")

    # The PathGuard solve replacement is complex — it implements the full
    # Woodbury-based exposure-gated solve with adaptive lambda search.
    # This references vendor-internal variables (targets, cov, layer_ks, etc.)
    # that the hook API doesn't expose.
    solve_replacement = _build_pathguard_solve(args)
    assert SOLVE_ANCHOR in src, "SOLVE_ANCHOR not found in memit_main.py"
    src = src.replace(SOLVE_ANCHOR, solve_replacement, 1)

    log_code = _build_pathguard_log_cache(args)
    assert DELTAS_ANCHOR in src, "DELTAS_ANCHOR not found"
    src = src.replace(DELTAS_ANCHOR, log_code + "\n" + DELTAS_ANCHOR, 1)

    ns = {
        "__name__": "memit.memit_main",
        "__file__": str(memit_path),
        **pg_state,
    }
    exec(compile(src, str(memit_path), "exec"), ns)
    return ns["apply_memit_to_model"], ns["get_context_templates"]


def _build_pathguard_solve(args) -> str:
    """Return the PathGuard Woodbury solve as a string for injection into memit_main.py."""
    # This is a large block that replaces the vendor's 3-line solve.
    # Kept as a string template because it references vendor loop variables.
    return r'''        # === PathGuard: hazard-gated Woodbury solve (injected) ===
        _K_prev = None
        _kpkp_norm = 0.0
        if _memit_lambda_prev > 0 and layer in _memit_prev_cache and len(_memit_prev_cache[layer]) > 0:
            _K_prev = torch.cat(_memit_prev_cache[layer], dim=1).to(layer_ks.device).double()
            _kpkp_mat = _K_prev @ _K_prev.T
            _kpkp_norm = torch.linalg.norm(_kpkp_mat, ord='fro').item()

        _mom2_w = hparams.mom2_update_weight
        if _pg_kernel_degree > 0:
            _G_lin = layer_ks.T @ layer_ks
            _G_kernel = (1.0 + _G_lin).pow(_pg_kernel_degree)
            _trace_lin = _G_lin.trace().clamp(min=1e-12)
            _frob_inner = (_G_kernel * _G_lin).sum().clamp(min=1e-12)
            _kernel_scale = (_trace_lin / _frob_inner).item()
            _KKT = layer_ks @ (_G_kernel * _kernel_scale) @ layer_ks.T
            _G_lin = _G_kernel = None
        else:
            _KKT = layer_ks @ layer_ks.T
        A_base = _mom2_w * cov.double() + _KKT
        _KKT = None
        _base_lhs_norm = torch.linalg.norm(A_base, ord='fro').item()
        if _K_prev is not None:
            A_base = A_base + _memit_lambda_prev * (_K_prev @ _K_prev.T)
        if _memit_lambda_delta > 0:
            A_base = A_base + _memit_lambda_delta * torch.eye(A_base.shape[0], device=A_base.device, dtype=A_base.dtype)

        _K_S_layer = _pg_selected_keys.get(int(layer))
        _pg_solve_lambda = 0.0
        _pg_solve_displacement[0] = 0.0

        if _K_S_layer is not None and _K_S_layer.shape[1] > 0:
            _K_S_d = _K_S_layer.to(layer_ks.device).double()
            _h_d = torch.tensor(_pg_h_vec, device=layer_ks.device, dtype=torch.float64) if _pg_h_vec and len(_pg_h_vec) > 0 else torch.ones(_K_S_d.shape[1], device=layer_ks.device, dtype=torch.float64)
            _h_d = _h_d[:_K_S_d.shape[1]]

            _combined = torch.cat([layer_ks, _K_S_d], dim=1)
            _solutions = torch.linalg.solve(A_base, _combined)
            _adj_k_base = _solutions[:, :layer_ks.shape[1]]
            _A_inv_KS = _solutions[:, layer_ks.shape[1]:]
            _core = _K_S_d.T @ _A_inv_KS
            _KS_adj_k = _K_S_d.T @ _adj_k_base

            if _pg_fixed_lambda is not None:
                _candidates = [_pg_fixed_lambda]
            elif _pg_adaptive:
                _candidates = _pg_lambda_candidates
            else:
                _candidates = [1.0]

            _best_adj_k = _adj_k_base
            _best_lambda = 0.0
            _h_sqrt = torch.sqrt(_h_d)
            _W_inv_diag = 1.0 / torch.clamp(_h_d, min=1e-10)

            for _lam in sorted(_candidates):
                if _lam <= 0:
                    _candidate_adj_k = _adj_k_base
                else:
                    _inner = (1.0 / _lam) * torch.diag(_W_inv_diag) + _core
                    _correction = _A_inv_KS @ torch.linalg.solve(_inner, _KS_adj_k)
                    _candidate_adj_k = _adj_k_base - _correction

                _resid_scale = 1.0 / (len(hparams.layers) - i) if i < len(hparams.layers) else 1.0
                _resid_for_D = targets * _resid_scale
                _proj = _candidate_adj_k.T @ _K_S_d
                _proj_w = _proj * _h_sqrt.unsqueeze(0)
                _dw_proj = _resid_for_D @ _proj_w
                _D_t = torch.sum(_dw_proj ** 2).item()

                _best_adj_k = _candidate_adj_k
                _best_lambda = _lam
                if _D_t <= _pg_epsilon_t[0] or not _pg_adaptive:
                    break

            adj_k = _best_adj_k
            _pg_solve_lambda = _best_lambda
            _pg_solve_displacement[0] = _D_t
            _memit_lhs_norms = {"base_lhs_norm": _base_lhs_norm, "kpkp_norm": _kpkp_norm, "pg_lambda": _pg_solve_lambda, "pg_displacement": _pg_solve_displacement[0]}
        else:
            adj_k = torch.linalg.solve(A_base, layer_ks)
            _memit_lhs_norms = {"base_lhs_norm": _base_lhs_norm, "kpkp_norm": _kpkp_norm, "pg_lambda": 0.0, "pg_displacement": 0.0}

        _solutions = _A_inv_KS = _core = _KS_adj_k = _combined = None
        _K_S_d = _h_d = _h_sqrt = _W_inv_diag = _candidate_adj_k = None
        _inner = _correction = _adj_k_base = _dw_proj = _proj = _proj_w = _resid_for_D = None
        A_base = _kpkp_mat = _K_S_layer = _best_adj_k = None
        _K_prev = None
        torch.cuda.empty_cache()
        # === END PathGuard solve ==='''


def _build_pathguard_log_cache(args) -> str:
    return r'''            # === PathGuard: log + cache keys ===
            _upd_norm = torch.linalg.norm(upd_matrix).item()
            _dw_kprev_norm = 0.0
            if _memit_lambda_prev > 0 and layer in _memit_prev_cache and len(_memit_prev_cache[layer]) > 0:
                _kp = torch.cat(_memit_prev_cache[layer], dim=1).to(upd_matrix.device).double()
                _dw_kprev_norm = torch.linalg.norm(upd_matrix.double() @ _kp).item()
                del _kp; torch.cuda.empty_cache()
            _log_entry = {"batch": _memit_batch_idx[0], "layer": int(layer), "upd_norm": _upd_norm, "dw_kprev_norm": _dw_kprev_norm}
            if '_memit_lhs_norms' in locals(): _log_entry.update(_memit_lhs_norms)
            _memit_log.append(_log_entry)
            if _memit_lambda_prev > 0 or _memit_cache_strategy == "all":
                if layer not in _memit_prev_cache: _memit_prev_cache[layer] = []
                _memit_prev_cache[layer].append(layer_ks.detach().cpu())
                if _memit_cache_max is not None and len(_memit_prev_cache[layer]) > _memit_cache_max:
                    if _memit_cache_strategy == "recent":
                        _memit_prev_cache[layer] = _memit_prev_cache[layer][-_memit_cache_max:]
            # === END log + cache ==='''


def _load_precomputed_keys(keys_dir: str, seed: int, layers: list, repr_layer: int):
    """Load precomputed keys from NPZ files."""
    keys = {}
    cids = []
    for lyr in layers:
        candidates = [
            Path(keys_dir) / f"keys_seed{seed}_layer{lyr}.npz",
            Path(keys_dir).parent / f"layer{lyr}" / f"seed{seed}" / f"keys_seed{seed}.npz",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found:
            d = np.load(str(found))
            keys[lyr] = torch.from_numpy(d["keys"]).T.float()
            if lyr == repr_layer:
                cids = d["case_ids"].tolist()
            print(f"  [PathGuard] Loaded keys layer {lyr}: {keys[lyr].shape}")
        else:
            fb = Path(keys_dir) / f"keys_seed{seed}_layer{repr_layer}.npz"
            if fb.exists():
                d = np.load(str(fb))
                keys[lyr] = torch.from_numpy(d["keys"]).T.float()
                if not cids:
                    cids = d["case_ids"].tolist()
    cid_map = {int(c): idx for idx, c in enumerate(cids)}
    print(f"  [PathGuard] Key index: {len(cid_map)} records")
    return keys, cids, cid_map


def _compute_exposure(precomp_keys, cid_map, history_cids, batch_cids, repr_layer, M, random_select):
    """Compute exposure scores and select top-M vulnerable edits."""
    hist_idx = [cid_map[c] for c in history_cids if c in cid_map]
    if not hist_idx or repr_layer not in precomp_keys:
        return {}, [], []

    K_hist = precomp_keys[repr_layer][:, hist_idx].cuda()
    b_idx = [cid_map[c] for c in batch_cids if c in cid_map]
    if not b_idx:
        return {}, [], []

    K_batch = precomp_keys[repr_layer][:, b_idx].cuda()
    ns = min(M, len(hist_idx))

    if random_select:
        sel = random.sample(range(len(hist_idx)), ns)
        h_vec = [0.5] * ns
    else:
        Khn = K_hist / torch.clamp(torch.linalg.norm(K_hist, dim=0, keepdim=True), min=1e-8)
        Kbn = K_batch / torch.clamp(torch.linalg.norm(K_batch, dim=0, keepdim=True), min=1e-8)
        cos_mat = Khn.T @ Kbn
        max_cos = cos_mat.max(dim=1).values
        tk = torch.topk(max_cos, ns)
        sel = tk.indices.tolist()
        h_vec = tk.values.tolist()

    selected_keys = {}
    for lyr in precomp_keys:
        gi = [hist_idx[s] for s in sel]
        selected_keys[lyr] = precomp_keys[lyr][:, gi]

    del K_hist, K_batch
    return selected_keys, sel, h_vec


def run(args: argparse.Namespace) -> None:
    alphaedit_root = get_alphaedit_root()
    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)
    _seed_everything(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cache_max = None if args.cache_max == "none" else int(args.cache_max)
    ordering = getattr(args, 'ordering', None)
    variant_name = _derive_variant_name(args)

    # Paths
    ckpt_dir = _resolve_ckpt_dir(args, variant_name, ordering)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if ordering:
        results_dir = get_result_root() / "matched_ordering" / ordering / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
    else:
        results_dir = get_result_root() / "pathguard" / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect resume
    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = min(latest[0] + 1, total_batches)
            print(f"  Auto-detected: resume from batch {start_from_batch}")
        else:
            start_from_batch = 0
            print("  No existing checkpoints found. Starting from batch 0.")

    # Print config
    eval_mode = "Milestone" if args.eval_at_checkpoints_only else ("Fast" if args.fast_checkpoint else "Full")
    print(f"\n{'=' * 70}")
    print("PathGuard Runner (harness-based)")
    print(f"  Variant:        {variant_name}")
    print(f"  Seed:           {args.seed}")
    print(f"  λ_prev:         {args.lambda_prev}")
    print(f"  PathGuard M:    {args.pathguard_M}")
    print(f"  Adaptive:       {args.pathguard_adaptive}")
    print(f"  ε_init:         {args.pathguard_epsilon_init}")
    print(f"  Evaluation:     {eval_mode}")
    print(f"  Checkpoint:     {ckpt_dir}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # Load model + dataset
    model, tok = load_model_and_tok(args.model_name)
    dataset = load_dataset(
        args.ds_name, args.dataset_size_limit, alphaedit_root,
        dataset_override=args.dataset_override,
    )

    # Load hparams
    sys.path.insert(0, str(alphaedit_root))
    from memit import MEMITHyperParams
    hparams = MEMITHyperParams.from_json(alphaedit_root / "hparams" / "MEMIT" / args.hparams_fname)

    # PathGuard mutable state — shared with exec'd memit_main.py
    pg_state = {
        "_memit_lambda_prev": args.lambda_prev,
        "_memit_lambda_delta": args.lambda_delta,
        "_memit_prev_cache": {},
        "_memit_cache_max": cache_max,
        "_memit_cache_strategy": args.cache_strategy,
        "_memit_batch_idx": [0],
        "_memit_log": [],
        "_memit_fast_mode": args.fast_checkpoint,
        "_pg_selected_keys": {},
        "_pg_selected_indices": [],
        "_pg_h_vec": [],
        "_pg_epsilon_t": [args.pathguard_epsilon_init],
        "_pg_kernel_degree": args.pathguard_kernel_degree,
        "_pg_adaptive": args.pathguard_adaptive,
        "_pg_fixed_lambda": args.pathguard_fixed_lambda,
        "_pg_lambda_candidates": [float(x) for x in args.pathguard_lambda_candidates.split(",")],
        "_pg_solve_displacement": [0.0],
    }

    # Load precomputed keys
    keys_dir = args.pathguard_keys_dir or str(get_result_root() / "key_vectors" / "full_mcf")
    precomp_keys, precomp_cids, cid_map = _load_precomputed_keys(
        keys_dir, args.seed, hparams.layers, args.pathguard_representative_layer
    )

    # PathGuard history tracking
    history_cids = []
    displacement_history = []
    epsilon_init = args.pathguard_epsilon_init

    # Load patched algorithm
    apply_fn, _ = _load_pathguard_apply(alphaedit_root, args, pg_state)

    # Load checkpoint
    if start_from_batch > 0:
        ckpt_result = load_checkpoint(
            model, hparams, str(ckpt_dir), start_from_batch - 1,
            extra_state_keys=["prev_cache.pt", "pathguard_state.pt", "mechanism_log.jsonl"],
        )
        if ckpt_result.get("prev_cache.pt"):
            pg_state["_memit_prev_cache"] = ckpt_result["prev_cache.pt"]
        if ckpt_result.get("mechanism_log.jsonl"):
            pg_state["_memit_log"] = ckpt_result["mechanism_log.jsonl"]
        if ckpt_result.get("pathguard_state.pt"):
            pg_loaded = ckpt_result["pathguard_state.pt"]
            history_cids = pg_loaded.get("key_history_cids", [])
            displacement_history = pg_loaded.get("displacement_history", [])
            pg_state["_pg_epsilon_t"][0] = pg_loaded.get("epsilon_t", epsilon_init)
        pg_state["_memit_batch_idx"] = [start_from_batch]

    # Build eval function
    mbe_fn = _make_eval_fn(fast_mode=args.fast_checkpoint)

    # Wrap apply_fn with exposure computation
    def apply_with_exposure(model, tok, requests, hparams, **kwargs):
        batch_cids = [r["case_id"] for r in requests]
        # Stage 1: compute exposure and select vulnerable edits
        sel_keys, sel_idx, h_vec = _compute_exposure(
            precomp_keys, cid_map, history_cids, batch_cids,
            args.pathguard_representative_layer, args.pathguard_M, args.pathguard_random,
        )
        pg_state["_pg_selected_keys"] = sel_keys
        pg_state["_pg_selected_indices"] = sel_idx
        pg_state["_pg_h_vec"] = h_vec

        result = apply_fn(model, tok, requests, hparams, **kwargs)

        # Post-batch: update history and adapt epsilon
        history_cids.extend(batch_cids)
        displacement_history.append(pg_state["_pg_solve_displacement"][0])
        if args.pathguard_adaptive and len(displacement_history) >= 10:
            baseline = sum(displacement_history[:10]) / 10
            if baseline > 0:
                recent = displacement_history[-5:]
                ratio = (sum(recent) / len(recent)) / baseline
                if ratio > 1.5:
                    nv = sum(1 for d in recent if d / baseline > 1.5)
                    pg_state["_pg_epsilon_t"][0] = max(epsilon_init * (0.5 ** nv), epsilon_init * 0.01)
                else:
                    pg_state["_pg_epsilon_t"][0] = epsilon_init

        return result

    # Build experiment hooks
    def after_edit(batch_idx, model, records, hparams, edit_extra, exec_time):
        pg_state["_memit_batch_idx"] = [batch_idx + 1]
        if should_save(batch_idx, args.save_interval):
            pg_st = {
                "key_history_cids": history_cids,
                "key_history_batches": list(range(batch_idx + 1)),
                "displacement_history": displacement_history,
                "epsilon_t": pg_state["_pg_epsilon_t"][0],
            }
            save_checkpoint(
                batch_idx, model, hparams, str(ckpt_dir), args.num_edits,
                extra_state={
                    "prev_cache.pt": pg_state["_memit_prev_cache"],
                    "pathguard_state.pt": pg_st,
                    "mechanism_log.jsonl": pg_state["_memit_log"],
                },
                metadata={
                    "lambda_prev": args.lambda_prev,
                    "pathguard_M": args.pathguard_M,
                    "pathguard_adaptive": args.pathguard_adaptive,
                    "pathguard_epsilon": pg_state["_pg_epsilon_t"][0],
                    "variant": variant_name,
                },
            )
        if (batch_idx + 1) % 10 == 0:
            print(f"=================================================================={(batch_idx+1)*args.num_edits}_edit==================================================================", flush=True)

    def should_eval_fn(batch_idx):
        if args.eval_at_checkpoints_only:
            return should_save(batch_idx, args.save_interval)
        return True

    hooks = ExperimentHooks(
        after_edit=after_edit,
        should_eval=should_eval_fn,
        eval_fn=mbe_fn,
    )

    alg_results_dir = results_dir / variant_name

    # Run
    summary = run_experiment(
        model=model, tok=tok, hparams=hparams,
        dataset=dataset, apply_fn=apply_with_exposure,
        alg_name="MEMIT", num_edits=args.num_edits,
        results_dir=alg_results_dir, ds_name=args.ds_name,
        conserve_memory=args.conserve_memory, hooks=hooks,
    )

    # Write mechanism log
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"log_{variant_name}_seed{args.seed}_{timestamp}.jsonl"
    with open(log_path, "w") as f:
        for entry in pg_state["_memit_log"]:
            f.write(json.dumps(entry) + "\n")

    # Metadata
    metadata = {
        "experiment": "pathguard", "variant": variant_name,
        "seed": args.seed, "lambda_prev": args.lambda_prev,
        "pathguard_M": args.pathguard_M, "pathguard_adaptive": args.pathguard_adaptive,
        "pathguard_epsilon_init": args.pathguard_epsilon_init,
        "model_name": args.model_name, "dataset_size_limit": args.dataset_size_limit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_config_hash": hash_eval_config(),
        "runner": "pathguard_runner (harness-based)",
    }
    with open(results_dir / f"metadata_{variant_name}_seed{args.seed}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("PathGuard completed.")
    print(f"  Batches: {summary['batches_run']}, Edits: {summary['total_edits']}")
    print(f"  Log: {log_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="PathGuard: Hazard-Gated Sequential Editing")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=10000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=10)
    parser.add_argument("--conserve_memory", action="store_true", default=True)
    parser.add_argument("--lambda_prev", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--cache_strategy", default="all", choices=["recent", "all"])
    parser.add_argument("--cache_max", default="none")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--start_from_batch", type=int, default=-1)
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--fast_checkpoint", action="store_true")
    eval_group.add_argument("--eval_at_checkpoints_only", action="store_true")
    parser.add_argument("--dataset_override", type=str, default=None)
    parser.add_argument("--ordering", type=str, default=None)
    parser.add_argument("--continue_from_run", type=str, default=None)
    # PathGuard-specific
    parser.add_argument("--pathguard", action="store_true")
    parser.add_argument("--pathguard_M", type=int, default=200)
    parser.add_argument("--pathguard_adaptive", action="store_true")
    parser.add_argument("--pathguard_fixed_lambda", type=float, default=None)
    parser.add_argument("--pathguard_random", action="store_true")
    parser.add_argument("--pathguard_keys_dir", type=str, default=None)
    parser.add_argument("--pathguard_representative_layer", type=int, default=6)
    parser.add_argument("--pathguard_q", type=int, default=10)
    parser.add_argument("--pathguard_no_margin_shield", action="store_true")
    parser.add_argument("--pathguard_epsilon_init", type=float, default=1.0)
    parser.add_argument("--pathguard_lambda_candidates", type=str, default="0.01,0.1,0.5,1.0,2.0,5.0,10.0")
    parser.add_argument("--pathguard_kernel_degree", type=int, default=0)
    parser.add_argument("--pathguard_signed_diag", action="store_true")
    parser.add_argument("--pathguard_signed_lambda", action="store_true")
    parser.add_argument("--pathguard_signed_max_screen", type=int, default=100)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
