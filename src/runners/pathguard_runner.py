#!/usr/bin/env python3
"""
PathGuard: Hazard-Gated Displacement-Constrained Sequential Editing.

Builds on MEMIT-Seq (non-projected sequential regularization) and adds:

  Stage 1 — Exposure retrieval:
    For batch B_t, compute e_{j,t} = max_{i in B_t} cos(k_j, k_i) for each
    historical edit j. Select top-M by exposure.

  Stage 2 — Risk-weighted covariance:
    G_t = K_S @ diag(h) @ K_S^T where K_S are the M selected keys and
    h = exposure scores. Augment the MEMIT-Seq LHS:
        (A_t + lambda_t * G_t) x = K_new

  Stage 3 — Adaptive displacement budget:
    Choose smallest lambda_t satisfying D_t <= epsilon_t AND editability
    constraint, using Woodbury identity for efficient lambda search.

  Stage 4 — Signed margin shield (optional):
    For top-q vulnerable edits, compute margin gradient and constrain
    <nabla_W M_j, DeltaW> >= -delta_j via dual correction.

The base LHS preserves MEMIT-Seq's terms:
    A_t = alpha * C0 + K_new @ K_new^T + lambda_prev * K_prev @ K_prev^T + lambda_delta * I

Variants (controlled by CLI flags):
    PathGuard-E:   --pathguard --pathguard_fixed_lambda X
    PathGuard-ED:  --pathguard --pathguard_adaptive
    PathGuard-EDS: --pathguard --pathguard_adaptive (with margin shield, default)
    PathGuard-ED-random: --pathguard --pathguard_adaptive --pathguard_random

Implementation: Dual source injection (same as memit_sequential_runner.py).

Usage:
    python src/runners/pathguard_runner.py \\
        --seed 42 --pathguard --pathguard_adaptive \\
        --lambda_prev 1.0 --lambda_delta 0.0 \\
        --dataset_size_limit 10000 --num_edits 100 \\
        --eval_at_checkpoints_only --save_interval 10 \\
        --ordering fb_high_exposure \\
        --dataset_override results/matched_ordering/orderings/fb_high_exposure_seed42.json
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

from model_resolve import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file, patch_glue_eval_file
from eval_config import hash_eval_config
from paths import get_project_root, get_alphaedit_root, get_result_root, get_checkpoint_root


_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def _model_tag(model_name: str | None) -> str:
    if not model_name or model_name == _DEFAULT_MODEL:
        return ""
    short = model_name.rsplit("/", 1)[-1].lower()
    if "qwen2.5-7b" in short:
        return "qwen2.5-7b"
    if "gpt-j" in short:
        return "gpt-j-6b"
    return short


def resolve_pathguard_checkpoint_dir(
    explicit_dir: str | None,
    seed: int,
    variant_name: str,
    ordering: str | None = None,
    model_name: str | None = None,
    **_kwargs,
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)

    tag = _model_tag(model_name)

    if ordering:
        if tag:
            return get_checkpoint_root() / "matched_ordering" / tag / variant_name / ordering / f"seed{seed}"
        return get_checkpoint_root() / "matched_ordering" / variant_name / ordering / f"seed{seed}"

    if tag:
        return get_checkpoint_root() / "pathguard" / tag / variant_name / f"seed{seed}"
    return get_checkpoint_root() / "pathguard" / variant_name / f"seed{seed}"


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path] | None:
    if not ckpt_dir.exists():
        return None
    batch_dirs = sorted(
        [d for d in ckpt_dir.glob("batch_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    if not batch_dirs:
        return None
    for batch_dir in reversed(batch_dirs):
        if (batch_dir / "metadata.json").exists():
            try:
                return (int(batch_dir.name.split("_")[1]), batch_dir)
            except (ValueError, IndexError):
                continue
    return None


# --- Source anchors (commit b84624f) ---

CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
POST_EDIT_ANCHOR = '        exec_time = time() - start'
MEMIT_IMPORT_ANCHOR = 'from memit.memit_main import apply_memit_to_model, get_context_templates'

SOLVE_ANCHOR = '        adj_k = torch.linalg.solve(\n            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,\n            layer_ks,\n        )'
DELTAS_ANCHOR = '            deltas[weight_name] = ('


def build_pathguard_script(
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
    lambda_prev: float,
    lambda_delta: float,
    cache_strategy: str,
    cache_max: int | None,
    output_jsonl: str,
    fast_checkpoint: bool = False,
    eval_at_checkpoints_only: bool = False,
    save_interval: int = 10,
    checkpoint_dir: str = "",
    start_from_batch: int = 0,
    dataset_override: str | None = None,
    eval_results_dir: str = "",
    variant_name: str = "",
    # PathGuard-specific
    pathguard_M: int = 200,
    pathguard_adaptive: bool = True,
    pathguard_fixed_lambda: float | None = None,
    pathguard_random: bool = False,
    pathguard_keys_dir: str = "",
    pathguard_representative_layer: int = 6,
    pathguard_q: int = 10,
    pathguard_no_margin_shield: bool = False,
    pathguard_epsilon_init: float = 1.0,
    pathguard_lambda_candidates: str = "0.01,0.1,1.0,10.0",
    pathguard_kernel_degree: int = 0,
    continue_from_run: str | None = None,
    # Signed diagnostic / lambda integration
    pathguard_signed_diag: bool = False,
    pathguard_signed_lambda: bool = False,
    pathguard_signed_max_screen: int = 100,
) -> str:
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
    if continue_from_run:
        argv_parts.append(f"--continue_from_run={continue_from_run}")

    argv_str = repr(argv_parts)
    cache_max_repr = repr(cache_max)

    # --- Solve replacement: PathGuard Woodbury solve ---
    solve_replacement = r'''        # === PathGuard: hazard-gated Woodbury solve (injected) ===
        # Stage 0: MEMIT-Seq base LHS
        _K_prev = None
        _kpkp_norm = 0.0
        if _memit_lambda_prev > 0 and layer in _memit_prev_cache and len(_memit_prev_cache[layer]) > 0:
            _K_prev = torch.cat(_memit_prev_cache[layer], dim=1).to(layer_ks.device).double()
            _kpkp_mat = _K_prev @ _K_prev.T
            _kpkp_norm = torch.linalg.norm(_kpkp_mat, ord='fro').item()

        _mom2_w = _memit_mom2_override if _memit_mom2_override is not None else hparams.mom2_update_weight
        # Current-batch K@K^T: kernel-weighted if poly2 enabled, else linear
        if _pg_kernel_degree > 0:
            _G_lin = layer_ks.T @ layer_ks
            _G_kernel = (1.0 + _G_lin).pow(_pg_kernel_degree)
            _trace_lin = _G_lin.trace().clamp(min=1e-12)
            _frob_inner = (_G_kernel * _G_lin).sum().clamp(min=1e-12)
            _kernel_scale = (_trace_lin / _frob_inner).item()
            _KKT = layer_ks @ (_G_kernel * _kernel_scale) @ layer_ks.T
            _G_lin = _G_kernel = None  # free
        else:
            _KKT = layer_ks @ layer_ks.T
        A_base = _mom2_w * cov.double() + _KKT
        _KKT = None  # free
        _base_lhs_norm = torch.linalg.norm(A_base, ord='fro').item()
        if _K_prev is not None:
            A_base = A_base + _memit_lambda_prev * (_K_prev @ _K_prev.T)
        if _memit_lambda_delta > 0:
            A_base = A_base + _memit_lambda_delta * torch.eye(A_base.shape[0], device=A_base.device, dtype=A_base.dtype)

        # Stages 1-3: PathGuard Woodbury solve
        _K_S_layer = _pg_selected_keys.get(int(layer))
        _pg_solve_lambda = 0.0
        _pg_solve_displacement[0] = 0.0
        _pg_solve_candidates_tried = 0

        if _K_S_layer is not None and _K_S_layer.shape[1] > 0:
            _K_S_d = _K_S_layer.to(layer_ks.device).double()
            _h_d = torch.tensor(_pg_h_vec, device=layer_ks.device, dtype=torch.float64) if _pg_h_vec is not None and len(_pg_h_vec) > 0 else torch.ones(_K_S_d.shape[1], device=layer_ks.device, dtype=torch.float64)
            _h_d = _h_d[:_K_S_d.shape[1]]

            # Single factorization: solve A @ [K_new | K_S]
            _combined = torch.cat([layer_ks, _K_S_d], dim=1)
            _solutions = torch.linalg.solve(A_base, _combined)
            _adj_k_base = _solutions[:, :layer_ks.shape[1]]
            _A_inv_KS = _solutions[:, layer_ks.shape[1]:]

            # M x M core for Woodbury
            _core = _K_S_d.T @ _A_inv_KS
            _KS_adj_k = _K_S_d.T @ _adj_k_base

            # Lambda candidates
            if _pg_fixed_lambda is not None:
                _candidates = [_pg_fixed_lambda]
            elif _pg_adaptive:
                _candidates = _pg_lambda_candidates
            else:
                _candidates = [1.0]

            _best_adj_k = _adj_k_base
            _best_lambda = 0.0
            _best_D = float('inf')
            _h_sqrt = torch.sqrt(_h_d)
            _W_inv_diag = 1.0 / torch.clamp(_h_d, min=1e-10)

            for _lam_idx, _lam in enumerate(sorted(_candidates)):
                if _lam <= 0:
                    _candidate_adj_k = _adj_k_base
                else:
                    _inner = (1.0 / _lam) * torch.diag(_W_inv_diag) + _core
                    _correction = _A_inv_KS @ torch.linalg.solve(_inner, _KS_adj_k)
                    _candidate_adj_k = _adj_k_base - _correction

                # Compute displacement
                _resid_scale = 1.0 / (len(hparams.layers) - i) if i < len(hparams.layers) else 1.0
                _resid_for_D = targets * _resid_scale
                _proj = _candidate_adj_k.T @ _K_S_d
                _proj_w = _proj * _h_sqrt.unsqueeze(0)
                _dw_proj = _resid_for_D @ _proj_w
                _D_t = torch.sum(_dw_proj ** 2).item()

                if _D_t <= _pg_epsilon_t[0] or not _pg_adaptive:
                    _best_adj_k = _candidate_adj_k
                    _best_lambda = _lam
                    _best_D = _D_t
                    _pg_solve_candidates_tried = _lam_idx + 1
                    break
                _best_adj_k = _candidate_adj_k
                _best_lambda = _lam
                _best_D = _D_t
                _pg_solve_candidates_tried = _lam_idx + 1

            adj_k = _best_adj_k
            _pg_solve_lambda = _best_lambda
            _pg_solve_displacement[0] = _best_D

            _memit_lhs_norms = {
                "base_lhs_norm": _base_lhs_norm,
                "kpkp_norm": _kpkp_norm,
                "mom2_weight_used": _mom2_w,
                "pg_lambda": _pg_solve_lambda,
                "pg_displacement": _pg_solve_displacement[0],
                "pg_candidates_tried": _pg_solve_candidates_tried,
            }
        else:
            adj_k = torch.linalg.solve(A_base, layer_ks)
            _memit_lhs_norms = {
                "base_lhs_norm": _base_lhs_norm,
                "kpkp_norm": _kpkp_norm,
                "mom2_weight_used": _mom2_w,
                "pg_lambda": 0.0,
                "pg_displacement": 0.0,
                "pg_candidates_tried": 0,
            }
        # Eagerly compute dw_kprev_norm while upd_matrix and K_prev are alive
        # (upd_matrix = resid @ adj_k.T is computed by vendor code AFTER this block,
        #  but adj_k is set above, so we can compute it with a proxy)
        _pg_cached_dw_kprev_norm = 0.0
        # Note: actual dw_kprev_norm computed in log block after upd_matrix exists

        # Free ALL large intermediates — set to None to release GPU memory
        # (use = None, not del, to avoid UnboundLocalError in exec'd code)
        _solutions = _A_inv_KS = _core = _KS_adj_k = _combined = None
        _K_S_d = _h_d = _h_sqrt = _W_inv_diag = _candidate_adj_k = None
        _inner = _correction = _adj_k_base = _dw_proj = _proj = _proj_w = _resid_for_D = None
        A_base = _kpkp_mat = _K_S_layer = _best_adj_k = _best_D = None
        # Release K_prev tensor — it's the biggest (~550MB+ in double at later batches)
        # Set to None instead of del to avoid UnboundLocalError in later references
        _K_prev = None
        torch.cuda.empty_cache()
        # === END PathGuard solve ==='''

    # --- Log and cache code ---
    log_and_cache_code = r'''            # === PathGuard: log + store keys (injected) ===
            _upd_norm = torch.linalg.norm(upd_matrix).item()
            _dw_kprev_norm = 0.0
            _cache_batches = len(_memit_prev_cache.get(layer, []))
            _cache_keys = sum(k.shape[1] for k in _memit_prev_cache.get(layer, []))
            # Recompute dw_kprev_norm from cache (K_prev was deleted in solve for memory)
            if _memit_lambda_prev > 0 and layer in _memit_prev_cache and len(_memit_prev_cache[layer]) > 0:
                _kp_for_log = torch.cat(_memit_prev_cache[layer], dim=1).to(upd_matrix.device).double()
                _dw_kprev_norm = torch.linalg.norm(upd_matrix.double() @ _kp_for_log).item()
                del _kp_for_log
                torch.cuda.empty_cache()

            _log_entry = {
                "batch": _memit_batch_idx[0], "layer": int(layer),
                "upd_norm": _upd_norm, "dw_kprev_norm": _dw_kprev_norm,
                "cache_batches": _cache_batches, "cache_keys": _cache_keys,
                "pg_lambda": _pg_solve_lambda,
                "pg_displacement": _pg_solve_displacement,
                "pg_n_vulnerable": len(_pg_selected_indices),
                "pg_max_exposure": max(_pg_h_vec) if _pg_h_vec and len(_pg_h_vec) > 0 else 0.0,
                "pg_mean_hazard": (sum(_pg_h_vec) / len(_pg_h_vec)) if _pg_h_vec and len(_pg_h_vec) > 0 else 0.0,
            }
            if '_memit_lhs_norms' in locals():
                _log_entry.update(_memit_lhs_norms)
            _memit_log.append(_log_entry)

            # Append current keys to MEMIT-Seq cache
            if _memit_lambda_prev > 0 or _memit_cache_strategy == "all":
                if layer not in _memit_prev_cache:
                    _memit_prev_cache[layer] = []
                _memit_prev_cache[layer].append(layer_ks.detach().cpu())
                if _memit_cache_max is not None and len(_memit_prev_cache[layer]) > _memit_cache_max:
                    if _memit_cache_strategy == "recent":
                        _memit_prev_cache[layer] = _memit_prev_cache[layer][-_memit_cache_max:]
            del _K_prev
            # === END PathGuard log + store keys ==='''

    batch_increment_hook = r'''        # === PathGuard: increment batch (injected) ===
        if '_memit_batch_idx' in globals():
            _memit_batch_idx[0] += 1
        # === END batch increment ===
'''

    # PathGuard post-batch hook: update history + adapt epsilon
    pg_post_batch_hook = r'''        # === PathGuard: update history and adapt epsilon (injected) ===
        if '_pg_key_history_cids' in globals():
            _pg_batch_cids_post = [r["case_id"] for r in record_chunks]
            _pg_key_history_cids.extend(_pg_batch_cids_post)
            _pg_key_history_batches.extend([_memit_batch_idx[0] - 1] * len(_pg_batch_cids_post))
            _pg_displacement_history.append(_pg_solve_displacement[0])
            if _pg_adaptive and len(_pg_displacement_history) >= 10:
                _pg_bl = sum(_pg_displacement_history[:10]) / 10
                if _pg_bl > 0:
                    _pg_rc = _pg_displacement_history[-5:]
                    _pg_rt = (sum(_pg_rc) / len(_pg_rc)) / _pg_bl
                    if _pg_rt > 1.5:
                        _pg_nv = sum(1 for _d in _pg_rc if _d / _pg_bl > 1.5)
                        _pg_epsilon_t[0] = max(_pg_epsilon_init * (0.5 ** _pg_nv), _pg_epsilon_init * 0.01)
                    else:
                        _pg_epsilon_t[0] = _pg_epsilon_init
        # === END PathGuard post-batch ===
'''

    # PathGuard exposure computation injection (before PRE_EDIT_ANCHOR in evaluate.py)
    pg_exposure_injection = r'''        # === PathGuard: compute exposure and select vulnerable edits (injected) ===
        if '_pg_precomputed_keys' in globals() and len(_pg_key_history_cids) > 0 and len(_pg_precomputed_keys) > 0:
            _pg_batch_cids_exp = [r["case_id"] for r in record_chunks]
            _pg_repr_layer = _pg_representative_layer
            _pg_hist_idx = []
            for _pg_hcid in _pg_key_history_cids:
                if _pg_hcid in _pg_precomputed_cids_map:
                    _pg_hist_idx.append(_pg_precomputed_cids_map[_pg_hcid])
            if _pg_hist_idx and _pg_repr_layer in _pg_precomputed_keys:
                _pg_K_hist_exp = _pg_precomputed_keys[_pg_repr_layer][:, _pg_hist_idx].to("cuda")
                _pg_b_idx = [_pg_precomputed_cids_map[c] for c in _pg_batch_cids_exp if c in _pg_precomputed_cids_map]
                if _pg_b_idx:
                    _pg_K_batch_exp = _pg_precomputed_keys[_pg_repr_layer][:, _pg_b_idx].to("cuda")
                    if _pg_random_select:
                        import random as _pg_rng_mod
                        _pg_ns = min(_pg_M, len(_pg_hist_idx))
                        _pg_sel_local = _pg_rng_mod.sample(range(len(_pg_hist_idx)), _pg_ns)
                        _pg_h_vec[:] = [0.5] * _pg_ns
                        _pg_selected_indices[:] = _pg_sel_local
                    else:
                        _pg_Khn = _pg_K_hist_exp / torch.clamp(torch.linalg.norm(_pg_K_hist_exp, dim=0, keepdim=True), min=1e-8)
                        _pg_Kbn = _pg_K_batch_exp / torch.clamp(torch.linalg.norm(_pg_K_batch_exp, dim=0, keepdim=True), min=1e-8)
                        _pg_cm = _pg_Khn.T @ _pg_Kbn
                        _pg_mc = _pg_cm.max(dim=1).values
                        _pg_ns = min(_pg_M, len(_pg_hist_idx))
                        _pg_tk = torch.topk(_pg_mc, _pg_ns)
                        _pg_selected_indices[:] = _pg_tk.indices.tolist()
                        _pg_h_vec[:] = _pg_tk.values.tolist()
                    _pg_selected_keys.clear()
                    for _pg_ly in _pg_precomputed_keys:
                        _pg_gi = [_pg_hist_idx[si] for si in _pg_selected_indices]
                        _pg_selected_keys[_pg_ly] = _pg_precomputed_keys[_pg_ly][:, _pg_gi]
                    del _pg_K_hist_exp, _pg_K_batch_exp
                else:
                    _pg_selected_keys.clear()
                    _pg_selected_indices[:] = []
                    _pg_h_vec[:] = []
            else:
                _pg_selected_keys.clear()
                _pg_selected_indices[:] = []
                _pg_h_vec[:] = []
        # === END PathGuard exposure ===
'''

    # PathGuard key loading injection (after checkpoint load, before loop)
    pg_key_load_injection = '''    # === PathGuard: load precomputed keys (injected) ===
    if '_pg_keys_dir' in globals() and _pg_keys_dir:
        from pathlib import Path as _pg_Path
        import numpy as _pg_np
        _pg_precomputed_cids = []
        # Load keys for each layer in hparams (resolved after model loads)
        def _pg_load_precomputed_keys_fn(hparams_obj):
            for _lyr in hparams_obj.layers:
                # Try multiple key path conventions
                _kp_candidates = [
                    _pg_Path(_pg_keys_dir) / f"keys_seed''' + str(seed) + '''_layer{_lyr}.npz",
                    _pg_Path(_pg_keys_dir).parent / f"layer{_lyr}" / f"seed''' + str(seed) + '''" / f"keys_seed''' + str(seed) + '''.npz",
                    _pg_Path(_pg_keys_dir).parent.parent / f"layer{_lyr}" / f"seed''' + str(seed) + '''" / f"keys_seed''' + str(seed) + '''.npz",
                ]
                _kp_found = None
                for _kpc in _kp_candidates:
                    if _kpc.exists():
                        _kp_found = _kpc
                        break
                if _kp_found:
                    _d = _pg_np.load(str(_kp_found))
                    _pg_precomputed_keys[_lyr] = torch.from_numpy(_d["keys"]).T.float()
                    if _lyr == _pg_representative_layer:
                        _pg_precomputed_cids[:] = _d["case_ids"].tolist()
                    print(f"  [PathGuard] Loaded keys for layer {_lyr}: {_pg_precomputed_keys[_lyr].shape} from {_kp_found}")
                else:
                    _fb = _pg_Path(_pg_keys_dir) / f"keys_seed''' + str(seed) + '''_layer{_pg_representative_layer}.npz"
                    if _fb.exists():
                        _d = _pg_np.load(str(_fb))
                        _pg_precomputed_keys[_lyr] = torch.from_numpy(_d["keys"]).T.float()
                        if not _pg_precomputed_cids:
                            _pg_precomputed_cids[:] = _d["case_ids"].tolist()
                        print(f"  [PathGuard] Fallback layer {_pg_representative_layer} keys for layer {_lyr}")
                    else:
                        print(f"  [PathGuard] WARNING: No keys found for layer {_lyr}")
            global _pg_precomputed_cids_map
            _pg_precomputed_cids_map = {int(c): idx for idx, c in enumerate(_pg_precomputed_cids)}
            print(f"  [PathGuard] Key index built: {len(_pg_precomputed_cids_map)} records")
    # === END PathGuard key loading ===
'''

    # Mega-batch eval injection (copied from memit_sequential_runner)
    mega_batch_eval_injection = '''    # === MEGA-BATCH EVAL: batched multi-token scoring (injected by pathguard_runner) ===
    def _mega_batch_eval(model, tok, records, case_result_template, num_edits, case_ids, exec_time, batch_size=4):
        import torch as _mbe_torch
        import numpy as _mbe_np
        import json as _mbe_json
        from time import time as _mbe_time
        from pathlib import Path as _mbe_Path

        _is_llama = 'llama' in model.config._name_or_path.lower()
        _mbe_start = _mbe_time()
        _mbe_total = len(records)
        _mbe_done = 0
        _mbe_skipped = 0

        for batch_start in range(0, _mbe_total, batch_size):
            batch_records = records[batch_start:batch_start + batch_size]
            all_sequences = []
            record_meta = []

            for record in batch_records:
                out_file = _mbe_Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    record_meta.append(None)
                    _mbe_skipped += 1
                    continue

                subject = record["requested_rewrite"]["subject"]
                target_new = record["requested_rewrite"]["target_new"]["str"]
                target_true = record["requested_rewrite"]["target_true"]["str"]

                rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
                paraphrase_prompts = record["paraphrase_prompts"]
                neighborhood_prompts = record["neighborhood_prompts"]

                prefixes = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
                which_correct = [0] * len(rewrite_prompts) + [0] * len(paraphrase_prompts) + [1] * len(neighborhood_prompts)

                a_tok = tok(f" {target_new}")["input_ids"]
                b_tok = tok(f" {target_true}")["input_ids"]
                if _is_llama:
                    a_tok = a_tok[1:]
                    b_tok = b_tok[1:]

                prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
                if _is_llama:
                    prefix_lens = [l - 1 for l in prefix_lens]

                seqs = [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]]
                seq_start_idx = len(all_sequences)
                all_sequences.extend(seqs)

                record_meta.append({
                    "record": record, "out_file": out_file,
                    "a_tok": a_tok, "b_tok": b_tok,
                    "prefix_lens": prefix_lens, "which_correct": which_correct,
                    "n_prefixes": len(prefixes), "n_rewrite": len(rewrite_prompts),
                    "n_paraphrase": len(paraphrase_prompts), "n_neighborhood": len(neighborhood_prompts),
                    "seq_start_idx": seq_start_idx, "n_seqs": len(seqs),
                })

            if not all_sequences:
                _mbe_done += len(batch_records)
                continue

            prompt_tok = tok(all_sequences, padding=True, return_tensors="pt").to("cuda")
            with _mbe_torch.no_grad():
                logits = model(**prompt_tok).logits
            if _is_llama:
                logits = logits[:, 1:, :]

            for meta in record_meta:
                if meta is None:
                    continue
                record = meta["record"]
                start_idx = meta["seq_start_idx"]
                n_seqs = meta["n_seqs"]
                a_tok = meta["a_tok"]
                b_tok = meta["b_tok"]
                prefix_lens = meta["prefix_lens"]
                which_correct = meta["which_correct"]
                choice_a_len = len(a_tok)
                choice_b_len = len(b_tok)
                rec_logits = logits[start_idx:start_idx + n_seqs]
                probs = _mbe_np.zeros((n_seqs,), dtype=_mbe_np.float32)
                targets_correct = []
                for i in range(n_seqs):
                    cur_len = choice_a_len if i % 2 == 0 else choice_b_len
                    for j in range(cur_len):
                        cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                        probs[i] += -_mbe_torch.nn.functional.log_softmax(
                            rec_logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
                        )[cur_tok].item()
                    probs[i] /= cur_len
                    if (which_correct[i // 2] == 0 and i % 2 == 0) or (which_correct[i // 2] == 1 and i % 2 == 1):
                        correct = True
                        for j in range(cur_len):
                            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                            if rec_logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                                correct = False
                                break
                        targets_correct.append(correct)
                ret_probs = [{"target_new": probs[i].item(), "target_true": probs[i + 1].item()} for i in range(0, n_seqs, 2)]
                n_rw = meta["n_rewrite"]
                n_para = meta["n_paraphrase"]
                n_neigh = meta["n_neighborhood"]
                cutoffs = [0, n_rw, n_rw + n_para, n_rw + n_para + n_neigh]
                ret_corrects = [targets_correct[cutoffs[i]:cutoffs[i+1]] for i in range(3)]
                post = {
                    "rewrite_prompts_probs": ret_probs[:n_rw], "rewrite_prompts_correct": ret_corrects[0],
                    "paraphrase_prompts_probs": ret_probs[n_rw:n_rw + n_para], "paraphrase_prompts_correct": ret_corrects[1],
                    "neighborhood_prompts_probs": ret_probs[n_rw + n_para:], "neighborhood_prompts_correct": ret_corrects[2],
                }
                metrics = {"case_id": record["case_id"], "grouped_case_ids": case_ids, "num_edits": num_edits,
                           "requested_rewrite": record["requested_rewrite"], "time": exec_time, "post": post}
                with open(meta["out_file"], "w") as f:
                    _mbe_json.dump(metrics, f, indent=1)

            del prompt_tok, logits
            _mbe_torch.cuda.empty_cache()
            _mbe_done += len(batch_records)
            _elapsed = _mbe_time() - _mbe_start
            _rate = _mbe_done / _elapsed if _elapsed > 0 else 0
            if _mbe_done % (batch_size * 4) < batch_size or _mbe_done >= _mbe_total:
                print(f"  [MEGA-BATCH EVAL] {_mbe_done}/{_mbe_total} records ({_mbe_skipped} skipped, {_rate:.1f} rec/s, {_elapsed:.0f}s elapsed)")
        print(f"  [MEGA-BATCH EVAL] Complete: {_mbe_total} records in {_mbe_time() - _mbe_start:.1f}s")

    if _do_final_eval:
        _records_to_eval = list(ds)
        if _memit_fast_mode:
            _records_to_eval = [r for r in ds if r["case_id"] in case_ids]
        _mega_batch_eval(edited_model, tok, _records_to_eval, case_result_template, num_edits, case_ids, exec_time)
    # === END mega-batch eval ===
    for record in ds:
        break  # Mega-batch handles all eval above; skip vendor fallback loop
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'''

    # ---- Build the full script ----
    script = textwrap.dedent(f"""\
import os, sys, random, json
import numpy as np
import torch

# 0. Add project root to sys.path for signed_diagnostic import
_project_root = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, _project_root)

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

# 3. MEMIT-Seq base parameters (shared state)
_memit_lambda_prev = {lambda_prev}
_memit_lambda_delta = {lambda_delta}
_memit_mom2_override = {repr(None)}
_memit_prev_cache = {{}}
_memit_cache_max = {cache_max_repr}
_memit_cache_strategy = "{cache_strategy}"
_memit_batch_idx = [0]
_memit_log = []
_memit_output_jsonl = "{output_jsonl}"
_memit_fast_mode = {fast_checkpoint}

# 3b. PathGuard parameters
_pg_M = {pathguard_M}
_pg_adaptive = {pathguard_adaptive}
_pg_fixed_lambda = {repr(pathguard_fixed_lambda)}
_pg_random_select = {pathguard_random}
_pg_keys_dir = "{pathguard_keys_dir}"
_pg_representative_layer = {pathguard_representative_layer}
_pg_q = {pathguard_q}
_pg_no_margin_shield = {pathguard_no_margin_shield}
_pg_epsilon_init = {pathguard_epsilon_init}
_pg_lambda_candidates = [{pathguard_lambda_candidates}]
_pg_kernel_degree = {pathguard_kernel_degree}

# 3c. PathGuard mutable state (lists/dicts for cross-namespace sharing)
_pg_key_history_cids = []
_pg_key_history_batches = []
_pg_precomputed_keys = {{}}
_pg_precomputed_cids = []
_pg_precomputed_cids_map = {{}}
_pg_selected_keys = {{}}
_pg_selected_indices = []
_pg_h_vec = []
_pg_displacement_history = []
_pg_epsilon_t = [{pathguard_epsilon_init}]
_pg_lambda_used = [0.0]
_pg_solve_displacement = [0.0]  # mutable container for cross-namespace sharing

# 3e. Signed diagnostic parameters
_pg_signed_diag = {pathguard_signed_diag}
_pg_signed_lambda = {pathguard_signed_lambda}
_pg_signed_max_screen = {pathguard_signed_max_screen}
_pg_signed_log = []
_pg_w_before_snapshot = {{}}  # {{param_name: tensor}} snapshot before each batch edit

# 3d. Checkpoint parameters
_ckpt_save_interval = {save_interval}
_ckpt_dir = "{checkpoint_dir}"
_ckpt_start_batch = {start_from_batch}
_ckpt_num_edits = {num_edits}
_ckpt_eval_at_checkpoints_only = {eval_at_checkpoints_only}

def _ckpt_save(cnt, model, hparams):
    from pathlib import Path
    from datetime import datetime, timezone

    batch_dir = Path(_ckpt_dir) / f"batch_{{cnt}}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    layer_weights = {{}}
    _all_params = dict(model.named_parameters())
    for layer_idx in hparams.layers:
        _rewrite_key = hparams.rewrite_module_tmp.format(layer_idx) + ".weight"
        if _rewrite_key in _all_params:
            layer_weights[_rewrite_key] = _all_params[_rewrite_key].data.cpu()
    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    torch.save(_memit_prev_cache, str(batch_dir / "prev_cache.pt"))

    _pg_state = {{
        "key_history_cids": _pg_key_history_cids,
        "key_history_batches": _pg_key_history_batches,
        "displacement_history": _pg_displacement_history,
        "epsilon_t": _pg_epsilon_t[0],
    }}
    torch.save(_pg_state, str(batch_dir / "pathguard_state.pt"))

    with open(str(batch_dir / "mechanism_log.jsonl"), "w") as f:
        for entry in _memit_log:
            f.write(json.dumps(entry) + "\\n")

    metadata = {{
        "batch_idx": cnt,
        "total_edits": (cnt + 1) * _ckpt_num_edits,
        "batch_idx_counter": _memit_batch_idx[0],
        "lambda_prev": _memit_lambda_prev,
        "lambda_delta": _memit_lambda_delta,
        "pathguard_M": _pg_M,
        "pathguard_adaptive": _pg_adaptive,
        "pathguard_epsilon": _pg_epsilon_t[0],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }}
    with open(str(batch_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [CHECKPOINT] Saved batch {{cnt}} ({{(cnt+1) * _ckpt_num_edits}} edits) -> {{batch_dir}}")

def _ckpt_load(model, hparams):
    from pathlib import Path

    if _ckpt_start_batch <= 0:
        return False

    batch_dir = Path(_ckpt_dir) / f"batch_{{_ckpt_start_batch - 1}}"
    if not batch_dir.exists():
        print(f"  [CHECKPOINT] WARNING: Expected checkpoint at {{batch_dir}} not found.")
        return False

    weights_file = batch_dir / "model_weights.pt"
    if weights_file.exists():
        layer_weights = torch.load(str(weights_file), map_location="cuda")
        param_dict = dict(model.named_parameters())
        loaded_count = 0
        for param_name, param_data in layer_weights.items():
            if param_name in param_dict:
                param_dict[param_name].data.copy_(param_data)
                loaded_count += 1
        print(f"  [CHECKPOINT] Loaded {{loaded_count}} parameter tensors")

    cache_file = batch_dir / "prev_cache.pt"
    if cache_file.exists():
        global _memit_prev_cache
        _loaded_cache = torch.load(str(cache_file), map_location="cpu")
        _memit_prev_cache.clear()
        _memit_prev_cache.update(_loaded_cache)
        del _loaded_cache
        total_keys = sum(sum(k.shape[1] for k in v) for v in _memit_prev_cache.values())
        print(f"  [CHECKPOINT] Loaded prev_cache ({{len(_memit_prev_cache)}} layers, {{total_keys}} total keys)")

    _pg_state_file = batch_dir / "pathguard_state.pt"
    if _pg_state_file.exists():
        _pg_loaded = torch.load(str(_pg_state_file), map_location="cpu")
        _pg_key_history_cids.clear()
        _pg_key_history_cids.extend(_pg_loaded["key_history_cids"])
        _pg_key_history_batches.clear()
        _pg_key_history_batches.extend(_pg_loaded["key_history_batches"])
        _pg_displacement_history.clear()
        _pg_displacement_history.extend(_pg_loaded["displacement_history"])
        _pg_epsilon_t[0] = _pg_loaded["epsilon_t"]
        print(f"  [PathGuard] Loaded state: {{len(_pg_key_history_cids)}} history edits, epsilon={{_pg_epsilon_t[0]:.4f}}")

    log_file = batch_dir / "mechanism_log.jsonl"
    if log_file.exists():
        global _memit_log
        _memit_log.clear()
        with open(str(log_file)) as f:
            for line in f:
                _memit_log.append(json.loads(line))
        print(f"  [CHECKPOINT] Loaded {{len(_memit_log)}} log entries")

    _memit_batch_idx[0] = _ckpt_start_batch
    print(f"  [CHECKPOINT] Resuming from batch {{_ckpt_start_batch}}")
    return True

def _ckpt_should_skip(cnt):
    return cnt < _ckpt_start_batch

def _ckpt_should_save(cnt):
    return _ckpt_dir and (cnt + 1) % _ckpt_save_interval == 0

# 4. Read and patch memit_main.py
with open("memit/memit_main.py", "r") as f:
    _memit_source = f.read()

_memit_source = _memit_source.replace("from .compute_ks", "from memit.compute_ks")
_memit_source = _memit_source.replace("from .compute_z", "from memit.compute_z")
_memit_source = _memit_source.replace("from .memit_hparams", "from memit.memit_hparams")

_solve_anchor = {repr(SOLVE_ANCHOR)}
assert _solve_anchor in _memit_source, "SOLVE_ANCHOR not found in memit_main.py."
_solve_replacement = {repr(solve_replacement)}
_memit_source = _memit_source.replace(_solve_anchor, _solve_replacement, 1)

_deltas_anchor = {repr(DELTAS_ANCHOR)}
assert _deltas_anchor in _memit_source, "DELTAS_ANCHOR not found in memit_main.py."
_log_cache_code = {repr(log_and_cache_code)}
_memit_source = _memit_source.replace(_deltas_anchor, _log_cache_code + "\\n" + _deltas_anchor, 1)

assert "PathGuard" in _memit_source, "PathGuard solve injection failed"

# 5. Compile and exec patched memit
_memit_ns = {{
    "__name__": "memit.memit_main",
    "__file__": "memit/memit_main.py",
    "_memit_lambda_prev": _memit_lambda_prev,
    "_memit_lambda_delta": _memit_lambda_delta,
    "_memit_mom2_override": _memit_mom2_override,
    "_memit_prev_cache": _memit_prev_cache,
    "_memit_cache_max": _memit_cache_max,
    "_memit_cache_strategy": _memit_cache_strategy,
    "_memit_batch_idx": _memit_batch_idx,
    "_memit_log": _memit_log,
    "_pg_selected_keys": _pg_selected_keys,
    "_pg_selected_indices": _pg_selected_indices,
    "_pg_h_vec": _pg_h_vec,
    "_pg_epsilon_t": _pg_epsilon_t,
    "_pg_kernel_degree": _pg_kernel_degree,
    "_pg_adaptive": _pg_adaptive,
    "_pg_fixed_lambda": _pg_fixed_lambda,
    "_pg_lambda_candidates": _pg_lambda_candidates,
    "_pg_solve_displacement": _pg_solve_displacement,
    "_pg_signed_diag": _pg_signed_diag,
    "_pg_signed_lambda": _pg_signed_lambda,
}}
exec(compile(_memit_source, "memit/memit_main.py", "exec"), _memit_ns)
_patched_apply_memit = _memit_ns["apply_memit_to_model"]
_patched_get_context_templates = _memit_ns["get_context_templates"]

print("[PathGuard] memit_main.py patched successfully")
print(f"  base: lambda_prev={{_memit_lambda_prev}}, lambda_delta={{_memit_lambda_delta}}")
print(f"  PathGuard: M={{_pg_M}}, adaptive={{_pg_adaptive}}, epsilon_init={{_pg_epsilon_init}}")

# 6. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    _eval_source = f.read()

_import_anchor = {repr(MEMIT_IMPORT_ANCHOR)}
assert _import_anchor in _eval_source, "MEMIT_IMPORT_ANCHOR not found."
_eval_source = _eval_source.replace(_import_anchor, "# apply_memit_to_model patched by pathguard_runner")

_cuda_target = {repr(CUDA_PATCH_TARGET)}
assert _cuda_target in _eval_source, "CUDA patch target not found."
_eval_source = _eval_source.replace(_cuda_target, "# CUDA managed by pathguard_runner")

_globals_import = 'from util.globals import *'
assert _globals_import in _eval_source, "globals import not found"
_eval_source = _eval_source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)

_eval_source = _eval_source.replace(
    'dir_name=args.alg_name,',
    'dir_name="{variant_name}",',
    1,
)

_loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert _loop_anchor in _eval_source, "Loop anchor not found."

# Inject fingerprint
_fp_code = '''    # === FINGERPRINT (injected) ===
    import hashlib as _fp_hashlib
    import json as _fp_json
    _fp_case_ids = [r["case_id"] for r in ds]
    _fp_sha256 = _fp_hashlib.sha256(_fp_json.dumps(_fp_case_ids, separators=(",", ":")).encode()).hexdigest()
    print(f"  [FINGERPRINT] Dataset: {{len(ds)}} records, SHA-256: {{_fp_sha256[:16]}}...")
    # === END fingerprint ===
'''
_eval_source = _eval_source.replace(_loop_anchor, _fp_code + _loop_anchor, 1)

# Inject dataset override
_ds_override_path = {repr(dataset_override) if dataset_override else 'None'}
if _ds_override_path:
    _ds_override_code = '''    # === DATASET OVERRIDE (injected) ===
    import json as _dsov_json
    with open("{dataset_override}", "r") as _dsov_f:
        _dsov_stream = _dsov_json.load(_dsov_f)
    _dsov_attr = "_data" if hasattr(ds, "_data") else "data"
    _dsov_existing = getattr(ds, _dsov_attr)
    _dsov_id_map = {{r.get("case_id", i): r for i, r in enumerate(_dsov_existing)}}
    _dsov_stream_ids = [r["case_id"] for r in _dsov_stream]
    _dsov_matched = [_dsov_id_map[cid] for cid in _dsov_stream_ids if cid in _dsov_id_map]
    if len(_dsov_matched) >= len(_dsov_stream_ids) * 0.95:
        setattr(ds, _dsov_attr, _dsov_matched)
        print(f"  [OVERRIDE] Reordered {{len(_dsov_matched)}} records from {dataset_override}")
    else:
        setattr(ds, _dsov_attr, _dsov_stream)
        print(f"  [OVERRIDE] Replaced with {{len(_dsov_stream)}} records from {dataset_override}")
    # === END dataset override ===
'''
    _eval_source = _eval_source.replace(_loop_anchor, _ds_override_code + _loop_anchor, 1)

# Inject checkpoint load
_ckpt_load_injection = '''    # === CHECKPOINT: load state (injected) ===
    exec_time = 0
    edited_model = model
    if _ckpt_start_batch > 0 and '_ckpt_load' in globals():
        _ckpt_load(model, hparams)
    # === END checkpoint load ===
'''
_eval_source = _eval_source.replace(_loop_anchor, _ckpt_load_injection + _loop_anchor, 1)

# Inject PathGuard key loading (after checkpoint load, before loop)
_pg_key_load_code = {repr(pg_key_load_injection)} + '''
    # Call key loading (hparams is always defined by this point in evaluate.py main())
    if '_pg_load_precomputed_keys_fn' in dir():
        _pg_load_precomputed_keys_fn(hparams)
    elif '_pg_load_precomputed_keys_fn' in globals():
        _pg_load_precomputed_keys_fn(hparams)
    else:
        print("  [PathGuard] WARNING: _pg_load_precomputed_keys_fn not defined")
'''
_eval_source = _eval_source.replace(_loop_anchor, _pg_key_load_code + _loop_anchor, 1)

# Inject skip guard
_pre_anchor = {repr(PRE_EDIT_ANCHOR)}
assert _pre_anchor in _eval_source, "PRE_EDIT_ANCHOR not found."
_skip_injection = '''        # === CHECKPOINT: skip (injected) ===
        if '_ckpt_should_skip' in globals() and _ckpt_should_skip(cnt):
            cnt += 1
            continue
        # === END skip ===

        # === SIGNED DIAG: snapshot W_before (injected) ===
        if _pg_signed_diag or _pg_signed_lambda:
            _pg_w_before_snapshot.clear()
            _all_p = dict(model.named_parameters())
            for _sl in hparams.layers:
                _skey = hparams.rewrite_module_tmp.format(_sl) + ".weight"
                if _skey in _all_p:
                    _pg_w_before_snapshot[_skey] = _all_p[_skey].data.detach().clone()
        # === END signed snapshot ===
'''
_eval_source = _eval_source.replace(_pre_anchor, _skip_injection + _pre_anchor, 1)

# Inject PathGuard exposure computation before edit
_pg_exposure_code = {repr(pg_exposure_injection)}
_eval_source = _eval_source.replace(_pre_anchor, _pg_exposure_code + _pre_anchor, 1)

# Inject batch increment + PathGuard post-batch + checkpoint save AFTER POST_EDIT_ANCHOR
_post_anchor = {repr(POST_EDIT_ANCHOR)}
assert _post_anchor in _eval_source, "POST_EDIT_ANCHOR not found."
_batch_hook = {repr(batch_increment_hook)}
_pg_post_hook = {repr(pg_post_batch_hook)}
_signed_diag_hook = '''        # === SIGNED DIAGNOSTIC: screen vulnerable edits (injected) ===
        if (_pg_signed_diag or _pg_signed_lambda) and _pg_w_before_snapshot:
            _sd_delta_w = {{}}
            _sd_all_p = dict(model.named_parameters())
            for _sd_pname, _sd_wb in _pg_w_before_snapshot.items():
                if _sd_pname in _sd_all_p:
                    _sd_delta_w[_sd_pname] = (_sd_all_p[_sd_pname].data.detach() - _sd_wb.to(_sd_all_p[_sd_pname].device))

            if _sd_delta_w and _pg_selected_indices and len(_pg_key_history_cids) > 0:
                # Get records for the top vulnerable edits (capped at _pg_signed_max_screen)
                _sd_vuln_cids = []
                for _sd_si in _pg_selected_indices[:_pg_signed_max_screen]:
                    if _sd_si < len(_pg_key_history_cids):
                        _sd_vuln_cids.append(_pg_key_history_cids[_sd_si])
                # Look up full records from dataset
                _sd_records = [r for r in ds if r["case_id"] in set(_sd_vuln_cids)]
                if _sd_records:
                    from src.mechanism.signed_diagnostic import compute_signed_hazard_batch, summarize_signed_diagnostics
                    _sd_results = compute_signed_hazard_batch(
                        model, tok, _sd_records, _sd_delta_w, hparams,
                        max_edits=_pg_signed_max_screen
                    )
                    _sd_summary = summarize_signed_diagnostics(_sd_results)
                    _sd_summary["batch"] = _memit_batch_idx[0] - 1

                    _pg_signed_log.append(_sd_summary)
                    _sd_nc = _sd_summary["n_predicted_crossings"]
                    _sd_nn = _sd_summary["n_negative_effect"]
                    _sd_ns = _sd_summary["n_screened"]
                    print("  [SIGNED DIAG] batch %d: %d screened, %d predicted crossings, "
                          "%d negative effects, mean_hazard=%.4f" % (
                          _memit_batch_idx[0]-1, _sd_ns, _sd_nc, _sd_nn,
                          _sd_summary["mean_signed_hazard"]))

                    # V4: signed-informed lambda adaptation
                    if _pg_signed_lambda and _sd_nc > 0:
                        _sd_old_eps = _pg_epsilon_t[0]
                        _sd_crossing_frac = _sd_nc / max(_sd_ns, 1)
                        if _sd_crossing_frac > 0.05:
                            _pg_epsilon_t[0] = max(_pg_epsilon_init * 0.25, _sd_old_eps * 0.5)
                            _pg_lambda_candidates_boosted = [c * 2.0 for c in _pg_lambda_candidates]
                            print("  [SIGNED LAMBDA] High crossing rate (%.2f): "
                                  "epsilon %.4f -> %.4f" % (_sd_crossing_frac, _sd_old_eps, _pg_epsilon_t[0]))
                        elif _sd_crossing_frac < 0.01 and _sd_old_eps < _pg_epsilon_init:
                            _pg_epsilon_t[0] = min(_pg_epsilon_init, _sd_old_eps * 1.5)
                            print("  [SIGNED LAMBDA] Low crossing rate (%.2f): "
                                  "epsilon %.4f -> %.4f" % (_sd_crossing_frac, _sd_old_eps, _pg_epsilon_t[0]))

            _pg_w_before_snapshot.clear()
            if "_sd_delta_w" in dir():
                del _sd_delta_w
            torch.cuda.empty_cache()
        # === END signed diagnostic ===
'''
_ckpt_save_hook = '''        # === CHECKPOINT: save (injected) ===
        if '_ckpt_should_save' in globals() and _ckpt_should_save(cnt):
            _ckpt_save(cnt, model, hparams)
        # === END checkpoint save ===
'''
_eval_source = _eval_source.replace(_post_anchor, _post_anchor + "\\n" + _batch_hook + _pg_post_hook + _signed_diag_hook + _ckpt_save_hook, 1)

# Inject checkpoint-only eval guard
_eval_start_anchor = '    # torch.save(hs, "post_edit_hs_memit.pt")\\n    start = time()'
assert _eval_start_anchor in _eval_source, "Eval start anchor not found."
_checkpoint_eval_skip = '''    # torch.save(hs, "post_edit_hs_memit.pt")
    # Always run final mega-batch eval (checkpoint-only mode skips INTERMEDIATE evals, not the final one)
    _do_final_eval = True
    start = time()'''
_eval_source = _eval_source.replace(_eval_start_anchor, _checkpoint_eval_skip, 1)

# Inject mega-batch eval
_eval_anchor = '    for record in ds:\\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
assert _eval_anchor in _eval_source, "Eval loop anchor not found."
_mega_batch_eval_injection = {repr(mega_batch_eval_injection)}
_eval_source = _eval_source.replace(_eval_anchor, _mega_batch_eval_injection, 1)

print("[PathGuard] evaluate.py patched successfully")

# 7. Execute patched evaluate.py
exec(compile(_eval_source, "experiments/evaluate.py", "exec"), {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "__builtins__": __builtins__,
    "apply_memit_to_model": _patched_apply_memit,
    "get_context_templates": _patched_get_context_templates,
    "_memit_lambda_prev": _memit_lambda_prev,
    "_memit_lambda_delta": _memit_lambda_delta,
    "_memit_prev_cache": _memit_prev_cache,
    "_memit_cache_max": _memit_cache_max,
    "_memit_cache_strategy": _memit_cache_strategy,
    "_memit_batch_idx": _memit_batch_idx,
    "_memit_log": _memit_log,
    "_memit_fast_mode": _memit_fast_mode,
    "_ckpt_start_batch": _ckpt_start_batch,
    "_ckpt_save_interval": _ckpt_save_interval,
    "_ckpt_dir": _ckpt_dir,
    "_ckpt_eval_at_checkpoints_only": _ckpt_eval_at_checkpoints_only,
    "_ckpt_save": _ckpt_save,
    "_ckpt_load": _ckpt_load,
    "_ckpt_should_skip": _ckpt_should_skip,
    "_ckpt_should_save": _ckpt_should_save,
    "_pg_M": _pg_M,
    "_pg_adaptive": _pg_adaptive,
    "_pg_fixed_lambda": _pg_fixed_lambda,
    "_pg_random_select": _pg_random_select,
    "_pg_keys_dir": _pg_keys_dir,
    "_pg_representative_layer": _pg_representative_layer,
    "_pg_epsilon_init": _pg_epsilon_init,
    "_pg_lambda_candidates": _pg_lambda_candidates,
    "_pg_key_history_cids": _pg_key_history_cids,
    "_pg_key_history_batches": _pg_key_history_batches,
    "_pg_precomputed_keys": _pg_precomputed_keys,
    "_pg_precomputed_cids": _pg_precomputed_cids,
    "_pg_precomputed_cids_map": _pg_precomputed_cids_map,
    "_pg_selected_keys": _pg_selected_keys,
    "_pg_selected_indices": _pg_selected_indices,
    "_pg_h_vec": _pg_h_vec,
    "_pg_displacement_history": _pg_displacement_history,
    "_pg_epsilon_t": _pg_epsilon_t,
    "_pg_solve_displacement": _pg_solve_displacement,
    "_pg_signed_diag": _pg_signed_diag,
    "_pg_signed_lambda": _pg_signed_lambda,
    "_pg_signed_max_screen": _pg_signed_max_screen,
    "_pg_signed_log": _pg_signed_log,
    "_pg_w_before_snapshot": _pg_w_before_snapshot,
}})

# 8. Write log
with open(_memit_output_jsonl, "w") as f:
    for entry in _memit_log:
        f.write(json.dumps(entry) + "\\n")

# 8b. Write signed diagnostic log if enabled
if _pg_signed_diag or _pg_signed_lambda:
    _sd_log_path = _memit_output_jsonl.replace(".jsonl", "_signed.jsonl")
    with open(_sd_log_path, "w") as f:
        for entry in _pg_signed_log:
            f.write(json.dumps(entry) + "\\n")
    print(f"[PathGuard] Signed diagnostic log: {{_sd_log_path}} ({{len(_pg_signed_log)}} entries)")

print(f"\\n[PathGuard] Log written: {{_memit_output_jsonl}} ({{len(_memit_log)}} entries)")
""")
    return script


def validate_anchors() -> None:
    alphaedit_root = get_alphaedit_root()

    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    for name, anchor in [
        ("CUDA_PATCH_TARGET", CUDA_PATCH_TARGET),
        ("PRE_EDIT_ANCHOR", PRE_EDIT_ANCHOR),
        ("POST_EDIT_ANCHOR", POST_EDIT_ANCHOR),
        ("MEMIT_IMPORT_ANCHOR", MEMIT_IMPORT_ANCHOR),
    ]:
        assert anchor in eval_source, f"{name} not found in evaluate.py"

    memit_source = (alphaedit_root / "memit" / "memit_main.py").read_text()
    for name, anchor in [
        ("SOLVE_ANCHOR", SOLVE_ANCHOR),
        ("DELTAS_ANCHOR", DELTAS_ANCHOR),
    ]:
        assert anchor in memit_source, f"{name} not found in memit_main.py"

    print("  All source anchors validated.")


def main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PathGuard: Hazard-Gated Displacement-Constrained Sequential Editing"
    )

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")

    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", _DEFAULT_MODEL))
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
    parser.add_argument("--continue_from_run", type=str, default=None,
                        help="Reuse existing run directory (e.g. run_004) to resume eval")

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
    parser.add_argument("--pathguard_kernel_degree", type=int, default=0,
                        help="Poly kernel degree for current-batch K@K^T (0=linear, 2=poly2-hybrid)")

    # Signed diagnostic / lambda integration
    parser.add_argument("--pathguard_signed_diag", action="store_true",
                        help="Enable signed hazard diagnostic screening (V1: log only, no repair)")
    parser.add_argument("--pathguard_signed_lambda", action="store_true",
                        help="Enable signed-informed lambda adaptation (V4: adjusts epsilon based on crossing load)")
    parser.add_argument("--pathguard_signed_max_screen", type=int, default=100,
                        help="Max number of vulnerable edits to screen per batch")

    return parser


def run(args: argparse.Namespace) -> None:
    alphaedit_root = get_alphaedit_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)
    patch_glue_eval_file(alphaedit_root)

    model_name = resolve_model_path(args.model_name)

    print("Validating source anchors...")
    validate_anchors()

    cache_max = None if args.cache_max == "none" else int(args.cache_max)

    # Derive variant name
    if args.pathguard_adaptive:
        if args.pathguard_no_margin_shield:
            pg_variant = "PathGuard-ED"
        else:
            pg_variant = "PathGuard-EDS"
    elif args.pathguard_fixed_lambda is not None:
        pg_variant = "PathGuard-E"
    else:
        pg_variant = "PathGuard"
    if args.pathguard_random:
        pg_variant += "-random"
    if args.pathguard_kernel_degree > 0:
        pg_variant += f"-poly{args.pathguard_kernel_degree}-hybrid"
    eps_str = f"-e{args.pathguard_epsilon_init}" if args.pathguard_epsilon_init != 1.0 else ""
    variant_name = f"{pg_variant}-M{args.pathguard_M}{eps_str}"

    ordering = getattr(args, 'ordering', None)
    if ordering:
        results_dir = get_result_root() / "matched_ordering" / ordering / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
    else:
        results_dir = get_result_root() / "pathguard" / f"seed{args.seed}" / f"{args.dataset_size_limit}edits"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = results_dir / f"log_{variant_name}_seed{args.seed}_{timestamp}.jsonl"

    ckpt_dir = resolve_pathguard_checkpoint_dir(
        args.checkpoint_dir, args.seed, variant_name,
        ordering=ordering, model_name=args.model_name,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_batches = args.dataset_size_limit // args.num_edits
    start_from_batch = args.start_from_batch
    if start_from_batch < 0:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            start_from_batch = latest[0] + 1
            if start_from_batch >= total_batches:
                start_from_batch = total_batches
            else:
                print(f"  Auto-detected: resume from batch {start_from_batch}")
        else:
            start_from_batch = 0

    # Resolve keys directory
    keys_dir = args.pathguard_keys_dir or str(get_result_root() / "key_vectors" / "full_mcf")

    script = build_pathguard_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name="MEMIT",
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        lambda_prev=args.lambda_prev,
        lambda_delta=args.lambda_delta,
        cache_strategy=args.cache_strategy,
        cache_max=cache_max,
        output_jsonl=str(output_jsonl),
        fast_checkpoint=args.fast_checkpoint,
        eval_at_checkpoints_only=args.eval_at_checkpoints_only,
        save_interval=args.save_interval,
        checkpoint_dir=str(ckpt_dir),
        start_from_batch=start_from_batch,
        dataset_override=args.dataset_override,
        eval_results_dir=str(results_dir),
        variant_name=variant_name,
        pathguard_M=args.pathguard_M,
        pathguard_adaptive=args.pathguard_adaptive,
        pathguard_fixed_lambda=args.pathguard_fixed_lambda,
        pathguard_random=args.pathguard_random,
        pathguard_keys_dir=keys_dir,
        pathguard_representative_layer=args.pathguard_representative_layer,
        pathguard_q=args.pathguard_q,
        pathguard_no_margin_shield=args.pathguard_no_margin_shield,
        pathguard_epsilon_init=args.pathguard_epsilon_init,
        pathguard_lambda_candidates=args.pathguard_lambda_candidates,
        pathguard_kernel_degree=args.pathguard_kernel_degree,
        continue_from_run=args.continue_from_run,
        pathguard_signed_diag=getattr(args, 'pathguard_signed_diag', False),
        pathguard_signed_lambda=getattr(args, 'pathguard_signed_lambda', False),
        pathguard_signed_max_screen=getattr(args, 'pathguard_signed_max_screen', 100),
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"\n{'=' * 70}")
    print("PathGuard Runner")
    print(f"  Variant:        {variant_name}")
    print(f"  Seed:           {args.seed}")
    print(f"  Base λ_prev:    {args.lambda_prev}")
    print(f"  PathGuard M:    {args.pathguard_M}")
    print(f"  Adaptive:       {args.pathguard_adaptive}")
    print(f"  Margin shield:  {not args.pathguard_no_margin_shield} (q={args.pathguard_q})")
    print(f"  ε_init:         {args.pathguard_epsilon_init}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Signed diag:    {getattr(args, 'pathguard_signed_diag', False)}")
    print(f"  Signed lambda:  {getattr(args, 'pathguard_signed_lambda', False)}")
    print(f"  Keys dir:       {keys_dir}")
    print(f"  Checkpoint:     {ckpt_dir}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    metadata = {
        "experiment": "pathguard",
        "variant": variant_name,
        "seed": args.seed,
        "lambda_prev": args.lambda_prev,
        "lambda_delta": args.lambda_delta,
        "pathguard_M": args.pathguard_M,
        "pathguard_adaptive": args.pathguard_adaptive,
        "pathguard_random": args.pathguard_random,
        "pathguard_q": args.pathguard_q,
        "pathguard_epsilon_init": args.pathguard_epsilon_init,
        "model_name": args.model_name,
        "dataset_size_limit": args.dataset_size_limit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eval_config_hash": hash_eval_config(),
    }
    meta_path = results_dir / f"metadata_{variant_name}_seed{args.seed}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(alphaedit_root),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Experiment failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("PathGuard completed.")
    print(f"  Log:       {output_jsonl}")
    print(f"{'=' * 70}")


def main():
    parser = main_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
