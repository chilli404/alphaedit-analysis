"""MEMIT algorithm with per-layer hook points.

Drop-in replacement for vendor apply_memit_to_model that supports
AlgorithmHooks for customizing the solve, filtering updates, and
capturing mechanism metrics — without exec(compile()) source injection.

The algorithm is identical to vendor/AlphaEdit/memit/memit_main.py
(commit b84624f) with hook call points at:
  1. build_lhs: before torch.linalg.solve
  2. post_solve: after solve, before weight update
  3. post_update: after weight update
"""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from algorithms.hooks import AlgorithmHooks

_VENDOR_ADDED = False


def _ensure_vendor_path():
    global _VENDOR_ADDED
    if not _VENDOR_ADDED:
        vendor = Path(__file__).resolve().parent.parent.parent / "vendor" / "AlphaEdit"
        sys.path.insert(0, str(vendor))
        _VENDOR_ADDED = True


def apply_memit_with_hooks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    hooks: Optional[AlgorithmHooks] = None,
    state: Optional[dict] = None,
    **_kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """MEMIT with per-layer hooks. Drop-in for apply_memit_to_model."""
    _ensure_vendor_path()
    from util import nethook

    if hooks is None:
        hooks = AlgorithmHooks()
    if state is None:
        state = hooks.get_state()

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = _execute_memit_with_hooks(
        model, tok, requests, hparams,
        cache_template=cache_template, hooks=hooks, state=state,
    )

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = _match_shape(upd_matrix, w.shape)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def _execute_memit_with_hooks(
    model, tok, requests, hparams,
    cache_template=None, hooks=None, state=None,
):
    """Core MEMIT loop with hook points. Mirrors vendor execute_memit exactly."""
    _ensure_vendor_path()
    from memit.compute_ks import compute_ks
    from memit.compute_z import compute_z, get_module_input_output_at_words
    from util import nethook
    from util.generate import generate_fast

    deltas = {}
    requests = deepcopy(requests)
    for i, req in enumerate(requests):
        if req["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + req["target_new"]["str"]

    for req in requests[:10]:
        print(f"MEMIT request sample: [{req['prompt'].format(req['subject'])}] -> [{req['target_new']['str']}]")

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z vectors
    context_templates = _get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
    for request in requests:
        cache_fname = (
            Path(str(cache_template).format(z_layer, hparams.clamp_norm_factor, request["case_id"]))
            if cache_template is not None else None
        )
        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache: {e}. Recomputing...")
        if not data_loaded:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, v_star=cur_z.detach().cpu().numpy())
    zs = torch.stack(z_list, dim=1)

    # Per-layer loop with hooks
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Covariance
        cov = _get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                       hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype)

        layer_ks, targets = layer_ks.double(), targets.double()

        # === HOOK: build_lhs ===
        if hooks.build_lhs is not None:
            lhs = hooks.build_lhs(layer, layer_ks, cov, hparams, state)
        else:
            lhs = hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T

        adj_k = torch.linalg.solve(lhs, layer_ks)
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = resid @ adj_k.T

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = _match_shape(upd_matrix, weights[weight_name].shape)

        # === HOOK: post_solve ===
        if hooks.post_solve is not None:
            upd_matrix = hooks.post_solve(layer, upd_matrix, adj_k, layer_ks, weight_name, state)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        # === HOOK: post_update ===
        if hooks.post_update is not None:
            hooks.post_update(layer, weight_name, weights[weight_name], state)

        # Cleanup
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore original weights
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas


# --- Vendor utilities (thin wrappers to avoid import issues) ---

_CONTEXT_TEMPLATES_CACHE = None
_COV_CACHE = {}


def _get_context_templates(model, tok):
    global _CONTEXT_TEMPLATES_CACHE
    if _CONTEXT_TEMPLATES_CACHE is None:
        _ensure_vendor_path()
        from util.generate import generate_fast
        _CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [f.replace("{", " ").replace("}", " ") + ". {}"
             for f in generate_fast(model, tok,
                                    ["The", "Therefore", "Because", "I", "You"],
                                    n_gen_per_prompt=n_gen // 5, max_out_len=length)]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {_CONTEXT_TEMPLATES_CACHE}")
    return _CONTEXT_TEMPLATES_CACHE


def _get_cov(model, tok, layer_name, dataset, n_samples, dtype, force=False):
    _ensure_vendor_path()
    from rome.layer_stats import layer_stats
    from util.globals import STATS_DIR

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in _COV_CACHE or force:
        stat = layer_stats(model, tok, layer_name, STATS_DIR, dataset,
                          to_collect=["mom2"], sample_size=n_samples, precision=dtype,
                          force_recompute=force)
        _COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return _COV_CACHE[key].to("cuda")


def _match_shape(matrix, shape):
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    raise ValueError(f"Shape mismatch: {matrix.shape} vs {shape}")
