"""AlphaEdit algorithm with per-layer hook points.

Drop-in replacement for vendor apply_AlphaEdit_to_model that supports
AlgorithmHooks. AlphaEdit differs from MEMIT:
  - Uses null-space projection P
  - Accepts/returns cache_c (accumulated covariance)
  - No weights_copy restore (edits are permanent per call)
  - Solve: P @ (K@K^T + cache_c) instead of alpha*C0 + K@K^T
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


def apply_alphaedit_with_hooks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    hooks: Optional[AlgorithmHooks] = None,
    state: Optional[dict] = None,
    **_kwargs,
) -> Tuple[AutoModelForCausalLM, Any]:
    """AlphaEdit with per-layer hooks. Drop-in for apply_AlphaEdit_to_model."""
    _ensure_vendor_path()
    from AlphaEdit.compute_ks import compute_ks
    from AlphaEdit.compute_z import compute_z, get_module_input_output_at_words
    from util import nethook
    from util.generate import generate_fast

    if hooks is None:
        hooks = AlgorithmHooks()
    if state is None:
        state = hooks.get_state()

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

    context_templates = _get_context_templates_ae(model, tok)
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
                z_list.append(torch.from_numpy(np.load(cache_fname)["v_star"]).to("cuda"))
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
        resid = targets / (len(hparams.layers) - i)

        # === HOOK: build_lhs ===
        # Hooks provide the INNER term only. AlphaEdit always wraps with:
        #   lhs = P @ (inner + cache_c) + L2*I
        # P is [14336, 14336] per layer — too large for GPU alongside the model.
        # Compute LHS on CPU, then move result to GPU for the solve.
        P_i = P[i, :, :].double()  # stays on CPU
        cache_c_i = cache_c[i, :, :].double() if cache_c is not None else torch.zeros(
            layer_ks.shape[0], layer_ks.shape[0], dtype=torch.float64)

        if hooks.build_lhs is not None:
            inner = hooks.build_lhs(layer, layer_ks, None, hparams, state).cpu().double()
        else:
            _ks_cpu = layer_ks.cpu().double()
            inner = _ks_cpu @ _ks_cpu.T

        lhs = (P_i @ (inner + cache_c_i)
               + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float64))

        # Move LHS and RHS to GPU for the solve (much smaller than P itself)
        rhs = (P_i @ layer_ks.cpu().double() @ resid.cpu().double().T)
        upd_matrix = torch.linalg.solve(lhs.cuda(), rhs.cuda())

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = _match_shape(upd_matrix, weights[weight_name].shape)

        # === HOOK: post_solve ===
        if hooks.post_solve is not None:
            upd_matrix = hooks.post_solve(layer, upd_matrix, None, layer_ks, weight_name, state)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        # === HOOK: post_update ===
        if hooks.post_update is not None:
            hooks.post_update(layer, weight_name, weights[weight_name], state)

        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Update cache_c with new keys
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


# --- Vendor utilities ---

_AE_CONTEXT_CACHE = None


def _get_context_templates_ae(model, tok):
    global _AE_CONTEXT_CACHE
    if _AE_CONTEXT_CACHE is None:
        _ensure_vendor_path()
        from util.generate import generate_fast
        _AE_CONTEXT_CACHE = [["{}"]] + [
            [f.replace("{", " ").replace("}", " ") + ". {}"
             for f in generate_fast(model, tok,
                                    ["The", "Therefore", "Because", "I", "You"],
                                    n_gen_per_prompt=n_gen // 5, max_out_len=length)]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {_AE_CONTEXT_CACHE}")
    return _AE_CONTEXT_CACHE


def _match_shape(matrix, shape):
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    raise ValueError(f"Shape mismatch: {matrix.shape} vs {shape}")
