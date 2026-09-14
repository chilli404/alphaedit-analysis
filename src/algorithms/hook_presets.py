"""Pre-built hook configurations for each algorithm variant.

Each function returns an AlgorithmHooks instance expressing the unique
logic that was previously injected via exec(compile(memit_main.py)).

Hooks can be composed: revive+seqreg = compose_hooks(seqreg_hooks(...), revive_hooks(...))
"""

import torch
from typing import Optional

from algorithms.hooks import AlgorithmHooks, compose_hooks


def seqreg_hooks(
    lambda_prev: float = 1.0,
    lambda_delta: float = 0.0,
    cache_strategy: str = "all",
    cache_max: Optional[int] = None,
    kernel_type: str = "poly",
    kernel_degree: int = 1,
    kernel_sigma: str = "median",
    kernel_prev: bool = True,
) -> AlgorithmHooks:
    """MEMIT-Seq / Polykernel-SeqReg: K_prev regularization + optional kernel weighting."""

    def init_state():
        return {
            "prev_cache": {},
            "mechanism_log": [],
            "batch_idx": [0],
            "lambda_prev": lambda_prev,
            "lambda_delta": lambda_delta,
        }

    def build_lhs(layer_idx, layer_ks, cov, hparams, state):
        alpha = getattr(hparams, 'mom2_update_weight', 1.0)
        if cov is not None:
            lhs_base = alpha * cov.double() + layer_ks @ layer_ks.T
        else:
            lhs_base = layer_ks @ layer_ks.T

        if lambda_prev > 0 and layer_idx in state["prev_cache"] and state["prev_cache"][layer_idx]:
            K_prev = torch.cat(state["prev_cache"][layer_idx], dim=1).to(layer_ks.device).double()
            lhs = lhs_base + lambda_prev * (K_prev @ K_prev.T)
            del K_prev
        else:
            lhs = lhs_base

        if lambda_delta > 0:
            lhs = lhs + lambda_delta * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)

        torch.cuda.empty_cache()
        return lhs

    def post_solve(layer_idx, upd_matrix, adj_k, layer_ks, weight_name, state):
        # Phase 2 call (layer_idx=None): skip logging and caching, just pass through.
        # SeqReg only needs Phase 1 for key caching; REVIVE handles Phase 2 filtering.
        if layer_idx is None:
            return upd_matrix

        upd_norm = torch.linalg.norm(upd_matrix).item()
        state["mechanism_log"].append({
            "batch": state["batch_idx"][0],
            "layer": int(layer_idx),
            "upd_norm": upd_norm,
        })

        if lambda_prev > 0 or cache_strategy == "all":
            if layer_idx not in state["prev_cache"]:
                state["prev_cache"][layer_idx] = []
            state["prev_cache"][layer_idx].append(layer_ks.detach().cpu())
            if cache_max is not None and len(state["prev_cache"][layer_idx]) > cache_max:
                if cache_strategy == "recent":
                    state["prev_cache"][layer_idx] = state["prev_cache"][layer_idx][-cache_max:]

        return upd_matrix

    return AlgorithmHooks(
        build_lhs=build_lhs,
        post_solve=post_solve,
        init_state=init_state,
    )


def revive_hooks(
    revive_tau: float = 0.1,
    revive_svd_device: str = "cuda",
    revive_svd_dtype: str = "float32",
) -> AlgorithmHooks:
    """REVIVE spectral filter: project updates away from dominant singular directions."""

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    svd_dtype = dtype_map.get(revive_svd_dtype, torch.float32)

    def post_solve(layer_idx, upd_matrix, adj_k, layer_ks, weight_name, state):
        current_weight = state.get("_current_weights", {}).get(weight_name)
        if current_weight is None:
            return upd_matrix

        import time
        t0 = time.perf_counter()

        w = current_weight.detach().to(dtype=svd_dtype, device=revive_svd_device)
        U, S, Vh = torch.linalg.svd(w, full_matrices=True)
        del w

        cumfrac = S.cumsum(dim=0) / S.sum()
        split_rank = int(torch.searchsorted(cumfrac, torch.tensor(revive_tau, device=cumfrac.device), right=False).item()) + 1
        split_rank = min(split_rank, S.numel())

        dev, dt = upd_matrix.device, upd_matrix.dtype
        upd_d = upd_matrix.double()

        # Use tail-only reconstruction to avoid full-matrix numerical noise
        # Instead of: U @ (zero top-k of U^T @ ΔW @ V^T) @ V
        # Use: U_tail @ (U_tail^T @ ΔW @ V_tail^T) @ V_tail
        U_tail = U[:, split_rank:].to(device=dev, dtype=torch.float64)
        Vh_tail = Vh[split_rank:, :].to(device=dev, dtype=torch.float64)
        del U, Vh

        coeff_tail = U_tail.T @ upd_d @ Vh_tail.T
        upd_safe = (U_tail @ coeff_tail @ Vh_tail).to(dtype=dt)
        del U_tail, Vh_tail, coeff_tail

        svd_time = time.perf_counter() - t0
        removed_frac = 1.0 - torch.linalg.norm(upd_safe).item() / max(torch.linalg.norm(upd_matrix).item(), 1e-10)
        print(f"  [REVIVE] layer={layer_idx} split_rank={split_rank}/{S.numel()} "
              f"removed={removed_frac:.1%} svd={svd_time:.1f}s")

        del S
        torch.cuda.empty_cache()
        return upd_safe

    return AlgorithmHooks(post_solve=post_solve)


def pathguard_hooks(
    pathguard_M: int = 200,
    pathguard_epsilon: float = 0.1,
    pathguard_adaptive: bool = True,
) -> AlgorithmHooks:
    """PathGuard: hazard-gated displacement-constrained editing."""

    def init_state():
        return {
            "key_history": {},
            "displacement_history": [],
            "epsilon": pathguard_epsilon,
            "pathguard_M": pathguard_M,
        }

    def post_update(layer_idx, weight_name, current_weight, state):
        state["displacement_history"].append({
            "layer": int(layer_idx),
            "weight_norm": torch.linalg.norm(current_weight).item(),
        })

    return AlgorithmHooks(
        init_state=init_state,
        post_update=post_update,
    )


def polykernel_hooks(
    kernel_type: str = "poly",
    kernel_degree: int = 2,
    kernel_sigma: str = "median",
) -> AlgorithmHooks:
    """Polykernel editor: kernel-weighted K@K^T for within-batch regularization."""

    def build_lhs(layer_idx, layer_ks, cov, hparams, state):
        alpha = hparams.mom2_update_weight
        K = layer_ks  # [d_in, n_batch]

        if kernel_type == "poly":
            G_lin = K.T @ K
            G_k = (1 + G_lin) ** kernel_degree - 1
            scale = G_lin.diag().mean() / max(G_k.diag().mean(), 1e-10)
            KKT = K @ (G_k * scale) @ K.T
        else:
            KKT = K @ K.T

        return alpha * cov.double() + KKT

    return AlgorithmHooks(build_lhs=build_lhs)


# Mapping from runner name to preset constructor
RUNNER_PRESETS = {
    "memit_sequential_runner": seqreg_hooks,
    "polykernel_seqreg_runner": seqreg_hooks,
    "pathguard_runner": pathguard_hooks,
    "polykernel_editor_runner": polykernel_hooks,
}
