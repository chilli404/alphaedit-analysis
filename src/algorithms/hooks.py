"""Per-layer algorithm hooks for MEMIT and AlphaEdit.

Replaces the exec(compile(memit_main.py)) source injection pattern.
Each hook is called once per layer per batch during the algorithm's solve loop.

Hook points:
  1. build_lhs: BEFORE the solve — customize the LHS matrix
  2. post_solve: AFTER the solve — log metrics, apply REVIVE filter, cache keys
  3. post_update: AFTER weight update — capture displacement, eigenspectrum
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AlgorithmHooks:
    """Per-layer hooks injected into the MEMIT/AlphaEdit solve loop.

    All hooks receive a mutable `state` dict for accumulating metrics,
    caching keys, and tracking algorithm-specific state across layers and batches.
    """

    build_lhs: Optional[Callable] = None
    """(layer_idx, layer_ks, cov, hparams, state) -> lhs_matrix [d_in, d_in]

    Called BEFORE torch.linalg.solve. Constructs the LHS of the linear system.
    Default (no hook): hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T

    Use cases:
      - SeqReg: add lambda_prev * K_prev @ K_prev.T + lambda_delta * I
      - Polykernel: replace K@K.T with K @ diag(kernel_weights) @ K.T
      - PathGuard: add hazard-gated displacement protection terms
    """

    post_solve: Optional[Callable] = None
    """(layer_idx, upd_matrix, adj_k, layer_ks, weight_name, state) -> upd_matrix

    Called AFTER solve, BEFORE weight update. Can modify the update matrix.
    Return the (possibly modified) upd_matrix.

    Use cases:
      - REVIVE: spectral filter on upd_matrix using current weight's SVD
      - Logging: record update norm, K_prev overlap, condition number
      - Key caching: append current layer_ks to prev_cache
    """

    post_update: Optional[Callable] = None
    """(layer_idx, weight_name, current_weight, state) -> None

    Called AFTER weight update. Cannot modify the update — observation only.

    Use cases:
      - Update interference: capture W_after - W_before delta
      - Displacement tracking: measure how historical keys moved
      - Eigenspectrum: capture singular values of accumulated cache
    """

    init_state: Optional[Callable] = None
    """() -> dict

    Create the initial state dict. Called once before the first batch.
    The state dict is passed to all hooks and persists across layers within a batch.
    Reset between batches is the caller's responsibility.

    Default: empty dict.
    """

    def get_state(self) -> dict:
        if self.init_state is not None:
            return self.init_state()
        return {}


def compose_hooks(*hook_sets: AlgorithmHooks) -> AlgorithmHooks:
    """Compose multiple hook sets into one.

    For build_lhs: chains them (each modifies the LHS in sequence).
    For post_solve: chains them (each modifies upd_matrix in sequence).
    For post_update: calls all (observation only, no return value).
    For init_state: merges all state dicts.
    """
    def _chain_build_lhs(layer_idx, layer_ks, cov, hparams, state):
        alpha = getattr(hparams, 'mom2_update_weight', 1.0)
        if cov is not None:
            lhs = alpha * cov.double() + layer_ks @ layer_ks.T
        else:
            lhs = layer_ks @ layer_ks.T
        for hs in hook_sets:
            if hs.build_lhs is not None:
                lhs = hs.build_lhs(layer_idx, layer_ks, cov, hparams, state)
        return lhs

    def _chain_post_solve(layer_idx, upd_matrix, adj_k, layer_ks, weight_name, state):
        for hs in hook_sets:
            if hs.post_solve is not None:
                upd_matrix = hs.post_solve(layer_idx, upd_matrix, adj_k, layer_ks, weight_name, state)
        return upd_matrix

    def _chain_post_update(layer_idx, weight_name, current_weight, state):
        for hs in hook_sets:
            if hs.post_update is not None:
                hs.post_update(layer_idx, weight_name, current_weight, state)

    def _merge_init_state():
        merged = {}
        for hs in hook_sets:
            if hs.init_state is not None:
                merged.update(hs.init_state())
        return merged

    has_build = any(hs.build_lhs for hs in hook_sets)
    has_post_s = any(hs.post_solve for hs in hook_sets)
    has_post_u = any(hs.post_update for hs in hook_sets)
    has_init = any(hs.init_state for hs in hook_sets)

    return AlgorithmHooks(
        build_lhs=_chain_build_lhs if has_build else None,
        post_solve=_chain_post_solve if has_post_s else None,
        post_update=_chain_post_update if has_post_u else None,
        init_state=_merge_init_state if has_init else None,
    )
