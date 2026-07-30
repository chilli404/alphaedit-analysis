"""
REVIVE spectral subspace filter for knowledge editing updates.

Given a pretrained weight W0 with compact SVD: W0 = U @ diag(S) @ Vh,
REVIVE removes components of a proposed update delta_W that align with the
dominant spectral directions of W0.

The protected rank k is the smallest integer satisfying:
    cumsum(S)[k-1] / sum(S) >= tau

The filtered update retains only components in tail singular directions:
    delta_w_safe = U_tail @ (U_tail.T @ delta_w_raw @ V_tail) @ V_tail.T

where U_tail = U[:, k:] and V_tail = Vh[k:, :].T

This formulation:
- Never materializes full m×m or n×n projection matrices
- Correctly handles rectangular matrices (m != n)
- For compact SVD with r = min(m,n): components outside the SVD basis
  (null space of W0.T for m < n) are implicitly removed

Reference: REVIVE paper (spectral subspace protection for model editing).
The paper defines energy using sum of singular values (not squared).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ReviveMetrics:
    """Diagnostics from a single REVIVE filter application."""

    param_name: str
    layer: int
    batch: int
    tau: float
    k: int
    r: int
    k_fraction: float
    protected_energy_fraction: float
    raw_norm_fro: float
    safe_norm_fro: float
    removed_norm_fro: float
    removed_fraction: float
    raw_safe_cosine: float
    raw_spectral_norm: float
    safe_spectral_norm: float
    left_overlap: float
    right_overlap: float
    dominant_block_overlap: float
    filter_time_ms: float
    had_nan_or_inf: bool

    def to_dict(self) -> dict:
        return {
            "phase": "revive",
            "param_name": self.param_name,
            "layer": self.layer,
            "batch": self.batch,
            "tau": self.tau,
            "k": self.k,
            "r": self.r,
            "k_fraction": round(self.k_fraction, 4),
            "protected_energy_fraction": round(self.protected_energy_fraction, 4),
            "raw_norm_fro": round(self.raw_norm_fro, 6),
            "safe_norm_fro": round(self.safe_norm_fro, 6),
            "removed_norm_fro": round(self.removed_norm_fro, 6),
            "removed_fraction": round(self.removed_fraction, 6),
            "raw_safe_cosine": round(self.raw_safe_cosine, 6),
            "raw_spectral_norm": round(self.raw_spectral_norm, 6),
            "safe_spectral_norm": round(self.safe_spectral_norm, 6),
            "left_overlap": round(self.left_overlap, 6),
            "right_overlap": round(self.right_overlap, 6),
            "dominant_block_overlap": round(self.dominant_block_overlap, 6),
            "filter_time_ms": round(self.filter_time_ms, 3),
            "had_nan_or_inf": self.had_nan_or_inf,
        }


def compute_protected_rank(S: Tensor, tau: float) -> int:
    """Find smallest k where cumsum(S)[:k] / sum(S) >= tau.

    Args:
        S: Singular values in descending order, shape [r].
        tau: Energy threshold in (0, 1).

    Returns:
        k: Protected rank (1 <= k < r).

    Raises:
        ValueError: If tau is not in (0, 1) or S is empty/invalid.
    """
    if tau <= 0.0 or tau >= 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if S.numel() == 0:
        raise ValueError("S must not be empty")
    if not torch.all(S >= 0):
        raise ValueError("Singular values must be non-negative")

    total = S.sum()
    if total <= 0:
        raise ValueError("Sum of singular values must be positive")

    cumulative = S.cumsum(dim=0)
    # Find first index where cumulative / total >= tau
    mask = cumulative >= tau * total
    if not mask.any():
        # tau is so high that even all singular values don't reach it
        # Return r-1 (protect all but last)
        return max(1, S.numel() - 1)

    k = int(mask.nonzero(as_tuple=False)[0].item()) + 1  # 1-indexed count
    # Clamp: k must be at least 1 and less than r
    k = max(1, min(k, S.numel() - 1))
    return k


def revive_filter(
    delta_w_raw: Tensor,
    U: Tensor,
    S: Tensor,
    Vh: Tensor,
    tau: float,
    *,
    param_name: str = "",
    layer: int = 0,
    batch: int = 0,
    compute_metrics: bool = True,
    compute_spectral_norm: bool = True,
) -> tuple[Tensor, ReviveMetrics | None]:
    """Apply REVIVE spectral subspace filter to a proposed weight update.

    Removes components of delta_w_raw that align with the top-k singular
    directions of the original weight matrix.

    Args:
        delta_w_raw: Proposed update, shape [m, n]. NOT modified in place.
        U: Left singular vectors of W0, shape [m, r].
        S: Singular values of W0, shape [r], descending.
        Vh: Right singular vectors of W0, shape [r, n].
        tau: Energy threshold for protected rank.
        param_name: For logging.
        layer: For logging.
        batch: For logging.
        compute_metrics: Whether to compute detailed diagnostics.
        compute_spectral_norm: Whether to compute spectral norms (expensive).

    Returns:
        (delta_w_safe, metrics): Filtered update and optional diagnostics.

    Raises:
        ValueError: On shape mismatch, invalid tau, non-finite inputs.
    """
    t0 = time.perf_counter()

    # --- Input validation ---
    m, n = delta_w_raw.shape
    r = S.numel()

    if U.shape != (m, r):
        raise ValueError(
            f"U shape {U.shape} doesn't match expected ({m}, {r}) "
            f"for delta_w_raw shape {delta_w_raw.shape}"
        )
    if Vh.shape != (r, n):
        raise ValueError(
            f"Vh shape {Vh.shape} doesn't match expected ({r}, {n}) "
            f"for delta_w_raw shape {delta_w_raw.shape}"
        )
    if not torch.isfinite(delta_w_raw).all():
        raise ValueError("delta_w_raw contains non-finite values")

    # --- Compute protected rank ---
    k = compute_protected_rank(S, tau)

    # --- Apply filter: retain only tail components ---
    # U_tail: [m, r-k], V_tail: [n, r-k]
    U_tail = U[:, k:]  # [m, r-k]
    Vh_tail = Vh[k:, :]  # [r-k, n]

    # Memory-efficient: compute A_tail = U_tail.T @ delta_w_raw @ V_tail.T
    # where V_tail = Vh_tail.T, so delta_w_raw @ V_tail = delta_w_raw @ Vh_tail.T
    # A_tail shape: [r-k, r-k]
    # Then reconstruct: delta_w_safe = U_tail @ A_tail @ Vh_tail

    # Step 1: project delta onto tail left subspace: [r-k, n]
    left_proj = U_tail.T @ delta_w_raw  # [r-k, n]
    # Step 2: project onto tail right subspace: [r-k, r-k]
    A_tail = left_proj @ Vh_tail.T  # [r-k, r-k]
    # Step 3: reconstruct in original space
    delta_w_safe = U_tail @ A_tail @ Vh_tail  # [m, n]

    # Cast back to original dtype if needed
    delta_w_safe = delta_w_safe.to(delta_w_raw.dtype)

    t1 = time.perf_counter()

    # --- Metrics ---
    metrics = None
    if compute_metrics:
        had_nan = not torch.isfinite(delta_w_safe).all()

        eps = 1e-12
        raw_fro = torch.linalg.norm(delta_w_raw, ord="fro").item()
        safe_fro = torch.linalg.norm(delta_w_safe, ord="fro").item()
        removed = delta_w_raw - delta_w_safe
        removed_fro = torch.linalg.norm(removed, ord="fro").item()
        removed_frac = removed_fro / (raw_fro + eps)

        # Cosine similarity
        inner = (delta_w_raw * delta_w_safe).sum().item()
        cos_sim = inner / (raw_fro * safe_fro + eps)

        # Spectral norms (optional, expensive)
        raw_spec = 0.0
        safe_spec = 0.0
        if compute_spectral_norm:
            raw_spec = torch.linalg.norm(delta_w_raw, ord=2).item()
            safe_spec = torch.linalg.norm(delta_w_safe, ord=2).item()

        # Overlap metrics: how much of raw update is in dominant subspaces
        U_top = U[:, :k]  # [m, k]
        Vh_top = Vh[:k, :]  # [k, n]

        # Left overlap: ||U_top.T @ delta_w_raw||_F / ||delta_w_raw||_F
        left_proj_top = U_top.T @ delta_w_raw  # [k, n]
        left_overlap = torch.linalg.norm(left_proj_top, ord="fro").item() / (raw_fro + eps)

        # Right overlap: ||delta_w_raw @ Vh_top.T||_F / ||delta_w_raw||_F
        right_proj_top = delta_w_raw @ Vh_top.T  # [m, k]
        right_overlap = torch.linalg.norm(right_proj_top, ord="fro").item() / (raw_fro + eps)

        # Dominant-dominant block overlap: ||U_top.T @ delta_w_raw @ Vh_top.T||_F / ||delta_w_raw||_F
        dom_block = U_top.T @ delta_w_raw @ Vh_top.T  # [k, k]
        dom_overlap = torch.linalg.norm(dom_block, ord="fro").item() / (raw_fro + eps)

        # Protected energy fraction
        protected_energy = S[:k].sum().item() / (S.sum().item() + eps)

        metrics = ReviveMetrics(
            param_name=param_name,
            layer=layer,
            batch=batch,
            tau=tau,
            k=k,
            r=r,
            k_fraction=k / r,
            protected_energy_fraction=protected_energy,
            raw_norm_fro=raw_fro,
            safe_norm_fro=safe_fro,
            removed_norm_fro=removed_fro,
            removed_fraction=removed_frac,
            raw_safe_cosine=cos_sim,
            raw_spectral_norm=raw_spec,
            safe_spectral_norm=safe_spec,
            left_overlap=left_overlap,
            right_overlap=right_overlap,
            dominant_block_overlap=dom_overlap,
            filter_time_ms=(t1 - t0) * 1000,
            had_nan_or_inf=had_nan,
        )

    return delta_w_safe, metrics


def revive_filter_disabled(delta_w_raw: Tensor) -> tuple[Tensor, None]:
    """Identity pass-through when REVIVE is disabled."""
    return delta_w_raw, None
