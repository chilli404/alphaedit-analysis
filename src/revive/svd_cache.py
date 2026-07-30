"""
Persistent SVD cache for REVIVE.

Computes and caches compact SVD factors for pretrained weight matrices.
Cache is keyed by model identifier + parameter name + weight fingerprint.

Layout:
    {cache_dir}/{model_slug}/{param_slug}/svd.pt

Each cache file contains:
    - param_name: str
    - shape: tuple
    - model_id: str
    - fingerprint: str (SHA256 of first 1024 elements)
    - U: Tensor [m, r]
    - S: Tensor [r]
    - Vh: Tensor [r, n]
    - dtype: str
    - timestamp: str (ISO 8601)
    - version: int
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

CACHE_VERSION = 1


def _weight_fingerprint(w: Tensor, n_elements: int = 1024) -> str:
    """Compute a stable fingerprint from the first n_elements of a weight tensor."""
    flat = w.detach().flatten()[:n_elements].float().cpu()
    raw_bytes = flat.numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:32]


def _slugify(name: str) -> str:
    """Convert a parameter name to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _model_slug(model_id: str) -> str:
    """Convert model identifier to a short slug."""
    parts = model_id.replace("/", "--").replace("\\", "--")
    return _slugify(parts)


class SVDCache:
    """Manages persistent SVD caching for REVIVE."""

    def __init__(
        self,
        cache_dir: Path | str,
        model_id: str,
        svd_dtype: torch.dtype = torch.float32,
        svd_device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.svd_dtype = svd_dtype
        self.svd_device = svd_device
        self._memory_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    def _cache_path(self, param_name: str) -> Path:
        return (
            self.cache_dir
            / _model_slug(self.model_id)
            / _slugify(param_name)
            / "svd.pt"
        )

    def get_or_compute(
        self,
        param_name: str,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        """Get SVD factors from cache or compute and store.

        Args:
            param_name: Full parameter name (e.g., "model.layers.4.mlp.down_proj.weight")
            weight: The ORIGINAL pretrained weight tensor.

        Returns:
            (U, S, Vh, cache_hit): SVD factors and whether cache was used.

        Raises:
            ValueError: If weight contains non-finite values.
        """
        # Check memory cache first (validate shape matches)
        if param_name in self._memory_cache:
            U, S, Vh = self._memory_cache[param_name]
            m, n = weight.shape
            r = min(m, n)
            if U.shape == (m, r) and Vh.shape == (r, n):
                return U, S, Vh, True
            # Shape mismatch: evict stale entry
            del self._memory_cache[param_name]

        if not torch.isfinite(weight).all():
            raise ValueError(f"Weight {param_name} contains non-finite values")

        fingerprint = _weight_fingerprint(weight)
        cache_path = self._cache_path(param_name)

        # Try loading from disk
        if cache_path.exists():
            loaded = self._load_and_validate(cache_path, param_name, weight.shape, fingerprint)
            if loaded is not None:
                U, S, Vh = loaded
                self._memory_cache[param_name] = (U, S, Vh)
                return U, S, Vh, True

        # Compute SVD
        U, S, Vh = self._compute_svd(weight)

        # Save to disk
        self._save(cache_path, param_name, weight.shape, fingerprint, U, S, Vh)

        # Store in memory
        self._memory_cache[param_name] = (U, S, Vh)
        return U, S, Vh, False

    def _compute_svd(self, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute compact SVD of weight matrix."""
        w = weight.to(dtype=self.svd_dtype, device=self.svd_device)
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        # Keep on SVD device (typically CPU to save GPU memory)
        return U, S, Vh

    def _save(
        self,
        path: Path,
        param_name: str,
        shape: torch.Size,
        fingerprint: str,
        U: Tensor,
        S: Tensor,
        Vh: Tensor,
    ) -> None:
        """Save SVD to disk with metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "param_name": param_name,
                "shape": tuple(shape),
                "model_id": self.model_id,
                "fingerprint": fingerprint,
                "U": U,
                "S": S,
                "Vh": Vh,
                "dtype": str(self.svd_dtype),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": CACHE_VERSION,
            },
            str(path),
        )

    def _load_and_validate(
        self,
        path: Path,
        param_name: str,
        shape: torch.Size,
        fingerprint: str,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Load and validate cached SVD. Returns None if validation fails."""
        try:
            data = torch.load(str(path), map_location=self.svd_device, weights_only=False)
        except Exception:
            return None

        # Validate metadata
        if data.get("param_name") != param_name:
            return None
        if tuple(data.get("shape", ())) != tuple(shape):
            return None
        if data.get("model_id") != self.model_id:
            return None
        if data.get("fingerprint") != fingerprint:
            return None
        if data.get("version", 0) != CACHE_VERSION:
            return None

        U = data["U"]
        S = data["S"]
        Vh = data["Vh"]

        # Validate tensor shapes
        m, n = shape
        r = min(m, n)
        if U.shape != (m, r) or S.shape != (r,) or Vh.shape != (r, n):
            return None

        # Validate finite values
        if not (torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()):
            return None

        # Validate descending singular values
        if S.numel() > 1 and not (S[:-1] >= S[1:]).all():
            return None

        return U, S, Vh

    def precompute_all(
        self,
        param_weights: dict[str, Tensor],
    ) -> dict[str, bool]:
        """Precompute SVDs for all given parameters. Returns {name: cache_hit}."""
        results = {}
        for name, weight in param_weights.items():
            _, _, _, hit = self.get_or_compute(name, weight)
            results[name] = hit
        return results
