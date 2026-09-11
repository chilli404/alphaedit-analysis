#!/usr/bin/env python3
"""
Generate a random orthogonal projector of specified rank for GPT-J-6B.

Creates a P matrix where the projection subspace is spanned by random
orthonormal vectors (via QR decomposition of a Gaussian matrix), rather
than the eigenspace of the covariance statistics.

This enables a controlled comparison: same rank, different orientation.
If AlphaEdit's performance depends on which directions are feasible
(not just how many), the eigenspace projector should outperform the
random one on preservation while matching on efficacy.

Usage:
    uv run python src/experiments/generate_random_projector.py --rank 3390 --seed 42
    uv run python src/experiments/generate_random_projector.py --rank 3390 --seed 42 --output data/stats/gpt-j-6b/wikipedia_stats/null_space_project_random_r3390_s42.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from paths import get_project_root

PROJECT_ROOT = get_project_root()
GPTJ_LAYERS = [3, 4, 5, 6, 7, 8]
GPTJ_DIM = 16384  # fc_out weight is [4096, 16384], keys are 16384-dim


def generate_random_projector(
    dim: int,
    rank: int,
    seed: int,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Generate a random orthogonal projector P of shape [dim, dim] with given rank.

    P = Q @ Q^T where Q is [dim, rank] with orthonormal columns
    obtained from QR decomposition of a random Gaussian matrix.
    """
    rng = np.random.default_rng(seed + layer_idx * 1000)
    G = rng.standard_normal((dim, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(G)  # Q is [dim, rank] orthonormal
    P = Q @ Q.T  # [dim, dim] symmetric idempotent
    return torch.from_numpy(P)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random orthogonal projector for GPT-J-6B"
    )
    parser.add_argument("--rank", type=int, required=True,
                        help="Rank of the projector (number of feasible dimensions)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for projector generation")
    parser.add_argument("--dim", type=int, default=GPTJ_DIM,
                        help=f"Ambient dimension (default: {GPTJ_DIM})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--per_layer_rank", type=int, nargs=6, default=None,
                        help="Per-layer ranks (6 values for layers 3-8). "
                             "If not provided, uses --rank for all layers.")
    args = parser.parse_args()

    if args.per_layer_rank:
        ranks = args.per_layer_rank
    else:
        ranks = [args.rank] * len(GPTJ_LAYERS)

    output_path = args.output
    if output_path is None:
        stats_dir = PROJECT_ROOT / "data" / "stats" / "gpt-j-6b" / "wikipedia_stats"
        output_path = stats_dir / f"null_space_project_random_r{args.rank}_s{args.seed}.pt"
    else:
        output_path = Path(output_path)

    print(f"Generating random projector:")
    print(f"  Dimension: {args.dim}")
    print(f"  Ranks: {ranks}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_path}")
    print()

    P_layers = []
    for layer_idx, (layer_num, rank) in enumerate(zip(GPTJ_LAYERS, ranks)):
        assert 0 < rank < args.dim, f"Rank must be in (0, {args.dim}), got {rank}"
        P = generate_random_projector(args.dim, rank, args.seed, layer_idx)

        # Verify properties
        trace = torch.trace(P).item()
        idempotent_err = torch.norm(P @ P - P).item()
        symmetric_err = torch.norm(P - P.T).item()

        print(f"  Layer {layer_num}: rank={rank}, trace={trace:.1f}, "
              f"idempotent_err={idempotent_err:.2e}, symmetric_err={symmetric_err:.2e}")

        P_layers.append(P)

    P_full = torch.stack(P_layers)  # [6, dim, dim]
    print(f"\n  Final shape: {P_full.shape}")
    print(f"  Saving to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(P_full, output_path)
    print("  Done.")


if __name__ == "__main__":
    main()
