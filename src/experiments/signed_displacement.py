#!/usr/bin/env python3
"""Signed displacement analysis: explains seed 137 anomaly.

Computes whether later edits systematically oppose or reinforce the
installation direction of earlier edits, using checkpoint weight deltas.

The hypothesis: seeds 42/2024 collapse under high-exposure because
concentrated batches create updates that destructively oppose earlier
installations. Seed 137 survives because the signed effects cancel.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_layer6_weight(ckpt_path: str) -> torch.Tensor:
    """Load only layer 6 down_proj weight from a checkpoint."""
    weights = torch.load(ckpt_path, map_location="cpu")
    for key in weights:
        if "layers.6" in key and "down_proj" in key:
            return weights[key].float()
    raise KeyError(f"Layer 6 down_proj not found in {ckpt_path}")


def analyze_seed(seed: int, ordering: str, keys: np.ndarray,
                 stream_case_ids: list, key_cid_to_idx: dict,
                 ckpt_template: str):
    """Analyze signed displacement for one seed × ordering."""

    batches_to_load = [9, 49, 99]
    W = {}
    for b in batches_to_load:
        path = ckpt_template.format(seed=seed, ordering=ordering, batch=b)
        if not Path(path).exists():
            print(f"  SKIP: {path} not found")
            return None
        W[b] = load_layer6_weight(path)
        print(f"  Loaded batch {b}: {W[b].shape}")

    # Compute interval deltas
    # Interval A: batch 9→49 (edits 1000→5000, the middle period)
    # Interval B: batch 49→99 (edits 5000→10000, the late period)
    delta_A = W[49] - W[9]   # effect of batches 10-49
    delta_B = W[99] - W[49]  # effect of batches 50-99
    delta_total = W[99] - W[9]  # total effect after installation

    # First 1K edits (batch 0-9): installed by batch_9 checkpoint
    first_1k_cids = stream_case_ids[:1000]
    first_1k_keys = []
    for cid in first_1k_cids:
        idx = key_cid_to_idx.get(cid)
        if idx is not None:
            first_1k_keys.append(keys[idx])
    K_first1k = torch.from_numpy(np.array(first_1k_keys)).float()  # [~1000, 14336]
    print(f"  First-1K keys: {K_first1k.shape}")

    # Installation direction for first-1K edits
    # e_i = W_batch9 @ k_i (the representation at installation time)
    # This is an approximation - ideally we'd have W_before_install
    install_repr = W[9] @ K_first1k.T  # [4096, ~1000]

    results = {}
    for interval_name, delta in [("mid_1K_5K", delta_A), ("late_5K_10K", delta_B), ("total_1K_10K", delta_total)]:
        # Displacement of each first-1K key under this interval's delta
        displacement = delta @ K_first1k.T  # [4096, ~1000]

        # Unsigned magnitude per edit
        unsigned_mag = torch.linalg.norm(displacement, dim=0)  # [~1000]

        # Signed alignment with installation direction
        # cos(displacement_i, install_repr_i)
        cos_align = torch.nn.functional.cosine_similarity(
            displacement, install_repr, dim=0
        )  # [~1000]

        # Signed damage = -alignment * magnitude
        # Negative alignment (opposing installation) → positive signed damage
        signed_damage = -cos_align * unsigned_mag

        # Cancellation factor: |mean(signed)| / mean(unsigned)
        # Low = lots of cancellation, High = consistent direction
        mean_signed = signed_damage.mean().item()
        mean_unsigned = unsigned_mag.mean().item()
        cancellation = abs(mean_signed) / (mean_unsigned + 1e-10)

        results[interval_name] = {
            "mean_unsigned_damage": round(mean_unsigned, 6),
            "mean_signed_damage": round(mean_signed, 6),
            "cancellation_factor": round(cancellation, 6),
            "mean_cos_alignment": round(cos_align.mean().item(), 6),
            "frac_opposing": round((cos_align < 0).float().mean().item(), 4),
            "frac_reinforcing": round((cos_align > 0).float().mean().item(), 4),
            "std_cos_alignment": round(cos_align.std().item(), 6),
            "delta_fro": round(torch.linalg.norm(delta).item(), 4),
        }

        print(f"  {interval_name}: unsigned={mean_unsigned:.4f}, signed={mean_signed:.4f}, "
              f"cancel={cancellation:.3f}, frac_opposing={results[interval_name]['frac_opposing']:.3f}")

    # Clean up
    del W, delta_A, delta_B, delta_total, displacement, install_repr
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    print("=" * 60)
    print("Signed Displacement Analysis")
    print("=" * 60)

    # Load key vectors
    keys_path = PROJECT_ROOT / "results" / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"
    npz = np.load(keys_path)
    keys = npz["keys"]
    key_cids = npz["case_ids"].tolist()
    key_cid_to_idx = {int(cid): i for i, cid in enumerate(key_cids)}
    print(f"Loaded keys: {keys.shape}")

    ckpt_template = "/tmp/ckpt_fb_high_s{seed}_b{batch}.pt"

    all_results = {}
    for seed in [42, 137]:
        for ordering in ["fb_high_exposure"]:
            print(f"\n--- Seed {seed}, {ordering} ---")

            # Load stream
            stream_path = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
            with open(stream_path) as f:
                stream = json.load(f)
            stream_cids = [r["case_id"] for r in stream]

            result = analyze_seed(
                seed, ordering, keys, stream_cids, key_cid_to_idx, ckpt_template
            )
            if result:
                all_results[f"seed{seed}_{ordering}"] = result

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Why does seed 137 survive?")
    print("=" * 60)

    for interval in ["mid_1K_5K", "late_5K_10K", "total_1K_10K"]:
        print(f"\n  {interval}:")
        for seed in [42, 137]:
            key = f"seed{seed}_fb_high_exposure"
            if key in all_results and interval in all_results[key]:
                r = all_results[key][interval]
                print(f"    Seed {seed}: unsigned={r['mean_unsigned_damage']:.4f}, "
                      f"signed={r['mean_signed_damage']:.4f}, "
                      f"cancel={r['cancellation_factor']:.3f}, "
                      f"opposing={r['frac_opposing']:.1%}")

    # Save
    out_path = PROJECT_ROOT / "results" / "signed_displacement_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
