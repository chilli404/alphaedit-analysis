#!/usr/bin/env python3
"""Compute signed displacement across multiple trajectories.

Downloads checkpoints from S3 one trajectory at a time, computes
signed/unsigned damage on first-1K edits, deletes checkpoints after.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
S3_CKPT = "s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit"
LAYER_KEY = "model.layers.6.mlp.down_proj.weight"


def download_checkpoint(ordering, seed, batch, dst):
    src = f"{S3_CKPT}/{ordering}/seed{seed}/batch_{batch}/model_weights.pt"
    result = subprocess.run(
        ["aws", "s3", "cp", src, dst, "--quiet"],
        capture_output=True, timeout=300,
    )
    return result.returncode == 0


def load_layer6_weight(path):
    weights = torch.load(path, map_location="cpu")
    if LAYER_KEY in weights:
        return weights[LAYER_KEY].float()
    for k, v in weights.items():
        if "layers.6" in k and "down_proj" in k:
            return v.float()
    raise KeyError(f"Layer 6 weight not found in {path}")


def compute_signed_for_trajectory(ordering, seed, keys, stream_cids, cid_to_kidx):
    """Download checkpoints, compute signed displacement, clean up."""
    tmp_dir = Path("/tmp/signed_ckpt")
    tmp_dir.mkdir(exist_ok=True)

    batches_needed = [9, 49, 99]
    weights = {}

    for batch in batches_needed:
        dst = str(tmp_dir / f"b{batch}.pt")
        ok = download_checkpoint(ordering, seed, batch, dst)
        if not ok:
            print(f"    FAILED to download batch_{batch}")
            # Clean up
            for f in tmp_dir.glob("*.pt"):
                f.unlink()
            return None
        weights[batch] = load_layer6_weight(dst)
        os.unlink(dst)

    # First 1K edits = positions 0-999 in the stream
    first_1k_cids = stream_cids[:1000]
    first_1k_kidx = [cid_to_kidx.get(cid, -1) for cid in first_1k_cids]
    valid = [(i, kidx) for i, kidx in enumerate(first_1k_kidx) if kidx >= 0]

    if not valid:
        return None

    valid_indices = [v[0] for v in valid]
    valid_kidx = [v[1] for v in valid]
    K = keys[valid_kidx]  # [n_valid, d_model]

    # Installation reference: W at batch_9
    W9 = weights[9]
    W49 = weights[49]
    W99 = weights[99]

    # Installation direction for each first-1K edit
    # e_i = W_9 @ k_i (representation after installation)
    e = (W9 @ K.T).T  # [n_valid, d_out]

    # Displacements
    dW_mid = W49 - W9   # batches 10-49
    dW_late = W99 - W49  # batches 50-99
    dW_total = W99 - W9  # batches 10-99

    disp_mid = (dW_mid @ K.T).T
    disp_late = (dW_late @ K.T).T
    disp_total = (dW_total @ K.T).T

    def compute_metrics(displacement, installation_dir):
        unsigned = torch.linalg.norm(displacement, dim=1).numpy()
        e_norm = torch.linalg.norm(installation_dir, dim=1, keepdim=True)
        d_norm = torch.linalg.norm(displacement, dim=1, keepdim=True)
        cos = (displacement * installation_dir).sum(dim=1) / (
            e_norm.squeeze() * d_norm.squeeze() + 1e-10
        )
        cos = cos.numpy()
        signed = np.maximum(0, -cos) * unsigned  # positive = destructive
        return {
            "mean_unsigned": float(unsigned.mean()),
            "mean_signed": float(signed.mean()),
            "mean_alignment": float(cos.mean()),
            "frac_opposing": float((cos < 0).mean()),
            "cancellation_factor": float(signed.sum() / (unsigned.sum() + 1e-10)),
        }

    result = {
        "mid_1K_5K": compute_metrics(disp_mid, e),
        "late_5K_10K": compute_metrics(disp_late, e),
        "total_1K_10K": compute_metrics(disp_total, e),
        "n_valid": len(valid),
    }

    # Net displacement (for R_t ratio)
    net_disp_norm = torch.linalg.norm(disp_total, dim=1).numpy()
    path_disp_norm = (
        torch.linalg.norm(disp_mid, dim=1).numpy()
        + torch.linalg.norm(disp_late, dim=1).numpy()
    )
    result["net_to_path_ratio"] = float(
        net_disp_norm.mean() / (path_disp_norm.mean() + 1e-10)
    )

    # Clean up any remaining files
    for f in tmp_dir.glob("*.pt"):
        f.unlink()

    return result


def main():
    # Load keys
    keys_path = PROJECT_ROOT / "results" / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"
    npz = np.load(keys_path)
    all_keys = torch.from_numpy(npz["keys"]).float()
    all_cids = npz["case_ids"].tolist()
    cid_to_kidx = {int(cid): i for i, cid in enumerate(all_cids)}

    conditions = []
    for ordering in ["fb_high_exposure", "fb_low_exposure", "fb_random0"]:
        for seed in [42, 2024, 137]:
            conditions.append((ordering, seed))

    # Load endpoint efficacy
    efficacy = {}
    for ordering, seed in conditions:
        f = PROJECT_ROOT / "results" / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}" / f"full_eval_seed{seed}.json"
        if f.exists():
            d = json.load(open(f))
            if "10000_edits" in d:
                efficacy[(ordering, seed)] = {
                    "all": d["10000_edits"]["all_facts"]["efficacy"],
                    "first_1k": d["10000_edits"].get("first_1k", {}).get("efficacy"),
                }

    results = {}
    for ordering, seed in conditions:
        key = f"seed{seed}_{ordering}"
        print(f"Processing {ordering}/seed{seed}...")

        # Load stream
        stream_path = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
        if not stream_path.exists():
            print(f"  Stream not found, skipping")
            continue

        stream = json.load(open(stream_path))
        stream_cids = [r["case_id"] for r in stream]

        signed = compute_signed_for_trajectory(
            ordering, seed, all_keys, stream_cids, cid_to_kidx
        )
        if signed is None:
            print(f"  Failed, skipping")
            continue

        eff = efficacy.get((ordering, seed), {})
        signed["endpoint_efficacy_all"] = eff.get("all")
        signed["endpoint_first_1k"] = eff.get("first_1k")
        signed["ordering"] = ordering
        signed["seed"] = seed

        results[key] = signed
        total = signed["total_1K_10K"]
        print(f"  unsigned={total['mean_unsigned']:.4f}, "
              f"signed={total['mean_signed']:.4f}, "
              f"cancel={total['cancellation_factor']:.3f}, "
              f"first_1k={eff.get('first_1k', '?')}")

    # Save
    out_path = PROJECT_ROOT / "results" / "signed_all_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Correlation analysis
    print("\n=== CORRELATION: cancellation_factor vs first_1k_efficacy ===")
    xs, ys, labels = [], [], []
    for key, r in results.items():
        cf = r["total_1K_10K"]["cancellation_factor"]
        f1k = r.get("endpoint_first_1k")
        if f1k is not None:
            xs.append(cf)
            ys.append(f1k)
            labels.append(key)

    if len(xs) >= 3:
        xs, ys = np.array(xs), np.array(ys)
        from scipy.stats import spearmanr, pearsonr
        rho_s, p_s = spearmanr(xs, ys)
        rho_p, p_p = pearsonr(xs, ys)
        print(f"  n = {len(xs)}")
        print(f"  Spearman: r={rho_s:.3f}, p={p_s:.4f}")
        print(f"  Pearson:  r={rho_p:.3f}, p={p_p:.4f}")

        # Also check signed_damage vs first_1k
        sds = [results[l]["total_1K_10K"]["mean_signed"] for l in labels]
        rho_sd, p_sd = spearmanr(sds, ys)
        print(f"\n  signed_damage vs first_1k:")
        print(f"  Spearman: r={rho_sd:.3f}, p={p_sd:.4f}")

        # And unsigned vs first_1k
        uds = [results[l]["total_1K_10K"]["mean_unsigned"] for l in labels]
        rho_ud, p_ud = spearmanr(uds, ys)
        print(f"\n  unsigned_damage vs first_1k:")
        print(f"  Spearman: r={rho_ud:.3f}, p={p_ud:.4f}")

        print(f"\n  Per-trajectory detail:")
        for label, cf, sd, ud, f1k in sorted(
            zip(labels, xs, sds, uds, ys), key=lambda x: x[-1]
        ):
            print(f"    {label:40s}: cancel={cf:.3f}, signed={sd:.4f}, unsigned={ud:.4f}, first_1k={f1k:.3f}")


if __name__ == "__main__":
    main()
