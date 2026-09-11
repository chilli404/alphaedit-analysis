#!/usr/bin/env python3
"""Decompose global update norm vs targeted historical-key displacement.

Tests whether forgetting is driven by generic norm growth (less novel, ENCORE
already showed this) or targeted interference on previously-edited keys
(the novel mechanism).

Computes for each trajectory:
  N  = global update norm ||ΔW||_F
  D  = mean historical displacement ||ΔW @ k_i|| over first-1K edits
  T  = D / (N × mean(||k_i||))  (normalized targeted transfer)
  D_unrelated = mean displacement on unrelated (unedited) keys

Then tests:
  1. retention ~ N  (global norm alone)
  2. retention ~ D  (historical displacement alone)
  3. retention ~ N + D  (does D add beyond N?)
  4. retention ~ T  (normalized transfer)
  5. D vs D_unrelated  (is displacement targeted?)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def load_trajectory_data():
    """Load the 9-trajectory signed displacement data."""
    with open(RESULTS / "signed_all_trajectories.json") as f:
        raw = json.load(f)

    trajectories = []
    for key, val in raw.items():
        total = val.get("total_1K_10K", {})
        trajectories.append({
            "label": key,
            "ordering": val.get("ordering", key.split("_", 1)[1] if "_" in key else key),
            "seed": val.get("seed", int(key.split("seed")[1].split("_")[0]) if "seed" in key else 0),
            "D_historical": total.get("mean_unsigned", 0.0),
            "signed_damage": total.get("mean_signed", 0.0),
            "cancellation_factor": total.get("cancellation_factor", 0.0),
            "alignment": total.get("mean_alignment", 0.0),
            "efficacy_all": val.get("endpoint_efficacy_all", 0.0),
            "first_1k": val.get("endpoint_first_1k", 0.0),
        })
    return trajectories


def compute_global_norms_and_unrelated(trajectories):
    """Download minimal checkpoints and compute global norm + unrelated displacement.

    Downloads batch_9 and batch_99 for each trajectory, computes:
    - N = ||W_99 - W_9||_F for layer 6
    - D_unrelated = mean ||ΔW @ k_j|| for 1000 unrelated keys
    """
    S3_CKPT = "s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit"

    # Load keys
    keys_data = np.load(str(RESULTS / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"))
    all_keys = keys_data["keys"]  # (20877, 14336)
    all_case_ids = set(keys_data["case_ids"].tolist())

    results = []
    for traj in trajectories:
        ordering = traj["ordering"]
        seed = traj["seed"]

        # Load stream to find edited case_ids
        stream_path = RESULTS / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
        if not stream_path.exists():
            print(f"  SKIP {traj['label']}: stream not found")
            results.append({"N": None, "D_unrelated": None})
            continue

        with open(stream_path) as f:
            stream = json.load(f)
        edited_cids = set(r["case_id"] for r in stream[:1000])

        # Download checkpoints
        ckpt_b9 = f"/tmp/norm_ckpt_{ordering}_s{seed}_b9.pt"
        ckpt_b99 = f"/tmp/norm_ckpt_{ordering}_s{seed}_b99.pt"

        for batch, dst in [(9, ckpt_b9), (99, ckpt_b99)]:
            src = f"{S3_CKPT}/{ordering}/seed{seed}/batch_{batch}/model_weights.pt"
            subprocess.run(["aws", "s3", "cp", src, dst, "--quiet"],
                           capture_output=True, timeout=120)

        try:
            import torch
            w9 = torch.load(ckpt_b9, map_location="cpu")
            w99 = torch.load(ckpt_b99, map_location="cpu")

            # Layer 6 weight key
            layer6_key = None
            for k in w9.keys():
                if "layers.6" in k and "down_proj" in k:
                    layer6_key = k
                    break

            if layer6_key is None:
                print(f"  SKIP {traj['label']}: layer 6 key not found")
                results.append({"N": None, "D_unrelated": None})
                continue

            delta_w = (w99[layer6_key].float() - w9[layer6_key].float()).numpy()

            # Global norm
            N = float(np.linalg.norm(delta_w))

            # Unrelated key displacement
            # Select 1000 keys NOT in the edited stream
            unrelated_indices = [i for i, cid in enumerate(keys_data["case_ids"])
                                 if int(cid) not in edited_cids]
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(unrelated_indices, size=min(1000, len(unrelated_indices)),
                                    replace=False)
            unrelated_keys = all_keys[sample_idx]

            effects_unrelated = delta_w @ unrelated_keys.T
            D_unrelated = float(np.linalg.norm(effects_unrelated, axis=0).mean())

            # Also compute D_historical from the same ΔW for verification
            edited_indices = [i for i, cid in enumerate(keys_data["case_ids"])
                              if int(cid) in edited_cids][:1000]
            edited_keys = all_keys[edited_indices]
            effects_edited = delta_w @ edited_keys.T
            D_historical_check = float(np.linalg.norm(effects_edited, axis=0).mean())

            results.append({
                "N": N,
                "D_unrelated": D_unrelated,
                "D_historical_recomputed": D_historical_check,
            })
            print(f"  {traj['label']}: N={N:.4f}, D_unrelated={D_unrelated:.4f}, "
                  f"D_hist={D_historical_check:.4f}")

        except Exception as e:
            print(f"  ERROR {traj['label']}: {e}")
            results.append({"N": None, "D_unrelated": None})
        finally:
            # Clean up
            for f in [ckpt_b9, ckpt_b99]:
                Path(f).unlink(missing_ok=True)

    return results


def analyze(trajectories, norms):
    """Run the decomposition analysis."""
    # Merge
    for traj, norm_data in zip(trajectories, norms):
        traj.update(norm_data)

    # Filter to trajectories with complete data
    complete = [t for t in trajectories if t.get("N") is not None]

    if len(complete) < 5:
        print(f"\nWARNING: Only {len(complete)} trajectories with complete data. "
              "Results may not be reliable.")

    labels = [t["label"] for t in complete]
    D = np.array([t["D_historical"] for t in complete])
    N = np.array([t["N"] for t in complete])
    first_1k = np.array([t["first_1k"] for t in complete])
    efficacy = np.array([t["efficacy_all"] for t in complete])
    D_unrel = np.array([t.get("D_unrelated", 0) for t in complete])
    seeds = np.array([t["seed"] for t in complete])

    # Mean key norms for normalization
    keys_data = np.load(str(RESULTS / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"))
    mean_key_norm = float(np.linalg.norm(keys_data["keys"][:1000], axis=1).mean())
    T = D / (N * mean_key_norm + 1e-10)  # Normalized targeted transfer

    print("\n" + "=" * 70)
    print("NORM vs DISPLACEMENT DECOMPOSITION")
    print("=" * 70)

    print(f"\n{'Label':40s} {'N':>8s} {'D_hist':>8s} {'D_unrel':>8s} {'T':>8s} {'1st1K':>8s}")
    print("-" * 80)
    for i, t in enumerate(complete):
        print(f"{t['label']:40s} {N[i]:8.4f} {D[i]:8.4f} {D_unrel[i]:8.4f} {T[i]:8.6f} {first_1k[i]:8.3f}")

    # 1. Spearman correlations with first_1k retention
    print(f"\n--- Spearman correlations with first_1k retention ---")
    for name, vals in [("N (global norm)", N), ("D (hist displacement)", D),
                       ("T (normalized transfer)", T), ("D_unrelated", D_unrel),
                       ("Signed damage", [t["signed_damage"] for t in complete]),
                       ("Cancellation factor", [t["cancellation_factor"] for t in complete])]:
        r, p = stats.spearmanr(vals, first_1k)
        print(f"  {name:30s}: r={r:+.3f}, p={p:.4f}")

    # 2. Is displacement targeted? D_historical vs D_unrelated
    print(f"\n--- Is displacement targeted? ---")
    ratio = D / (D_unrel + 1e-10)
    print(f"  Mean D_hist / D_unrel ratio: {ratio.mean():.3f} ± {ratio.std():.3f}")
    print(f"  Per trajectory:")
    for i, t in enumerate(complete):
        print(f"    {t['label']:35s}: D_hist={D[i]:.4f}, D_unrel={D_unrel[i]:.4f}, ratio={ratio[i]:.3f}")

    if ratio.mean() > 1.2:
        print(f"  → Displacement IS targeted: edited keys displaced {ratio.mean():.1f}× more than unrelated")
    else:
        print(f"  → Displacement is NOT strongly targeted: ratio only {ratio.mean():.2f}×")

    # 3. Does D add beyond N? (partial correlation)
    print(f"\n--- Does D add beyond N? ---")
    # Partial correlation: D vs first_1k, controlling for N
    from scipy.stats import pearsonr
    # Residualize D on N
    slope_dn = np.polyfit(N, D, 1)
    D_resid = D - np.polyval(slope_dn, N)
    slope_fn = np.polyfit(N, first_1k, 1)
    f1k_resid = first_1k - np.polyval(slope_fn, N)
    r_partial, p_partial = pearsonr(D_resid, f1k_resid)
    print(f"  Partial corr (D | N) vs first_1k: r={r_partial:+.3f}, p={p_partial:.4f}")

    # Partial correlation: N vs first_1k, controlling for D
    slope_nd = np.polyfit(D, N, 1)
    N_resid = N - np.polyval(slope_nd, D)
    slope_fd = np.polyfit(D, first_1k, 1)
    f1k_resid2 = first_1k - np.polyval(slope_fd, D)
    r_partial2, p_partial2 = pearsonr(N_resid, f1k_resid2)
    print(f"  Partial corr (N | D) vs first_1k: r={r_partial2:+.3f}, p={p_partial2:.4f}")

    if abs(r_partial) > abs(r_partial2):
        print(f"  → D adds more beyond N than N adds beyond D")
    else:
        print(f"  → N adds more beyond D than D adds beyond N")

    # 4. Within-seed analysis
    print(f"\n--- Within-seed analysis ---")
    for seed in sorted(set(seeds)):
        mask = seeds == seed
        if mask.sum() < 3:
            continue
        r_d, _ = stats.spearmanr(D[mask], first_1k[mask])
        r_n, _ = stats.spearmanr(N[mask], first_1k[mask])
        print(f"  Seed {seed} (n={mask.sum()}): "
              f"D vs f1k r={r_d:+.3f}, N vs f1k r={r_n:+.3f}")

    # 5. Summary table for paper
    results_dict = {
        "n_trajectories": len(complete),
        "correlations": {
            "N_vs_first1k": {"spearman_r": float(stats.spearmanr(N, first_1k)[0]),
                             "p": float(stats.spearmanr(N, first_1k)[1])},
            "D_vs_first1k": {"spearman_r": float(stats.spearmanr(D, first_1k)[0]),
                             "p": float(stats.spearmanr(D, first_1k)[1])},
            "T_vs_first1k": {"spearman_r": float(stats.spearmanr(T, first_1k)[0]),
                             "p": float(stats.spearmanr(T, first_1k)[1])},
            "D_unrel_vs_first1k": {"spearman_r": float(stats.spearmanr(D_unrel, first_1k)[0]),
                                    "p": float(stats.spearmanr(D_unrel, first_1k)[1])},
        },
        "partial_correlations": {
            "D_given_N_vs_first1k": {"r": float(r_partial), "p": float(p_partial)},
            "N_given_D_vs_first1k": {"r": float(r_partial2), "p": float(p_partial2)},
        },
        "targeting": {
            "mean_D_hist_to_D_unrel_ratio": float(ratio.mean()),
            "std_ratio": float(ratio.std()),
            "is_targeted": bool(ratio.mean() > 1.2),
        },
        "per_trajectory": [
            {
                "label": t["label"],
                "seed": int(t["seed"]),
                "ordering": t["ordering"],
                "N": float(t["N"]) if t["N"] is not None else None,
                "D_historical": float(t["D_historical"]),
                "D_unrelated": float(t.get("D_unrelated", 0)),
                "T_normalized": float(T[i]),
                "D_hist_to_D_unrel": float(ratio[i]),
                "first_1k": float(t["first_1k"]),
                "efficacy_all": float(t["efficacy_all"]),
            }
            for i, t in enumerate(complete)
        ],
    }

    with open(RESULTS / "norm_vs_displacement.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {RESULTS / 'norm_vs_displacement.json'}")

    return results_dict


if __name__ == "__main__":
    print("Loading trajectory data...")
    trajectories = load_trajectory_data()
    print(f"  {len(trajectories)} trajectories loaded")

    print("\nComputing global norms and unrelated-key displacement...")
    norms = compute_global_norms_and_unrelated(trajectories)

    results = analyze(trajectories, norms)
