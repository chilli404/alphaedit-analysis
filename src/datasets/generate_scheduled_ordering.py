#!/usr/bin/env python3
"""Conditioning-aware interference scheduler for batch ordering.

Greedily builds a batch sequence that balances three objectives:
  1. Risk:    minimize damage to already-installed edits (key-cosine proxy)
  2. Spectral: avoid concentrating the cache in a narrow subspace
  3. Deferral: prevent indefinite postponement of high-overlap batches

Score for candidate batch B at step t:
    J_t(B) = Risk(B, S_t) + β · SpectralCost(H_t, B) + γ · DeferralCost(B, t)

The batch with the LOWEST J_t is selected at each step.

Produces orderings compatible with the run_matched_ordering.sh pipeline.

Usage:
    uv run python src/datasets/generate_scheduled_ordering.py --seed 42
    uv run python src/datasets/generate_scheduled_ordering.py --seed 42 --beta 1.0 --gamma 0.1
    uv run python src/datasets/generate_scheduled_ordering.py --seed 42 --preset all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Scoring Components ──────────────────────────────────────────────────────


def compute_batch_keys(batch_cids, keys, cid_to_kidx):
    """Extract and L2-normalize key matrix for a batch."""
    indices = [cid_to_kidx[cid] for cid in batch_cids if cid in cid_to_kidx]
    if not indices:
        return None
    batch_keys = keys[indices]
    norms = np.linalg.norm(batch_keys, axis=1, keepdims=True)
    return batch_keys / np.maximum(norms, 1e-8)


def risk_score(batch_normed, installed_normed):
    """Mean max-cosine from installed edits to this batch's keys.

    Proxy for η(i, B) — how much damage this batch would cause to
    installed edits through key-space overlap.
    """
    if installed_normed is None or len(installed_normed) == 0:
        return 0.0
    cos = installed_normed @ batch_normed.T  # [n_installed, n_batch]
    return float(cos.max(axis=1).mean())


def spectral_cost(prefix_keys, batch_keys_normed, max_prefix=500):
    """Effective-rank-based spectral cost using batch centroids.

    Uses per-batch centroids (already in prefix_keys when subsampled)
    to keep the Gram small.  When the prefix exceeds max_prefix rows,
    subsample uniformly to keep computation bounded.

    Returns: 1 − (effective_rank / numerical_rank), in [0, 1].
    Higher = more concentrated = worse.
    """
    if prefix_keys is None or len(prefix_keys) == 0:
        return 0.0
    if len(prefix_keys) > max_prefix:
        idx = np.linspace(0, len(prefix_keys) - 1, max_prefix, dtype=int)
        prefix_sub = prefix_keys[idx]
    else:
        prefix_sub = prefix_keys
    combined = np.concatenate([prefix_sub, batch_keys_normed], axis=0)
    G = combined @ combined.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) < 2:
        return 0.0
    p = eigvals / eigvals.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    num_rank = float(len(eigvals))
    return 1.0 - eff_rank / num_rank


def deferral_cost(batch_idx, step, n_batches):
    """Linear deferral penalty: grows with how long this batch has waited."""
    return step / max(n_batches, 1)


# ─── Greedy Scheduler ────────────────────────────────────────────────────────


def schedule_batches(
    batch_cids_list,
    keys_raw,
    cid_to_kidx,
    beta=1.0,
    gamma=0.1,
    use_risk=True,
    use_spectral=True,
    verbose=True,
):
    """Greedily schedule batches to minimize J_t.

    Returns ordered list of batch indices.
    """
    n_batches = len(batch_cids_list)

    # Precompute normalized + raw keys per batch
    batch_normed = []
    batch_raw = []
    for cids in batch_cids_list:
        indices = [cid_to_kidx[cid] for cid in cids if cid in cid_to_kidx]
        raw = keys_raw[indices]
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normed = raw / np.maximum(norms, 1e-8)
        batch_normed.append(normed)
        batch_raw.append(raw)

    # State
    order = []
    remaining = set(range(n_batches))
    installed_normed_list = []

    for step in range(n_batches):
        best_idx = None
        best_score = float("inf")

        # Rebuild installed_normed once per step
        if installed_normed_list:
            installed_normed = np.concatenate(installed_normed_list, axis=0)
        else:
            installed_normed = None

        for b in remaining:
            score = 0.0

            if use_risk:
                score += risk_score(batch_normed[b], installed_normed)

            if use_spectral and beta > 0:
                score += beta * spectral_cost(installed_normed, batch_normed[b])

            if gamma > 0:
                score += gamma * deferral_cost(b, step, n_batches)

            if score < best_score:
                best_score = score
                best_idx = b

        order.append(best_idx)
        remaining.remove(best_idx)

        # Update state
        installed_normed_list.append(batch_normed[best_idx])

        if verbose and (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{n_batches}: selected batch {best_idx}, "
                  f"J={best_score:.4f}")

    return order


# ─── Main ────────────────────────────────────────────────────────────────────


PRESETS = {
    "balanced": {"beta": 1.0, "gamma": 0.1, "use_risk": True, "use_spectral": True},
    "exposure_only": {"beta": 0.0, "gamma": 0.0, "use_risk": True, "use_spectral": False},
    "conditioning_only": {"beta": 1.0, "gamma": 0.0, "use_risk": False, "use_spectral": True},
}


def main():
    parser = argparse.ArgumentParser(
        description="Conditioning-aware interference scheduler"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for spectral/conditioning cost")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Weight for deferral penalty")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()) + ["all"],
                        help="Named preset (overrides beta/gamma)")
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--batch_assignment", type=str, default=None,
                        help="Path to fixed_batch_assignment JSON")
    parser.add_argument("--base_ordering", type=str, default="key_clustered",
                        help="Base ordering to derive batches from (if no assignment)")
    parser.add_argument("--output_dir", type=str, default="results/matched_ordering")
    parser.add_argument("--max_prefix", type=int, default=500,
                        help="Max prefix keys for spectral cost (subsampled if larger)")
    args = parser.parse_args()

    # Resolve paths
    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / keys_path

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / args.output_dir
    ord_dir = out_base / "orderings"
    diag_dir = out_base / "diagnostics"
    ord_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Load keys
    print(f"Loading keys from {keys_path.name}...")
    npz = np.load(keys_path)
    all_keys = npz["keys"]
    all_case_ids = npz["case_ids"].tolist()
    cid_to_kidx = {int(cid): i for i, cid in enumerate(all_case_ids)}
    print(f"  Keys: {all_keys.shape}")

    # Load batch assignment
    if args.batch_assignment:
        ba_path = Path(args.batch_assignment)
    else:
        ba_path = diag_dir / f"fixed_batch_assignment_seed{args.seed}.json"

    if ba_path.exists():
        with open(ba_path) as f:
            ba = json.load(f)
        batch_cids_list = ba["batches"]
        print(f"  Loaded batch assignment: {len(batch_cids_list)} batches")
    else:
        # Fall back to deriving from base ordering
        base_path = ord_dir / f"{args.base_ordering}_seed{args.seed}.json"
        if not base_path.exists():
            print(f"ERROR: Neither batch assignment nor base ordering found")
            print(f"  Tried: {ba_path}")
            print(f"  Tried: {base_path}")
            sys.exit(1)
        with open(base_path) as f:
            base_ordering = json.load(f)
        batch_size = 100
        batches = [base_ordering[i:i+batch_size]
                    for i in range(0, len(base_ordering), batch_size)]
        batch_cids_list = [[r["case_id"] for r in b] for b in batches]
        print(f"  Derived {len(batch_cids_list)} batches from {args.base_ordering}")

    # We need full records to write the output JSON. Load the base ordering.
    base_path = ord_dir / f"{args.base_ordering}_seed{args.seed}.json"
    if base_path.exists():
        with open(base_path) as f:
            base_records = json.load(f)
        cid_to_record = {r["case_id"]: r for r in base_records}
    else:
        print(f"WARNING: Base ordering not found at {base_path}")
        print(f"  Output will contain case_id-only stub records")
        cid_to_record = None

    # Determine which presets to run
    if args.preset == "all":
        presets_to_run = list(PRESETS.items())
    elif args.preset:
        presets_to_run = [(args.preset, PRESETS[args.preset])]
    else:
        presets_to_run = [
            (f"sched_b{args.beta}_g{args.gamma}",
             {"beta": args.beta, "gamma": args.gamma,
              "use_risk": True, "use_spectral": args.beta > 0})
        ]

    print(f"\n{'='*70}")
    print(f"Conditioning-Aware Interference Scheduler")
    print(f"  Seed:    {args.seed}")
    print(f"  Batches: {len(batch_cids_list)}")
    print(f"  Presets: {[name for name, _ in presets_to_run]}")
    print(f"{'='*70}")

    all_results = {}

    for name, params in presets_to_run:
        print(f"\n  === Scheduling: {name} (β={params['beta']}, γ={params.get('gamma', 0)}) ===")

        order = schedule_batches(
            batch_cids_list,
            all_keys,
            cid_to_kidx,
            beta=params["beta"],
            gamma=params.get("gamma", 0),
            use_risk=params.get("use_risk", True),
            use_spectral=params.get("use_spectral", True),
        )

        # Build flat record list in scheduled order
        flat = []
        for b_idx in order:
            for cid in batch_cids_list[b_idx]:
                if cid_to_record and cid in cid_to_record:
                    flat.append(cid_to_record[cid])
                else:
                    flat.append({"case_id": cid})

        # Compute diagnostics
        norms_all = np.linalg.norm(all_keys, axis=1, keepdims=True)
        normed_all = all_keys / np.maximum(norms_all, 1e-8)

        within_cos = []
        future_exp = []
        batch_size = len(batch_cids_list[0])
        for pos, b_idx in enumerate(order):
            indices = [cid_to_kidx[c] for c in batch_cids_list[b_idx] if c in cid_to_kidx]
            bk = normed_all[indices]
            n = len(indices)
            if n > 1:
                cm = bk @ bk.T
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                within_cos.append(float(cm[mask].mean()))

            fut_indices = []
            for fut_pos in range(pos + 1, min(pos + 11, len(order))):
                fut_b = order[fut_pos]
                fut_indices.extend(
                    cid_to_kidx[c] for c in batch_cids_list[fut_b] if c in cid_to_kidx
                )
            if fut_indices:
                cross = bk @ normed_all[fut_indices].T
                future_exp.append(float(cross.max(axis=1).mean()))

        ordering_name = f"sched_{name}"
        out_path = ord_dir / f"{ordering_name}_seed{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(flat, f)
        print(f"  Saved: {out_path.name} ({len(flat)} records)")

        metrics = {
            "name": ordering_name,
            "params": params,
            "batch_order": order,
            "n_records": len(flat),
            "mean_within_batch_cosine": float(np.mean(within_cos)) if within_cos else 0,
            "mean_future_exposure": float(np.mean(future_exp)) if future_exp else 0,
        }
        all_results[name] = metrics
        print(f"  Within-batch cosine: {metrics['mean_within_batch_cosine']:.4f} "
              f"(identical to other fb orderings)")
        print(f"  Future exposure: {metrics['mean_future_exposure']:.4f}")

    # Save diagnostics
    diag_path = diag_dir / f"scheduled_ordering_report_seed{args.seed}.json"
    with open(diag_path, "w") as f:
        json.dump({"seed": args.seed, "schedules": all_results}, f, indent=2)
    print(f"\n  Diagnostics: {diag_path.name}")

    print(f"\n{'='*70}")
    print("Scheduled orderings generated. Run experiments with:")
    for name, _ in presets_to_run:
        print(f"  bash scripts/run_matched_ordering.sh {args.seed} AlphaEdit sched_{name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
