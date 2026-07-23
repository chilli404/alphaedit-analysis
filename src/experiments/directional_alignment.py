#!/usr/bin/env python3
"""
Directional Alignment Analysis: Does later displacement oppose the installation direction?

For each first-cohort edit i:
  e_i = (W_1K - W_0) @ k_i        # installation direction (what installing edit i did)
  d_i(T) = (W_T - W_1K) @ k_i     # later net displacement

  a_i = cos(d_i, e_i)             # alignment: <0 means opposition, >0 means reinforcement
  D_i = -<d_i, e_i> / (||e_i||^2 + eps)  # damage score: positive = reversal fraction

Compare retained vs forgotten edits, controlling for:
  - ||e_i|| (installation magnitude)
  - position within first cohort
  - key norm ||k_i||
  - initial success at 1K

This is cheap (CPU only once base model weight is extracted) and tests the most
plausible explanation for the magnitude null: direction, not magnitude, predicts forgetting.

Usage:
    # Extract base model weight (GPU, one-time, ~30s):
    python src/experiments/directional_alignment.py --extract_base_weight

    # Run analysis (CPU only, uses checkpoints + extracted base weight):
    python src/experiments/directional_alignment.py --seed 42

    # Both:
    python src/experiments/directional_alignment.py --seed 42 --extract_base_weight
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from paths import get_checkpoint_root, get_result_root

# Constants
DEFAULT_LAYER = 6
WEIGHT_SHAPE = (4096, 14336)  # Same for all layers in Llama 3 8B
NUM_EDITS_PER_BATCH = 100


def weight_key_for_layer(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def get_base_weight_path(layer: int = DEFAULT_LAYER) -> Path:
    """Path where extracted base model weight is stored."""
    return get_result_root() / "interference" / f"base_weight_layer{layer}.npy"


def extract_base_weight(model_name: str, layers: list[int] | None = None):
    """Extract down_proj weight from base (unedited) model. GPU needed once.

    Args:
        model_name: HuggingFace model name or local path.
        layers: Which layers to extract. Defaults to [DEFAULT_LAYER].
    """
    if layers is None:
        layers = [DEFAULT_LAYER]

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import download_model, _artifactory_reachable  # patches filelock

    import torch
    from transformers import AutoModelForCausalLM

    # Download model explicitly if on Artifactory infra, then load from local path
    if _artifactory_reachable():
        model_path = download_model(model_name)
    else:
        model_path = model_name

    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    param_dict = dict(model.named_parameters())
    last_w = None

    for layer in layers:
        wkey = weight_key_for_layer(layer)
        w = param_dict[wkey].detach().float().numpy()
        assert w.shape == WEIGHT_SHAPE, f"Unexpected shape for layer {layer}: {w.shape}"

        out_path = get_base_weight_path(layer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, w)
        print(f"Base weight saved: {out_path} (shape={w.shape})")
        last_w = w

    del model
    return last_w


def load_base_weight(layer: int = DEFAULT_LAYER) -> np.ndarray:
    """Load pre-extracted base model weight."""
    path = get_base_weight_path(layer)
    if not path.exists():
        raise FileNotFoundError(
            f"Base weight not found at {path}. "
            f"Run with --extract_base_weight --layers {layer} first."
        )
    w = np.load(path)
    assert w.shape == WEIGHT_SHAPE
    return w


def load_checkpoint_weight(ckpt_path: Path, layer: int = DEFAULT_LAYER) -> np.ndarray:
    """Load down_proj weight from checkpoint for given layer."""
    import torch
    weights = torch.load(ckpt_path / "model_weights.pt", map_location="cpu")
    wkey = weight_key_for_layer(layer)
    w = weights[wkey].float().numpy()
    assert w.shape == WEIGHT_SHAPE
    return w


def load_keys(seed: int, layer: int = DEFAULT_LAYER) -> Tuple[np.ndarray, np.ndarray]:
    """Load keys [5000, 14336] and case_ids [5000]."""
    path = get_result_root() / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer{layer}.npz"
    data = np.load(path)
    return data["keys"], data["case_ids"]


def load_ordering(ordering_name: str, seed: int) -> List[int]:
    """Load ordering -> list of case_ids in edit order."""
    path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering_name}_seed{seed}.json"
    with open(path) as f:
        records = json.load(f)
    return [r["case_id"] for r in records]


def load_percase_eval(alg: str, ordering: str, seed: int) -> Dict[int, Dict]:
    """Load per-case behavioral eval. Returns {case_id: {efficacy, paraphrase}}."""
    path = get_result_root() / "interference" / alg / ordering / f"seed{seed}" / "percase_eval.json"
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    return {r["case_id"]: r for r in records}


def compute_directional_alignment(
    W_0: np.ndarray,
    W_1K: np.ndarray,
    W_T: np.ndarray,
    keys: np.ndarray,
    case_ids: np.ndarray,
    ordering_case_ids: List[int],
    ordering_name: str,
) -> Dict:
    """
    Compute directional alignment metrics for first-1K cohort.

    Returns per-edit: a_i (cosine alignment), D_i (damage score),
    ||e_i|| (installation magnitude), ||d_i|| (displacement magnitude).
    """
    print(f"\n  Computing directional alignment for {ordering_name}...")

    # Map case_id -> key index
    cid_to_kidx = {int(cid): i for i, cid in enumerate(case_ids)}

    # First-1K key indices
    first_1k_kidx = np.array([cid_to_kidx[cid] for cid in ordering_case_ids[:1000]])
    K_first1K = keys[first_1k_kidx]  # [1000, 14336]

    # Installation direction: e_i = (W_1K - W_0) @ k_i for each key
    # This is the net effect of the ENTIRE first 1K edits on key i's direction.
    # NOTE: This includes within-batch effects from ALL 1K edits, not just edit i.
    # It's the total change that occurred during the installation cohort.
    delta_install = W_1K - W_0  # [4096, 14336]
    E = delta_install @ K_first1K.T  # [4096, 1000] — installation directions

    # Later net displacement: d_i(T) = (W_T - W_1K) @ k_i
    delta_later = W_T - W_1K  # [4096, 14336]
    D_vecs = delta_later @ K_first1K.T  # [4096, 1000] — displacement vectors

    # Per-edit metrics
    n = E.shape[1]
    e_norms = np.linalg.norm(E, axis=0)  # [1000]
    d_norms = np.linalg.norm(D_vecs, axis=0)  # [1000]
    key_norms = np.linalg.norm(K_first1K, axis=1)  # [1000]

    # Cosine alignment: a_i = cos(d_i, e_i)
    dot_products = np.sum(E * D_vecs, axis=0)  # [1000]
    denom = e_norms * d_norms + 1e-10
    a_i = dot_products / denom  # [1000]

    # Damage score: D_i = -<d_i, e_i> / (||e_i||^2 + eps)
    # Positive D_i means later updates reverse some fraction of installation
    damage_i = -dot_products / (e_norms ** 2 + 1e-10)  # [1000]

    # Summary
    print(f"    ||e_i|| (install mag): mean={e_norms.mean():.6f}, std={e_norms.std():.6f}")
    print(f"    ||d_i|| (later disp):  mean={d_norms.mean():.6f}, std={d_norms.std():.6f}")
    print(f"    a_i (alignment):  mean={a_i.mean():.4f}, median={np.median(a_i):.4f}")
    print(f"      fraction a_i < 0 (opposing): {(a_i < 0).mean():.3f}")
    print(f"      fraction a_i > 0 (reinforcing): {(a_i > 0).mean():.3f}")
    print(f"    D_i (damage):     mean={damage_i.mean():.4f}, median={np.median(damage_i):.4f}")
    print(f"      fraction D_i > 0 (reversal): {(damage_i > 0).mean():.3f}")

    return {
        "ordering": ordering_name,
        "first_1K_case_ids": [int(ordering_case_ids[i]) for i in range(1000)],
        "key_indices": first_1k_kidx.tolist(),
        "a_i": a_i.tolist(),
        "damage_i": damage_i.tolist(),
        "e_norm": e_norms.tolist(),
        "d_norm": d_norms.tolist(),
        "key_norm": key_norms.tolist(),
        "dot_product": dot_products.tolist(),
        "summary": {
            "mean_alignment": float(a_i.mean()),
            "median_alignment": float(np.median(a_i)),
            "frac_opposing": float((a_i < 0).mean()),
            "frac_reinforcing": float((a_i > 0).mean()),
            "mean_damage": float(damage_i.mean()),
            "median_damage": float(np.median(damage_i)),
            "frac_reversal": float((damage_i > 0).mean()),
            "mean_e_norm": float(e_norms.mean()),
            "mean_d_norm": float(d_norms.mean()),
        },
    }


def retained_vs_forgotten_directional(
    result: Dict,
    percase_eval: Dict[int, Dict],
    ordering_name: str,
) -> Dict:
    """
    Compare directional metrics between retained and forgotten edits.
    Controls for installation magnitude, position, and key norm.
    """
    print(f"\n  === Retained vs Forgotten: {ordering_name} ===")

    case_ids = result["first_1K_case_ids"]
    a_i = np.array(result["a_i"])
    damage_i = np.array(result["damage_i"])
    e_norm = np.array(result["e_norm"])
    d_norm = np.array(result["d_norm"])
    key_norm = np.array(result["key_norm"])

    # Classify retained vs forgotten
    retained_idx = []
    forgotten_idx = []
    for i, cid in enumerate(case_ids):
        if cid in percase_eval:
            eff = percase_eval[cid]["efficacy"]
            if eff >= 0.5:
                retained_idx.append(i)
            else:
                forgotten_idx.append(i)

    n_ret = len(retained_idx)
    n_forg = len(forgotten_idx)
    print(f"    Retained: {n_ret}, Forgotten: {n_forg}")

    if n_forg < 5:
        print(f"    SKIP: too few forgotten")
        return {"skip": True, "n_retained": n_ret, "n_forgotten": n_forg}

    # Compare distributions
    stats = {"n_retained": n_ret, "n_forgotten": n_forg}

    for name, values in [("a_i", a_i), ("damage_i", damage_i),
                          ("e_norm", e_norm), ("d_norm", d_norm)]:
        v_ret = values[retained_idx]
        v_forg = values[forgotten_idx]
        median_diff = float(np.median(v_forg) - np.median(v_ret))
        mean_diff = float(np.mean(v_forg) - np.mean(v_ret))

        stats[name] = {
            "retained_median": float(np.median(v_ret)),
            "retained_mean": float(np.mean(v_ret)),
            "forgotten_median": float(np.median(v_forg)),
            "forgotten_mean": float(np.mean(v_forg)),
            "median_diff": median_diff,
            "mean_diff": mean_diff,
        }

        # Cliff's delta (vectorized)
        if len(v_forg) * len(v_ret) < 1e8:
            diff_matrix = v_forg[:, None] - v_ret[None, :]
            cliffs_d = float(np.mean(np.sign(diff_matrix)))
        else:
            rng = np.random.default_rng(42)
            xi = rng.choice(v_forg, 100000)
            yi = rng.choice(v_ret, 100000)
            cliffs_d = float(np.mean(np.sign(xi - yi)))
        stats[name]["cliffs_delta"] = cliffs_d

        direction = "forgotten > retained" if median_diff > 0 else "retained >= forgotten"
        print(f"    {name}: forg_med={np.median(v_forg):.4f}, ret_med={np.median(v_ret):.4f}, "
              f"diff={median_diff:.4f}, Cliff's d={cliffs_d:.4f} ({direction})")

    # Key question: do forgotten edits have MORE negative alignment (more opposition)?
    # Expected: a_i(forgotten) < a_i(retained) → Cliff's delta < 0 for a_i
    # Expected: damage_i(forgotten) > damage_i(retained) → Cliff's delta > 0 for damage_i

    # Controlled analysis: partial correlation controlling for e_norm and position
    try:
        from scipy import stats as sp_stats
        positions = np.arange(1000, dtype=np.float64)

        # Simple logistic regression: P(retained) ~ position + e_norm + damage_i
        try:
            import statsmodels.api as sm
            y = np.array([1 if i in set(retained_idx) else 0
                         for i in range(len(case_ids))
                         if i in set(retained_idx) | set(forgotten_idx)])
            valid_idx = sorted(retained_idx + forgotten_idx)
            X = np.column_stack([
                positions[valid_idx] / 1000,  # position (normalized)
                e_norm[valid_idx] / (e_norm.max() + 1e-10),  # install magnitude
                key_norm[valid_idx] / (key_norm.max() + 1e-10),  # key norm
                damage_i[valid_idx],  # damage score (primary predictor)
            ])
            X_const = sm.add_constant(X)
            logit_model = sm.Logit(y, X_const).fit(disp=0)

            stats["logistic"] = {
                "coef_names": ["const", "position", "e_norm", "key_norm", "damage_i"],
                "coefs": [round(float(c), 4) for c in logit_model.params],
                "pvalues": [round(float(p), 6) for p in logit_model.pvalues],
                "pseudo_r2": round(float(logit_model.prsquared), 4),
            }
            print(f"    Logistic (P(retained) ~ pos + e_norm + key_norm + damage_i):")
            print(f"      coef_damage = {logit_model.params[4]:.4f}, "
                  f"p = {logit_model.pvalues[4]:.4e}")
            print(f"      pseudo-R2 = {logit_model.prsquared:.4f}")
        except ImportError:
            print("    statsmodels not available, skipping logistic")
        except Exception as e:
            print(f"    Logistic failed: {e}")

    except ImportError:
        pass

    return stats


def run_directional_analysis(seed: int, ckpt_base: Path, output_dir: Path, layer: int = DEFAULT_LAYER):
    """Run full directional alignment analysis for all 4 conditions."""
    print("=" * 70)
    print(f"DIRECTIONAL ALIGNMENT ANALYSIS (layer {layer})")
    print("=" * 70)

    # Load base weight
    W_0 = load_base_weight(layer)
    print(f"  Base weight loaded: {W_0.shape}")

    # Load keys
    keys, case_ids = load_keys(seed, layer)

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    all_results = {}

    for alg, ordering in conditions:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"

        # Check checkpoints
        batch9_path = ckpt_dir / "batch_9"
        batch49_path = ckpt_dir / "batch_49"
        if not (batch9_path / "model_weights.pt").exists():
            print(f"\n  SKIP {condition_name}: no batch_9 checkpoint")
            continue
        if not (batch49_path / "model_weights.pt").exists():
            print(f"\n  SKIP {condition_name}: no batch_49 checkpoint")
            continue

        W_1K = load_checkpoint_weight(batch9_path, layer)
        W_5K = load_checkpoint_weight(batch49_path, layer)

        ordering_case_ids = load_ordering(ordering, seed)

        result = compute_directional_alignment(
            W_0=W_0, W_1K=W_1K, W_T=W_5K,
            keys=keys, case_ids=case_ids,
            ordering_case_ids=ordering_case_ids,
            ordering_name=condition_name,
        )
        all_results[condition_name] = result

        # Retained vs forgotten (if per-case eval available)
        percase = load_percase_eval(alg, ordering, seed)
        if percase:
            comparison = retained_vs_forgotten_directional(result, percase, condition_name)
            all_results[f"{condition_name}/comparison"] = comparison
        else:
            print(f"    No per-case eval found for {condition_name}")

    # Also compute at intermediate checkpoints for progressive analysis
    print("\n\n" + "=" * 70)
    print("PROGRESSIVE ALIGNMENT (intermediate checkpoints)")
    print("=" * 70)

    for alg, ordering in [("AlphaEdit", "key_clustered"), ("AlphaEdit", "key_dispersed")]:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"
        ordering_case_ids = load_ordering(ordering, seed)
        first_1k_kidx = np.array([
            {int(cid): i for i, cid in enumerate(case_ids)}[cid]
            for cid in ordering_case_ids[:1000]
        ])
        K_first1K = keys[first_1k_kidx]

        W_1K = load_checkpoint_weight(ckpt_dir / "batch_9", layer)
        delta_install = W_1K - W_0
        E = delta_install @ K_first1K.T  # [4096, 1000]
        e_norms = np.linalg.norm(E, axis=0)

        progressive = []
        for batch_idx in [19, 29, 39, 49]:
            batch_path = ckpt_dir / f"batch_{batch_idx}"
            if not (batch_path / "model_weights.pt").exists():
                continue
            W_T = load_checkpoint_weight(batch_path, layer)
            D_vecs = (W_T - W_1K) @ K_first1K.T
            dot_products = np.sum(E * D_vecs, axis=0)
            d_norms = np.linalg.norm(D_vecs, axis=0)
            a_i = dot_products / (e_norms * d_norms + 1e-10)
            damage_i = -dot_products / (e_norms ** 2 + 1e-10)

            progressive.append({
                "checkpoint": f"batch_{batch_idx}",
                "total_edits": (batch_idx + 1) * 100,
                "mean_alignment": float(a_i.mean()),
                "median_alignment": float(np.median(a_i)),
                "frac_opposing": float((a_i < 0).mean()),
                "mean_damage": float(damage_i.mean()),
                "median_damage": float(np.median(damage_i)),
            })
            print(f"  {condition_name} @ {(batch_idx+1)*100} edits: "
                  f"mean_a={a_i.mean():.4f}, frac_opposing={(a_i < 0).mean():.3f}, "
                  f"mean_D={damage_i.mean():.4f}")

        all_results[f"{condition_name}/progressive"] = progressive

    # Save all results (include layer suffix for non-default layers)
    suffix = "" if layer == DEFAULT_LAYER else f"_layer{layer}"
    for cond_name, result in all_results.items():
        if "/" not in cond_name:
            continue
        parts = cond_name.split("/")
        if len(parts) == 2:
            alg, ordering = parts
            cond_dir = output_dir / alg / ordering / f"seed{seed}"
            cond_dir.mkdir(parents=True, exist_ok=True)
            with open(cond_dir / f"directional_alignment{suffix}.json", "w") as f:
                json.dump(result, f, indent=2)
        elif len(parts) == 3 and parts[2] == "comparison":
            alg, ordering = parts[0], parts[1]
            cond_dir = output_dir / alg / ordering / f"seed{seed}"
            cond_dir.mkdir(parents=True, exist_ok=True)
            with open(cond_dir / f"directional_comparison{suffix}.json", "w") as f:
                json.dump(result, f, indent=2)
        elif len(parts) == 3 and parts[2] == "progressive":
            alg, ordering = parts[0], parts[1]
            cond_dir = output_dir / alg / ordering / f"seed{seed}"
            cond_dir.mkdir(parents=True, exist_ok=True)
            with open(cond_dir / f"directional_progressive{suffix}.json", "w") as f:
                json.dump(result, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Directional alignment analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="Which layer to analyze (default: 6)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers for extraction, e.g. '4,5,6,7,8'")
    parser.add_argument("--extract_base_weight", action="store_true",
                        help="Extract base model weight (GPU, one-time)")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract base weight, skip analysis")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--checkpoint_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Parse layers for extraction
    if args.layers:
        extract_layers = [int(x.strip()) for x in args.layers.split(",")]
    else:
        extract_layers = [args.layer]

    if args.extract_base_weight:
        extract_base_weight(args.model_name, layers=extract_layers)
        if args.extract_only:
            return

    # Check base weight exists before analysis
    if not get_base_weight_path(args.layer).exists():
        print(f"ERROR: Base weight not found for layer {args.layer}. "
              f"Run with --extract_base_weight --layers {args.layer} first.")
        sys.exit(1)

    # Resolve checkpoint base
    if args.checkpoint_base:
        ckpt_base = Path(args.checkpoint_base).expanduser()
    else:
        ckpt_base = get_checkpoint_root()

    output_dir = Path(args.output_dir) if args.output_dir else get_result_root() / "interference"

    run_directional_analysis(args.seed, ckpt_base, output_dir, layer=args.layer)


if __name__ == "__main__":
    main()
