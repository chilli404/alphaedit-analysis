#!/usr/bin/env python3
"""
Installation Strength Analysis: Does ordering affect initial installation quality?

Hypothesis: Edits that are eventually forgotten were already installed more weakly,
and dispersed ordering generates more weakly-installed edits.

Measures at installation time (batch_9 = 1K checkpoint):
  - Immediate full-target efficacy at 1K
  - Target probability margin (log P(target_new) - log P(target_true))
  - Installation vector magnitude ||e_i|| = ||(W_1K - W_0) @ k_i||
  - Within-batch key similarity (mean cosine to other keys in same 100-edit batch)
  - Future-key similarity (mean/max cosine to keys in batches 10-49)

Model: P(retained at 5K) ~ initial_margin + e_norm + future_key_sim + position + within_batch_sim

Then: does future-key similarity contribute after controlling for initial strength?

Usage:
    # GPU phase: evaluate first-1K at 1K checkpoint (need model)
    python src/experiments/installation_strength.py --seed 42 --eval_at_1k

    # CPU phase: compute features + fit model (no GPU)
    python src/experiments/installation_strength.py --seed 42 --analyze

    # Both:
    python src/experiments/installation_strength.py --seed 42 --eval_at_1k --analyze
"""

import argparse
import json
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


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_keys(seed: int, layer: int = DEFAULT_LAYER) -> Tuple[np.ndarray, np.ndarray]:
    """Load keys [5000, 14336] and case_ids [5000]."""
    path = get_result_root() / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer{layer}.npz"
    data = np.load(path)
    return data["keys"], data["case_ids"]


def load_ordering(ordering_name: str, seed: int) -> List[Dict]:
    """Load ordering -> list of records with case_id + requested_rewrite."""
    path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering_name}_seed{seed}.json"
    with open(path) as f:
        return json.load(f)


def load_percase_eval_5k(alg: str, ordering: str, seed: int) -> Dict[int, Dict]:
    """Load per-case eval at 5K. Returns {case_id: {efficacy, paraphrase}}."""
    path = get_result_root() / "interference" / alg / ordering / f"seed{seed}" / "percase_eval.json"
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    return {r["case_id"]: r for r in records}


def load_checkpoint_weight(ckpt_path: Path, layer: int = DEFAULT_LAYER) -> np.ndarray:
    """Load down_proj weight from checkpoint for given layer."""
    import torch
    weights = torch.load(ckpt_path / "model_weights.pt", map_location="cpu")
    wkey = weight_key_for_layer(layer)
    w = weights[wkey].float().numpy()
    assert w.shape == WEIGHT_SHAPE
    return w


def load_base_weight(layer: int = DEFAULT_LAYER) -> np.ndarray:
    """Load pre-extracted base model weight."""
    path = get_result_root() / "interference" / f"base_weight_layer{layer}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Base weight not found at {path}. "
            f"Run directional_alignment.py --extract_base_weight --layers {layer} first."
        )
    return np.load(path)


# ─── GPU Phase: Evaluate at 1K Checkpoint ─────────────────────────────────────

def eval_at_1k_checkpoint(seed: int, ckpt_base: Path, output_dir: Path, model_name: str):
    """Evaluate first-1K edits at 1K checkpoint (batch_9) for all 4 conditions.

    Measures both binary efficacy AND probability margin.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import download_model, _artifactory_reachable

    if _artifactory_reachable():
        model_name = download_model(model_name)

    print("=" * 70)
    print("INSTALLATION STRENGTH: Evaluate at 1K checkpoint")
    print("=" * 70)

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    model = None
    tok = None

    for alg, ordering in conditions:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"
        ckpt_1k = ckpt_dir / "batch_9"

        if not (ckpt_1k / "model_weights.pt").exists():
            print(f"\n  SKIP {condition_name}: no batch_9 checkpoint")
            continue

        print(f"\n  Evaluating {condition_name} at 1K checkpoint...")

        # Load ordering records
        ordering_path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
        with open(ordering_path) as f:
            records = json.load(f)

        # Load/reuse model
        if model is None:
            print(f"    Loading base model: {model_name}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
            tok.pad_token = tok.eos_token
            tok.padding_side = "right"
        else:
            print("    Reusing loaded model, swapping weights...")

        # Apply 1K checkpoint weights
        weights = torch.load(str(ckpt_1k / "model_weights.pt"), map_location="cuda")
        param_dict = dict(model.named_parameters())
        for name, tensor in weights.items():
            if name in param_dict:
                param_dict[name].data.copy_(tensor.cuda().half())
        del weights
        torch.cuda.empty_cache()

        # Evaluate first 1K with margin measurement
        first_1k_records = records[:1000]
        percase_results = _evaluate_with_margin(model, tok, first_1k_records)

        # Save
        cond_dir = output_dir / alg / ordering / f"seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_path = cond_dir / "percase_eval_1k.json"
        with open(out_path, "w") as f:
            json.dump(percase_results, f, indent=2)

        eff_vals = [r["efficacy"] for r in percase_results]
        margins = [r["margin"] for r in percase_results]
        n_successful = sum(1 for e in eff_vals if e >= 0.5)
        print(f"    First-1K at 1K: mean_eff={np.mean(eff_vals):.4f}, "
              f"successful={n_successful}/1000, "
              f"mean_margin={np.mean(margins):.4f}")
        print(f"  Saved: {out_path}")

    del model
    torch.cuda.empty_cache()


def _evaluate_with_margin(model, tok, records: List[Dict]) -> List[Dict]:
    """Full multi-token eval + probability margin measurement.

    Returns per-record: {case_id, efficacy, paraphrase, margin, target_prob, true_prob}
    Margin = mean log P(target_new tokens) - mean log P(target_true tokens)
    """
    import torch
    from itertools import chain

    results = []
    is_llama = "llama" in model.config._name_or_path.lower()

    for i, record in enumerate(records):
        rw = record["requested_rewrite"]
        subject = rw["subject"]
        target_new = rw["target_new"]["str"]
        target_true = rw["target_true"]["str"]

        rewrite_prompts = [rw["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]

        prob_prompts = [rewrite_prompts, paraphrase_prompts]
        which_correct = [
            [0] * len(rewrite_prompts),
            [0] * len(paraphrase_prompts),
        ]

        all_prefixes = list(chain(*prob_prompts))
        all_which = list(chain(*which_correct))

        # Compute correctness + margin
        targets_correct, margin_info = _test_batch_prediction_with_margin(
            model, tok, all_prefixes, all_which, target_new, target_true, is_llama
        )

        n_rw = len(rewrite_prompts)
        results.append({
            "case_id": record["case_id"],
            "efficacy": float(np.mean(targets_correct[:n_rw])),
            "paraphrase": float(np.mean(targets_correct[n_rw:])),
            "margin": margin_info["margin"],
            "target_new_logprob": margin_info["target_new_logprob"],
            "target_true_logprob": margin_info["target_true_logprob"],
        })

        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(records)}] "
                  f"eff={np.mean([r['efficacy'] for r in results]):.4f}, "
                  f"margin={np.mean([r['margin'] for r in results]):.4f}")

    return results


def _test_batch_prediction_with_margin(
    model, tok, prefixes: List[str], which_correct: List[int],
    target_new: str, target_true: str, is_llama: bool,
) -> Tuple[List[bool], Dict]:
    """Full multi-token eval with probability margin.

    Returns (targets_correct, margin_info) where margin_info has:
      margin: mean log-prob of target_new tokens - mean log-prob of target_true tokens
              (for the first prefix only, which is the rewrite prompt)
    """
    import torch
    import torch.nn.functional as F

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok = tok(f" {target_new}")["input_ids"]
    b_tok = tok(f" {target_true}")["input_ids"]

    if is_llama:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [l - 1 for l in prefix_lens]

    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if is_llama:
        logits = logits[:, 1:, :]

    targets_correct = []
    target_new_logprob = None
    target_true_logprob = None

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        cur_tok = a_tok if i % 2 == 0 else b_tok

        wc = which_correct[i // 2]
        is_correct_target = (wc == 0 and i % 2 == 0) or (wc == 1 and i % 2 == 1)

        if is_correct_target:
            correct = True
            for j in range(cur_len):
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok[j]:
                    correct = False
                    break
            targets_correct.append(correct)

        # Compute log-probs for first prefix (rewrite prompt) — both targets
        if i // 2 == 0:
            log_probs = F.log_softmax(logits[i], dim=-1)
            token_logprobs = []
            for j in range(cur_len):
                lp = log_probs[prefix_lens[0] + j - 1, cur_tok[j]].item()
                token_logprobs.append(lp)
            mean_lp = float(np.mean(token_logprobs))

            if i % 2 == 0:
                target_new_logprob = mean_lp
            else:
                target_true_logprob = mean_lp

    margin_info = {
        "margin": (target_new_logprob - target_true_logprob)
                  if (target_new_logprob is not None and target_true_logprob is not None)
                  else 0.0,
        "target_new_logprob": target_new_logprob if target_new_logprob is not None else 0.0,
        "target_true_logprob": target_true_logprob if target_true_logprob is not None else 0.0,
    }

    del logits, prompt_tok
    torch.cuda.empty_cache()

    return targets_correct, margin_info


# ─── CPU Phase: Key Geometry Features + Model Fitting ─────────────────────────

def compute_key_geometry_features(seed: int, ckpt_base: Path, output_dir: Path, layer: int = DEFAULT_LAYER):
    """Compute within-batch similarity, future-key similarity, installation magnitude.

    All CPU — uses pre-computed keys and base weight.
    """
    print("\n" + "=" * 70)
    print(f"INSTALLATION STRENGTH: Key Geometry Features (layer {layer})")
    print("=" * 70)

    # Load keys
    keys, case_ids = load_keys(seed, layer)
    cid_to_kidx = {int(cid): i for i, cid in enumerate(case_ids)}
    print(f"  Keys loaded: {keys.shape}")

    # Load base weight
    W_0 = load_base_weight(layer)
    print(f"  Base weight loaded: {W_0.shape}")

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    for alg, ordering in conditions:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"
        ckpt_1k = ckpt_dir / "batch_9"

        if not (ckpt_1k / "model_weights.pt").exists():
            print(f"\n  SKIP {condition_name}: no batch_9 checkpoint")
            continue

        print(f"\n  Computing features for {condition_name}...")

        # Load ordering
        ordering_path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
        with open(ordering_path) as f:
            records = json.load(f)

        # Get first-1K case IDs and their keys
        first_1k_cids = [r["case_id"] for r in records[:1000]]
        first_1k_kidx = np.array([cid_to_kidx[cid] for cid in first_1k_cids])
        K_first1K = keys[first_1k_kidx]  # [1000, 14336]

        # Get future keys (batches 10-49 = positions 1000-4999)
        future_cids = [r["case_id"] for r in records[1000:]]
        future_kidx = np.array([cid_to_kidx[cid] for cid in future_cids])
        K_future = keys[future_kidx]  # [4000, 14336]

        # 1. Installation magnitude: ||(W_1K - W_0) @ k_i||
        W_1K = load_checkpoint_weight(ckpt_1k, layer)
        delta_install = W_1K - W_0  # [4096, 14336]
        E = delta_install @ K_first1K.T  # [4096, 1000]
        e_norms = np.linalg.norm(E, axis=0)  # [1000]

        # 2. Within-batch key similarity
        # For each edit i in batch b (batch = position // 100), compute mean cosine
        # to other keys in the same batch
        K_first1K_normed = K_first1K / (np.linalg.norm(K_first1K, axis=1, keepdims=True) + 1e-10)
        within_batch_sim = np.zeros(1000)
        for batch_idx in range(10):
            start = batch_idx * 100
            end = start + 100
            batch_keys_normed = K_first1K_normed[start:end]  # [100, 14336]
            # Pairwise cosine similarity within batch
            sim_matrix = batch_keys_normed @ batch_keys_normed.T  # [100, 100]
            # Mean cosine excluding self (diagonal)
            np.fill_diagonal(sim_matrix, 0)
            within_batch_sim[start:end] = sim_matrix.sum(axis=1) / 99.0

        # 3. Future-key similarity
        # For each first-1K edit, compute mean and max cosine to future keys
        K_future_normed = K_future / (np.linalg.norm(K_future, axis=1, keepdims=True) + 1e-10)
        # [1000, 4000] cosine matrix — too large for memory? 1000*4000*4 = 16MB, fine.
        future_sim_matrix = K_first1K_normed @ K_future_normed.T  # [1000, 4000]
        future_key_mean_sim = future_sim_matrix.mean(axis=1)  # [1000]
        future_key_max_sim = future_sim_matrix.max(axis=1)  # [1000]

        # 4. Key norm
        key_norms = np.linalg.norm(K_first1K, axis=1)  # [1000]

        # 5. Position (0-indexed)
        positions = np.arange(1000)

        # 6. Batch-level aggregates: mean within-batch similarity for the installation batch
        batch_mean_sims = np.array([within_batch_sim[b*100:(b+1)*100].mean() for b in range(10)])

        # Save features as JSON (avoids np.savez seek issues on FUSE filesystems)
        suffix = "" if layer == DEFAULT_LAYER else f"_layer{layer}"
        cond_dir = output_dir / alg / ordering / f"seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_path = cond_dir / f"installation_features{suffix}.json"
        feat_data = {
            "case_ids": [int(c) for c in first_1k_cids],
            "positions": positions.tolist(),
            "e_norm": e_norms.tolist(),
            "key_norm": key_norms.tolist(),
            "within_batch_sim": within_batch_sim.tolist(),
            "future_key_mean_sim": future_key_mean_sim.tolist(),
            "future_key_max_sim": future_key_max_sim.tolist(),
            "batch_mean_sims": batch_mean_sims.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(feat_data, f)
        print(f"    Saved features: {out_path}")

        # Print summary
        print(f"    e_norm: mean={e_norms.mean():.4f}, std={e_norms.std():.4f}")
        print(f"    within_batch_sim: mean={within_batch_sim.mean():.4f}, std={within_batch_sim.std():.4f}")
        print(f"    future_key_mean_sim: mean={future_key_mean_sim.mean():.4f}, std={future_key_mean_sim.std():.4f}")
        print(f"    future_key_max_sim: mean={future_key_max_sim.mean():.4f}, std={future_key_max_sim.std():.4f}")


def _load_condition_data(alg: str, ordering: str, seed: int, output_dir: Path, layer: int = DEFAULT_LAYER):
    """Load features + evals for one condition. Returns None if missing."""
    suffix = "" if layer == DEFAULT_LAYER else f"_layer{layer}"
    cond_dir = output_dir / alg / ordering / f"seed{seed}"
    feat_path = cond_dir / f"installation_features{suffix}.json"
    eval_1k_path = cond_dir / "percase_eval_1k.json"
    eval_5k_path = cond_dir / "percase_eval.json"

    if not feat_path.exists() or not eval_5k_path.exists() or not eval_1k_path.exists():
        return None

    with open(feat_path) as f:
        feats = json.load(f)
    with open(eval_5k_path) as f:
        eval_5k = {r["case_id"]: r for r in json.load(f)}
    with open(eval_1k_path) as f:
        eval_1k = {r["case_id"]: r for r in json.load(f)}

    case_ids = np.array(feats["case_ids"])
    e_norm = np.array(feats["e_norm"])
    future_key_max_sim = np.array(feats["future_key_max_sim"])
    positions = np.array(feats["positions"])

    # Build per-edit arrays: margin_1k, retained (binary), valid mask
    n = len(case_ids)
    margin_1k = np.zeros(n)
    retained = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)

    n_never = 0
    for i, cid in enumerate(case_ids):
        cid = int(cid)
        eff_5k = eval_5k.get(cid, {}).get("efficacy", None)
        eff_1k = eval_1k.get(cid, {}).get("efficacy", None)
        m = eval_1k.get(cid, {}).get("margin", None)

        if eff_5k is None or eff_1k is None or m is None:
            continue
        if eff_1k < 0.5:
            n_never += 1
            continue

        margin_1k[i] = m
        retained[i] = 1 if eff_5k >= 0.5 else 0
        valid[i] = True

    return {
        "case_ids": case_ids,
        "positions": positions,
        "e_norm": e_norm,
        "future_key_max_sim": future_key_max_sim,
        "margin_1k": margin_1k,
        "retained": retained,
        "valid": valid,
        "n_never": n_never,
    }


def _block_bootstrap_logistic(X, y, positions, n_boot=2000, block_size=100, seed=42):
    """Block bootstrap for logistic regression coefficients.

    Blocks by position (100-edit batches). Returns percentile CIs.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = int(np.ceil(n / block_size))

    # Assign each observation to a block based on position
    block_ids = (positions / block_size).astype(int)
    unique_blocks = np.unique(block_ids)

    # Fit original model
    X_c = sm.add_constant(X)
    orig_model = sm.Logit(y, X_c).fit(disp=0)
    orig_params = orig_model.params.copy()

    # Bootstrap
    boot_params = []
    for _ in range(n_boot):
        # Resample blocks with replacement
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        boot_idx = np.concatenate([np.where(block_ids == b)[0] for b in sampled_blocks])

        X_boot = X_c[boot_idx]
        y_boot = y[boot_idx]

        # Skip degenerate samples
        if y_boot.sum() < 5 or (len(y_boot) - y_boot.sum()) < 5:
            continue

        try:
            m = sm.Logit(y_boot, X_boot).fit(disp=0, maxiter=50)
            if m.mle_retvals["converged"]:
                boot_params.append(m.params)
        except Exception:
            continue

    if len(boot_params) < 100:
        return orig_params, None, None

    boot_params = np.array(boot_params)
    ci_lo = np.percentile(boot_params, 2.5, axis=0)
    ci_hi = np.percentile(boot_params, 97.5, axis=0)

    return orig_params, ci_lo, ci_hi


def run_analysis(seed: int, output_dir: Path, layer: int = DEFAULT_LAYER):
    """Final joint model with block bootstrap. CPU only."""
    print("\n" + "=" * 70)
    print(f"INSTALLATION STRENGTH: Joint Model Analysis (layer {layer})")
    print("=" * 70)

    all_results = {}

    for alg in ["AlphaEdit", "MEMIT-Seq-lp1.0-ld0.0-cache0"]:
        print(f"\n{'='*70}")
        print(f"  {alg}: Joint Model (clustered + dispersed)")
        print(f"{'='*70}")

        data_c = _load_condition_data(alg, "key_clustered", seed, output_dir, layer)
        data_d = _load_condition_data(alg, "key_dispersed", seed, output_dir, layer)

        if data_c is None or data_d is None:
            print(f"  SKIP: missing data for {alg}")
            continue

        # Extract valid observations from each ordering
        vc = data_c["valid"]
        vd = data_d["valid"]
        n_c = vc.sum()
        n_d = vd.sum()
        n_ret_c = data_c["retained"][vc].sum()
        n_ret_d = data_d["retained"][vd].sum()

        print(f"\n  Clustered: {n_c} valid ({n_ret_c} retained, {n_c - n_ret_c} forgotten, "
              f"{data_c['n_never']} never installed)")
        print(f"  Dispersed: {n_d} valid ({n_ret_d} retained, {n_d - n_ret_d} forgotten, "
              f"{data_d['n_never']} never installed)")

        # --- Per-ordering models first ---
        import statsmodels.api as sm

        for ordering_label, data in [("clustered", data_c), ("dispersed", data_d)]:
            v = data["valid"]
            n_forg = int((1 - data["retained"][v]).sum())
            if n_forg < 10:
                print(f"\n  {ordering_label}: SKIP (only {n_forg} forgotten)")
                continue

            print(f"\n  --- {ordering_label} (per-ordering model) ---")

            y = data["retained"][v]
            X = np.column_stack([
                data["positions"][v] / 1000.0,
                data["margin_1k"][v] / 10.0,  # scale for interpretability
                data["e_norm"][v],
                data["future_key_max_sim"][v],
            ])
            col_names = ["const", "position", "margin_1k/10", "e_norm", "future_max_cos"]

            params, ci_lo, ci_hi = _block_bootstrap_logistic(
                X, y, data["positions"][v], n_boot=2000, block_size=100
            )

            print(f"    N={v.sum()}, n_retained={int(y.sum())}, n_forgotten={int((1-y).sum())}")
            print(f"    {'Feature':<16} {'Coef':>8} {'95% CI':>20} {'Sig':>5}")
            print(f"    {'-'*55}")
            for j, name in enumerate(col_names):
                ci_str = f"[{ci_lo[j]:.3f}, {ci_hi[j]:.3f}]" if ci_lo is not None else "n/a"
                sig = "*" if (ci_lo is not None and (ci_lo[j] > 0 or ci_hi[j] < 0)) else ""
                print(f"    {name:<16} {params[j]:>8.4f} {ci_str:>20} {sig:>5}")

            # Marginal effect: change in P(retained) for 1-SD increase
            X_c = sm.add_constant(X)
            m = sm.Logit(y, X_c).fit(disp=0)
            p_mean = m.predict(X_c).mean()
            print(f"\n    Mean P(retained) = {p_mean:.4f}")
            print(f"    Pseudo-R² = {m.prsquared:.4f}")

            # Effect sizes: OR for 1-SD increase
            sds = X.std(axis=0)
            print(f"\n    Effect sizes (OR per 1-SD increase):")
            for j, name in enumerate(col_names[1:], 1):
                or_val = np.exp(params[j] * sds[j-1])
                print(f"      {name}: OR={or_val:.3f} (1-SD = {sds[j-1]:.4f})")

            all_results[f"{alg}/{ordering_label}"] = {
                "n_valid": int(v.sum()),
                "n_retained": int(y.sum()),
                "n_forgotten": int((1-y).sum()),
                "pseudo_r2": float(m.prsquared),
                "coefs": {n: float(params[j]) for j, n in enumerate(col_names)},
                "ci_lo": {n: float(ci_lo[j]) for j, n in enumerate(col_names)} if ci_lo is not None else None,
                "ci_hi": {n: float(ci_hi[j]) for j, n in enumerate(col_names)} if ci_hi is not None else None,
            }

        # --- Joint model across both orderings ---
        print(f"\n  --- JOINT MODEL (both orderings) ---")

        # Combine data
        y_joint = np.concatenate([data_c["retained"][vc], data_d["retained"][vd]])
        ordering_indicator = np.concatenate([np.zeros(n_c), np.ones(n_d)])  # 1=dispersed
        pos_joint = np.concatenate([data_c["positions"][vc], data_d["positions"][vd]])
        margin_joint = np.concatenate([data_c["margin_1k"][vc], data_d["margin_1k"][vd]])
        e_norm_joint = np.concatenate([data_c["e_norm"][vc], data_d["e_norm"][vd]])
        fmc_joint = np.concatenate([data_c["future_key_max_sim"][vc], data_d["future_key_max_sim"][vd]])

        # Interaction: ordering × future_max_cos
        interaction = ordering_indicator * fmc_joint

        X_joint = np.column_stack([
            ordering_indicator,
            pos_joint / 1000.0,
            margin_joint / 10.0,
            e_norm_joint,
            fmc_joint,
            interaction,
        ])
        joint_col_names = [
            "const", "dispersed", "position", "margin_1k/10",
            "e_norm", "future_max_cos", "dispersed×future_max_cos"
        ]

        # Block bootstrap on joint (blocks within each ordering)
        # Assign blocks: clustered blocks 0-9, dispersed blocks 10-19
        block_ids_joint = np.concatenate([
            (data_c["positions"][vc] / 100).astype(int),
            (data_d["positions"][vd] / 100).astype(int) + 10,
        ])

        params_j, ci_lo_j, ci_hi_j = _block_bootstrap_logistic(
            X_joint, y_joint, block_ids_joint * 100,  # pass as positions for block assignment
            n_boot=2000, block_size=100
        )

        print(f"    N={len(y_joint)} ({n_c} clustered + {n_d} dispersed)")
        print(f"    Retained: {int(y_joint.sum())}, Forgotten: {int((1-y_joint).sum())}")
        print(f"\n    {'Feature':<28} {'Coef':>8} {'95% CI':>20} {'Sig':>5}")
        print(f"    {'-'*65}")
        for j, name in enumerate(joint_col_names):
            ci_str = f"[{ci_lo_j[j]:.3f}, {ci_hi_j[j]:.3f}]" if ci_lo_j is not None else "n/a"
            sig = "*" if (ci_lo_j is not None and (ci_lo_j[j] > 0 or ci_hi_j[j] < 0)) else ""
            print(f"    {name:<28} {params_j[j]:>8.4f} {ci_str:>20} {sig:>5}")

        # Fit with statsmodels for pseudo-R² and LR tests
        X_joint_c = sm.add_constant(X_joint)
        m_joint = sm.Logit(y_joint, X_joint_c).fit(disp=0)
        print(f"\n    Pseudo-R² = {m_joint.prsquared:.4f}")

        # LR test: full model vs model without interaction
        X_no_int = np.column_stack([
            ordering_indicator, pos_joint / 1000.0, margin_joint / 10.0,
            e_norm_joint, fmc_joint,
        ])
        X_no_int_c = sm.add_constant(X_no_int)
        m_no_int = sm.Logit(y_joint, X_no_int_c).fit(disp=0)

        lr_int = 2 * (m_joint.llf - m_no_int.llf)
        from scipy.stats import chi2
        p_int = chi2.sf(lr_int, df=1)
        print(f"    LR test (interaction): χ²={lr_int:.4f}, p={p_int:.4e}")

        # LR test: model with ordering vs without ordering (controlling for all else)
        X_no_ord = np.column_stack([
            pos_joint / 1000.0, margin_joint / 10.0, e_norm_joint, fmc_joint,
        ])
        X_no_ord_c = sm.add_constant(X_no_ord)
        m_no_ord = sm.Logit(y_joint, X_no_ord_c).fit(disp=0)

        lr_ord = 2 * (m_no_int.llf - m_no_ord.llf)
        p_ord = chi2.sf(lr_ord, df=1)
        print(f"    LR test (ordering main effect): χ²={lr_ord:.4f}, p={p_ord:.4e}")

        # Effect sizes: OR
        sds_j = X_joint.std(axis=0)
        print(f"\n    Effect sizes (OR per 1-SD or indicator flip):")
        for j, name in enumerate(joint_col_names[1:], 1):
            if name == "dispersed":
                or_val = np.exp(params_j[j])  # binary indicator
                print(f"      {name}: OR={or_val:.3f} (indicator: 0→1)")
            else:
                or_val = np.exp(params_j[j] * sds_j[j-1])
                print(f"      {name}: OR={or_val:.3f} (1-SD = {sds_j[j-1]:.4f})")

        # Installation quality comparison
        print(f"\n    Installation quality comparison:")
        mean_margin_c = data_c["margin_1k"][vc].mean()
        mean_margin_d = data_d["margin_1k"][vd].mean()
        mean_e_c = data_c["e_norm"][vc].mean()
        mean_e_d = data_d["e_norm"][vd].mean()
        print(f"      Clustered: mean_margin={mean_margin_c:.4f}, mean_e_norm={mean_e_c:.4f}")
        print(f"      Dispersed: mean_margin={mean_margin_d:.4f}, mean_e_norm={mean_e_d:.4f}")

        all_results[f"{alg}/joint"] = {
            "n_total": int(len(y_joint)),
            "n_clustered": int(n_c),
            "n_dispersed": int(n_d),
            "pseudo_r2": float(m_joint.prsquared),
            "coefs": {n: float(params_j[j]) for j, n in enumerate(joint_col_names)},
            "ci_lo": {n: float(ci_lo_j[j]) for j, n in enumerate(joint_col_names)} if ci_lo_j is not None else None,
            "ci_hi": {n: float(ci_hi_j[j]) for j, n in enumerate(joint_col_names)} if ci_hi_j is not None else None,
            "lr_interaction": {"chi2": float(lr_int), "p": float(p_int)},
            "lr_ordering": {"chi2": float(lr_ord), "p": float(p_ord)},
            "installation_quality": {
                "clustered_mean_margin": float(mean_margin_c),
                "dispersed_mean_margin": float(mean_margin_d),
                "clustered_mean_e_norm": float(mean_e_c),
                "dispersed_mean_e_norm": float(mean_e_d),
            },
        }

    # Save
    suffix = "" if layer == DEFAULT_LAYER else f"_layer{layer}"
    out_path = output_dir / f"installation_strength_seed{seed}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    return all_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Installation strength analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help="Which layer to analyze (default: 6)")
    parser.add_argument("--eval_at_1k", action="store_true",
                        help="Evaluate first-1K at 1K checkpoint (GPU)")
    parser.add_argument("--analyze", action="store_true",
                        help="Compute features + fit model (CPU)")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--checkpoint_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths
    if args.checkpoint_base:
        ckpt_base = Path(args.checkpoint_base).expanduser()
    else:
        ckpt_base = get_checkpoint_root()

    output_dir = Path(args.output_dir) if args.output_dir else get_result_root() / "interference"

    if args.eval_at_1k:
        eval_at_1k_checkpoint(args.seed, ckpt_base, output_dir, args.model_name)

    if args.analyze:
        compute_key_geometry_features(args.seed, ckpt_base, output_dir, layer=args.layer)
        run_analysis(args.seed, output_dir, layer=args.layer)

    if not args.eval_at_1k and not args.analyze:
        print("Specify --eval_at_1k (GPU) and/or --analyze (CPU)")
        sys.exit(1)


if __name__ == "__main__":
    main()
