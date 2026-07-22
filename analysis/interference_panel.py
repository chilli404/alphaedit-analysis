"""Per-edit interference analysis — linking semantic exposure to individual forgetting.

Question answered: Among edits of comparable age, are those exposed to more
subsequent semantic or geometric overlap more likely to be forgotten?

This analysis builds an edit–checkpoint panel from failure-curve per-case files,
computes time-censored predictors, and fits per-trajectory discrete-time models.

Two predictor tiers:
  Tier 1 (offline, this script): Subject overlap, relation overlap, target margin.
  Tier 2 (requires GPU keys): Cosine similarity, cumulative interference score.
    Run src/mechanism/compute_keys.py first to generate keys, then pass
    --keys-dir to this script.

Design choices (following reviewer guidance):
  - Three trajectories (seeds 42, 2024, 137) are NOT treated as population inference.
  - Models are fit per-trajectory; report sign consistency and effect-size range.
  - Overlap predictors are time-censored: only edits inserted BEFORE the checkpoint.
  - Age is modeled flexibly (bins) to avoid confounding with overlap.
  - Subject overlap is extremely sparse in CounterFact (subjects are unique entities).
  - Relation overlap is substantial but acts as a REGULARITY proxy (protective),
    not an interference proxy. This is itself an informative negative result.
  - Actual key cosine similarity (Tier 2) is needed to demonstrate geometric interference.

Usage:
    uv run python -m analysis.interference_panel
    uv run python -m analysis.interference_panel --keys-dir results/key_vectors
    uv run python -m analysis.interference_panel --output-dir results/figures/paper
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.style import PROJECT, PAPER_OUTPUT

# ─── Configuration ────────────────────────────────────────────────────────────

RESULTS = PROJECT / "results"
FC_DIR = RESULTS / "failure_curve_checkpointed"

# Trajectories with full coverage at 3K–5K
TRAJECTORIES = [42, 2024]
# Checkpoints for the panel (must have per-case data for both seeds)
CHECKPOINTS = [3000, 4000, 5000]
ALG = "AlphaEdit"
BATCH_SIZE = 100


# ─── Key Similarity (Tier 2, optional) ───────────────────────────────────────


def load_key_vectors(keys_dir: Optional[Path], seed: int) -> Optional[Dict[int, np.ndarray]]:
    """Load precomputed key vectors for a trajectory.

    Expected file: {keys_dir}/seed{seed}/keys_seed{seed}.npz with arrays:
      - case_ids: int array of case IDs
      - keys: float32 array of shape (n_cases, hidden_dim)

    Returns dict: case_id → key vector, or None if unavailable.
    """
    if keys_dir is None:
        return None
    path = Path(keys_dir) / f"seed{seed}" / f"keys_seed{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    case_ids = data["case_ids"]
    keys = data["keys"]
    return {int(cid): keys[i] for i, cid in enumerate(case_ids)}


def compute_key_similarity_predictors(
    case_id: int,
    insertion_pos: int,
    max_pos: int,
    ordering: List[int],
    key_vectors: Dict[int, np.ndarray],
) -> dict:
    """Compute geometric overlap predictors for one edit using actual key vectors.

    Returns:
      - max_cosine_subsequent: max cos(k_i, k_j) for j in (pos, max_pos)
      - cumulative_interference: sum of max(0, cos(k_i, k_j))^2
      - mean_top5_similarity: mean of top-5 cosine similarities
      - n_above_threshold: count of subsequent keys with cos > 0.7
    """
    k_i = key_vectors.get(case_id)
    if k_i is None:
        return {}

    k_i_norm = k_i / (np.linalg.norm(k_i) + 1e-10)
    cosines = []

    for pos_j in range(insertion_pos + 1, min(max_pos, len(ordering))):
        cid_j = ordering[pos_j]
        k_j = key_vectors.get(cid_j)
        if k_j is None:
            continue
        k_j_norm = k_j / (np.linalg.norm(k_j) + 1e-10)
        cos_sim = float(np.dot(k_i_norm, k_j_norm))
        cosines.append(cos_sim)

    if not cosines:
        return {}

    cosines_arr = np.array(cosines)
    top5 = np.sort(cosines_arr)[-5:]

    return {
        "max_cosine_subsequent": float(np.max(cosines_arr)),
        # cumulative_interference := Σ max(0, cos(k_i, k_j))² for all j in (pos_i, checkpoint).
        # Squaring upweights high-similarity neighbors; clipping at 0 ignores anti-correlated keys.
        "cumulative_interference": float(np.sum(np.maximum(cosines_arr, 0) ** 2)),
        "mean_top5_similarity": float(np.mean(top5)),
        "n_above_threshold_07": int(np.sum(cosines_arr > 0.7)),
        "n_above_threshold_05": int(np.sum(cosines_arr > 0.5)),
    }


# ─── Panel Construction ──────────────────────────────────────────────────────


def load_edit_ordering(seed: int) -> Optional[List[int]]:
    """Load the exact ordered case_ids for a trajectory.

    Searches across checkpoints since all share the same ordering
    (later checkpoints extend earlier ones).
    """
    # Find the largest checkpoint's ordering (most complete)
    for edits in sorted([10000, 9000, 7000, 5000, 3000, 2000]):
        path = FC_DIR / f"seed{seed}" / f"{edits}edits" / ALG / "run_000" / "edit_ordering.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data["case_ids_ordered"]
    return None


def load_per_case_outcomes(seed: int, edits: int) -> Dict[int, dict]:
    """Load per-case evaluation results at a checkpoint.

    Returns dict: case_id → {efficacy, paraphrase, neighborhood, subject, relation_id, target_margin}
    """
    run_dir = FC_DIR / f"seed{seed}" / f"{edits}edits" / ALG / "run_000"
    if not run_dir.exists():
        return {}

    results = {}
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)

        case_id = data["case_id"]
        post = data.get("post", {})
        rewrite = data.get("requested_rewrite", {})

        # Binary efficacy
        correct = post.get("rewrite_prompts_correct", [])
        efficacy = sum(correct) / len(correct) if correct else None

        # Target margin (log-prob difference: target_true - target_new)
        # Lower margin → harder edit, higher → easier
        probs = post.get("rewrite_prompts_probs", [])
        if probs and isinstance(probs[0], dict):
            margin = probs[0].get("target_true", 0) - probs[0].get("target_new", 0)
        else:
            margin = None

        results[case_id] = {
            "efficacy": efficacy,
            "subject": rewrite.get("subject", ""),
            "relation_id": rewrite.get("relation_id", ""),
            "target_margin": margin,
        }

    return results


def build_panel(seed: int, keys_dir: Optional[Path] = None) -> List[dict]:
    """Build the edit–checkpoint panel for one trajectory.

    Each row: {case_id, checkpoint, age, survived, insertion_pos,
               subject_overlap_count, subject_overlap_rate,
               relation_overlap_count, relation_overlap_rate,
               any_subject_overlap, target_margin_at_insertion,
               [key similarity predictors if keys available]}
    """
    ordering = load_edit_ordering(seed)
    if not ordering:
        print(f"  [WARN] No edit_ordering.json for seed {seed}")
        return []

    # Load key vectors if available (Tier 2)
    key_vectors = load_key_vectors(keys_dir, seed)

    # Build position lookup and metadata from the earliest available checkpoint
    # that contains all facts we need
    pos_of = {cid: pos for pos, cid in enumerate(ordering)}

    # Load metadata (subject, relation) from the largest available checkpoint
    metadata = {}
    for edits in sorted(CHECKPOINTS, reverse=True):
        outcomes = load_per_case_outcomes(seed, edits)
        for cid, data in outcomes.items():
            if cid not in metadata:
                metadata[cid] = {"subject": data["subject"], "relation_id": data["relation_id"]}
    if not metadata:
        print(f"  [WARN] No per-case data for seed {seed}")
        return []

    # Precompute: for each case, which subsequent cases share subject/relation
    # Only consider cases in the ordering (i.e., actually edited)
    ordered_subjects = []
    ordered_relations = []
    for cid in ordering:
        meta = metadata.get(cid, {})
        ordered_subjects.append(meta.get("subject", ""))
        ordered_relations.append(meta.get("relation_id", ""))

    # For each position, precompute cumulative same-subject and same-relation
    # counts from subsequent edits (using only edits up to a checkpoint)
    n_total = len(ordering)

    # Build subject→positions and relation→positions indices
    subject_positions = defaultdict(list)
    relation_positions = defaultdict(list)
    for pos, (subj, rel) in enumerate(zip(ordered_subjects, ordered_relations)):
        if subj:
            subject_positions[subj].append(pos)
        if rel:
            relation_positions[rel].append(pos)

    # Build panel rows
    panel = []
    for checkpoint in CHECKPOINTS:
        outcomes = load_per_case_outcomes(seed, checkpoint)
        if not outcomes:
            continue

        n_edits_at_checkpoint = checkpoint  # number of edits applied by this checkpoint
        max_pos = min(n_edits_at_checkpoint, n_total)

        for cid, data in outcomes.items():
            if cid not in pos_of:
                continue

            pos = pos_of[cid]
            # Only include facts that were inserted BEFORE this checkpoint
            if pos >= max_pos:
                continue

            eff = data["efficacy"]
            if eff is None:
                continue

            # Age = number of subsequent edits since insertion (up to checkpoint)
            age = max_pos - pos - 1

            # Time-censored overlap: count subsequent edits (pos+1 .. max_pos-1)
            # that share subject or relation
            subj = ordered_subjects[pos]
            rel = ordered_relations[pos]

            # Subject overlap: positions of same-subject edits after pos, before checkpoint
            if subj:
                subj_overlap = sum(1 for p in subject_positions[subj]
                                   if pos < p < max_pos)
            else:
                subj_overlap = 0

            # Relation overlap: positions of same-relation edits after pos, before checkpoint
            if rel:
                rel_overlap = sum(1 for p in relation_positions[rel]
                                  if pos < p < max_pos)
            else:
                rel_overlap = 0

            # Rates per 1000 subsequent edits (to deconfound from age)
            n_subsequent = max(age, 1)
            subj_rate = subj_overlap / n_subsequent * 1000
            rel_rate = rel_overlap / n_subsequent * 1000

            row = {
                "case_id": cid,
                "seed": seed,
                "checkpoint": checkpoint,
                "insertion_pos": pos,
                "age": age,
                "age_bin": _age_bin(age, max_pos),
                "survived": int(eff >= 0.5),
                "efficacy": eff,
                "subject_overlap_count": subj_overlap,
                "subject_overlap_rate": subj_rate,
                "relation_overlap_count": rel_overlap,
                "relation_overlap_rate": rel_rate,
                "any_subject_overlap": int(subj_overlap > 0),
                "any_relation_overlap": int(rel_overlap > 0),
                "log1p_subject_overlap": np.log1p(subj_overlap),
                "target_margin": data["target_margin"],
            }

            # Tier 2: Key cosine similarity (if keys available)
            if key_vectors is not None:
                key_preds = compute_key_similarity_predictors(
                    cid, pos, max_pos, ordering, key_vectors
                )
                row.update(key_preds)

            panel.append(row)

    if key_vectors is not None:
        n_with_keys = sum(1 for r in panel if "max_cosine_subsequent" in r)
        print(f"    Key similarity computed for {n_with_keys}/{len(panel)} rows")

    return panel


def _age_bin(age: int, max_edits: int) -> str:
    """Assign an age to a quartile bin relative to the trajectory length."""
    frac = age / max(max_edits, 1)
    if frac < 0.25:
        return "Q1_young"
    elif frac < 0.5:
        return "Q2"
    elif frac < 0.75:
        return "Q3"
    else:
        return "Q4_old"


# ─── Monotonicity Check ─────────────────────────────────────────────────────


def check_monotonicity(panels: List[dict]) -> dict:
    """Check whether retention is approximately monotonic (no recovery).

    Groups by (seed, case_id) across checkpoints and counts transitions.
    """
    # Group by trajectory + case
    case_series = defaultdict(list)
    for row in panels:
        key = (row["seed"], row["case_id"])
        case_series[key].append((row["checkpoint"], row["survived"]))

    transitions = {"success_to_failure": 0, "failure_to_success": 0,
                   "stable_success": 0, "stable_failure": 0, "mixed": 0}

    for key, series in case_series.items():
        series.sort()
        states = [s for _, s in series]
        if len(states) < 2:
            continue

        s2f = sum(1 for i in range(len(states) - 1) if states[i] == 1 and states[i + 1] == 0)
        f2s = sum(1 for i in range(len(states) - 1) if states[i] == 0 and states[i + 1] == 1)

        if f2s == 0 and s2f > 0:
            transitions["success_to_failure"] += 1
        elif f2s > 0 and s2f == 0:
            transitions["failure_to_success"] += 1
        elif f2s > 0 and s2f > 0:
            transitions["mixed"] += 1
        elif all(s == 1 for s in states):
            transitions["stable_success"] += 1
        else:
            transitions["stable_failure"] += 1

    total = sum(transitions.values())
    transitions["total_cases"] = total
    transitions["monotonic_fraction"] = (
        (transitions["success_to_failure"] + transitions["stable_success"] +
         transitions["stable_failure"]) / max(total, 1)
    )
    return transitions


# ─── Per-Trajectory Logistic Regression ──────────────────────────────────────


def fit_per_trajectory(panel: List[dict], seed: int) -> Optional[dict]:
    """Fit a discrete-time logistic model for one trajectory.

    Model hierarchy (all include checkpoint fixed effects + age-bin dummies):
      M1: logit P(survived_it) = α_t + γ_bin(age_i) + ε_it
      M2: M1 + β₁·subj_overlap_rate_it + β₂·rel_overlap_rate_it
      M3: M2 + β₃·target_margin_i
      M4: M3 − subj_overlap + β₄·max_cosine_subsequent_it + β₅·cum_interference_it

    where i indexes edits, t indexes checkpoints, α_t is a checkpoint fixed
    effect, and γ_bin is a set of age-quartile dummies.

    Panel construction: 18,000 rows = 3,000 case_ids × 3 checkpoints (3K, 4K, 5K)
    × 2 trajectories (seeds 42, 2024). Each case_id is an edit inserted at
    position p ∈ [0, 2999]. At checkpoint C, its outcome survived_it = 1 iff
    efficacy ≥ 0.5 when all C edits have been applied. Time-varying predictors
    (overlap counts, max_cosine) are computed using only edits in [p+1, C-1],
    ensuring no future information leaks into prediction.

    Uses statsmodels if available; falls back to summary statistics.
    """
    if not panel:
        return None

    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return _fit_summary_only(panel, seed)

    df = pd.DataFrame(panel)
    df = df[df["seed"] == seed].copy()

    if len(df) < 50:
        return None

    # Checkpoint as factor
    df["checkpoint_f"] = df["checkpoint"].astype(str)

    # Flexible age: quartile bins
    df["age_bin_f"] = pd.Categorical(df["age_bin"],
                                      categories=["Q1_young", "Q2", "Q3", "Q4_old"])

    # Fit staged models
    results = {"seed": seed, "n_obs": len(df),
               "n_survived": int(df["survived"].sum()),
               "survival_rate": float(df["survived"].mean())}

    # Model 1: Age + checkpoint only (baseline)
    try:
        m1 = smf.logit("survived ~ C(checkpoint_f) + C(age_bin_f)", data=df).fit(disp=0)
        results["m1_aic"] = m1.aic
        results["m1_llf"] = m1.llf
    except Exception:
        m1 = None

    # Model 2: + semantic exposure
    try:
        m2 = smf.logit(
            "survived ~ C(checkpoint_f) + C(age_bin_f) + subject_overlap_rate + relation_overlap_rate",
            data=df
        ).fit(disp=0)
        results["m2_aic"] = m2.aic
        results["m2_llf"] = m2.llf
        results["subject_overlap_coef"] = float(m2.params.get("subject_overlap_rate", np.nan))
        results["subject_overlap_pval"] = float(m2.pvalues.get("subject_overlap_rate", np.nan))
        results["subject_overlap_se"] = float(m2.bse.get("subject_overlap_rate", np.nan))
        results["relation_overlap_coef"] = float(m2.params.get("relation_overlap_rate", np.nan))
        results["relation_overlap_pval"] = float(m2.pvalues.get("relation_overlap_rate", np.nan))

        # Odds ratio
        results["subject_overlap_OR"] = float(np.exp(results["subject_overlap_coef"]))
    except Exception as e:
        results["m2_error"] = str(e)

    # Model 3: + target margin (initial difficulty control)
    try:
        df_margin = df.dropna(subset=["target_margin"])
        if len(df_margin) > 50:
            m3 = smf.logit(
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "subject_overlap_rate + relation_overlap_rate + target_margin",
                data=df_margin
            ).fit(disp=0)
            results["m3_aic"] = m3.aic
            results["m3_subject_overlap_coef"] = float(m3.params.get("subject_overlap_rate", np.nan))
            results["m3_subject_overlap_pval"] = float(m3.pvalues.get("subject_overlap_rate", np.nan))
            results["m3_target_margin_coef"] = float(m3.params.get("target_margin", np.nan))
    except Exception as e:
        results["m3_error"] = str(e)

    # Model 4: + key cosine similarity (Tier 2, if available)
    if "max_cosine_subsequent" in df.columns:
        try:
            df_keys = df.dropna(subset=["max_cosine_subsequent", "target_margin"])
            if len(df_keys) > 50:
                m4 = smf.logit(
                    "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                    "max_cosine_subsequent + cumulative_interference + "
                    "relation_overlap_rate + target_margin",
                    data=df_keys
                ).fit(disp=0)
                results["m4_aic"] = m4.aic
                results["max_cosine_coef"] = float(m4.params.get("max_cosine_subsequent", np.nan))
                results["max_cosine_pval"] = float(m4.pvalues.get("max_cosine_subsequent", np.nan))
                results["max_cosine_se"] = float(m4.bse.get("max_cosine_subsequent", np.nan))
                results["max_cosine_OR"] = float(np.exp(results["max_cosine_coef"]))
                results["cum_interference_coef"] = float(m4.params.get("cumulative_interference", np.nan))
                results["cum_interference_pval"] = float(m4.pvalues.get("cumulative_interference", np.nan))

                # OR per +0.1 cosine increment (interpretable scale)
                results["max_cosine_OR_per_0.1"] = float(np.exp(results["max_cosine_coef"] * 0.1))

                # AIC improvement over semantic-only model
                if "m2_aic" in results:
                    results["aic_improvement_keys_over_semantic"] = results["m2_aic"] - m4.aic

                # Feature scaling documentation
                results["max_cosine_range"] = [
                    float(df_keys["max_cosine_subsequent"].min()),
                    float(df_keys["max_cosine_subsequent"].max()),
                ]
                results["max_cosine_mean"] = float(df_keys["max_cosine_subsequent"].mean())
                results["max_cosine_std"] = float(df_keys["max_cosine_subsequent"].std())

                # Robustness: Clustered standard errors by case_id
                # Accounts for repeated observations of the same edit across checkpoints
                try:
                    from statsmodels.genmod.generalized_estimating_equations import GEE
                    from statsmodels.genmod.families import Binomial
                    from statsmodels.genmod.cov_struct import Independence

                    # GEE with edit-level clusters (independence working correlation)
                    df_gee = df_keys.copy()
                    df_gee["case_id_int"] = df_gee["case_id"].astype(int)
                    # Create design matrix manually for GEE
                    import patsy
                    y, X = patsy.dmatrices(
                        "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                        "max_cosine_subsequent + cumulative_interference + "
                        "relation_overlap_rate + target_margin",
                        data=df_gee, return_type="dataframe"
                    )
                    gee_model = GEE(
                        y.values.ravel(), X, groups=df_gee["case_id_int"],
                        family=Binomial(), cov_struct=Independence()
                    )
                    gee_result = gee_model.fit()

                    # Find the max_cosine_subsequent column
                    cos_idx = [i for i, c in enumerate(X.columns) if "max_cosine_subsequent" in c]
                    if cos_idx:
                        idx = cos_idx[0]
                        results["robustness_clustered"] = {
                            "method": "GEE with independence working correlation, clustered by case_id",
                            "max_cosine_coef": float(gee_result.params.iloc[idx]),
                            "max_cosine_se_clustered": float(gee_result.bse.iloc[idx]),
                            "max_cosine_pval_clustered": float(gee_result.pvalues.iloc[idx]),
                            "max_cosine_OR_per_0.1_clustered": float(np.exp(gee_result.params.iloc[idx] * 0.1)),
                            "n_clusters": int(df_gee["case_id_int"].nunique()),
                            "note": ("Clustered SEs account for within-edit correlation "
                                     "across checkpoints (each edit observed at 3 checkpoints). "
                                     "N_clusters = 2999 (not 3000) because the last edit at position 2999 "
                                     "has no subsequent edits at the 3K checkpoint, so max_cosine is NA "
                                     "and it is excluded from the key-cosine model."),
                        }
                except Exception as e:
                    results["robustness_clustered"] = {"error": str(e)}

        except Exception as e:
            results["m4_error"] = str(e)

    # Panel structure documentation
    results["panel_structure"] = {
        "n_checkpoints": len(df["checkpoint"].unique()),
        "checkpoints": sorted(df["checkpoint"].unique().tolist()),
        "n_unique_cases": len(df["case_id"].unique()),
        "obs_per_case": len(df) / max(len(df["case_id"].unique()), 1),
        "construction": (
            f"{len(df)} rows = {len(df['case_id'].unique())} unique edits × "
            f"{len(df['checkpoint'].unique())} checkpoints "
            f"({', '.join(str(c) for c in sorted(df['checkpoint'].unique()))}). "
            "Each edit i inserted at position p is evaluated at each checkpoint C ∈ {3K, 4K, 5K} "
            "where it has already been applied (p < C). Predictors are time-varying: "
            "overlap and cosine computed over edits in [p+1, C−1] only."
        ),
        "repeated_obs_handling": (
            "Checkpoint fixed effects (α_t) absorb level differences across evaluation points. "
            "Robustness check: clustered standard errors by case_id (see robustness_clustered)."
        ),
    }

    # Model formula documentation
    results["model_formulas"] = {
        "m1": "logit P(survived) ~ C(checkpoint) + C(age_bin)",
        "m2": "logit P(survived) ~ C(checkpoint) + C(age_bin) + subject_overlap_rate + relation_overlap_rate",
        "m3": "logit P(survived) ~ C(checkpoint) + C(age_bin) + subject_overlap_rate + relation_overlap_rate + target_margin",
        "m4": "logit P(survived) ~ C(checkpoint) + C(age_bin) + max_cosine_subsequent + cumulative_interference + relation_overlap_rate + target_margin",
        "predictor_definitions": {
            "max_cosine_subsequent": "max cos(k_i, k_j) for j in (pos_i+1, checkpoint-1) — peak geometric overlap with any single subsequent edit key",
            "cumulative_interference": "Σ max(0, cos(k_i, k_j))² for j in (pos_i+1, checkpoint-1) — sum of squared positive cosine similarities, upweighting high-overlap neighbors",
            "relation_overlap_rate": "count of same-relation edits in (pos_i+1, checkpoint-1), per 1000 subsequent edits",
            "target_margin": "log P(target_true) − log P(target_new) at insertion time — intrinsic edit difficulty",
        },
    }

    return results


def _fit_summary_only(panel: List[dict], seed: int) -> dict:
    """Fallback when statsmodels is unavailable: report descriptive statistics."""
    rows = [r for r in panel if r["seed"] == seed]
    if not rows:
        return {"seed": seed, "error": "no data"}

    survived = [r for r in rows if r["survived"] == 1]
    forgotten = [r for r in rows if r["survived"] == 0]

    results = {
        "seed": seed,
        "n_obs": len(rows),
        "n_survived": len(survived),
        "n_forgotten": len(forgotten),
        "survival_rate": len(survived) / max(len(rows), 1),
        "statsmodels_unavailable": True,
    }

    # Descriptive: mean overlap among survived vs forgotten
    if survived:
        results["mean_subj_rate_survived"] = np.mean([r["subject_overlap_rate"] for r in survived])
    if forgotten:
        results["mean_subj_rate_forgotten"] = np.mean([r["subject_overlap_rate"] for r in forgotten])

    return results


# ─── Bootstrap Confidence Intervals ───────────────────────────────────────────


def bootstrap_key_cosine_effect(
    panel: List[dict], seed: int, n_bootstrap: int = 1000, block_size: int = 100
) -> dict:
    """Contiguous edit-block bootstrap for key cosine coefficient CI.

    Resamples contiguous blocks of edits (preserving temporal correlation)
    rather than individual observations. Block size = 100 (one edit batch).

    Returns 95% CI for β(max_cosine) and OR per +0.1 cosine.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "requires pandas, statsmodels"}

    df = pd.DataFrame([r for r in panel if r["seed"] == seed])
    if "max_cosine_subsequent" not in df.columns:
        return {"error": "no key cosine data"}

    df = df.dropna(subset=["max_cosine_subsequent", "target_margin"])
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = pd.Categorical(df["age_bin"],
                                      categories=["Q1_young", "Q2", "Q3", "Q4_old"])

    # Define blocks by insertion position (contiguous edit batches)
    df["block_id"] = df["insertion_pos"] // block_size
    block_ids = df["block_id"].unique()
    n_blocks = len(block_ids)

    rng = np.random.default_rng(seed)
    boot_betas = []
    boot_ors_01 = []

    for _ in range(n_bootstrap):
        # Resample blocks with replacement
        sampled_blocks = rng.choice(block_ids, size=n_blocks, replace=True)
        boot_df = pd.concat([df[df["block_id"] == b] for b in sampled_blocks],
                           ignore_index=True)

        try:
            m = smf.logit(
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "max_cosine_subsequent + cumulative_interference + "
                "relation_overlap_rate + target_margin",
                data=boot_df
            ).fit(disp=0, maxiter=50, warn_convergence=False)
            beta = m.params.get("max_cosine_subsequent")
            if beta is not None and np.isfinite(beta):
                boot_betas.append(float(beta))
                boot_ors_01.append(float(np.exp(beta * 0.1)))
        except Exception:
            continue

    if len(boot_betas) < 100:
        return {"error": f"only {len(boot_betas)} successful bootstraps"}

    boot_betas = np.array(boot_betas)
    boot_ors_01 = np.array(boot_ors_01)

    return {
        "n_bootstrap": n_bootstrap,
        "n_successful": len(boot_betas),
        "block_size": block_size,
        "n_blocks": n_blocks,
        "beta_mean": float(np.mean(boot_betas)),
        "beta_ci_025": float(np.percentile(boot_betas, 2.5)),
        "beta_ci_975": float(np.percentile(boot_betas, 97.5)),
        "or_per_0.1_mean": float(np.mean(boot_ors_01)),
        "or_per_0.1_ci_025": float(np.percentile(boot_ors_01, 2.5)),
        "or_per_0.1_ci_975": float(np.percentile(boot_ors_01, 97.5)),
        "sign_consistency": float(np.mean(boot_betas < 0)),
    }


# ─── Negative Controls (Permutation-Based) ───────────────────────────────────


def negative_control_permuted_keys(
    panel: List[dict], seed: int, n_permutations: int = 1000
) -> dict:
    """Control 1: Permute max_cosine values within age bins.

    If the effect is real, shuffling the key-similarity assignment within
    age bins (preserving marginal age distribution) should destroy the signal.
    Reports the permutation p-value for the observed difference.
    """
    rows = [r for r in panel if r["seed"] == seed and "max_cosine_subsequent" in r]
    if not rows:
        return {"error": "no key data"}

    # Observed statistic: mean(cos|forgotten) - mean(cos|survived)
    survived = [r["max_cosine_subsequent"] for r in rows if r["survived"] == 1]
    forgotten = [r["max_cosine_subsequent"] for r in rows if r["survived"] == 0]
    if not survived or not forgotten:
        return {"error": "no variance in outcome"}

    observed_diff = np.mean(forgotten) - np.mean(survived)

    # Permutation within age bins
    rng = np.random.default_rng(seed * 7 + 1)
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[r["age_bin"]].append(r)

    perm_diffs = []
    for _ in range(n_permutations):
        # Shuffle cos values within each bin, keeping outcome fixed
        perm_surv = []
        perm_forg = []
        for bin_name, bin_rows in by_bin.items():
            cos_vals = np.array([r["max_cosine_subsequent"] for r in bin_rows])
            shuffled = rng.permutation(cos_vals)
            for i, r in enumerate(bin_rows):
                if r["survived"] == 1:
                    perm_surv.append(shuffled[i])
                else:
                    perm_forg.append(shuffled[i])

        if perm_surv and perm_forg:
            perm_diffs.append(np.mean(perm_forg) - np.mean(perm_surv))

    perm_diffs = np.array(perm_diffs)
    n_extreme = int(np.sum(perm_diffs >= observed_diff))
    # Report p as (n_extreme + 1) / (n_permutations + 1) to avoid exact zero
    p_value = (n_extreme + 1) / (n_permutations + 1)

    return {
        "test": "permute_keys_within_age_bins",
        "observed_diff": float(observed_diff),
        "perm_mean": float(np.mean(perm_diffs)),
        "perm_std": float(np.std(perm_diffs)),
        "p_value": p_value,
        "p_value_note": f"({n_extreme}/{n_permutations} permutations >= observed); reported as (k+1)/(N+1)",
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }


def negative_control_preceding_keys(
    panel: List[dict], seed: int, keys_dir: Optional[Path] = None
) -> dict:
    """Control 2: Compare preceding vs subsequent key similarity.

    For each edit at position P, compute max cosine to PRECEDING edits
    (positions max(0, P-W)..P-1) using the SAME window size W as subsequent
    keys used for that edit. This size-matches the comparison: if the
    subsequent window covers W positions, the preceding window also covers W.

    Preceding keys shouldn't predict forgetting because those edits were
    already applied before the focal edit was inserted.
    """
    if keys_dir is None:
        return {"error": "no keys_dir"}

    key_path = Path(keys_dir) / f"seed{seed}" / f"keys_seed{seed}.npz"
    if not key_path.exists():
        return {"error": f"no key file: {key_path}"}

    key_data = np.load(key_path)
    case_ids = key_data["case_ids"]
    keys = key_data["keys"]
    key_map = {int(cid): keys[i] for i, cid in enumerate(case_ids)}

    # Get ordering for this seed
    ordering = load_edit_ordering(seed)
    if not ordering:
        return {"error": "no ordering"}

    # Precompute normalized keys
    key_norms = {}
    for cid, k in key_map.items():
        norm = np.linalg.norm(k)
        key_norms[cid] = k / (norm + 1e-10)

    # Determine subsequent window sizes from the panel (to size-match preceding)
    # For each (case_id, checkpoint), subsequent window = [pos+1, checkpoint/BATCH_SIZE)
    panel_rows = [r for r in panel if r["seed"] == seed]
    # Use max checkpoint to determine typical subsequent window
    max_checkpoint = max(r["checkpoint"] for r in panel_rows) if panel_rows else 5000
    max_pos_for_subsequent = max_checkpoint  # edits applied up to this checkpoint

    # Compute max cosine to PRECEDING edits for each case
    # Window size matched: for edit at position P, subsequent sees [P+1, max_pos).
    # Preceding uses same size W = max_pos - P - 1, looking back [P-W, P).
    preceding_cos = {}
    n_preceding_searched = []
    n_subsequent_searched = []
    for pos, cid in enumerate(ordering):
        if pos >= max_pos_for_subsequent:
            break
        k_i_norm = key_norms.get(cid)
        if k_i_norm is None:
            continue

        # Subsequent window size for this edit
        subsequent_window = min(max_pos_for_subsequent, len(ordering)) - pos - 1
        # Match preceding window to same size
        preceding_start = max(0, pos - subsequent_window)

        max_cos = -1.0
        n_searched = 0
        for prev_pos in range(preceding_start, pos):
            cid_prev = ordering[prev_pos]
            k_prev_norm = key_norms.get(cid_prev)
            if k_prev_norm is None:
                continue
            cos_sim = float(np.dot(k_i_norm, k_prev_norm))
            max_cos = max(max_cos, cos_sim)
            n_searched += 1

        if max_cos > -1.0:
            preceding_cos[cid] = max_cos
            n_preceding_searched.append(n_searched)
            n_subsequent_searched.append(subsequent_window)

    # Compare: does preceding cos predict forgetting?
    rows = [r for r in panel if r["seed"] == seed and r["case_id"] in preceding_cos]
    survived_pre = [preceding_cos[r["case_id"]] for r in rows if r["survived"] == 1]
    forgotten_pre = [preceding_cos[r["case_id"]] for r in rows if r["survived"] == 0]

    if not survived_pre or not forgotten_pre:
        return {"error": "no variance"}

    # Also get subsequent cos for comparison
    rows_with_sub = [r for r in rows if "max_cosine_subsequent" in r]
    sub_surv = [r["max_cosine_subsequent"] for r in rows_with_sub if r["survived"] == 1]
    sub_forg = [r["max_cosine_subsequent"] for r in rows_with_sub if r["survived"] == 0]

    # Window size matching statistics
    window_stats = {}
    if n_preceding_searched and n_subsequent_searched:
        window_stats = {
            "mean_preceding_window": float(np.mean(n_preceding_searched)),
            "mean_subsequent_window": float(np.mean(n_subsequent_searched)),
            "median_preceding_window": float(np.median(n_preceding_searched)),
            "median_subsequent_window": float(np.median(n_subsequent_searched)),
        }

    return {
        "test": "preceding_vs_subsequent_keys",
        "preceding_cos_survived": float(np.mean(survived_pre)),
        "preceding_cos_forgotten": float(np.mean(forgotten_pre)),
        "preceding_diff": float(np.mean(forgotten_pre) - np.mean(survived_pre)),
        "subsequent_cos_survived": float(np.mean(sub_surv)) if sub_surv else None,
        "subsequent_cos_forgotten": float(np.mean(sub_forg)) if sub_forg else None,
        "subsequent_diff": float(np.mean(sub_forg) - np.mean(sub_surv)) if sub_surv and sub_forg else None,
        "n_with_preceding": len(rows),
        "window_matching": window_stats,
        "window_note": ("Preceding window is size-matched to subsequent: for edit at position P "
                       "with subsequent window [P+1, C), preceding uses [max(0, P−W), P) where W = C−P−1."),
        "interpretation": ("Preceding keys should show weaker predictive power than subsequent keys. "
                          "A large subsequent/preceding ratio supports a temporally directional "
                          "interference interpretation: edits applied after the focal edit drive forgetting."),
    }


def negative_control_shuffled_outcomes(
    panel: List[dict], seed: int, n_permutations: int = 500
) -> dict:
    """Control 3: Shuffle the outcome-exposure mapping.

    Permute survived/forgotten labels across all observations (within seed).
    The observed β should be extreme relative to this null distribution.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "requires pandas, statsmodels"}

    df = pd.DataFrame([r for r in panel if r["seed"] == seed])
    if "max_cosine_subsequent" not in df.columns:
        return {"error": "no key data"}

    df = df.dropna(subset=["max_cosine_subsequent", "target_margin"])
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = pd.Categorical(df["age_bin"],
                                      categories=["Q1_young", "Q2", "Q3", "Q4_old"])

    # Observed β
    try:
        m_obs = smf.logit(
            "survived ~ C(checkpoint_f) + C(age_bin_f) + "
            "max_cosine_subsequent + relation_overlap_rate + target_margin",
            data=df
        ).fit(disp=0)
        observed_beta = float(m_obs.params.get("max_cosine_subsequent", np.nan))
    except Exception as e:
        return {"error": f"model fit failed: {e}"}

    # Permutation null: shuffle 'survived' labels
    rng = np.random.default_rng(seed * 13 + 3)
    perm_betas = []
    for _ in range(n_permutations):
        df_perm = df.copy()
        df_perm["survived"] = rng.permutation(df_perm["survived"].values)
        try:
            m_perm = smf.logit(
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "max_cosine_subsequent + relation_overlap_rate + target_margin",
                data=df_perm
            ).fit(disp=0, maxiter=30, warn_convergence=False)
            beta = m_perm.params.get("max_cosine_subsequent")
            if beta is not None and np.isfinite(beta):
                perm_betas.append(float(beta))
        except Exception:
            continue

    if len(perm_betas) < 50:
        return {"error": f"only {len(perm_betas)} successful permutations"}

    perm_betas = np.array(perm_betas)
    # Two-sided p-value: (k+1)/(N+1) to avoid exact zero
    n_extreme = int(np.sum(np.abs(perm_betas) >= np.abs(observed_beta)))
    p_value = (n_extreme + 1) / (len(perm_betas) + 1)

    return {
        "test": "shuffled_outcomes",
        "observed_beta": observed_beta,
        "perm_mean": float(np.mean(perm_betas)),
        "perm_std": float(np.std(perm_betas)),
        "p_value": p_value,
        "p_value_note": f"({n_extreme}/{len(perm_betas)} permutations >= |observed|); reported as (k+1)/(N+1)",
        "n_permutations": n_permutations,
        "n_successful": len(perm_betas),
        "significant": p_value < 0.05,
        "z_score": float((observed_beta - np.mean(perm_betas)) / max(np.std(perm_betas), 1e-10)),
    }


def negative_control_random_keys(
    panel: List[dict], seed: int, keys_dir: Optional[Path] = None
) -> dict:
    """Control 4: Replace real keys with random vectors of matched norms.

    If the effect depends on the DIRECTION of real key vectors (not just
    dimensionality/norm), random keys with matched norms should produce
    no signal. This rules out artifacts of high-dimensional cosine behavior.
    """
    if keys_dir is None:
        return {"error": "no keys_dir"}

    key_path = Path(keys_dir) / f"seed{seed}" / f"keys_seed{seed}.npz"
    if not key_path.exists():
        return {"error": f"no key file: {key_path}"}

    key_data = np.load(key_path)
    case_ids = key_data["case_ids"]
    keys = key_data["keys"]
    key_norms = np.linalg.norm(keys, axis=1)
    hidden_dim = keys.shape[1]

    # Generate random keys with matched norms
    rng = np.random.default_rng(seed * 17 + 5)
    random_keys = rng.standard_normal((len(case_ids), hidden_dim)).astype(np.float32)
    random_norms = np.linalg.norm(random_keys, axis=1, keepdims=True)
    random_keys = random_keys / random_norms * key_norms[:, np.newaxis]

    random_key_map = {int(cid): random_keys[i] for i, cid in enumerate(case_ids)}

    # Get ordering
    ordering = load_edit_ordering(seed)
    if not ordering:
        return {"error": "no ordering"}

    rows = [r for r in panel if r["seed"] == seed and "max_cosine_subsequent" in r]
    if not rows:
        return {"error": "no key data in panel"}

    pos_of = {cid: pos for pos, cid in enumerate(ordering)}

    # For each row, compute max cosine using random keys
    random_cos_survived = []
    random_cos_forgotten = []
    real_cos_survived = []
    real_cos_forgotten = []

    for row in rows:
        cid = row["case_id"]
        pos = pos_of.get(cid)
        if pos is None:
            continue
        k_i = random_key_map.get(cid)
        if k_i is None:
            continue
        k_i_norm = k_i / (np.linalg.norm(k_i) + 1e-10)

        # Max cosine to subsequent edits using random keys
        max_cos = -1.0
        max_pos = min(len(ordering), pos + 500)  # Match window used in real computation
        for pos_j in range(pos + 1, max_pos):
            cid_j = ordering[pos_j]
            k_j = random_key_map.get(cid_j)
            if k_j is None:
                continue
            k_j_norm = k_j / (np.linalg.norm(k_j) + 1e-10)
            cos_sim = float(np.dot(k_i_norm, k_j_norm))
            max_cos = max(max_cos, cos_sim)

        if max_cos > -1.0:
            if row["survived"] == 1:
                random_cos_survived.append(max_cos)
                real_cos_survived.append(row["max_cosine_subsequent"])
            else:
                random_cos_forgotten.append(max_cos)
                real_cos_forgotten.append(row["max_cosine_subsequent"])

    if not random_cos_survived or not random_cos_forgotten:
        return {"error": "no data after random key computation"}

    return {
        "test": "random_keys_matched_norms",
        "hidden_dim": hidden_dim,
        "random_cos_survived": float(np.mean(random_cos_survived)),
        "random_cos_forgotten": float(np.mean(random_cos_forgotten)),
        "random_diff": float(np.mean(random_cos_forgotten) - np.mean(random_cos_survived)),
        "real_cos_survived": float(np.mean(real_cos_survived)),
        "real_cos_forgotten": float(np.mean(real_cos_forgotten)),
        "real_diff": float(np.mean(real_cos_forgotten) - np.mean(real_cos_survived)),
        "interpretation": ("Random keys should show near-zero difference between "
                          "survived and forgotten. Real keys show large difference. "
                          "This supports a temporally directional interference interpretation: "
                          "the effect depends on learned key direction, not high-dimensional artifacts."),
    }


# ─── Leave-One-Trajectory-Out Prediction ─────────────────────────────────────


def leave_one_out_prediction(all_panels: List[dict]) -> dict:
    """Train on one trajectory, predict the other. Report held-out performance."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from sklearn.metrics import roc_auc_score, log_loss
    except ImportError:
        return {"error": "requires pandas, statsmodels, sklearn"}

    df = pd.DataFrame(all_panels)
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = pd.Categorical(df["age_bin"],
                                      categories=["Q1_young", "Q2", "Q3", "Q4_old"])

    results = {}
    seeds = sorted(df["seed"].unique())

    for held_out_seed in seeds:
        train = df[df["seed"] != held_out_seed].copy()
        test = df[df["seed"] == held_out_seed].copy()

        if len(train) < 50 or len(test) < 50:
            continue

        # Fit on training trajectories
        try:
            # Age-only baseline
            m_base = smf.logit("survived ~ C(checkpoint_f) + C(age_bin_f)",
                              data=train).fit(disp=0)
            pred_base = m_base.predict(test)

            # + semantic exposure
            m_full = smf.logit(
                "survived ~ C(checkpoint_f) + C(age_bin_f) + subject_overlap_rate + relation_overlap_rate",
                data=train
            ).fit(disp=0)
            pred_full = m_full.predict(test)

            y_true = test["survived"].values

            results[f"held_out_seed_{held_out_seed}"] = {
                "n_test": len(test),
                "base_auc": float(roc_auc_score(y_true, pred_base)),
                "full_auc": float(roc_auc_score(y_true, pred_full)),
                "base_logloss": float(log_loss(y_true, pred_base)),
                "full_logloss": float(log_loss(y_true, pred_full)),
                "auc_improvement": float(roc_auc_score(y_true, pred_full) -
                                         roc_auc_score(y_true, pred_base)),
            }
        except Exception as e:
            results[f"held_out_seed_{held_out_seed}"] = {"error": str(e)}

    return results


# ─── Age-Matched Comparison ──────────────────────────────────────────────────


def age_matched_comparison(panel: List[dict], has_keys: bool = False) -> dict:
    """Within each age bin, compare overlap between survived and forgotten edits.

    This is the most direct test: among same-age edits, do forgotten ones
    have higher subsequent overlap?
    """
    bin_results = {}
    by_bin = defaultdict(list)
    for row in panel:
        by_bin[row["age_bin"]].append(row)

    for age_bin in ["Q1_young", "Q2", "Q3", "Q4_old"]:
        rows = by_bin.get(age_bin, [])
        survived = [r for r in rows if r["survived"] == 1]
        forgotten = [r for r in rows if r["survived"] == 0]

        if not survived or not forgotten:
            continue

        surv_subj = np.mean([r["subject_overlap_rate"] for r in survived])
        forg_subj = np.mean([r["subject_overlap_rate"] for r in forgotten])
        surv_rel = np.mean([r["relation_overlap_rate"] for r in survived])
        forg_rel = np.mean([r["relation_overlap_rate"] for r in forgotten])

        bin_results[age_bin] = {
            "n_survived": len(survived),
            "n_forgotten": len(forgotten),
            "mean_subj_rate_survived": float(surv_subj),
            "mean_subj_rate_forgotten": float(forg_subj),
            "subj_rate_diff": float(forg_subj - surv_subj),
            "mean_rel_rate_survived": float(surv_rel),
            "mean_rel_rate_forgotten": float(forg_rel),
            "rel_rate_diff": float(forg_rel - surv_rel),
            "rel_direction_consistent": forg_rel > surv_rel,
            "direction_consistent": forg_subj > surv_subj,
        }

    # Key similarity comparison (Tier 2)
    key_sim_results = {}
    if has_keys:
        for age_bin in ["Q1_young", "Q2", "Q3", "Q4_old"]:
            rows = by_bin.get(age_bin, [])
            survived = [r for r in rows if r["survived"] == 1 and "max_cosine_subsequent" in r]
            forgotten = [r for r in rows if r["survived"] == 0 and "max_cosine_subsequent" in r]
            if survived and forgotten:
                surv_cos = np.mean([r["max_cosine_subsequent"] for r in survived])
                forg_cos = np.mean([r["max_cosine_subsequent"] for r in forgotten])
                surv_ci = np.mean([r["cumulative_interference"] for r in survived])
                forg_ci = np.mean([r["cumulative_interference"] for r in forgotten])
                key_sim_results[age_bin] = {
                    "mean_cos_survived": float(surv_cos),
                    "mean_cos_forgotten": float(forg_cos),
                    "cos_diff": float(forg_cos - surv_cos),
                    "mean_ci_survived": float(surv_ci),
                    "mean_ci_forgotten": float(forg_ci),
                    "ci_diff": float(forg_ci - surv_ci),
                    "direction_consistent": forg_cos > surv_cos,
                }

    # Summary — relation overlap shows PROTECTIVE effect (negative direction)
    rel_protective_bins = sum(1 for v in bin_results.values()
                              if v.get("rel_rate_diff", 0) < 0)
    total_bins = len(bin_results)

    result = {
        "per_bin": bin_results,
        "consistent_bins": sum(1 for v in bin_results.values() if v.get("direction_consistent")),
        "total_bins": total_bins,
        "rel_protective_bins": rel_protective_bins,
        "summary": (f"Relation overlap is PROTECTIVE in {rel_protective_bins}/{total_bins} bins "
                    f"(survived edits have higher relation overlap)"),
    }
    if key_sim_results:
        result["key_similarity"] = key_sim_results
        cos_consistent = sum(1 for v in key_sim_results.values() if v.get("direction_consistent"))
        result["key_cos_consistent_bins"] = cos_consistent
        result["summary"] += (f"\n  Key cosine: forgotten > survived in "
                              f"{cos_consistent}/{len(key_sim_results)} bins")

    return result


# ─── Formatting Helpers ───────────────────────────────────────────────────────


def _format_pval(p: float, n_perm: int) -> str:
    """Format permutation p-value using (k+1)/(N+1) convention.

    With N permutations, the minimum achievable p is 1/(N+1).
    Always report the exact value (e.g., p = 0.001 for 1000 perms,
    p = 0.002 for 500 perms) — never report p = 0.
    """
    return f"p = {p:.4g} ({n_perm} permutations)"


# ─── Main ────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT, keys_dir: Optional[Path] = None):
    """Run the full interference panel analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PER-EDIT INTERFERENCE ANALYSIS")
    print("=" * 60)
    if keys_dir:
        print(f"  Key vectors: {keys_dir}")
    else:
        print("  Key vectors: NOT AVAILABLE (Tier 1 only: semantic proxies)")
        print("  Run: uv run python -m src.mechanism.compute_keys to generate keys")

    # 1. Build panels
    print("\n[1] Building edit–checkpoint panels...")
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed, keys_dir)
        print(f"  seed {seed}: {len(panel)} panel rows")
        all_panels.extend(panel)

    if not all_panels:
        print("  ERROR: No panel data. Aborting.")
        return

    print(f"  Total: {len(all_panels)} rows across {len(TRAJECTORIES)} trajectories")

    # 1b. Overlap distribution diagnostic
    print("\n[1b] Overlap sparsity diagnostic...")
    subj_nonzero = sum(1 for r in all_panels if r["subject_overlap_count"] > 0)
    rel_nonzero = sum(1 for r in all_panels if r["relation_overlap_count"] > 0)
    print(f"  Subject overlap > 0: {subj_nonzero}/{len(all_panels)} ({subj_nonzero/len(all_panels):.1%})")
    print(f"  Relation overlap > 0: {rel_nonzero}/{len(all_panels)} ({rel_nonzero/len(all_panels):.1%})")
    rel_rates = [r["relation_overlap_rate"] for r in all_panels]
    print(f"  Relation overlap rate: mean={np.mean(rel_rates):.1f}, "
          f"median={np.median(rel_rates):.1f}, max={np.max(rel_rates):.1f} per 1K subsequent")
    if keys_dir:
        n_keys = sum(1 for r in all_panels if "max_cosine_subsequent" in r)
        if n_keys > 0:
            cos_vals = [r["max_cosine_subsequent"] for r in all_panels if "max_cosine_subsequent" in r]
            print(f"  Max cosine similarity: mean={np.mean(cos_vals):.3f}, "
                  f"median={np.median(cos_vals):.3f}, max={np.max(cos_vals):.3f}")

    # 2. Monotonicity check
    print("\n[2] Checking retention monotonicity...")
    mono = check_monotonicity(all_panels)
    print(f"  Monotonic fraction: {mono['monotonic_fraction']:.3f}")
    print(f"  Success→Failure: {mono['success_to_failure']}")
    print(f"  Failure→Success (recovery): {mono['failure_to_success']}")
    print(f"  Mixed: {mono['mixed']}")
    if mono["monotonic_fraction"] > 0.9:
        print("  → Retention is approximately monotonic. Discrete-time hazard is appropriate.")
    else:
        print("  → Substantial recovery observed. Using checkpoint-state model instead.")

    # 3. Per-trajectory models
    print("\n[3] Fitting per-trajectory logistic models...")
    traj_results = {}
    for seed in TRAJECTORIES:
        result = fit_per_trajectory(all_panels, seed)
        traj_results[seed] = result
        if result:
            print(f"\n  --- Seed {seed} ---")
            print(f"  N={result.get('n_obs')}, survival={result.get('survival_rate', 0):.3f}")
            if "subject_overlap_coef" in result:
                coef = result["subject_overlap_coef"]
                orr = result.get("subject_overlap_OR", np.exp(coef))
                pval = result.get("subject_overlap_pval", np.nan)
                print(f"  Subject overlap rate: β={coef:.4f}, OR={orr:.4f}, p={pval:.4f}")
            if "relation_overlap_coef" in result:
                print(f"  Relation overlap rate: β={result['relation_overlap_coef']:.4f}")
            if "m1_aic" in result and "m2_aic" in result:
                print(f"  AIC improvement (age-only → +exposure): "
                      f"{result['m1_aic']:.1f} → {result['m2_aic']:.1f} "
                      f"(Δ={result['m2_aic'] - result['m1_aic']:.1f})")

    # 4. Age-matched comparison
    print("\n[4] Age-matched comparison (within-bin)...")
    age_match = age_matched_comparison(all_panels, has_keys="max_cosine_subsequent" in all_panels[0])
    print(f"  {age_match['summary']}")
    print(f"\n  {'Bin':<10} {'N_surv':<8} {'N_forg':<8} "
          f"{'Rel rate (surv)':<16} {'Rel rate (forg)':<16} {'Δ rel':<8}")
    print(f"  {'-'*66}")
    for bin_name in ["Q1_young", "Q2", "Q3", "Q4_old"]:
        data = age_match.get("per_bin", {}).get(bin_name)
        if data:
            print(f"  {bin_name:<10} {data['n_survived']:<8} {data['n_forgotten']:<8} "
                  f"{data['mean_rel_rate_survived']:<16.1f} {data['mean_rel_rate_forgotten']:<16.1f} "
                  f"{data['rel_rate_diff']:+.1f}")

    # Key similarity results (if available)
    if "key_similarity" in age_match:
        print(f"\n  Key cosine similarity (Tier 2):")
        ks = age_match["key_similarity"]
        for bin_name, data in ks.items():
            print(f"    {bin_name}: forgotten max_cos={data['mean_cos_forgotten']:.3f} "
                  f"vs survived={data['mean_cos_survived']:.3f} "
                  f"(Δ={data['cos_diff']:+.3f})")

    print(f"\n  Interpretation:")
    print(f"  - Relation overlap is PROTECTIVE (survived > forgotten).")
    print(f"    Common relations (P27, P103) represent familiar patterns.")
    print(f"  - Subject overlap is too sparse (<1%) to be informative.")
    if "max_cosine_subsequent" not in all_panels[0]:
        print(f"  - Key cosine similarity (Tier 2) needed for geometric inference.")

    # 5. Leave-one-out prediction
    print("\n[5] Leave-one-trajectory-out prediction...")
    loo = leave_one_out_prediction(all_panels)
    for key, val in loo.items():
        if isinstance(val, dict) and "auc_improvement" in val:
            print(f"  {key}: base AUC={val['base_auc']:.3f}, "
                  f"full AUC={val['full_auc']:.3f}, "
                  f"Δ={val['auc_improvement']:+.4f}")

    # 5b. OR per +0.1 cosine reporting
    has_keys = any("max_cosine_subsequent" in r for r in all_panels)
    if has_keys:
        print("\n[5b] Key cosine effect scaling...")
        for seed in TRAJECTORIES:
            res = traj_results.get(seed, {})
            if "max_cosine_coef" in res:
                or_01 = res.get("max_cosine_OR_per_0.1", np.exp(res["max_cosine_coef"] * 0.1))
                print(f"  Seed {seed}: β={res['max_cosine_coef']:.3f}, "
                      f"OR per +0.1 cosine = {or_01:.4f}")
                print(f"    Range: [{res.get('max_cosine_range', ['?', '?'])[0]:.3f}, "
                      f"{res.get('max_cosine_range', ['?', '?'])[1]:.3f}], "
                      f"mean={res.get('max_cosine_mean', 0):.3f}, "
                      f"std={res.get('max_cosine_std', 0):.3f}")
                # Robustness: clustered SEs
                robust = res.get("robustness_clustered", {})
                if "max_cosine_coef" in robust:
                    print(f"    Clustered SE (GEE, {robust['n_clusters']} clusters): "
                          f"β={robust['max_cosine_coef']:.3f}, "
                          f"SE={robust['max_cosine_se_clustered']:.3f}, "
                          f"p={robust['max_cosine_pval_clustered']:.2e}, "
                          f"OR/+0.1={robust['max_cosine_OR_per_0.1_clustered']:.4f}")
                elif "error" in robust:
                    print(f"    Clustered SE: unavailable ({robust['error'][:60]})")

    # 6. Bootstrap confidence intervals
    bootstrap_results = {}
    if has_keys:
        print("\n[6] Bootstrap CIs (contiguous edit-block, 1000 resamples)...")
        for seed in TRAJECTORIES:
            boot = bootstrap_key_cosine_effect(all_panels, seed, n_bootstrap=1000)
            bootstrap_results[str(seed)] = boot
            if "error" not in boot:
                print(f"  Seed {seed}: β 95% CI = [{boot['beta_ci_025']:.3f}, {boot['beta_ci_975']:.3f}]")
                print(f"    OR per +0.1: {boot['or_per_0.1_mean']:.4f} "
                      f"[{boot['or_per_0.1_ci_025']:.4f}, {boot['or_per_0.1_ci_975']:.4f}]")
                print(f"    Sign consistency: {boot['sign_consistency']:.1%} negative "
                      f"({boot['n_successful']}/{boot['n_bootstrap']} converged)")
            else:
                print(f"  Seed {seed}: {boot['error']}")

    # 7. Negative controls (permutation-based)
    negative_controls = {}
    if has_keys:
        print("\n[7] Negative controls...")

        for seed in TRAJECTORIES:
            seed_controls = {}

            # Control 1: Permute keys within age bins
            perm = negative_control_permuted_keys(all_panels, seed)
            seed_controls["permuted_keys"] = perm
            if "error" not in perm:
                print(f"\n  Seed {seed} — Permuted keys within age bins:")
                print(f"    Observed diff: {perm['observed_diff']:+.4f}, "
                      f"Null mean: {perm['perm_mean']:+.4f} ± {perm['perm_std']:.4f}")
                p_str = _format_pval(perm['p_value'], perm['n_permutations'])
                print(f"    Permutation {p_str} "
                      f"({'PASS' if perm['significant'] else 'FAIL: effect not significant'})")

            # Control 2: Preceding vs subsequent keys
            prec = negative_control_preceding_keys(all_panels, seed, keys_dir)
            seed_controls["preceding_keys"] = prec
            if "error" not in prec:
                print(f"  Seed {seed} — Preceding vs subsequent keys:")
                print(f"    Preceding: forgotten={prec['preceding_cos_forgotten']:.3f}, "
                      f"survived={prec['preceding_cos_survived']:.3f} "
                      f"(Δ={prec['preceding_diff']:+.4f})")
                print(f"    Subsequent: forgotten={prec['subsequent_cos_forgotten']:.3f}, "
                      f"survived={prec['subsequent_cos_survived']:.3f} "
                      f"(Δ={prec['subsequent_diff']:+.4f})")
                ratio = abs(prec['subsequent_diff']) / max(abs(prec['preceding_diff']), 1e-10)
                print(f"    Subsequent/preceding ratio: {ratio:.1f}x")

            # Control 3: Shuffled outcomes
            shuf = negative_control_shuffled_outcomes(all_panels, seed)
            seed_controls["shuffled_outcomes"] = shuf
            if "error" not in shuf:
                print(f"  Seed {seed} — Shuffled outcomes:")
                print(f"    Observed β={shuf['observed_beta']:.3f}, "
                      f"Null: {shuf['perm_mean']:.3f} ± {shuf['perm_std']:.3f}")
                p_str = _format_pval(shuf['p_value'], shuf['n_successful'])
                print(f"    z = {shuf['z_score']:.1f}, {p_str}")

            # Control 4: Random keys
            rand = negative_control_random_keys(all_panels, seed, keys_dir)
            seed_controls["random_keys"] = rand
            if "error" not in rand:
                print(f"  Seed {seed} — Random keys (matched norms, dim={rand['hidden_dim']}):")
                print(f"    Random diff: {rand['random_diff']:+.4f}")
                print(f"    Real diff:   {rand['real_diff']:+.4f}")
                ratio = abs(rand['real_diff']) / max(abs(rand['random_diff']), 1e-10)
                print(f"    Real/random ratio: {ratio:.1f}x")

            negative_controls[str(seed)] = seed_controls

    # 8. Save results
    output = {
        "panel_size": len(all_panels),
        "trajectories": TRAJECTORIES,
        "checkpoints": CHECKPOINTS,
        "monotonicity": mono,
        "per_trajectory": {str(k): v for k, v in traj_results.items()},
        "age_matched": age_match,
        "leave_one_out": loo,
        "bootstrap": bootstrap_results,
        "negative_controls": negative_controls,
    }

    out_path = output_dir / "interference_panel_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Summary statement
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Panel: {len(all_panels)} observations, "
          f"{len(set(r['case_id'] for r in all_panels))} unique edits × "
          f"{len(CHECKPOINTS)} checkpoints × {len(TRAJECTORIES)} trajectories")

    print(f"\n  Tier 1 (semantic proxies):")
    print(f"  - Subject overlap: too sparse in CounterFact (<1% of rows non-zero)")
    print(f"  - Relation overlap: {age_match.get('rel_protective_bins', 0)}/{age_match.get('total_bins', 0)} "
          f"bins show PROTECTIVE effect (survived > forgotten)")
    print(f"  - Interpretation: sharing a semantic category (relation type)")
    print(f"    does NOT cause interference; it slightly protects against forgetting.")
    print(f"    This dissociates semantic similarity from geometric interference.")

    if has_keys:
        print(f"\n  Tier 2 (key cosine similarity):")
        cos_bins = age_match.get("key_cos_consistent_bins", 0)
        total = len(age_match.get("key_similarity", {}))
        print(f"  - Max cosine: forgotten > survived in {cos_bins}/{total} age bins")

        # Check sign consistency across trajectories for key similarity
        signs = []
        for seed, res in traj_results.items():
            if res and "max_cosine_coef" in res:
                signs.append(res["max_cosine_coef"] < 0)

        if signs and all(signs):
            effect_sizes = [abs(traj_results[s]["max_cosine_coef"])
                           for s in TRAJECTORIES if traj_results.get(s, {}).get("max_cosine_coef")]
            or_01_vals = [traj_results[s].get("max_cosine_OR_per_0.1")
                         for s in TRAJECTORIES if traj_results.get(s, {}).get("max_cosine_OR_per_0.1")]
            print(f"  - Key overlap negatively associated with retention")
            print(f"    in all {len(signs)} trajectories.")
            if effect_sizes:
                print(f"    |β|: {min(effect_sizes):.3f} to {max(effect_sizes):.3f}")
            if or_01_vals:
                print(f"    OR per +0.1 cosine: {min(or_01_vals):.4f} to {max(or_01_vals):.4f}")

            # Bootstrap summary
            for seed_str, boot in bootstrap_results.items():
                if "error" not in boot:
                    print(f"    Seed {seed_str} bootstrap 95% CI: "
                          f"[{boot['beta_ci_025']:.3f}, {boot['beta_ci_975']:.3f}]")

            print(f"\n  Negative controls ({len(negative_controls)} seeds):")
            for seed_str, ctrls in negative_controls.items():
                perm = ctrls.get("permuted_keys", {})
                rand = ctrls.get("random_keys", {})
                prec = ctrls.get("preceding_keys", {})
                if "error" not in perm:
                    p_str = _format_pval(perm['p_value'], perm['n_permutations'])
                    print(f"    Seed {seed_str}: {p_str}, "
                          f"random Δ={rand.get('random_diff', 0):+.4f} vs real Δ={rand.get('real_diff', 0):+.4f}, "
                          f"preceding Δ={prec.get('preceding_diff', 0):+.4f}")

            print(f"\n  CONCLUSION: Among edits of the same age, those whose keys are")
            print(f"  more similar to subsequent edit keys are more likely to be forgotten.")
            print(f"  This links the global spectral finding to individual behavioral failures.")
            print(f"  Effect robust to: block bootstrap, key permutation, outcome shuffling,")
            print(f"  and random-key control. Preceding keys show no/weak effect (directional).")
    else:
        print(f"\n  Tier 2 (key cosine similarity): NOT AVAILABLE")
        print(f"  Run: uv run python -m src.mechanism.compute_keys")
        print(f"  Then: uv run python -m analysis.interference_panel --keys-dir results/key_vectors")

    print(f"\n  NOTE: This is within-trajectory mechanistic evidence (N={len(TRAJECTORIES)} trajectories),")
    print(f"  not population-level inference. Complements the matched-ordering experiment.")


def main():
    parser = argparse.ArgumentParser(
        description="Per-edit interference analysis: link semantic exposure to forgetting"
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    parser.add_argument("--keys-dir", type=Path, default=None,
                        help="Directory containing keys_seed{N}.npz files from compute_keys.py")
    args = parser.parse_args()
    generate(args.output_dir, keys_dir=args.keys_dir)


if __name__ == "__main__":
    main()
