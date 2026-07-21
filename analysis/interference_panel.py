"""Per-edit interference analysis — linking semantic exposure to individual forgetting.

Question answered: Among edits of comparable age, are those exposed to more
subsequent semantic overlap more likely to be forgotten?

This analysis builds an edit–checkpoint panel from failure-curve per-case files,
computes time-censored semantic exposure predictors (subject/relation overlap),
and fits per-trajectory discrete-time hazard models.

Design choices (following reviewer guidance):
  - Three trajectories (seeds 42, 2024, 137) are NOT treated as population inference.
  - Models are fit per-trajectory; report sign consistency and effect-size range.
  - Overlap predictors are time-censored: only edits inserted BEFORE the checkpoint.
  - Age is modeled flexibly (bins or spline) to avoid confounding with overlap.
  - Key cosine similarity requires a GPU forward pass and is handled separately
    (see src/mechanism/compute_keys.py). This script uses semantic proxies only.

Usage:
    uv run python -m analysis.interference_panel
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


def build_panel(seed: int) -> List[dict]:
    """Build the edit–checkpoint panel for one trajectory.

    Each row: {case_id, checkpoint, age, survived, insertion_pos,
               subject_overlap_count, subject_overlap_rate,
               relation_overlap_count, relation_overlap_rate,
               any_subject_overlap, target_margin_at_insertion}
    """
    ordering = load_edit_ordering(seed)
    if not ordering:
        print(f"  [WARN] No edit_ordering.json for seed {seed}")
        return []

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

            panel.append({
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
            })

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

    Model: logit P(survived) = checkpoint_FE + f(age) + β₁·subj_rate +
                                β₂·rel_rate + β₃·target_margin

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


# ─── Negative Controls ───────────────────────────────────────────────────────


def negative_control_pre_overlap(panel: List[dict], ordering: List[int],
                                  metadata: Dict[int, dict]) -> dict:
    """Negative control: overlap from edits BEFORE the focal edit.

    Pre-edit overlap should NOT predict forgetting (it's already baked in).
    """
    pos_of = {cid: pos for pos, cid in enumerate(ordering)}

    # Build subject index
    subject_at_pos = {}
    for pos, cid in enumerate(ordering):
        meta = metadata.get(cid, {})
        subject_at_pos[pos] = meta.get("subject", "")

    for row in panel:
        pos = row["insertion_pos"]
        subj = subject_at_pos.get(pos, "")
        # Count same-subject edits BEFORE this one
        pre_overlap = sum(1 for p in range(pos) if subject_at_pos.get(p) == subj and subj)
        row["pre_subject_overlap"] = pre_overlap

    # Split by high/low pre-overlap
    survived = [r for r in panel if r["survived"] == 1]
    forgotten = [r for r in panel if r["survived"] == 0]

    return {
        "mean_pre_overlap_survived": np.mean([r["pre_subject_overlap"] for r in survived]) if survived else 0,
        "mean_pre_overlap_forgotten": np.mean([r["pre_subject_overlap"] for r in forgotten]) if forgotten else 0,
        "description": "Pre-edit overlap should not predict forgetting (null control)",
    }


def negative_control_future_leak(panel: List[dict], ordering: List[int],
                                   metadata: Dict[int, dict]) -> dict:
    """Negative control: overlap with edits AFTER the checkpoint.

    Future-of-checkpoint overlap is information leakage and should have
    no explanatory value in a well-constructed model.
    """
    pos_of = {cid: pos for pos, cid in enumerate(ordering)}
    subject_at_pos = {}
    for pos, cid in enumerate(ordering):
        meta = metadata.get(cid, {})
        subject_at_pos[pos] = meta.get("subject", "")

    n_total = len(ordering)

    for row in panel:
        pos = row["insertion_pos"]
        checkpoint_pos = row["checkpoint"]  # number of edits at this checkpoint
        max_pos = min(checkpoint_pos, n_total)
        subj = subject_at_pos.get(pos, "")

        # Overlap from edits AFTER checkpoint (future leak)
        future_overlap = sum(1 for p in range(max_pos, n_total)
                            if subject_at_pos.get(p) == subj and subj)
        row["future_subject_overlap"] = future_overlap

    survived = [r for r in panel if r["survived"] == 1]
    forgotten = [r for r in panel if r["survived"] == 0]

    return {
        "mean_future_overlap_survived": np.mean([r["future_subject_overlap"] for r in survived]) if survived else 0,
        "mean_future_overlap_forgotten": np.mean([r["future_subject_overlap"] for r in forgotten]) if forgotten else 0,
        "description": "Future-after-checkpoint overlap should have no explanatory value",
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


def age_matched_comparison(panel: List[dict]) -> dict:
    """Within each age bin, compare overlap between survived and forgotten edits.

    This is the most direct test: among same-age edits, do forgotten ones
    have higher subsequent overlap?
    """
    from collections import defaultdict

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
            "direction_consistent": forg_subj > surv_subj,
        }

    # Summary
    consistent_bins = sum(1 for v in bin_results.values() if v.get("direction_consistent"))
    total_bins = len(bin_results)

    return {
        "per_bin": bin_results,
        "consistent_bins": consistent_bins,
        "total_bins": total_bins,
        "summary": f"Forgotten edits had higher subject overlap in {consistent_bins}/{total_bins} age bins",
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Run the full interference panel analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PER-EDIT INTERFERENCE ANALYSIS")
    print("=" * 60)

    # 1. Build panels
    print("\n[1] Building edit–checkpoint panels...")
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed)
        print(f"  seed {seed}: {len(panel)} panel rows")
        all_panels.extend(panel)

    if not all_panels:
        print("  ERROR: No panel data. Aborting.")
        return

    print(f"  Total: {len(all_panels)} rows across {len(TRAJECTORIES)} trajectories")

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
    age_match = age_matched_comparison(all_panels)
    print(f"  {age_match['summary']}")
    for bin_name, data in age_match.get("per_bin", {}).items():
        direction = "↑" if data["direction_consistent"] else "↓"
        print(f"    {bin_name}: forgotten subj_rate={data['mean_subj_rate_forgotten']:.1f} "
              f"vs survived={data['mean_subj_rate_survived']:.1f} "
              f"(Δ={data['subj_rate_diff']:+.1f}) {direction}")

    # 5. Leave-one-out prediction
    print("\n[5] Leave-one-trajectory-out prediction...")
    loo = leave_one_out_prediction(all_panels)
    for key, val in loo.items():
        if isinstance(val, dict) and "auc_improvement" in val:
            print(f"  {key}: base AUC={val['base_auc']:.3f}, "
                  f"full AUC={val['full_auc']:.3f}, "
                  f"Δ={val['auc_improvement']:+.4f}")

    # 6. Negative controls
    print("\n[6] Negative controls...")
    for seed in TRAJECTORIES:
        ordering = load_edit_ordering(seed)
        if not ordering:
            continue
        # Build metadata from outcomes
        metadata = {}
        for edits in CHECKPOINTS:
            outcomes = load_per_case_outcomes(seed, edits)
            for cid, data in outcomes.items():
                if cid not in metadata:
                    metadata[cid] = {"subject": data["subject"], "relation_id": data["relation_id"]}

        seed_panel = [r for r in all_panels if r["seed"] == seed]

        pre = negative_control_pre_overlap(seed_panel, ordering, metadata)
        print(f"\n  Seed {seed} — Pre-edit overlap (should be null):")
        print(f"    Survived: {pre['mean_pre_overlap_survived']:.2f}, "
              f"Forgotten: {pre['mean_pre_overlap_forgotten']:.2f}")

        future = negative_control_future_leak(seed_panel, ordering, metadata)
        print(f"  Seed {seed} — Future-leak overlap (should be null):")
        print(f"    Survived: {future['mean_future_overlap_survived']:.2f}, "
              f"Forgotten: {future['mean_future_overlap_forgotten']:.2f}")

    # 7. Save results
    output = {
        "panel_size": len(all_panels),
        "trajectories": TRAJECTORIES,
        "checkpoints": CHECKPOINTS,
        "monotonicity": mono,
        "per_trajectory": {str(k): v for k, v in traj_results.items()},
        "age_matched": age_match,
        "leave_one_out": loo,
    }

    out_path = output_dir / "interference_panel_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Summary statement
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check sign consistency
    signs = []
    for seed, res in traj_results.items():
        if res and "subject_overlap_coef" in res:
            signs.append(res["subject_overlap_coef"] < 0)  # negative = more overlap → less survival

    if signs and all(signs):
        effect_sizes = [abs(traj_results[s]["subject_overlap_coef"])
                       for s in TRAJECTORIES if traj_results.get(s, {}).get("subject_overlap_coef")]
        print(f"\n  Subject overlap was negatively associated with retention")
        print(f"  in all {len(signs)} trajectories.")
        if effect_sizes:
            print(f"  Effect sizes (|β|): {min(effect_sizes):.4f} to {max(effect_sizes):.4f}")
        print(f"\n  Interpretation: Among edits of the same age, those with more")
        print(f"  subsequent same-subject edits are more likely to be forgotten.")
    elif signs:
        print(f"\n  Sign NOT consistent across trajectories ({sum(signs)}/{len(signs)} negative).")
        print(f"  Cannot conclude that semantic overlap predicts forgetting.")
    else:
        print(f"\n  Insufficient data to assess sign consistency.")

    print(f"\n  NOTE: This is within-trajectory mechanistic evidence (N={len(TRAJECTORIES)} trajectories),")
    print(f"  not population-level inference. Complements the controlled-coupling experiment.")


def main():
    parser = argparse.ArgumentParser(
        description="Per-edit interference analysis: link semantic exposure to forgetting"
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
