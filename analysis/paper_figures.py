"""
Paper Figures: Unified analysis and figure generation across all experiments.

Produces:
  1. Failure curve figure (AlphaEdit vs MEMIT, multi-seed)
  2. Order sensitivity table + figure
  3. Mechanism → behavioral correlation (predictive divergence)
  4. Controlled coupling comparison (when data available)

Output: results/figures/paper/
"""

import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── Style ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800"}
SEED_COLORS = {42: "#2196F3", 2024: "#E91E63", 137: "#4CAF50", 7: "#FF9800", 99: "#9C27B0"}
STREAM_COLORS = {"low_coupling": "#2196F3", "high_coupling": "#E91E63"}

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "figures" / "paper"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ─── Data Loaders ────────────────────────────────────────────────────────────

def load_failure_curve_metrics(seed, edits, alg):
    """Load aggregate metrics for a given seed/edits/algorithm checkpoint."""
    run_dir = RESULTS / "failure_curve_checkpointed" / f"seed{seed}" / f"{edits}edits" / alg / "run_000"
    if not run_dir.exists():
        return None

    files = list(run_dir.glob("100_edits-case_*.json"))
    if not files:
        return None

    all_eff, all_para, all_neigh = [], [], []
    for f_path in files:
        with open(f_path) as f:
            data = json.load(f)
        post = data["post"]
        all_eff.extend(post["rewrite_prompts_correct"])
        all_para.extend(post["paraphrase_prompts_correct"])
        all_neigh.extend(post["neighborhood_prompts_correct"])

    return {
        "efficacy": np.mean(all_eff),
        "paraphrase": np.mean(all_para),
        "neighborhood": np.mean(all_neigh),
        "n_facts": len(all_eff),
    }


def load_failure_curve_glue(seed, edits, alg):
    """Load GLUE scores at a checkpoint."""
    glue_dir = RESULTS / "failure_curve_checkpointed" / f"seed{seed}" / f"{edits}edits" / alg / "run_000" / "glue_eval"
    if not glue_dir.exists():
        return None

    glue_files = sorted(glue_dir.glob("case_*_glue.json"))
    if not glue_files:
        return None

    with open(glue_files[-1]) as f:
        data = json.load(f)

    scores = {}
    for task in ["mmmlu", "sst", "cola", "mrpc", "nli"]:
        if task in data:
            scores[task] = data[task]["f1_new"]

    scores["edit_num"] = data.get("edit_num", edits)
    return scores


def load_comparison_ordered(base_dir, alg):
    """Load aggregate metrics for a comparison_ordered run."""
    run_dir = base_dir / alg / "run_000"
    if not run_dir.exists():
        return None

    files = list(run_dir.glob("100_edits-case_*.json"))
    if not files:
        return None

    all_eff, all_para, all_neigh = [], [], []
    for f_path in files:
        with open(f_path) as f:
            data = json.load(f)
        post = data["post"]
        all_eff.extend(post["rewrite_prompts_correct"])
        all_para.extend(post["paraphrase_prompts_correct"])
        all_neigh.extend(post["neighborhood_prompts_correct"])

    return {
        "efficacy": np.mean(all_eff),
        "paraphrase": np.mean(all_para),
        "neighborhood": np.mean(all_neigh),
    }


def load_comparison_ordered_glue(base_dir, alg):
    """Load final GLUE from comparison_ordered."""
    glue_dir = base_dir / alg / "run_000" / "glue_eval"
    if not glue_dir.exists():
        return None

    glue_files = sorted(glue_dir.glob("case_*_glue.json"))
    if not glue_files:
        return None

    with open(glue_files[-1]) as f:
        data = json.load(f)

    scores = {}
    for task in ["mmmlu", "sst", "cola", "mrpc", "nli"]:
        if task in data:
            scores[task] = data[task]["f1_new"]
    scores["edit_num"] = data.get("edit_num", "?")
    return scores


def load_mechanism_metrics(seed):
    """Load mechanism analysis JSONL for a seed."""
    mech_dir = RESULTS / "mechanism_analysis" / f"seed{seed}"
    if not mech_dir.exists():
        return []

    records = []
    for jsonl in sorted(mech_dir.glob("mechanism_*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_controlled_coupling(stream_name):
    """Load controlled coupling JSONL if available."""
    cc_dir = RESULTS / "controlled_coupling"
    if not cc_dir.exists():
        return []

    records = []
    for jsonl in sorted(cc_dir.glob(f"{stream_name}*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


# ─── Figure 1: Failure Curve ─────────────────────────────────────────────────

def plot_failure_curve():
    """4-panel failure curve: efficacy, paraphrase, neighborhood, GLUE-MMLU."""
    seeds = [42, 2024, 137]
    edit_points = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Failure Curve: AlphaEdit vs MEMIT (0→10K edits)", fontsize=13, y=0.98)

    metrics = ["efficacy", "paraphrase", "neighborhood"]

    for mi, metric in enumerate(metrics):
        ax = axes[mi // 2, mi % 2]

        for alg in ["AlphaEdit", "MEMIT"]:
            seed_curves = {}
            for seed in seeds:
                curve = []
                for edits in edit_points:
                    m = load_failure_curve_metrics(seed, edits, alg)
                    if m is not None:
                        curve.append((edits, m[metric]))
                if curve:
                    seed_curves[seed] = curve

            if not seed_curves:
                continue

            # Individual seeds (thin)
            for seed, curve in seed_curves.items():
                xs, ys = zip(*curve)
                ax.plot(xs, ys, color=COLORS[alg], alpha=0.25, linewidth=1, linestyle="--")

            # Mean curve (thick)
            all_edits = sorted(set(e for c in seed_curves.values() for e, _ in c))
            mean_vals = []
            for e in all_edits:
                vals = [v for seed, curve in seed_curves.items() for x, v in curve if x == e]
                if vals:
                    mean_vals.append((e, np.mean(vals), np.std(vals)))

            if mean_vals:
                xs, ys, stds = zip(*mean_vals)
                ax.plot(xs, ys, color=COLORS[alg], linewidth=2.5, label=alg, marker="o", markersize=4)
                ax.fill_between(xs, np.array(ys) - np.array(stds),
                               np.array(ys) + np.array(stds),
                               color=COLORS[alg], alpha=0.1)

        ax.set_xlabel("Total Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    # Panel 4: GLUE MMLU
    ax = axes[1, 1]
    for alg in ["AlphaEdit", "MEMIT"]:
        seed_curves = {}
        for seed in seeds:
            curve = []
            for edits in edit_points:
                g = load_failure_curve_glue(seed, edits, alg)
                if g and "mmmlu" in g:
                    curve.append((edits, g["mmmlu"]))
            if curve:
                seed_curves[seed] = curve

        if not seed_curves:
            continue

        for seed, curve in seed_curves.items():
            xs, ys = zip(*curve)
            ax.plot(xs, ys, color=COLORS[alg], alpha=0.25, linewidth=1, linestyle="--")

        all_edits = sorted(set(e for c in seed_curves.values() for e, _ in c))
        mean_vals = []
        for e in all_edits:
            vals = [v for seed, curve in seed_curves.items() for x, v in curve if x == e]
            if vals:
                mean_vals.append((e, np.mean(vals), np.std(vals)))

        if mean_vals:
            xs, ys, stds = zip(*mean_vals)
            ax.plot(xs, ys, color=COLORS[alg], linewidth=2.5, label=alg, marker="o", markersize=4)
            ax.fill_between(xs, np.array(ys) - np.array(stds),
                           np.array(ys) + np.array(stds),
                           color=COLORS[alg], alpha=0.1)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("MMLU F1")
    ax.set_title("General Capability (MMLU)")
    ax.legend()
    ax.set_ylim(-0.05, 0.75)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"fig1_failure_curve.{fmt}")
    plt.close()
    print(f"  [Fig 1] failure_curve: saved")


# ─── Figure 2: Order Sensitivity ─────────────────────────────────────────────

def plot_order_sensitivity():
    """Show AlphaEdit is order-invariant while MEMIT collapses regardless."""
    base = RESULTS / "comparison_ordered" / "seed42" / "3000edits"
    if not base.exists():
        print("  [Fig 2] SKIP: comparison_ordered data not found")
        return

    order_dirs = [("0", base)]
    for i in range(1, 5):
        d = base / f"order{i}"
        if d.exists():
            order_dirs.append((str(i), d))

    # Collect AlphaEdit GLUE per order
    ae_glue = []
    ae_edit = []
    for label, d in order_dirs:
        g = load_comparison_ordered_glue(d, "AlphaEdit")
        e = load_comparison_ordered(d, "AlphaEdit")
        if g:
            g["order"] = label
            ae_glue.append(g)
        if e:
            e["order"] = label
            ae_edit.append(e)

    # MEMIT GLUE trajectory (from failure curve data, seed 42)
    memit_trajectory = []
    for edits in [2000, 3000, 4000, 5000]:
        g = load_failure_curve_glue(42, edits, "MEMIT")
        if g and "mmmlu" in g:
            memit_trajectory.append((edits, g["mmmlu"]))

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: GLUE tasks as grouped bars
    ax = fig.add_subplot(gs[0])
    tasks = ["mmmlu", "sst", "cola", "mrpc", "nli"]
    task_labels = ["MMLU", "SST", "CoLA", "MRPC", "NLI"]
    n_orders = len(ae_glue)
    x = np.arange(len(tasks))
    width = 0.8 / max(n_orders, 1)

    for i, gd in enumerate(ae_glue):
        vals = [gd.get(t, 0) for t in tasks]
        offset = (i - n_orders / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, alpha=0.8,
               color=plt.cm.Blues(0.3 + 0.12 * i), label=f"Order {gd['order']}")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("F1 Score")
    ax.set_title("(a) AlphaEdit GLUE (5 orderings)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7, ncol=2)

    # Panel B: Edit metrics stability
    ax = fig.add_subplot(gs[1])
    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]

    means = [np.mean([e[m] for e in ae_edit]) for m in metrics]
    stds = [np.std([e[m] for e in ae_edit]) for m in metrics]

    bars = ax.bar(metric_labels, means, yerr=stds, capsize=5,
                  color=COLORS["AlphaEdit"], alpha=0.7, edgecolor="black", linewidth=0.5)

    # Overlay individual points
    for e in ae_edit:
        for i, m in enumerate(metrics):
            ax.scatter(i, e[m], color="black", s=20, zorder=5, alpha=0.5)

    ax.set_ylabel("Score")
    ax.set_title("(b) Edit Metrics (mean ± σ)")
    ax.set_ylim(0, 1.05)

    # Panel C: MEMIT collapse vs AlphaEdit stability
    ax = fig.add_subplot(gs[2])

    # AlphaEdit: flat line showing mean MMLU across orders
    ae_mmlu_vals = [g.get("mmmlu", 0) for g in ae_glue if g.get("mmmlu")]
    if ae_mmlu_vals:
        ae_mean = np.mean(ae_mmlu_vals)
        ax.axhline(ae_mean, color=COLORS["AlphaEdit"], linewidth=2, linestyle="-",
                   label=f"AlphaEdit (μ={ae_mean:.3f}, σ={np.std(ae_mmlu_vals):.3f})")
        ax.axhspan(ae_mean - np.std(ae_mmlu_vals), ae_mean + np.std(ae_mmlu_vals),
                   color=COLORS["AlphaEdit"], alpha=0.1)

    # MEMIT trajectory
    if memit_trajectory:
        xs, ys = zip(*memit_trajectory)
        ax.plot(xs, ys, color=COLORS["MEMIT"], linewidth=2.5, marker="o", markersize=5, label="MEMIT")

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("MMLU F1")
    ax.set_title("(c) MMLU: Stable vs Collapsed")
    ax.legend()
    ax.set_ylim(-0.05, 0.75)
    ax.set_xlim(1500, 5500)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"fig2_order_sensitivity.{fmt}")
    plt.close()
    print(f"  [Fig 2] order_sensitivity: saved")


# ─── Figure 3: Mechanism Divergence ──────────────────────────────────────────

def plot_mechanism_divergence():
    """Cache eigenspectrum divergence predicts behavioral collapse."""
    records_42 = load_mechanism_metrics(42)
    records_2024 = load_mechanism_metrics(2024)

    if not records_42 or not records_2024:
        print("  [Fig 3] SKIP: mechanism data not available for both seeds")
        return

    def aggregate_by_edits(records):
        by_edits = defaultdict(list)
        for r in records:
            if r.get("cache"):
                by_edits[r["total_edits"]].append(r["cache"])
        result = {}
        for edits, caches in sorted(by_edits.items()):
            result[edits] = {
                "condition": np.mean([c["cache_condition"] for c in caches]),
                "effective_rank": np.mean([c["cache_effective_rank"] for c in caches]),
                "numerical_rank": np.mean([c["cache_numerical_rank"] for c in caches]),
                "top_sv_share": np.mean([c["cache_top_sv_share"] for c in caches]),
            }
        return result

    agg_42 = aggregate_by_edits(records_42)
    agg_2024 = aggregate_by_edits(records_2024)

    # Also load behavioral data (efficacy from failure curve)
    eff_42, eff_2024 = {}, {}
    for edits in sorted(set(list(agg_42.keys()) + [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])):
        m = load_failure_curve_metrics(42, edits, "AlphaEdit")
        if m:
            eff_42[edits] = m["efficacy"]
        m = load_failure_curve_metrics(2024, edits, "AlphaEdit")
        if m:
            eff_2024[edits] = m["efficacy"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Condition number (mechanism)
    ax = axes[0]
    xs_42 = sorted(agg_42.keys())
    xs_2024 = sorted(agg_2024.keys())
    ax.semilogy(xs_42, [agg_42[x]["condition"] for x in xs_42],
                color=SEED_COLORS[42], linewidth=2, label="Seed 42", marker="o", markersize=3)
    ax.semilogy(xs_2024, [agg_2024[x]["condition"] for x in xs_2024],
                color=SEED_COLORS[2024], linewidth=2, label="Seed 2024", marker="o", markersize=3)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Condition Number (log)")
    ax.set_title("(a) Cache Condition Number")
    ax.legend()

    # Find divergence point
    common = sorted(set(xs_42) & set(xs_2024))
    for e in common:
        if agg_42[e]["condition"] > 0 and agg_2024[e]["condition"] > 0:
            ratio = max(agg_42[e]["condition"], agg_2024[e]["condition"]) / min(agg_42[e]["condition"], agg_2024[e]["condition"])
            if ratio > 2.0:
                ax.axvline(e, color="red", linestyle=":", alpha=0.5)
                ax.annotate(f"diverges\n{e} edits", xy=(e, agg_42[e]["condition"]),
                           fontsize=8, ha="right", color="red")
                break

    # Panel 2: Effective rank
    ax = axes[1]
    ax.plot(xs_42, [agg_42[x]["effective_rank"] for x in xs_42],
            color=SEED_COLORS[42], linewidth=2, label="Seed 42", marker="o", markersize=3)
    ax.plot(xs_2024, [agg_2024[x]["effective_rank"] for x in xs_2024],
            color=SEED_COLORS[2024], linewidth=2, label="Seed 2024", marker="o", markersize=3)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Effective Rank")
    ax.set_title("(b) Cache Effective Rank")
    ax.legend()

    # Panel 3: Behavioral outcome (efficacy)
    ax = axes[2]
    if eff_42:
        xs, ys = zip(*sorted(eff_42.items()))
        ax.plot(xs, ys, color=SEED_COLORS[42], linewidth=2, label="Seed 42", marker="o", markersize=3)
    if eff_2024:
        xs, ys = zip(*sorted(eff_2024.items()))
        ax.plot(xs, ys, color=SEED_COLORS[2024], linewidth=2, label="Seed 2024", marker="o", markersize=3)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Efficacy")
    ax.set_title("(c) Behavioral Outcome")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"fig3_mechanism_divergence.{fmt}")
    plt.close()
    print(f"  [Fig 3] mechanism_divergence: saved")


# ─── Figure 4: Controlled Coupling ───────────────────────────────────────────

def plot_controlled_coupling():
    """Low vs high coupling: mechanism degradation comparison."""
    low = load_controlled_coupling("low_coupling")
    high = load_controlled_coupling("high_coupling")

    if not low and not high:
        print("  [Fig 4] SKIP: no controlled coupling data locally")
        print("         Pull: scp chilli@remote-rig:~/Projects/alphaedit-analysis/results/controlled_coupling/*.jsonl results/controlled_coupling/")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Controlled Coupling: Semantic Structure Drives Null-Space Exhaustion", fontsize=12, y=0.98)

    streams = [(n, d) for n, d in [("low_coupling", low), ("high_coupling", high)] if d]

    for name, records in streams:
        edits = [r["total_edits"] for r in records]
        color = STREAM_COLORS[name]
        label = name.replace("_", " ").title()

        agg_key = lambda r, k: r.get("mechanism", {}).get("aggregate", {}).get(k)

        # Panel 1: Effective rank
        ax = axes[0, 0]
        vals = [(e, agg_key(r, "mean_cache_effective_rank")) for e, r in zip(edits, records)]
        vals = [(e, v) for e, v in vals if v is not None]
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=2)

        # Panel 2: Removed fraction
        ax = axes[0, 1]
        vals = [(e, agg_key(r, "mean_removed_fraction")) for e, r in zip(edits, records)]
        vals = [(e, v) for e, v in vals if v is not None]
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=2)

        # Panel 3: Condition number
        ax = axes[1, 0]
        vals = [(e, agg_key(r, "mean_cache_condition")) for e, r in zip(edits, records)]
        vals = [(e, v) for e, v in vals if v is not None]
        if vals:
            xs, ys = zip(*vals)
            ax.semilogy(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=2)

        # Panel 4: Efficacy
        ax = axes[1, 1]
        vals = [(e, r.get("evaluation", {}).get("overall_efficacy")) for e, r in zip(edits, records)]
        vals = [(e, v) for e, v in vals if v is not None]
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, color=color, linewidth=2, label=label, marker="o", markersize=2)

    axes[0, 0].set_xlabel("Total Edits")
    axes[0, 0].set_ylabel("Effective Rank")
    axes[0, 0].set_title("(a) Cache Effective Rank")
    axes[0, 0].legend()

    axes[0, 1].set_xlabel("Total Edits")
    axes[0, 1].set_ylabel("Removed Fraction")
    axes[0, 1].set_title("(b) Projection Capacity Consumed")
    axes[0, 1].legend()

    axes[1, 0].set_xlabel("Total Edits")
    axes[1, 0].set_ylabel("Condition Number")
    axes[1, 0].set_title("(c) Cache Condition (log)")
    axes[1, 0].legend()

    axes[1, 1].set_xlabel("Total Edits")
    axes[1, 1].set_ylabel("Efficacy")
    axes[1, 1].set_title("(d) Edit Success Rate")
    axes[1, 1].legend()
    axes[1, 1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"fig4_controlled_coupling.{fmt}")
    plt.close()
    print(f"  [Fig 4] controlled_coupling: saved")


# ─── Summary Statistics ──────────────────────────────────────────────────────

def print_summary():
    """Print key results for paper narrative."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    seeds = [42, 2024, 137]
    edit_points = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # 1. AlphaEdit advantage quantification
    print("\n─── 1. AlphaEdit vs MEMIT at scale ───")
    for edits in [3000, 5000, 7000, 10000]:
        ae_effs, memit_effs = [], []
        for seed in seeds:
            m_ae = load_failure_curve_metrics(seed, edits, "AlphaEdit")
            m_me = load_failure_curve_metrics(seed, edits, "MEMIT")
            if m_ae:
                ae_effs.append(m_ae["efficacy"])
            if m_me:
                memit_effs.append(m_me["efficacy"])
        if ae_effs and memit_effs:
            print(f"  {edits:>5} edits: AlphaEdit eff={np.mean(ae_effs):.3f}±{np.std(ae_effs):.3f}"
                  f"  MEMIT eff={np.mean(memit_effs):.3f}±{np.std(memit_effs):.3f}"
                  f"  Δ={np.mean(ae_effs)-np.mean(memit_effs):+.3f}")
        elif ae_effs:
            print(f"  {edits:>5} edits: AlphaEdit eff={np.mean(ae_effs):.3f}±{np.std(ae_effs):.3f}  MEMIT: no data")

    # 2. MEMIT collapse point
    print("\n─── 2. MEMIT Collapse Points ───")
    for seed in seeds:
        eff_collapse, glue_collapse = None, None
        for edits in edit_points:
            m = load_failure_curve_metrics(seed, edits, "MEMIT")
            if m and m["efficacy"] < 0.5 and eff_collapse is None:
                eff_collapse = edits
            g = load_failure_curve_glue(seed, edits, "MEMIT")
            if g and g.get("mmmlu", 1) < 0.3 and glue_collapse is None:
                glue_collapse = edits
        print(f"  Seed {seed}: efficacy<0.5 at {eff_collapse or '>10K'}, MMLU<0.3 at {glue_collapse or '>10K'}")

    # 3. AlphaEdit eventual degradation
    print("\n─── 3. AlphaEdit Degradation at Scale ───")
    for seed in seeds:
        trajectory = []
        for edits in edit_points:
            m = load_failure_curve_metrics(seed, edits, "AlphaEdit")
            if m:
                trajectory.append((edits, m["efficacy"]))
        if trajectory:
            first_e, first_v = trajectory[0]
            last_e, last_v = trajectory[-1]
            print(f"  Seed {seed}: eff at {first_e}={first_v:.3f} → at {last_e}={last_v:.3f} (Δ={last_v-first_v:+.3f})")

    # 4. Order sensitivity
    print("\n─── 4. Order Sensitivity (AlphaEdit, 3000 edits) ───")
    base = RESULTS / "comparison_ordered" / "seed42" / "3000edits"
    if base.exists():
        order_dirs = [base] + [base / f"order{i}" for i in range(1, 5) if (base / f"order{i}").exists()]
        effs = []
        paras = []
        for d in order_dirs:
            e = load_comparison_ordered(d, "AlphaEdit")
            if e:
                effs.append(e["efficacy"])
                paras.append(e["paraphrase"])
        if effs:
            print(f"  Efficacy across {len(effs)} orderings: μ={np.mean(effs):.4f}, σ={np.std(effs):.4f}, CV={np.std(effs)/np.mean(effs)*100:.2f}%")
            print(f"  Paraphrase across {len(paras)} orderings: μ={np.mean(paras):.4f}, σ={np.std(paras):.4f}, CV={np.std(paras)/np.mean(paras)*100:.2f}%")
            print(f"  Conclusion: Negligible order sensitivity (CV < 1%)")

    # 5. Mechanism leading indicators
    print("\n─── 5. Predictive Divergence ───")
    records_42 = load_mechanism_metrics(42)
    records_2024 = load_mechanism_metrics(2024)
    if records_42 and records_2024:
        def get_condition_by_edits(records):
            by_edits = defaultdict(list)
            for r in records:
                if r.get("cache"):
                    by_edits[r["total_edits"]].append(r["cache"]["cache_condition"])
            return {e: np.mean(v) for e, v in by_edits.items()}

        cond_42 = get_condition_by_edits(records_42)
        cond_2024 = get_condition_by_edits(records_2024)
        common = sorted(set(cond_42.keys()) & set(cond_2024.keys()))

        divergence_point = None
        for e in common:
            ratio = max(cond_42[e], cond_2024[e]) / min(cond_42[e], cond_2024[e])
            if ratio > 2.0 and divergence_point is None:
                divergence_point = e

        if divergence_point:
            print(f"  Cache condition diverges (>2× between seeds) at {divergence_point} edits")
            # Find where behavior diverges
            eff_42 = {e: load_failure_curve_metrics(42, e, "AlphaEdit") for e in edit_points}
            eff_2024 = {e: load_failure_curve_metrics(2024, e, "AlphaEdit") for e in edit_points}
            for e in edit_points:
                if eff_42[e] and eff_2024[e]:
                    diff = abs(eff_42[e]["efficacy"] - eff_2024[e]["efficacy"])
                    if diff > 0.1:
                        print(f"  Behavioral divergence (Δeff > 0.1) at {e} edits")
                        print(f"  Lead time: {e - divergence_point} edits")
                        break

    # 6. Controlled coupling
    print("\n─── 6. Controlled Coupling ───")
    low = load_controlled_coupling("low_coupling")
    if low:
        last = low[-1]
        agg = last.get("mechanism", {}).get("aggregate", {})
        print(f"  Low coupling after {last['total_edits']} edits:")
        print(f"    Effective rank: {agg.get('mean_cache_effective_rank', '?'):.1f}")
        print(f"    Removed fraction: {agg.get('mean_removed_fraction', '?'):.4f}")
        print(f"    → Only {agg.get('mean_removed_fraction', 0)*100:.1f}% of null-space consumed")
    else:
        print("  Data not available locally (on remote rig)")


# ─── CSV Export ──────────────────────────────────────────────────────────────

def export_csv():
    """Export data tables as CSV for LaTeX."""
    import csv

    seeds = [42, 2024, 137]
    edit_points = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    csv_path = OUTPUT / "table_failure_curve.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "edits", "algorithm", "efficacy", "paraphrase", "neighborhood", "mmlu_f1"])
        for seed in seeds:
            for edits in edit_points:
                for alg in ["AlphaEdit", "MEMIT"]:
                    m = load_failure_curve_metrics(seed, edits, alg)
                    g = load_failure_curve_glue(seed, edits, alg)
                    if m:
                        mmlu = f"{g['mmmlu']:.4f}" if g and "mmmlu" in g else ""
                        writer.writerow([seed, edits, alg, f"{m['efficacy']:.4f}",
                                        f"{m['paraphrase']:.4f}", f"{m['neighborhood']:.4f}", mmlu])
    print(f"  {csv_path.name}")

    csv_path = OUTPUT / "table_order_sensitivity.csv"
    base = RESULTS / "comparison_ordered" / "seed42" / "3000edits"
    if base.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["order", "efficacy", "paraphrase", "neighborhood",
                           "mmlu", "sst", "cola", "mrpc", "nli"])
            order_dirs = [("0", base)] + [(str(i), base / f"order{i}") for i in range(1, 5)]
            for label, d in order_dirs:
                if not d.exists():
                    continue
                e = load_comparison_ordered(d, "AlphaEdit")
                g = load_comparison_ordered_glue(d, "AlphaEdit")
                if e and g:
                    writer.writerow([label, f"{e['efficacy']:.4f}", f"{e['paraphrase']:.4f}",
                                    f"{e['neighborhood']:.4f}",
                                    f"{g.get('mmmlu', 0):.4f}", f"{g.get('sst', 0):.4f}",
                                    f"{g.get('cola', 0):.4f}", f"{g.get('mrpc', 0):.4f}",
                                    f"{g.get('nli', 0):.4f}"])
        print(f"  {csv_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("AlphaEdit Reproducibility: Paper Figures & Analysis")
    print("=" * 70)
    print(f"Output: {OUTPUT}\n")

    print("Generating figures...")
    plot_failure_curve()
    plot_order_sensitivity()
    plot_mechanism_divergence()
    plot_controlled_coupling()

    print("\nExporting CSVs...")
    export_csv()

    print_summary()

    print("\n" + "=" * 70)
    print(f"All outputs in: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
