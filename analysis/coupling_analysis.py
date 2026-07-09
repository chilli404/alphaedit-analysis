#!/usr/bin/env python3
"""
Analysis and visualization for the semantic coupling stress test.

Reads JSONL trace files from results/coupling_stress/ and produces:
1. Box/violin plot: projection_loss by coupling type
2. Scatter: projection_loss vs upd_matrix_norm
3. Heatmap: mean projection_loss by (coupling_type x layer)
4. Temporal: projection_loss over edit index, colored by type

Statistical tests:
- Kruskal-Wallis: does projection_loss differ across types?
- Pairwise Mann-Whitney U with Holm-Bonferroni correction
- Spearman correlation: projection_loss <-> upd_matrix_norm
- Effect size: Cliff's delta for Type 3 vs Type 0
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


COUPLING_TYPE_NAMES = {
    -1: "warmup",
    0: "unrelated",
    1: "relation_match",
    2: "subject_match",
    3: "full_conflict",
}

PROBE_TYPES = [0, 1, 2, 3]  # Only analyze probe edits, not warmup


def load_trace(trace_path: Path) -> List[dict]:
    """Load a JSONL trace file."""
    records = []
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_traces(results_dir: Path) -> List[dict]:
    """Load all trace JSONL files from a directory."""
    all_records = []
    for path in sorted(results_dir.glob("coupling_trace_*.jsonl")):
        all_records.extend(load_trace(path))
    return all_records


def filter_probes(records: List[dict]) -> List[dict]:
    """Keep only probe edits (not warmup or anchors)."""
    return [r for r in records if r.get("role") == "probe"]


def extract_projection_losses(records: List[dict]) -> Dict[int, List[float]]:
    """Group mean projection_loss by coupling type."""
    by_type = {t: [] for t in PROBE_TYPES}
    for record in records:
        ct = record.get("coupling_type", -1)
        if ct in by_type:
            loss = record.get("aggregate", {}).get("mean_projection_loss", None)
            if loss is not None:
                by_type[ct].append(loss)
    return by_type


def extract_per_layer(records: List[dict]) -> Dict[int, Dict[str, List[float]]]:
    """Group projection_loss by (coupling_type, layer)."""
    result = {t: {} for t in PROBE_TYPES}
    for record in records:
        ct = record.get("coupling_type", -1)
        if ct not in PROBE_TYPES:
            continue
        layers = record.get("layers", {})
        for layer_name, layer_data in layers.items():
            if layer_name not in result[ct]:
                result[ct][layer_name] = []
            loss = layer_data.get("projection_loss", None)
            if loss is not None:
                result[ct][layer_name].append(loss)
    return result


# --- Statistical Tests ---


def kruskal_wallis_test(by_type: Dict[int, List[float]]) -> Optional[dict]:
    """Kruskal-Wallis H-test: do projection losses differ across coupling types?"""
    if not HAS_SCIPY:
        return None
    groups = [by_type[t] for t in PROBE_TYPES if len(by_type[t]) > 0]
    if len(groups) < 2:
        return None
    stat, p = scipy_stats.kruskal(*groups)
    return {"H_statistic": stat, "p_value": p, "n_groups": len(groups)}


def pairwise_mannwhitney(by_type: Dict[int, List[float]]) -> List[dict]:
    """Pairwise Mann-Whitney U tests with Holm-Bonferroni correction."""
    if not HAS_SCIPY:
        return []

    comparisons = []
    for i in range(len(PROBE_TYPES)):
        for j in range(i + 1, len(PROBE_TYPES)):
            t_a, t_b = PROBE_TYPES[i], PROBE_TYPES[j]
            a, b = by_type[t_a], by_type[t_b]
            if len(a) < 2 or len(b) < 2:
                continue
            stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            comparisons.append({
                "type_a": t_a,
                "type_b": t_b,
                "name_a": COUPLING_TYPE_NAMES[t_a],
                "name_b": COUPLING_TYPE_NAMES[t_b],
                "U_statistic": stat,
                "p_value": p,
                "n_a": len(a),
                "n_b": len(b),
            })

    # Holm-Bonferroni correction
    if comparisons:
        comparisons.sort(key=lambda x: x["p_value"])
        m = len(comparisons)
        for rank, comp in enumerate(comparisons):
            adjusted_alpha = 0.05 / (m - rank)
            comp["holm_significant"] = comp["p_value"] < adjusted_alpha
            comp["holm_rank"] = rank + 1

    return comparisons


def cliffs_delta(a: List[float], b: List[float]) -> float:
    """Cliff's delta effect size (non-parametric)."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    count = 0
    for x in a:
        for y in b:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    return count / (n_a * n_b)


def spearman_correlation(
    records: List[dict], x_key: str = "mean_projection_loss", y_key: str = "total_upd_norm"
) -> Optional[dict]:
    """Spearman rank correlation between two aggregate metrics."""
    if not HAS_SCIPY:
        return None
    xs, ys = [], []
    for r in records:
        agg = r.get("aggregate", {})
        x_val = agg.get(x_key)
        y_val = agg.get(y_key)
        if x_val is not None and y_val is not None:
            xs.append(x_val)
            ys.append(y_val)
    if len(xs) < 5:
        return None
    rho, p = scipy_stats.spearmanr(xs, ys)
    return {"rho": rho, "p_value": p, "n": len(xs)}


# --- Plotting ---


def plot_projection_loss_by_type(
    by_type: Dict[int, List[float]], output_path: Path
) -> None:
    """Box/violin plot of projection_loss by coupling type."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    data = [by_type[t] for t in PROBE_TYPES]
    labels = [COUPLING_TYPE_NAMES[t] for t in PROBE_TYPES]

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Projection Loss")
    ax.set_title("Projection Loss by Semantic Coupling Type")
    ax.set_ylim(0, 1)

    # Add sample sizes
    for i, t in enumerate(PROBE_TYPES):
        ax.text(i + 1, -0.05, f"n={len(by_type[t])}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_projection_loss_heatmap(
    per_layer: Dict[int, Dict[str, List[float]]], output_path: Path
) -> None:
    """Heatmap: mean projection_loss by (coupling_type x layer)."""
    if not HAS_MPL:
        return

    # Collect all layer names
    all_layers = set()
    for ct_data in per_layer.values():
        all_layers.update(ct_data.keys())
    layers_sorted = sorted(all_layers, key=lambda x: int(x))

    # Build matrix
    matrix = np.zeros((len(PROBE_TYPES), len(layers_sorted)))
    for i, ct in enumerate(PROBE_TYPES):
        for j, layer in enumerate(layers_sorted):
            values = per_layer[ct].get(layer, [])
            matrix[i, j] = np.mean(values) if values else 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers_sorted)))
    ax.set_xticklabels([f"Layer {l}" for l in layers_sorted])
    ax.set_yticks(range(len(PROBE_TYPES)))
    ax.set_yticklabels([COUPLING_TYPE_NAMES[t] for t in PROBE_TYPES])
    ax.set_title("Mean Projection Loss by Coupling Type and Layer")
    plt.colorbar(im, ax=ax, label="Projection Loss")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal(records: List[dict], output_path: Path) -> None:
    """projection_loss over edit index, colored by coupling type."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = {0: "gray", 1: "blue", 2: "orange", 3: "red", -1: "lightgray"}

    for record in records:
        ct = record.get("coupling_type", -1)
        idx = record.get("edit_idx", 0)
        loss = record.get("aggregate", {}).get("mean_projection_loss", None)
        if loss is not None:
            ax.scatter(idx, loss, c=colors.get(ct, "black"), s=10, alpha=0.6)

    # Legend
    for ct in PROBE_TYPES:
        ax.scatter([], [], c=colors[ct], s=30, label=COUPLING_TYPE_NAMES[ct])
    ax.legend(loc="upper left")

    ax.set_xlabel("Edit Index")
    ax.set_ylabel("Mean Projection Loss")
    ax.set_title("Projection Loss Over Editing Sequence")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# --- Main ---


def run_analysis(results_dir: Path, output_dir: Optional[Path] = None) -> dict:
    """Run full analysis on coupling stress results."""
    if output_dir is None:
        output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_records = load_all_traces(results_dir)
    probes = filter_probes(all_records)
    print(f"Loaded {len(all_records)} total records, {len(probes)} probes")

    if not probes:
        print("No probe records found. Exiting.")
        return {}

    # Extract grouped data
    by_type = extract_projection_losses(probes)
    per_layer = extract_per_layer(probes)

    # Print summary
    print("\n--- Projection Loss Summary ---")
    for ct in PROBE_TYPES:
        values = by_type[ct]
        if values:
            print(f"  {COUPLING_TYPE_NAMES[ct]:16s}: "
                  f"mean={np.mean(values):.4f}, "
                  f"median={np.median(values):.4f}, "
                  f"n={len(values)}")

    # Statistical tests
    results = {"summary": {}}
    for ct in PROBE_TYPES:
        values = by_type[ct]
        if values:
            results["summary"][COUPLING_TYPE_NAMES[ct]] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }

    print("\n--- Statistical Tests ---")

    kw = kruskal_wallis_test(by_type)
    if kw:
        print(f"  Kruskal-Wallis: H={kw['H_statistic']:.2f}, p={kw['p_value']:.4e}")
        results["kruskal_wallis"] = kw

    pairwise = pairwise_mannwhitney(by_type)
    if pairwise:
        print("  Pairwise Mann-Whitney U (Holm-Bonferroni):")
        for comp in pairwise:
            sig = "*" if comp.get("holm_significant") else ""
            print(f"    {comp['name_a']} vs {comp['name_b']}: "
                  f"p={comp['p_value']:.4e} {sig}")
        results["pairwise_mannwhitney"] = pairwise

    # Cliff's delta: Type 3 vs Type 0
    if by_type[3] and by_type[0]:
        delta = cliffs_delta(by_type[3], by_type[0])
        print(f"  Cliff's delta (full_conflict vs unrelated): {delta:.3f}")
        results["cliffs_delta_3v0"] = delta

    # Spearman correlation
    spearman = spearman_correlation(probes)
    if spearman:
        print(f"  Spearman (projection_loss ~ upd_norm): "
              f"rho={spearman['rho']:.3f}, p={spearman['p_value']:.4e}")
        results["spearman_loss_vs_norm"] = spearman

    # Plots
    if HAS_MPL:
        print("\n--- Generating Figures ---")
        plot_projection_loss_by_type(by_type, output_dir / "projection_loss_by_type.png")
        plot_projection_loss_heatmap(per_layer, output_dir / "projection_loss_heatmap.png")
        plot_temporal(probes, output_dir / "projection_loss_temporal.png")
        print(f"  Figures saved to {output_dir}")

    # Save results JSON
    results_path = output_dir / "coupling_stats.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Stats saved to {results_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze coupling stress test results")
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing coupling trace JSONL files"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures and stats"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else project_root / "results" / "coupling_stress"
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_analysis(results_dir, output_dir)


if __name__ == "__main__":
    main()
