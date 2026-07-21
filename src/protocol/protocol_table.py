"""Protocol table formatter — CSV and LaTeX output.

Produces:
  - table5_protocol.csv
  - table5_protocol.tex
  - protocol_numbers.json (all protocol values for paper prose)

Usage:
    uv run python -m src.protocol.protocol_table
    uv run python -m src.protocol.protocol_table --output-dir results/figures/paper
    uv run python -m src.protocol.protocol_table --edits 5000 --seeds 42 137 2024
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.protocol.sequential_memory_eval import (
    MethodReport,
    evaluate_method,
    evaluate_method_multiseed,
)

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT / "results" / "figures" / "paper"

METRIC_LABELS = {
    "current_batch_efficacy": "Current-batch eff.",
    "latest_1k_efficacy": "Latest-1K eff.",
    "first_1k_retention": "First-1K retention",
    "retention_auc": "Retention AUC",
    "order_variance": "Order CV (%)",
    "concentration_sensitivity": "Coupling gap",
    "cost_seconds_per_batch": "Runtime (s/batch)",
    "cost_memory_mb": "Memory (MB)",
}

METRIC_KEYS = list(METRIC_LABELS.keys())

METHOD_DISPLAY = {
    "AlphaEdit": "AlphaEdit",
    "MEMIT": "MEMIT (per-batch)",
    "MEMIT+SeqReg": "MEMIT+SeqReg (capped)",
}


# ─── Table Generation ────────────────────────────────────────────────────────


def format_table(
    reports: List[MethodReport],
    output_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Format protocol reports into table rows and write CSV + LaTeX.

    Args:
        reports: List of MethodReport objects (one per method).
        output_dir: Where to write output files.

    Returns:
        List of row dicts.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for report in reports:
        row = {"method": METHOD_DISPLAY.get(report.method, report.method)}
        for key in METRIC_KEYS:
            val = getattr(report, key)
            row[key] = val
        rows.append(row)

    # Write CSV
    _write_csv(rows, output_dir / "table5_protocol.csv")

    # Write LaTeX
    _write_latex(rows, output_dir / "table5_protocol.tex")

    # Write protocol numbers JSON
    _write_numbers(reports, output_dir / "protocol_numbers.json")

    return rows


def format_table_multiseed(
    methods: List[str],
    seeds: List[int],
    edits: int,
    batch_size: int = 100,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Generate protocol table with multi-seed mean +/- std.

    Args:
        methods: List of algorithm names.
        seeds: List of seeds.
        edits: Edit count to evaluate at.
        batch_size: Edits per batch.
        output_dir: Where to write output files.

    Returns:
        List of row dicts with mean ± std formatting.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_results = []

    for alg in methods:
        result = evaluate_method_multiseed(alg, seeds, edits, batch_size)
        raw_results.append(result)

        row = {"method": METHOD_DISPLAY.get(alg, alg)}
        for key in METRIC_KEYS:
            val = result.get(key)
            if val and isinstance(val, dict):
                if val["std"] > 0:
                    row[key] = f"{val['mean']:.3f} ± {val['std']:.3f}"
                else:
                    row[key] = f"{val['mean']:.3f}"
            else:
                row[key] = "—"
        rows.append(row)

    _write_csv(rows, output_dir / "table5_protocol_multiseed.csv")
    _write_latex(rows, output_dir / "table5_protocol_multiseed.tex")

    # Write raw numbers
    numbers_path = output_dir / "protocol_numbers.json"
    numbers = {}
    for result in raw_results:
        alg = result["method"]
        for key in METRIC_KEYS:
            val = result.get(key)
            if val and isinstance(val, dict):
                numbers[f"{alg}_{edits}_{key}_mean"] = val["mean"]
                numbers[f"{alg}_{edits}_{key}_std"] = val["std"]
    with open(numbers_path, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"  protocol_numbers.json: {len(numbers)} values")

    return rows


# ─── Output Writers ──────────────────────────────────────────────────────────


def _write_csv(rows: List[Dict], path: Path):
    """Write rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_fmt(v) for k, v in row.items()})
    print(f"  {path.name}: {len(rows)} rows")


def _write_latex(rows: List[Dict], path: Path):
    """Write rows as LaTeX tabular."""
    if not rows:
        return

    n_cols = len(METRIC_KEYS) + 1
    col_spec = "l" + "r" * len(METRIC_KEYS)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sequential-Memory Evaluation Protocol}")
    lines.append(r"\label{tab:protocol}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = "Method"
    for key in METRIC_KEYS:
        header += f" & {METRIC_LABELS[key]}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for row in rows:
        line = row["method"]
        for key in METRIC_KEYS:
            val = row.get(key)
            line += f" & {_latex_fmt(val)}"
        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {path.name}: written")


def _write_numbers(reports: List[MethodReport], path: Path):
    """Write all protocol numbers as JSON for paper prose."""
    numbers = {}
    for report in reports:
        prefix = f"{report.method}_seed{report.seed}_{report.edits}"
        for key in METRIC_KEYS:
            val = getattr(report, key)
            if val is not None:
                numbers[f"{prefix}_{key}"] = val
    with open(path, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"  protocol_numbers.json: {len(numbers)} values")


def _csv_fmt(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _latex_fmt(val) -> str:
    if val is None:
        return "---"
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate Sequential-Memory Protocol Table"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--edits", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["AlphaEdit", "MEMIT", "MEMIT+SeqReg"])
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    print(f"Generating protocol table at {args.edits} edits...")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print()

    if len(args.seeds) == 1:
        # Single-seed mode
        reports = []
        for alg in args.methods:
            report = evaluate_method(alg, args.seeds[0], args.edits, args.batch_size)
            reports.append(report)
        format_table(reports, args.output_dir)
    else:
        # Multi-seed mode
        format_table_multiseed(
            args.methods, args.seeds, args.edits, args.batch_size, args.output_dir
        )


if __name__ == "__main__":
    main()
