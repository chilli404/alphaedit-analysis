"""Shared visual configuration for all paper figures."""

from pathlib import Path

import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
PAPER_OUTPUT = RESULTS / "figures" / "paper"
APPENDIX_OUTPUT = RESULTS / "figures" / "appendix"

# ─── Colors ───────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "AlphaEdit": "#2196F3",
    "MEMIT": "#FF9800",
    "MEMIT+SeqReg": "#4CAF50",
}

SEED_COLORS = {
    42: "#2196F3",
    2024: "#E91E63",
    137: "#4CAF50",
    7: "#FF9800",
    99: "#9C27B0",
}

STREAM_COLORS = {
    "low_coupling": "#2196F3",
    "high_coupling": "#E91E63",
}

KERNEL_COLORS = {
    "AlphaEdit": "#2196F3",   # Standard (same as ALGO_COLORS)
    "poly2": "#9C27B0",       # Purple
    "rbf": "#795548",         # Brown
}

# ─── Matplotlib Style ─────────────────────────────────────────────────────────

RCPARAMS = {
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
}


def setup_style():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update(RCPARAMS)


def save_figure(fig, name, output_dir=None):
    """Save figure as both PNG and PDF.

    Args:
        fig: matplotlib Figure.
        name: filename stem (without extension).
        output_dir: output directory (defaults to PAPER_OUTPUT).
    """
    out = Path(output_dir) if output_dir else PAPER_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(out / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  [{name}] saved to {out}")
