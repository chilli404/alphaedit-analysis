"""Method registry: maps short method names to runner + default parameters.

Every editing method in the project is registered here with its runner,
algorithm name, and default hyperparameters. The CLI uses this to dispatch.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MethodDef:
    """Definition of a single editing method."""
    name: str
    runner: str          # relative path under src/ or scripts/
    shell: bool = False  # True for baseline shell scripts
    # Default args passed to the runner
    defaults: dict = field(default_factory=dict)
    description: str = ""


def _src(rel: str) -> str:
    return rel


METHODS: dict[str, MethodDef] = {
    # --- checkpoint_runner methods ---
    "alphaedit": MethodDef(
        name="alphaedit",
        runner=_src("runners/checkpoint_runner.py"),
        defaults={"alg_name": "AlphaEdit", "ds_name": "mcf"},
        description="AlphaEdit (null-space projection)",
    ),
    "memit": MethodDef(
        name="memit",
        runner=_src("runners/checkpoint_runner.py"),
        defaults={"alg_name": "MEMIT", "ds_name": "mcf"},
        description="MEMIT (unconstrained mass editing)",
    ),

    # --- polykernel_seqreg_runner methods ---
    "memit-seq": MethodDef(
        name="memit-seq",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "MEMIT", "lambda_prev": 1.0, "lambda_delta": 0.0,
            "kernel_degree": 1, "cache_strategy": "all", "cache_max": "none",
        },
        description="MEMIT-Seq (history-aware regularization)",
    ),
    "revive+memit": MethodDef(
        name="revive+memit",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "MEMIT", "lambda_prev": 0.0, "lambda_delta": 0.0,
            "kernel_degree": 1, "cache_strategy": "all", "cache_max": "none",
            "revive": True, "revive_tau": 0.1,
        },
        description="REVIVE + MEMIT (spectral filter on plain MEMIT)",
    ),
    "revive+alphaedit": MethodDef(
        name="revive+alphaedit",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "AlphaEdit", "lambda_prev": 0.0, "lambda_delta": 0.0,
            "kernel_degree": 1, "cache_strategy": "all", "cache_max": "none",
            "revive": True, "revive_tau": 0.1,
        },
        description="REVIVE + AlphaEdit (spectral filter on AlphaEdit)",
    ),
    "revive+nse": MethodDef(
        name="revive+nse",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "NSE", "lambda_prev": 0.0, "lambda_delta": 0.0,
            "kernel_degree": 1, "cache_strategy": "all", "cache_max": "none",
            "revive": True, "revive_tau": 0.1,
        },
        description="REVIVE + NSE (spectral filter on neuron-level editing)",
    ),
    "revive+rect": MethodDef(
        name="revive+rect",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "MEMIT_rect", "lambda_prev": 0.0, "lambda_delta": 0.0,
            "kernel_degree": 1, "cache_strategy": "all", "cache_max": "none",
            "revive": True, "revive_tau": 0.1,
        },
        description="REVIVE + RECT-Aligned (spectral filter on RECT)",
    ),
    "poly2-hybrid": MethodDef(
        name="poly2-hybrid",
        runner=_src("polykernel/polykernel_seqreg_runner.py"),
        defaults={
            "base_alg": "MEMIT", "lambda_prev": 1.0, "lambda_delta": 0.0,
            "kernel_degree": 2, "no_kernel_prev": True,
            "cache_strategy": "all", "cache_max": "none",
        },
        description="MEMIT-Seq + poly2-hybrid kernel",
    ),

    # --- pathguard_runner ---
    "pathguard": MethodDef(
        name="pathguard",
        runner=_src("runners/pathguard_runner.py"),
        defaults={
            "pathguard": True, "pathguard_M": 200, "pathguard_adaptive": True,
            "pathguard_kernel_degree": 2,
            "lambda_prev": 1.0, "lambda_delta": 0.0,
            "cache_strategy": "all", "cache_max": "none",
        },
        description="PathGuard (hazard-gated displacement-constrained editing)",
    ),

    # --- baseline shell scripts ---
    "evoedit": MethodDef(
        name="evoedit",
        runner="scripts/run_evoedit_baseline.sh",
        shell=True,
        description="EvoEdit (evolving sequential null-space alignment)",
    ),
    "nse": MethodDef(
        name="nse",
        runner="scripts/run_nse_baseline.sh",
        shell=True,
        description="NSE (neuron-level selection editing)",
    ),
    "rect": MethodDef(
        name="rect",
        runner="scripts/run_rect_aligned_paper_replication.sh",
        shell=True,
        description="RECT-Aligned (rectified alignment + error cache)",
    ),

    # --- seeded_runner (MVE reproductions) ---
    "mve-alphaedit": MethodDef(
        name="mve-alphaedit",
        runner=_src("runners/seeded_runner.py"),
        defaults={"alg_name": "AlphaEdit", "ds_name": "mcf", "dataset_size_limit": 2000},
        description="AlphaEdit 2K reproduction (5-seed MVE)",
    ),
    "mve-memit": MethodDef(
        name="mve-memit",
        runner=_src("runners/seeded_runner.py"),
        defaults={"alg_name": "MEMIT", "ds_name": "mcf", "dataset_size_limit": 2000},
        description="MEMIT 2K reproduction (5-seed MVE)",
    ),
}


def get_method(name: str) -> MethodDef:
    if name not in METHODS:
        raise ValueError(
            f"Unknown method: {name}. "
            f"Available: {', '.join(sorted(METHODS.keys()))}"
        )
    return METHODS[name]


def list_methods() -> list[MethodDef]:
    return sorted(METHODS.values(), key=lambda m: m.name)
