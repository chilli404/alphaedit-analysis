"""Shared data loading for all paper figures.

All loaders extract BOTH binary (_correct) and probability (_probs) metrics
from the local results/ directory.

Expected local directory layout:
─────────────────────────────────
results/
├── failure_curve_checkpointed/
│   └── seed{N}/
│       └── {E}edits/              # E = 2000, 3000, ..., 10000
│           └── {Alg}/             # AlphaEdit or MEMIT
│               └── run_000/
│                   ├── {E}_edits-case_0.json
│                   └── ...
├── comparison_ordered/
│   └── seed{N}/
│       └── {E}edits/
│           ├── order{0-9}/{Alg}/run_000/*_edits-case_*.json
│           └── (legacy) {Alg}/run_000/*_edits-case_*.json
├── polykernel_editor/
│   └── seed{N}/
│       └── {E}edits/
│           ├── {Alg}-{kernel}/run_000/*_edits-case_*.json
│           ├── eval_{E}/                         # flat eval dir (10k)
│           ├── log_{Alg}_seed{N}_{kernel}_*.jsonl
│           └── metadata_{Alg}_seed{N}_{kernel}.json
├── memit_seqreg/
│   ├── full_eval_seed{N}_lp{X}_ld{Y}.json
│   ├── log_seed{N}_lp{X}_ld{Y}_*.jsonl
│   └── behavioral_run_*/
│       └── *_edits-case_*.json
├── mechanism_analysis/
│   └── seed{N}/
│       └── mechanism_seed{N}_*.jsonl
├── matched_ordering/
│   ├── orderings/                       # stream definitions (input datasets)
│   │   ├── clustered_seed{N}.json
│   │   ├── dispersed_seed{N}.json
│   │   ├── key_clustered_seed{N}.json
│   │   └── key_dispersed_seed{N}.json
│   ├── key_geometry/                    # precomputed key vectors
│   │   └── keys_seed{N}_layer{L}.npz
│   ├── diagnostics/                     # validation & diagnostics
│   │   ├── cohort_balance_seed{N}.json
│   │   ├── k_sweep_seed{N}.json
│   │   ├── key_stream_properties_seed{N}.json
│   │   └── validation_report_seed{N}.json
│   └── {ALG}/{ORDERING}/seed{SEED}/     # runtime results & evals
│       └── *.jsonl / full_eval_seed{N}.json
├── mve1_alphaedit_mcf/
│   └── seed{N}/alphaedit_results/AlphaEdit/run_000/*_edits-case_*.json
├── mve2_memit_mcf/
│   └── seed{N}/alphaedit_results/MEMIT/run_000/*_edits-case_*.json
├── mve3_alphaedit_zsre/
│   └── seed{N}/alphaedit_results/AlphaEdit/run_000/*_edits-case_*.json
└── figures/paper/
    └── stream_matching_audit_seed{N}.json
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from analysis.style import PROJECT, RESULTS

# ─── Core Metric Extraction ──────────────────────────────────────────────────


def extract_case_metrics(case_json: dict) -> dict:
    """Extract all metrics from a single per-case JSON file.

    Returns dict with:
      - efficacy, paraphrase, neighborhood (binary, 0-1)
      - efficacy_prob, paraphrase_prob, neighborhood_prob (continuous)
      - case_id, num_edits
    """
    post = case_json.get("post", {})
    row = {
        "case_id": case_json.get("case_id"),
        "num_edits": case_json.get("num_edits"),
    }

    # Binary metrics (mean of boolean list)
    for json_key, metric_name in [
        ("rewrite_prompts_correct", "efficacy"),
        ("paraphrase_prompts_correct", "paraphrase"),
        ("neighborhood_prompts_correct", "neighborhood"),
    ]:
        vals = post.get(json_key)
        if isinstance(vals, list) and vals:
            row[metric_name] = sum(vals) / len(vals)
        else:
            row[metric_name] = None

    # Probability metrics (mean of target_new probabilities)
    for json_key, metric_name in [
        ("rewrite_prompts_probs", "efficacy_prob"),
        ("paraphrase_prompts_probs", "paraphrase_prob"),
        ("neighborhood_prompts_probs", "neighborhood_prob"),
    ]:
        vals = post.get(json_key)
        if isinstance(vals, list) and vals:
            if isinstance(vals[0], dict):
                row[metric_name] = np.mean([d["target_new"] for d in vals])
            else:
                row[metric_name] = np.mean(vals)
        else:
            row[metric_name] = None

    return row


def _aggregate_case_files(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate metrics from a directory of case JSON files.

    Returns dict with: efficacy, paraphrase, neighborhood,
    efficacy_prob, paraphrase_prob, neighborhood_prob, n_facts.
    """
    case_files = list(run_dir.glob("*_edits-case_*.json"))
    if not case_files:
        return None

    metrics = defaultdict(list)
    for f_path in case_files:
        with open(f_path) as f:
            data = json.load(f)
        row = extract_case_metrics(data)
        for k in ("efficacy", "paraphrase", "neighborhood",
                  "efficacy_prob", "paraphrase_prob", "neighborhood_prob"):
            if row.get(k) is not None:
                metrics[k].append(row[k])

    if not metrics.get("efficacy"):
        return None

    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    result["n_facts"] = len(metrics["efficacy"])
    return result


# ─── Failure Curve Loaders ────────────────────────────────────────────────────


def _find_run_dir(seed: int, edits: int, alg: str) -> Optional[Path]:
    """Locate the run directory for a failure curve checkpoint."""
    local = RESULTS / "failure_curve_checkpointed" / f"seed{seed}" / f"{edits}edits" / alg / "run_000"
    if local.exists() and any(local.glob("*_edits-case_*.json")):
        return local
    return None


def load_checkpoint_metrics(seed: int, edits: int, alg: str) -> Optional[Dict[str, Any]]:
    """Load aggregate metrics for a failure curve checkpoint.

    Returns dict with: efficacy, paraphrase, neighborhood,
    efficacy_prob, paraphrase_prob, neighborhood_prob, n_facts.
    """
    run_dir = _find_run_dir(seed, edits, alg)
    if run_dir is None:
        return None
    return _aggregate_case_files(run_dir)


def load_checkpoint_cohorts(
    seed: int,
    edits: int,
    alg: str,
    batch_size: int = 100,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load per-cohort metrics for a failure curve checkpoint.

    Groups facts by their insertion batch (case_id // batch_size).
    Returns dict mapping cohort_index → {efficacy, paraphrase, neighborhood, n_facts}.
    """
    run_dir = _find_run_dir(seed, edits, alg)
    if run_dir is None:
        return None

    cohorts = defaultdict(lambda: defaultdict(list))
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)
        row = extract_case_metrics(data)
        if row["case_id"] is None:
            continue
        cohort_idx = row["case_id"] // batch_size
        for k in ("efficacy", "paraphrase", "neighborhood"):
            if row.get(k) is not None:
                cohorts[cohort_idx][k].append(row[k])

    if not cohorts:
        return None

    result = {}
    for idx, vals in sorted(cohorts.items()):
        result[idx] = {
            k: float(np.mean(v)) for k, v in vals.items()
        }
        result[idx]["n_facts"] = len(vals.get("efficacy", []))
    return result


def load_checkpoint_glue(seed: int, edits: int, alg: str) -> Optional[Dict[str, float]]:
    """Load GLUE/MMLU scores at a failure curve checkpoint.

    Looks for post-edit GLUE first, falls back to base GLUE.
    """
    run_dir = _find_run_dir(seed, edits, alg)
    if run_dir is None:
        return None

    glue_dir = run_dir / "glue_eval"
    if not glue_dir.exists():
        return None

    # Prefer post-edit GLUE (case_*_glue.json)
    post_glue = sorted(glue_dir.glob("case_*_glue.json"))
    if post_glue:
        with open(post_glue[-1]) as f:
            data = json.load(f)
    else:
        # Fall back to base GLUE
        base_glue = glue_dir / "base_glue.json"
        if not base_glue.exists():
            return None
        with open(base_glue) as f:
            data = json.load(f)

    scores = {}
    for task in ("mmmlu", "sst", "cola", "mrpc", "nli", "rte"):
        if task in data:
            scores[task] = data[task].get("f1_new", data[task].get("f1", 0))
    scores["is_post_edit"] = bool(post_glue)
    scores["edit_num"] = data.get("edit_num", edits)
    return scores


# ─── Order Sensitivity Loaders ───────────────────────────────────────────────


def load_comparison_ordered(
    seed: int,
    edits: int,
) -> List[Dict[str, Any]]:
    """Load all orderings for a given seed and edit count (comparison_ordered experiment).

    Returns list of dicts, one per ordering × algorithm, with:
    order_id, algorithm, efficacy, paraphrase, neighborhood, glue (if available).
    """
    base = RESULTS / "comparison_ordered" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return []

    results = []

    # All orderings live in order0/, order1/, ... subdirectories.
    # Legacy fallback: if order0/ doesn't exist but AlphaEdit/ does at base level,
    # treat base as order0.
    order_dirs = []
    for i in range(10):
        d = base / f"order{i}"
        if d.exists():
            order_dirs.append((str(i), d))
    # Fallback: base directory has AlphaEdit/MEMIT directly (legacy layout)
    if not order_dirs and (base / "AlphaEdit").exists():
        order_dirs.append(("0", base))
        for i in range(1, 10):
            d = base / f"order{i}"
            if d.exists():
                order_dirs.append((str(i), d))

    for order_id, d in order_dirs:
        for alg in ("AlphaEdit", "MEMIT"):
            run_dir = d / alg / "run_000"
            if not run_dir.exists():
                continue

            case_files = list(run_dir.glob("*_edits-case_*.json"))
            if not case_files:
                continue

            metrics = defaultdict(list)
            for f_path in case_files:
                with open(f_path) as f:
                    data = json.load(f)
                row = extract_case_metrics(data)
                for k in ("efficacy", "paraphrase", "neighborhood",
                          "neighborhood_prob"):
                    if row.get(k) is not None:
                        metrics[k].append(row[k])

            if not metrics.get("efficacy"):
                continue

            entry = {
                "order_id": order_id,
                "algorithm": alg,
                **{k: float(np.mean(v)) for k, v in metrics.items()},
                "n_facts": len(metrics["efficacy"]),
            }

            # Load GLUE if available
            glue_dir = run_dir / "glue_eval"
            glue_files = sorted(glue_dir.glob("case_*_glue.json")) if glue_dir.exists() else []
            if glue_files:
                with open(glue_files[-1]) as f:
                    gdata = json.load(f)
                for task in ("mmmlu", "sst", "cola", "mrpc", "nli"):
                    if task in gdata:
                        entry[task] = gdata[task].get("f1_new", 0)

            results.append(entry)

    return results


# ─── Polykernel Editor Loaders ──────────────────────────────────────────────


def load_polykernel_metrics(
    seed: int,
    edits: int,
    kernel: str = "poly2",
    alg: str = "AlphaEdit",
) -> Optional[Dict[str, Any]]:
    """Load aggregate metrics for a polykernel editor run.

    Layout: polykernel_editor/seed{N}/{E}edits/{Alg}-{kernel}/run_000/

    Also checks eval_{E}/ flat directory (used for 10k evals).

    Args:
        seed: Random seed.
        edits: Edit count (2000, 10000).
        kernel: Kernel type ("poly2", "rbf").
        alg: Algorithm name ("AlphaEdit").
    """
    base = RESULTS / "polykernel_editor" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return None

    # Primary: {Alg}-{kernel}/run_000/
    run_dir = base / f"{alg}-{kernel}" / "run_000"
    if run_dir.exists():
        result = _aggregate_case_files(run_dir)
        if result is not None:
            return result

    # Fallback: eval_{edits}/ flat dir (10k milestone eval)
    eval_dir = base / f"eval_{edits // 1000}k"
    if eval_dir.exists():
        return _aggregate_case_files(eval_dir)

    return None


def load_polykernel_cohorts(
    seed: int,
    edits: int,
    kernel: str = "poly2",
    alg: str = "AlphaEdit",
    batch_size: int = 100,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load per-cohort metrics for polykernel editor."""
    base = RESULTS / "polykernel_editor" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return None

    # Find case files
    run_dir = base / f"{alg}-{kernel}" / "run_000"
    if not run_dir.exists():
        run_dir = base / f"eval_{edits // 1000}k"
    if not run_dir.exists():
        return None

    cohorts = defaultdict(lambda: defaultdict(list))
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)
        row = extract_case_metrics(data)
        if row["case_id"] is None:
            continue
        cohort_idx = row["case_id"] // batch_size
        for k in ("efficacy", "paraphrase", "neighborhood"):
            if row.get(k) is not None:
                cohorts[cohort_idx][k].append(row[k])

    if not cohorts:
        return None

    result = {}
    for idx, vals in sorted(cohorts.items()):
        result[idx] = {k: float(np.mean(v)) for k, v in vals.items()}
        result[idx]["n_facts"] = len(vals.get("efficacy", []))
    return result


def load_polykernel_logs(
    seed: int,
    edits: int,
    kernel: str = "poly2",
    alg: str = "AlphaEdit",
) -> List[Dict]:
    """Load per-batch JSONL mechanism logs for polykernel editor.

    Returns list of records with: batch, layer, trace_ratio, G_lin_rank,
    kernel_type, phase.
    """
    base = RESULTS / "polykernel_editor" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return []

    # Logs live inside {Alg}-{kernel}/ subdirectory
    subdir = base / f"{alg}-{kernel}"
    pattern = f"log_{alg}_seed{seed}_{kernel}_*.jsonl"

    records = []
    search_dirs = [subdir, base] if subdir.exists() else [base]
    for search_dir in search_dirs:
        for jsonl in sorted(search_dir.glob(pattern)):
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if records:
            break
    return records


def load_polykernel_metadata(
    seed: int,
    edits: int,
    kernel: str = "poly2",
    alg: str = "AlphaEdit",
) -> Optional[Dict]:
    """Load experiment metadata for a polykernel run."""
    base = RESULTS / "polykernel_editor" / f"seed{seed}" / f"{edits}edits"
    # Try kernel-specific naming (deg2, rbf_median)
    kernel_short = kernel.replace("poly", "deg") if kernel.startswith("poly") else kernel
    path = base / f"metadata_{alg}_seed{seed}_{kernel_short}.json"
    if not path.exists():
        path = base / f"metadata_{alg}_seed{seed}_{kernel}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Polykernel Diagnostic Loaders ───────────────────────────────────────────


def load_polykernel_diagnostic(
    seed: int,
    alg: str = "AlphaEdit",
) -> Optional[Dict]:
    """Load Gram matrix diagnostic analysis for polykernel experiment.

    Layout: polykernel_diagnostic/analysis_{alg}_seed{seed}.json

    Returns dict with: metadata, analysis_params, per_batch, cumulative,
    sliding, by_coupling_type, cross_group_separation, summary.
    """
    path = RESULTS / "polykernel_diagnostic" / f"analysis_{alg}_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Mechanism Analysis Loaders ──────────────────────────────────────────────


def load_mechanism_metrics(seed: int) -> List[Dict]:
    """Load mechanism analysis JSONL records."""
    mech_dir = RESULTS / "mechanism_analysis" / f"seed{seed}"
    if not mech_dir.exists():
        return []

    records = []
    for jsonl in sorted(mech_dir.glob("mechanism_*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─── MEMIT+SeqReg Loaders ────────────────────────────────────────────────────


def load_seqreg_eval(
    seed: int,
    lambda_prev: float = 1.0,
    lambda_delta: float = 1.0,
) -> Optional[Dict]:
    """Load full evaluation JSON for MEMIT+SeqReg.

    Returns dict keyed by "{N}_edits" with metrics at each checkpoint.
    """
    path = RESULTS / "memit_seqreg" / f"full_eval_seed{seed}_lp{lambda_prev}_ld{lambda_delta}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_seqreg_logs(
    seed: int,
    lambda_prev: float = 1.0,
    lambda_delta: float = 1.0,
) -> List[Dict]:
    """Load per-batch JSONL logs for MEMIT+SeqReg.

    Returns list of records with: batch, layer, upd_norm, dw_kprev_norm,
    cache_batches, cache_keys, base_lhs_norm, kpkp_norm.
    """
    pattern = f"log_seed{seed}_lp{lambda_prev}_ld{lambda_delta}_*.jsonl"
    logs = sorted(RESULTS.glob(f"memit_seqreg/{pattern}"))
    if not logs:
        return []

    records = []
    for logfile in logs:
        with open(logfile) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_seqreg_behavioral(
    seed: int,
    edits: int,
) -> Optional[Dict[str, Any]]:
    """Load SeqReg per-case results from behavioral eval directories.

    Returns aggregated metrics dict.
    """
    # Check behavioral_run_{batch} directories
    for subdir in RESULTS.glob("memit_seqreg/behavioral_run_*"):
        if not subdir.is_dir():
            continue
        case_files = list(subdir.glob("*_edits-case_*.json"))
        if not case_files:
            continue

        # Check if this matches the requested edit count
        with open(case_files[0]) as f:
            sample = json.load(f)
        if sample.get("num_edits") != edits:
            continue

        return _aggregate_case_files(subdir)

    return None


# ─── Weight Drift Loaders ────────────────────────────────────────────────────


def load_weight_drift(seed: int) -> Optional[Dict]:
    """Load weight drift analysis."""
    path = RESULTS / "figures" / "paper" / f"weight_drift_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── MVE (Reproduction) Loaders ──────────────────────────────────────────────


def load_mve_metrics(experiment: str, seed: int, alg: str) -> Optional[Dict[str, Any]]:
    """Load aggregate metrics for an MVE experiment.

    Args:
        experiment: e.g. "mve1_alphaedit_mcf", "mve2_memit_mcf",
                   "mve3_alphaedit_zsre"
        seed: random seed
        alg: "AlphaEdit" or "MEMIT"
    """
    seed_dir = RESULTS / experiment / f"seed{seed}"
    if not seed_dir.exists():
        return None

    # Try multiple layout conventions
    candidates = [
        seed_dir / "alphaedit_results" / alg / "run_000",  # mve1-3 standard
        seed_dir / "results" / alg / "run_000",            # legacy
    ]

    for run_dir in candidates:
        if not run_dir.exists():
            continue
        result = _aggregate_case_files(run_dir)
        if result is not None:
            return result

    return None


# ─── Matched Ordering Loaders ─────────────────────────────────────────────────


def load_matched_ordering_validation(seed: int) -> Optional[Dict]:
    """Load the pre-experiment validation report for key-clustered orderings.

    Returns the full validation report with cosine stats, prefix geometry,
    future-key exposure, and cohort balance.
    """
    path = RESULTS / "matched_ordering" / "diagnostics" / f"validation_report_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_matched_ordering_properties(seed: int) -> Optional[Dict]:
    """Load stream properties (cosine ratio, cluster stats, etc.)."""
    path = RESULTS / "matched_ordering" / "diagnostics" / f"key_stream_properties_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_matched_ordering_ksweep(seed: int) -> Optional[List[Dict]]:
    """Load k-means cluster count sweep results."""
    path = RESULTS / "matched_ordering" / "diagnostics" / f"k_sweep_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_matched_ordering_keys(seed: int, layer: int = 6) -> Optional[Dict]:
    """Load precomputed MEMIT keys and case_ids.

    Returns dict with 'keys' (ndarray N×D), 'case_ids' (list of int).
    """
    path = RESULTS / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer{layer}.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    return {
        "keys": npz["keys"],
        "case_ids": npz["case_ids"].tolist(),
        "layer": int(npz["layer"]),
    }


def load_matched_ordering_stream(seed: int, ordering: str) -> Optional[List[Dict]]:
    """Load a matched ordering stream (key_clustered or key_dispersed).

    Args:
        seed: Random seed.
        ordering: One of 'key_clustered', 'key_dispersed', 'clustered', 'dispersed'.

    Returns list of MCF records in the stream's order.
    """
    path = RESULTS / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_matched_ordering_results(
    seed: int,
    ordering: str,
    alg: str = "AlphaEdit",
) -> List[Dict]:
    """Load runtime JSONL results from matched ordering experiment.

    Layout: matched_ordering/{ALG}/{ORDERING}/seed{SEED}/*.jsonl

    Args:
        seed: Random seed.
        ordering: Stream ordering (e.g. "clustered", "dispersed",
                  "key_clustered", "key_dispersed").
        alg: Algorithm name ("AlphaEdit" or "MEMIT-Seq-1-0").

    Returns list of per-batch records with mechanism and evaluation data.
    """
    result_dir = RESULTS / "matched_ordering" / alg / ordering / f"seed{seed}"
    if not result_dir.exists():
        return []

    records = []
    for jsonl in sorted(result_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─── Discovery ───────────────────────────────────────────────────────────────


def discover_available_data() -> Dict[str, Any]:
    """Report what data is available locally.

    Returns a structured dict describing available experiments, seeds,
    and checkpoints.
    """
    summary = {}

    # Failure curve
    fc_dir = RESULTS / "failure_curve_checkpointed"
    if fc_dir.exists():
        fc = {}
        for seed_dir in sorted(fc_dir.glob("seed*")):
            seed = seed_dir.name
            edits_available = {}
            for edit_dir in sorted(seed_dir.glob("*edits")):
                algos = [d.name for d in edit_dir.iterdir() if d.is_dir()]
                has_data = {}
                for alg in algos:
                    run = edit_dir / alg / "run_000"
                    n = len(list(run.glob("*_edits-case_*.json"))) if run.exists() else 0
                    if n > 0:
                        has_data[alg] = n
                if has_data:
                    edits_available[edit_dir.name] = has_data
            if edits_available:
                fc[seed] = edits_available
        summary["failure_curve"] = fc

    # Order sensitivity (comparison_ordered)
    co_dir = RESULTS / "comparison_ordered"
    if co_dir.exists():
        co = {}
        for seed_dir in sorted(co_dir.glob("seed*")):
            co[seed_dir.name] = [d.name for d in sorted(seed_dir.iterdir()) if d.is_dir()]
        summary["comparison_ordered"] = co

    # Polykernel editor
    pk_dir = RESULTS / "polykernel_editor"
    if pk_dir.exists():
        pk = {}
        for seed_dir in sorted(pk_dir.glob("seed*")):
            edits_info = {}
            for edit_dir in sorted(seed_dir.glob("*edits")):
                alg_dirs = [d.name for d in edit_dir.iterdir() if d.is_dir()]
                logs = [f.name for f in edit_dir.glob("log_*.jsonl")]
                edits_info[edit_dir.name] = {
                    "algorithms": alg_dirs,
                    "logs": logs,
                }
            if edits_info:
                pk[seed_dir.name] = edits_info
        summary["polykernel_editor"] = pk

    # SeqReg
    sr_dir = RESULTS / "memit_seqreg"
    if sr_dir.exists():
        summary["memit_seqreg"] = {
            "logs": [f.name for f in sr_dir.glob("*.jsonl")],
            "evals": [f.name for f in sr_dir.glob("full_eval_*.json")],
            "behavioral_dirs": [d.name for d in sr_dir.iterdir()
                                if d.is_dir() and d.name.startswith("behavioral")],
        }

    # MVE experiments (reproduction at standard scale)
    for mve_name in ("mve1_alphaedit_mcf", "mve2_memit_mcf",
                     "mve3_alphaedit_zsre"):
        mve_dir = RESULTS / mve_name
        if mve_dir.exists():
            mve = {}
            for seed_dir in sorted(mve_dir.glob("seed*")):
                n_cases = 0
                # Standard layout: alphaedit_results/{Alg}/run_000/
                for run_dir in seed_dir.glob("alphaedit_results/*/run_000"):
                    n_cases += len(list(run_dir.glob("*_edits-case_*.json")))
                if n_cases > 0:
                    mve[seed_dir.name] = n_cases
            if mve:
                summary[mve_name] = mve

    # Matched ordering
    mo_dir = RESULTS / "matched_ordering"
    if mo_dir.exists():
        mo = {}
        ord_dir = mo_dir / "orderings"
        if ord_dir.exists():
            for stream_file in sorted(ord_dir.glob("*_seed*.json")):
                mo.setdefault("streams", []).append(stream_file.name)
        diag_dir = mo_dir / "diagnostics"
        if diag_dir.exists():
            for val_file in sorted(diag_dir.glob("*.json")):
                mo.setdefault("diagnostics", []).append(val_file.name)
        kg_dir = mo_dir / "key_geometry"
        if kg_dir.exists():
            mo["keys"] = [f.name for f in sorted(kg_dir.glob("keys_*.npz"))]
        # Runtime results: {ALG}/{ORDERING}/seed{N}/
        result_dirs = []
        for alg_dir in sorted(mo_dir.iterdir()):
            if not alg_dir.is_dir():
                continue
            if alg_dir.name in ("key_geometry", "diagnostics", "orderings"):
                continue
            for ordering_dir in sorted(alg_dir.iterdir()):
                if not ordering_dir.is_dir():
                    continue
                for seed_dir in sorted(ordering_dir.glob("seed*")):
                    n_jsonl = len(list(seed_dir.glob("*.jsonl")))
                    if n_jsonl > 0:
                        result_dirs.append(
                            f"{alg_dir.name}/{ordering_dir.name}/{seed_dir.name}: {n_jsonl} files"
                        )
        if result_dirs:
            mo["results"] = result_dirs
        if mo:
            summary["matched_ordering"] = mo

    # Mechanism analysis
    mech_dir = RESULTS / "mechanism_analysis"
    if mech_dir.exists():
        mech = {}
        for seed_dir in sorted(mech_dir.glob("seed*")):
            jsonl_files = list(seed_dir.glob("mechanism_*.jsonl"))
            if jsonl_files:
                mech[seed_dir.name] = [f.name for f in sorted(jsonl_files)]
        if mech:
            summary["mechanism_analysis"] = mech

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    """Print data availability report."""
    import pprint
    data = discover_available_data()
    print("=" * 60)
    print("DATA AVAILABILITY REPORT")
    print("=" * 60)
    pprint.pprint(data, width=100)
