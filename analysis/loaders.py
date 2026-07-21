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
│                   ├── {E}_edits-case_1.json
│                   └── ...
├── controlled_coupling/
│   ├── {stream}_seed{N}*.jsonl    # stream = low_coupling or high_coupling
│   ├── behavioral_eval_seed{N}.json
│   └── stream_properties_seed{N}.json
├── comparison_ordered/
│   └── seed{N}/
│       └── {E}edits/
│           ├── {Alg}/run_000/*_edits-case_*.json   # order 0 (base)
│           └── order{1-9}/{Alg}/run_000/...        # additional orderings
├── memit_seqreg/
│   ├── full_eval_seed{N}_lp{X}_ld{Y}.json
│   ├── log_seed{N}_lp{X}_ld{Y}_*.jsonl
│   └── behavioral_run_*/
│       └── *_edits-case_*.json
├── mechanism_analysis/
│   └── seed{N}/
│       └── mechanism_seed{N}_*.jsonl      # per-layer cache spectrum data
├── mve1_alphaedit_mcf/
│   └── seed{N}/alphaedit_results/AlphaEdit/run_000/*_edits-case_*.json
├── mve2_memit_mcf/
│   └── seed{N}/alphaedit_results/MEMIT/run_000/*_edits-case_*.json
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

    files = list(run_dir.glob("*_edits-case_*.json"))
    if not files:
        return None

    metrics = defaultdict(list)
    for f_path in files:
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


# ─── Controlled Coupling Loaders ─────────────────────────────────────────────


def load_controlled_coupling_jsonl(
    stream: str,
    seed: int,
) -> List[Dict]:
    """Load per-batch mechanism records from controlled coupling JSONL.

    Args:
        stream: "low_coupling" or "high_coupling"
        seed: random seed (42, 137)

    Returns list of records with keys: stream, batch_idx, total_edits,
    mechanism.{layers, aggregate, projection_layers}, evaluation.
    """
    cc_dir = RESULTS / "controlled_coupling"
    if not cc_dir.exists():
        return []

    records = []
    for jsonl in sorted(cc_dir.glob(f"{stream}_seed{seed}*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_controlled_coupling_behavioral(seed: int) -> Optional[Dict]:
    """Load behavioral evaluation results for controlled coupling.

    Returns dict with keys "low_coupling" and "high_coupling", each containing:
    overall_efficacy, first_1k_mean_efficacy, last_1k_mean_efficacy,
    retention_auc, cohort_retention, per_fact_results.
    """
    path = RESULTS / "controlled_coupling" / f"behavioral_eval_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_stream_audit(seed: int) -> Optional[Dict]:
    """Load stream-matching audit results."""
    path = RESULTS / "figures" / "paper" / f"stream_matching_audit_seed{seed}.json"
    if not path.exists():
        # Try controlled_coupling dir
        path = RESULTS / "controlled_coupling" / f"stream_properties_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── Order Sensitivity Loaders ───────────────────────────────────────────────


def load_comparison_ordered(
    seed: int,
    edits: int,
) -> List[Dict[str, Any]]:
    """Load all orderings for a given seed and edit count.

    Returns list of dicts, one per ordering, with:
    order_id, efficacy, paraphrase, neighborhood, glue (if available).
    """
    base = RESULTS / "comparison_ordered" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return []

    results = []

    # Two conventions for order 0:
    #   - 3K style: AlphaEdit/MEMIT directly under base (base IS order 0)
    #   - 7K style: explicit order0/ subdirectory
    order_dirs = []
    if (base / "order0").exists():
        # Explicit order directories (order0, order1, ...)
        for i in range(10):
            d = base / f"order{i}"
            if d.exists():
                order_dirs.append((str(i), d))
    else:
        # Base directory is order 0, then order1, order2, ...
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

        metrics = defaultdict(list)
        for f_path in case_files:
            with open(f_path) as f:
                data = json.load(f)
            row = extract_case_metrics(data)
            for k in ("efficacy", "paraphrase", "neighborhood",
                      "efficacy_prob", "paraphrase_prob", "neighborhood_prob"):
                if row.get(k) is not None:
                    metrics[k].append(row[k])

        if metrics:
            result = {k: float(np.mean(v)) for k, v in metrics.items()}
            result["n_facts"] = len(metrics["efficacy"])
            return result

    return None


# ─── Weight Drift Loaders ────────────────────────────────────────────────────


def load_weight_drift(seed: int) -> Optional[Dict]:
    """Load weight drift analysis for controlled coupling."""
    path = RESULTS / "figures" / "paper" / f"weight_drift_controlled_coupling_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─── MVE (Reproduction) Loaders ──────────────────────────────────────────────


def load_mve_metrics(experiment: str, seed: int, alg: str) -> Optional[Dict[str, Any]]:
    """Load aggregate metrics for an MVE experiment.

    Args:
        experiment: e.g. "mve1_alphaedit_mcf", "mve2_memit_mcf"
        seed: random seed
        alg: "AlphaEdit" or "MEMIT"
    """
    # Try the results subdirectory
    run_dir = RESULTS / experiment / f"seed{seed}" / "alphaedit_results" / alg / "run_000"
    if not run_dir.exists():
        run_dir = RESULTS / experiment / f"seed{seed}" / "results" / alg / "run_000"
    if not run_dir.exists():
        return None

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

    # Controlled coupling
    cc_dir = RESULTS / "controlled_coupling"
    if cc_dir.exists():
        cc = {
            "jsonl_files": [f.name for f in cc_dir.glob("*.jsonl")],
            "behavioral_evals": [f.name for f in cc_dir.glob("behavioral_eval_*.json")],
        }
        summary["controlled_coupling"] = cc

    # Order sensitivity
    co_dir = RESULTS / "comparison_ordered"
    if co_dir.exists():
        co = {}
        for seed_dir in sorted(co_dir.glob("seed*")):
            co[seed_dir.name] = [d.name for d in sorted(seed_dir.iterdir()) if d.is_dir()]
        summary["comparison_ordered"] = co

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
    for mve_name in ("mve1_alphaedit_mcf", "mve2_memit_mcf", "mve3_alphaedit_zsre"):
        mve_dir = RESULTS / mve_name
        if mve_dir.exists():
            mve = {}
            for seed_dir in sorted(mve_dir.glob("seed*")):
                # Count case files under alphaedit_results/{Alg}/run_000/
                n_cases = 0
                for run_dir in seed_dir.glob("alphaedit_results/*/run_000"):
                    n_cases += len(list(run_dir.glob("*_edits-case_*.json")))
                if n_cases > 0:
                    mve[seed_dir.name] = n_cases
            if mve:
                summary[mve_name] = mve

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
