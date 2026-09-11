#!/usr/bin/env python3
"""
Probability-preference evaluation matching the official AlphaEdit metrics.

Official AlphaEdit/EvoEdit/REVIVE papers use PAIRWISE probability comparisons:
  - Efficacy:     P(target_new) > P(target_true) on rewrite prompts
  - Paraphrase:   P(target_new) > P(target_true) on paraphrase prompts
  - Neighborhood:  P(target_true) > P(target_new) on neighborhood prompts

Since NLL is stored (lower NLL = higher probability), the comparisons become:
  - Efficacy:     NLL(target_true) > NLL(target_new)
  - Paraphrase:   NLL(target_true) > NLL(target_new)
  - Neighborhood: NLL(target_true) < NLL(target_new)

This matches vendor/AlphaEdit/experiments/summarize.py lines 62-99 exactly.

Our previous evaluation used argmax metrics (is target the top-1 prediction?), which
is much stricter and likely explains 10-15% neighborhood scores vs published 50-66%.

Two modes:
  1. RESCORE: Re-score existing per-case result JSON files (no GPU needed)
     python scripts/eval_prob_preference.py rescore --result_dir results/matched_ordering/AlphaEdit/key_clustered/seed42/AlphaEdit/run_000

  2. EVALUATE: Load model from checkpoint, run inference, compute both metric sets
     python scripts/eval_prob_preference.py evaluate --checkpoint_dir ~/.cache/alphaedit_checkpoints/matched_ordering/AlphaEdit/key_clustered/seed42 --checkpoints 9 19 29 49

Usage examples:
    # Re-score all experiments for one seed
    python scripts/eval_prob_preference.py rescore \\
        --result_dir results/matched_ordering/AlphaEdit/key_clustered/seed42/AlphaEdit/run_000

    # Re-score failure curve
    python scripts/eval_prob_preference.py rescore \\
        --result_dir results/failure_curve_checkpointed/seed42/6000edits/AlphaEdit/run_000

    # Full checkpoint evaluation with both metric sets
    python scripts/eval_prob_preference.py evaluate \\
        --seed 42 --alg_name AlphaEdit --ordering key_clustered \\
        --checkpoints 9 19 29 49 99

    # Evaluate with custom checkpoint dir
    python scripts/eval_prob_preference.py evaluate \\
        --checkpoint_dir ~/.cache/alphaedit_checkpoints/matched_ordering/AlphaEdit/key_clustered/seed42 \\
        --dataset_path results/matched_ordering/orderings/key_clustered_seed42.json \\
        --checkpoints 19 29 49
"""

import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Metric computation from raw NLL values (matches summarize.py exactly)
# ---------------------------------------------------------------------------

def compute_prob_preference_metrics(
    rewrite_probs: List[Dict[str, float]],
    paraphrase_probs: List[Dict[str, float]],
    neighborhood_probs: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute probability-preference metrics from NLL values.

    Each *_probs entry is {"target_new": nll_new, "target_true": nll_true}.
    Lower NLL = higher probability.

    Returns dict with:
        rewrite_success:     fraction where NLL(true) > NLL(new)  [= P(new) > P(true)]
        paraphrase_success:  fraction where NLL(true) > NLL(new)
        neighborhood_success: fraction where NLL(true) < NLL(new) [= P(true) > P(new)]
        rewrite_diff:        mean( P(new) - P(true) )
        paraphrase_diff:     mean( P(new) - P(true) )
        neighborhood_diff:   mean( P(true) - P(new) )
    """
    metrics = {}

    # Efficacy: target_new should be more probable (lower NLL)
    # summarize.py line 65: x["target_true"] > x["target_new"]
    if rewrite_probs:
        metrics["rewrite_success"] = float(np.mean([
            x["target_true"] > x["target_new"] for x in rewrite_probs
        ]))
        metrics["rewrite_diff"] = float(np.mean([
            np.exp(-x["target_new"]) - np.exp(-x["target_true"])
            for x in rewrite_probs
        ]))
    else:
        metrics["rewrite_success"] = float("nan")
        metrics["rewrite_diff"] = float("nan")

    # Paraphrase: target_new should be more probable (lower NLL)
    # summarize.py line 65: x["target_true"] > x["target_new"]
    if paraphrase_probs:
        metrics["paraphrase_success"] = float(np.mean([
            x["target_true"] > x["target_new"] for x in paraphrase_probs
        ]))
        metrics["paraphrase_diff"] = float(np.mean([
            np.exp(-x["target_new"]) - np.exp(-x["target_true"])
            for x in paraphrase_probs
        ]))
    else:
        metrics["paraphrase_success"] = float("nan")
        metrics["paraphrase_diff"] = float("nan")

    # Neighborhood: target_true should be more probable (lower NLL)
    # summarize.py line 87: x["target_true"] < x["target_new"]
    if neighborhood_probs:
        metrics["neighborhood_success"] = float(np.mean([
            x["target_true"] < x["target_new"] for x in neighborhood_probs
        ]))
        metrics["neighborhood_diff"] = float(np.mean([
            np.exp(-x["target_true"]) - np.exp(-x["target_new"])
            for x in neighborhood_probs
        ]))
    else:
        metrics["neighborhood_success"] = float("nan")
        metrics["neighborhood_diff"] = float("nan")

    return metrics


def compute_argmax_metrics(
    rewrite_correct: List[bool],
    paraphrase_correct: List[bool],
    neighborhood_correct: List[bool],
) -> Dict[str, float]:
    """Compute argmax-based metrics (is correct token the top-1 prediction?)."""
    return {
        "rewrite_acc": float(np.mean(rewrite_correct)) if rewrite_correct else float("nan"),
        "paraphrase_acc": float(np.mean(paraphrase_correct)) if paraphrase_correct else float("nan"),
        "neighborhood_acc": float(np.mean(neighborhood_correct)) if neighborhood_correct else float("nan"),
    }


# ---------------------------------------------------------------------------
# MODE 1: RESCORE existing per-case result files (no GPU)
# ---------------------------------------------------------------------------

def rescore_result_dir(
    result_dir: Path,
    output_path: Optional[Path] = None,
    num_edits_per_batch: int = 100,
) -> Dict:
    """Re-score existing per-case result files using probability-preference metrics.

    Reads *_edits-case_*.json files from result_dir, extracts stored NLL values,
    and computes both probability-preference and argmax metrics.

    Args:
        result_dir: Directory containing per-case result JSON files.
        output_path: Where to save the rescore results. If None, auto-generated.
        num_edits_per_batch: Number of edits per batch (for cohort analysis).

    Returns:
        Dict with aggregate metrics and per-case details.
    """
    # Find all case files
    case_files = sorted(
        result_dir.glob("*case_*.json"),
        key=lambda x: int(str(x).split("_")[-1].split(".")[0]),
    )

    if not case_files:
        print(f"ERROR: No case files found in {result_dir}")
        sys.exit(1)

    print(f"Found {len(case_files)} per-case result files in {result_dir}")

    # Accumulate per-case metrics
    per_case_prob_pref = []   # probability-preference metrics per case
    per_case_argmax = []      # argmax metrics per case
    raw_per_case = []         # full per-case detail

    n_missing_probs = 0
    n_missing_correct = 0
    n_corrupt = 0

    for case_file in case_files:
        try:
            with open(case_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            n_corrupt += 1
            continue

        case_id = data["case_id"]
        post = data.get("post", {})

        # Extract raw probability data
        rewrite_probs = post.get("rewrite_prompts_probs", [])
        paraphrase_probs = post.get("paraphrase_prompts_probs", [])
        neighborhood_probs = post.get("neighborhood_prompts_probs", [])

        rewrite_correct = post.get("rewrite_prompts_correct", [])
        paraphrase_correct = post.get("paraphrase_prompts_correct", [])
        neighborhood_correct = post.get("neighborhood_prompts_correct", [])

        if not rewrite_probs:
            n_missing_probs += 1
            continue

        # Compute probability-preference metrics
        pp_metrics = compute_prob_preference_metrics(
            rewrite_probs, paraphrase_probs, neighborhood_probs,
        )
        per_case_prob_pref.append(pp_metrics)

        # Compute argmax metrics
        am_metrics = compute_argmax_metrics(
            rewrite_correct, paraphrase_correct, neighborhood_correct,
        )
        if not rewrite_correct:
            n_missing_correct += 1
        per_case_argmax.append(am_metrics)

        raw_per_case.append({
            "case_id": case_id,
            "prob_preference": pp_metrics,
            "argmax": am_metrics,
        })

    if n_corrupt > 0:
        print(f"  WARNING: {n_corrupt} corrupt/empty JSON files skipped")
    if n_missing_probs > 0:
        print(f"  WARNING: {n_missing_probs} cases missing probability data")
    if n_missing_correct > 0:
        print(f"  NOTE: {n_missing_correct} cases missing argmax correctness data")

    n_cases = len(per_case_prob_pref)
    print(f"  Scored {n_cases} cases")

    # Aggregate metrics
    def aggregate(cases, key):
        vals = [c[key] for c in cases if not np.isnan(c[key])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    # Probability-preference (official)
    pp_agg = {
        "rewrite_success": aggregate(per_case_prob_pref, "rewrite_success"),
        "paraphrase_success": aggregate(per_case_prob_pref, "paraphrase_success"),
        "neighborhood_success": aggregate(per_case_prob_pref, "neighborhood_success"),
        "rewrite_diff": aggregate(per_case_prob_pref, "rewrite_diff"),
        "paraphrase_diff": aggregate(per_case_prob_pref, "paraphrase_diff"),
        "neighborhood_diff": aggregate(per_case_prob_pref, "neighborhood_diff"),
    }

    # Argmax (our previous metric)
    am_agg = {
        "rewrite_acc": aggregate(per_case_argmax, "rewrite_acc"),
        "paraphrase_acc": aggregate(per_case_argmax, "paraphrase_acc"),
        "neighborhood_acc": aggregate(per_case_argmax, "neighborhood_acc"),
    }

    # Cohort analysis (groups of num_edits_per_batch)
    n_cohorts = (n_cases + num_edits_per_batch - 1) // num_edits_per_batch
    cohort_metrics = {}
    for c in range(n_cohorts):
        start = c * num_edits_per_batch
        end = min((c + 1) * num_edits_per_batch, n_cases)
        cohort_pp = per_case_prob_pref[start:end]
        cohort_am = per_case_argmax[start:end]

        cohort_metrics[str(c)] = {
            "edits_range": f"{start}-{end}",
            "n_facts": len(cohort_pp),
            "prob_preference": {
                "rewrite_success": aggregate(cohort_pp, "rewrite_success"),
                "paraphrase_success": aggregate(cohort_pp, "paraphrase_success"),
                "neighborhood_success": aggregate(cohort_pp, "neighborhood_success"),
            },
            "argmax": {
                "rewrite_acc": aggregate(cohort_am, "rewrite_acc"),
                "paraphrase_acc": aggregate(cohort_am, "paraphrase_acc"),
                "neighborhood_acc": aggregate(cohort_am, "neighborhood_acc"),
            },
        }

    # Named cohort slices
    def slice_metrics(cases_pp, cases_am):
        return {
            "prob_preference": {
                "rewrite_success": aggregate(cases_pp, "rewrite_success"),
                "paraphrase_success": aggregate(cases_pp, "paraphrase_success"),
                "neighborhood_success": aggregate(cases_pp, "neighborhood_success"),
            },
            "argmax": {
                "rewrite_acc": aggregate(cases_am, "rewrite_acc"),
                "paraphrase_acc": aggregate(cases_am, "paraphrase_acc"),
                "neighborhood_acc": aggregate(cases_am, "neighborhood_acc"),
            },
        }

    first_1k_pp = per_case_prob_pref[:1000] if n_cases >= 1000 else per_case_prob_pref
    first_1k_am = per_case_argmax[:1000] if n_cases >= 1000 else per_case_argmax
    latest_1k_pp = per_case_prob_pref[-1000:] if n_cases >= 1000 else per_case_prob_pref
    latest_1k_am = per_case_argmax[-1000:] if n_cases >= 1000 else per_case_argmax
    latest_100_pp = per_case_prob_pref[-100:]
    latest_100_am = per_case_argmax[-100:]

    result = {
        "source_dir": str(result_dir),
        "n_cases": n_cases,
        "num_edits_per_batch": num_edits_per_batch,
        "metric_definitions": {
            "prob_preference": "Official AlphaEdit metric: pairwise NLL comparison (P(target) > P(alternative))",
            "argmax": "Strict metric: is the correct token the argmax prediction at every position?",
        },
        "all_facts": {
            "prob_preference": pp_agg,
            "argmax": am_agg,
        },
        "first_1k": slice_metrics(first_1k_pp, first_1k_am),
        "latest_1k": slice_metrics(latest_1k_pp, latest_1k_am),
        "latest_100": slice_metrics(latest_100_pp, latest_100_am),
        "cohort_metrics": cohort_metrics,
    }

    # Print comparison table
    _print_rescore_summary(result)

    # Save
    if output_path is None:
        output_path = result_dir / "prob_preference_rescore.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {output_path}")

    return result


def _print_rescore_summary(result: Dict):
    """Print a comparison of prob-preference vs argmax metrics."""
    print(f"\n{'='*85}")
    print("PROBABILITY-PREFERENCE vs ARGMAX METRICS COMPARISON")
    print(f"{'='*85}")
    print(f"Source: {result['source_dir']}")
    print(f"Cases:  {result['n_cases']}")
    print()

    pp = result["all_facts"]["prob_preference"]
    am = result["all_facts"]["argmax"]

    header = f"{'Metric':<22} {'Prob-Pref (official)':>20} {'Argmax (strict)':>18} {'Delta':>10}"
    print(header)
    print("-" * 75)

    for name, pp_key, am_key in [
        ("Efficacy", "rewrite_success", "rewrite_acc"),
        ("Paraphrase", "paraphrase_success", "paraphrase_acc"),
        ("Neighborhood", "neighborhood_success", "neighborhood_acc"),
    ]:
        pp_val = pp[pp_key]
        am_val = am[am_key]
        if np.isnan(pp_val) or np.isnan(am_val):
            delta_str = "N/A"
        else:
            delta = pp_val - am_val
            delta_str = f"{delta:+.4f}"
        print(f"{name:<22} {pp_val:>20.4f} {am_val:>18.4f} {delta_str:>10}")

    # Prob diffs (continuous)
    print()
    print(f"{'Continuous diffs':}")
    for name, key in [
        ("  Rewrite diff", "rewrite_diff"),
        ("  Paraphrase diff", "paraphrase_diff"),
        ("  Neighborhood diff", "neighborhood_diff"),
    ]:
        print(f"{name:<22} {pp[key]:>20.4f}")

    # Cohort breakdown (first 5 + last 5)
    cohorts = result.get("cohort_metrics", {})
    if cohorts:
        n = len(cohorts)
        show_indices = list(range(min(5, n))) + list(range(max(5, n - 5), n))
        show_indices = sorted(set(show_indices))

        print(f"\n{'Cohort':<12} {'PP Eff':>8} {'PP Para':>8} {'PP Neigh':>9} "
              f"{'AM Eff':>8} {'AM Para':>8} {'AM Neigh':>9}")
        print("-" * 68)
        for i in show_indices:
            si = str(i)
            if si not in cohorts:
                continue
            c = cohorts[si]
            pp_c = c["prob_preference"]
            am_c = c["argmax"]
            label = c["edits_range"]
            print(f"{label:<12} "
                  f"{pp_c['rewrite_success']:>8.4f} "
                  f"{pp_c['paraphrase_success']:>8.4f} "
                  f"{pp_c['neighborhood_success']:>9.4f} "
                  f"{am_c['rewrite_acc']:>8.4f} "
                  f"{am_c['paraphrase_acc']:>8.4f} "
                  f"{am_c['neighborhood_acc']:>9.4f}")
            if i == show_indices[min(4, len(show_indices) - 1)] and n > 10:
                print(f"{'  ...':}")


# ---------------------------------------------------------------------------
# MODE 2: EVALUATE from checkpoints (requires GPU)
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(model_name: str, ckpt_path: Path):
    """Load base model and apply checkpoint weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_resolve import resolve_model_path

    token = os.environ.get("HF_TOKEN")
    model_path = resolve_model_path(model_name)
    print(f"  Loading base model: {model_path} (float16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, token=token,
    ).cuda()
    tok = AutoTokenizer.from_pretrained(model_path, token=token)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # Required for multi-token NLL scoring

    weights_file = ckpt_path / "model_weights.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights at {weights_file}")

    print(f"  Applying checkpoint weights: {ckpt_path}")
    weights = torch.load(str(weights_file), map_location="cuda")
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.cuda().half())
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    print(f"  Loaded {loaded} weight tensors")

    return model, tok


def swap_checkpoint_weights(model, ckpt_path: Path):
    """Swap model weights from a new checkpoint (reuse loaded model)."""
    import torch

    weights_file = ckpt_path / "model_weights.pt"
    print(f"  Swapping weights from: {ckpt_path}")
    weights = torch.load(str(weights_file), map_location="cuda")
    param_dict = dict(model.named_parameters())
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.cuda().half())
    del weights
    torch.cuda.empty_cache()


def test_batch_prediction_vendor(
    model,
    tok,
    prefixes: List[str],
    which_correct: List[int],
    target_new: str,
    target_true: str,
) -> Tuple[List[Dict[str, float]], List[bool]]:
    """Compute NLL-based probability scores and argmax correctness.

    This is a faithful reproduction of the vendor's test_batch_prediction
    from vendor/AlphaEdit/experiments/py/eval_utils_counterfact.py.

    Returns:
        probs: List of {"target_new": nll_new, "target_true": nll_true} per prefix.
               Lower NLL = higher probability.
        targets_correct: List of bool, whether argmax matches correct target at every position.
    """
    import torch

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])

    if "llama" in model.config._name_or_path.lower():
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [lengths - 1 for lengths in prefix_lens]

    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if "llama" in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities (average NLL per token)
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on correct targets (argmax at every position)
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def evaluate_record_full(model, tok, record: Dict) -> Dict:
    """Evaluate a single record with both probability-preference and argmax metrics.

    Matches the vendor's compute_rewrite_quality_counterfact exactly for the
    probability computation, then applies both metric types.
    """
    subject, target_new, target_true = (
        record["requested_rewrite"][x]
        for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]

    probs, targets_correct = test_batch_prediction_vendor(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )

    # Unflatten
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    # Probability-preference metrics
    pp = compute_prob_preference_metrics(ret_probs[0], ret_probs[1], ret_probs[2])

    # Argmax metrics
    am = compute_argmax_metrics(ret_corrects[0], ret_corrects[1], ret_corrects[2])

    return {
        "case_id": record["case_id"],
        "prob_preference": pp,
        "argmax": am,
        # Also store raw data for future re-scoring
        "raw": {
            "rewrite_prompts_probs": ret_probs[0],
            "paraphrase_prompts_probs": ret_probs[1],
            "neighborhood_prompts_probs": ret_probs[2],
            "rewrite_prompts_correct": ret_corrects[0],
            "paraphrase_prompts_correct": ret_corrects[1],
            "neighborhood_prompts_correct": ret_corrects[2],
        },
    }


def evaluate_records_batched_full(
    model, tok, records: List[Dict], batch_size: int = 8
) -> List[Dict]:
    """Evaluate many records using the full vendor-compatible multi-token NLL protocol.

    Unlike the mega-batch first-token approach in eval_matched_ordering.py, this uses
    the exact same multi-token scoring as the vendor's test_batch_prediction.

    batch_size controls how many RECORDS are processed before logging progress.
    Each record is evaluated independently to match the vendor protocol exactly
    (since different records have different target tokens with different lengths).
    """
    all_results = []

    for i, record in enumerate(records):
        result = evaluate_record_full(model, tok, record)
        all_results.append(result)

        if (i + 1) % 500 == 0 or (i + 1) == len(records):
            pp_eff = np.mean([r["prob_preference"]["rewrite_success"] for r in all_results])
            pp_neigh = np.mean([r["prob_preference"]["neighborhood_success"] for r in all_results])
            am_eff = np.mean([r["argmax"]["rewrite_acc"] for r in all_results])
            am_neigh = np.mean([r["argmax"]["neighborhood_acc"] for r in all_results])
            print(
                f"    [{i+1}/{len(records)}] "
                f"PP: eff={pp_eff:.4f} neigh={pp_neigh:.4f}  "
                f"AM: eff={am_eff:.4f} neigh={am_neigh:.4f}"
            )

    return all_results


def evaluate_checkpoint_full(
    model_name: str,
    ckpt_path: Path,
    records: List[Dict],
    num_edits_per_batch: int,
    total_edits: int,
    model_tok_cache: Optional[tuple] = None,
) -> Tuple[Dict, tuple]:
    """Full evaluation at one checkpoint with both metric sets.

    Returns (summary, (model, tok)) for model reuse across checkpoints.
    """
    import torch

    print(f"\n{'='*70}")
    print(f"Evaluating checkpoint: {ckpt_path.name} ({total_edits} edits)")
    print(f"  Records to evaluate: {len(records)}")
    print(f"  Mode: full multi-token NLL (vendor-compatible)")
    print(f"{'='*70}")

    if model_tok_cache is not None:
        model, tok = model_tok_cache
        swap_checkpoint_weights(model, ckpt_path)
    else:
        model, tok = load_model_from_checkpoint(model_name, ckpt_path)

    # Evaluate all records
    results = evaluate_records_batched_full(model, tok, records)

    # Aggregate metrics
    def agg(results_slice, metric_type, key):
        vals = [r[metric_type][key] for r in results_slice]
        return round(float(np.mean(vals)), 4)

    # Cohort breakdown
    n_cohorts = (total_edits + num_edits_per_batch - 1) // num_edits_per_batch
    cohort_metrics = {}
    for c in range(n_cohorts):
        start = c * num_edits_per_batch
        end = min((c + 1) * num_edits_per_batch, len(results))
        cohort = results[start:end]
        if cohort:
            cohort_metrics[str(c)] = {
                "edits_range": f"{start}-{end}",
                "n_facts": len(cohort),
                "prob_preference": {
                    "rewrite_success": agg(cohort, "prob_preference", "rewrite_success"),
                    "paraphrase_success": agg(cohort, "prob_preference", "paraphrase_success"),
                    "neighborhood_success": agg(cohort, "prob_preference", "neighborhood_success"),
                },
                "argmax": {
                    "rewrite_acc": agg(cohort, "argmax", "rewrite_acc"),
                    "paraphrase_acc": agg(cohort, "argmax", "paraphrase_acc"),
                    "neighborhood_acc": agg(cohort, "argmax", "neighborhood_acc"),
                },
            }

    # Named cohort slices
    def slice_summary(res_slice):
        return {
            "prob_preference": {
                "rewrite_success": agg(res_slice, "prob_preference", "rewrite_success"),
                "paraphrase_success": agg(res_slice, "prob_preference", "paraphrase_success"),
                "neighborhood_success": agg(res_slice, "prob_preference", "neighborhood_success"),
            },
            "argmax": {
                "rewrite_acc": agg(res_slice, "argmax", "rewrite_acc"),
                "paraphrase_acc": agg(res_slice, "argmax", "paraphrase_acc"),
                "neighborhood_acc": agg(res_slice, "argmax", "neighborhood_acc"),
            },
        }

    first_1k = results[:1000] if len(results) >= 1000 else results
    latest_1k = results[-1000:] if len(results) >= 1000 else results
    latest_100 = results[-100:]

    # Retention AUC on prob-preference efficacy
    cohort_effs = [
        cohort_metrics[str(c)]["prob_preference"]["rewrite_success"]
        for c in range(n_cohorts) if str(c) in cohort_metrics
    ]
    retention_auc = float(np.trapezoid(cohort_effs) / max(len(cohort_effs) - 1, 1)) if len(cohort_effs) > 1 else (cohort_effs[0] if cohort_effs else 0.0)

    summary = {
        "checkpoint": ckpt_path.name,
        "total_edits": total_edits,
        "n_evaluated": len(results),
        "all_facts": slice_summary(results),
        "first_1k": slice_summary(first_1k),
        "latest_1k": slice_summary(latest_1k),
        "latest_100": slice_summary(latest_100),
        "retention_auc_pp": round(retention_auc, 4),
        "cohort_metrics": cohort_metrics,
    }

    # Print summary
    pp = summary["all_facts"]["prob_preference"]
    am = summary["all_facts"]["argmax"]
    print(f"\n  Results at {total_edits} edits:")
    print(f"    Prob-Pref: eff={pp['rewrite_success']:.4f}  "
          f"para={pp['paraphrase_success']:.4f}  "
          f"neigh={pp['neighborhood_success']:.4f}")
    print(f"    Argmax:    eff={am['rewrite_acc']:.4f}  "
          f"para={am['paraphrase_acc']:.4f}  "
          f"neigh={am['neighborhood_acc']:.4f}")

    return summary, (model, tok)


def evaluate_from_checkpoints(args):
    """Main evaluation pipeline: load checkpoints, run inference, compute metrics."""
    import torch

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_resolve import resolve_model_path

    model_name = resolve_model_path(args.model_name)

    # Find checkpoint directory
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir).expanduser()
    else:
        # Auto-resolve from matched ordering conventions
        ckpt_root = Path(os.environ.get(
            "CHECKPOINT_ROOT",
            str(Path.home() / ".cache" / "alphaedit_checkpoints"),
        ))
        if args.ordering:
            ckpt_dir = ckpt_root / "matched_ordering" / args.alg_name / args.ordering / f"seed{args.seed}"
        else:
            ckpt_dir = ckpt_root / "failure_curve" / args.alg_name / f"seed{args.seed}"

    print(f"Checkpoint dir: {ckpt_dir}")

    # Verify checkpoints exist
    for batch_idx in args.checkpoints:
        batch_path = ckpt_dir / f"batch_{batch_idx}"
        if not batch_path.exists():
            print(f"ERROR: Checkpoint batch_{batch_idx} not found at {batch_path}")
            available = sorted([d.name for d in ckpt_dir.iterdir() if d.is_dir()])
            print(f"  Available: {available}")
            sys.exit(1)
    print(f"  Checkpoints verified: {['batch_' + str(b) for b in args.checkpoints]}")

    # Load dataset
    if args.dataset_path:
        ds_path = Path(args.dataset_path)
    else:
        data_root = Path(os.environ.get("DATA_ROOT", "data/dsets"))
        candidates = [
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
            data_root / "multi_counterfact.json",
            PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json",
        ]
        ds_path = None
        for c in candidates:
            if c.exists():
                ds_path = c
                break
        if ds_path is None:
            print("ERROR: Cannot find multi_counterfact.json")
            print("  Tried:", [str(c) for c in candidates])
            print("  Specify with --dataset_path")
            sys.exit(1)

    print(f"  Dataset: {ds_path}")
    with open(ds_path) as f:
        all_records = json.load(f)
    print(f"  Total records in dataset: {len(all_records)}")

    # Evaluate each checkpoint
    all_summaries = {}
    model_cache = None
    for batch_idx in args.checkpoints:
        total_edits = (batch_idx + 1) * args.num_edits
        records_to_eval = all_records[:total_edits]
        ckpt_path = ckpt_dir / f"batch_{batch_idx}"

        summary, model_cache = evaluate_checkpoint_full(
            model_name=model_name,
            ckpt_path=ckpt_path,
            records=records_to_eval,
            num_edits_per_batch=args.num_edits,
            total_edits=total_edits,
            model_tok_cache=model_cache,
        )
        all_summaries[f"{total_edits}_edits"] = summary

    # Cleanup
    if model_cache:
        del model_cache
        torch.cuda.empty_cache()

    # Determine output path
    result_root = Path(os.environ.get("RESULT_ROOT", str(PROJECT_ROOT / "results")))

    alg_name = args.alg_name or "unknown"
    if args.ordering:
        out_dir = result_root / "prob_preference" / alg_name / args.ordering / f"seed{args.seed}"
    else:
        out_dir = result_root / "prob_preference" / alg_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prob_pref_eval_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Final comparison table
    print(f"\n{'='*100}")
    print("PROBABILITY-PREFERENCE EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"{'Edits':<8} {'PP Eff':>8} {'PP Para':>8} {'PP Neigh':>9} "
          f"{'AM Eff':>8} {'AM Para':>8} {'AM Neigh':>9} "
          f"{'1K PP Eff':>10} {'1K AM Eff':>10} {'AUC':>6}")
    print("-" * 100)
    for key in sorted(all_summaries.keys()):
        s = all_summaries[key]
        pp = s["all_facts"]["prob_preference"]
        am = s["all_facts"]["argmax"]
        pp1k = s["first_1k"]["prob_preference"]
        am1k = s["first_1k"]["argmax"]
        print(
            f"{s['total_edits']:<8} "
            f"{pp['rewrite_success']:>8.4f} "
            f"{pp['paraphrase_success']:>8.4f} "
            f"{pp['neighborhood_success']:>9.4f} "
            f"{am['rewrite_acc']:>8.4f} "
            f"{am['paraphrase_acc']:>8.4f} "
            f"{am['neighborhood_acc']:>9.4f} "
            f"{pp1k['rewrite_success']:>10.4f} "
            f"{am1k['rewrite_acc']:>10.4f} "
            f"{s['retention_auc_pp']:>6.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probability-preference evaluation (official AlphaEdit metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # --- RESCORE subcommand ---
    rescore_parser = subparsers.add_parser(
        "rescore",
        help="Re-score existing per-case result files (no GPU needed)",
    )
    rescore_parser.add_argument(
        "--result_dir", type=str, required=True,
        help="Directory containing per-case result JSON files (e.g., 100_edits-case_*.json)",
    )
    rescore_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: {result_dir}/prob_preference_rescore.json)",
    )
    rescore_parser.add_argument(
        "--num_edits", type=int, default=100,
        help="Number of edits per batch (for cohort analysis, default: 100)",
    )

    # --- EVALUATE subcommand ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Full checkpoint evaluation with both metric sets (requires GPU)",
    )
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--alg_name", type=str, default=None,
                             help="Algorithm name (AlphaEdit, MEMIT, MEMIT-Seq-lp1.0-ld0.0-cache0)")
    eval_parser.add_argument("--ordering", type=str, default=None,
                             help="Ordering type (key_clustered, key_dispersed)")
    eval_parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    eval_parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=[9, 19, 29, 49],
        help="Batch indices to evaluate (default: 9 19 29 49 = 1K, 2K, 3K, 5K edits)",
    )
    eval_parser.add_argument("--num_edits", type=int, default=100,
                             help="Number of edits per batch (default: 100)")
    eval_parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Explicit checkpoint directory (overrides auto-resolved path)",
    )
    eval_parser.add_argument(
        "--dataset_path", default=None,
        help="Path to dataset JSON (auto-detected if not specified)",
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "rescore":
        result_dir = Path(args.result_dir).expanduser()
        output_path = Path(args.output) if args.output else None
        rescore_result_dir(result_dir, output_path, args.num_edits)

    elif args.mode == "evaluate":
        evaluate_from_checkpoints(args)


if __name__ == "__main__":
    main()
