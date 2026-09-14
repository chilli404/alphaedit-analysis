#!/usr/bin/env python3
"""
Compare probability-preference vs argmax evaluation metrics on any model checkpoint.

This script evaluates a model checkpoint using BOTH metrics that appear in the
knowledge-editing literature:

  1. PROBABILITY-PREFERENCE (used by EvoEdit, MEMIT, ROME papers):
     Success = P(target_new) > P(target_true) for the edited prompt
     Computed as NLL(target_true) > NLL(target_new) (lower NLL = higher prob)

  2. ARGMAX (used by some reproducibility studies):
     Success = every token of the target is the argmax prediction
     Stricter: requires the model to confidently predict EVERY token

The probability-preference metric is always >= the argmax metric. The gap
quantifies how much "soft preference" editing achieves vs "hard commitment."

Supports three checkpoint formats:
  - Lightweight layer weights (model_weights.pt) — our checkpoint format
  - Full model directory (config.json + safetensors/bin) — HuggingFace format
  - Base model only (no checkpoint) — for baseline measurement

Usage:
  # Evaluate with lightweight checkpoint (our format)
  python scripts/compare_eval_metrics.py \\
      --checkpoint_dir ~/.cache/evoedit_anchor/fb_random0_seed42/edits_010000 \\
      --edits 10000

  # Evaluate a full HuggingFace checkpoint
  python scripts/compare_eval_metrics.py \\
      --checkpoint_dir /path/to/hf_checkpoint \\
      --checkpoint_format hf --edits 2000

  # Evaluate base model (pre-edit baseline)
  python scripts/compare_eval_metrics.py --base_only --edits 1000

  # Use a specific ordering stream
  python scripts/compare_eval_metrics.py \\
      --checkpoint_dir /path/to/ckpt --edits 5000 \\
      --stream_path results/matched_ordering/orderings/key_clustered_seed42.json

  # Quick test with fewer records
  python scripts/compare_eval_metrics.py \\
      --checkpoint_dir /path/to/ckpt --edits 10000 --max_records 500
"""

import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_model(model_name: str) -> str:
    """Resolve model name to loadable path."""
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_resolve import resolve_model_path
    return resolve_model_path(model_name)


def load_base_model(model_name: str, dtype=None):
    """Load base model without any checkpoint."""
    token = os.environ.get("HF_TOKEN")
    path = resolve_model(model_name)
    print(f"Loading base model: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, token=token,
    ).cuda()
    tok = AutoTokenizer.from_pretrained(path, token=token)
    tok.pad_token = tok.eos_token
    return model, tok


def load_lightweight_checkpoint(model_name: str, ckpt_dir: Path, dtype=None):
    """Load base model + apply lightweight layer-weight checkpoint."""
    model, tok = load_base_model(model_name, dtype)
    weights_file = ckpt_dir / "model_weights.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"No model_weights.pt at {weights_file}")
    print(f"Applying layer weights from: {ckpt_dir}")
    weights = torch.load(str(weights_file), map_location="cuda", weights_only=True)
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.cuda().to(param_dict[name].dtype))
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    print(f"  Applied {loaded} weight tensors")
    return model, tok


def load_hf_checkpoint(ckpt_dir: Path, dtype=None):
    """Load a full HuggingFace-format checkpoint."""
    token = os.environ.get("HF_TOKEN")
    print(f"Loading HF checkpoint: {ckpt_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir), torch_dtype=dtype, token=token,
    ).cuda()
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), token=token)
    tok.pad_token = tok.eos_token
    return model, tok


# ---------------------------------------------------------------------------
# Dual-metric evaluation core
# ---------------------------------------------------------------------------

def test_batch_prediction_dual(
    model,
    tok,
    prefixes: List[str],
    which_correct: List[int],
    target_new: str,
    target_true: str,
) -> Tuple[List[Dict], List[bool]]:
    """
    Evaluate prompts using the EXACT same logic as EvoEdit/MEMIT/AlphaEdit
    eval_utils_counterfact.py::test_batch_prediction.

    Returns:
      probs: list of {"target_new": nll, "target_true": nll} per prompt pair
      targets_correct: list of bool (argmax) for the "correct" side
    """
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

    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if "llama" in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

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


def evaluate_record_dual(model, tok, record: Dict) -> Dict:
    """Evaluate one MCF record with both probability-preference and argmax metrics."""
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

    probs, targets_correct = test_batch_prediction_dual(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )

    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    prob_eff = np.mean([x["target_true"] > x["target_new"] for x in ret_probs[0]])
    prob_gen = np.mean([x["target_true"] > x["target_new"] for x in ret_probs[1]])
    prob_spec = np.mean([x["target_true"] < x["target_new"] for x in ret_probs[2]])

    argmax_eff = float(np.mean(ret_corrects[0]))
    argmax_gen = float(np.mean(ret_corrects[1]))
    argmax_spec = float(np.mean(ret_corrects[2]))

    # Also compute probability difference (continuous metric)
    prob_diff_eff = np.mean([
        np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in ret_probs[0]
    ])
    prob_diff_gen = np.mean([
        np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in ret_probs[1]
    ])
    prob_diff_spec = np.mean([
        np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in ret_probs[2]
    ])

    return {
        "case_id": record["case_id"],
        "prob_efficacy": float(prob_eff),
        "prob_generalization": float(prob_gen),
        "prob_specificity": float(prob_spec),
        "prob_diff_efficacy": float(prob_diff_eff),
        "prob_diff_generalization": float(prob_diff_gen),
        "prob_diff_specificity": float(prob_diff_spec),
        "argmax_efficacy": argmax_eff,
        "argmax_generalization": argmax_gen,
        "argmax_specificity": argmax_spec,
        "rewrite_probs": ret_probs[0],
        "paraphrase_probs": ret_probs[1],
        "neighborhood_probs": ret_probs[2],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare probability-preference vs argmax metrics on any checkpoint"
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--checkpoint_format", choices=["lightweight", "hf"],
                        default="lightweight",
                        help="Checkpoint format: lightweight (model_weights.pt) or hf (full model)")
    parser.add_argument("--base_only", action="store_true",
                        help="Evaluate base model without any checkpoint (pre-edit baseline)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--edits", type=int, required=True,
                        help="Number of edits to evaluate (selects first N records)")
    parser.add_argument("--stream_path", type=str, default=None,
                        help="Path to ordering stream JSON (for exact fact order)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to multi_counterfact.json")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Evaluate at most this many records")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to this JSON file")
    parser.add_argument("--label", type=str, default=None,
                        help="Label for this run in the output table")
    args = parser.parse_args()

    if not args.base_only and not args.checkpoint_dir:
        parser.error("Specify --checkpoint_dir or --base_only")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if args.stream_path:
        ds_path = Path(args.stream_path)
        with open(ds_path) as f:
            all_records = json.load(f)
        print(f"Dataset: {ds_path} ({len(all_records)} records, stream-ordered)")
    else:
        candidates = [
            Path(args.dataset_path) if args.dataset_path else None,
            PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
        ]
        ds_path = next((c for c in candidates if c and c.exists()), None)
        if ds_path is None:
            print("ERROR: Cannot find multi_counterfact.json")
            sys.exit(1)
        with open(ds_path) as f:
            all_records = json.load(f)
        print(f"Dataset: {ds_path} ({len(all_records)} records)")

    n_eval = min(args.edits, len(all_records))
    if args.max_records:
        n_eval = min(n_eval, args.max_records)
    records = all_records[:n_eval]
    print(f"Evaluating: {n_eval} records")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    if args.base_only:
        model, tok = load_base_model(args.model_name)
        ckpt_label = "base (no edits)"
    elif args.checkpoint_format == "hf":
        model, tok = load_hf_checkpoint(Path(args.checkpoint_dir))
        ckpt_label = args.checkpoint_dir
    else:
        model, tok = load_lightweight_checkpoint(
            args.model_name, Path(args.checkpoint_dir)
        )
        ckpt_label = args.checkpoint_dir

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print(f"\nEvaluating {len(records)} records with both metrics...")
    results = []
    for i, record in enumerate(records):
        result = evaluate_record_dual(model, tok, record)
        results.append(result)
        if (i + 1) % 200 == 0:
            pe = np.mean([r["prob_efficacy"] for r in results])
            ae = np.mean([r["argmax_efficacy"] for r in results])
            ps = np.mean([r["prob_specificity"] for r in results])
            a_s = np.mean([r["argmax_specificity"] for r in results])
            print(
                f"  [{i+1}/{len(records)}] "
                f"prob(eff/spec)={pe:.4f}/{ps:.4f}  "
                f"argmax(eff/spec)={ae:.4f}/{a_s:.4f}"
            )

    # ------------------------------------------------------------------
    # 4. Aggregate
    # ------------------------------------------------------------------
    prob_metrics = {
        "efficacy": np.mean([r["prob_efficacy"] for r in results]) * 100,
        "generalization": np.mean([r["prob_generalization"] for r in results]) * 100,
        "specificity": np.mean([r["prob_specificity"] for r in results]) * 100,
    }
    argmax_metrics = {
        "efficacy": np.mean([r["argmax_efficacy"] for r in results]) * 100,
        "generalization": np.mean([r["argmax_generalization"] for r in results]) * 100,
        "specificity": np.mean([r["argmax_specificity"] for r in results]) * 100,
    }
    prob_diff_metrics = {
        "efficacy": np.mean([r["prob_diff_efficacy"] for r in results]) * 100,
        "generalization": np.mean([r["prob_diff_generalization"] for r in results]) * 100,
        "specificity": np.mean([r["prob_diff_specificity"] for r in results]) * 100,
    }

    label = args.label or ckpt_label

    # ------------------------------------------------------------------
    # 5. Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Dual-Metric Comparison: {label}")
    print(f"  Edits evaluated: {n_eval}")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Prob-pref (%)':>14} {'Argmax (%)':>14} {'Gap (pp)':>12} {'Prob-diff':>12}")
    print(f"{'-'*80}")
    for key in ["efficacy", "generalization", "specificity"]:
        gap = prob_metrics[key] - argmax_metrics[key]
        print(
            f"{key.capitalize():<20} "
            f"{prob_metrics[key]:>14.2f} "
            f"{argmax_metrics[key]:>14.2f} "
            f"{gap:>+12.2f} "
            f"{prob_diff_metrics[key]:>12.2f}"
        )
    print(f"{'='*80}")

    # Cohort breakdown (every 1K edits)
    cohort_size = 1000
    n_cohorts = max(1, len(results) // cohort_size)
    if n_cohorts > 1:
        print(f"\nCohort Breakdown (every {cohort_size} edits):")
        print(f"{'Cohort':<12} {'P-Eff':>8} {'A-Eff':>8} {'P-Gen':>8} {'A-Gen':>8} {'P-Spec':>8} {'A-Spec':>8}")
        print("-" * 68)
        for c in range(n_cohorts):
            start = c * cohort_size
            end = min((c + 1) * cohort_size, len(results))
            chunk = results[start:end]
            print(
                f"{start}-{end:<8} "
                f"{np.mean([r['prob_efficacy'] for r in chunk])*100:>8.2f} "
                f"{np.mean([r['argmax_efficacy'] for r in chunk])*100:>8.2f} "
                f"{np.mean([r['prob_generalization'] for r in chunk])*100:>8.2f} "
                f"{np.mean([r['argmax_generalization'] for r in chunk])*100:>8.2f} "
                f"{np.mean([r['prob_specificity'] for r in chunk])*100:>8.2f} "
                f"{np.mean([r['argmax_specificity'] for r in chunk])*100:>8.2f} "
            )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    if args.output:
        summary = {
            "config": {
                "checkpoint_dir": args.checkpoint_dir,
                "checkpoint_format": args.checkpoint_format,
                "base_only": args.base_only,
                "model_name": args.model_name,
                "edits": args.edits,
                "n_evaluated": len(results),
                "label": label,
            },
            "probability_preference": {k: round(v, 2) for k, v in prob_metrics.items()},
            "argmax": {k: round(v, 2) for k, v in argmax_metrics.items()},
            "probability_difference": {k: round(v, 2) for k, v in prob_diff_metrics.items()},
            "metric_gap_pp": {
                k: round(prob_metrics[k] - argmax_metrics[k], 2)
                for k in prob_metrics
            },
            "per_case": [
                {
                    "case_id": r["case_id"],
                    "prob_efficacy": r["prob_efficacy"],
                    "prob_generalization": r["prob_generalization"],
                    "prob_specificity": r["prob_specificity"],
                    "argmax_efficacy": r["argmax_efficacy"],
                    "argmax_generalization": r["argmax_generalization"],
                    "argmax_specificity": r["argmax_specificity"],
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved: {args.output}")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
