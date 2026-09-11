#!/usr/bin/env python3
"""
Anchor evaluation: Compare probability-preference vs argmax metrics on EvoEdit checkpoints.

Evaluates an EvoEdit checkpoint using BOTH metrics:
  1. Probability-preference (EvoEdit paper metric): P(target_new) > P(target_true)
  2. Argmax (our pipeline metric): every target token must be the argmax prediction

Published EvoEdit results (10K edits, BS=100, LLaMA-3-8B, MCF):
  Efficacy: 98.29%   Generalization: 91.21%   Specificity: 63.91%

This script verifies whether probability-preference on our checkpoint reproduces
those numbers, establishing the metric translation factor for all our experiments.

Usage:
  # Quick validation on 2K checkpoint (downloads from S3 if needed)
  python scripts/eval_anchor_evoedit.py --edits 2000

  # Full 10K anchor comparison
  python scripts/eval_anchor_evoedit.py --edits 10000

  # Custom checkpoint path
  python scripts/eval_anchor_evoedit.py --checkpoint_dir /path/to/ckpt --edits 10000

  # Use stream-ordered dataset (matches the exact facts that were edited)
  python scripts/eval_anchor_evoedit.py --edits 10000 --stream_path results/matched_ordering/orderings/fb_random0_seed42.json
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

def resolve_model_path(model_name: str) -> str:
    """Resolve model name to a loadable path."""
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_resolve import resolve_model_path as _resolve
    return _resolve(model_name)


def load_model_from_checkpoint(
    model_name: str,
    ckpt_path: Path,
    dtype=torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and apply lightweight layer-weight checkpoint."""
    token = os.environ.get("HF_TOKEN")
    model_path = resolve_model_path(model_name)
    print(f"Loading base model: {model_path} ({dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, token=token,
    ).cuda()
    tok = AutoTokenizer.from_pretrained(model_path, token=token)
    tok.pad_token = tok.eos_token

    weights_file = ckpt_path / "model_weights.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"No model_weights.pt at {weights_file}")

    print(f"Applying checkpoint weights from: {ckpt_path}")
    weights = torch.load(str(weights_file), map_location="cuda", weights_only=True)
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.to(dtype).cuda())
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    print(f"  Applied {loaded} weight tensors")

    return model, tok


# ---------------------------------------------------------------------------
# Dual-metric evaluation (probability-preference + argmax)
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
    Evaluate prompts using the EXACT same logic as EvoEdit's eval_utils_counterfact.py.

    Returns:
      probs: list of {"target_new": nll_new, "target_true": nll_true} per prompt
      targets_correct: list of bool (argmax correctness) per correct-side prompt
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

        # Compute per-token NLL (same as EvoEdit)
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute argmax correctness (same as EvoEdit)
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
    """
    Evaluate one MCF record with BOTH metrics. Returns per-case results
    containing both raw NLL probabilities and argmax correctness.
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

    # Probability-preference metric (EvoEdit paper):
    #   rewrite/paraphrase: target_true_nll > target_new_nll (new is more likely)
    #   neighborhood: target_true_nll < target_new_nll (true is more likely)
    prob_eff = np.mean([x["target_true"] > x["target_new"] for x in ret_probs[0]])
    prob_gen = np.mean([x["target_true"] > x["target_new"] for x in ret_probs[1]])
    prob_spec = np.mean([x["target_true"] < x["target_new"] for x in ret_probs[2]])

    # Argmax metric (our pipeline):
    argmax_eff = float(np.mean(ret_corrects[0]))
    argmax_gen = float(np.mean(ret_corrects[1]))
    argmax_spec = float(np.mean(ret_corrects[2]))

    return {
        "case_id": record["case_id"],
        # Probability-preference
        "prob_efficacy": float(prob_eff),
        "prob_generalization": float(prob_gen),
        "prob_specificity": float(prob_spec),
        # Argmax
        "argmax_efficacy": argmax_eff,
        "argmax_generalization": argmax_gen,
        "argmax_specificity": argmax_spec,
        # Raw data for debugging
        "rewrite_probs": ret_probs[0],
        "paraphrase_probs": ret_probs[1],
        "neighborhood_probs": ret_probs[2],
    }


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_checkpoint(edits: int, seed: int, ordering: str, result_root: Path) -> Path:
    """Resolve checkpoint path from result_root (local or S3 FUSE mount).

    On SkyPilot clusters, RESULT_ROOT points to the S3 FUSE mount at
    /s3-data/continual-learning/alphaedit/results — no aws CLI needed.
    """
    candidates = [
        result_root / "evoedit" / ordering / f"seed{seed}"
        / "10000edits" / "EvoEdit" / "run_000" / "checkpoints" / f"edits_{edits:06d}",
        result_root / "evoedit" / ordering / f"seed{seed}"
        / f"{edits}edits" / "EvoEdit" / "run_000" / "checkpoints" / f"edits_{edits:06d}",
    ]
    for ckpt_path in candidates:
        if (ckpt_path / "model_weights.pt").exists():
            return ckpt_path
    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Anchor evaluation: probability-preference vs argmax metrics on EvoEdit"
    )
    parser.add_argument("--edits", type=int, default=10000,
                        help="Number of edits (checkpoint to evaluate). Default: 10000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ordering", type=str, default="fb_random0",
                        help="Ordering used in the EvoEdit run. Default: fb_random0")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Explicit checkpoint directory (overrides S3 download)")
    parser.add_argument("--model_name", type=str,
                        default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--stream_path", type=str, default=None,
                        help="Path to ordering stream JSON (for exact fact matching)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to multi_counterfact.json (auto-detected if omitted)")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Evaluate at most this many records (for quick testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save detailed results JSON")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve checkpoint
    # ------------------------------------------------------------------
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir).expanduser()
    else:
        result_root = Path(os.environ.get("RESULT_ROOT", str(PROJECT_ROOT / "results")))
        ckpt_dir = resolve_checkpoint(args.edits, args.seed, args.ordering, result_root)

    if not (ckpt_dir / "model_weights.pt").exists():
        print(f"ERROR: No checkpoint at {ckpt_dir}")
        sys.exit(1)
    print(f"Checkpoint: {ckpt_dir}")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    if args.stream_path:
        ds_path = Path(args.stream_path)
        print(f"Using stream-ordered dataset: {ds_path}")
        with open(ds_path) as f:
            all_records = json.load(f)
    else:
        # Auto-detect MCF dataset
        candidates = [
            Path(args.dataset_path) if args.dataset_path else None,
            PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
        ]
        ds_path = None
        for c in candidates:
            if c and c.exists():
                ds_path = c
                break
        if ds_path is None:
            print("ERROR: Cannot find multi_counterfact.json. Specify with --dataset_path")
            sys.exit(1)
        print(f"Using default MCF dataset: {ds_path}")
        with open(ds_path) as f:
            all_records = json.load(f)

    # Select records to evaluate (first N edits)
    n_eval = min(args.edits, len(all_records))
    if args.max_records:
        n_eval = min(n_eval, args.max_records)
    records = all_records[:n_eval]
    print(f"Evaluating {n_eval} records")

    # ------------------------------------------------------------------
    # 3. Load model + checkpoint
    # ------------------------------------------------------------------
    model, tok = load_model_from_checkpoint(args.model_name, ckpt_dir)

    # ------------------------------------------------------------------
    # 4. Evaluate all records with both metrics
    # ------------------------------------------------------------------
    print(f"\nRunning dual-metric evaluation on {len(records)} records...")
    results = []
    for i, record in enumerate(records):
        result = evaluate_record_dual(model, tok, record)
        results.append(result)
        if (i + 1) % 200 == 0:
            pe = np.mean([r["prob_efficacy"] for r in results])
            ae = np.mean([r["argmax_efficacy"] for r in results])
            print(f"  [{i+1}/{len(records)}] prob_eff={pe:.4f}  argmax_eff={ae:.4f}")

    # ------------------------------------------------------------------
    # 5. Aggregate and compare
    # ------------------------------------------------------------------
    prob_eff = np.mean([r["prob_efficacy"] for r in results]) * 100
    prob_gen = np.mean([r["prob_generalization"] for r in results]) * 100
    prob_spec = np.mean([r["prob_specificity"] for r in results]) * 100

    argmax_eff = np.mean([r["argmax_efficacy"] for r in results]) * 100
    argmax_gen = np.mean([r["argmax_generalization"] for r in results]) * 100
    argmax_spec = np.mean([r["argmax_specificity"] for r in results]) * 100

    # Published EvoEdit numbers (10K edits, BS=100, LLaMA-3-8B, MCF)
    pub_eff, pub_gen, pub_spec = 98.29, 91.21, 63.91

    print(f"\n{'='*75}")
    print(f"EvoEdit Anchor Comparison @ {args.edits} edits (seed {args.seed}, {args.ordering})")
    print(f"{'='*75}")
    print(f"{'Metric':<22} {'Published':>10} {'Prob-pref':>10} {'Argmax':>10} {'Gap(pub)':>10}")
    print(f"{'-'*75}")
    print(f"{'Efficacy':<22} {pub_eff:>10.2f} {prob_eff:>10.2f} {argmax_eff:>10.2f} {prob_eff - pub_eff:>+10.2f}")
    print(f"{'Generalization':<22} {pub_gen:>10.2f} {prob_gen:>10.2f} {argmax_gen:>10.2f} {prob_gen - pub_gen:>+10.2f}")
    print(f"{'Specificity':<22} {pub_spec:>10.2f} {prob_spec:>10.2f} {argmax_spec:>10.2f} {prob_spec - pub_spec:>+10.2f}")
    print(f"{'-'*75}")
    print(f"{'Metric gap (prob - argmax):'}")
    print(f"  Efficacy:       {prob_eff - argmax_eff:+.2f} pp")
    print(f"  Generalization: {prob_gen - argmax_gen:+.2f} pp")
    print(f"  Specificity:    {prob_spec - argmax_spec:+.2f} pp")

    # Verdict
    print(f"\n{'='*75}")
    eff_close = abs(prob_eff - pub_eff) < 3.0
    gen_close = abs(prob_gen - pub_gen) < 5.0
    spec_close = abs(prob_spec - pub_spec) < 5.0
    if eff_close and gen_close and spec_close:
        print("VERDICT: Probability-preference metric REPRODUCES published results")
        print("         (all within tolerance: eff<3pp, gen<5pp, spec<5pp)")
    else:
        print("VERDICT: Numbers DIFFER from published results")
        if not eff_close:
            print(f"         Efficacy gap: {abs(prob_eff - pub_eff):.2f}pp (>3pp threshold)")
        if not gen_close:
            print(f"         Generalization gap: {abs(prob_gen - pub_gen):.2f}pp (>5pp threshold)")
        if not spec_close:
            print(f"         Specificity gap: {abs(prob_spec - pub_spec):.2f}pp (>5pp threshold)")
        print("         Note: differences may be due to ordering (fb_random0 vs default MCF)")
        print("         or model variant (NousResearch vs meta-llama)")
    print(f"{'='*75}")

    # ------------------------------------------------------------------
    # 6. Save detailed results
    # ------------------------------------------------------------------
    summary = {
        "config": {
            "edits": args.edits,
            "seed": args.seed,
            "ordering": args.ordering,
            "model_name": args.model_name,
            "checkpoint_dir": str(ckpt_dir),
            "n_evaluated": len(results),
        },
        "published": {
            "efficacy": pub_eff,
            "generalization": pub_gen,
            "specificity": pub_spec,
        },
        "probability_preference": {
            "efficacy": round(prob_eff, 2),
            "generalization": round(prob_gen, 2),
            "specificity": round(prob_spec, 2),
        },
        "argmax": {
            "efficacy": round(argmax_eff, 2),
            "generalization": round(argmax_gen, 2),
            "specificity": round(argmax_spec, 2),
        },
        "metric_gap_pp": {
            "efficacy": round(prob_eff - argmax_eff, 2),
            "generalization": round(prob_gen - argmax_gen, 2),
            "specificity": round(prob_spec - argmax_spec, 2),
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

    out_path = args.output
    if not out_path:
        out_dir = PROJECT_ROOT / "results" / "anchor_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"evoedit_{args.ordering}_s{args.seed}_{args.edits}edits.json")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved: {out_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
