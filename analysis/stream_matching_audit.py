#!/usr/bin/env python3
"""
Stream-Matching Audit (Step 5): Verify that low- and high-coupling streams
have comparable intrinsic difficulty before editing.

Compares:
  - Base-model target probability (is new target already likely?)
  - Target probability margin (how far from correct answer?)
  - Prompt length distribution
  - Target token length distribution
  - Relation distribution (are certain relations over-represented?)
  - Immediate first-batch efficacy (from behavioral eval)

The behavioral pattern (recent edits retained, old edits forgotten) already
argues against a difficulty explanation, but this table prevents reviewers
from claiming high-coupling simply contains harder examples.

Usage:
    uv run python analysis/stream_matching_audit.py
    uv run python analysis/stream_matching_audit.py --with_model  # adds base-model logit analysis (GPU)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results" / "controlled_coupling"
OUTPUT = PROJECT_ROOT / "results" / "figures" / "paper"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_stream(name: str, seed: int):
    path = RESULTS / f"{name}_seed{seed}.json"
    if not path.exists():
        print(f"  ERROR: {path} not found")
        print(f"  Pull from remote: scp chilli@remote-rig:~/Projects/alphaedit-analysis/results/controlled_coupling/{name}_seed{seed}.json results/controlled_coupling/")
        return None
    with open(path) as f:
        return json.load(f)


def analyze_stream_properties(records, name):
    """Compute difficulty/distribution metrics for a stream."""
    props = {"name": name, "n_records": len(records)}

    # Prompt lengths (in characters)
    prompts = [r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]) for r in records]
    props["prompt_len_mean"] = np.mean([len(p) for p in prompts])
    props["prompt_len_std"] = np.std([len(p) for p in prompts])
    props["prompt_len_median"] = np.median([len(p) for p in prompts])

    # Subject lengths
    subjects = [r["requested_rewrite"]["subject"] for r in records]
    props["subject_len_mean"] = np.mean([len(s) for s in subjects])
    props["subject_len_std"] = np.std([len(s) for s in subjects])

    # Target token lengths (new target)
    targets_new = [r["requested_rewrite"]["target_new"]["str"] for r in records]
    props["target_new_len_mean"] = np.mean([len(t) for t in targets_new])
    props["target_new_len_std"] = np.std([len(t) for t in targets_new])

    # Target true lengths
    targets_true = [r["requested_rewrite"]["target_true"]["str"] for r in records]
    props["target_true_len_mean"] = np.mean([len(t) for t in targets_true])
    props["target_true_len_std"] = np.std([len(t) for t in targets_true])

    # Relation distribution
    relations = [r["requested_rewrite"].get("relation_id", "unknown") for r in records]
    rel_counts = Counter(relations)
    props["n_unique_relations"] = len(rel_counts)
    props["top5_relations"] = rel_counts.most_common(5)
    props["relation_entropy"] = -sum(
        (c / len(relations)) * np.log(c / len(relations))
        for c in rel_counts.values()
    )

    # Subject uniqueness (key coupling metric)
    subject_counts = Counter(subjects)
    props["n_unique_subjects"] = len(subject_counts)
    props["subject_reuse_rate"] = 1.0 - len(subject_counts) / len(subjects)
    props["max_subject_repeats"] = max(subject_counts.values())
    props["mean_subject_repeats"] = np.mean(list(subject_counts.values()))

    # Per-batch subject overlap (within 100-edit windows)
    batch_size = 100
    intra_batch_overlaps = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_subjects = [r["requested_rewrite"]["subject"] for r in batch]
        n_unique = len(set(batch_subjects))
        overlap = 1.0 - n_unique / len(batch_subjects)
        intra_batch_overlaps.append(overlap)
    props["mean_intra_batch_overlap"] = np.mean(intra_batch_overlaps)
    props["max_intra_batch_overlap"] = max(intra_batch_overlaps)

    # Case ID distribution (are they from similar parts of the dataset?)
    case_ids = [r["case_id"] for r in records]
    props["case_id_min"] = min(case_ids)
    props["case_id_max"] = max(case_ids)
    props["case_id_mean"] = np.mean(case_ids)
    props["case_id_std"] = np.std(case_ids)

    return props


def analyze_base_model_difficulty(records, model_name, name):
    """GPU analysis: compute base-model probabilities for targets."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import resolve_model_path
    model_name = resolve_model_path(model_name)

    print(f"  Loading model for base-difficulty analysis...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model.eval()

    target_new_probs = []
    target_true_probs = []
    margins = []

    # Sample for speed (every 10th fact)
    sample_indices = list(range(0, len(records), 10))

    for idx in sample_indices:
        record = records[idx]
        rw = record["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        target_new = rw["target_new"]["str"]
        target_true = rw["target_true"]["str"]

        input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        new_ids = tok(f" {target_new}", add_special_tokens=False).input_ids
        true_ids = tok(f" {target_true}", add_special_tokens=False).input_ids

        new_prob = probs[new_ids[0]].item()
        true_prob = probs[true_ids[0]].item()

        target_new_probs.append(new_prob)
        target_true_probs.append(true_prob)
        margins.append(true_prob - new_prob)

    del model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "n_sampled": len(sample_indices),
        "base_target_new_prob_mean": float(np.mean(target_new_probs)),
        "base_target_new_prob_std": float(np.std(target_new_probs)),
        "base_target_true_prob_mean": float(np.mean(target_true_probs)),
        "base_target_true_prob_std": float(np.std(target_true_probs)),
        "base_margin_mean": float(np.mean(margins)),
        "base_margin_std": float(np.std(margins)),
        "base_target_new_prob_median": float(np.median(target_new_probs)),
        "base_margin_median": float(np.median(margins)),
    }


def main():
    parser = argparse.ArgumentParser(description="Stream-matching audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_model", action="store_true",
                        help="Include base-model target probability analysis (requires GPU)")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    print("=" * 70)
    print("Stream-Matching Audit: Low vs High Coupling")
    print("=" * 70)

    low = load_stream("low_coupling", args.seed)
    high = load_stream("high_coupling", args.seed)

    if not low or not high:
        sys.exit(1)

    # Structural analysis (no GPU needed)
    print("\nAnalyzing stream structure...")
    low_props = analyze_stream_properties(low, "low_coupling")
    high_props = analyze_stream_properties(high, "high_coupling")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<35} {'Low Coupling':>15} {'High Coupling':>15}")
    print(f"{'-' * 70}")

    comparison_keys = [
        ("n_records", "N records"),
        ("prompt_len_mean", "Prompt length (mean)"),
        ("prompt_len_std", "Prompt length (std)"),
        ("subject_len_mean", "Subject length (mean)"),
        ("target_new_len_mean", "Target new length (mean)"),
        ("target_true_len_mean", "Target true length (mean)"),
        ("n_unique_relations", "Unique relations"),
        ("relation_entropy", "Relation entropy"),
        ("n_unique_subjects", "Unique subjects"),
        ("subject_reuse_rate", "Subject reuse rate"),
        ("max_subject_repeats", "Max subject repeats"),
        ("mean_subject_repeats", "Mean subject repeats"),
        ("mean_intra_batch_overlap", "Mean intra-batch overlap"),
        ("max_intra_batch_overlap", "Max intra-batch overlap"),
        ("case_id_mean", "Case ID mean"),
        ("case_id_std", "Case ID std"),
    ]

    for key, label in comparison_keys:
        lv = low_props.get(key, "?")
        hv = high_props.get(key, "?")
        if isinstance(lv, float):
            print(f"  {label:<33} {lv:>15.4f} {hv:>15.4f}")
        else:
            print(f"  {label:<33} {str(lv):>15} {str(hv):>15}")

    # Coupling-specific metrics
    print(f"\n{'=' * 70}")
    print("Coupling Structure Verification:")
    print(f"  Low: {low_props['n_unique_subjects']} unique subjects in {low_props['n_records']} edits "
          f"(reuse={low_props['subject_reuse_rate']:.1%})")
    print(f"  High: {high_props['n_unique_subjects']} unique subjects in {high_props['n_records']} edits "
          f"(reuse={high_props['subject_reuse_rate']:.1%})")
    print(f"  Intra-batch overlap: Low={low_props['mean_intra_batch_overlap']:.3f}, "
          f"High={high_props['mean_intra_batch_overlap']:.3f}")

    # Check for difficulty confounds
    print(f"\n{'=' * 70}")
    print("Difficulty Confound Check:")
    prompt_diff = abs(low_props["prompt_len_mean"] - high_props["prompt_len_mean"])
    target_diff = abs(low_props["target_new_len_mean"] - high_props["target_new_len_mean"])
    print(f"  Prompt length difference: {prompt_diff:.1f} chars "
          f"({'OK' if prompt_diff < 5 else 'WARN'})")
    print(f"  Target length difference: {target_diff:.1f} chars "
          f"({'OK' if target_diff < 3 else 'WARN'})")
    print(f"  Relation entropy difference: "
          f"{abs(low_props['relation_entropy'] - high_props['relation_entropy']):.3f} "
          f"({'OK' if abs(low_props['relation_entropy'] - high_props['relation_entropy']) < 0.5 else 'WARN'})")

    # Base model analysis (GPU)
    if args.with_model:
        print(f"\n{'=' * 70}")
        print("Base-Model Difficulty Analysis (GPU)...")
        low_diff = analyze_base_model_difficulty(low, args.model_name, "low_coupling")
        high_diff = analyze_base_model_difficulty(high, args.model_name, "high_coupling")

        print(f"\n{'Metric':<35} {'Low Coupling':>15} {'High Coupling':>15}")
        print(f"{'-' * 70}")
        for key in ["base_target_new_prob_mean", "base_target_true_prob_mean",
                    "base_margin_mean", "base_margin_median"]:
            label = key.replace("base_", "").replace("_", " ").title()
            print(f"  {label:<33} {low_diff[key]:>15.6f} {high_diff[key]:>15.6f}")

        # Save
        audit_results = {
            "low_structure": low_props,
            "high_structure": high_props,
            "low_difficulty": low_diff,
            "high_difficulty": high_diff,
        }
    else:
        audit_results = {
            "low_structure": low_props,
            "high_structure": high_props,
        }

    # Save results
    out_path = OUTPUT / f"stream_matching_audit_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
