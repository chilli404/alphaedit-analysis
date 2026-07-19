#!/usr/bin/env python3
"""
Evaluate a completed controlled coupling checkpoint.

Loads the final model checkpoint and evaluates behavioral metrics
(efficacy, cohort retention, etc.) WITHOUT re-running the edit loop.

Usage:
    python scripts/eval_controlled_coupling.py --stream low_coupling --seed 42
    python scripts/eval_controlled_coupling.py --stream high_coupling --seed 42
    python scripts/eval_controlled_coupling.py --stream both --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


def find_checkpoint(ckpt_base: Path, stream_name: str, seed: int) -> Path | None:
    """Find the latest checkpoint for a stream."""
    ckpt_dir = ckpt_base / stream_name / f"seed{seed}"
    if not ckpt_dir.exists():
        return None
    batch_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not batch_dirs:
        return None
    last = batch_dirs[-1]
    if (last / "model_weights.pt").exists():
        return last
    return None


def load_checkpoint_model(model_name: str, ckpt_dir: Path):
    """Load base model and apply checkpoint weights."""
    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    print(f"  Applying checkpoint weights from: {ckpt_dir}")
    weights = torch.load(ckpt_dir / "model_weights.pt", map_location="cuda")
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.cuda())
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    print(f"  Loaded {loaded} weight tensors")

    return model, tok


def evaluate_efficacy(model, tok, records, batch_size=100):
    """Evaluate edit efficacy: does the model produce the target for each edited fact?"""
    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        for record in batch:
            rw = record["requested_rewrite"]
            prompt = rw["prompt"].format(rw["subject"])
            target_new = rw["target_new"]["str"]
            target_true = rw["target_true"]["str"]

            input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                logits = model(input_ids).logits[0, -1, :]

            # Check if target_new is more likely than target_true
            new_ids = tok(f" {target_new}", add_special_tokens=False).input_ids
            true_ids = tok(f" {target_true}", add_special_tokens=False).input_ids

            # Use first token probability
            new_prob = logits[new_ids[0]].item()
            true_prob = logits[true_ids[0]].item()

            results.append({
                "case_id": record["case_id"],
                "efficacy": new_prob > true_prob,
                "new_logit": new_prob,
                "true_logit": true_prob,
            })

        if (i + batch_size) % 500 == 0:
            done = min(i + batch_size, len(records))
            eff_so_far = np.mean([r["efficacy"] for r in results])
            print(f"    Evaluated {done}/{len(records)} facts, efficacy={eff_so_far:.4f}")

    return results


def compute_cohort_retention(eval_results, num_edits=100):
    """Compute retention by edit cohort (which batch a fact was in)."""
    # Group by cohort (batch index)
    cohorts = {}
    for r in eval_results:
        cohort = r["case_id"] // num_edits
        if cohort not in cohorts:
            cohorts[cohort] = []
        cohorts[cohort].append(r["efficacy"])

    retention_by_cohort = {}
    for cohort_idx in sorted(cohorts.keys()):
        retention_by_cohort[cohort_idx] = {
            "cohort": cohort_idx,
            "edits_range": f"{cohort_idx * num_edits}-{(cohort_idx + 1) * num_edits}",
            "efficacy": float(np.mean(cohorts[cohort_idx])),
            "n_facts": len(cohorts[cohort_idx]),
        }

    return retention_by_cohort


def evaluate_stream(model_name: str, ckpt_dir: Path, stream_path: Path, stream_name: str, seed: int):
    """Full behavioral evaluation for one stream."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {stream_name} (seed {seed})")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Stream:     {stream_path}")
    print(f"{'=' * 60}")

    # Load model with checkpoint
    model, tok = load_checkpoint_model(model_name, ckpt_dir)

    # Load stream data
    with open(stream_path) as f:
        records = json.load(f)
    print(f"  Stream has {len(records)} facts")

    # Evaluate efficacy
    print("\n  Evaluating efficacy...")
    eval_results = evaluate_efficacy(model, tok, records)

    overall_efficacy = np.mean([r["efficacy"] for r in eval_results])
    print(f"\n  Overall efficacy: {overall_efficacy:.4f}")

    # Cohort retention
    cohort_retention = compute_cohort_retention(eval_results)
    n_cohorts = len(cohort_retention)

    first_cohort_eff = cohort_retention[0]["efficacy"] if 0 in cohort_retention else None
    last_cohort_eff = cohort_retention[n_cohorts - 1]["efficacy"] if (n_cohorts - 1) in cohort_retention else None

    # First 10 cohorts (1000 edits) and last 10 cohorts
    first_10 = [cohort_retention[c]["efficacy"] for c in range(min(10, n_cohorts))]
    last_10 = [cohort_retention[c]["efficacy"] for c in range(max(0, n_cohorts - 10), n_cohorts)]

    print(f"  First cohort (edits 0-100):     efficacy={first_cohort_eff:.4f}")
    print(f"  Last cohort (edits {(n_cohorts-1)*100}-{n_cohorts*100}): efficacy={last_cohort_eff:.4f}")
    print(f"  First 1K mean:  {np.mean(first_10):.4f}")
    print(f"  Last 1K mean:   {np.mean(last_10):.4f}")

    # Retention AUC (area under the cohort retention curve)
    cohort_effs = [cohort_retention[c]["efficacy"] for c in sorted(cohort_retention.keys())]
    retention_auc = float(np.trapz(cohort_effs) / max(len(cohort_effs) - 1, 1))

    print(f"  Retention AUC:  {retention_auc:.4f}")

    # Compile results
    result = {
        "stream": stream_name,
        "seed": seed,
        "checkpoint": str(ckpt_dir),
        "n_facts": len(records),
        "overall_efficacy": round(overall_efficacy, 4),
        "first_cohort_efficacy": round(first_cohort_eff, 4) if first_cohort_eff else None,
        "last_cohort_efficacy": round(last_cohort_eff, 4) if last_cohort_eff else None,
        "first_1k_mean_efficacy": round(float(np.mean(first_10)), 4),
        "last_1k_mean_efficacy": round(float(np.mean(last_10)), 4),
        "retention_auc": round(retention_auc, 4),
        "cohort_retention": cohort_retention,
        "per_fact_results": eval_results,
    }

    # Cleanup
    del model, tok
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate controlled coupling checkpoint")
    parser.add_argument("--stream", choices=["low_coupling", "high_coupling", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    results_dir = PROJECT_ROOT / "results" / "controlled_coupling"
    ckpt_base = results_dir / "checkpoints"

    streams = []
    if args.stream in ("both", "low_coupling"):
        streams.append("low_coupling")
    if args.stream in ("both", "high_coupling"):
        streams.append("high_coupling")

    all_results = {}
    for stream_name in streams:
        ckpt_dir = find_checkpoint(ckpt_base, stream_name, args.seed)
        if not ckpt_dir:
            print(f"  SKIP {stream_name}: no checkpoint found at {ckpt_base / stream_name / f'seed{args.seed}'}")
            continue

        stream_path = results_dir / f"{stream_name}_seed{args.seed}.json"
        if not stream_path.exists():
            print(f"  SKIP {stream_name}: stream file not found at {stream_path}")
            continue

        result = evaluate_stream(model_name, ckpt_dir, stream_path, stream_name, args.seed)
        all_results[stream_name] = result

    # Save results
    out_path = results_dir / f"behavioral_eval_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary comparison
    if len(all_results) == 2:
        print(f"\n{'=' * 60}")
        print("COMPARISON: Low Coupling vs High Coupling")
        print(f"{'=' * 60}")
        low = all_results["low_coupling"]
        high = all_results["high_coupling"]
        print(f"  {'Metric':<25} {'Low':>8} {'High':>8} {'Δ':>8}")
        print(f"  {'-' * 50}")
        for key in ["overall_efficacy", "first_1k_mean_efficacy", "last_1k_mean_efficacy", "retention_auc"]:
            lv = low[key]
            hv = high[key]
            delta = hv - lv if lv is not None and hv is not None else None
            print(f"  {key:<25} {lv:>8.4f} {hv:>8.4f} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
