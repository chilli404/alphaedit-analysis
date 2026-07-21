#!/usr/bin/env python3
"""
Per-record re-evaluation of coupling checkpoints with multiple scoring methods.

Evaluates existing checkpoints (AlphaEdit and MEMIT-seq) and reports:
  1. Raw efficacy (all records weighted equally)
  2. Deduplicated efficacy (each unique (subject, target) scored once, last occurrence)
  3. Nonconflicting-only efficacy (exclude subjects with multiple targets)
  4. Per-subject efficacy (weight each subject equally)
  5. Latest-target efficacy (for conflicting subjects, score against latest target only)

Usage:
    uv run python scripts/rescore_coupling_perrecord.py \
        --checkpoint_dir ~/.cache/memit_seqreg_checkpoints/coupling_low_seed42 \
        --dataset_path results/controlled_coupling/low_coupling_seed42.json \
        --batch 49 --method memit_seq --stream low

    uv run python scripts/rescore_coupling_perrecord.py \
        --checkpoint_dir results/controlled_coupling/checkpoints/low_coupling/seed42 \
        --dataset_path results/controlled_coupling/low_coupling_seed42.json \
        --batch 49 --method alphaedit --stream low
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Model Loading ────────────────────────────────────────────────────────────


def load_model_and_apply_checkpoint(model_name: str, ckpt_path: Path):
    """Load base model in fp16, then apply checkpoint weights."""
    print(f"  Loading base model: {model_name} (float16)")
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    weights_file = ckpt_path / "model_weights.pt"
    if not weights_file.exists():
        print(f"  ERROR: No model_weights.pt at {ckpt_path}")
        sys.exit(1)

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


# ─── Evaluation Protocol (full multi-token, vendor-compatible) ────────────────


def test_batch_prediction(model, tok, prefixes, which_correct, target_new_str, target_true_str):
    """Multi-token argmax evaluation (same protocol as vendor evaluate.py)."""
    prefix_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prefixes]

    # Prepare interleaved prompts: target_new and target_true for each prefix
    prompts = []
    for p in prefixes:
        prompts.append(p + " " + target_new_str)
        prompts.append(p + " " + target_true_str)

    prompt_tok = tok(prompts, padding=True, return_tensors="pt").to("cuda")

    a_tok = tok(" " + target_new_str, add_special_tokens=False).input_ids
    b_tok = tok(" " + target_true_str, add_special_tokens=False).input_ids
    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if "llama" in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    targets_correct = []
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        cur_toks = a_tok if i % 2 == 0 else b_tok

        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_toks[j]:
                    correct = False
                    break
            targets_correct.append(correct)

    return targets_correct


def evaluate_record(model, tok, record: dict) -> dict:
    """Evaluate a single MCF record with full multi-token protocol."""
    subject = record["requested_rewrite"]["subject"]
    target_new = record["requested_rewrite"]["target_new"]
    target_true = record["requested_rewrite"]["target_true"]

    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [rewrite_prompts, paraphrase_prompts, neighborhood_prompts]
    which_correct = [
        [0] * len(rewrite_prompts),
        [0] * len(paraphrase_prompts),
        [1] * len(neighborhood_prompts),
    ]

    targets_correct = test_batch_prediction(
        model, tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )

    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    return {
        "case_id": record["case_id"],
        "subject": subject,
        "target_new": target_new["str"],
        "target_true": target_true["str"],
        "efficacy": float(np.mean(ret_corrects[0])) if ret_corrects[0] else 0.0,
        "paraphrase": float(np.mean(ret_corrects[1])) if ret_corrects[1] else 0.0,
        "neighborhood": float(np.mean(ret_corrects[2])) if ret_corrects[2] else 0.0,
    }


# ─── Aggregation Methods ─────────────────────────────────────────────────────


def compute_aggregates(per_record_results: list, records: list) -> dict:
    """Compute 5 aggregation methods from per-record evaluation results."""
    n = len(per_record_results)

    # Build index structures
    subjects = [r["requested_rewrite"]["subject"] for r in records]
    targets = [r["requested_rewrite"]["target_new"]["str"] for r in records]

    # Map subject -> list of (index, target)
    subject_entries = defaultdict(list)
    for i, (s, t) in enumerate(zip(subjects, targets)):
        subject_entries[s].append((i, t))

    # Identify conflicting subjects (multiple distinct targets)
    conflicting_subjects = set()
    for s, entries in subject_entries.items():
        if len(set(t for _, t in entries)) > 1:
            conflicting_subjects.add(s)

    # Identify duplicates: (subject, target) seen more than once
    pair_indices = defaultdict(list)  # (subject, target) -> [indices]
    for i, (s, t) in enumerate(zip(subjects, targets)):
        pair_indices[(s, t)].append(i)

    # ─── 1. Raw efficacy ───
    raw_eff = np.mean([r["efficacy"] for r in per_record_results])
    raw_para = np.mean([r["paraphrase"] for r in per_record_results])
    raw_neigh = np.mean([r["neighborhood"] for r in per_record_results])

    # ─── 2. Deduplicated efficacy (last occurrence of each unique pair) ───
    dedup_indices = set()
    for pair, indices in pair_indices.items():
        dedup_indices.add(indices[-1])  # keep last occurrence
    dedup_results = [per_record_results[i] for i in sorted(dedup_indices)]
    dedup_eff = np.mean([r["efficacy"] for r in dedup_results])
    dedup_para = np.mean([r["paraphrase"] for r in dedup_results])
    dedup_neigh = np.mean([r["neighborhood"] for r in dedup_results])

    # ─── 3. Nonconflicting-only efficacy ───
    nonconflict_results = [
        per_record_results[i] for i in range(n)
        if subjects[i] not in conflicting_subjects
    ]
    if nonconflict_results:
        nonconflict_eff = np.mean([r["efficacy"] for r in nonconflict_results])
        nonconflict_para = np.mean([r["paraphrase"] for r in nonconflict_results])
        nonconflict_neigh = np.mean([r["neighborhood"] for r in nonconflict_results])
    else:
        nonconflict_eff = nonconflict_para = nonconflict_neigh = float("nan")

    # ─── 4. Per-subject efficacy (weight each subject equally) ───
    subject_effs = defaultdict(list)
    for i, r in enumerate(per_record_results):
        subject_effs[subjects[i]].append(r["efficacy"])
    per_subj_eff = np.mean([np.mean(v) for v in subject_effs.values()])
    subject_paras = defaultdict(list)
    for i, r in enumerate(per_record_results):
        subject_paras[subjects[i]].append(r["paraphrase"])
    per_subj_para = np.mean([np.mean(v) for v in subject_paras.values()])

    # ─── 5. Latest-target efficacy ───
    # For conflicting subjects, only score the record whose target matches the
    # LAST target assigned to that subject. For non-conflicting, include all.
    latest_target_for_subject = {}
    for i, (s, t) in enumerate(zip(subjects, targets)):
        latest_target_for_subject[s] = t  # last write wins

    latest_target_results = []
    for i, r in enumerate(per_record_results):
        s = subjects[i]
        t = targets[i]
        if s in conflicting_subjects:
            # Only include if this record's target matches the latest
            if t == latest_target_for_subject[s]:
                latest_target_results.append(r)
        else:
            latest_target_results.append(r)

    if latest_target_results:
        latest_eff = np.mean([r["efficacy"] for r in latest_target_results])
        latest_para = np.mean([r["paraphrase"] for r in latest_target_results])
        latest_neigh = np.mean([r["neighborhood"] for r in latest_target_results])
    else:
        latest_eff = latest_para = latest_neigh = float("nan")

    return {
        "n_total": n,
        "n_unique_pairs": len(pair_indices),
        "n_duplicates": n - len(pair_indices),
        "n_conflicting_subjects": len(conflicting_subjects),
        "n_nonconflicting_records": len(nonconflict_results),
        "n_latest_target_records": len(latest_target_results),
        "raw": {"efficacy": float(raw_eff), "paraphrase": float(raw_para), "neighborhood": float(raw_neigh)},
        "deduplicated": {"efficacy": float(dedup_eff), "paraphrase": float(dedup_para), "neighborhood": float(dedup_neigh), "n_records": len(dedup_results)},
        "nonconflicting": {"efficacy": float(nonconflict_eff), "paraphrase": float(nonconflict_para), "neighborhood": float(nonconflict_neigh), "n_records": len(nonconflict_results)},
        "per_subject": {"efficacy": float(per_subj_eff), "paraphrase": float(per_subj_para), "n_subjects": len(subject_effs)},
        "latest_target": {"efficacy": float(latest_eff), "paraphrase": float(latest_para), "neighborhood": float(latest_neigh), "n_records": len(latest_target_results)},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Per-record re-evaluation with multiple scoring methods")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch", type=int, required=True, help="Batch index (e.g., 49 for 5K edits)")
    parser.add_argument("--method", type=str, required=True, choices=["memit_seq", "alphaedit"])
    parser.add_argument("--stream", type=str, required=True, choices=["low", "high"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num_edits", type=int, default=100)
    args = parser.parse_args()

    # Resolve model
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from model_download import resolve_model_path
        model_name = resolve_model_path(args.model_name)
    except ImportError:
        model_name = args.model_name

    # Load dataset
    ds_path = Path(args.dataset_path)
    if not ds_path.is_absolute():
        ds_path = PROJECT_ROOT / ds_path
    with open(ds_path) as f:
        all_records = json.load(f)

    total_edits = (args.batch + 1) * args.num_edits
    records = all_records[:total_edits]
    print(f"  Dataset: {ds_path}")
    print(f"  Records to evaluate: {len(records)}")

    # Load checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path.home() / args.checkpoint_dir.lstrip("~/")
    ckpt_path = ckpt_dir / f"batch_{args.batch}"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, tok = load_model_and_apply_checkpoint(model_name, ckpt_path)

    # Evaluate all records
    print(f"\n  Evaluating {len(records)} records (full multi-token protocol)...")
    per_record_results = []
    for i, record in enumerate(records):
        result = evaluate_record(model, tok, record)
        per_record_results.append(result)
        if (i + 1) % 500 == 0:
            eff_so_far = np.mean([r["efficacy"] for r in per_record_results])
            print(f"    [{i+1}/{len(records)}] efficacy={eff_so_far:.4f}")

    # Compute aggregates
    aggregates = compute_aggregates(per_record_results, records)

    # Save per-record results
    out_dir = PROJECT_ROOT / "results" / "coupling_rescore"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_record_path = out_dir / f"perrecord_{args.method}_{args.stream}_seed42_batch{args.batch}.jsonl"
    with open(per_record_path, "w") as f:
        for r in per_record_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Per-record results: {per_record_path}")

    aggregate_path = out_dir / f"aggregates_{args.method}_{args.stream}_seed42_batch{args.batch}.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregates, f, indent=2)
    print(f"  Aggregates: {aggregate_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Re-scoring: {args.method} / {args.stream} coupling / {total_edits} edits")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'Efficacy':>9} {'Paraphrase':>11} {'Neighborhood':>13} {'N':>6}")
    print(f"  {'-'*65}")
    print(f"  {'Raw':<22} {aggregates['raw']['efficacy']:>9.4f} {aggregates['raw']['paraphrase']:>11.4f} {aggregates['raw']['neighborhood']:>13.4f} {aggregates['n_total']:>6}")
    print(f"  {'Deduplicated':<22} {aggregates['deduplicated']['efficacy']:>9.4f} {aggregates['deduplicated']['paraphrase']:>11.4f} {aggregates['deduplicated']['neighborhood']:>13.4f} {aggregates['deduplicated']['n_records']:>6}")
    print(f"  {'Nonconflicting':<22} {aggregates['nonconflicting']['efficacy']:>9.4f} {aggregates['nonconflicting']['paraphrase']:>11.4f} {aggregates['nonconflicting']['neighborhood']:>13.4f} {aggregates['nonconflicting']['n_records']:>6}")
    print(f"  {'Per-subject':<22} {aggregates['per_subject']['efficacy']:>9.4f} {aggregates['per_subject']['paraphrase']:>11.4f} {'—':>13} {aggregates['per_subject']['n_subjects']:>6}")
    print(f"  {'Latest-target':<22} {aggregates['latest_target']['efficacy']:>9.4f} {aggregates['latest_target']['paraphrase']:>11.4f} {aggregates['latest_target']['neighborhood']:>13.4f} {aggregates['latest_target']['n_records']:>6}")
    print(f"\n  Structural: {aggregates['n_duplicates']} duplicates, {aggregates['n_conflicting_subjects']} conflicting subjects")

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
