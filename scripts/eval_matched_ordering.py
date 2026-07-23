#!/usr/bin/env python3
"""
Post-hoc evaluation of matched ordering checkpoints.

Loads model weights from each saved checkpoint and evaluates ALL edited facts
from the stream file (not just the last batch). Works with any algorithm
(AlphaEdit, MEMIT-Seq, etc.) that saves checkpoints during matched ordering runs.

Produces the decisive comparison table:

    Method    Checkpoint    All facts    First 1K    Latest 1K    Locality

Checkpoints are at batches 9, 19, 29, 39, 49 (= 1K, 2K, 3K, 4K, 5K edits).
We evaluate at 2K (batch 19), 3K (batch 29), and 5K (batch 49).

Usage:
    python scripts/eval_matched_ordering.py --seed 42 --alg_name AlphaEdit --ordering key_clustered
    python scripts/eval_matched_ordering.py --seed 42 --checkpoints 19 29 49
    python scripts/eval_matched_ordering.py --seed 42 --checkpoints 29  # just 3K
"""

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


def resolve_checkpoint_dir(seed: int, lambda_prev: float, lambda_delta: float) -> Path:
    """Resolve checkpoint directory for MEMIT+SeqReg."""
    base = Path.home() / ".cache" / "memit_seqreg_checkpoints"
    return base / f"seed{seed}_lp{lambda_prev}_ld{lambda_delta}"


def load_model_from_checkpoint(model_name: str, ckpt_path: Path):
    """Load base model and apply checkpoint weights."""
    print(f"  Loading base model: {model_name} (float16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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


def test_batch_prediction(
    model,
    tok,
    prefixes: List[str],
    which_correct: List[int],
    target_new: str,
    target_true: str,
) -> tuple:
    """
    Evaluate batch of prompts. Matches vendor protocol exactly.
    which_correct: 0 = target_new is correct, 1 = target_true is correct.
    Returns (probs, targets_correct).
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

    a_tok = tok(f" {target_new}")["input_ids"]
    b_tok = tok(f" {target_true}")["input_ids"]

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


def evaluate_record(model, tok, record: Dict) -> Dict:
    """Evaluate a single MCF record: efficacy, paraphrase, neighborhood."""
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

    probs, targets_correct = test_batch_prediction(
        model,
        tok,
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
        "rewrite_correct": ret_corrects[0],  # list of bool
        "paraphrase_correct": ret_corrects[1],  # list of bool
        "neighborhood_correct": ret_corrects[2],  # list of bool
        "efficacy": float(np.mean(ret_corrects[0])),
        "paraphrase": float(np.mean(ret_corrects[1])),
        "neighborhood": float(np.mean(ret_corrects[2])),
    }


def evaluate_record_fast(model, tok, record: Dict) -> Dict:
    """Fast evaluation: only first-token logit comparison (no multi-token scoring)."""
    rw = record["requested_rewrite"]
    subject = rw["subject"]
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]

    # Gather all prompts
    rewrite_prompts = [rw["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    all_prompts = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
    # which_correct: 0 = target_new correct, 1 = target_true correct
    which_correct = (
        [0] * len(rewrite_prompts)
        + [0] * len(paraphrase_prompts)
        + [1] * len(neighborhood_prompts)
    )

    # Tokenize targets (first token only for speed)
    new_tok_id = tok(f" {target_new}", add_special_tokens=False).input_ids[0]
    true_tok_id = tok(f" {target_true}", add_special_tokens=False).input_ids[0]

    # Batch all prompts
    inputs = tok(all_prompts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get last-token logits for each prompt
    # With left-padding, last token is always at position -1
    last_logits = logits[:, -1, :]  # [n_prompts, vocab_size]
    new_scores = last_logits[:, new_tok_id].cpu().numpy()
    true_scores = last_logits[:, true_tok_id].cpu().numpy()

    # Determine correctness
    corrects = []
    for i, wc in enumerate(which_correct):
        if wc == 0:  # target_new should win
            corrects.append(bool(new_scores[i] > true_scores[i]))
        else:  # target_true should win
            corrects.append(bool(true_scores[i] > new_scores[i]))

    n_rw = len(rewrite_prompts)
    n_para = len(paraphrase_prompts)

    return {
        "case_id": record["case_id"],
        "efficacy": float(np.mean(corrects[:n_rw])),
        "paraphrase": float(np.mean(corrects[n_rw:n_rw + n_para])),
        "neighborhood": float(np.mean(corrects[n_rw + n_para:])),
    }


def evaluate_records_batched(model, tok, records: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Evaluate many records with mega-batching across records.

    Batches prompts from multiple records into single forward passes.
    With 96GB VRAM and fp16 Llama-3-8B (~16GB), we can fit ~50+ records per batch.
    Each record has ~13 prompts, so batch_size=32 → ~416 sequences per forward pass.
    """
    all_results = []

    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start:batch_start + batch_size]

        # Collect all prompts across all records in this batch
        all_prompts = []
        record_metadata = []  # (record_idx, n_rewrite, n_para, n_neigh, new_tok_id, true_tok_id)

        for rec_idx, record in enumerate(batch_records):
            rw = record["requested_rewrite"]
            subject = rw["subject"]
            target_new = rw["target_new"]["str"]
            target_true = rw["target_true"]["str"]

            rewrite_prompts = [rw["prompt"].format(subject)]
            paraphrase_prompts = record["paraphrase_prompts"]
            neighborhood_prompts = record["neighborhood_prompts"]

            prompts = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
            all_prompts.extend(prompts)

            new_tok_id = tok(f" {target_new}", add_special_tokens=False).input_ids[0]
            true_tok_id = tok(f" {target_true}", add_special_tokens=False).input_ids[0]

            record_metadata.append({
                "case_id": record["case_id"],
                "start_idx": len(all_prompts) - len(prompts),
                "n_rewrite": len(rewrite_prompts),
                "n_para": len(paraphrase_prompts),
                "n_neigh": len(neighborhood_prompts),
                "new_tok_id": new_tok_id,
                "true_tok_id": true_tok_id,
            })

        # Single forward pass for entire batch
        inputs = tok(all_prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Extract last-token logits (left-padding → position -1)
        last_logits = logits[:, -1, :]  # [total_prompts, vocab_size]

        # Score each record
        for meta in record_metadata:
            start = meta["start_idx"]
            n_rw = meta["n_rewrite"]
            n_para = meta["n_para"]
            n_neigh = meta["n_neigh"]
            total = n_rw + n_para + n_neigh

            record_logits = last_logits[start:start + total]
            new_scores = record_logits[:, meta["new_tok_id"]]
            true_scores = record_logits[:, meta["true_tok_id"]]

            # Rewrite + paraphrase: target_new should beat target_true
            rw_para_correct = (new_scores[:n_rw + n_para] > true_scores[:n_rw + n_para]).cpu().numpy()
            # Neighborhood: target_true should beat target_new
            neigh_correct = (true_scores[n_rw + n_para:] > new_scores[n_rw + n_para:]).cpu().numpy()

            all_results.append({
                "case_id": meta["case_id"],
                "efficacy": float(np.mean(rw_para_correct[:n_rw])),
                "paraphrase": float(np.mean(rw_para_correct[n_rw:])),
                "neighborhood": float(np.mean(neigh_correct)),
            })

        # Free GPU memory between mega-batches
        del inputs, logits, last_logits
        torch.cuda.empty_cache()

        if (batch_start + batch_size) % 500 < batch_size:
            done = min(batch_start + batch_size, len(records))
            eff_so_far = np.mean([r["efficacy"] for r in all_results])
            neigh_so_far = np.mean([r["neighborhood"] for r in all_results])
            print(
                f"    [{done}/{len(records)}] "
                f"efficacy={eff_so_far:.4f}, "
                f"neighborhood={neigh_so_far:.4f}"
            )

    return all_results


def evaluate_checkpoint(
    model_name: str,
    ckpt_path: Path,
    records: List[Dict],
    num_edits_per_batch: int,
    total_edits: int,
    fast: bool = True,
    model_tok_cache: Optional[tuple] = None,
) -> tuple:
    """Full evaluation at one checkpoint. Returns (summary, (model, tok)) for reuse."""
    print(f"\n{'='*70}")
    print(f"Evaluating checkpoint: {ckpt_path.name} ({total_edits} edits)")
    print(f"  Records to evaluate: {len(records)}")
    print(f"  Mode: {'fast (first-token)' if fast else 'full (multi-token)'}")
    print(f"{'='*70}")

    if model_tok_cache is not None:
        model, tok = model_tok_cache
        # Just swap weights
        weights_file = ckpt_path / "model_weights.pt"
        print(f"  Swapping weights from: {ckpt_path}")
        weights = torch.load(str(weights_file), map_location="cuda")
        param_dict = dict(model.named_parameters())
        for name, tensor in weights.items():
            if name in param_dict:
                param_dict[name].data.copy_(tensor.cuda().half())
        del weights
        torch.cuda.empty_cache()
    else:
        model, tok = load_model_from_checkpoint(model_name, ckpt_path)

    # Evaluate all records
    if fast:
        # Mega-batched: ~32 records per forward pass (96GB VRAM)
        print("  Using mega-batch evaluation (batch_size=32)")
        results = evaluate_records_batched(model, tok, records, batch_size=32)
    else:
        tok.padding_side = "right"  # Full protocol requires right-padding
        results = []
        for i, record in enumerate(records):
            result = evaluate_record(model, tok, record)
            results.append(result)
            if (i + 1) % 500 == 0:
                eff_so_far = np.mean([r["efficacy"] for r in results])
                neigh_so_far = np.mean([r["neighborhood"] for r in results])
                print(
                    f"    [{i+1}/{len(records)}] "
                    f"efficacy={eff_so_far:.4f}, "
                    f"neighborhood={neigh_so_far:.4f}"
                )

    # Compute aggregate metrics
    all_eff = [r["efficacy"] for r in results]
    all_para = [r["paraphrase"] for r in results]
    all_neigh = [r["neighborhood"] for r in results]

    # Cohort breakdown (100 edits per cohort)
    n_cohorts = total_edits // num_edits_per_batch
    cohort_metrics = {}
    for c in range(n_cohorts):
        start = c * num_edits_per_batch
        end = min((c + 1) * num_edits_per_batch, len(results))
        cohort_results = results[start:end]
        if cohort_results:
            cohort_metrics[c] = {
                "edits_range": f"{start}-{end}",
                "efficacy": float(np.mean([r["efficacy"] for r in cohort_results])),
                "paraphrase": float(np.mean([r["paraphrase"] for r in cohort_results])),
                "neighborhood": float(np.mean([r["neighborhood"] for r in cohort_results])),
                "n_facts": len(cohort_results),
            }

    # Named cohort slices
    first_1k = results[:1000] if len(results) >= 1000 else results
    latest_1k = results[-1000:] if len(results) >= 1000 else results
    latest_100 = results[-100:]

    # Middle cohort: edits 1000-2000 (if available)
    middle = results[1000:2000] if len(results) >= 2000 else results[len(results)//3: 2*len(results)//3]

    # Retention AUC (area under cohort efficacy curve)
    cohort_effs = [cohort_metrics[c]["efficacy"] for c in sorted(cohort_metrics.keys())]
    retention_auc = float(np.trapezoid(cohort_effs) / max(len(cohort_effs) - 1, 1))

    summary = {
        "checkpoint": ckpt_path.name,
        "total_edits": total_edits,
        "n_evaluated": len(results),
        "all_facts": {
            "efficacy": round(float(np.mean(all_eff)), 4),
            "paraphrase": round(float(np.mean(all_para)), 4),
            "neighborhood": round(float(np.mean(all_neigh)), 4),
        },
        "first_1k": {
            "efficacy": round(float(np.mean([r["efficacy"] for r in first_1k])), 4),
            "paraphrase": round(float(np.mean([r["paraphrase"] for r in first_1k])), 4),
            "neighborhood": round(float(np.mean([r["neighborhood"] for r in first_1k])), 4),
        },
        "middle_cohort": {
            "efficacy": round(float(np.mean([r["efficacy"] for r in middle])), 4),
            "paraphrase": round(float(np.mean([r["paraphrase"] for r in middle])), 4),
            "neighborhood": round(float(np.mean([r["neighborhood"] for r in middle])), 4),
        },
        "latest_1k": {
            "efficacy": round(float(np.mean([r["efficacy"] for r in latest_1k])), 4),
            "paraphrase": round(float(np.mean([r["paraphrase"] for r in latest_1k])), 4),
            "neighborhood": round(float(np.mean([r["neighborhood"] for r in latest_1k])), 4),
        },
        "latest_100": {
            "efficacy": round(float(np.mean([r["efficacy"] for r in latest_100])), 4),
            "paraphrase": round(float(np.mean([r["paraphrase"] for r in latest_100])), 4),
            "neighborhood": round(float(np.mean([r["neighborhood"] for r in latest_100])), 4),
        },
        "retention_auc": round(retention_auc, 4),
        "cohort_metrics": cohort_metrics,
    }

    # Print summary
    print(f"\n  Results at {total_edits} edits:")
    print(f"    All facts:   eff={summary['all_facts']['efficacy']:.4f}  "
          f"para={summary['all_facts']['paraphrase']:.4f}  "
          f"neigh={summary['all_facts']['neighborhood']:.4f}")
    print(f"    First 1K:    eff={summary['first_1k']['efficacy']:.4f}  "
          f"para={summary['first_1k']['paraphrase']:.4f}  "
          f"neigh={summary['first_1k']['neighborhood']:.4f}")
    print(f"    Latest 1K:   eff={summary['latest_1k']['efficacy']:.4f}  "
          f"para={summary['latest_1k']['paraphrase']:.4f}  "
          f"neigh={summary['latest_1k']['neighborhood']:.4f}")
    print(f"    Latest 100:  eff={summary['latest_100']['efficacy']:.4f}  "
          f"para={summary['latest_100']['paraphrase']:.4f}  "
          f"neigh={summary['latest_100']['neighborhood']:.4f}")
    print(f"    Retention AUC: {summary['retention_auc']:.4f}")

    return summary, (model, tok)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of MEMIT+SeqReg checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_prev", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--alg_name", type=str, default=None,
                        help="Algorithm name for output path (default: auto-detect from checkpoint_dir or MEMIT-Seq)")
    parser.add_argument("--ordering", type=str, default=None,
                        help="Ordering type (e.g. key_clustered, key_dispersed) — used in output path")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=[19, 29, 49],
        help="Batch indices to evaluate (default: 19 29 49 = 2K, 3K, 5K edits)"
    )
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument(
        "--checkpoint_dir", default=None,
        help="Explicit checkpoint directory (overrides auto-resolved path)"
    )
    parser.add_argument(
        "--dataset_path", default=None,
        help="Path to multi_counterfact.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast first-token evaluation (~5x faster, less accurate)"
    )
    parser.add_argument(
        "--full", action="store_true", default=True,
        help="Use full multi-token evaluation (matches vendor exactly, default)"
    )
    args = parser.parse_args()
    if args.fast:
        args.full = False

    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    # Find checkpoint directory
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir).expanduser()
    else:
        ckpt_dir = resolve_checkpoint_dir(args.seed, args.lambda_prev, args.lambda_delta)
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
        # Auto-detect: check vendor data dir, then S3 mount
        candidates = [
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
            Path("/s3-data/continual-learning/alphaedit/dsets/multi_counterfact.json"),
            Path.home() / "Projects" / "alphaedit-analysis" / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
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

    # Evaluate each checkpoint (reuse model across checkpoints)
    all_summaries = {}
    model_cache = None
    for batch_idx in args.checkpoints:
        total_edits = (batch_idx + 1) * args.num_edits
        records_to_eval = all_records[:total_edits]
        ckpt_path = ckpt_dir / f"batch_{batch_idx}"

        summary, model_cache = evaluate_checkpoint(
            model_name=model_name,
            ckpt_path=ckpt_path,
            records=records_to_eval,
            num_edits_per_batch=args.num_edits,
            total_edits=total_edits,
            fast=args.fast,
            model_tok_cache=model_cache,
        )
        all_summaries[f"{total_edits}_edits"] = summary

    # Cleanup
    if model_cache:
        del model_cache
        torch.cuda.empty_cache()

    # Save results in matched_ordering format:
    #   results/matched_ordering/{alg_name}/{ordering}/seed{seed}/full_eval_seed{seed}.json
    if args.alg_name:
        variant_name = args.alg_name
    elif args.checkpoint_dir:
        # Derive from checkpoint_dir structure:
        #   .../AlphaEdit/key_clustered/seed42 → alg=AlphaEdit, ordering=key_clustered
        #   .../matched_key_clustered_seed42 → fall back to MEMIT-Seq
        ckpt_parts = Path(args.checkpoint_dir).parts
        if "AlphaEdit" in ckpt_parts:
            variant_name = "AlphaEdit"
        else:
            variant_name = f"MEMIT-Seq-lp{args.lambda_prev}-ld{args.lambda_delta}"
    else:
        variant_name = f"MEMIT-Seq-lp{args.lambda_prev}-ld{args.lambda_delta}"

    # Determine ordering from --ordering flag or dataset_path filename
    ordering = args.ordering
    if not ordering and args.dataset_path:
        ds_stem = Path(args.dataset_path).stem  # e.g. "key_clustered_seed42"
        # Strip _seed{N} suffix
        for suffix in [f"_seed{args.seed}", f"_seed"]:
            if suffix in ds_stem:
                ordering = ds_stem[:ds_stem.index(suffix)]
                break

    if ordering:
        out_dir = PROJECT_ROOT / "results" / "matched_ordering" / variant_name / ordering / f"seed{args.seed}"
    else:
        out_dir = PROJECT_ROOT / "results" / "matched_ordering" / variant_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_eval_seed{args.seed}.json"
    with open(str(out_path), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Final comparison table
    print(f"\n{'='*80}")
    print("MEMIT+SeqReg Full Evaluation Summary")
    print(f"{'='*80}")
    print(f"{'Checkpoint':<12} {'All Eff':>8} {'All Para':>9} {'All Neigh':>10} "
          f"{'1st1K Eff':>10} {'Lat1K Eff':>10} {'Lat100':>8} {'AUC':>6}")
    print("-" * 80)
    for key in sorted(all_summaries.keys()):
        s = all_summaries[key]
        print(
            f"{s['total_edits']:>5} edits  "
            f"{s['all_facts']['efficacy']:>8.4f} "
            f"{s['all_facts']['paraphrase']:>9.4f} "
            f"{s['all_facts']['neighborhood']:>10.4f} "
            f"{s['first_1k']['efficacy']:>10.4f} "
            f"{s['latest_1k']['efficacy']:>10.4f} "
            f"{s['latest_100']['efficacy']:>8.4f} "
            f"{s['retention_auc']:>6.4f}"
        )


if __name__ == "__main__":
    main()
