#!/usr/bin/env python3
"""
Capability Probe Runner: Editing + perplexity/MMLU measurement pipeline.

Uses evaluate_harness.run_experiment() with probe hooks instead of
exec(compile(evaluate.py)).

Usage:
    python src/capability_probe_runner.py \
        --seed 42 --cuda_device 0 --alg_name AlphaEdit \
        --ds_name mcf --dataset_size_limit 2000 --num_edits 100 \
        --probe_interval 5
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))
sys.path.insert(0, str(_SRC_DIR))

from model_registry import DEFAULT_MODEL
from model_resolve import resolve_model_path
from setup_hparams import link_hparams
from paths import get_project_root, get_alphaedit_root, get_result_root
from evaluate_harness import (
    run_experiment, load_model_and_tok, load_dataset,
    ExperimentHooks, _canonical_name_or_path,
)


ALG_DICT = None

def _init_alg_dict():
    global ALG_DICT
    alphaedit_root = get_alphaedit_root()
    sys.path.insert(0, str(alphaedit_root))
    from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
    from AlphaEdit import AlphaEditHyperParams
    from memit.memit_main import apply_memit_to_model
    from memit import MEMITHyperParams
    ALG_DICT = {
        "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
        "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    }


def _compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=4):
    """Compute corpus-level perplexity on WikiText texts."""
    from torch.nn import functional as F
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts, return_tensors="pt",
            max_length=max_length, truncation=True, padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**encodings).logits
        for j in range(encodings["input_ids"].shape[0]):
            mask = encodings["attention_mask"][j] == 1
            valid_ids = encodings["input_ids"][j][mask]
            valid_logits = logits[j][mask]
            if len(valid_ids) < 2:
                continue
            nll = F.cross_entropy(valid_logits[:-1], valid_ids[1:], reduction="sum").item()
            total_nll += nll
            total_tokens += len(valid_ids) - 1

    if total_tokens == 0:
        return {"mean_perplexity": float("nan"), "n_tokens": 0}
    return {
        "mean_perplexity": float(np.exp(total_nll / total_tokens)),
        "n_tokens": total_tokens,
    }


def _load_wikitext(n_samples=200):
    """Load WikiText-103 test split for perplexity measurement."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [item["text"].strip() for item in ds if len(item["text"].strip()) > 100]
    return texts[:n_samples]


def run(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if args.alg_name.startswith("MEMIT-Seq"):
        print(f"ERROR: Online probing not supported for '{args.alg_name}'. Use offline probe.")
        sys.exit(1)

    link_hparams()
    alphaedit_root = get_alphaedit_root()
    sys.path.insert(0, str(alphaedit_root))
    _init_alg_dict()

    if args.alg_name not in ALG_DICT:
        print(f"ERROR: Unknown algorithm '{args.alg_name}'. Available: {list(ALG_DICT.keys())}")
        sys.exit(1)

    HparamsClass, apply_fn = ALG_DICT[args.alg_name]
    hparams = HparamsClass.from_json(str(alphaedit_root / "hparams" / args.alg_name / args.hparams_fname))

    model, tok = load_model_and_tok(args.model_name)
    dataset = load_dataset(args.ds_name, args.dataset_size_limit, alphaedit_root)

    # Probe output
    output_dir = get_result_root() / "capability_probe" / f"seed{args.seed}" / f"{args.dataset_size_limit}edits" / args.alg_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = output_dir / f"probe_{timestamp}.jsonl"

    results_dir = output_dir.parent
    wikitext_texts = None
    probe_records = []

    def after_edit(batch_idx, model, records, hparams, edit_result, exec_time):
        nonlocal wikitext_texts
        total_edits = (batch_idx + 1) * args.num_edits
        if (batch_idx + 1) % args.probe_interval != 0:
            return
        print(f"  [PROBE] Running at {total_edits} edits...")
        if wikitext_texts is None:
            wikitext_texts = _load_wikitext()
        ppl = _compute_perplexity(model, tok, wikitext_texts)
        record = {"edit_count": total_edits, "timestamp_utc": time.time(), **ppl}
        probe_records.append(record)
        with open(output_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [PROBE] Perplexity: {ppl['mean_perplexity']:.2f}")

    hooks = ExperimentHooks(
        after_edit=after_edit,
        should_eval=lambda _: False,  # skip MCF eval, we only want probes
    )

    print(f"{'=' * 70}")
    print("Capability Probe Runner (harness)")
    print(f"  Algorithm:      {args.alg_name}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Probe interval: every {args.probe_interval} batches")
    print(f"  Output:         {output_jsonl}")
    print(f"{'=' * 70}")

    run_experiment(
        model=model, tok=tok, hparams=hparams,
        dataset=dataset, apply_fn=apply_fn,
        alg_name=args.alg_name, num_edits=args.num_edits,
        results_dir=results_dir, ds_name=args.ds_name,
        hooks=hooks,
    )

    print(f"\n=== Capability probe complete ===")
    print(f"  Recorded {len(probe_records)} measurement points")
    print(f"  Output: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Run editing with capability probing")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", required=True)
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--probe_interval", type=int, default=5)
    parser.add_argument("--no_mmlu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
