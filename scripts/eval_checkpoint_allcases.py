#!/usr/bin/env python3
"""Evaluate ALL MCF cases at a given checkpoint and save per-case JSONs.

Used to build multi-timepoint panels for the survival model on any architecture.
Produces output identical to the failure_curve_checkpointed format.

Usage:
    uv run python scripts/eval_checkpoint_allcases.py \
        --checkpoint_path /path/to/batch_29/model_weights.pt \
        --output_dir results/failure_curve_gptj/seed42/3000edits/AlphaEdit/run_000 \
        --model_name EleutherAI/gpt-j-6b \
        --num_edits 3000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


def load_model_with_checkpoint(model_name: str, checkpoint_path: Path, device: str = "cuda"):
    """Load base model and apply checkpoint weights."""
    from model_resolve import resolve_model_path
    token = os.environ.get("HF_TOKEN")
    model_path = resolve_model_path(model_name)

    print(f"  Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, token=token,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(model_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    weights_file = checkpoint_path / "model_weights.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"No weights at {weights_file}")

    print(f"  Applying checkpoint: {checkpoint_path}")
    weights = torch.load(str(weights_file), map_location=device)
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.to(device).half())
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    print(f"  Loaded {loaded} weight tensors")

    model.eval()
    return model, tok


def evaluate_case(model, tok, record: dict) -> dict:
    """Evaluate a single case: efficacy, paraphrase, neighborhood."""
    rw = record["requested_rewrite"]
    subject = rw["subject"]
    target_new = rw["target_new"]["str"]
    target_true = rw["target_true"]["str"]

    rewrite_prompts = [rw["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    all_prompts = rewrite_prompts + paraphrase_prompts + neighborhood_prompts
    if not all_prompts:
        return None

    new_tok_id = tok(f" {target_new}", add_special_tokens=False).input_ids[0]
    true_tok_id = tok(f" {target_true}", add_special_tokens=False).input_ids[0]

    inputs = tok(all_prompts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    last_logits = logits[:, -1, :]
    new_scores = last_logits[:, new_tok_id].cpu().numpy()
    true_scores = last_logits[:, true_tok_id].cpu().numpy()

    n_rw = len(rewrite_prompts)
    n_para = len(paraphrase_prompts)

    rw_correct = [bool(new_scores[i] > true_scores[i]) for i in range(n_rw)]
    para_correct = [bool(new_scores[i] > true_scores[i]) for i in range(n_rw, n_rw + n_para)]
    neigh_correct = [bool(true_scores[i] > new_scores[i]) for i in range(n_rw + n_para, len(all_prompts))]

    probs = [
        {"target_new": float(new_scores[i]), "target_true": float(true_scores[i])}
        for i in range(n_rw)
    ]

    return {
        "case_id": record["case_id"],
        "requested_rewrite": rw,
        "num_edits": 0,
        "post": {
            "rewrite_prompts_correct": rw_correct,
            "rewrite_prompts_probs": probs,
            "paraphrase_prompts_correct": para_correct,
            "neighborhood_prompts_correct": neigh_correct,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all MCF cases at a checkpoint")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    parser.add_argument("--num_edits", type=int, default=10000)
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    # Load dataset
    ds_paths = [
        args.dataset_path,
        PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
        PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json",
        Path(os.environ.get("DATA_ROOT", "")) / "multi_counterfact.json",
    ]
    dataset = None
    for p in ds_paths:
        if p and p.exists():
            print(f"Loading dataset: {p}")
            with open(p) as f:
                dataset = json.load(f)
            break
    if dataset is None:
        print("ERROR: multi_counterfact.json not found")
        sys.exit(1)

    # Limit to num_edits
    records = dataset[:args.num_edits]
    print(f"Evaluating {len(records)} cases")

    # Load model + checkpoint
    model, tok = load_model_with_checkpoint(args.model_name, args.checkpoint_path)

    # Evaluate all cases
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0

    for i, record in enumerate(records):
        result = evaluate_case(model, tok, record)
        if result is None:
            continue

        result["num_edits"] = args.num_edits
        batch_num = i // 100
        out_path = args.output_dir / f"100_edits-case_{record['case_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f)

        n_done += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  [{i+1}/{len(records)}] {rate:.1f} cases/sec, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} cases evaluated in {elapsed:.1f}s")
    print(f"Results at: {args.output_dir}")


if __name__ == "__main__":
    main()
