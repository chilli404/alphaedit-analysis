#!/usr/bin/env python3
"""
General Capability Probe for edited models.

Measures whether sequential knowledge editing degrades the model's general
capabilities. This addresses a key reviewer concern: "editing may be
destroying the model quietly while still passing CounterFact metrics."

Evaluates:
1. Perplexity on WikiText-103 (held-out text, sliding window)
2. 5-shot MMLU accuracy (subset: 4 diverse categories)

The probe runs at each downstream_eval_step during editing, producing a
timeseries of (total_edits, perplexity, mmlu_accuracy) that can be plotted
alongside the editing metrics.

This module is injected into the evaluate.py execution via source patching
in seeded_runner.py. It can also run standalone for post-hoc analysis.

Usage (standalone, requires model on GPU):
    python src/capability_probe.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --output results/capability_baseline.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


# Number of WikiText-103 samples for perplexity (balances accuracy vs speed)
WIKITEXT_N_SAMPLES = 200
WIKITEXT_MAX_LENGTH = 512  # tokens per sample (sliding window stride)
MMLU_N_SHOTS = 5
MMLU_N_QUESTIONS = 50  # per category


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = WIKITEXT_MAX_LENGTH,
    batch_size: int = 4,
) -> dict:
    """
    Compute perplexity on a list of text passages using a sliding window.

    Returns dict with: mean_perplexity, std_perplexity, n_samples, n_tokens
    """
    model.eval()
    device = next(model.parameters()).device

    sample_nlls = []  # (total_nll, n_tokens) per sample
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Compute per-sample NLL
        for j in range(input_ids.shape[0]):
            # Get valid tokens (non-padding)
            mask = attention_mask[j] == 1
            valid_ids = input_ids[j][mask]
            valid_logits = logits[j][mask]

            if len(valid_ids) < 2:
                continue

            # Shift: predict next token from current
            shift_logits = valid_logits[:-1]
            shift_labels = valid_ids[1:]

            nll = F.cross_entropy(
                shift_logits, shift_labels, reduction="sum"
            ).item()

            n_tokens = len(shift_labels)
            sample_nlls.append((nll, n_tokens))
            total_nll += nll
            total_tokens += n_tokens

    if not sample_nlls:
        return {"mean_perplexity": float("nan"), "std_perplexity": float("nan"),
                "n_samples": 0, "n_tokens": 0}

    # Corpus-level perplexity: exp(total_NLL / total_tokens)
    # This weights each token equally, which is the standard definition.
    corpus_ppl = float(np.exp(total_nll / total_tokens))

    # Per-sample perplexities (for confidence intervals / spread reporting)
    per_sample_ppl = [np.exp(nll / n) for nll, n in sample_nlls]

    return {
        "mean_perplexity": corpus_ppl,
        "median_perplexity": float(np.median(per_sample_ppl)),
        "std_perplexity": float(np.std(per_sample_ppl)),
        "n_samples": len(sample_nlls),
        "n_tokens": total_tokens,
    }


def load_wikitext_samples(n_samples: int = WIKITEXT_N_SAMPLES) -> list[str]:
    """
    Load text passages from WikiText-103 test set.

    Uses the test split to avoid any overlap with the Wikipedia-based
    covariance statistics used by AlphaEdit.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Filter out empty lines and very short passages
    texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:  # At least 100 characters
            texts.append(text)
        if len(texts) >= n_samples:
            break

    return texts


def compute_mmlu_accuracy(
    model,
    tokenizer,
    n_shots: int = MMLU_N_SHOTS,
    n_questions: int = MMLU_N_QUESTIONS,
    categories: list[str] | None = None,
) -> dict:
    """
    Compute few-shot MMLU accuracy on a diverse subset of categories.

    Uses 4 categories spanning different knowledge domains to detect
    broad capability degradation without running the full 57-category MMLU.
    """
    from datasets import load_dataset

    if categories is None:
        # Diverse subset: science, humanities, social science, STEM
        categories = [
            "abstract_algebra",
            "world_religions",
            "us_foreign_policy",
            "college_biology",
        ]

    model.eval()
    device = next(model.parameters()).device

    results = {}
    total_correct = 0
    total_questions = 0

    for category in categories:
        try:
            ds = load_dataset("cais/mmlu", category, split="test")
            val_ds = load_dataset("cais/mmlu", category, split="validation")
        except Exception:
            # Fall back to older MMLU format
            try:
                ds = load_dataset("lukaemon/mmlu", category, split="test")
                val_ds = load_dataset("lukaemon/mmlu", category, split="train")
            except Exception:
                continue

        # Use validation set for few-shot examples
        few_shot_examples = list(val_ds)[:n_shots]

        correct = 0
        tested = 0
        choices = ["A", "B", "C", "D"]

        for item in list(ds)[:n_questions]:
            # Build few-shot prompt
            prompt = f"The following are multiple choice questions about {category.replace('_', ' ')}.\n\n"

            for ex in few_shot_examples:
                prompt += _format_mmlu_question(ex, choices) + "\n\n"

            prompt += _format_mmlu_question(item, choices, include_answer=False)

            # Get model's prediction
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=2048).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]

            # Get logits for A, B, C, D tokens
            choice_logits = []
            for choice in choices:
                token_id = tokenizer.encode(choice, add_special_tokens=False)
                if token_id:
                    choice_logits.append(next_token_logits[token_id[0]].item())
                else:
                    choice_logits.append(float("-inf"))

            predicted = choices[np.argmax(choice_logits)]
            answer_idx = item.get("answer", item.get("target", 0))
            if isinstance(answer_idx, int):
                correct_answer = choices[answer_idx]
            else:
                correct_answer = str(answer_idx)

            if predicted == correct_answer:
                correct += 1
            tested += 1

        if tested > 0:
            results[category] = {
                "accuracy": correct / tested,
                "correct": correct,
                "total": tested,
            }
            total_correct += correct
            total_questions += tested

    overall_accuracy = total_correct / total_questions if total_questions > 0 else float("nan")

    return {
        "mmlu_accuracy": overall_accuracy,
        "mmlu_correct": total_correct,
        "mmlu_total": total_questions,
        "mmlu_per_category": results,
    }


def _format_mmlu_question(item: dict, choices: list[str], include_answer: bool = True) -> str:
    """Format a single MMLU question as text."""
    question = item.get("question", item.get("input", ""))
    options = item.get("choices", [])

    text = f"Question: {question}\n"
    for i, opt in enumerate(options):
        text += f"{choices[i]}. {opt}\n"
    text += "Answer:"

    if include_answer:
        answer_idx = item.get("answer", item.get("target", 0))
        if isinstance(answer_idx, int):
            text += f" {choices[answer_idx]}"
        else:
            text += f" {answer_idx}"

    return text


def run_capability_probe(
    model,
    tokenizer,
    edit_count: int = 0,
    compute_mmlu: bool = True,
) -> dict:
    """
    Run the full capability probe on the current model state.

    Called at each downstream_eval_step during editing.

    Args:
        model: The (possibly edited) model
        tokenizer: The tokenizer
        edit_count: How many edits have been applied so far
        compute_mmlu: Whether to run MMLU (slower but more informative)

    Returns:
        Dict with perplexity and optionally MMLU scores
    """
    result = {
        "edit_count": edit_count,
        "timestamp": time.time(),
    }

    # 1. Perplexity on WikiText-103 test set
    texts = load_wikitext_samples(n_samples=WIKITEXT_N_SAMPLES)
    ppl_result = compute_perplexity(model, tokenizer, texts)
    result.update(ppl_result)

    # 2. MMLU (optional, adds ~5-10 min per eval point)
    if compute_mmlu:
        mmlu_result = compute_mmlu_accuracy(model, tokenizer)
        result.update(mmlu_result)

    return result


def main():
    """Standalone mode: measure baseline capabilities of an unedited model."""
    parser = argparse.ArgumentParser(
        description="Measure general model capabilities (perplexity + MMLU)"
    )
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("results/capability_baseline.json"))
    parser.add_argument("--no_mmlu", action="store_true", help="Skip MMLU (faster)")
    parser.add_argument("--n_samples", type=int, default=WIKITEXT_N_SAMPLES)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_download import resolve_model_path

    model_path = resolve_model_path(args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    print("Running capability probe (baseline, no edits)...")
    result = run_capability_probe(
        model, tokenizer,
        edit_count=0,
        compute_mmlu=not args.no_mmlu,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults:")
    print(f"  Perplexity: {result['mean_perplexity']:.2f} (±{result['std_perplexity']:.2f})")
    print(f"  Samples: {result['n_samples']}, Tokens: {result['n_tokens']}")
    if "mmlu_accuracy" in result:
        print(f"  MMLU accuracy: {result['mmlu_accuracy']:.3f} ({result['mmlu_correct']}/{result['mmlu_total']})")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
