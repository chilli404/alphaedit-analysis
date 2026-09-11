#!/usr/bin/env python3
"""
Extract key vectors for all prompt variants of MCF records.

For each MCF record, extracts keys from:
  - The original prompt template (e.g., "{}, which is located in")
  - Paraphrase prompt 0 (longer context before subject)
  - Paraphrase prompt 1 (longer context before subject)

These keys are used by the same-fact-damage runner to construct
HIGH-overlap and LOW-overlap batches with identical factual content
but different key geometry.

Output: .npz with keys_original, keys_para0, keys_para1, case_ids

Usage:
    uv run python src/experiments/prompt_variant_keys.py --seed 42
    uv run python src/experiments/prompt_variant_keys.py --seed 42 --n_records 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
from model_resolve import resolve_model_path


def find_subject_last_token(tok, text, subject):
    """Find the token index of the subject's last token in the full text."""
    prefix = text[:text.index(subject) + len(subject)]
    prefix_ids = tok(prefix, return_tensors="pt")["input_ids"][0]
    return len(prefix_ids) - 1


def extract_key_at_subject(model, tok, text, subject, layer, module_template):
    """Extract the MLP input representation at the subject's last token."""
    weight_name = f"{module_template.format(layer)}.weight"

    captured = {}
    def hook(module, input, output):
        captured["input"] = input[0].detach()

    param = dict(model.named_parameters())[weight_name]
    handle = param.register_hook if hasattr(param, "register_hook") else None

    # Use a forward hook on the module
    module_name = module_template.format(layer)
    parts = module_name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)

    handle = mod.register_forward_hook(hook)

    try:
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        tok_idx = find_subject_last_token(tok, text, subject)
        key = captured["input"][0, tok_idx].cpu().float()
        return key
    finally:
        handle.remove()


def make_prompt_variants(record):
    """Generate prompt variants for a single MCF record.

    Returns list of (variant_name, full_prompt_text) tuples.
    """
    rw = record["requested_rewrite"]
    subject = rw["subject"]
    original = rw["prompt"].format(subject)

    variants = [("original", original)]

    for i, para in enumerate(record.get("paraphrase_prompts", [])):
        if subject in para:
            variants.append((f"para{i}", para))

    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Extract key vectors for prompt variants of MCF records"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_records", type=int, default=5000,
                        help="Number of records to extract keys for")
    parser.add_argument("--model_name", default=os.environ.get(
        "MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--stream_path", default=None,
                        help="Path to ordering JSON (defaults to key_clustered)")
    parser.add_argument("--output_dir", default="results/key_vectors/prompt_variants")
    args = parser.parse_args()

    model_name = resolve_model_path(args.model_name)

    # Load stream
    if args.stream_path:
        stream_path = Path(args.stream_path)
    else:
        stream_path = (PROJECT_ROOT / "results" / "matched_ordering" / "orderings"
                       / f"key_clustered_seed{args.seed}.json")
    with open(stream_path) as f:
        records = json.load(f)[:args.n_records]

    print(f"Extracting prompt-variant keys for {len(records)} records")
    print(f"  Model: {model_name}")
    print(f"  Layer: {args.layer}")

    # Determine module template from model name
    if "gpt-j" in model_name.lower():
        module_template = "transformer.h.{}.mlp.fc_out"
    else:
        module_template = "model.layers.{}.mlp.down_proj"

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model.eval()

    # Extract keys
    all_keys = {"original": [], "para0": [], "para1": []}
    case_ids = []
    errors = 0

    for idx, record in enumerate(records):
        rw = record["requested_rewrite"]
        subject = rw["subject"]
        case_id = record["case_id"]

        variants = make_prompt_variants(record)

        try:
            keys_for_record = {}
            for vname, text in variants:
                key = extract_key_at_subject(
                    model, tok, text, subject, args.layer, module_template
                )
                keys_for_record[vname] = key

            # Store (use zero vector if variant missing)
            d = keys_for_record.get("original", torch.zeros_like(next(iter(keys_for_record.values()))))
            all_keys["original"].append(keys_for_record.get("original", d))
            all_keys["para0"].append(keys_for_record.get("para0", d))
            all_keys["para1"].append(keys_for_record.get("para1", d))
            case_ids.append(case_id)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on case {case_id}: {e}")

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(records)}] extracted, {errors} errors")

    # Save — write to /tmp first then copy (S3 FUSE can't handle np.savez seeks)
    import shutil, tempfile
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt_variant_keys_seed{args.seed}_layer{args.layer}.npz"

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    np.savez(
        tmp_path,
        keys_original=torch.stack(all_keys["original"]).numpy(),
        keys_para0=torch.stack(all_keys["para0"]).numpy(),
        keys_para1=torch.stack(all_keys["para1"]).numpy(),
        case_ids=np.array(case_ids, dtype=np.int32),
    )
    shutil.copyfile(tmp_path, str(out_path))
    os.unlink(tmp_path)
    print(f"\nSaved: {out_path}")
    print(f"  {len(case_ids)} records, {errors} errors")
    print(f"  Keys shape: {torch.stack(all_keys['original']).shape}")

    # Quick diagnostic: how much do keys vary across variants?
    orig = torch.stack(all_keys["original"])
    p0 = torch.stack(all_keys["para0"])
    p1 = torch.stack(all_keys["para1"])

    orig_n = orig / orig.norm(dim=1, keepdim=True).clamp(min=1e-8)
    p0_n = p0 / p0.norm(dim=1, keepdim=True).clamp(min=1e-8)
    p1_n = p1 / p1.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cos_orig_p0 = (orig_n * p0_n).sum(dim=1)
    cos_orig_p1 = (orig_n * p1_n).sum(dim=1)
    cos_p0_p1 = (p0_n * p1_n).sum(dim=1)

    print(f"\n  Within-fact cosine (same fact, different prompt):")
    print(f"    original ↔ para0: {cos_orig_p0.mean():.4f} ± {cos_orig_p0.std():.4f}")
    print(f"    original ↔ para1: {cos_orig_p1.mean():.4f} ± {cos_orig_p1.std():.4f}")
    print(f"    para0 ↔ para1:    {cos_p0_p1.mean():.4f} ± {cos_p0_p1.std():.4f}")
    print(f"  (lower = more variation = more room for the experiment)")


if __name__ == "__main__":
    main()
