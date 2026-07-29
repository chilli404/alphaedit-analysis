#!/usr/bin/env python3
"""
Offline Capability Probe: Runs perplexity/MMLU probes on existing failure
curve checkpoints WITHOUT re-editing the model.

Loads the base model once, then for each checkpoint:
  1. Applies saved layer weights to the model
  2. Runs perplexity measurement (WikiText-103)
  3. Optionally runs MMLU
  4. Restores base model weights before next checkpoint

Produces output identical in format to capability_probe_runner.py.

Requirements:
  - Failure curve checkpoints must already exist (from checkpoint_runner.py)
  - GPU with enough VRAM for the model (~16GB for Llama-3-8B in fp16)

Usage:
    python src/capability_probe_offline.py \\
        --seed 42 \\
        --alg_name AlphaEdit \\
        --checkpoint_dir ~/.cache/alphaedit_checkpoints/failure_curve/AlphaEdit/seed42

    # Auto-detect checkpoint dir, skip MMLU for speed
    python src/capability_probe_offline.py --seed 42 --alg_name AlphaEdit --no_mmlu
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from capability_probe import (
    compute_mmlu_accuracy,
    compute_perplexity,
    load_wikitext_samples,
    PROBE_DTYPE,
    PROBE_VERSION,
    WIKITEXT_N_SAMPLES,
    WIKITEXT_MAX_LENGTH,
)
from model_download import resolve_model_path
from paths import get_project_root, get_result_root, get_checkpoint_root


def resolve_checkpoint_dir(explicit_dir: str | None, alg_name: str, seed: int) -> Path:
    """Resolve checkpoint directory using CHECKPOINT_ROOT env var.

    For MEMIT-Seq variants (e.g. MEMIT-Seq-lp1.0-ld1.0-cache0), checkpoints
    live at: {CHECKPOINT_ROOT}/failure_curve/{variant}/seed{N}/
    For standard algorithms (AlphaEdit, MEMIT): {CHECKPOINT_ROOT}/{alg}/seed{N}/
    """
    if explicit_dir:
        return Path(explicit_dir)

    ckpt_root = get_checkpoint_root()

    # MEMIT-Seq variants always live under failure_curve/
    if alg_name.startswith("MEMIT-Seq"):
        return ckpt_root / "failure_curve" / alg_name / f"seed{seed}"

    # Standard algorithms: try failure_curve/ first, then flat layout
    fc_path = ckpt_root / "failure_curve" / alg_name / f"seed{seed}"
    if fc_path.exists():
        return fc_path

    flat_path = ckpt_root / alg_name / f"seed{seed}"
    if flat_path.exists():
        return flat_path

    # Default to failure_curve convention
    return fc_path


def find_all_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Find all valid checkpoints sorted by batch index."""
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for batch_dir in sorted(ckpt_dir.glob("batch_*"), key=lambda p: int(p.name.split("_")[1])):
        metadata_file = batch_dir / "metadata.json"
        weights_file = batch_dir / "model_weights.pt"

        if not metadata_file.exists():
            continue
        if not weights_file.exists():
            continue

        try:
            batch_idx = int(batch_dir.name.split("_")[1])
            checkpoints.append((batch_idx, batch_dir))
        except (ValueError, IndexError):
            continue

    return checkpoints


def apply_checkpoint_weights(model, weights_path: Path) -> int:
    """Apply checkpoint layer weights to model. Returns count of loaded params.

    Copies checkpoint to local /tmp before loading to avoid S3 FUSE read
    failures on large files (torch.load needs reliable random access).
    """

    # Copy to local disk if on a FUSE mount (heuristic: path starts with /s3-)
    source = Path(weights_path)
    if str(source).startswith("/s3"):
        local_copy = Path(tempfile.mktemp(suffix=".pt", prefix="ckpt_"))
        shutil.copy2(str(source), str(local_copy))
        load_path = str(local_copy)
    else:
        local_copy = None
        load_path = str(source)

    try:
        layer_weights = torch.load(
            load_path,
            map_location="cuda",
            weights_only=True,
        )
    finally:
        if local_copy is not None:
            local_copy.unlink(missing_ok=True)

    param_dict = dict(model.named_parameters())
    loaded = 0
    for param_name, param_data in layer_weights.items():
        if param_name in param_dict:
            param_dict[param_name].data.copy_(param_data)
            loaded += 1
    return loaded


def run_offline_probes(
    model,
    tokenizer,
    ckpt_dir: Path,
    output_jsonl: Path,
    compute_mmlu: bool = True,
    seed: int | None = None,
    alg_name: str | None = None,
) -> list[dict]:
    """Run capability probes on all checkpoints."""
    checkpoints = find_all_checkpoints(ckpt_dir)

    if not checkpoints:
        print(f"ERROR: No valid checkpoints found in {ckpt_dir}")
        print("  Checkpoints must contain both metadata.json and model_weights.pt")
        sys.exit(1)

    print(f"  Found {len(checkpoints)} checkpoints")

    # Pre-load WikiText once
    print("  Loading WikiText-103 test samples...")
    texts = load_wikitext_samples(n_samples=200)
    print(f"  Loaded {len(texts)} text passages")

    # Save base model weights for restoration between checkpoints
    print("  Saving base model state for restoration...")
    base_state = {}
    # Only save params that checkpoints modify (edited layers)
    # Determine which params are in the first checkpoint
    first_ckpt_path = checkpoints[0][1] / "model_weights.pt"
    if str(first_ckpt_path).startswith("/s3"):
        _tmp = Path(tempfile.mktemp(suffix=".pt", prefix="ckpt_init_"))
        shutil.copy2(str(first_ckpt_path), str(_tmp))
        first_weights = torch.load(str(_tmp), map_location="cpu", weights_only=True)
        _tmp.unlink(missing_ok=True)
    else:
        first_weights = torch.load(str(first_ckpt_path), map_location="cpu", weights_only=True)
    param_dict = dict(model.named_parameters())
    for param_name in first_weights.keys():
        if param_name in param_dict:
            base_state[param_name] = param_dict[param_name].data.clone()
    del first_weights
    print(f"  Saved {len(base_state)} base parameter tensors")

    # Write to a local temp file, then copy to final destination.
    # S3 FUSE mounts do not support repeated append-mode writes reliably.
    local_tmp = Path(tempfile.mktemp(suffix=".jsonl", prefix="capability_probe_"))
    print(f"  Writing to local temp: {local_tmp}")
    print(f"  Final destination:     {output_jsonl}")

    # Build shared metadata for all records
    model_dtype = str(next(model.parameters()).dtype)
    shared_metadata = {
        "probe_version": PROBE_VERSION,
        "model_dtype": model_dtype,
        "wikitext_split": "test",
        "wikitext_dataset": "wikitext-103-raw-v1",
        "wikitext_n_samples": WIKITEXT_N_SAMPLES,
        "wikitext_max_length": WIKITEXT_MAX_LENGTH,
        "seed": seed,
        "alg_name": alg_name,
    }

    # Run baseline measurement (0 edits)
    print("\n  [PROBE] Baseline (0 edits)...")
    t0 = time.time()
    ppl_result = compute_perplexity(model, tokenizer, texts)
    baseline_record = {
        "edit_count": 0,
        "timestamp_utc": time.time(),
        "source": "offline_probe",
        "metadata": {**shared_metadata, "checkpoint_path": None},
        **ppl_result,
    }
    if compute_mmlu:
        mmlu_result = compute_mmlu_accuracy(model, tokenizer)
        baseline_record.update(mmlu_result)
    elapsed = time.time() - t0
    print(f"         Perplexity: {ppl_result['mean_perplexity']:.2f} ({elapsed:.1f}s)")

    records = [baseline_record]
    with open(local_tmp, "a") as f:
        f.write(json.dumps(baseline_record) + "\n")

    # Process each checkpoint
    for batch_idx, batch_dir in checkpoints:
        # Load metadata for edit count
        with open(batch_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        total_edits = metadata.get("total_edits", (batch_idx + 1) * 100)

        # Restore base weights first (clean slate)
        for param_name, base_data in base_state.items():
            param_dict[param_name].data.copy_(base_data)

        # Apply this checkpoint's weights (skip if corrupted)
        try:
            loaded = apply_checkpoint_weights(model, batch_dir / "model_weights.pt")
        except (RuntimeError, OSError) as e:
            print(f"\n  [SKIP] Batch {batch_idx} ({total_edits} edits): checkpoint unreadable — {e}")
            continue

        # Run probe
        print(f"\n  [PROBE] Batch {batch_idx} ({total_edits} edits, {loaded} params loaded)...")
        t0 = time.time()
        ppl_result = compute_perplexity(model, tokenizer, texts)

        record = {
            "edit_count": total_edits,
            "batch_idx": batch_idx,
            "timestamp_utc": time.time(),
            "source": "offline_probe",
            "metadata": {**shared_metadata, "checkpoint_path": str(batch_dir)},
            **ppl_result,
        }

        if compute_mmlu:
            mmlu_result = compute_mmlu_accuracy(model, tokenizer)
            record.update(mmlu_result)

        elapsed = time.time() - t0
        ppl_str = f"Perplexity: {ppl_result['mean_perplexity']:.2f}"
        mmlu_str = ""
        if "mmlu_accuracy" in record:
            mmlu_str = f", MMLU: {record['mmlu_accuracy']:.3f}"
        print(f"         {ppl_str}{mmlu_str} ({elapsed:.1f}s)")

        records.append(record)
        with open(local_tmp, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Restore base weights at the end
    for param_name, base_data in base_state.items():
        param_dict[param_name].data.copy_(base_data)

    # Copy local temp to final output (works on S3 FUSE as a single write)
    # Use shutil.copyfile — FUSE rejects chmod/utime (copy/copy2 both fail)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(local_tmp), str(output_jsonl))
    local_tmp.unlink()
    print(f"\n  Results written to: {output_jsonl}")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Offline capability probing on failure curve checkpoints"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed used for the failure curve run")
    parser.add_argument("--alg_name", required=True,
                        help="Algorithm whose checkpoints to probe (AlphaEdit, MEMIT, or MEMIT-Seq-lp{X}-ld{Y}-cache{Z})")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Explicit checkpoint directory (default: auto-resolve)")
    parser.add_argument("--model_name",
                        default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"),
                        help="Model to load as base")
    parser.add_argument("--no_mmlu", action="store_true",
                        help="Skip MMLU evaluation (perplexity only, faster)")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: results/capability_probe/offline_seed{seed}_{alg}.jsonl)")

    args = parser.parse_args()

    project_root = get_project_root()

    # Resolve checkpoint directory
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.alg_name, args.seed)

    # Determine max edit count from checkpoints
    checkpoints = find_all_checkpoints(ckpt_dir)
    if checkpoints:
        last_batch_dir = checkpoints[-1][1]
        with open(last_batch_dir / "metadata.json", "r") as f:
            last_meta = json.load(f)
        max_edits = last_meta.get("total_edits", (checkpoints[-1][0] + 1) * 100)
    else:
        max_edits = 0

    # Resolve output path
    if args.output:
        output_jsonl = Path(args.output)
    else:
        output_dir = get_result_root() / "capability_probe" / f"seed{args.seed}" / f"{max_edits}edits" / args.alg_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_jsonl = output_dir / f"offline_probe_{timestamp}.jsonl"

    print(f"{'=' * 70}")
    print("Offline Capability Probe")
    print(f"  Seed:           {args.seed}")
    print(f"  Algorithm:      {args.alg_name}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"  Model:          {args.model_name}")
    print(f"  MMLU:           {'yes' if not args.no_mmlu else 'no (perplexity only)'}")
    print(f"  Output:         {output_jsonl}")
    print(f"  Started:        {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    if not ckpt_dir.exists():
        print(f"\nERROR: Checkpoint directory does not exist: {ckpt_dir}")
        print("  Run the failure curve first:")
        print(f"    EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_checkpointed.sh {args.seed} {args.alg_name} 10000")
        sys.exit(1)

    # Load model once with consistent dtype
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ensure model is downloaded (uses Artifactory endpoint + HF_TOKEN if available)
    from huggingface_hub import snapshot_download
    resolve_model_path(args.model_name)  # sets HF_ENDPOINT if on corporate infra
    token = os.environ.get("HF_TOKEN")
    print(f"  Ensuring model is downloaded: {args.model_name}")
    snapshot_download(
        repo_id=args.model_name,
        token=token,
        endpoint=os.environ.get("HF_ENDPOINT"),
    )
    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loading model weights (dtype={PROBE_DTYPE})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=token, torch_dtype=PROBE_DTYPE,
    ).cuda(args.cuda_device)
    print(f"  Model loaded: {args.model_name} ({PROBE_DTYPE})")

    # Run probes
    records = run_offline_probes(
        model=model,
        tokenizer=tokenizer,
        ckpt_dir=ckpt_dir,
        output_jsonl=output_jsonl,
        compute_mmlu=not args.no_mmlu,
        seed=args.seed,
        alg_name=args.alg_name,
    )

    print(f"\n{'=' * 70}")
    print("Offline capability probe complete.")
    print(f"  Checkpoints probed: {len(records) - 1} + baseline")
    print(f"  Output: {output_jsonl}")
    if records:
        baseline_ppl = records[0]["mean_perplexity"]
        final_ppl = records[-1]["mean_perplexity"]
        print(f"  Perplexity: {baseline_ppl:.2f} (baseline) → {final_ppl:.2f} (final)")
        if "mmlu_accuracy" in records[0]:
            baseline_mmlu = records[0]["mmlu_accuracy"]
            final_mmlu = records[-1]["mmlu_accuracy"]
            print(f"  MMLU:       {baseline_mmlu:.3f} (baseline) → {final_mmlu:.3f} (final)")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
