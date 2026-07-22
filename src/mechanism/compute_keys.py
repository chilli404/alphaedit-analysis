#!/usr/bin/env python3
"""Compute key vectors for all CounterFact edits using the base model.

Keys are the INPUT representations to `model.layers.{L}.mlp.down_proj`
at the last subject token position — exactly as AlphaEdit's compute_ks.py
computes them during editing.

This script runs a single forward pass per edit (no model editing needed).
The resulting key vectors enable Tier 2 geometric interference analysis
in analysis/interference_panel.py.

Output:
    {output_dir}/keys_seed{seed}.npz containing:
      - case_ids: int array of shape (N,)
      - keys: float32 array of shape (N, hidden_dim)
      - layer: int (which layer was used)
      - metadata: dict with model_name, n_cases, timestamp

Requirements:
    - GPU with ~16GB VRAM (Llama-3-8B in float16)
    - ~30 minutes for 10K edits

Usage:
    uv run python -m src.mechanism.compute_keys --seed 42
    uv run python -m src.mechanism.compute_keys --seed 42 --layer 5
    uv run python -m src.mechanism.compute_keys --seed 42 2024 --output-dir results/key_vectors
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent.parent
VENDOR = PROJECT / "vendor" / "AlphaEdit"
RESULTS = PROJECT / "results"
FC_DIR = RESULTS / "failure_curve_checkpointed"

# Add src/util to path for resolve_model_path
sys.path.insert(0, str(PROJECT / "src" / "util"))

# AlphaEdit uses layers [4, 5, 6, 7, 8] for Llama-3-8B
# We use the middle layer (6) by default as representative
DEFAULT_LAYER = 6
EDIT_LAYERS = [4, 5, 6, 7, 8]

# Module template for Llama-3
MODULE_TEMPLATE = "model.layers.{}.mlp.down_proj"

# Model — same default as all experiment scripts
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")


# ─── Key Extraction ──────────────────────────────────────────────────────────


class KeyExtractor:
    """Extract key vectors (input to down_proj) at subject's last token."""

    def __init__(self, model, tokenizer, layer: int):
        self.model = model
        self.tok = tokenizer
        self.layer = layer
        self._captured = None

        # Register hook on the target module
        module = self._get_module(layer)
        module.register_forward_hook(self._hook)

    def _get_module(self, layer: int) -> nn.Module:
        """Navigate to model.layers.{layer}.mlp.down_proj."""
        return self.model.model.layers[layer].mlp.down_proj

    def _hook(self, module, input, output):
        """Capture the INPUT to down_proj."""
        # input is a tuple; first element is the hidden states
        self._captured = input[0].detach()

    def extract_key(self, prompt: str, subject: str) -> Optional[np.ndarray]:
        """Extract the key vector for one edit.

        Args:
            prompt: The edit prompt template with {} for subject, already formatted
            subject: The subject entity

        Returns:
            Key vector of shape (hidden_dim,) or None if extraction fails.
        """
        # Format prompt with subject
        text = prompt.replace("{}", subject)

        # Tokenize
        inputs = self.tok(text, return_tensors="pt", padding=False).to(self.model.device)
        input_ids = inputs["input_ids"][0]

        # Find the last token position of the subject
        subject_token_pos = self._find_subject_last_token(text, subject, input_ids)
        if subject_token_pos is None:
            return None

        # Forward pass (no grad needed)
        self._captured = None
        with torch.no_grad():
            self.model(**inputs)

        if self._captured is None:
            return None

        # Extract at subject position: shape (1, seq_len, hidden_dim) → (hidden_dim,)
        key = self._captured[0, subject_token_pos].cpu().numpy().astype(np.float32)
        return key

    def _find_subject_last_token(self, text: str, subject: str, input_ids: torch.Tensor) -> Optional[int]:
        """Find the last token position of the subject in the input."""
        # Strategy: tokenize the text up to the end of the subject,
        # the last token of that prefix corresponds to the subject's last token
        subj_start = text.find(subject)
        if subj_start == -1:
            # Subject not found in text
            return None
        subj_end = subj_start + len(subject)

        prefix = text[:subj_end]
        prefix_ids = self.tok(prefix, return_tensors="pt", padding=False)["input_ids"][0]
        # Last token of prefix = last subject token
        return len(prefix_ids) - 1


# ─── Dataset Loading ─────────────────────────────────────────────────────────


def load_edit_ordering(seed: int) -> Optional[List[int]]:
    """Load the exact case_id ordering for a trajectory."""
    for edits in [10000, 9000, 7000, 5000, 3000, 2000]:
        path = FC_DIR / f"seed{seed}" / f"{edits}edits" / "AlphaEdit" / "run_000" / "edit_ordering.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)["case_ids_ordered"]
    return None


def load_case_metadata(seed: int) -> dict:
    """Load prompt/subject for each case_id from per-case result files."""
    metadata = {}
    for edits_dir in sorted(FC_DIR.glob(f"seed{seed}/*edits")):
        run_dir = edits_dir / "AlphaEdit" / "run_000"
        if not run_dir.exists():
            continue
        for f_path in run_dir.glob("*_edits-case_*.json"):
            with open(f_path) as f:
                data = json.load(f)
            cid = data["case_id"]
            if cid not in metadata:
                rewrite = data.get("requested_rewrite", {})
                metadata[cid] = {
                    "prompt": rewrite.get("prompt", ""),
                    "subject": rewrite.get("subject", ""),
                    "relation_id": rewrite.get("relation_id", ""),
                }
    return metadata


# ─── Main ────────────────────────────────────────────────────────────────────


def compute_keys_for_seed(
    seed: int,
    model,
    tokenizer,
    layer: int,
    output_dir: Path,
    max_cases: Optional[int] = None,
):
    """Compute and save key vectors for all edits in a trajectory."""
    print(f"\n{'='*50}")
    print(f"Computing keys for seed {seed}, layer {layer}")
    print(f"{'='*50}")

    # Load ordering and metadata
    ordering = load_edit_ordering(seed)
    if not ordering:
        print(f"  ERROR: No edit_ordering.json for seed {seed}")
        return

    metadata = load_case_metadata(seed)
    if not metadata:
        print(f"  ERROR: No case metadata for seed {seed}")
        return

    print(f"  Ordering: {len(ordering)} edits")
    print(f"  Metadata: {len(metadata)} cases")

    # Filter to cases with metadata
    valid_cases = [cid for cid in ordering if cid in metadata]
    if max_cases:
        valid_cases = valid_cases[:max_cases]
    print(f"  Computing keys for {len(valid_cases)} cases")

    # Extract keys
    extractor = KeyExtractor(model, tokenizer, layer)
    case_ids = []
    keys = []
    failed = 0

    t0 = time.time()
    for i, cid in enumerate(valid_cases):
        meta = metadata[cid]
        prompt = meta["prompt"]
        subject = meta["subject"]

        if not prompt or not subject:
            failed += 1
            continue

        key = extractor.extract_key(prompt, subject)
        if key is not None:
            case_ids.append(cid)
            keys.append(key)
        else:
            failed += 1

        # Progress
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(valid_cases) - i - 1) / rate
            print(f"  [{i+1}/{len(valid_cases)}] {rate:.1f} cases/sec, ETA {eta:.0f}s "
                  f"({failed} failed)")

    elapsed = time.time() - t0
    print(f"  Done: {len(keys)} keys extracted in {elapsed:.1f}s ({failed} failed)")

    if not keys:
        print("  ERROR: No keys extracted")
        return

    # Save
    seed_dir = output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = seed_dir / f"keys_seed{seed}.npz"

    np.savez_compressed(
        out_path,
        case_ids=np.array(case_ids, dtype=np.int32),
        keys=np.stack(keys, axis=0),
        layer=np.array(layer),
    )

    # Also save metadata JSON
    meta_path = seed_dir / f"keys_seed{seed}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "seed": seed,
            "layer": layer,
            "model_name": MODEL_NAME,
            "n_cases": len(keys),
            "n_failed": failed,
            "hidden_dim": keys[0].shape[0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
        }, f, indent=2)

    print(f"  Saved: {out_path} ({len(keys)} × {keys[0].shape[0]})")
    print(f"  Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute base-model key vectors for interference analysis"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[42, 2024],
                        help="Seeds to compute keys for")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help=f"Layer to extract keys from (default: {DEFAULT_LAYER})")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "key_vectors",
                        help="Output directory for .npz files")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases (for testing)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name or path (default: $MODEL_NAME or Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Resolve model path (handles Artifactory auth on corporate infra)
    from model_download import resolve_model_path
    model_id = resolve_model_path(args.model)

    # Download model via Artifactory endpoint if on corporate infra
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT")
    print(f"Ensuring model is downloaded: {model_id}")
    snapshot_download(
        repo_id=model_id,
        token=token,
        endpoint=endpoint,
    )

    print(f"Loading model: {model_id}")
    print(f"Layer: {args.layer} (AlphaEdit edit layers: {EDIT_LAYERS})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded on {args.device}")

    # Compute keys for each seed
    for seed in args.seed:
        compute_keys_for_seed(
            seed=seed,
            model=model,
            tokenizer=tokenizer,
            layer=args.layer,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
        )

    print(f"\nAll done. Use with:")
    print(f"  uv run python -m analysis.interference_panel --keys-dir {args.output_dir}")


if __name__ == "__main__":
    main()
