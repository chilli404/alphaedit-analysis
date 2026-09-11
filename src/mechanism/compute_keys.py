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
RESULTS = Path(os.environ.get("RESULT_ROOT", PROJECT / "results"))
FC_DIR = RESULTS / "failure_curve_checkpointed"


# Model — same default as all experiment scripts
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# Per-architecture config: (edit_layers, default_layer, module_path_fn)
ARCH_CONFIG = {
    "llama": {
        "edit_layers": [4, 5, 6, 7, 8],
        "default_layer": 6,
    },
    "gptj": {
        "edit_layers": [3, 4, 5, 6, 7, 8],
        "default_layer": 6,
    },
    "qwen": {
        "edit_layers": [4, 5, 6, 7, 8],
        "default_layer": 6,
    },
}


def detect_architecture(model_name: str) -> str:
    """Detect model architecture from name."""
    name_lower = model_name.lower()
    if "gpt-j" in name_lower or "gptj" in name_lower:
        return "gptj"
    if "qwen" in name_lower:
        return "qwen"
    return "llama"


def get_edit_layers(model_name: str) -> list:
    """Get edit layers for a model."""
    arch = detect_architecture(model_name)
    return ARCH_CONFIG[arch]["edit_layers"]


def get_default_layer(model_name: str) -> int:
    """Get default layer for a model."""
    arch = detect_architecture(model_name)
    return ARCH_CONFIG[arch]["default_layer"]


# Legacy constants for backward compat
DEFAULT_LAYER = get_default_layer(MODEL_NAME)
EDIT_LAYERS = get_edit_layers(MODEL_NAME)


# ─── Key Extraction ──────────────────────────────────────────────────────────


class KeyExtractor:
    """Extract key vectors (input to MLP output projection) at subject's last token."""

    def __init__(self, model, tokenizer, layer: int):
        self.model = model
        self.tok = tokenizer
        self.layer = layer
        self._captured = None

        # Register hook on the target module
        module = self._get_module(layer)
        module.register_forward_hook(self._hook)

    def _get_module(self, layer: int) -> nn.Module:
        """Navigate to the MLP output projection for the given layer.

        Llama-3: model.model.layers.{L}.mlp.down_proj
        GPT-J:   model.transformer.h.{L}.mlp.fc_out
        Qwen:    model.model.layers.{L}.mlp.down_proj
        """
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer].mlp.fc_out
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


def load_edit_ordering(seed: int, min_cases: int = 10000) -> Optional[List[int]]:
    """Load the case_id ordering for a trajectory.

    Prefers orderings with at least min_cases. Falls back to full MCF dataset
    (all 20K+ cases) for key extraction where we want maximum coverage.
    """
    for edits in [10000, 9000, 7000, 5000, 3000, 2000]:
        path = FC_DIR / f"seed{seed}" / f"{edits}edits" / "AlphaEdit" / "run_000" / "edit_ordering.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            ordering = data["case_ids_ordered"]
            if len(ordering) >= min_cases:
                print(f"  Ordering from {path}: {len(ordering)} case IDs")
                return ordering
            else:
                print(f"  Found {path} but only {len(ordering)} cases (need {min_cases})")

    # Use all MCF case IDs — for key extraction we want full coverage
    print(f"  Using full MCF dataset as ordering (all cases)")
    metadata = load_case_metadata(seed)
    if metadata:
        ordering = list(metadata.keys())
        print(f"  Full MCF: {len(ordering)} case IDs")
        return ordering
    return None


def load_case_metadata(seed: int) -> dict:
    """Load prompt/subject for each case_id from MCF dataset (fast) or per-case files (slow)."""
    metadata = {}

    # Prefer MCF dataset (single file read — fast on S3 FUSE)
    for mcf_path in [
        VENDOR / "data" / "multi_counterfact.json",
        PROJECT / "data" / "dsets" / "multi_counterfact.json",
    ]:
        if mcf_path.exists():
            print(f"  Loading metadata from {mcf_path}...")
            with open(mcf_path) as f:
                mcf_data = json.load(f)
            for record in mcf_data:
                cid = record["case_id"]
                rw = record.get("requested_rewrite", {})
                metadata[cid] = {
                    "prompt": rw.get("prompt", ""),
                    "subject": rw.get("subject", ""),
                    "relation_id": rw.get("relation_id", ""),
                }
            print(f"  Loaded {len(metadata)} cases from MCF dataset")
            return metadata

    # Fallback: per-case result files (slow on S3 FUSE — thousands of small reads)
    print("  MCF dataset not found, falling back to per-case files (slow)...")
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
    print(f"  Loaded {len(metadata)} cases from per-case files")
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

    # Save — write to /tmp first then copy (S3 FUSE can't handle compressed writes)
    import tempfile

    seed_dir = output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = seed_dir / f"keys_seed{seed}.npz"

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    np.savez_compressed(
        tmp_path,
        case_ids=np.array(case_ids, dtype=np.int32),
        keys=np.stack(keys, axis=0),
        layer=np.array(layer),
    )
    with open(tmp_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    tmp_path.unlink()
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

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

    model_id = os.environ.get("MODEL_PATH", args.model)
    token = os.environ.get("HF_TOKEN")
    arch = detect_architecture(model_id)
    edit_layers = ARCH_CONFIG[arch]["edit_layers"]

    if not os.path.isdir(model_id):
        print(f"Model will be downloaded by transformers: {model_id}")
    else:
        print(f"Model already local: {model_id}")

    print(f"Loading model: {model_id}")
    print(f"Architecture: {arch}, Layer: {args.layer} (edit layers: {edit_layers})")

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
