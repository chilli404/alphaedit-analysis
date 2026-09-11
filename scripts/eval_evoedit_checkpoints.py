#!/usr/bin/env python3
"""Prepare EvoEdit checkpoints for eval and run eval_matched_ordering.py.

EvoEdit saves checkpoints as edits_001000/model_weights.pt (our lightweight format).
This script creates the batch_N/ directory structure that eval_matched_ordering.py
expects, then runs eval.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def prepare_checkpoints(ckpt_base: Path, output_dir: Path):
    """Create batch_N/ symlinks from edits_NNNNNN/ checkpoint dirs."""
    for ckpt_path in sorted(ckpt_base.glob("edits_*")):
        edits = int(ckpt_path.name.split("_")[1])
        batch_idx = (edits // 100) - 1

        out_dir = output_dir / f"batch_{batch_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Link model_weights.pt
        src = ckpt_path / "model_weights.pt"
        dst = out_dir / "model_weights.pt"
        if src.exists() and not dst.exists():
            os.symlink(str(src.resolve()), str(dst))

        # Create metadata.json
        meta_path = out_dir / "metadata.json"
        if not meta_path.exists():
            meta = {"batch_idx": batch_idx, "total_edits": edits, "source": "evoedit"}
            with open(str(meta_path), "w") as f:
                json.dump(meta, f, indent=2)

        print(f"  {ckpt_path.name} -> batch_{batch_idx} ({edits} edits)")

    print(f"\nPrepared checkpoints at: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ordering", type=str, required=True)
    parser.add_argument("--result_root", type=str, default=os.environ.get("RESULT_ROOT", "results"))
    args = parser.parse_args()

    ckpt_base_dir = Path(args.result_root) / "evoedit" / args.ordering / f"seed{args.seed}" / "10000edits" / "EvoEdit"

    run_dirs = sorted(ckpt_base_dir.glob("run_*/checkpoints"))
    if not run_dirs:
        print(f"ERROR: No checkpoint dirs found at {ckpt_base_dir}")
        sys.exit(1)

    latest_ckpts = run_dirs[-1]
    print(f"Using: {latest_ckpts.parent}")
    print(f"Checkpoints: {sorted(d.name for d in latest_ckpts.iterdir() if d.is_dir())}")

    prepared_dir = Path(f"/tmp/evoedit_prepared/{args.ordering}")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepare_checkpoints(latest_ckpts, prepared_dir)

    batches = sorted(
        int(d.name.split("_")[1])
        for d in prepared_dir.iterdir()
        if d.is_dir() and d.name.startswith("batch_")
    )
    print(f"\nRunning eval on batches: {batches}")

    stream_path = Path(args.result_root) / "matched_ordering" / "orderings" / f"{args.ordering}_seed{args.seed}.json"

    cmd = [
        sys.executable, "scripts/eval_matched_ordering.py",
        "--seed", str(args.seed),
        "--alg_name", "EvoEdit",
        "--ordering", args.ordering,
        "--checkpoints", *[str(b) for b in batches],
        "--num_edits", "100",
        "--checkpoint_dir", str(prepared_dir),
        "--dataset_path", str(stream_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
