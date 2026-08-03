#!/usr/bin/env python3
"""
Build covariance statistics and null-space projectors for a model.

Wraps vendor/AlphaEdit/rome/layer_stats.py to compute the second-moment matrix
(mom2) from Wikipedia text for each edit layer. After all layers are computed,
optionally builds the null-space projector P via SVD and reports the retained-
dimension fraction per layer.

Resumable: skips layers whose .npz file already exists.

Usage:
    # Full computation (requires GPU)
    uv run python scripts/build_stats.py --model Qwen/Qwen2.5-7B-Instruct

    # Specific layers only
    uv run python scripts/build_stats.py --model EleutherAI/gpt-j-6b --layers 3 4 5 6 7 8

    # Verify existing stats + compute retained-dim report (no new stats)
    uv run python scripts/build_stats.py --model Qwen/Qwen2.5-7B-Instruct --verify_only

    # Skip projector computation (stats only)
    uv run python scripts/build_stats.py --model EleutherAI/gpt-j-6b --no_projector
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "AlphaEdit"

sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))

# Monkey-patch datasets.load_dataset to fix deprecated Wikipedia config.
# The vendor code (rome/layer_stats.py:103) hardcodes "20200501.en" which
# HuggingFace has removed. The equivalent is now "20220301.en".
import datasets as _datasets
_original_load_dataset = _datasets.load_dataset


def _patched_load_dataset(path, name=None, *args, **kwargs):
    if name == "20200501.en":
        name = "20220301.en"
    return _original_load_dataset(path, name, *args, **kwargs)


_datasets.load_dataset = _patched_load_dataset

from model_registry import get_model_spec


def load_hparams(model_spec, alg="AlphaEdit"):
    """Load hparams for the model from the project configs or vendor hparams."""
    from util.hparams import HyperParams

    # Check project configs first
    project_hparams = PROJECT_DIR / "configs" / "hparams" / alg / model_spec.hparams_fname
    if project_hparams.exists():
        return HyperParams.from_json(project_hparams)

    # Fall back to vendor hparams
    vendor_hparams = VENDOR_DIR / "hparams" / alg / model_spec.hparams_fname
    if vendor_hparams.exists():
        return HyperParams.from_json(vendor_hparams)

    raise FileNotFoundError(
        f"No hparams found for {model_spec.short_name} at:\n"
        f"  {project_hparams}\n"
        f"  {vendor_hparams}"
    )


def compute_stats_for_layer(model, tok, layer_name, stats_dir, hparams, sample_size=100000):
    """Compute covariance stats for a single layer using the vendor layer_stats code."""
    from rome.layer_stats import layer_stats

    print(f"\n{'='*60}")
    print(f"Computing stats for: {layer_name}")
    print(f"  Output dir: {stats_dir}")
    print(f"  Samples: {sample_size}")
    print(f"{'='*60}")

    start = time.time()

    stat = layer_stats(
        model,
        tok,
        layer_name,
        stats_dir,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=sample_size,
        precision=hparams.mom2_dtype,
        batch_tokens=None,
        progress=lambda x, **kw: x,  # No tqdm in batch mode
    )

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return stat


def compute_projector(cov_matrix, threshold=0.02):
    """Compute null-space projector from covariance via SVD.

    Returns:
        P: The null-space projector matrix (d x d)
        info: Dict with retained_dims, total_dims, fraction
    """
    U, S, _ = torch.linalg.svd(cov_matrix.float(), full_matrices=False)
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    total_dims = S.shape[0]
    retained_dims = small_singular_indices.shape[0]

    P = U[:, small_singular_indices] @ U[:, small_singular_indices].T

    info = {
        "total_dims": total_dims,
        "retained_dims": retained_dims,
        "fraction": retained_dims / total_dims,
        "threshold": threshold,
        "min_singular": S.min().item(),
        "max_singular": S.max().item(),
        "n_above_threshold": (S >= threshold).sum().item(),
    }

    return P, info


def check_existing_stats(stats_dir, layer_name, precision="float32", sample_size=100000):
    """Check if stats already exist for a layer."""
    size_suffix = f"_{sample_size}"
    filename = stats_dir / f"{layer_name}_{precision}_mom2{size_suffix}.npz"
    return filename.exists(), filename


def main():
    parser = argparse.ArgumentParser(description="Build covariance statistics for a model")
    parser.add_argument("--model", required=True, help="Model name or HuggingFace repo ID")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to compute (default: all edit layers from hparams)")
    parser.add_argument("--sample_size", type=int, default=100000,
                        help="Number of Wikipedia samples (default: 100000)")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Null-space SVD threshold tau (default: 0.02)")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing stats and compute projector report")
    parser.add_argument("--no_projector", action="store_true",
                        help="Skip null-space projector computation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: data/stats/{model}/wikipedia_stats)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Resolve model
    spec = get_model_spec(args.model)
    print(f"Model: {spec.short_name} ({spec.hf_repo})")
    print(f"  Hidden size: {spec.hidden_size}")
    print(f"  Layers: {spec.n_layers}")
    print(f"  Edit layers: {list(spec.edit_layers)}")

    # Determine output directory
    if args.output_dir:
        stats_dir = Path(args.output_dir)
    else:
        stats_dir = PROJECT_DIR / "data" / "stats" / spec.stats_dir_name / "wikipedia_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Determine which layers to process
    layers = args.layers if args.layers else list(spec.edit_layers)
    print(f"  Target layers: {layers}")

    # Load hparams to get module path templates
    # We parse the JSON directly since we don't need the full dataclass
    hparams_path = PROJECT_DIR / "configs" / "hparams" / "AlphaEdit" / spec.hparams_fname
    if not hparams_path.exists():
        hparams_path = VENDOR_DIR / "hparams" / "AlphaEdit" / spec.hparams_fname
    with open(hparams_path) as f:
        hparams_dict = json.load(f)

    rewrite_module_tmp = hparams_dict["rewrite_module_tmp"]
    layer_names = [rewrite_module_tmp.format(l) for l in layers]

    # Check existing stats
    print(f"\nChecking existing stats in: {stats_dir}")
    missing_layers = []
    for layer, layer_name in zip(layers, layer_names):
        exists, path = check_existing_stats(stats_dir, layer_name, "float32", args.sample_size)
        status = "EXISTS" if exists else "MISSING"
        print(f"  Layer {layer} ({layer_name}): {status}")
        if not exists:
            missing_layers.append((layer, layer_name))

    if args.verify_only:
        if missing_layers:
            print(f"\nERROR: {len(missing_layers)} layers missing stats. "
                  f"Run without --verify_only to compute them.")
            sys.exit(1)
        print("\nAll stats files present.")
    else:
        if not missing_layers:
            print("\nAll stats already computed. Skipping to projector.")
        else:
            print(f"\nWill compute stats for {len(missing_layers)} layers.")

            # Set seeds
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            # Load model
            print(f"\nLoading model: {spec.hf_repo}")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            tok = AutoTokenizer.from_pretrained(spec.hf_repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"  Model loaded. Device: {next(model.parameters()).device}")

            # Verify module paths exist
            print("\nVerifying module paths...")
            for layer, layer_name in missing_layers:
                try:
                    param = None
                    for name, p in model.named_parameters():
                        if layer_name in name:
                            param = p
                            break
                    assert param is not None, f"Module {layer_name} not found in model"
                    print(f"  Layer {layer} ({layer_name}): OK "
                          f"[shape={list(param.shape)}]")
                except AssertionError as e:
                    print(f"  Layer {layer} ({layer_name}): FAILED - {e}")
                    print("\nAborting. Check module path templates in hparams.")
                    sys.exit(1)

            # Compute stats for missing layers
            # We need to set the STATS_DIR that layer_stats expects
            # The function builds its own path: stats_dir / model_name / ds_name_stats / ...
            # So we pass the parent of where we want files to land
            vendor_stats_dir = stats_dir.parent.parent  # data/stats/ (layer_stats adds model/ds_stats/)
            os.chdir(str(VENDOR_DIR))  # layer_stats uses relative paths internally

            for layer, layer_name in missing_layers:
                compute_stats_for_layer(
                    model, tok, layer_name,
                    str(vendor_stats_dir),
                    type("HParams", (), {
                        "mom2_dataset": hparams_dict.get("mom2_dataset", "wikipedia"),
                        "mom2_dtype": hparams_dict.get("mom2_dtype", "float32"),
                    })(),
                    sample_size=args.sample_size,
                )

            del model
            torch.cuda.empty_cache()

    # Compute null-space projector and retained-dimension report
    if not args.no_projector:
        print(f"\n{'='*60}")
        print("Computing null-space projectors and retained-dimension report")
        print(f"  Threshold tau: {args.threshold}")
        print(f"{'='*60}")

        report = {
            "model": spec.short_name,
            "hf_repo": spec.hf_repo,
            "threshold": args.threshold,
            "sample_size": args.sample_size,
            "hidden_size": spec.hidden_size,
            "layers": {},
        }

        P_tensors = []
        for layer, layer_name in zip(layers, layer_names):
            _, stats_path = check_existing_stats(stats_dir, layer_name, "float32", args.sample_size)
            if not stats_path.exists():
                print(f"  Layer {layer}: stats missing, skipping projector")
                continue

            # Load the covariance matrix
            data = np.load(stats_path)
            # The mom2 stat stores the second moment; we need to get it
            # layer_stats saves via CombinedStat which uses .npz with specific keys
            if "mom2" in data:
                cov = torch.from_numpy(data["mom2"])
            else:
                # Fallback: try loading the full stat object
                print(f"  Layer {layer}: unexpected npz format, keys={list(data.keys())}")
                continue

            P, info = compute_projector(cov, threshold=args.threshold)
            P_tensors.append(P)

            report["layers"][str(layer)] = info
            print(f"  Layer {layer}: retained {info['retained_dims']}/{info['total_dims']} "
                  f"dims ({info['fraction']:.4f}), "
                  f"singular range [{info['min_singular']:.4e}, {info['max_singular']:.4e}]")

        # Save report
        report_path = stats_dir / "retained_dims_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {report_path}")

        # Save stacked P tensor
        if P_tensors:
            P_stacked = torch.stack(P_tensors)
            p_path = stats_dir / "null_space_project.pt"
            torch.save(P_stacked, str(p_path))
            print(f"  Projector saved to: {p_path} [shape={list(P_stacked.shape)}]")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
