#!/usr/bin/env python3
"""
Weight-Norm and Spectral-Drift Control (Step 3):

Computes per-layer weight drift metrics from existing checkpoints to
determine whether late capability collapse is driven by weight damage
(norm growth) vs cache geometry (null-space exhaustion).

For each edited layer at each checkpoint:
  - Relative Frobenius drift: ||W_t - W_0||_F / ||W_0||_F
  - Spectral norm of cumulative change: σ_max(W_t - W_0)
  - Relative total norm: ||W_t||_F / ||W_0||_F
  - Top singular-value changes
  - Stable rank of the perturbation
  - Dominant-subspace angle between W_0 and W_t

Ideal outcome: Cache-spectrum tracks edit forgetting, weight deformation
tracks general-capability collapse. Even a negative result rules out a
major alternative explanation.

Usage:
    # From controlled coupling checkpoints (remote rig)
    uv run python analysis/weight_drift_analysis.py \
        --checkpoint_base results/controlled_coupling/checkpoints \
        --streams low_coupling,high_coupling --seed 42

    # From failure curve checkpoints (S3 or local)
    uv run python analysis/weight_drift_analysis.py \
        --checkpoint_base /s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed42 \
        --mode failure_curve

    # From comparison_ordered checkpoints
    uv run python analysis/weight_drift_analysis.py \
        --checkpoint_base /s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed42/order0 \
        --mode failure_curve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_checkpoints(base_dir: Path) -> list[Path]:
    """Find all batch_N directories sorted by batch index."""
    if not base_dir.exists():
        return []
    batch_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    return [d for d in batch_dirs if (d / "model_weights.pt").exists()]


def load_base_weights(model_name: str, layers: list[int], module_template: str = "model.layers.{}.mlp.down_proj"):
    """Load base model weights for comparison."""
    from transformers import AutoModelForCausalLM
    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    base_weights = {}
    for name, param in model.named_parameters():
        for layer in layers:
            expected = module_template.format(layer) + ".weight"
            if expected.replace(".", "_") in name.replace(".", "_") or name == expected:
                base_weights[name] = param.detach().cpu().clone()
                break

    # Also try direct matching
    if not base_weights:
        for name, param in model.named_parameters():
            if "down_proj" in name and any(f"layers.{l}." in name for l in layers):
                base_weights[name] = param.detach().cpu().clone()

    del model
    torch.cuda.empty_cache()
    print(f"  Loaded {len(base_weights)} base weight tensors")
    return base_weights


def compute_drift_metrics(W_base: torch.Tensor, W_t: torch.Tensor) -> dict:
    """Compute all weight drift metrics for a single layer."""
    delta = W_t - W_base

    # Frobenius norms
    base_fro = torch.linalg.norm(W_base, ord="fro").item()
    delta_fro = torch.linalg.norm(delta, ord="fro").item()
    current_fro = torch.linalg.norm(W_t, ord="fro").item()

    # Relative Frobenius drift
    rel_fro_drift = delta_fro / max(base_fro, 1e-10)

    # Spectral norm of change (largest singular value of delta)
    # Use a few iterations of power method for speed on large matrices
    try:
        svs_delta = torch.linalg.svdvals(delta.float())
        spectral_norm_delta = svs_delta[0].item()
        # Stable rank of perturbation
        stable_rank_delta = (delta_fro ** 2) / max(spectral_norm_delta ** 2, 1e-10)
        # Top-5 SVs of delta
        top5_svs_delta = svs_delta[:5].tolist()
    except Exception:
        spectral_norm_delta = float("nan")
        stable_rank_delta = float("nan")
        top5_svs_delta = []

    # Top SV changes in the full weight
    try:
        svs_base = torch.linalg.svdvals(W_base.float())[:10]
        svs_current = torch.linalg.svdvals(W_t.float())[:10]
        top_sv_base = svs_base[0].item()
        top_sv_current = svs_current[0].item()
        top_sv_change = (top_sv_current - top_sv_base) / max(top_sv_base, 1e-10)
    except Exception:
        top_sv_base = float("nan")
        top_sv_current = float("nan")
        top_sv_change = float("nan")

    # Dominant subspace angle (angle between top-k singular subspaces)
    try:
        k = min(5, min(W_base.shape) // 2)
        U_base = torch.linalg.svd(W_base.float(), full_matrices=False)[0][:, :k]
        U_current = torch.linalg.svd(W_t.float(), full_matrices=False)[0][:, :k]
        # Principal angle via SVD of U_base^T @ U_current
        cos_angles = torch.linalg.svdvals(U_base.T @ U_current)
        cos_angles = cos_angles.clamp(-1, 1)
        angles_deg = torch.acos(cos_angles).rad2deg()
        max_subspace_angle = angles_deg.max().item()
        mean_subspace_angle = angles_deg.mean().item()
    except Exception:
        max_subspace_angle = float("nan")
        mean_subspace_angle = float("nan")

    return {
        "base_frobenius": round(base_fro, 4),
        "delta_frobenius": round(delta_fro, 6),
        "current_frobenius": round(current_fro, 4),
        "relative_frobenius_drift": round(rel_fro_drift, 6),
        "relative_total_norm": round(current_fro / max(base_fro, 1e-10), 6),
        "spectral_norm_delta": round(spectral_norm_delta, 6),
        "stable_rank_delta": round(stable_rank_delta, 2),
        "top5_svs_delta": [round(v, 6) for v in top5_svs_delta],
        "top_sv_base": round(top_sv_base, 4),
        "top_sv_current": round(top_sv_current, 4),
        "top_sv_relative_change": round(top_sv_change, 6),
        "max_subspace_angle_deg": round(max_subspace_angle, 4),
        "mean_subspace_angle_deg": round(mean_subspace_angle, 4),
    }


def analyze_controlled_coupling(args):
    """Analyze weight drift for controlled coupling checkpoints."""
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    layers = [4, 5, 6, 7, 8]  # AlphaEdit edited layers for Llama-3-8B
    base_weights = load_base_weights(model_name, layers)

    streams = args.streams.split(",")
    all_results = {}

    for stream_name in streams:
        ckpt_base = Path(args.checkpoint_base) / stream_name / f"seed{args.seed}"
        checkpoints = find_checkpoints(ckpt_base)

        if not checkpoints:
            print(f"  SKIP {stream_name}: no checkpoints at {ckpt_base}")
            continue

        print(f"\n  Analyzing {stream_name} ({len(checkpoints)} checkpoints)...")
        stream_results = []

        for ckpt_dir in checkpoints:
            batch_idx = int(ckpt_dir.name.split("_")[1])
            total_edits = (batch_idx + 1) * 100

            weights = torch.load(ckpt_dir / "model_weights.pt", map_location="cpu")

            batch_metrics = {
                "batch_idx": batch_idx,
                "total_edits": total_edits,
                "layers": {},
            }

            for wname, w_t in weights.items():
                if wname in base_weights:
                    layer_num = None
                    for l in layers:
                        if f"layers.{l}." in wname:
                            layer_num = l
                            break

                    metrics = compute_drift_metrics(base_weights[wname], w_t)
                    batch_metrics["layers"][str(layer_num)] = metrics

            # Aggregate across layers
            layer_drifts = [m["relative_frobenius_drift"] for m in batch_metrics["layers"].values()]
            layer_spectral = [m["spectral_norm_delta"] for m in batch_metrics["layers"].values()]
            layer_angles = [m["max_subspace_angle_deg"] for m in batch_metrics["layers"].values()
                          if not np.isnan(m["max_subspace_angle_deg"])]

            batch_metrics["aggregate"] = {
                "mean_relative_frobenius_drift": round(float(np.mean(layer_drifts)), 6) if layer_drifts else None,
                "max_relative_frobenius_drift": round(float(np.max(layer_drifts)), 6) if layer_drifts else None,
                "mean_spectral_norm_delta": round(float(np.mean(layer_spectral)), 6) if layer_spectral else None,
                "max_spectral_norm_delta": round(float(np.max(layer_spectral)), 6) if layer_spectral else None,
                "mean_subspace_angle": round(float(np.mean(layer_angles)), 4) if layer_angles else None,
                "max_subspace_angle": round(float(np.max(layer_angles)), 4) if layer_angles else None,
            }

            stream_results.append(batch_metrics)
            del weights

            if (batch_idx + 1) % 10 == 0 or batch_idx == checkpoints[-1].name.split("_")[1]:
                agg = batch_metrics["aggregate"]
                print(f"    Batch {batch_idx} ({total_edits} edits): "
                      f"fro_drift={agg['mean_relative_frobenius_drift']:.6f}, "
                      f"spectral={agg['mean_spectral_norm_delta']:.6f}, "
                      f"angle={agg['mean_subspace_angle']:.2f}°")

        all_results[stream_name] = stream_results

    return all_results


def analyze_failure_curve(args):
    """Analyze weight drift for failure curve checkpoints."""
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    layers = [4, 5, 6, 7, 8]
    base_weights = load_base_weights(model_name, layers)

    ckpt_base = Path(args.checkpoint_base)
    checkpoints = find_checkpoints(ckpt_base)

    if not checkpoints:
        print(f"  No checkpoints at {ckpt_base}")
        return {}

    print(f"\n  Analyzing failure curve ({len(checkpoints)} checkpoints)...")
    results = []

    for ckpt_dir in checkpoints:
        batch_idx = int(ckpt_dir.name.split("_")[1])
        total_edits = (batch_idx + 1) * 100

        weights = torch.load(ckpt_dir / "model_weights.pt", map_location="cpu")

        batch_metrics = {"batch_idx": batch_idx, "total_edits": total_edits, "layers": {}}

        for wname, w_t in weights.items():
            if wname in base_weights:
                layer_num = None
                for l in layers:
                    if f"layers.{l}." in wname:
                        layer_num = l
                        break
                metrics = compute_drift_metrics(base_weights[wname], w_t)
                batch_metrics["layers"][str(layer_num)] = metrics

        layer_drifts = [m["relative_frobenius_drift"] for m in batch_metrics["layers"].values()]
        layer_spectral = [m["spectral_norm_delta"] for m in batch_metrics["layers"].values()]
        layer_angles = [m["max_subspace_angle_deg"] for m in batch_metrics["layers"].values()
                      if not np.isnan(m["max_subspace_angle_deg"])]

        batch_metrics["aggregate"] = {
            "mean_relative_frobenius_drift": round(float(np.mean(layer_drifts)), 6) if layer_drifts else None,
            "max_relative_frobenius_drift": round(float(np.max(layer_drifts)), 6) if layer_drifts else None,
            "mean_spectral_norm_delta": round(float(np.mean(layer_spectral)), 6) if layer_spectral else None,
            "max_spectral_norm_delta": round(float(np.max(layer_spectral)), 6) if layer_spectral else None,
            "mean_subspace_angle": round(float(np.mean(layer_angles)), 4) if layer_angles else None,
            "max_subspace_angle": round(float(np.max(layer_angles)), 4) if layer_angles else None,
        }

        results.append(batch_metrics)
        del weights

        agg = batch_metrics["aggregate"]
        print(f"    Batch {batch_idx} ({total_edits} edits): "
              f"fro_drift={agg['mean_relative_frobenius_drift']:.6f}, "
              f"spectral={agg['mean_spectral_norm_delta']:.6f}, "
              f"angle={agg['mean_subspace_angle']:.2f}°")

    return {"failure_curve": results}


def main():
    parser = argparse.ArgumentParser(description="Weight drift analysis from checkpoints")
    parser.add_argument("--checkpoint_base", type=str, required=True,
                        help="Base directory for checkpoints")
    parser.add_argument("--mode", choices=["controlled_coupling", "failure_curve"],
                        default="controlled_coupling")
    parser.add_argument("--streams", default="low_coupling,high_coupling",
                        help="Comma-separated stream names (controlled_coupling mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Weight-Norm and Spectral-Drift Analysis")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Checkpoint base: {args.checkpoint_base}")

    if args.mode == "controlled_coupling":
        results = analyze_controlled_coupling(args)
    else:
        results = analyze_failure_curve(args)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = PROJECT_ROOT / "results" / "figures" / "paper"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"weight_drift_{args.mode}_seed{args.seed}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    if results:
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        for stream_name, stream_data in results.items():
            if not stream_data:
                continue
            last = stream_data[-1]
            agg = last["aggregate"]
            print(f"\n  {stream_name} (final checkpoint: {last['total_edits']} edits):")
            print(f"    Mean Frobenius drift:  {agg['mean_relative_frobenius_drift']:.6f}")
            print(f"    Max Frobenius drift:   {agg['max_relative_frobenius_drift']:.6f}")
            print(f"    Mean spectral norm Δ:  {agg['mean_spectral_norm_delta']:.6f}")
            print(f"    Mean subspace angle:   {agg['mean_subspace_angle']:.2f}°")


if __name__ == "__main__":
    main()
