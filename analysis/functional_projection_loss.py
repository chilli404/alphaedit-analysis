#!/usr/bin/env python3
"""
Functional Projection Loss Analysis (Step 2):

Computes the functional signal preservation ratio q_t from existing checkpoints
WITHOUT re-running the full edit experiment. For each checkpoint:
  - Loads the checkpoint model weights
  - Loads cache_c and P (null-space projection)
  - Selects a test batch of facts (the NEXT batch that would be edited)
  - Runs model forward to compute keys (K) and target residuals (resid)
  - Computes both projected and unconstrained solves
  - Measures q_t = ||ΔW_proj.T @ K|| / ||ΔW_raw.T @ K||

This directly shows whether the null-space projection is removing task-relevant
edit signal — the mechanism for capability loss BEYOND simple cache saturation.

Key metrics:
  q_t: functional signal preservation ratio (1.0 = no loss, 0.0 = total loss)
  fit_quality_proj: how well projected update achieves edit target
  fit_quality_raw: how well unconstrained update achieves edit target
  removed_fraction: RHS-level signal removal (fast proxy)

Usage:
    # From controlled coupling checkpoints (remote rig)
    uv run python analysis/functional_projection_loss.py \
        --checkpoint_base results/controlled_coupling/checkpoints \
        --streams low_coupling,high_coupling --seed 42

    # From failure curve checkpoints
    uv run python analysis/functional_projection_loss.py \
        --checkpoint_base /s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed42 \
        --mode failure_curve

    # Specific checkpoint indices only (fast test)
    uv run python analysis/functional_projection_loss.py \
        --checkpoint_base results/controlled_coupling/checkpoints \
        --streams low_coupling --seed 42 --checkpoint_indices 0,9,19,29,39,49
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAEDIT_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"

# Add vendor paths for imports
sys.path.insert(0, str(ALPHAEDIT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


def find_checkpoints(base_dir: Path) -> list[Path]:
    """Find all batch_N directories sorted by batch index."""
    if not base_dir.exists():
        return []
    batch_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    return [d for d in batch_dirs if (d / "model_weights.pt").exists()]


def load_model_and_tok(model_name: str):
    """Load base model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def apply_checkpoint_weights(model, ckpt_dir: Path) -> int:
    """Apply checkpoint weights to model. Returns count of restored params."""
    weights = torch.load(ckpt_dir / "model_weights.pt", map_location="cuda")
    param_dict = dict(model.named_parameters())
    loaded = 0
    for name, tensor in weights.items():
        if name in param_dict:
            param_dict[name].data.copy_(tensor.cuda())
            loaded += 1
    del weights
    torch.cuda.empty_cache()
    return loaded


def load_hparams(hparams_fname: str = "Llama3-8B.json"):
    """Load AlphaEdit hyperparameters."""
    sys.path.insert(0, str(ALPHAEDIT_ROOT / "AlphaEdit"))
    from AlphaEdit_hparams import AlphaEditHyperParams
    hparams_path = ALPHAEDIT_ROOT / "hparams" / "AlphaEdit" / hparams_fname
    return AlphaEditHyperParams.from_json(str(hparams_path))


def load_P(p_path: Path = None) -> torch.Tensor:
    """Load the null-space projection matrix P."""
    if p_path is None:
        # Try common locations
        candidates = [
            ALPHAEDIT_ROOT / "null_space_project.pt",
            Path("/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct/null_space_project.pt"),
        ]
        for c in candidates:
            if c.exists():
                p_path = c
                break
    if p_path is None or not p_path.exists():
        raise FileNotFoundError(
            "null_space_project.pt not found. Compute it first by running an AlphaEdit experiment."
        )
    P = torch.load(p_path, map_location="cpu")
    print(f"  Loaded P: shape={P.shape}")
    return P


def compute_functional_metrics(
    model, tok, hparams, P: torch.Tensor, cache_c: torch.Tensor,
    records: list[dict], layer_idx: int = None
) -> list[dict]:
    """
    Compute functional projection loss for a batch of edit records.

    For each edited layer:
      1. Compute keys (K) via model forward pass
      2. Compute target residuals (z* - z_current)
      3. Compute projected solve: (P(KK^T + C) + λI) X = P K resid^T
      4. Compute raw solve: (KK^T + C + λI) X = K resid^T
      5. Measure q_t and fit quality

    Returns per-layer metrics.
    """
    from AlphaEdit.compute_ks import compute_ks
    from AlphaEdit.compute_z import compute_z, get_module_input_output_at_words
    from util.generate import generate_fast

    # Get context templates (needed for compute_ks/compute_z)
    context_templates = [["{}"]] + [
        [
            f.replace("{", " ").replace("}", " ") + ". {}"
            for f in generate_fast(
                model, tok,
                ["The", "Therefore", "Because", "I", "You"],
                n_gen_per_prompt=1,
                max_out_len=10,
            )
        ]
    ]

    # Prepare requests (add space to target_new)
    from copy import deepcopy
    requests = deepcopy(records)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    # Compute z targets for last layer
    z_layer = hparams.layers[-1]
    z_list = []
    for request in requests:
        cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)

    results = []
    layers_to_process = [layer_idx] if layer_idx is not None else range(len(hparams.layers))

    for i in layers_to_process:
        if isinstance(i, int) and i >= len(hparams.layers):
            continue
        layer = hparams.layers[i] if layer_idx is None else hparams.layers[layer_idx]
        i_actual = i if layer_idx is None else layer_idx

        # Compute keys for this layer
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        # shape: [d_in, n]

        # Compute current model outputs to get residual
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i_actual)

        # --- Projected solve (what AlphaEdit does) ---
        P_i = P[i_actual, :, :].cuda()
        C_i = cache_c[i_actual, :, :].cuda()
        lhs_proj = P_i @ (layer_ks @ layer_ks.T + C_i) + hparams.L2 * torch.eye(
            layer_ks.shape[0], device="cuda", dtype=torch.float)
        rhs = layer_ks @ resid.T  # [d_in, d_out]
        rhs_proj = P_i @ rhs

        upd_proj = torch.linalg.solve(lhs_proj, rhs_proj)  # [d_in, d_out]

        # --- Raw solve (unconstrained) ---
        lhs_raw = layer_ks @ layer_ks.T + C_i + hparams.L2 * torch.eye(
            layer_ks.shape[0], device="cuda", dtype=torch.float)
        upd_raw = torch.linalg.solve(lhs_raw, rhs)  # [d_in, d_out]

        # --- Functional metrics ---
        # Effect on current edit keys
        effect_proj = upd_proj.T @ layer_ks  # [d_out, n]
        effect_raw = upd_raw.T @ layer_ks    # [d_out, n]

        effect_proj_norm = torch.linalg.norm(effect_proj).item()
        effect_raw_norm = torch.linalg.norm(effect_raw).item()

        # q_t: functional signal preservation ratio
        q_t = effect_proj_norm / max(effect_raw_norm, 1e-10)

        # Fit quality
        resid_norm = torch.linalg.norm(resid).item()
        fit_proj = 1.0 - (torch.linalg.norm(resid - effect_proj).item() / max(resid_norm, 1e-10))
        fit_raw = 1.0 - (torch.linalg.norm(resid - effect_raw).item() / max(resid_norm, 1e-10))

        # RHS-level removed fraction
        rhs_norm = torch.linalg.norm(rhs).item()
        proj_rhs_norm = torch.linalg.norm(rhs_proj).item()
        removed_fraction = max(0.0, 1.0 - (proj_rhs_norm / max(rhs_norm, 1e-10)))

        # Update norm comparison
        upd_proj_norm = torch.linalg.norm(upd_proj).item()
        upd_raw_norm = torch.linalg.norm(upd_raw).item()

        results.append({
            "layer": layer,
            "layer_position": i_actual,
            "q_t": round(q_t, 6),
            "fit_quality_projected": round(fit_proj, 6),
            "fit_quality_raw": round(fit_raw, 6),
            "effect_norm_projected": round(effect_proj_norm, 6),
            "effect_norm_raw": round(effect_raw_norm, 6),
            "target_norm": round(resid_norm, 6),
            "removed_fraction": round(removed_fraction, 6),
            "update_norm_projected": round(upd_proj_norm, 6),
            "update_norm_raw": round(upd_raw_norm, 6),
            "update_norm_ratio": round(upd_proj_norm / max(upd_raw_norm, 1e-10), 6),
        })

        # Cleanup per-layer GPU tensors
        del lhs_proj, rhs_proj, upd_proj, lhs_raw, upd_raw
        del effect_proj, effect_raw, layer_ks, cur_zs, targets, resid
        torch.cuda.empty_cache()

    return results


def analyze_controlled_coupling(args):
    """Analyze functional projection loss for controlled coupling checkpoints."""
    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    # Load model
    model, tok = load_model_and_tok(model_name)
    hparams = load_hparams(args.hparams_fname)
    P = load_P(Path(args.p_path) if args.p_path else None)

    streams = args.streams.split(",")
    all_results = {}

    # Parse checkpoint indices if specified
    ckpt_indices = None
    if args.checkpoint_indices:
        ckpt_indices = [int(x) for x in args.checkpoint_indices.split(",")]

    for stream_name in streams:
        ckpt_base = Path(args.checkpoint_base) / stream_name / f"seed{args.seed}"
        checkpoints = find_checkpoints(ckpt_base)

        if not checkpoints:
            print(f"  SKIP {stream_name}: no checkpoints at {ckpt_base}")
            continue

        # Load stream data for test batches
        stream_path = (PROJECT_ROOT / "results" / "controlled_coupling" /
                       f"{stream_name}_seed{args.seed}.json")
        if not stream_path.exists():
            print(f"  SKIP {stream_name}: stream file not found at {stream_path}")
            continue
        with open(stream_path) as f:
            stream_data = json.load(f)

        print(f"\n  Analyzing {stream_name} ({len(checkpoints)} checkpoints)...")
        stream_results = []

        for ckpt_dir in checkpoints:
            batch_idx = int(ckpt_dir.name.split("_")[1])

            # Skip if not in requested indices
            if ckpt_indices and batch_idx not in ckpt_indices:
                continue

            total_edits = (batch_idx + 1) * args.num_edits

            # Load checkpoint weights
            loaded = apply_checkpoint_weights(model, ckpt_dir)

            # Load cache_c from checkpoint
            cache_c_path = ckpt_dir / "cache_c.pt"
            if not cache_c_path.exists():
                print(f"    SKIP batch_{batch_idx}: no cache_c.pt")
                continue
            cache_c = torch.load(cache_c_path, map_location="cpu")

            # Select test batch: the NEXT batch of facts after this checkpoint
            next_batch_start = (batch_idx + 1) * args.num_edits
            next_batch_end = min(next_batch_start + args.num_edits, len(stream_data))
            if next_batch_start >= len(stream_data):
                # Use the last batch that was edited at this checkpoint
                next_batch_start = batch_idx * args.num_edits
                next_batch_end = min(next_batch_start + args.num_edits, len(stream_data))

            test_records = stream_data[next_batch_start:next_batch_end]
            if not test_records:
                print(f"    SKIP batch_{batch_idx}: no test records")
                continue

            # Sample if batch is large (for speed)
            if len(test_records) > args.max_test_facts:
                rng = np.random.RandomState(args.seed + batch_idx)
                indices = rng.choice(len(test_records), args.max_test_facts, replace=False)
                test_records = [test_records[i] for i in sorted(indices)]

            print(f"    Batch {batch_idx} ({total_edits} edits): testing {len(test_records)} facts...")

            # Compute functional metrics
            try:
                layer_metrics = compute_functional_metrics(
                    model, tok, hparams, P, cache_c, test_records
                )
            except Exception as e:
                print(f"    ERROR at batch_{batch_idx}: {e}")
                continue

            # Aggregate across layers
            q_ts = [m["q_t"] for m in layer_metrics]
            fits_proj = [m["fit_quality_projected"] for m in layer_metrics]
            fits_raw = [m["fit_quality_raw"] for m in layer_metrics]
            fracs = [m["removed_fraction"] for m in layer_metrics]

            batch_result = {
                "batch_idx": batch_idx,
                "total_edits": total_edits,
                "n_test_facts": len(test_records),
                "layers": layer_metrics,
                "aggregate": {
                    "mean_q_t": round(float(np.mean(q_ts)), 6),
                    "min_q_t": round(float(min(q_ts)), 6),
                    "max_q_t": round(float(max(q_ts)), 6),
                    "mean_fit_quality_projected": round(float(np.mean(fits_proj)), 6),
                    "mean_fit_quality_raw": round(float(np.mean(fits_raw)), 6),
                    "mean_removed_fraction": round(float(np.mean(fracs)), 6),
                    "fit_quality_gap": round(float(np.mean(fits_raw)) - float(np.mean(fits_proj)), 6),
                },
            }
            stream_results.append(batch_result)

            agg = batch_result["aggregate"]
            print(f"      q_t={agg['mean_q_t']:.4f}, "
                  f"fit_proj={agg['mean_fit_quality_projected']:.4f}, "
                  f"fit_raw={agg['mean_fit_quality_raw']:.4f}, "
                  f"removed={agg['mean_removed_fraction']:.4f}")

            del cache_c
            torch.cuda.empty_cache()

        all_results[stream_name] = stream_results

    del model, tok
    torch.cuda.empty_cache()
    return all_results


def analyze_failure_curve(args):
    """Analyze functional projection loss for failure curve checkpoints."""
    from model_download import resolve_model_path
    model_name = resolve_model_path(args.model_name)

    model, tok = load_model_and_tok(model_name)
    hparams = load_hparams(args.hparams_fname)
    P = load_P(Path(args.p_path) if args.p_path else None)

    ckpt_base = Path(args.checkpoint_base)
    checkpoints = find_checkpoints(ckpt_base)

    if not checkpoints:
        print(f"  No checkpoints at {ckpt_base}")
        return {}

    # Load MCF dataset for test batches
    mcf_path = ALPHAEDIT_ROOT / "data" / "multi_counterfact.json"
    if not mcf_path.exists():
        print(f"  ERROR: dataset not found at {mcf_path}")
        return {}
    with open(mcf_path) as f:
        all_records = json.load(f)

    # Parse checkpoint indices
    ckpt_indices = None
    if args.checkpoint_indices:
        ckpt_indices = [int(x) for x in args.checkpoint_indices.split(",")]

    print(f"\n  Analyzing failure curve ({len(checkpoints)} checkpoints)...")
    results = []

    for ckpt_dir in checkpoints:
        batch_idx = int(ckpt_dir.name.split("_")[1])

        if ckpt_indices and batch_idx not in ckpt_indices:
            continue

        total_edits = (batch_idx + 1) * args.num_edits

        loaded = apply_checkpoint_weights(model, ckpt_dir)
        cache_c_path = ckpt_dir / "cache_c.pt"
        if not cache_c_path.exists():
            print(f"    SKIP batch_{batch_idx}: no cache_c.pt")
            continue
        cache_c = torch.load(cache_c_path, map_location="cpu")

        # Test on the next batch
        next_start = (batch_idx + 1) * args.num_edits
        next_end = min(next_start + args.num_edits, len(all_records))
        if next_start >= len(all_records):
            next_start = batch_idx * args.num_edits
            next_end = min(next_start + args.num_edits, len(all_records))

        test_records = all_records[next_start:next_end]
        if len(test_records) > args.max_test_facts:
            rng = np.random.RandomState(args.seed + batch_idx)
            indices = rng.choice(len(test_records), args.max_test_facts, replace=False)
            test_records = [test_records[i] for i in sorted(indices)]

        print(f"    Batch {batch_idx} ({total_edits} edits): testing {len(test_records)} facts...")

        try:
            layer_metrics = compute_functional_metrics(
                model, tok, hparams, P, cache_c, test_records
            )
        except Exception as e:
            print(f"    ERROR at batch_{batch_idx}: {e}")
            continue

        q_ts = [m["q_t"] for m in layer_metrics]
        fits_proj = [m["fit_quality_projected"] for m in layer_metrics]
        fits_raw = [m["fit_quality_raw"] for m in layer_metrics]
        fracs = [m["removed_fraction"] for m in layer_metrics]

        batch_result = {
            "batch_idx": batch_idx,
            "total_edits": total_edits,
            "n_test_facts": len(test_records),
            "layers": layer_metrics,
            "aggregate": {
                "mean_q_t": round(float(np.mean(q_ts)), 6),
                "min_q_t": round(float(min(q_ts)), 6),
                "max_q_t": round(float(max(q_ts)), 6),
                "mean_fit_quality_projected": round(float(np.mean(fits_proj)), 6),
                "mean_fit_quality_raw": round(float(np.mean(fits_raw)), 6),
                "mean_removed_fraction": round(float(np.mean(fracs)), 6),
                "fit_quality_gap": round(float(np.mean(fits_raw)) - float(np.mean(fits_proj)), 6),
            },
        }
        results.append(batch_result)

        agg = batch_result["aggregate"]
        print(f"      q_t={agg['mean_q_t']:.4f}, fit_proj={agg['mean_fit_quality_projected']:.4f}, "
              f"fit_raw={agg['mean_fit_quality_raw']:.4f}")

        del cache_c
        torch.cuda.empty_cache()

    del model, tok
    torch.cuda.empty_cache()
    return {"failure_curve": results}


def main():
    parser = argparse.ArgumentParser(description="Functional projection loss analysis from checkpoints")
    parser.add_argument("--checkpoint_base", type=str, required=True,
                        help="Base directory for checkpoints")
    parser.add_argument("--mode", choices=["controlled_coupling", "failure_curve"],
                        default="controlled_coupling")
    parser.add_argument("--streams", default="low_coupling,high_coupling",
                        help="Comma-separated stream names (controlled_coupling mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--p_path", type=str, default=None,
                        help="Path to null_space_project.pt (auto-detected if not set)")
    parser.add_argument("--num_edits", type=int, default=100,
                        help="Batch size (edits per batch)")
    parser.add_argument("--max_test_facts", type=int, default=25,
                        help="Max facts per test batch (for speed; 25 ≈ 3min/checkpoint)")
    parser.add_argument("--checkpoint_indices", type=str, default=None,
                        help="Comma-separated batch indices to analyze (e.g., '9,19,29,39,49')")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Functional Projection Loss Analysis")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Checkpoint base: {args.checkpoint_base}")
    print(f"  Max test facts per checkpoint: {args.max_test_facts}")
    if args.checkpoint_indices:
        print(f"  Checkpoint indices: {args.checkpoint_indices}")

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
        out_path = out_dir / f"functional_projection_loss_{args.mode}_seed{args.seed}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("SUMMARY: Functional Projection Loss")
        print(f"{'=' * 70}")
        for stream_name, stream_data in results.items():
            if not stream_data:
                continue
            print(f"\n  {stream_name}:")
            print(f"    {'Edits':<8} {'q_t':<10} {'fit_proj':<12} {'fit_raw':<12} {'gap':<10} {'removed':<10}")
            print(f"    {'-' * 60}")
            for batch in stream_data:
                agg = batch["aggregate"]
                print(f"    {batch['total_edits']:<8} "
                      f"{agg['mean_q_t']:<10.4f} "
                      f"{agg['mean_fit_quality_projected']:<12.4f} "
                      f"{agg['mean_fit_quality_raw']:<12.4f} "
                      f"{agg['fit_quality_gap']:<10.4f} "
                      f"{agg['mean_removed_fraction']:<10.4f}")

            # Highlight key finding
            if len(stream_data) >= 2:
                first_qt = stream_data[0]["aggregate"]["mean_q_t"]
                last_qt = stream_data[-1]["aggregate"]["mean_q_t"]
                print(f"\n    Signal preservation decline: {first_qt:.4f} → {last_qt:.4f} "
                      f"(Δ = {last_qt - first_qt:+.4f})")


if __name__ == "__main__":
    main()
