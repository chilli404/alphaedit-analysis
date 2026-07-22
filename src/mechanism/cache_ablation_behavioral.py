#!/usr/bin/env python3
"""
Cache Ablation — Behavioral Evaluation.

Loads the 7K checkpoint and applies the NEXT batch (100 edits) using different
cache scaling factors γ ∈ {0, 0.1, 0.25, 0.5, 1.0}. For each γ, measures:

  - New-batch efficacy (does the edit produce the target token?)
  - New-batch paraphrase generalization
  - Previous-edit retention (sample of 200 earlier edits still correct?)
  - Neighborhood/locality (unrelated facts intact?)
  - Update norm ||ΔW||_F per layer
  - Residual attainment per layer

This establishes the Pareto tradeoff:
  lower γ → higher new-edit efficacy
  lower γ → worse previous-edit retention

Usage:
    uv run python src/cache_ablation_behavioral.py --seed 42
    uv run python src/cache_ablation_behavioral.py --seed 42 --gamma_values 0 0.5 1.0
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# NOTE: Do NOT add SRC_DIR to sys.path — src/datasets/ shadows the
# HuggingFace 'datasets' package which vendor code needs.
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))

from paths import get_alphaedit_root, get_result_root, get_checkpoint_root

ALPHAEDIT_ROOT = get_alphaedit_root()


def main():
    parser = argparse.ArgumentParser(description="Cache ablation behavioral evaluation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_batch", type=int, default=69,
                        help="Batch index to resume from (69 = 7000 edits)")
    parser.add_argument("--gamma_values", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--model_name", type=str,
                        default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", type=str, default="Llama3-8B.json")
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--num_edits", type=int, default=100,
                        help="Edits per batch")
    parser.add_argument("--retention_sample", type=int, default=200,
                        help="Number of previous edits to sample for retention")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # ─── Paths ───
    if args.checkpoint_dir:
        ckpt_base = Path(args.checkpoint_dir) / "AlphaEdit" / f"seed{args.seed}"
    else:
        ckpt_base = get_checkpoint_root() / "failure_curve" / "AlphaEdit" / f"seed{args.seed}"

    ckpt_dir = ckpt_base / f"batch_{args.checkpoint_batch}"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_result_root() / "cache_ablation_behavioral"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"behavioral_seed{args.seed}_batch{args.checkpoint_batch}_{timestamp}.jsonl"

    print("=" * 70)
    print("Cache Ablation — Behavioral Evaluation")
    print(f"  Seed:             {args.seed}")
    print(f"  Checkpoint batch: {args.checkpoint_batch} ({(args.checkpoint_batch+1)*100} edits)")
    print(f"  Gamma values:     {args.gamma_values}")
    print(f"  Model:            {args.model_name}")
    print(f"  Checkpoint dir:   {ckpt_dir}")
    print(f"  Retention sample: {args.retention_sample}")
    print(f"  Output:           {output_path}")
    print("=" * 70)

    # ─── Verify checkpoint ───
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_dir}")
        sys.exit(1)

    # ─── Import vendor code (needs cwd = ALPHAEDIT_ROOT for globals.yml) ───
    sys.path.insert(0, str(ALPHAEDIT_ROOT / "experiments"))
    sys.path.insert(0, str(ALPHAEDIT_ROOT))

    original_cwd = os.getcwd()
    os.chdir(ALPHAEDIT_ROOT)

    from dsets import CounterFactDataset
    from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact

    os.chdir(original_cwd)

    # ─── Load model ───
    print(f"\n  Loading model...")
    print(f"  HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'NOT SET')}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=token).cuda()
    tok = AutoTokenizer.from_pretrained(args.model_name, token=token)
    tok.pad_token = tok.eos_token
    print(f"  Model loaded: {args.model_name}")

    # ─── Load checkpoint ───
    print(f"\n  Loading checkpoint from {ckpt_dir}...")
    weights_path = ckpt_dir / "model_weights.pt"
    cache_path = ckpt_dir / "cache_c.pt"

    checkpoint_weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint_weights, strict=False)
    print(f"  Model weights restored ({len(checkpoint_weights)} params)")

    # Save the checkpoint state so we can reset between gamma runs
    checkpoint_state = {k: v.clone() for k, v in checkpoint_weights.items()}

    cache_c_original = torch.load(cache_path, map_location="cpu")
    print(f"  cache_c loaded: shape={cache_c_original.shape}")

    # ─── Load P ───
    import glob
    p_candidates = glob.glob(str(ckpt_base / "**" / "null_space_project*"), recursive=True)
    p_candidates += glob.glob("/s3-data/continual-learning/alphaedit/stats/**/null_space_project*", recursive=True)
    if not p_candidates:
        print("  ERROR: null_space_project.pt not found")
        sys.exit(1)
    P = torch.load(p_candidates[0], map_location="cpu").float()
    print(f"  P loaded: shape={P.shape} from {p_candidates[0]}")

    # ─── Load hparams ───
    hparams_path = ALPHAEDIT_ROOT / "hparams" / "AlphaEdit" / args.hparams_fname
    hparams = AlphaEditHyperParams.from_json(str(hparams_path))
    print(f"  Layers: {hparams.layers}, L2={hparams.L2}")

    # ─── Load dataset ───
    print(f"\n  Loading dataset...")
    os.chdir(ALPHAEDIT_ROOT)
    DATA_DIR = ALPHAEDIT_ROOT / "data"
    ds = CounterFactDataset(str(DATA_DIR), multi=True)
    os.chdir(original_cwd)

    # Target batch (the NEXT batch after checkpoint)
    target_start = (args.checkpoint_batch + 1) * args.num_edits
    target_end = target_start + args.num_edits
    new_batch = [ds[i] for i in range(target_start, min(target_end, len(ds)))]
    print(f"  New batch: cases {target_start}-{target_end-1} ({len(new_batch)} records)")

    # Previous edits sample (for retention measurement)
    import random
    random.seed(args.seed)
    all_previous_indices = list(range(0, target_start))
    retention_indices = random.sample(all_previous_indices, min(args.retention_sample, len(all_previous_indices)))
    retention_batch = [ds[i] for i in retention_indices]
    print(f"  Retention sample: {len(retention_batch)} records from 0-{target_start-1}")

    # ─── Run behavioral evaluation for each gamma ───
    print(f"\n  Running behavioral evaluation...")
    results = []

    for gamma in args.gamma_values:
        print(f"\n{'='*70}")
        print(f"  γ = {gamma}")
        print(f"{'='*70}")

        # Reset model to checkpoint state
        model.load_state_dict(checkpoint_state, strict=False)
        torch.cuda.empty_cache()

        # Scale cache_c by gamma
        scaled_cache = gamma * cache_c_original.clone()

        # Prepare requests (pass raw requested_rewrite — AlphaEdit expects {} placeholder)
        requests = [record["requested_rewrite"] for record in new_batch]

        # Apply AlphaEdit with scaled cache
        print(f"    Applying edits (100 facts, γ={gamma})...")
        edited_model, updated_cache = apply_AlphaEdit_to_model(
            model, tok, requests, hparams,
            cache_c=scaled_cache,
            P=P,
        )

        # Compute update norms per layer
        update_norms = {}
        for param_name, ckpt_val in checkpoint_state.items():
            current_val = dict(model.named_parameters())[param_name].detach().cpu()
            delta = current_val - ckpt_val
            layer_idx = None
            for l in hparams.layers:
                if f"layers.{l}" in param_name:
                    layer_idx = l
                    break
            if layer_idx is not None:
                update_norms[layer_idx] = {
                    "delta_fro": delta.norm().item(),
                    "weight_fro": ckpt_val.norm().item(),
                    "relative": delta.norm().item() / ckpt_val.norm().item(),
                }

        # ─── Evaluate new-batch efficacy ───
        print(f"    Evaluating new-batch efficacy ({len(new_batch)} cases)...")
        new_batch_metrics = []
        for record in new_batch:
            metrics = compute_rewrite_quality_counterfact(
                model, tok, record,
                snips=None, vec=None,
            )
            new_batch_metrics.append(metrics)

        # Aggregate new-batch metrics
        def aggregate_metrics(metrics_list, prefix=""):
            """Extract mean of key metrics."""
            agg = {}
            # Efficacy: rewrite_prompts_correct
            vals = [m.get("rewrite_prompts_correct", []) for m in metrics_list]
            flat = [v for sublist in vals for v in (sublist if isinstance(sublist, list) else [sublist])]
            if flat:
                agg[f"{prefix}efficacy"] = sum(flat) / len(flat)

            # Paraphrase: paraphrase_prompts_correct
            vals = [m.get("paraphrase_prompts_correct", []) for m in metrics_list]
            flat = [v for sublist in vals for v in (sublist if isinstance(sublist, list) else [sublist])]
            if flat:
                agg[f"{prefix}paraphrase"] = sum(flat) / len(flat)

            # Neighborhood: neighborhood_prompts_correct
            vals = [m.get("neighborhood_prompts_correct", []) for m in metrics_list]
            flat = [v for sublist in vals for v in (sublist if isinstance(sublist, list) else [sublist])]
            if flat:
                agg[f"{prefix}neighborhood"] = sum(flat) / len(flat)

            return agg

        new_agg = aggregate_metrics(new_batch_metrics, "new_")
        print(f"    New-batch: efficacy={new_agg.get('new_efficacy', 'N/A'):.4f}, "
              f"paraphrase={new_agg.get('new_paraphrase', 'N/A'):.4f}, "
              f"neighborhood={new_agg.get('new_neighborhood', 'N/A'):.4f}")

        # ─── Evaluate retention on previous edits ───
        print(f"    Evaluating retention ({len(retention_batch)} sampled previous edits)...")
        retention_metrics = []
        for record in retention_batch:
            metrics = compute_rewrite_quality_counterfact(
                model, tok, record,
                snips=None, vec=None,
            )
            retention_metrics.append(metrics)

        ret_agg = aggregate_metrics(retention_metrics, "retention_")
        print(f"    Retention: efficacy={ret_agg.get('retention_efficacy', 'N/A'):.4f}, "
              f"paraphrase={ret_agg.get('retention_paraphrase', 'N/A'):.4f}, "
              f"neighborhood={ret_agg.get('retention_neighborhood', 'N/A'):.4f}")

        # ─── Record results ───
        gamma_result = {
            "gamma": gamma,
            "seed": args.seed,
            "checkpoint_batch": args.checkpoint_batch,
            "total_prior_edits": (args.checkpoint_batch + 1) * args.num_edits,
            "new_batch_size": len(new_batch),
            "retention_sample_size": len(retention_batch),
            **new_agg,
            **ret_agg,
            "update_norms": update_norms,
        }
        results.append(gamma_result)

        print(f"\n    Summary γ={gamma}:")
        print(f"      New efficacy:       {new_agg.get('new_efficacy', 'N/A'):.4f}")
        print(f"      New paraphrase:     {new_agg.get('new_paraphrase', 'N/A'):.4f}")
        print(f"      New neighborhood:   {new_agg.get('new_neighborhood', 'N/A'):.4f}")
        print(f"      Retention efficacy: {ret_agg.get('retention_efficacy', 'N/A'):.4f}")
        for l, norms in sorted(update_norms.items()):
            print(f"      Layer {l} ||ΔW||/||W|| = {norms['relative']:.6f}")

    # ─── Save results ───
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\n{'='*70}")
    print(f"Results saved: {output_path}")
    print(f"{'='*70}")

    # ─── Print Pareto summary ───
    print(f"\n{'='*70}")
    print("PARETO TRADEOFF SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'γ':<6} {'New Eff':<10} {'New Para':<10} {'Ret Eff':<10} {'Ret Para':<10} {'Neighborhood':<12}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['gamma']:<6.2f} "
              f"{r.get('new_efficacy', 0):<10.4f} "
              f"{r.get('new_paraphrase', 0):<10.4f} "
              f"{r.get('retention_efficacy', 0):<10.4f} "
              f"{r.get('retention_paraphrase', 0):<10.4f} "
              f"{r.get('new_neighborhood', 0):<12.4f}")

    print(f"\n  Prediction: lower γ → higher new_efficacy, lower retention_efficacy")
    print(f"  If confirmed: cache-induced over-regularization is the causal mechanism")


if __name__ == "__main__":
    main()
