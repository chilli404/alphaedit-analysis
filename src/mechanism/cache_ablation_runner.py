#!/usr/bin/env python3
"""
Cache Ablation Experiment: Causal test of over-regularization hypothesis.

Loads a late checkpoint (e.g., 7K edits) and solves the SAME next batch
under different cache scaling factors γ ∈ {0, 0.1, 0.25, 0.5, 1.0}.

Measures per γ:
  - ||ΔW||_F                     (update norm — should increase as γ decreases)
  - ||ΔW@K - R||_F / ||R||_F     (residual attainment — should improve)
  - efficacy                      (new-edit success — should recover)
  - specificity                   (locality — should worsen)

Also measures projection ablation:
  - ||Pk - k||₂ / ||k||₂         (key projection loss)
  - ||PA - A||_F / ||A||_F       (update-driving projection loss)
  - P vs I comparison

Hypothesis prediction: reducing γ increases update norm, improves residual
attainment, recovers new-edit efficacy, but worsens old-edit retention.

Usage:
    uv run python src/cache_ablation_runner.py \\
        --seed 42 \\
        --checkpoint_batch 69 \\
        --gamma_values 0 0.1 0.25 0.5 1.0 \\
        --model_name NousResearch/Meta-Llama-3-8B-Instruct

On cluster:
    bash scripts/run_cache_ablation.sh 42
"""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# NOTE: Do NOT add SRC_DIR to sys.path — src/datasets/ shadows HuggingFace 'datasets'
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))

from paths import get_alphaedit_root, get_result_root, get_checkpoint_root

ALPHAEDIT_ROOT = get_alphaedit_root()

if str(ALPHAEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAEDIT_ROOT))


def build_injection_code(gamma_values: list[float], checkpoint_batch: int) -> str:
    """
    Build source injection code that:
    1. Patches the AlphaEdit solve to scale cache_c by gamma
    2. Records per-layer diagnostics (update norm, residual, key projection loss)
    3. Loops over gamma values for the same batch
    """
    gamma_str = repr(gamma_values)
    return textwrap.dedent(f"""\
    # ═══ CACHE ABLATION INJECTION ═══
    import json as _json
    from pathlib import Path as _Path

    _GAMMA_VALUES = {gamma_str}
    _CHECKPOINT_BATCH = {checkpoint_batch}
    _ABLATION_RESULTS = []
    """)


def build_alphaedit_patch(gamma_values: list[float]) -> str:
    """
    Patch for AlphaEdit_main.py that:
    1. Accepts a `cache_gamma` parameter
    2. Records solve diagnostics (update norm, residual, key projection loss)
    3. Returns diagnostics alongside the update
    """
    gamma_str = repr(gamma_values)
    return textwrap.dedent("""\
    # ═══ CACHE ABLATION: AlphaEdit_main.py PATCH ═══
    # Store diagnostics from the solve
    _ABLATION_SOLVE_DIAGNOSTICS = []

    # Patch: find the solve and insert gamma scaling + diagnostics
    # The key anchor is the line that computes upd_matrix
    """)


def main():
    parser = argparse.ArgumentParser(description="Cache ablation experiment")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_batch", type=int, default=69,
                        help="Batch index to resume from (69 = 7000 edits)")
    parser.add_argument("--gamma_values", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--model_name", type=str,
                        default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", type=str, default="Llama3-8B.json")
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--dataset_size_limit", type=int, default=7100,
                        help="Must be > checkpoint_batch * 100 + 100")
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Paths
    if args.checkpoint_dir:
        ckpt_base = Path(args.checkpoint_dir) / "AlphaEdit" / f"seed{args.seed}"
    else:
        ckpt_base = get_checkpoint_root() / "failure_curve" / "AlphaEdit" / f"seed{args.seed}"

    ckpt_dir = ckpt_base / f"batch_{args.checkpoint_batch}"

    edits = (args.checkpoint_batch + 1) * 100
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_result_root() / "cache_ablation" / f"seed{args.seed}" / f"{edits}edits" / "AlphaEdit"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cache_ablation_seed{args.seed}_batch{args.checkpoint_batch}_{timestamp}.jsonl"

    print("=" * 70)
    print("Cache Ablation Experiment")
    print(f"  Seed:             {args.seed}")
    print(f"  Checkpoint batch: {args.checkpoint_batch} ({(args.checkpoint_batch+1)*100} edits)")
    print(f"  Gamma values:     {args.gamma_values}")
    print(f"  Model:            {args.model_name}")
    print(f"  Checkpoint dir:   {ckpt_dir}")
    print(f"  Output:           {output_path}")
    print("=" * 70)

    # Verify checkpoint exists
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_dir}")
        sys.exit(1)

    # ─── Source injection approach ───
    # Read evaluate.py and AlphaEdit_main.py, inject ablation code
    evaluate_path = ALPHAEDIT_ROOT / "experiments" / "evaluate.py"
    alphaedit_main_path = ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py"

    evaluate_src = evaluate_path.read_text()
    alphaedit_src = alphaedit_main_path.read_text()

    # ─── Patch AlphaEdit_main.py: inject gamma scaling into the solve ───
    # The solve in AlphaEdit_main.py:
    #   upd_matrix = (right_vector @ (left_pseudo @ P_).T).T
    # But the actual constraint uses cache_c in the normal equations.
    # We need to find where cache_c is used in the LHS and scale it.

    # The key section is where the solve happens. In AlphaEdit_main.py,
    # the constraint is built as:
    #   layer_ks @ layer_ks.T + cache_c[i,:,:]  (in the LHS)
    # We inject gamma scaling on cache_c.

    # Find the anchor: "cache_c[i,:,:].cuda()" or similar
    cache_anchor = "layer_ks.cpu() @ layer_ks.cpu().T"

    if cache_anchor not in alphaedit_src:
        # Try alternative anchors
        for candidate in ["layer_ks @ layer_ks.T", "cache_c[i", "cache_c["]:
            if candidate in alphaedit_src:
                cache_anchor = candidate
                break

    print(f"\n  AlphaEdit source anchor found: '{cache_anchor[:50]}...'")

    # ─── Inject into evaluate.py: checkpoint loading + multi-gamma loop ───
    # Instead of complex source injection, use a simpler approach:
    # Load model + checkpoint, then manually run the solve with different gammas

    # Load model (HF_ENDPOINT is set in shell script before Python starts,
    # so huggingface_hub picks up Artifactory at import time — same as seeded_runner subprocess)
    print(f"\n  Loading model: {args.model_name}")
    print(f"  HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'NOT SET')}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=token).cuda()
    tok = AutoTokenizer.from_pretrained(args.model_name, token=token)
    tok.pad_token = tok.eos_token
    print(f"  Model loaded: {args.model_name}")

    # Load checkpoint weights
    print(f"\n  Loading checkpoint from {ckpt_dir}...")
    weights_path = ckpt_dir / "model_weights.pt"
    cache_path = ckpt_dir / "cache_c.pt"

    if weights_path.exists():
        checkpoint_weights = torch.load(weights_path, map_location="cpu")
        # Restore model weights
        model.load_state_dict(checkpoint_weights, strict=False)
        print(f"  Model weights restored ({len(checkpoint_weights)} params)")
    else:
        print(f"  WARNING: No model_weights.pt found, using base model")

    cache_c = torch.load(cache_path, map_location="cpu") if cache_path.exists() else None
    if cache_c is not None:
        print(f"  cache_c loaded: shape={cache_c.shape}")
    else:
        print(f"  WARNING: No cache_c.pt found")
        sys.exit(1)

    # Load P (null-space projector)
    # Search common locations
    import glob
    p_candidates = glob.glob(str(ckpt_base / "**" / "null_space_project*"), recursive=True)
    p_candidates += glob.glob("/s3-data/continual-learning/alphaedit/stats/**/null_space_project*", recursive=True)
    if not p_candidates:
        print("  ERROR: null_space_project.pt not found")
        sys.exit(1)

    P = torch.load(p_candidates[0], map_location="cpu").float()
    print(f"  P loaded: shape={P.shape} from {p_candidates[0]}")

    # Load hparams
    hparams_path = ALPHAEDIT_ROOT / "hparams" / "AlphaEdit" / args.hparams_fname
    with open(hparams_path) as f:
        hparams_data = json.load(f)
    layers = hparams_data["layers"]
    L2 = hparams_data.get("L2", 1e-5)
    print(f"  Layers: {layers}, L2={L2}")

    # Load dataset and get the target batch
    print(f"\n  Loading dataset...")
    sys.path.insert(0, str(ALPHAEDIT_ROOT / "experiments"))
    sys.path.insert(0, str(ALPHAEDIT_ROOT))

    # AlphaEdit vendor code reads globals.yml relative to cwd
    original_cwd = os.getcwd()
    os.chdir(ALPHAEDIT_ROOT)

    DATA_DIR = ALPHAEDIT_ROOT / "data"
    from dsets import CounterFactDataset
    from AlphaEdit.compute_ks import compute_ks

    os.chdir(original_cwd)

    ds = CounterFactDataset(str(DATA_DIR), multi=True)

    # Get the batch right after checkpoint
    target_batch_start = (args.checkpoint_batch + 1) * args.num_edits
    target_batch_end = target_batch_start + args.num_edits
    batch_records = [ds[i] for i in range(target_batch_start, min(target_batch_end, len(ds)))]
    print(f"  Target batch: cases {target_batch_start}-{target_batch_end-1} ({len(batch_records)} records)")

    # ─── Compute keys for the target batch ───
    print(f"\n  Computing keys for target batch...")

    # Prepare requests in the format AlphaEdit expects
    requests = []
    for record in batch_records:
        rw = record["requested_rewrite"]
        requests.append({
            "prompt": rw["prompt"],
            "subject": rw["subject"],
            "target_new": rw["target_new"],
            "target_true": rw["target_true"],
        })

    # Context templates: list of lists (each inner list = one context type)
    context_templates = [["{}"]]

    # Compute keys for each layer
    print(f"  Computing keys for {len(layers)} layers...")
    all_layer_ks = {}
    for i, layer in enumerate(layers):
        layer_name = hparams_data["rewrite_module_tmp"].format(layer)
        # Use AlphaEdit's compute_ks
        try:
            layer_ks = compute_ks(model, tok, requests, type("H", (), hparams_data)(), layer, context_templates).T
            all_layer_ks[i] = layer_ks.cuda()
            print(f"    Layer {layer}: keys shape={layer_ks.shape}")
        except Exception as e:
            print(f"    Layer {layer}: FAILED to compute keys: {e}")
            all_layer_ks[i] = None

    # ─── Run ablation for each gamma ───
    print(f"\n  Running cache ablation...")
    results = []

    # Precompute base model weight norm for relative update size
    W_base_norms = {}
    for i, layer in enumerate(layers):
        param_name = f"model.layers.{layer}.mlp.down_proj.weight"
        W = dict(model.named_parameters()).get(param_name)
        if W is not None:
            W_base_norms[i] = W.detach().norm().item()
        else:
            W_base_norms[i] = 1.0  # fallback

    for gamma in args.gamma_values:
        print(f"\n  --- γ = {gamma} ---")
        gamma_result = {
            "gamma": gamma,
            "seed": args.seed,
            "checkpoint_batch": args.checkpoint_batch,
            "target_batch": args.checkpoint_batch + 1,
            "total_prior_edits": (args.checkpoint_batch + 1) * args.num_edits,
            "layers": {},
        }

        for i, layer in enumerate(layers):
            if all_layer_ks[i] is None:
                continue

            layer_ks = all_layer_ks[i]  # (d, n_keys) on cuda
            P_i = P[i].cuda()  # (d, d)
            C_i = cache_c[i].cuda()  # (d, d)
            d = P_i.shape[0]

            # Build LHS: P @ (K_new @ K_new^T + gamma * cache_c) + L2 * I
            K_gram = layer_ks @ layer_ks.T  # (d, d)
            LHS = P_i @ (K_gram + gamma * C_i) + L2 * torch.eye(d, device="cuda")

            # ─── Diagnostic 1: cache dominance (Frobenius energy ratio) ───
            # ||γC||_F / ||KK^T + λI||_F
            key_term_fro = (K_gram + L2 * torch.eye(d, device="cuda")).norm().item()
            cache_term_fro = (gamma * C_i).norm().item()
            cache_dominance = cache_term_fro / key_term_fro if key_term_fro > 0 else 0.0

            # ─── Diagnostic 2: per-key cache alignment k^T C k / ||k||² ───
            # Shows how much cache penalty each incoming key faces
            Ck = C_i @ layer_ks  # (d, n_keys)
            k_sq_norms = (layer_ks * layer_ks).sum(dim=0)  # (n_keys,)
            kCk = (layer_ks * Ck).sum(dim=0)  # (n_keys,) — k^T C k per key
            cache_alignment = (kCk / k_sq_norms.clamp(min=1e-8))  # per-key
            mean_cache_alignment = cache_alignment.mean().item()
            max_cache_alignment = cache_alignment.max().item()

            # ─── Diagnostic 3: solve + update norm + residual attainment ───
            # Use random target vectors v of shape (n_keys,) representing
            # scalar targets per key (one "row" of the edit problem)
            torch.manual_seed(args.seed + i)  # reproducible per layer
            n_rhs = 10  # average over 10 random target directions
            # v: (n_keys, n_rhs) — random targets scaled to realistic magnitude
            v = torch.randn(layer_ks.shape[1], n_rhs, device="cuda")
            v = v / v.norm(dim=0, keepdim=True)
            # RHS = P @ K @ v: (d, n_rhs) — one column of ΔW^T per rhs
            Pk = P_i @ layer_ks  # (d, n_keys)
            RHS = Pk @ v  # (d, n_rhs)

            try:
                X = torch.linalg.solve(LHS, RHS)  # (d, n_rhs) — solution

                # Relative update size: ||ΔW||_F / ||W||_F (averaged per-column)
                update_norm = X.norm().item()
                relative_update = update_norm / W_base_norms[i]

                # Residual attainment: ||ΔW @ K||_F / ||R||_F
                # X = ΔW^T columns, so ΔW @ K = X^T @ K... but simpler:
                # The target is that LHS @ X = RHS (the constrained normal eq)
                # ΔW^T @ K should approximately recover v (the targets)
                achieved = X.T @ layer_ks  # (n_rhs, n_keys)
                residual_attainment = achieved.norm().item() / v.norm().item()

                # Relative solve residual: ||LX - B||_F / ||B||_F
                solve_residual_abs = (LHS @ X - RHS).norm().item()
                solve_residual = solve_residual_abs / RHS.norm().item() if RHS.norm().item() > 0 else 0.0

            except Exception as e:
                update_norm = relative_update = float("nan")
                residual_attainment = float("nan")
                solve_residual = float("nan")
                print(f"    Solve failed: {e}")

            # ─── Diagnostic 4: inverse gain along keys ───
            # g_t = k^T @ (LHS)^{-1} @ P @ k  (averaged over keys in batch)
            try:
                inv_LHS_Pk = torch.linalg.solve(LHS, Pk)  # (d, n_keys)
                gains = (layer_ks * inv_LHS_Pk).sum(dim=0)  # per-key gain
                mean_gain = gains.mean().item()
                min_gain = gains.min().item()
                max_gain = gains.max().item()
            except Exception as e:
                mean_gain = min_gain = max_gain = float("nan")
                print(f"    Gain solve failed: {e}")

            # ─── Diagnostic 5: key projection loss ───
            k_norms = layer_ks.norm(dim=0)  # (n_keys,)
            Pk_norms = Pk.norm(dim=0)  # (n_keys,)
            projection_loss = ((k_norms - Pk_norms) / k_norms.clamp(min=1e-8)).mean().item()

            layer_result = {
                "layer_idx": layer,
                "cache_dominance_fro": round(cache_dominance, 4),
                "mean_cache_alignment_kCk": round(mean_cache_alignment, 4),
                "max_cache_alignment_kCk": round(max_cache_alignment, 4),
                "relative_update_size": round(relative_update, 8),
                "update_norm_fro": round(update_norm, 6),
                "residual_attainment": round(residual_attainment, 6),
                "solve_residual": round(solve_residual, 10),
                "mean_inverse_gain": round(mean_gain, 8),
                "min_inverse_gain": round(min_gain, 8),
                "max_inverse_gain": round(max_gain, 8),
                "key_projection_loss": round(projection_loss, 6),
                "key_term_fro": round(key_term_fro, 4),
                "cache_term_fro": round(cache_term_fro, 4),
            }
            gamma_result["layers"][str(layer)] = layer_result
            print(f"    Layer {layer}: dom={cache_dominance:.3f}, "
                  f"gain={mean_gain:.6f}, ||ΔW||/||W||={relative_update:.6f}, "
                  f"attain={residual_attainment:.4f}, res={solve_residual:.2e}")

            del P_i, C_i, K_gram, LHS, Pk, Ck
            torch.cuda.empty_cache()

        results.append(gamma_result)

    # ─── Also run P vs I comparison ───
    print(f"\n  --- Projection ablation: P vs I ---")
    proj_ablation = {"type": "projection_ablation", "layers": {}}
    for i, layer in enumerate(layers):
        if all_layer_ks[i] is None:
            continue
        layer_ks = all_layer_ks[i]
        P_i = P[i].cuda()
        C_i = cache_c[i].cuda()
        d = P_i.shape[0]

        K_gram = layer_ks @ layer_ks.T
        I_d = torch.eye(d, device="cuda")

        # With P (standard AlphaEdit)
        LHS_P = P_i @ (K_gram + C_i) + L2 * I_d
        # With I (no projection — tests whether P matters)
        LHS_I = (K_gram + C_i) + L2 * I_d

        Pk = P_i @ layer_ks

        # Solve with same RHS for fair comparison
        torch.manual_seed(args.seed + i)
        v = torch.randn(layer_ks.shape[1], 10, device="cuda")
        v = v / v.norm(dim=0, keepdim=True)

        try:
            RHS_P = Pk @ v
            X_P = torch.linalg.solve(LHS_P, RHS_P)
            gain_P = (layer_ks * torch.linalg.solve(LHS_P, Pk)).sum(dim=0).mean().item()
            update_P = X_P.norm().item()
            res_P = (LHS_P @ X_P - RHS_P).norm().item() / RHS_P.norm().item()
        except:
            gain_P = update_P = res_P = float("nan")

        try:
            RHS_I = layer_ks @ v  # I @ K = K
            X_I = torch.linalg.solve(LHS_I, RHS_I)
            gain_I = (layer_ks * torch.linalg.solve(LHS_I, layer_ks)).sum(dim=0).mean().item()
            update_I = X_I.norm().item()
            res_I = (LHS_I @ X_I - RHS_I).norm().item() / RHS_I.norm().item()
        except:
            gain_I = update_I = res_I = float("nan")

        # Key projection loss: ||Pk - k|| / ||k||
        diff = (P_i @ layer_ks - layer_ks).norm().item()
        k_norm = layer_ks.norm().item()
        key_loss = diff / k_norm if k_norm > 0 else 0.0

        proj_ablation["layers"][str(layer)] = {
            "gain_with_P": round(gain_P, 8),
            "gain_with_I": round(gain_I, 8),
            "gain_ratio_P_over_I": round(gain_P / gain_I, 6) if gain_I and gain_I != 0 else float("nan"),
            "update_norm_P": round(update_P, 6),
            "update_norm_I": round(update_I, 6),
            "solve_residual_P": round(res_P, 10),
            "solve_residual_I": round(res_I, 10),
            "key_projection_loss": round(key_loss, 6),
        }
        print(f"    Layer {layer}: gain(P)={gain_P:.6f}, gain(I)={gain_I:.6f}, "
              f"ratio={gain_P/gain_I:.4f}, ||Pk-k||/||k||={key_loss:.6f}, "
              f"res(P)={res_P:.2e}, res(I)={res_I:.2e}")

        del P_i, C_i, K_gram, LHS_P, LHS_I, I_d
        torch.cuda.empty_cache()

    results.append(proj_ablation)

    # ─── Save results ───
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n{'=' * 70}")
    print(f"Results saved: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
