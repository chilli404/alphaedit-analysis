#!/usr/bin/env python3
"""
Phase 0 + Phase 1: Update-Level Interference from Checkpoints

Phase 0: Verification checks (no GPU needed for most)
  0a: Key provenance (dim 14336, layer 6)
  0b: Off-by-one (batch metadata)
  0c: Case alignment (ordering -> key bank -> behavioral eval)
  0d: Cohort overlap between orderings
  0e: Applied-delta capture (GPU needed - separate flag)
  0f: Reconstruction (checkpoint differences compose)

Phase 1: Coarse checkpoint-difference interference
  - Loads W from batch_{9,19,29,39,49} checkpoints
  - Computes DeltaW for each 1K-edit interval
  - Measures path interference, net displacement, Frobenius-normalized
  - Only accumulates for keys installed BEFORE interval start

Usage:
    # Phase 0 only (quick, no GPU for 0a-0d,0f):
    python src/experiments/interference_from_checkpoints.py --phase 0

    # Phase 1 (needs checkpoints, no GPU):
    python src/experiments/interference_from_checkpoints.py --phase 1

    # Both:
    python src/experiments/interference_from_checkpoints.py --phase 0 1

    # With per-case behavioral eval (GPU needed):
    python src/experiments/interference_from_checkpoints.py --phase 1 --eval_behavioral
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from paths import get_checkpoint_root, get_result_root

# Constants
LAYER = 6
WEIGHT_KEY = "model.layers.6.mlp.down_proj.weight"
WEIGHT_SHAPE = (4096, 14336)  # [d_out, d_in]
KEY_DIM = 14336
NUM_EDITS_PER_BATCH = 100
BATCH_INDICES = [9, 19, 29, 39, 49]  # 1K, 2K, 3K, 4K, 5K
INTERVALS = [(9, 19), (19, 29), (29, 39), (39, 49)]


def resolve_checkpoint_base() -> Path:
    """Resolve checkpoint base directory.

    Uses CHECKPOINT_ROOT env var (set by shell launcher) or falls back
    to get_checkpoint_root() from paths.py (~/.cache/alphaedit_checkpoints).
    The shell launcher is responsible for choosing S3 vs local.
    """
    return get_checkpoint_root()


def load_ordering(ordering_name: str, seed: int) -> List[int]:
    """Load ordering JSON -> list of case_ids in edit order."""
    path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering_name}_seed{seed}.json"
    with open(path) as f:
        records = json.load(f)
    return [r["case_id"] for r in records]


def load_keys(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load key geometry. Returns (keys [5000, 14336], case_ids [5000])."""
    path = get_result_root() / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer6.npz"
    data = np.load(path)
    assert data["keys"].shape == (5000, KEY_DIM), f"Unexpected key shape: {data['keys'].shape}"
    assert data["layer"].item() == LAYER, f"Unexpected layer: {data['layer'].item()}"
    return data["keys"], data["case_ids"]


def load_checkpoint_weight(ckpt_path: Path) -> np.ndarray:
    """Load layer-6 down_proj weight from checkpoint. Returns [4096, 14336] numpy."""
    import torch
    weights = torch.load(ckpt_path / "model_weights.pt", map_location="cpu")
    if WEIGHT_KEY not in weights:
        available = list(weights.keys())
        raise KeyError(f"{WEIGHT_KEY} not in checkpoint. Available: {available}")
    w = weights[WEIGHT_KEY]
    assert w.shape == WEIGHT_SHAPE, f"Unexpected weight shape: {w.shape}"
    return w.float().numpy()


def case_id_to_key_index(case_ids: np.ndarray) -> Dict[int, int]:
    """Map case_id -> index in key bank."""
    return {int(cid): i for i, cid in enumerate(case_ids)}


def get_installation_batch(ordering_case_ids: List[int]) -> Dict[int, int]:
    """Map case_id -> batch index where it was installed (0-indexed)."""
    result = {}
    for pos, cid in enumerate(ordering_case_ids):
        batch_idx = pos // NUM_EDITS_PER_BATCH
        result[cid] = batch_idx
    return result


# ─── Phase 0: Verification ─────────────────────────────────────────────────

def check_0a_key_provenance(keys: np.ndarray, case_ids: np.ndarray):
    """0a: Verify keys are 14336-dim input activations to layer-6 down_proj."""
    print("\n[0a] Key Provenance")
    print(f"  Key shape: {keys.shape} (expect (5000, {KEY_DIM}))")
    print(f"  Key dtype: {keys.dtype}")
    print(f"  Case IDs shape: {case_ids.shape}")
    print(f"  Key norm stats: mean={np.linalg.norm(keys, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(keys, axis=1).std():.4f}")

    assert keys.shape == (5000, KEY_DIM), f"FAIL: shape {keys.shape}"
    assert keys.dtype == np.float32, f"FAIL: dtype {keys.dtype}"
    assert len(case_ids) == 5000, f"FAIL: {len(case_ids)} case_ids"
    # Keys should not be all-zero or trivially identical
    assert np.linalg.norm(keys, axis=1).min() > 0, "FAIL: zero-norm keys found"
    assert keys.std() > 1e-6, "FAIL: keys have trivial variance"
    print("  PASS")


def check_0b_off_by_one(ckpt_base: Path, alg: str, ordering: str, seed: int):
    """0b: Verify checkpoint metadata matches expected edit counts."""
    print(f"\n[0b] Off-by-one ({alg}/{ordering}/seed{seed})")
    ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"

    for batch_idx in BATCH_INDICES:
        batch_path = ckpt_dir / f"batch_{batch_idx}"
        meta_path = batch_path / "metadata.json"
        expected_edits = (batch_idx + 1) * NUM_EDITS_PER_BATCH

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            actual_edits = meta.get("total_edits", "?")
            status = "PASS" if actual_edits == expected_edits else "FAIL"
            print(f"  batch_{batch_idx}: total_edits={actual_edits} (expect {expected_edits}) [{status}]")
            if status == "FAIL":
                raise AssertionError(
                    f"batch_{batch_idx}: expected {expected_edits} edits, got {actual_edits}"
                )
        elif (batch_path / "model_weights.pt").exists():
            # Metadata missing but weights exist — infer from batch index
            print(f"  batch_{batch_idx}: no metadata.json, weights exist, "
                  f"inferred total_edits={expected_edits}")
        else:
            print(f"  batch_{batch_idx}: NOT FOUND at {batch_path}")
            raise FileNotFoundError(f"Checkpoint missing: {batch_path}")

    print("  PASS")


def check_0c_case_alignment(ordering_case_ids: List[int], key_case_ids: np.ndarray):
    """0c: Verify ordering case_ids are a subset of key bank case_ids."""
    print("\n[0c] Case Alignment")
    ordering_set = set(ordering_case_ids)
    key_set = set(int(c) for c in key_case_ids)

    in_both = ordering_set & key_set
    only_ordering = ordering_set - key_set
    only_keys = key_set - ordering_set

    print(f"  Ordering case_ids: {len(ordering_set)}")
    print(f"  Key bank case_ids: {len(key_set)}")
    print(f"  Intersection: {len(in_both)}")
    print(f"  Only in ordering: {len(only_ordering)}")
    print(f"  Only in key bank: {len(only_keys)}")

    assert len(only_ordering) == 0, (
        f"FAIL: {len(only_ordering)} case_ids in ordering not in key bank"
    )
    assert len(in_both) == 5000, f"FAIL: expected 5000 matches, got {len(in_both)}"
    print("  PASS")


def check_0d_cohort_overlap(seed: int):
    """0d: Compute first-1K cohort overlap between orderings."""
    print("\n[0d] Cohort Overlap")
    clustered = load_ordering("key_clustered", seed)
    dispersed = load_ordering("key_dispersed", seed)

    first_1k_clust = set(clustered[:1000])
    first_1k_disp = set(dispersed[:1000])
    overlap = first_1k_clust & first_1k_disp

    print(f"  First-1K clustered: {len(first_1k_clust)} unique case_ids")
    print(f"  First-1K dispersed: {len(first_1k_disp)} unique case_ids")
    print(f"  Overlap: {len(overlap)} ({100*len(overlap)/1000:.1f}%)")
    print(f"  -> Confirms need for all-5000 tracking in secondary analysis")

    # Also compute eligible paired cases at different horizons
    clust_install = get_installation_batch(clustered)
    disp_install = get_installation_batch(dispersed)

    for horizon_batches, horizon_name in [(10, "1K-exposure"), (20, "2K-exposure")]:
        max_install_batch = 49 - horizon_batches
        eligible = [
            cid for cid in set(clustered) & set(dispersed)
            if clust_install[cid] <= max_install_batch
            and disp_install[cid] <= max_install_batch
        ]
        print(f"  {horizon_name} eligible pairs: {len(eligible)}")

    print("  PASS (informational)")
    return len(overlap)


def check_0f_reconstruction(ckpt_base: Path, alg: str, ordering: str, seed: int):
    """0f: Verify sum of interval deltas equals endpoint difference."""
    print(f"\n[0f] Reconstruction ({alg}/{ordering}/seed{seed})")
    ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"

    W_start = load_checkpoint_weight(ckpt_dir / "batch_9")
    W_end = load_checkpoint_weight(ckpt_dir / "batch_49")

    # Sum interval deltas
    accumulated = np.zeros_like(W_start)
    prev_W = W_start
    for _, end_batch in INTERVALS:
        W_next = load_checkpoint_weight(ckpt_dir / f"batch_{end_batch}")
        accumulated += (W_next - prev_W)
        prev_W = W_next

    # Compare
    direct = W_end - W_start
    error = accumulated - direct
    rel_error = np.linalg.norm(error) / (np.linalg.norm(direct) + 1e-10)
    max_abs_error = np.abs(error).max()
    # Compute in float64 to avoid dot product exceeding product of norms
    acc_f64 = accumulated.astype(np.float64).ravel()
    dir_f64 = direct.astype(np.float64).ravel()
    cosine_sim = np.dot(acc_f64, dir_f64) / (
        np.linalg.norm(acc_f64) * np.linalg.norm(dir_f64) + 1e-10
    )

    print(f"  ||direct delta||_F: {np.linalg.norm(direct):.6f}")
    print(f"  ||accumulated delta||_F: {np.linalg.norm(accumulated):.6f}")
    print(f"  Relative Frobenius error: {rel_error:.2e}")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.10f}")

    # These should be essentially zero (float32 accumulation noise only)
    if rel_error > 1e-5:
        print(f"  WARNING: relative error {rel_error:.2e} larger than expected")
    else:
        print("  PASS")

    return {"rel_error": float(rel_error), "max_abs_error": float(max_abs_error),
            "cosine_sim": float(cosine_sim)}


def run_phase0(seed: int, ckpt_base: Path):
    """Run all Phase 0 verification checks."""
    print("=" * 70)
    print("PHASE 0: Verification Checks")
    print("=" * 70)

    keys, case_ids = load_keys(seed)

    # 0a: Key provenance
    check_0a_key_provenance(keys, case_ids)

    # 0b: Off-by-one for all 4 conditions
    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]
    for alg, ordering in conditions:
        try:
            check_0b_off_by_one(ckpt_base, alg, ordering, seed)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    # 0c: Case alignment
    clustered_ids = load_ordering("key_clustered", seed)
    check_0c_case_alignment(clustered_ids, case_ids)

    dispersed_ids = load_ordering("key_dispersed", seed)
    check_0c_case_alignment(dispersed_ids, case_ids)

    # 0d: Cohort overlap
    check_0d_cohort_overlap(seed)

    # 0f: Reconstruction (for conditions with checkpoints available)
    for alg, ordering in conditions:
        try:
            check_0f_reconstruction(ckpt_base, alg, ordering, seed)
        except (FileNotFoundError, KeyError) as e:
            print(f"\n[0f] Reconstruction ({alg}/{ordering}) SKIP: {e}")

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)


# ─── Phase 1: Coarse Checkpoint-Difference Interference ────────────────────

def compute_coarse_interference(
    ckpt_dir: Path,
    keys: np.ndarray,
    case_ids: np.ndarray,
    ordering_case_ids: List[int],
    ordering_name: str,
) -> Dict:
    """
    Compute coarse interference metrics from checkpoint differences.

    Returns dict with per-key path interference, net displacement, etc.
    """
    print(f"\n  Computing interference for {ordering_name}...")

    # Build case_id -> key_index mapping
    cid_to_kidx = case_id_to_key_index(case_ids)

    # Build case_id -> installation batch
    install_batch = get_installation_batch(ordering_case_ids)

    # Map ordering positions to key indices
    # ordering_case_ids[pos] -> case_id -> key_index
    key_indices = np.array([cid_to_kidx[cid] for cid in ordering_case_ids])

    # Load all checkpoint weights
    print("    Loading checkpoints...")
    weights = {}
    for batch_idx in BATCH_INDICES:
        batch_path = ckpt_dir / f"batch_{batch_idx}"
        weights[batch_idx] = load_checkpoint_weight(batch_path)
        print(f"      batch_{batch_idx}: loaded ({weights[batch_idx].shape})")

    # W_{1K} = batch_9 (shared baseline for first-1K cohort)
    W_1K = weights[9]

    # Initialize accumulators
    n_keys = 5000
    U_path = np.zeros(n_keys, dtype=np.float64)
    I_fro_accum = np.zeros(n_keys, dtype=np.float64)
    n_intervals_accumulated = np.zeros(n_keys, dtype=np.int32)

    # Per-interval results
    interval_results = []

    for start_batch, end_batch in INTERVALS:
        delta_W = weights[end_batch] - weights[start_batch]  # [4096, 14336]
        dW_fro = np.linalg.norm(delta_W)

        # Eligible keys: installed STRICTLY before interval start
        # batch_9 means installed at or before batch 9 (first 1K edits)
        # For interval [9, 19]: eligible = installed before batch 9 end = batches 0-9
        # "installed before interval start" means installation_batch < start_batch
        # BUT batch_9 IS the 1K checkpoint. Keys in batches 0-9 were installed BY batch_9.
        # For interval [9, 19], we want keys already installed at batch_9 = first 1K.
        # Correct interpretation: eligible if installation_batch <= start_batch
        eligible_mask = np.zeros(n_keys, dtype=bool)
        for pos, cid in enumerate(ordering_case_ids):
            kidx = cid_to_kidx[cid]
            batch = pos // NUM_EDITS_PER_BATCH
            if batch <= start_batch:
                eligible_mask[kidx] = True

        n_eligible = eligible_mask.sum()
        K_eligible = keys[eligible_mask]  # [n_eligible, 14336]

        # Compute effects: delta_W @ K.T -> [4096, n_eligible]
        effects = delta_W @ K_eligible.T
        norms = np.linalg.norm(effects, axis=0)  # [n_eligible]

        # Key norms for Frobenius normalization
        key_norms = np.linalg.norm(K_eligible, axis=1)
        I_fro = norms / (dW_fro * key_norms + 1e-10)

        # Accumulate
        eligible_indices = np.where(eligible_mask)[0]
        U_path[eligible_indices] += norms
        I_fro_accum[eligible_indices] += I_fro
        n_intervals_accumulated[eligible_indices] += 1

        interval_results.append({
            "interval": f"batch_{start_batch}_to_{end_batch}",
            "edits_range": f"{(start_batch+1)*100}-{(end_batch+1)*100}",
            "n_eligible": int(n_eligible),
            "dW_fro": float(dW_fro),
            "mean_interference_norm": float(norms.mean()),
            "median_interference_norm": float(np.median(norms)),
            "std_interference_norm": float(norms.std()),
            "mean_I_fro": float(I_fro.mean()),
            "median_I_fro": float(np.median(I_fro)),
        })

        print(f"    [{start_batch}->{end_batch}] eligible={n_eligible}, "
              f"||dW||_F={dW_fro:.4f}, mean_interf={norms.mean():.6f}, "
              f"mean_I_fro={I_fro.mean():.6f}")

    # Net displacement for first-1K (W_{1K} is valid baseline)
    first_1K_positions = key_indices[:1000]  # key indices for first 1K edits
    K_first1K = keys[first_1K_positions]  # [1000, 14336]

    W_5K = weights[49]
    cumulative_dW = W_5K - W_1K  # [4096, 14336]
    net_effects = cumulative_dW @ K_first1K.T  # [4096, 1000]
    U_net_first1K = np.linalg.norm(net_effects, axis=0)  # [1000]

    # Baseline output norms (W_{1K} @ k_i)
    baseline_effects = W_1K @ K_first1K.T  # [4096, 1000]
    baseline_norms = np.linalg.norm(baseline_effects, axis=0)  # [1000]

    # Relative net displacement
    d_rel_first1K = U_net_first1K / (baseline_norms + 1e-10)

    # Also compute path for first-1K only (subset of U_path)
    U_path_first1K = U_path[first_1K_positions]
    I_fro_first1K = I_fro_accum[first_1K_positions]

    print(f"\n    First-1K summary:")
    print(f"      U_path: mean={U_path_first1K.mean():.6f}, "
          f"median={np.median(U_path_first1K):.6f}, std={U_path_first1K.std():.6f}")
    print(f"      U_net: mean={U_net_first1K.mean():.6f}, "
          f"median={np.median(U_net_first1K):.6f}")
    print(f"      d_rel: mean={d_rel_first1K.mean():.6f}, "
          f"median={np.median(d_rel_first1K):.6f}")
    print(f"      baseline_norm: mean={baseline_norms.mean():.6f}")

    return {
        "ordering": ordering_name,
        "intervals": interval_results,
        "first_1K": {
            "case_ids": [int(ordering_case_ids[i]) for i in range(1000)],
            "key_indices": first_1K_positions.tolist(),
            "U_path": U_path_first1K.tolist(),
            "U_net": U_net_first1K.tolist(),
            "d_rel": d_rel_first1K.tolist(),
            "I_fro_mean": (I_fro_first1K / 4).tolist(),  # avg over 4 intervals
            "baseline_output_norm": baseline_norms.tolist(),
        },
        "all_5K": {
            "case_ids": [int(cid) for cid in ordering_case_ids],
            "key_indices": key_indices.tolist(),
            "U_path": U_path.tolist(),
            "I_fro_accum": I_fro_accum.tolist(),
            "n_intervals": n_intervals_accumulated.tolist(),
            "installation_batch": [
                install_batch[cid] for cid in ordering_case_ids
            ],
        },
    }


def run_phase1(seed: int, ckpt_base: Path, output_dir: Path):
    """Run Phase 1 coarse interference for all 4 conditions."""
    print("\n" + "=" * 70)
    print("PHASE 1: Coarse Checkpoint-Difference Interference")
    print("=" * 70)

    keys, case_ids = load_keys(seed)

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    all_results = {}

    for alg, ordering in conditions:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"

        # Check checkpoints exist
        missing = [
            b for b in BATCH_INDICES
            if not (ckpt_dir / f"batch_{b}" / "model_weights.pt").exists()
        ]
        if missing:
            print(f"\n  SKIP {condition_name}: missing batches {missing}")
            continue

        ordering_case_ids = load_ordering(f"key_{ordering.replace('key_', '')}" if not ordering.startswith("key_") else ordering, seed)

        result = compute_coarse_interference(
            ckpt_dir=ckpt_dir,
            keys=keys,
            case_ids=case_ids,
            ordering_case_ids=ordering_case_ids,
            ordering_name=condition_name,
        )
        all_results[condition_name] = result

    # Save per-condition results: results/interference/{alg}/{ordering}/seed{seed}/phase1_coarse.json
    for cond_name, result in all_results.items():
        alg, ordering = cond_name.split("/")
        cond_dir = output_dir / alg / ordering / f"seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_path = cond_dir / "phase1_coarse.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Print stop/no-go summary
    print("\n" + "=" * 70)
    print("STOP/NO-GO ASSESSMENT")
    print("=" * 70)

    for cond_name, result in all_results.items():
        first_1k = result["first_1K"]
        U_path = np.array(first_1k["U_path"])

        # Basic variation check
        cv = U_path.std() / (U_path.mean() + 1e-10)
        print(f"\n  {cond_name}:")
        print(f"    U_path CV (coefficient of variation): {cv:.4f}")
        print(f"    U_path range: [{U_path.min():.6f}, {U_path.max():.6f}]")
        print(f"    -> {'Meaningful variation' if cv > 0.1 else 'LOW variation'}")

    print("\n  NOTE: Retained/forgotten classification requires per-case behavioral eval.")
    print("  Run with --eval_behavioral to classify and complete stop/no-go.")

    return all_results


# ─── Per-Case Behavioral Evaluation ────────────────────────────────────────

def run_percase_eval(
    seed: int,
    ckpt_base: Path,
    output_dir: Path,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
):
    """
    Evaluate per-case efficacy at the 5K checkpoint for retained/forgotten classification.
    GPU REQUIRED.
    """
    # Import model_download first — patches filelock to prevent hangs
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from model_download import resolve_model_path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print("PER-CASE BEHAVIORAL EVALUATION (GPU)")
    print("=" * 70)

    model_name = resolve_model_path(model_name)

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    all_percase = {}

    for alg, ordering in conditions:
        condition_name = f"{alg}/{ordering}"
        ckpt_dir = ckpt_base / "matched_ordering" / alg / ordering / f"seed{seed}"
        ckpt_5k = ckpt_dir / "batch_49"

        if not (ckpt_5k / "model_weights.pt").exists():
            print(f"\n  SKIP {condition_name}: no batch_49 checkpoint")
            continue

        print(f"\n  Evaluating {condition_name} at 5K checkpoint...")

        # Load ordering to get the records
        ordering_name = ordering if ordering.startswith("key_") else f"key_{ordering}"
        ordering_path = get_result_root() / "matched_ordering" / "orderings" / f"{ordering_name}_seed{seed}.json"
        with open(ordering_path) as f:
            records = json.load(f)

        # Load model with 5K checkpoint weights
        if alg == conditions[0][0] and ordering == conditions[0][1]:
            # First condition: load from scratch
            # Use float16 + right-padding to match vendor evaluate.py protocol
            print(f"    Loading base model: {model_name}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
            tok.pad_token = tok.eos_token
            tok.padding_side = "right"  # Full protocol requires right-padding
        else:
            print("    Reusing loaded model, swapping weights...")

        # Apply checkpoint weights (float16 to match model dtype)
        weights = torch.load(str(ckpt_5k / "model_weights.pt"), map_location="cuda")
        param_dict = dict(model.named_parameters())
        for name, tensor in weights.items():
            if name in param_dict:
                param_dict[name].data.copy_(tensor.cuda().half())
        del weights
        torch.cuda.empty_cache()

        # Evaluate first 1K records using full multi-token protocol
        # (matches vendor evaluate.py and original full_eval results)
        first_1k_records = records[:1000]
        percase_results = _evaluate_full_protocol(model, tok, first_1k_records)

        all_percase[condition_name] = percase_results
        eff_vals = [r["efficacy"] for r in percase_results]
        n_forgotten = sum(1 for e in eff_vals if e < 0.5)
        print(f"    First-1K: mean_eff={np.mean(eff_vals):.4f}, "
              f"forgotten={n_forgotten}/{len(eff_vals)}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Save per-condition: results/interference/{alg}/{ordering}/seed{seed}/percase_eval.json
    for cond_name, results in all_percase.items():
        alg, ordering = cond_name.split("/")
        cond_dir = output_dir / alg / ordering / f"seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_path = cond_dir / "percase_eval.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out_path}")

    return all_percase


def _evaluate_full_protocol(model, tok, records: List[Dict]) -> List[Dict]:
    """
    Full multi-token evaluation matching vendor evaluate.py protocol.

    For each prompt, appends both target_new and target_true, computes
    per-token log-probability, and checks argmax correctness at each position.
    This matches the metric used in the paper's retention measurements.
    """
    import torch
    from itertools import chain

    results = []
    is_llama = "llama" in model.config._name_or_path.lower()

    for i, record in enumerate(records):
        rw = record["requested_rewrite"]
        subject = rw["subject"]
        target_new = rw["target_new"]["str"]
        target_true = rw["target_true"]["str"]

        rewrite_prompts = [rw["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]

        prob_prompts = [rewrite_prompts, paraphrase_prompts]
        which_correct = [
            [0] * len(rewrite_prompts),    # target_new is correct
            [0] * len(paraphrase_prompts), # target_new is correct
        ]

        all_prefixes = list(chain(*prob_prompts))
        all_which = list(chain(*which_correct))

        # Compute correctness using full multi-token protocol
        targets_correct = _test_batch_prediction(
            model, tok, all_prefixes, all_which, target_new, target_true, is_llama
        )

        n_rw = len(rewrite_prompts)
        results.append({
            "case_id": record["case_id"],
            "efficacy": float(np.mean(targets_correct[:n_rw])),
            "paraphrase": float(np.mean(targets_correct[n_rw:])),
        })

        if (i + 1) % 200 == 0:
            print(f"      [{i+1}/{len(records)}] "
                  f"eff={np.mean([r['efficacy'] for r in results]):.4f}")

    return results


def _test_batch_prediction(
    model, tok, prefixes: List[str], which_correct: List[int],
    target_new: str, target_true: str, is_llama: bool,
) -> List[bool]:
    """
    Full multi-token evaluation: checks argmax correctness at each target token position.
    Matches vendor evaluate.py / eval_seqreg_checkpoints.py test_batch_prediction exactly.
    """
    import torch

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok = tok(f" {target_new}")["input_ids"]
    b_tok = tok(f" {target_true}")["input_ids"]

    if is_llama:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [l - 1 for l in prefix_lens]

    choice_a_len, choice_b_len = len(a_tok), len(b_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if is_llama:
        logits = logits[:, 1:, :]

    targets_correct = []
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        cur_tok = a_tok if i % 2 == 0 else b_tok

        # Check if this is a "correct" entry (even=target_new, odd=target_true)
        wc = which_correct[i // 2]
        is_correct_target = (wc == 0 and i % 2 == 0) or (wc == 1 and i % 2 == 1)

        if is_correct_target:
            correct = True
            for j in range(cur_len):
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok[j]:
                    correct = False
                    break
            targets_correct.append(correct)

    del logits, prompt_tok
    torch.cuda.empty_cache()

    return targets_correct


# ─── Stop/No-Go with Behavioral Labels ─────────────────────────────────────

def run_stop_nogo_from_conditions(output_dir: Path, seed: int):
    """Load per-condition phase1 + percase_eval and run stop/no-go."""
    print("\n" + "=" * 70)
    print("STOP/NO-GO ANALYSIS (with behavioral labels)")
    print("=" * 70)

    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"),
        ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"),
    ]

    for alg, ordering in conditions:
        cond_dir = output_dir / alg / ordering / f"seed{seed}"
        phase1_path = cond_dir / "phase1_coarse.json"
        percase_path = cond_dir / "percase_eval.json"

        cond_name = f"{alg}/{ordering}"

        if not phase1_path.exists():
            print(f"\n  {cond_name}: no phase1 data, skip")
            continue
        if not percase_path.exists():
            print(f"\n  {cond_name}: no per-case eval, skip")
            continue

        with open(phase1_path) as f:
            phase1 = json.load(f)
        with open(percase_path) as f:
            percase = json.load(f)

        first_1k = phase1["first_1K"]
        case_ids = first_1k["case_ids"]
        U_path = np.array(first_1k["U_path"])
        d_rel = np.array(first_1k["d_rel"])

        # Match per-case eval to ordering
        eval_by_cid = {r["case_id"]: r for r in percase}

        # Classify retained vs forgotten (earliest observed retention >= 0.5)
        retained_mask = np.zeros(len(case_ids), dtype=bool)
        forgotten_mask = np.zeros(len(case_ids), dtype=bool)
        no_eval = 0

        for i, cid in enumerate(case_ids):
            if cid in eval_by_cid:
                eff = eval_by_cid[cid]["efficacy"]
                if eff >= 0.5:
                    retained_mask[i] = True
                else:
                    forgotten_mask[i] = True
            else:
                no_eval += 1

        n_retained = retained_mask.sum()
        n_forgotten = forgotten_mask.sum()

        print(f"\n  {cond_name}:")
        print(f"    Retained: {n_retained}, Forgotten: {n_forgotten}, No eval: {no_eval}")

        if n_forgotten < 30:
            print(f"    -> DESCRIPTIVE ONLY (forgotten < 30)")

        # Compute statistics
        U_path_retained = U_path[retained_mask]
        U_path_forgotten = U_path[forgotten_mask]
        d_rel_retained = d_rel[retained_mask]
        d_rel_forgotten = d_rel[forgotten_mask]

        if n_forgotten > 0 and n_retained > 0:
            median_diff = np.median(U_path_forgotten) - np.median(U_path_retained)
            mean_diff = np.mean(U_path_forgotten) - np.mean(U_path_retained)

            print(f"    U_path (forgotten): median={np.median(U_path_forgotten):.6f}, "
                  f"mean={np.mean(U_path_forgotten):.6f}")
            print(f"    U_path (retained):  median={np.median(U_path_retained):.6f}, "
                  f"mean={np.mean(U_path_retained):.6f}")
            print(f"    Median diff (forg - ret): {median_diff:.6f}")
            print(f"    Mean diff (forg - ret): {mean_diff:.6f}")
            print(f"    Direction: {'forgotten > retained' if median_diff > 0 else 'retained >= forgotten'}")

            # Cliff's delta
            from itertools import product
            n_greater = sum(1 for f, r in product(U_path_forgotten, U_path_retained) if f > r)
            n_less = sum(1 for f, r in product(U_path_forgotten, U_path_retained) if f < r)
            n_total = n_forgotten * n_retained
            cliffs_d = (n_greater - n_less) / n_total
            print(f"    Cliff's delta: {cliffs_d:.4f}")

            # Same for d_rel
            print(f"    d_rel (forgotten): median={np.median(d_rel_forgotten):.6f}")
            print(f"    d_rel (retained):  median={np.median(d_rel_retained):.6f}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0+1: Update-Level Interference from Checkpoints"
    )
    parser.add_argument("--phase", nargs="*", type=int, default=None,
                        help="Which phases to run (0, 1, or both). Omit to skip phases.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_base", type=str, default=None,
                        help="Checkpoint base directory (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/interference/)")
    parser.add_argument("--eval_behavioral", action="store_true",
                        help="Run per-case behavioral eval at 5K checkpoint (GPU needed)")
    parser.add_argument("--stop_nogo", action="store_true",
                        help="Run stop/no-go analysis (requires prior phase1 + eval_behavioral)")
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    # Resolve paths
    if args.checkpoint_base:
        ckpt_base = Path(args.checkpoint_base).expanduser()
    else:
        ckpt_base = resolve_checkpoint_base()
    print(f"Checkpoint base: {ckpt_base}")

    output_dir = Path(args.output_dir) if args.output_dir else get_result_root() / "interference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run phases (default to [0, 1] if --phase not specified and no other action requested)
    phases = args.phase if args.phase is not None else (
        [] if (args.eval_behavioral or args.stop_nogo) else [0, 1]
    )

    if 0 in phases:
        run_phase0(args.seed, ckpt_base)

    if 1 in phases:
        run_phase1(args.seed, ckpt_base, output_dir)

    if args.eval_behavioral:
        run_percase_eval(args.seed, ckpt_base, output_dir, args.model_name)

    if args.stop_nogo:
        run_stop_nogo_from_conditions(output_dir, args.seed)


if __name__ == "__main__":
    main()
