#!/usr/bin/env python3
"""
Direct Logit-Damage Intervention Runner

A/B experiment: install N batches of edits, then apply a HIGH-cosine or
LOW-cosine future batch and measure the immediate target-logit change for
every focal edit. Provides direct causal evidence that key-geometry similarity
→ interference damage.

Design:
  1. Use fixed-batch assignment (batch membership constant across conditions)
  2. Install first N batches in canonical order → save model state + cache_c
  3. Rank remaining batches by mean max cosine to focal-edit keys
  4. For each (HIGH, LOW) trial pair:
     a. Measure baseline target logprobs for all focal edits
     b. Apply HIGH batch → measure logprobs + ||ΔW @ k_i|| → restore
     c. Apply LOW batch → measure logprobs + ||ΔW @ k_i|| → restore
  5. Save paired results + summary statistics

Output:
  results/logit_damage/seed{SEED}/intervention_results.json
  - Per-focal-edit × per-trial: paired logprob differences + damage norms
  - Summary: trial-level paired t-test, fraction more damaged by HIGH

Usage:
    python src/runners/logit_damage_runner.py --seed 42
    python src/runners/logit_damage_runner.py --seed 42 --install_batches 10 --n_trials 10
    python src/runners/logit_damage_runner.py --seed 42 --checkpoint_base /path/to/ckpt
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))

from paths import get_alphaedit_root, get_checkpoint_root, get_result_root

ALPHAEDIT_ROOT = get_alphaedit_root()


# ─── Batch Ranking (CPU, runs in outer process) ─────────────────────────────


def rank_future_batches(
    batches, install_batches, keys, case_id_to_kidx,
):
    """Rank future batches by mean max-cosine to focal-edit keys.

    Returns list of (batch_idx, mean_max_cosine) sorted descending.
    """
    focal_keys = []
    for b_idx in range(install_batches):
        for r in batches[b_idx]:
            kidx = case_id_to_kidx.get(r["case_id"], -1)
            if kidx >= 0:
                focal_keys.append(keys[kidx])

    focal = np.array(focal_keys)
    focal_norms = np.linalg.norm(focal, axis=1, keepdims=True)
    focal_normed = focal / np.maximum(focal_norms, 1e-8)

    scores = []
    for b_idx in range(install_batches, len(batches)):
        batch_keys = []
        for r in batches[b_idx]:
            kidx = case_id_to_kidx.get(r["case_id"], -1)
            if kidx >= 0:
                batch_keys.append(keys[kidx])
        if not batch_keys:
            scores.append((b_idx, 0.0))
            continue

        bk = np.array(batch_keys)
        bk_norms = np.linalg.norm(bk, axis=1, keepdims=True)
        bk_normed = bk / np.maximum(bk_norms, 1e-8)
        cos = focal_normed @ bk_normed.T
        mean_max = float(cos.max(axis=1).mean())
        scores.append((b_idx, mean_max))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def build_intervention_config(
    batches, ranked_scores, install_batches, n_trials, case_id_to_kidx,
):
    """Build the trial configuration for the inner GPU script."""
    high_indices = [s[0] for s in ranked_scores[:n_trials]]
    low_indices = [s[0] for s in ranked_scores[-n_trials:]][::-1]

    focal_case_ids = []
    for b_idx in range(install_batches):
        for r in batches[b_idx]:
            focal_case_ids.append(r["case_id"])

    trials = []
    for t in range(n_trials):
        hi_b = high_indices[t]
        lo_b = low_indices[t]
        hi_cos = ranked_scores[t][1]
        lo_cos = ranked_scores[-(t + 1)][1]

        hi_records = batches[hi_b]
        lo_records = batches[lo_b]

        hi_relations = {}
        lo_relations = {}
        for r in hi_records:
            rel = r["requested_rewrite"]["relation_id"]
            hi_relations[rel] = hi_relations.get(rel, 0) + 1
        for r in lo_records:
            rel = r["requested_rewrite"]["relation_id"]
            lo_relations[rel] = lo_relations.get(rel, 0) + 1

        trials.append({
            "trial": t,
            "high_batch_idx": hi_b,
            "low_batch_idx": lo_b,
            "high_cosine": hi_cos,
            "low_cosine": lo_cos,
            "high_case_ids": [r["case_id"] for r in hi_records],
            "low_case_ids": [r["case_id"] for r in lo_records],
            "high_relations": hi_relations,
            "low_relations": lo_relations,
        })

    return {
        "focal_case_ids": focal_case_ids,
        "install_batches": install_batches,
        "n_trials": n_trials,
        "trials": trials,
    }


# ─── Inner GPU Script ────────────────────────────────────────────────────────


def build_inner_script(
    seed, model_name, hparams_fname,
    batch_assignment_path, stream_path, keys_path,
    checkpoint_dir, config_path, output_path,
    install_batches, num_edits,
):
    return textwrap.dedent(f"""\
import os, sys, random, json, time, gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

seed = {seed}
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
INSTALL_BATCHES = {install_batches}
NUM_EDITS = {num_edits}

# ─── Source-inject AlphaEdit_main.py ───────────────────────────────
ae_src = Path("AlphaEdit/AlphaEdit_main.py").read_text()
ae_src = ae_src.replace("from .compute_ks", "from AlphaEdit.compute_ks")
ae_src = ae_src.replace("from .compute_z", "from AlphaEdit.compute_z")
ae_src = ae_src.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")
exec(compile(ae_src, "AlphaEdit/AlphaEdit_main.py", "exec"),
     _ae_ns := {{"__name__": "AlphaEdit.AlphaEdit_main",
                 "__file__": "AlphaEdit/AlphaEdit_main.py"}})
apply_ae = _ae_ns["apply_AlphaEdit_to_model"]
print("[LD] AlphaEdit_main.py compiled")

# ─── Load model + tokenizer ───────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[LD] Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}", torch_dtype=torch.float32).cuda()
tok = AutoTokenizer.from_pretrained("{model_name}")
tok.pad_token = tok.eos_token
print("[LD] Model loaded")

# ─── Load hparams ─────────────────────────────────────────────────
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
hparams = AlphaEditHyperParams.from_json("hparams/AlphaEdit/{hparams_fname}")
print(f"[LD] Layers: {{hparams.layers}}")

# ─── Load P ───────────────────────────────────────────────────────
P_path = Path("null_space_project.pt")
if not P_path.exists():
    P_path = Path("data/stats/Llama3-8B/wikipedia_stats/null_space_project.pt")
if not P_path.exists():
    _mn = hparams.model_name.replace("/", "_")
    P_path = Path(f"data/stats/{{_mn}}/wikipedia_stats/null_space_project.pt")
assert P_path.exists(), f"P not found at {{P_path}}"
P = torch.load(str(P_path), map_location="cpu")
print(f"[LD] P: {{P.shape}}")

# ─── Initialize cache_c ──────────────────────────────────────────
_wkey = hparams.rewrite_module_tmp.format(hparams.layers[0]) + ".weight"
d = dict(model.named_parameters())[_wkey].shape[1]
cache_c = torch.zeros(len(hparams.layers), d, d)

# ─── Load stream + batch assignment + keys ─────────────────────────
with open("{stream_path}") as f:
    stream_records = json.load(f)
with open("{batch_assignment_path}") as f:
    batch_assign = json.load(f)
with open("{config_path}") as f:
    config = json.load(f)

cid_to_rec = {{r["case_id"]: r for r in stream_records}}
batches = [[cid_to_rec[cid] for cid in bcids] for bcids in batch_assign["batches"]]
print(f"[LD] {{len(batches)}} batches loaded")

keys_data = np.load("{keys_path}")
all_keys = torch.from_numpy(keys_data["keys"]).float()
all_cids = keys_data["case_ids"].tolist()
cid_to_kidx = {{int(c): i for i, c in enumerate(all_cids)}}

# ─── Install / load checkpoint ────────────────────────────────────
ckpt_dir = Path("{checkpoint_dir}")
ckpt_batch = ckpt_dir / f"batch_{{INSTALL_BATCHES - 1}}"
if (ckpt_batch / "model_weights.pt").exists():
    print(f"[LD] Restoring checkpoint: {{ckpt_batch}}")
    _cw = torch.load(str(ckpt_batch / "model_weights.pt"), map_location="cuda")
    _pd = dict(model.named_parameters())
    for wn, wt in _cw.items():
        if wn in _pd: _pd[wn].data.copy_(wt.cuda())
    del _cw
    if (ckpt_batch / "cache_c.pt").exists():
        cache_c = torch.load(str(ckpt_batch / "cache_c.pt"), map_location="cpu")
    torch.cuda.empty_cache()
    print(f"[LD] Restored ({{INSTALL_BATCHES * NUM_EDITS}} edits)")
else:
    print(f"[LD] Installing {{INSTALL_BATCHES}} batches from scratch...")
    def _make_requests(records):
        return [{{"case_id": r["case_id"], **r["requested_rewrite"]}}
                 for r in records]
    for b in range(INSTALL_BATCHES):
        model, cache_c = apply_ae(model, tok, _make_requests(batches[b]),
                                  hparams, cache_c=cache_c, P=P)
        if (b + 1) % 5 == 0:
            print(f"  Batch {{b+1}}/{{INSTALL_BATCHES}}")
    # Save checkpoint
    ckpt_batch.mkdir(parents=True, exist_ok=True)
    _pd = dict(model.named_parameters())
    _ew = {{}}
    for layer in hparams.layers:
        wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
        _ew[wn] = _pd[wn].detach().cpu()
    torch.save(_ew, str(ckpt_batch / "model_weights.pt"))
    torch.save(cache_c.cpu(), str(ckpt_batch / "cache_c.pt"))
    with open(str(ckpt_batch / "metadata.json"), "w") as f:
        json.dump({{"batch_idx": INSTALL_BATCHES - 1, "seed": seed,
                    "total_edits": INSTALL_BATCHES * NUM_EDITS}}, f)
    print(f"[LD] Installed + saved checkpoint")
    del _ew; torch.cuda.empty_cache()

# ─── Save baseline state ──────────────────────────────────────────
_pd = dict(model.named_parameters())
baseline_w = {{}}
for layer in hparams.layers:
    wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
    baseline_w[wn] = _pd[wn].detach().cpu().clone()
baseline_cc = cache_c.clone()

# Focal-edit keys
focal_keys = []
focal_cids = config["focal_case_ids"]
for cid in focal_cids:
    kidx = cid_to_kidx.get(cid, -1)
    if kidx >= 0:
        focal_keys.append(all_keys[kidx])
    else:
        focal_keys.append(torch.zeros(all_keys.shape[1]))
focal_keys_t = torch.stack(focal_keys).float()
print(f"[LD] {{len(focal_cids)}} focal edits, keys: {{focal_keys_t.shape}}")

# ─── Measurement functions ─────────────────────────────────────────
def measure_logprobs(model, tok, records):
    lps = []
    for r in records:
        prompt = r["requested_rewrite"]["prompt"].format(
            r["requested_rewrite"]["subject"])
        target = " " + r["requested_rewrite"]["target_new"]["str"]
        enc = tok(prompt + target, return_tensors="pt").to("cuda")
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits
        tids = enc["input_ids"][0, plen:]
        if len(tids) == 0:
            lps.append(0.0); continue
        lsm = torch.log_softmax(
            logits[0, plen-1:plen-1+len(tids)].float(), dim=-1)
        lps.append(float(lsm.gather(1, tids.unsqueeze(1)).sum()))
    return lps

def compute_damage(model, baseline_w, focal_keys_t, hparams):
    damage = np.zeros(len(focal_keys_t))
    _pd = dict(model.named_parameters())
    for layer in hparams.layers:
        wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
        dw = _pd[wn].detach().cpu().float() - baseline_w[wn].float()
        eff = dw @ focal_keys_t.T
        damage += torch.linalg.norm(eff, dim=0).numpy()
    return damage

def restore():
    global cache_c
    _pd = dict(model.named_parameters())
    for wn, wt in baseline_w.items():
        _pd[wn].data.copy_(wt.cuda())
    cache_c = baseline_cc.clone()
    torch.cuda.empty_cache()

def make_requests(records):
    return [{{"case_id": r["case_id"], **r["requested_rewrite"]}}
             for r in records]

# ─── Baseline logprobs ─────────────────────────────────────────────
focal_records = [cid_to_rec[cid] for cid in focal_cids]
print("[LD] Measuring baseline logprobs...")
baseline_lps = measure_logprobs(model, tok, focal_records)
print(f"  Baseline mean: {{np.mean(baseline_lps):.4f}}")

# ─── A/B Intervention Loop ─────────────────────────────────────────
trials_out = []
for trial_cfg in config["trials"]:
    t = trial_cfg["trial"]
    hi_b = trial_cfg["high_batch_idx"]
    lo_b = trial_cfg["low_batch_idx"]
    print(f"\\nTrial {{t}}: HIGH=b{{hi_b}} (cos={{trial_cfg['high_cosine']:.4f}}) "
          f"vs LOW=b{{lo_b}} (cos={{trial_cfg['low_cosine']:.4f}})")

    # HIGH treatment
    model, cache_c = apply_ae(model, tok, make_requests(batches[hi_b]),
                              hparams, cache_c=cache_c, P=P)
    hi_lps = measure_logprobs(model, tok, focal_records)
    hi_dmg = compute_damage(model, baseline_w, focal_keys_t, hparams)
    hi_dlps = [h - b for h, b in zip(hi_lps, baseline_lps)]
    restore()
    print(f"  HIGH: mean_dmg={{hi_dmg.mean():.6f}}, mean_Δlp={{np.mean(hi_dlps):.4f}}")

    # LOW treatment
    model, cache_c = apply_ae(model, tok, make_requests(batches[lo_b]),
                              hparams, cache_c=cache_c, P=P)
    lo_lps = measure_logprobs(model, tok, focal_records)
    lo_dmg = compute_damage(model, baseline_w, focal_keys_t, hparams)
    lo_dlps = [l - b for l, b in zip(lo_lps, baseline_lps)]
    restore()
    print(f"  LOW:  mean_dmg={{lo_dmg.mean():.6f}}, mean_Δlp={{np.mean(lo_dlps):.4f}}")

    paired_lp = [h - l for h, l in zip(hi_dlps, lo_dlps)]
    frac = sum(1 for p in paired_lp if p < 0) / len(paired_lp)
    print(f"  PAIRED: mean={{np.mean(paired_lp):.4f}}, "
          f"{{100*frac:.1f}}% more damaged by HIGH")

    trials_out.append({{
        **trial_cfg,
        "high_damage": hi_dmg.tolist(),
        "low_damage": lo_dmg.tolist(),
        "high_delta_logprob": hi_dlps,
        "low_delta_logprob": lo_dlps,
        "paired_logprob_diff": paired_lp,
        "paired_damage_diff": (hi_dmg - lo_dmg).tolist(),
    }})

# ─── Summary ───────────────────────────────────────────────────────
all_paired = []
for t in trials_out:
    all_paired.extend(t["paired_logprob_diff"])
all_paired = np.array(all_paired)

from scipy import stats as _stats
_hi_means = [np.mean(t["high_delta_logprob"]) for t in trials_out]
_lo_means = [np.mean(t["low_delta_logprob"]) for t in trials_out]
if len(_hi_means) > 1:
    t_stat, p_val = _stats.ttest_rel(_hi_means, _lo_means)
else:
    t_stat, p_val = 0.0, 1.0

print(f"\\n{{'='*60}}")
print(f"SUMMARY ({{len(trials_out)}} trials × {{len(focal_cids)}} focal edits)")
print(f"  Mean paired Δlogprob (HIGH-LOW): {{all_paired.mean():.4f}}")
print(f"  More damaged by HIGH: {{(all_paired < 0).sum()}}/{{len(all_paired)}} "
      f"({{100*(all_paired < 0).mean():.1f}}%)")
print(f"  Trial-level paired t: t={{t_stat:.3f}}, p={{p_val:.4f}}")
print(f"{{'='*60}}")

out = {{
    "metadata": {{
        "seed": seed, "install_batches": INSTALL_BATCHES,
        "n_focal": len(focal_cids), "n_trials": len(trials_out),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }},
    "baseline_mean_logprob": float(np.mean(baseline_lps)),
    "trials": trials_out,
    "summary": {{
        "mean_paired_delta_lp": float(all_paired.mean()),
        "median_paired_delta_lp": float(np.median(all_paired)),
        "frac_more_damaged_by_high": float((all_paired < 0).mean()),
        "trial_t_stat": float(t_stat),
        "trial_p_value": float(p_val),
    }},
}}

_op = Path("{output_path}")
_op.parent.mkdir(parents=True, exist_ok=True)
with open(str(_op), "w") as f:
    json.dump(out, f)
print(f"Saved: {{_op}}")
""")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Direct logit-damage A/B intervention experiment"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--install_batches", type=int, default=10,
                        help="Number of batches to install before A/B (default: 10 = 1K edits)")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of HIGH/LOW trial pairs")
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--checkpoint_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from model_registry import DEFAULT_MODEL
    from model_resolve import resolve_model_path
    from setup_hparams import link_hparams
    from source_patches import patch_evaluate_file, patch_glue_eval_file

    link_hparams()
    patch_evaluate_file(ALPHAEDIT_ROOT)
    patch_glue_eval_file(ALPHAEDIT_ROOT)
    model_name = resolve_model_path(args.model_name)

    result_root = get_result_root()
    ckpt_root = get_checkpoint_root()

    # ── Load keys + stream for CPU-side batch ranking ────────────────
    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path
    assert keys_path.exists(), f"Keys not found: {keys_path}"

    keys_data = np.load(keys_path)
    all_keys = keys_data["keys"]
    all_cids = keys_data["case_ids"].tolist()
    cid_to_kidx = {int(c): i for i, c in enumerate(all_cids)}

    # ── Load or generate fixed-batch assignment ──────────────────────
    diag_dir = result_root / "matched_ordering" / "diagnostics"
    ba_path = diag_dir / f"fixed_batch_assignment_seed{args.seed}.json"
    if not ba_path.exists():
        print(f"Fixed-batch assignment not found at {ba_path}")
        print(f"Generating with: uv run python src/datasets/generate_orderings.py "
              f"--seed {args.seed} --fixed_batch")
        sys.exit(1)

    with open(ba_path) as f:
        batch_assign = json.load(f)

    # Need a stream file (any fb_ ordering) to load full records
    stream_path = result_root / "matched_ordering" / "orderings" / f"fb_high_exposure_seed{args.seed}.json"
    if not stream_path.exists():
        stream_path = result_root / "matched_ordering" / "orderings" / f"fb_low_exposure_seed{args.seed}.json"
    if not stream_path.exists():
        stream_path = result_root / "matched_ordering" / "orderings" / f"fb_random0_seed{args.seed}.json"
    assert stream_path.exists(), (
        f"No fb_ stream found. Generate with: "
        f"uv run python src/datasets/generate_orderings.py --seed {args.seed} --fixed_batch"
    )

    with open(stream_path) as f:
        stream_records = json.load(f)
    cid_to_rec = {r["case_id"]: r for r in stream_records}
    batches = [
        [cid_to_rec[cid] for cid in bcids]
        for bcids in batch_assign["batches"]
    ]

    # ── Rank future batches (CPU) ────────────────────────────────────
    print(f"\nRanking {len(batches) - args.install_batches} future batches "
          f"by cosine to {args.install_batches * args.num_edits} focal-edit keys...")
    ranked = rank_future_batches(batches, args.install_batches, all_keys, cid_to_kidx)
    n_avail = len(ranked)
    n_trials = min(args.n_trials, n_avail // 2)

    print(f"  Top-{n_trials} HIGH batches: "
          + ", ".join(f"b{b}({s:.4f})" for b, s in ranked[:n_trials]))
    print(f"  Top-{n_trials} LOW batches:  "
          + ", ".join(f"b{b}({s:.4f})" for b, s in ranked[-n_trials:]))

    config = build_intervention_config(
        batches, ranked, args.install_batches, n_trials, cid_to_kidx,
    )

    # ── Write config for inner script ────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = result_root / "logit_damage" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "trial_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    output_path = out_dir / "intervention_results.json"

    if args.checkpoint_base:
        ckpt_dir = Path(args.checkpoint_base)
    else:
        ckpt_dir = ckpt_root / "logit_damage" / f"seed{args.seed}"

    # ── Build + run inner script ─────────────────────────────────────
    print(f"\nLaunching GPU intervention script...")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Output:     {output_path}")

    script = build_inner_script(
        seed=args.seed,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        batch_assignment_path=str(ba_path),
        stream_path=str(stream_path),
        keys_path=str(keys_path),
        checkpoint_dir=str(ckpt_dir),
        config_path=str(config_path),
        output_path=str(output_path),
        install_batches=args.install_batches,
        num_edits=args.num_edits,
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ALPHAEDIT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        print(f"\nERROR: Inner script failed (rc={result.returncode})")
        sys.exit(result.returncode)

    print(f"\nDone. Results: {output_path}")


if __name__ == "__main__":
    main()
