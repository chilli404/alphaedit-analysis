#!/usr/bin/env python3
"""
Same-Fact, Different-Key Logit-Damage Intervention Runner

The cleanest causal test: for each future edit, select the prompt variant
whose key has HIGH or LOW overlap with focal edits. Same subject, same
relation, same target — only the prompt-induced key geometry differs.

Prerequisites:
  - Prompt-variant keys extracted:
    results/key_vectors/prompt_variants/prompt_variant_keys_seed{SEED}_layer6.npz
  - Fixed-batch assignment:
    results/matched_ordering/diagnostics/fixed_batch_assignment_seed{SEED}.json

Design:
  1. Install N batches of focal edits (using original prompts)
  2. Save model state + cache_c
  3. For each trial batch from the remaining pool:
     a. Construct HIGH-overlap version: for each edit, pick the prompt
        variant whose key has max cosine to any focal key
     b. Construct LOW-overlap version: pick min-cosine variant
     c. Apply HIGH batch → measure focal-edit logprobs → restore
     d. Apply LOW batch → measure focal-edit logprobs → restore
  4. Paired comparison: same facts, same targets, different keys

Output: results/same_fact_damage/seed{SEED}/intervention_results.json

Usage:
    python src/runners/same_fact_damage_runner.py --seed 42
    python src/runners/same_fact_damage_runner.py --seed 42 --install_batches 10 --n_trials 10
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


def select_prompt_variants(
    batch_records, variant_keys, focal_keys_normed, case_id_to_vidx, mode="high",
):
    """For each record in a batch, select the prompt variant with
    highest (mode='high') or lowest (mode='low') max-cosine to focal keys.

    Returns list of (record, selected_prompt_template, variant_name, cosine_score).
    """
    results = []
    for record in batch_records:
        cid = record["case_id"]
        vidx = case_id_to_vidx.get(cid)
        if vidx is None:
            # No variant keys — use original
            results.append((record, record["requested_rewrite"]["prompt"], "original", 0.0))
            continue

        subject = record["requested_rewrite"]["subject"]

        # Get keys for each variant
        variant_cosines = []
        for vname in ["original", "para0", "para1"]:
            key = variant_keys[vname][vidx]
            key_norm = key / max(np.linalg.norm(key), 1e-8)
            # Max cosine to any focal key
            cos = focal_keys_normed @ key_norm
            max_cos = float(cos.max())
            variant_cosines.append((vname, max_cos))

        # Select best variant
        if mode == "high":
            best = max(variant_cosines, key=lambda x: x[1])
        else:
            best = min(variant_cosines, key=lambda x: x[1])

        vname, score = best

        # Convert to prompt template
        if vname == "original":
            template = record["requested_rewrite"]["prompt"]
        else:
            idx = int(vname.replace("para", ""))
            para_text = record["paraphrase_prompts"][idx]
            template = para_text.replace(subject, "{}")

        results.append((record, template, vname, score))

    return results


def build_inner_script(
    seed, model_name, hparams_fname,
    batch_assignment_path, stream_path, keys_path, variant_keys_path,
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
# Cast keys to float32 for compatibility with P and cache_c (model is bfloat16)
ae_src = ae_src.replace(
    "layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T",
    "layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T.float()"
)
exec(compile(ae_src, "AlphaEdit/AlphaEdit_main.py", "exec"),
     _ae_ns := {{"__name__": "AlphaEdit.AlphaEdit_main",
                 "__file__": "AlphaEdit/AlphaEdit_main.py"}})
apply_ae = _ae_ns["apply_AlphaEdit_to_model"]
print("[SF] AlphaEdit_main.py compiled")

# ─── Load model + tokenizer ───────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[SF] Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("{model_name}")
tok.pad_token = tok.eos_token
print("[SF] Model loaded")

# ─── Load hparams ─────────────────────────────────────────────────
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
hparams = AlphaEditHyperParams.from_json("hparams/AlphaEdit/{hparams_fname}")

# ─── Load P ───────────────────────────────────────────────────────
_p_candidates = [
    Path("null_space_project.pt"),
    Path("data/stats/Llama3-8B/wikipedia_stats/null_space_project.pt"),
    Path("/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct/null_space_project.pt"),
    Path("/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct/wikipedia_stats/null_space_project.pt"),
]
_mn = hparams.model_name.replace("/", "_")
_p_candidates.append(Path(f"data/stats/{{_mn}}/wikipedia_stats/null_space_project.pt"))
P_path = None
for _pc in _p_candidates:
    if _pc.exists():
        P_path = _pc
        break
assert P_path is not None, f"P not found at any of: {{[str(p) for p in _p_candidates]}}"
P = torch.load(str(P_path), map_location="cpu")
print(f"[SF] Loaded P from {{P_path}}")

# ─── Initialize cache_c ──────────────────────────────────────────
_wkey = hparams.rewrite_module_tmp.format(hparams.layers[0]) + ".weight"
d = dict(model.named_parameters())[_wkey].shape[1]
cache_c = torch.zeros(len(hparams.layers), d, d)

# ─── Load config + stream + batch assignment ──────────────────────
with open("{config_path}") as f:
    config = json.load(f)
with open("{stream_path}") as f:
    stream_records = json.load(f)
with open("{batch_assignment_path}") as f:
    batch_assign = json.load(f)

cid_to_rec = {{r["case_id"]: r for r in stream_records}}
batches = [[cid_to_rec[cid] for cid in bcids] for bcids in batch_assign["batches"]]

# ─── Install focal edits (using original prompts) ─────────────────
ckpt_dir = Path("{checkpoint_dir}")
ckpt_batch = ckpt_dir / f"batch_{{INSTALL_BATCHES - 1}}"
if (ckpt_batch / "model_weights.pt").exists():
    print(f"[SF] Restoring checkpoint: {{ckpt_batch}}")
    _cw = torch.load(str(ckpt_batch / "model_weights.pt"), map_location="cuda")
    _pd = dict(model.named_parameters())
    for wn, wt in _cw.items():
        if wn in _pd: _pd[wn].data.copy_(wt.cuda())
    del _cw
    if (ckpt_batch / "cache_c.pt").exists():
        cache_c = torch.load(str(ckpt_batch / "cache_c.pt"), map_location="cpu")
    torch.cuda.empty_cache()
else:
    print(f"[SF] Installing {{INSTALL_BATCHES}} batches...")
    def _make_reqs(records):
        return [{{"case_id": r["case_id"], **r["requested_rewrite"]}} for r in records]
    for b in range(INSTALL_BATCHES):
        model, cache_c = apply_ae(model, tok, _make_reqs(batches[b]),
                                  hparams, cache_c=cache_c, P=P)
        if (b + 1) % 5 == 0: print(f"  Batch {{b+1}}/{{INSTALL_BATCHES}}")
    ckpt_batch.mkdir(parents=True, exist_ok=True)
    _pd = dict(model.named_parameters())
    _ew = {{}}
    for layer in hparams.layers:
        wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
        _ew[wn] = _pd[wn].detach().cpu()
    torch.save(_ew, str(ckpt_batch / "model_weights.pt"))
    torch.save(cache_c.cpu(), str(ckpt_batch / "cache_c.pt"))
    print("[SF] Installed + saved checkpoint")
    del _ew; torch.cuda.empty_cache()

# ─── Save baseline ────────────────────────────────────────────────
_pd = dict(model.named_parameters())
baseline_w = {{}}
for layer in hparams.layers:
    wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
    baseline_w[wn] = _pd[wn].detach().cpu().clone()
baseline_cc = cache_c.clone()

# ─── Measurement functions ────────────────────────────────────────
def measure_logprobs(model, tok, records):
    lps = []
    for r in records:
        prompt = r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"])
        target = " " + r["requested_rewrite"]["target_new"]["str"]
        enc = tok(prompt + target, return_tensors="pt").to("cuda")
        plen = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        with torch.no_grad():
            logits = model(**enc).logits
        tids = enc["input_ids"][0, plen:]
        if len(tids) == 0:
            lps.append(0.0); continue
        lsm = torch.log_softmax(logits[0, plen-1:plen-1+len(tids)].float(), dim=-1)
        lps.append(float(lsm.gather(1, tids.unsqueeze(1)).sum()))
    return lps

def restore():
    global cache_c
    _pd = dict(model.named_parameters())
    for wn, wt in baseline_w.items():
        _pd[wn].data.copy_(wt.cuda())
    cache_c = baseline_cc.clone()
    torch.cuda.empty_cache()

# ─── Focal-edit records ───────────────────────────────────────────
focal_cids = config["focal_case_ids"]
focal_records = [cid_to_rec[cid] for cid in focal_cids]
print(f"[SF] {{len(focal_cids)}} focal edits")

# ─── Baseline logprobs ────────────────────────────────────────────
print("[SF] Measuring baseline logprobs...")
baseline_lps = measure_logprobs(model, tok, focal_records)
print(f"  Baseline mean: {{np.mean(baseline_lps):.4f}}")

# ─── A/B Intervention with prompt variants ────────────────────────
trials_out = []
for trial_cfg in config["trials"]:
    t = trial_cfg["trial"]
    print(f"\\nTrial {{t}}: batch with {{len(trial_cfg['high_requests'])}} edits")
    print(f"  Mean cosine: HIGH={{trial_cfg['high_mean_cosine']:.4f}}, "
          f"LOW={{trial_cfg['low_mean_cosine']:.4f}}")

    # Apply HIGH-variant batch
    hi_reqs = trial_cfg["high_requests"]
    model, cache_c = apply_ae(model, tok, hi_reqs, hparams, cache_c=cache_c, P=P)
    hi_lps = measure_logprobs(model, tok, focal_records)
    hi_dlps = [h - b for h, b in zip(hi_lps, baseline_lps)]
    restore()

    # Apply LOW-variant batch (SAME facts, different prompts)
    lo_reqs = trial_cfg["low_requests"]
    model, cache_c = apply_ae(model, tok, lo_reqs, hparams, cache_c=cache_c, P=P)
    lo_lps = measure_logprobs(model, tok, focal_records)
    lo_dlps = [l - b for l, b in zip(lo_lps, baseline_lps)]
    restore()

    paired = [h - l for h, l in zip(hi_dlps, lo_dlps)]
    frac = sum(1 for p in paired if p < 0) / len(paired)
    print(f"  HIGH mean Δlp={{np.mean(hi_dlps):.4f}}, LOW mean Δlp={{np.mean(lo_dlps):.4f}}")
    print(f"  PAIRED: mean={{np.mean(paired):.4f}}, {{100*frac:.1f}}% more damaged by HIGH")

    trials_out.append({{
        **{{k: v for k, v in trial_cfg.items() if k not in ("high_requests", "low_requests")}},
        "high_delta_logprob": hi_dlps,
        "low_delta_logprob": lo_dlps,
        "paired_logprob_diff": paired,
        "high_variant_dist": trial_cfg.get("high_variant_dist", {{}}),
        "low_variant_dist": trial_cfg.get("low_variant_dist", {{}}),
    }})

# ─── Summary ──────────────────────────────────────────────────────
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
print(f"SAME-FACT DAMAGE SUMMARY")
print(f"  Mean paired Δlogprob (HIGH-LOW): {{all_paired.mean():.4f}}")
print(f"  More damaged by HIGH: {{(all_paired < 0).sum()}}/{{len(all_paired)}}")
print(f"  Trial-level t: t={{t_stat:.3f}}, p={{p_val:.4f}}")
print(f"{{'='*60}}")

out = {{
    "metadata": {{
        "seed": seed, "install_batches": INSTALL_BATCHES,
        "n_focal": len(focal_cids), "n_trials": len(trials_out),
        "experiment_type": "same_fact_different_key",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }},
    "baseline_mean_logprob": float(np.mean(baseline_lps)),
    "trials": trials_out,
    "summary": {{
        "mean_paired_delta_lp": float(all_paired.mean()),
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


def main():
    parser = argparse.ArgumentParser(
        description="Same-fact, different-key A/B intervention"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--install_batches", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--variant_keys_path", default=None)
    parser.add_argument("--keys_path", default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--output_dir", default=None)
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

    # Load variant keys
    if args.variant_keys_path:
        vk_path = Path(args.variant_keys_path)
    else:
        vk_path = (PROJECT_ROOT / "results" / "key_vectors" / "prompt_variants"
                    / f"prompt_variant_keys_seed{args.seed}_layer6.npz")

    if not vk_path.exists():
        print(f"ERROR: Variant keys not found at {vk_path}")
        print(f"Run: uv run python src/experiments/prompt_variant_keys.py --seed {args.seed}")
        sys.exit(1)

    vk_data = np.load(vk_path)
    variant_keys = {
        "original": vk_data["keys_original"],
        "para0": vk_data["keys_para0"],
        "para1": vk_data["keys_para1"],
    }
    vk_case_ids = vk_data["case_ids"].tolist()
    case_id_to_vidx = {int(cid): i for i, cid in enumerate(vk_case_ids)}
    print(f"Loaded variant keys: {len(vk_case_ids)} records, 3 variants each")

    # Load batch assignment + stream
    ba_path = result_root / "matched_ordering" / "diagnostics" / f"fixed_batch_assignment_seed{args.seed}.json"
    assert ba_path.exists(), f"Batch assignment not found: {ba_path}"
    with open(ba_path) as f:
        batch_assign = json.load(f)

    stream_path = result_root / "matched_ordering" / "orderings" / f"fb_high_exposure_seed{args.seed}.json"
    if not stream_path.exists():
        stream_path = result_root / "matched_ordering" / "orderings" / f"key_clustered_seed{args.seed}.json"
    with open(stream_path) as f:
        stream_records = json.load(f)
    cid_to_rec = {r["case_id"]: r for r in stream_records}
    batches = [[cid_to_rec[cid] for cid in bcids] for bcids in batch_assign["batches"]]

    # Load focal-edit keys for cosine computation
    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path
    keys_data = np.load(keys_path)
    all_keys = keys_data["keys"]
    all_cids = keys_data["case_ids"].tolist()
    cid_to_kidx = {int(c): i for i, c in enumerate(all_cids)}

    # Focal keys
    focal_keys = []
    focal_cids = []
    for b in range(args.install_batches):
        for r in batches[b]:
            kidx = cid_to_kidx.get(r["case_id"], -1)
            if kidx >= 0:
                focal_keys.append(all_keys[kidx])
                focal_cids.append(r["case_id"])
    focal_keys = np.array(focal_keys)
    focal_norms = np.linalg.norm(focal_keys, axis=1, keepdims=True)
    focal_normed = focal_keys / np.maximum(focal_norms, 1e-8)

    # Build trials: for each future batch, construct HIGH and LOW prompt-variant versions
    print(f"\nBuilding {args.n_trials} trial pairs...")
    trials = []
    future_batches = list(range(args.install_batches, len(batches)))

    # Score each future batch by achievable cosine gap
    batch_gaps = []
    for b_idx in future_batches:
        hi_scores = []
        lo_scores = []
        for r in batches[b_idx]:
            vidx = case_id_to_vidx.get(r["case_id"])
            if vidx is None:
                continue
            variant_cos = []
            for vname in ["original", "para0", "para1"]:
                key = variant_keys[vname][vidx]
                key_n = key / max(np.linalg.norm(key), 1e-8)
                max_cos = float((focal_normed @ key_n).max())
                variant_cos.append(max_cos)
            hi_scores.append(max(variant_cos))
            lo_scores.append(min(variant_cos))
        if hi_scores:
            gap = np.mean(hi_scores) - np.mean(lo_scores)
            batch_gaps.append((b_idx, gap, np.mean(hi_scores), np.mean(lo_scores)))

    # Sort by gap and take top n_trials
    batch_gaps.sort(key=lambda x: x[1], reverse=True)
    selected = batch_gaps[:args.n_trials]

    for rank, (b_idx, gap, hi_mean, lo_mean) in enumerate(selected):
        batch = batches[b_idx]
        hi_selected = select_prompt_variants(batch, variant_keys, focal_normed, case_id_to_vidx, "high")
        lo_selected = select_prompt_variants(batch, variant_keys, focal_normed, case_id_to_vidx, "low")

        hi_requests = [{"case_id": r["case_id"], "prompt": template,
                        "subject": r["requested_rewrite"]["subject"],
                        "target_new": r["requested_rewrite"]["target_new"],
                        "target_true": r["requested_rewrite"]["target_true"]}
                       for r, template, _, _ in hi_selected]
        lo_requests = [{"case_id": r["case_id"], "prompt": template,
                        "subject": r["requested_rewrite"]["subject"],
                        "target_new": r["requested_rewrite"]["target_new"],
                        "target_true": r["requested_rewrite"]["target_true"]}
                       for r, template, _, _ in lo_selected]

        from collections import Counter
        hi_vdist = dict(Counter(vname for _, _, vname, _ in hi_selected))
        lo_vdist = dict(Counter(vname for _, _, vname, _ in lo_selected))

        trials.append({
            "trial": rank,
            "batch_idx": b_idx,
            "cosine_gap": float(gap),
            "high_mean_cosine": float(hi_mean),
            "low_mean_cosine": float(lo_mean),
            "high_requests": hi_requests,
            "low_requests": lo_requests,
            "high_variant_dist": hi_vdist,
            "low_variant_dist": lo_vdist,
        })

        print(f"  Trial {rank}: batch {b_idx}, gap={gap:.4f}, "
              f"hi={hi_mean:.4f}, lo={lo_mean:.4f}, "
              f"hi_variants={hi_vdist}, lo_variants={lo_vdist}")

    config = {
        "focal_case_ids": focal_cids,
        "install_batches": args.install_batches,
        "n_trials": args.n_trials,
        "trials": trials,
    }

    # Write config + run
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = result_root / "same_fact_damage" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "trial_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    output_path = out_dir / "intervention_results.json"

    ckpt_dir = ckpt_root / "same_fact_damage" / f"seed{args.seed}"

    script = build_inner_script(
        seed=args.seed, model_name=model_name, hparams_fname=args.hparams_fname,
        batch_assignment_path=str(ba_path), stream_path=str(stream_path),
        keys_path=str(keys_path), variant_keys_path=str(vk_path),
        checkpoint_dir=str(ckpt_dir), config_path=str(config_path),
        output_path=str(output_path),
        install_batches=args.install_batches, num_edits=args.num_edits,
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(ALPHAEDIT_ROOT), env=env,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nDone. Results: {output_path}")


if __name__ == "__main__":
    main()
