#!/usr/bin/env python3
"""
MEMIT-Seq Logit-Damage A/B Intervention Runner

Cross-editor replication of the logit damage experiment under MEMIT-Seq
(history-aware, no null-space projection). Tests whether the interference
mechanism attenuates under a different editing method.

Uses the same fixed-batch assignment and trial configuration as the
AlphaEdit logit_damage_runner, enabling direct comparison.

Usage:
    python src/runners/logit_damage_memit_runner.py --seed 42
    python src/runners/logit_damage_memit_runner.py --seed 42 --lambda_prev 1.0 --lambda_delta 1.0
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

# Reuse batch ranking from the AlphaEdit runner
from logit_damage_runner import rank_future_batches, build_intervention_config


def build_inner_script(
    seed, model_name, hparams_fname,
    batch_assignment_path, stream_path, keys_path,
    checkpoint_dir, config_path, output_path,
    install_batches, num_edits,
    lambda_prev, lambda_delta,
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
LAMBDA_PREV = {lambda_prev}
LAMBDA_DELTA = {lambda_delta}

# ─── Source-inject memit_main.py with SeqReg augmentation ─────────
memit_src = Path("memit/memit_main.py").read_text()
memit_src = memit_src.replace("from .compute_ks", "from memit.compute_ks")
memit_src = memit_src.replace("from .compute_z", "from memit.compute_z")
memit_src = memit_src.replace("from .memit_hparams", "from memit.memit_hparams")

# Inject SeqReg LHS augmentation: replace the solve line
_SOLVE_ANCHOR = '        adj_k = torch.linalg.solve(\\n            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,\\n            layer_ks,\\n        )'

_SOLVE_REPLACEMENT = '''        # === MEMIT+SeqReg: augmented solve (injected) ===
        _K_prev = None
        if LAMBDA_PREV > 0 and layer in _prev_cache and len(_prev_cache[layer]) > 0:
            _K_prev = torch.cat(_prev_cache[layer], dim=1).to(layer_ks.device).float()
        _lhs = hparams.mom2_update_weight * cov.float() + layer_ks @ layer_ks.T
        if _K_prev is not None:
            _lhs = _lhs + LAMBDA_PREV * (_K_prev @ _K_prev.T)
        if LAMBDA_DELTA > 0:
            _lhs = _lhs + LAMBDA_DELTA * torch.eye(_lhs.shape[0], device=_lhs.device, dtype=_lhs.dtype)
        adj_k = torch.linalg.solve(_lhs, layer_ks)
        # === END augmented solve ==='''

assert _SOLVE_ANCHOR in memit_src, "MEMIT solve anchor not found"
memit_src = memit_src.replace(_SOLVE_ANCHOR, _SOLVE_REPLACEMENT)

# Inject key cache storage after deltas
_DELTAS_ANCHOR = '            deltas[weight_name] = ('
_CACHE_CODE = '''            # === MEMIT+SeqReg: store keys in cache (injected) ===
            if LAMBDA_PREV > 0:
                if layer not in _prev_cache:
                    _prev_cache[layer] = []
                _prev_cache[layer].append(layer_ks.detach().cpu())
            if '_K_prev' in dir():
                del _K_prev
            # === END store keys ===
'''
assert _DELTAS_ANCHOR in memit_src, "MEMIT deltas anchor not found"
memit_src = memit_src.replace(_DELTAS_ANCHOR, _CACHE_CODE + _DELTAS_ANCHOR)

# Initialize the key cache
_prev_cache = {{}}

exec(compile(memit_src, "memit/memit_main.py", "exec"),
     _memit_ns := {{"__name__": "memit.memit_main",
                    "__file__": "memit/memit_main.py",
                    "LAMBDA_PREV": LAMBDA_PREV,
                    "LAMBDA_DELTA": LAMBDA_DELTA,
                    "_prev_cache": _prev_cache}})
apply_memit = _memit_ns["apply_memit_to_model"]
print("[MS] memit_main.py compiled with SeqReg (λ_prev={{LAMBDA_PREV}}, λ_delta={{LAMBDA_DELTA}})")

# ─── Load model + tokenizer ───────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"[MS] Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("{model_name}")
tok.pad_token = tok.eos_token
# Override _name_or_path so get_cov() finds precomputed stats under Llama3-8B/
model.config._name_or_path = "Llama3-8B"

# Ensure stats symlinks exist (link_stats.sh may not persist into subprocess)
import os as _os
_stats_dst = Path("data/stats/Llama3-8B/wikipedia_stats")
_stats_dst.mkdir(parents=True, exist_ok=True)
_s3_src = Path("/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct")
if _s3_src.is_dir():
    for _f in _s3_src.iterdir():
        if _f.name.endswith(".npz"):
            _dst = _stats_dst / _f.name
            if not _dst.exists():
                _os.symlink(str(_f), str(_dst))
    print(f"[MS] Linked {{len(list(_stats_dst.glob('*.npz')))}} stat files from S3")
else:
    print(f"[MS] WARNING: S3 stats not found at {{_s3_src}}")
    # List what IS available
    _alt = Path("data/stats")
    if _alt.exists():
        print(f"[MS] Available: {{[str(p) for p in _alt.iterdir()]}}")
print("[MS] Model loaded (stats path: Llama3-8B)")

# ─── Load hparams ─────────────────────────────────────────────────
from memit.memit_hparams import MEMITHyperParams
hparams = MEMITHyperParams.from_json("hparams/MEMIT/{hparams_fname}")
print(f"[MS] Layers: {{hparams.layers}}")

# ─── Load stream + batch assignment + keys ─────────────────────────
with open("{stream_path}") as f:
    stream_records = json.load(f)
with open("{batch_assignment_path}") as f:
    batch_assign = json.load(f)
with open("{config_path}") as f:
    config = json.load(f)

cid_to_rec = {{r["case_id"]: r for r in stream_records}}
batches = [[cid_to_rec[cid] for cid in bcids] for bcids in batch_assign["batches"]]
print(f"[MS] {{len(batches)}} batches loaded")

keys_data = np.load("{keys_path}")
all_keys = torch.from_numpy(keys_data["keys"]).float()
all_cids = keys_data["case_ids"].tolist()
cid_to_kidx = {{int(c): i for i, c in enumerate(all_cids)}}

# ─── Install batches ──────────────────────────────────────────────
print(f"[MS] Installing {{INSTALL_BATCHES}} batches with MEMIT-Seq...")
def make_requests(records):
    return [{{"case_id": r["case_id"], **r["requested_rewrite"]}}
             for r in records]

for b in range(INSTALL_BATCHES):
    model, _ = apply_memit(model, tok, make_requests(batches[b]), hparams)
    if (b + 1) % 5 == 0:
        print(f"  Batch {{b+1}}/{{INSTALL_BATCHES}}")
print(f"[MS] Installed {{INSTALL_BATCHES * NUM_EDITS}} edits")

# ─── Save baseline state ──────────────────────────────────────────
_pd = dict(model.named_parameters())
baseline_w = {{}}
for layer in hparams.layers:
    wn = hparams.rewrite_module_tmp.format(layer) + ".weight"
    baseline_w[wn] = _pd[wn].detach().cpu().clone()
# Save key cache state
import copy
baseline_cache = copy.deepcopy(_prev_cache)

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
print(f"[MS] {{len(focal_cids)}} focal edits, keys: {{focal_keys_t.shape}}")

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
    _pd = dict(model.named_parameters())
    for wn, wt in baseline_w.items():
        _pd[wn].data.copy_(wt.cuda())
    # Restore key cache
    _prev_cache.clear()
    _prev_cache.update(copy.deepcopy(baseline_cache))
    torch.cuda.empty_cache()

# ─── Baseline logprobs ─────────────────────────────────────────────
focal_records = [cid_to_rec[cid] for cid in focal_cids]
print("[MS] Measuring baseline logprobs...")
baseline_lps = measure_logprobs(model, tok, focal_records)
print(f"  Baseline mean: {{np.mean(baseline_lps):.4f}}")

# ─── A/B Intervention Loop ─────────────────────────────────────────
trials_out = []
for trial_cfg in config["trials"]:
    t = trial_cfg["trial"]
    hi_b = trial_cfg["high_batch_idx"]
    lo_b = trial_cfg["low_batch_idx"]
    print(f"\\nTrial {{t}}: HIGH=b{{hi_b}} vs LOW=b{{lo_b}}")

    # HIGH treatment
    model, _ = apply_memit(model, tok, make_requests(batches[hi_b]), hparams)
    hi_lps = measure_logprobs(model, tok, focal_records)
    hi_dmg = compute_damage(model, baseline_w, focal_keys_t, hparams)
    hi_dlps = [h - b for h, b in zip(hi_lps, baseline_lps)]
    restore()
    print(f"  HIGH: mean_dmg={{hi_dmg.mean():.6f}}, mean_Δlp={{np.mean(hi_dlps):.4f}}")

    # LOW treatment
    model, _ = apply_memit(model, tok, make_requests(batches[lo_b]), hparams)
    lo_lps = measure_logprobs(model, tok, focal_records)
    lo_dmg = compute_damage(model, baseline_w, focal_keys_t, hparams)
    lo_dlps = [l - b for l, b in zip(lo_lps, baseline_lps)]
    restore()
    print(f"  LOW:  mean_dmg={{lo_dmg.mean():.6f}}, mean_Δlp={{np.mean(lo_dlps):.4f}}")

    paired_lp = [h - l for h, l in zip(hi_dlps, lo_dlps)]
    frac = sum(1 for p in paired_lp if p < 0) / len(paired_lp)
    print(f"  PAIRED: mean={{np.mean(paired_lp):.4f}}, {{100*frac:.1f}}% more damaged by HIGH")

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
print(f"MEMIT-Seq SUMMARY ({{len(trials_out)}} trials × {{len(focal_cids)}} focal edits)")
print(f"  λ_prev={{LAMBDA_PREV}}, λ_delta={{LAMBDA_DELTA}}")
print(f"  Mean paired Δlogprob (HIGH-LOW): {{all_paired.mean():.4f}}")
print(f"  More damaged by HIGH: {{(all_paired < 0).sum()}}/{{len(all_paired)}} "
      f"({{100*(all_paired < 0).mean():.1f}}%)")
print(f"  Trial-level paired t: t={{t_stat:.3f}}, p={{p_val:.4f}}")
print(f"{{'='*60}}")

out = {{
    "metadata": {{
        "seed": seed, "install_batches": INSTALL_BATCHES,
        "n_focal": len(focal_cids), "n_trials": len(trials_out),
        "algorithm": "MEMIT-Seq",
        "lambda_prev": LAMBDA_PREV, "lambda_delta": LAMBDA_DELTA,
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
        description="MEMIT-Seq logit-damage A/B intervention"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--install_batches", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--lambda_prev", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
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

    # Load keys for batch ranking
    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path
    keys_data = np.load(keys_path)
    all_keys = keys_data["keys"]
    all_cids = keys_data["case_ids"].tolist()
    cid_to_kidx = {int(c): i for i, c in enumerate(all_cids)}

    # Load batch assignment (same as AlphaEdit experiments)
    diag_dir = result_root / "matched_ordering" / "diagnostics"
    ba_path = diag_dir / f"fixed_batch_assignment_seed{args.seed}.json"
    assert ba_path.exists(), f"Fixed-batch assignment not found: {ba_path}"

    with open(ba_path) as f:
        batch_assign = json.load(f)

    stream_path = result_root / "matched_ordering" / "orderings" / f"fb_high_exposure_seed{args.seed}.json"
    if not stream_path.exists():
        stream_path = result_root / "matched_ordering" / "orderings" / f"fb_low_exposure_seed{args.seed}.json"
    assert stream_path.exists(), f"No stream found for seed {args.seed}"

    with open(stream_path) as f:
        stream_records = json.load(f)
    cid_to_rec = {r["case_id"]: r for r in stream_records}
    batches = [
        [cid_to_rec[cid] for cid in bcids]
        for bcids in batch_assign["batches"]
    ]

    # Rank and build config (same as AlphaEdit)
    ranked = rank_future_batches(batches, args.install_batches, all_keys, cid_to_kidx)
    n_trials = min(args.n_trials, len(ranked) // 2)
    config = build_intervention_config(batches, ranked, args.install_batches, n_trials, cid_to_kidx)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = result_root / "logit_damage_memit_seq" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "trial_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    output_path = out_dir / "intervention_results.json"
    ckpt_dir = get_checkpoint_root() / "logit_damage_memit_seq" / f"seed{args.seed}"

    script = build_inner_script(
        seed=args.seed, model_name=model_name, hparams_fname=args.hparams_fname,
        batch_assignment_path=str(ba_path), stream_path=str(stream_path),
        keys_path=str(keys_path), checkpoint_dir=str(ckpt_dir),
        config_path=str(config_path), output_path=str(output_path),
        install_batches=args.install_batches, num_edits=args.num_edits,
        lambda_prev=args.lambda_prev, lambda_delta=args.lambda_delta,
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\nLaunching MEMIT-Seq A/B intervention...")
    print(f"  λ_prev={args.lambda_prev}, λ_delta={args.lambda_delta}")
    print(f"  Output: {output_path}")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ALPHAEDIT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nDone. Results: {output_path}")


if __name__ == "__main__":
    main()
