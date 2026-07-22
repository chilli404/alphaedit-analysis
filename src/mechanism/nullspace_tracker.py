#!/usr/bin/env python3
"""
Null-Space Rank Consumption Tracker for AlphaEdit.

Provides mechanistic insight into WHY AlphaEdit eventually degrades:
as sequential edits accumulate, the null-space projection matrix P
remains static while the accumulated covariance cache_c grows in rank.
When the edit directions saturate the available null-space dimensions,
updates can no longer avoid interfering with prior edits.

Implementation approach:
  Instead of fragile monkey-patching, this injects measurement code
  directly into the evaluate.py source string at known anchor points
  (pinned commit b84624f). The injected code runs inline where cache_c
  and P are already in scope — no wrappers, no closures, no module
  attribute patching.

  Uses hparams.nullspace_threshold for rank computation (same threshold
  AlphaEdit uses in get_project()), ensuring the consumption ratio
  compares apples to apples.

Output: JSONL file with one record per edit batch containing:
  - batch_idx, num_requests, total_edits_so_far
  - Per-layer: nullspace_rank_initial, cache_c_numerical_rank,
    cache_c_effective_rank, consumption_ratio, top singular values

Usage:
    python src/nullspace_tracker.py \\
        --seed 42 \\
        --cuda_device 0 \\
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \\
        --hparams_fname Llama3-8B.json \\
        --ds_name mcf \\
        --dataset_size_limit 2000 \\
        --num_edits 100 \\
        --downstream_eval_steps 5
"""

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from model_download import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file


def get_project_root() -> Path:
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


# --- Source anchors from evaluate.py at commit b84624f ---
# These are the exact strings we inject code around.

# Anchor 1: Just before the AlphaEdit apply_algo call.
# We inject AFTER this line to capture pre-edit state.
PRE_EDIT_ANCHOR = '        start = time()\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'

# Anchor 2: The line immediately after the AlphaEdit apply_algo call closes.
# We inject BEFORE this line to capture post-edit state.
POST_EDIT_ANCHOR = '        elif alg_name == "MEMIT_prune":'


def build_tracker_script(
    seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    downstream_eval_steps: int,
    conserve_memory: bool,
    output_jsonl: str,
    eval_results_dir: str = "",
) -> str:
    """
    Build an inline Python script that:
    1. Seeds all RNGs
    2. Injects tracking code directly into evaluate.py source at known anchors
    3. Executes the patched evaluate.py as __main__
    """
    argv_parts = [
        "experiments.evaluate",
        f"--alg_name={alg_name}",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
        f"--downstream_eval_steps={downstream_eval_steps}",
        "--generation_test_interval=1",
    ]
    if conserve_memory:
        argv_parts.append("--conserve_memory")

    argv_str = repr(argv_parts)

    script = textwrap.dedent(f"""\
import os, sys, random, json, math
import numpy as np
import torch

# 1. Seed all sources of randomness
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# 2. Set sys.argv
sys.argv = {argv_str}

# 3. Define tracking functions (these will be in the exec namespace)
_ns_track_output = "{output_jsonl}"
_ns_track_records = []

def _ns_track_pre_batch(cnt, cache_c, P, hparams, num_edits, n_requests):
    \"\"\"Record null-space state BEFORE an edit batch is applied.\"\"\"
    record = {{
        "batch_idx": cnt,
        "num_requests": n_requests * num_edits,
        "total_edits_so_far": cnt * num_edits,
        "layers": {{}},
    }}

    threshold = hparams.nullspace_threshold  # Same threshold as get_project()

    for i, layer in enumerate(hparams.layers):
        layer_record = {{}}

        # Null-space rank from P (static, computed once at init)
        # P[i] is a projection matrix; rank = trace (eigenvalues are 0 or 1)
        P_layer = P[i].float()
        nullspace_rank = int(torch.trace(P_layer).round().item())
        layer_record["nullspace_rank_initial"] = nullspace_rank
        layer_record["hidden_dim"] = P_layer.shape[0]

        # Cache_c rank (grows with each edit batch)
        cache_layer = cache_c[i].float()
        if cache_layer.abs().max() > 0:
            svs = torch.linalg.svdvals(cache_layer)
            # Numerical rank using SAME threshold as AlphaEdit's null-space
            numerical_rank = int((svs > threshold).sum().item())
            layer_record["cache_c_numerical_rank"] = numerical_rank

            # Effective rank (entropy-based): exp(H(normalized_svs))
            svs_pos = svs[svs > 0]
            if len(svs_pos) > 1:
                p = svs_pos / svs_pos.sum()
                entropy = -(p * torch.log(p)).sum().item()
                effective_rank = math.exp(entropy)
            else:
                effective_rank = float(len(svs_pos))
            layer_record["cache_c_effective_rank"] = round(effective_rank, 2)

            # Top 10 singular values
            layer_record["cache_c_top_svs"] = svs[:10].tolist()

            # Consumption ratio
            if nullspace_rank > 0:
                layer_record["consumption_ratio"] = round(
                    numerical_rank / nullspace_rank, 4
                )
            else:
                layer_record["consumption_ratio"] = None
        else:
            layer_record["cache_c_numerical_rank"] = 0
            layer_record["cache_c_effective_rank"] = 0.0
            layer_record["cache_c_top_svs"] = []
            layer_record["consumption_ratio"] = 0.0

        record["layers"][str(layer)] = layer_record

    _ns_track_records.append(record)
    return record

def _ns_track_post_batch(cnt, cache_c, P, hparams, num_edits):
    \"\"\"Record cache_c rank AFTER the edit batch is applied.\"\"\"
    if not _ns_track_records:
        return

    record = _ns_track_records[-1]
    threshold = hparams.nullspace_threshold

    for i, layer in enumerate(hparams.layers):
        cache_layer = cache_c[i].float()
        if cache_layer.abs().max() > 0:
            svs = torch.linalg.svdvals(cache_layer)
            post_rank = int((svs > threshold).sum().item())
        else:
            post_rank = 0
        record["layers"][str(layer)]["cache_c_rank_post_edit"] = post_rank

    record["total_edits_after"] = (cnt + 1) * num_edits

    # Write incrementally
    with open(_ns_track_output, "a") as f:
        f.write(json.dumps(record) + "\\n")

# 4. Read evaluate.py source
with open("experiments/evaluate.py", "r") as f:
    source = f.read()

# 5. Patch CUDA_VISIBLE_DEVICES
cuda_patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
assert cuda_patch_target in source, (
    "CUDA_VISIBLE_DEVICES patch target not found in evaluate.py. "
    "Upstream code has changed from pinned commit b84624f."
)
source = source.replace(
    cuda_patch_target,
    '# CUDA_VISIBLE_DEVICES managed by nullspace_tracker',
)

# 5a. Override RESULTS_DIR
_globals_import = 'from util.globals import *'
assert _globals_import in source, "globals import not found in evaluate.py"
source = source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)
print(f"  [RESULTS_DIR] Overridden to: {eval_results_dir}")

# 6. Inject pre-edit tracking code
pre_anchor = '        start = time()\\n        if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE"]):'
assert pre_anchor in source, (
    "Pre-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

pre_injection = '        # === NULLSPACE TRACKING: pre-edit measurement (injected) ===\\n'
pre_injection += "        if alg_name == \\"AlphaEdit\\" and '_ns_track_output' in globals():\\n"
pre_injection += '            _ns_track_pre_batch(cnt, cache_c, P, hparams, num_edits, len(record_chunks))\\n'
pre_injection += '        # === END pre-edit tracking ===\\n'
source = source.replace(
    pre_anchor,
    pre_injection + pre_anchor,
    1,  # Replace only first occurrence
)

# 7. Inject post-edit tracking code
post_anchor = '        elif alg_name == "MEMIT_prune":'
assert post_anchor in source, (
    "Post-edit anchor not found in evaluate.py source. "
    "Upstream code has changed from pinned commit b84624f."
)

post_injection = '        # === NULLSPACE TRACKING: post-edit measurement (injected) ===\\n'
post_injection += "        if alg_name == \\"AlphaEdit\\" and '_ns_track_output' in globals():\\n"
post_injection += '            _ns_track_post_batch(cnt, cache_c, P, hparams, num_edits)\\n'
post_injection += '        # === END post-edit tracking ===\\n'
source = source.replace(
    post_anchor,
    post_injection + post_anchor,
    1,
)

# 8. Verify injection succeeded
assert "NULLSPACE TRACKING: pre-edit" in source, "Pre-edit injection failed"
assert "NULLSPACE TRACKING: post-edit" in source, "Post-edit injection failed"

# 9. Execute
exec(compile(source, "experiments/evaluate.py", "exec"),
     {{
         "__name__": "__main__",
         "__file__": "experiments/evaluate.py",
         "_ns_track_output": _ns_track_output,
         "_ns_track_records": _ns_track_records,
         "_ns_track_pre_batch": _ns_track_pre_batch,
         "_ns_track_post_batch": _ns_track_post_batch,
     }})

# 10. Final summary
print(f"\\n=== Null-space tracking complete ===")
print(f"  Recorded {{len(_ns_track_records)}} edit batches")
print(f"  Output: {{_ns_track_output}}")
if _ns_track_records:
    last = _ns_track_records[-1]
    for layer_str, data in last["layers"].items():
        ratio = data.get("consumption_ratio")
        if ratio is not None:
            print(f"  Layer {{layer_str}}: consumption ratio = {{ratio:.3f}}")
""")
    return script


def run(args: argparse.Namespace) -> None:
    """Launch the instrumented experiment."""
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        print("Run: git submodule update --init --recursive")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    # Resolve model path (falls back to Artifactory mirror if HF access fails)
    model_name = resolve_model_path(args.model_name)

    # Validate anchors exist in the source before launching
    eval_source = (alphaedit_root / "experiments" / "evaluate.py").read_text()
    if PRE_EDIT_ANCHOR not in eval_source:
        print("ERROR: Pre-edit anchor not found in evaluate.py.")
        print("  The upstream code has diverged from pinned commit b84624f.")
        print(f"  Expected: {PRE_EDIT_ANCHOR[:60]}...")
        sys.exit(1)
    if POST_EDIT_ANCHOR not in eval_source:
        print("ERROR: Post-edit anchor not found in evaluate.py.")
        print("  The upstream code has diverged from pinned commit b84624f.")
        sys.exit(1)

    # Output file
    output_dir = (
        project_root / "results" / "nullspace_tracking"
        / f"seed{args.seed}" / f"{args.dataset_size_limit}edits" / "AlphaEdit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = output_dir / f"rank_trace_seed{args.seed}_{args.dataset_size_limit}edits_{timestamp}.jsonl"

    script = build_tracker_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name="AlphaEdit",
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        downstream_eval_steps=args.downstream_eval_steps,
        conserve_memory=args.conserve_memory,
        output_jsonl=str(output_jsonl),
        eval_results_dir=str(output_dir.parent),
    )

    # Environment
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'=' * 70}")
    print("Null-Space Rank Consumption Tracker")
    print("  Algorithm:  AlphaEdit (with inline instrumentation)")
    print(f"  Dataset:    {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Num edits:  {args.num_edits}")
    print(f"  Seed:       {args.seed}")
    print(f"  CUDA:       device {args.cuda_device}")
    print(f"  Model:      {args.model_name}")
    print("  Threshold:  from hparams (nullspace_threshold)")
    print(f"  Output:     {output_jsonl}")
    print(f"  Started:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Tracking run failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Null-space tracking completed.")
    print(f"  Output: {output_jsonl}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Track null-space rank consumption during AlphaEdit editing"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--downstream_eval_steps", type=int, default=5)
    parser.add_argument("--conserve_memory", action="store_true", default=True)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
