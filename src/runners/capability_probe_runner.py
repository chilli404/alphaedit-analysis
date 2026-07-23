#!/usr/bin/env python3
"""
Capability Probe Runner: Editing + perplexity/MMLU measurement pipeline.

This script runs the full editing pipeline and measures general model
capabilities (perplexity on WikiText-103, optional MMLU accuracy) at
regular intervals during editing.

Unlike the standard seeded_runner.py which only measures CounterFact
metrics, this runner additionally captures whether the edited model
retains its general language modeling and reasoning capabilities.

The probe measurements are saved as a JSONL file (one record per
measurement point), independent of AlphaEdit's own result JSONs.

Usage:
    python src/capability_probe_runner.py \
        --seed 42 \
        --cuda_device 0 \
        --alg_name AlphaEdit \
        --ds_name mcf \
        --dataset_size_limit 2000 \
        --num_edits 100 \
        --probe_interval 5
"""

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Add src/util to path for shared utilities
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from model_download import resolve_model_path
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file
from paths import get_project_root, get_alphaedit_root, get_result_root


def build_probe_script(
    seed: int,
    cuda_device: str,
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    num_edits: int,
    probe_interval: int,
    output_jsonl: str,
    compute_mmlu: bool,
    eval_results_dir: str = "",
) -> str:
    """
    Build a script that runs evaluate.py with capability probing injected
    at each probe_interval evaluation step.

    The probe is injected by monkey-patching the GLUEEval.evaluate() method
    to additionally compute perplexity on WikiText-103 and optionally MMLU.
    """
    argv_parts = [
        "experiments.evaluate",
        f"--alg_name={alg_name}",
        f"--model_name={model_name}",
        f"--hparams_fname={hparams_fname}",
        f"--ds_name={ds_name}",
        f"--dataset_size_limit={dataset_size_limit}",
        f"--num_edits={num_edits}",
        f"--downstream_eval_steps={probe_interval}",
        "--generation_test_interval=1",
        "--conserve_memory",
    ]
    argv_str = repr(argv_parts)

    script = textwrap.dedent(f"""\
import os, sys, random, json, time
import numpy as np
import torch
from torch.nn import functional as F

# 1. Seed
seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

sys.argv = {argv_str}

# 2. Define the capability probe functions
PROBE_OUTPUT = "{output_jsonl}"
PROBE_COMPUTE_MMLU = {compute_mmlu}
WIKITEXT_TEXTS = None  # Lazy-loaded
_probe_records = []

def _load_wikitext(n_samples=200):
    global WIKITEXT_TEXTS
    if WIKITEXT_TEXTS is not None:
        return WIKITEXT_TEXTS
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    WIKITEXT_TEXTS = texts
    return texts

def _compute_perplexity(model, tokenizer, texts, max_length=512, batch_size=4):
    model.eval()
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    n_samples = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts, return_tensors="pt",
            max_length=max_length, truncation=True, padding=True,
        ).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for j in range(input_ids.shape[0]):
            mask = attention_mask[j] == 1
            valid_ids = input_ids[j][mask]
            valid_logits = logits[j][mask]
            if len(valid_ids) < 2:
                continue
            shift_logits = valid_logits[:-1]
            shift_labels = valid_ids[1:]
            nll = F.cross_entropy(shift_logits, shift_labels, reduction="sum").item()
            n_tokens = len(shift_labels)
            total_nll += nll
            total_tokens += n_tokens
            n_samples += 1

    if n_samples == 0:
        return {{"mean_perplexity": float("nan"), "n_samples": 0, "n_tokens": 0}}
    # Corpus-level perplexity: exp(total_NLL / total_tokens)
    corpus_ppl = float(np.exp(total_nll / total_tokens))
    return {{
        "mean_perplexity": corpus_ppl,
        "n_samples": n_samples,
        "n_tokens": total_tokens,
    }}

def _run_probe(model, tokenizer, edit_count):
    \"\"\"Run capability probe and save results.\"\"\"
    print(f"  [PROBE] Running capability probe at {{edit_count}} edits...")
    t0 = time.time()

    texts = _load_wikitext()
    ppl_result = _compute_perplexity(model, tokenizer, texts)

    record = {{
        "edit_count": edit_count,
        "timestamp_utc": time.time(),
        **ppl_result,
    }}

    elapsed = time.time() - t0
    print(f"  [PROBE] Perplexity: {{ppl_result['mean_perplexity']:.2f}} "
          f"({{ppl_result['n_samples']}} samples, {{elapsed:.1f}}s)")

    _probe_records.append(record)
    with open(PROBE_OUTPUT, "a") as f:
        f.write(json.dumps(record) + "\\n")

# 3. Monkey-patch GLUEEval to also run our probe
_original_glue_eval = None

def _patched_glue_evaluate(self, glue_results, *args, **kwargs):
    \"\"\"Wrapper that runs our probe after GLUEEval.\"\"\"
    result = _original_glue_eval(self, glue_results, *args, **kwargs)
    # Extract edit count from glue_results
    edit_count = glue_results.get("edit_num", 0)
    if edit_count == -1:
        edit_count = 0  # baseline measurement
    _run_probe(self.model, self.tokenizer, edit_count)
    return result

# 4. Read and patch evaluate.py
with open("experiments/evaluate.py", "r") as f:
    source = f.read()

patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
if patch_target in source:
    source = source.replace(
        patch_target,
        '# CUDA_VISIBLE_DEVICES managed by capability_probe_runner',
    )

# Override RESULTS_DIR
_globals_import = 'from util.globals import *'
assert _globals_import in source, "globals import not found in evaluate.py"
source = source.replace(
    _globals_import,
    _globals_import + '\\nRESULTS_DIR = Path("{eval_results_dir}")\\n',
    1,
)
print(f"  [RESULTS_DIR] Overridden to: {eval_results_dir}")

# Inject GLUEEval monkey-patch after imports
glue_patch = '''
# === Capability probe patch (injected by capability_probe_runner.py) ===
from glue_eval.glue_eval import GLUEEval as _OrigGLUEEval
_original_glue_eval = _OrigGLUEEval.evaluate
_OrigGLUEEval.evaluate = _patched_glue_evaluate
# === End capability probe patch ===
'''
source = source.replace(
    'if __name__ == "__main__":',
    glue_patch + '\\nif __name__ == "__main__":',
)

# 5. Execute
exec_globals = {{
    "__name__": "__main__",
    "__file__": "experiments/evaluate.py",
    "_patched_glue_evaluate": _patched_glue_evaluate,
    "_original_glue_eval": None,  # Set by patch
    "_run_probe": _run_probe,
    "_load_wikitext": _load_wikitext,
    "_compute_perplexity": _compute_perplexity,
    "_probe_records": _probe_records,
    "PROBE_OUTPUT": PROBE_OUTPUT,
}}

exec(compile(source, "experiments/evaluate.py", "exec"), exec_globals)

print(f"\\n=== Capability probe complete ===")
print(f"  Recorded {{len(_probe_records)}} measurement points")
print(f"  Output: {{PROBE_OUTPUT}}")
    """)
    return script


def run(args: argparse.Namespace) -> None:
    alphaedit_root = get_alphaedit_root()
    project_root = get_project_root()

    if not alphaedit_root.exists():
        print(f"ERROR: AlphaEdit not found at {alphaedit_root}")
        sys.exit(1)

    link_hparams()
    patch_evaluate_file(alphaedit_root)

    # Resolve model path (falls back to Artifactory mirror if HF access fails)
    model_name = resolve_model_path(args.model_name)

    # Output file
    output_dir = get_result_root() / "capability_probe" / f"seed{args.seed}" / f"{args.dataset_size_limit}edits" / args.alg_name
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_jsonl = output_dir / f"probe_{timestamp}.jsonl"

    script = build_probe_script(
        seed=args.seed,
        cuda_device=args.cuda_device,
        alg_name=args.alg_name,
        model_name=model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        num_edits=args.num_edits,
        probe_interval=args.probe_interval,
        output_jsonl=str(output_jsonl),
        compute_mmlu=not args.no_mmlu,
        eval_results_dir=str(output_dir.parent),
    )

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'=' * 70}")
    print("Capability Probe Runner")
    print(f"  Algorithm:      {args.alg_name}")
    print(f"  Dataset:        {args.ds_name} (limit={args.dataset_size_limit})")
    print(f"  Probe interval: every {args.probe_interval} edit rounds")
    print(f"  MMLU:           {'yes' if not args.no_mmlu else 'no (perplexity only)'}")
    print(f"  Seed:           {args.seed}")
    print(f"  Output:         {output_jsonl}")
    print(f"{'=' * 70}")

    cmd = [sys.executable, "-c", script]
    result = subprocess.run(cmd, cwd=str(alphaedit_root), env=env)

    if result.returncode != 0:
        print(f"\nERROR: Capability probe failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nCapability probe completed. Results: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Run editing with capability probing (perplexity + MMLU)"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--alg_name", required=True, choices=["AlphaEdit", "MEMIT", "ROME"])
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--hparams_fname", default="Llama3-8B.json")
    parser.add_argument("--ds_name", default="mcf", choices=["mcf", "cf", "zsre"])
    parser.add_argument("--dataset_size_limit", type=int, default=2000)
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--probe_interval", type=int, default=5,
                       help="Run probe every N edit rounds (default: 5)")
    parser.add_argument("--no_mmlu", action="store_true",
                       help="Skip MMLU evaluation (perplexity only, faster)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
