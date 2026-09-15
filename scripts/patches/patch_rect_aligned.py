#!/usr/bin/env python3
"""Add MEMIT_seq_rect and MEMIT_seq_rect_err (RECT-Aligned) support to baselines evaluate.py.

Adds:
1. Import for apply_memit_seq_rect_to_model and apply_memit_seq_rect_err_to_model
2. ALG_DICT entries for MEMIT_seq_rect and MEMIT_seq_rect_err
3. error_cache initialization for rect algorithms
4. error_cache passing to apply_algo
5. 3-tuple result unpacking (model, cache_c, error_cache)
6. MEMIT_seq_rect / MEMIT_seq_rect_err in all alg_name check lists

Idempotent — returns 0 patches if already applied.
"""
import sys
from pathlib import Path


def apply(baselines_root: Path):
    eval_path = baselines_root / "experiments" / "evaluate.py"
    if not eval_path.exists():
        return 0

    source = eval_path.read_text()

    # Check if FULLY patched (not just partially — old manual edits may have ALG_DICT but not choices)
    choices_region = source[source.find("choices="):source.find("choices=")+300] if "choices=" in source else ""
    if "MEMIT_seq_rect_err" in source and "MEMIT_seq_rect_err" in choices_region:
        print("  [rect-aligned] Already patched")
        return 0
    # If partially patched (MEMIT_seq_rect in ALG_DICT but not choices), we can't re-apply
    # cleanly. Reset the key sections.
    if "MEMIT_seq_rect" in source:
        print("  [rect-aligned] Partially patched — forcing re-patch")

    patched = source

    # 1. Add imports
    patched = patched.replace(
        "from memit.memit_rect_main import apply_memit_rect_to_model",
        "from memit.memit_rect_main import apply_memit_rect_to_model\n"
        "from memit.memit_seq_rect_main import apply_memit_seq_rect_to_model\n"
        "from memit.memit_seq_rect_err_main import apply_memit_seq_rect_err_to_model",
    )

    # 2. Add ALG_DICT entries
    patched = patched.replace(
        '    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),',
        '    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),\n'
        '    "MEMIT_seq_rect": (MEMITHyperParams, apply_memit_seq_rect_to_model),\n'
        '    "MEMIT_seq_rect_err": (MEMITHyperParams, apply_memit_seq_rect_err_to_model),',
    )

    # 3. Add MEMIT_seq_rect and MEMIT_seq_rect_err to argparse choices
    patched = patched.replace(
        '"MEMIT_seq","MEMIT_prune"',
        '"MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err","MEMIT_prune"',
    )

    # 4. Add MEMIT_seq_rect and MEMIT_seq_rect_err to all alg_name check lists
    # cache_template check
    patched = patched.replace(
        '"MEMIT","AlphaEdit", "EvoEdit", "MEMIT_seq", "MEMIT_prune", "MEMIT_rect"',
        '"MEMIT","AlphaEdit", "EvoEdit", "MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err", "MEMIT_prune", "MEMIT_rect"',
    )
    # Sequential editing check (cache_c init)
    patched = patched.replace(
        '"EvoEdit","AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"',
        '"EvoEdit","AlphaEdit", "MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err", "MEMIT_prune", "NSE"',
    )
    # etc_args
    patched = patched.replace(
        '"ROME", "MEMIT", "EvoEdit", "AlphaEdit", "MEMIT_seq", "MEMIT_prune", "NSE"',
        '"ROME", "MEMIT", "EvoEdit", "AlphaEdit", "MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err", "MEMIT_prune", "NSE"',
    )
    # seq_args
    patched = patched.replace(
        'dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "EvoEdit", "MEMIT_seq", "NSE"])',
        'dict(cache_c=cache_c) if any(alg in alg_name for alg in ["AlphaEdit", "EvoEdit", "MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err", "NSE"])',
    )
    # apply_algo dispatch
    patched = patched.replace(
        'if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "NSE", "EvoEdit"]):',
        'if any(alg in alg_name for alg in ["AlphaEdit", "MEMIT_seq", "MEMIT_seq_rect", "MEMIT_seq_rect_err", "NSE", "EvoEdit"]):',
    )

    # 5. Add error_cache init after cache_c init
    patched = patched.replace(
        '            if alg_name == "AlphaEdit" or alg_name == "EvoEdit":\n                P = torch.zeros',
        '            if "rect" in alg_name:\n'
        '                error_cache = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[1]), device="cpu", dtype=torch.double)\n'
        '            if alg_name == "AlphaEdit" or alg_name == "EvoEdit":\n                P = torch.zeros',
    )

    # 6. Add error_args to apply_algo call
    patched = patched.replace(
        '        seq_args = dict(cache_c=cache_c)',
        '        error_args = dict(error_cache=error_cache) if "rect" in alg_name and \'error_cache\' in dir() else dict()\n'
        '        seq_args = dict(cache_c=cache_c)',
    )
    patched = patched.replace(
        '                **seq_args,\n                **nc_args,',
        '                **seq_args,\n                **error_args,\n                **nc_args,',
    )

    # 7. Handle 3-tuple result (model, cache_c, error_cache)
    patched = patched.replace(
        '            edited_model, cache_c = apply_algo(',
        '            _result = apply_algo(',
    )
    # Add unpacking after the apply_algo call block
    patched = patched.replace(
        '            )\n        elif alg_name == "MEMIT_prune":',
        '            )\n'
        '            if isinstance(_result, tuple) and len(_result) == 3:\n'
        '                edited_model, cache_c, error_cache = _result\n'
        '            else:\n'
        '                edited_model, cache_c = _result\n'
        '        elif alg_name == "MEMIT_prune":',
    )

    # 8. P matrix caching (avoid recomputing 45-min SVD)
    patched = patched.replace(
        '    if alg_name == "AlphaEdit" or alg_name == "EvoEdit":\n'
        '        for i, layer in enumerate(hparams.layers):\n'
        '            P[i,:,:] = get_project(model,tok,layer,hparams)\n'
        '        torch.save(P, "null_space_project.pt")',
        '    if alg_name == "AlphaEdit" or alg_name == "EvoEdit":\n'
        '        if Path("null_space_project.pt").exists():\n'
        '            P = torch.load("null_space_project.pt", map_location="cpu")\n'
        '            print(f"Loaded cached null-space projection from null_space_project.pt (shape={P.shape})")\n'
        '        else:\n'
        '            for i, layer in enumerate(hparams.layers):\n'
        '                P[i,:,:] = get_project(model,tok,layer,hparams)\n'
        '            torch.save(P, "null_space_project.pt")\n'
        '            print(f"Computed and cached null-space projection to null_space_project.pt")',
    )

    # 9. Add Model dtype print after model loading
    patched = patched.replace(
        '        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()\n'
        '        tok = AutoTokenizer.from_pretrained(model_name)',
        '        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()\n'
        '        print(f"  Model dtype: {next(model.parameters()).dtype}")\n'
        '        tok = AutoTokenizer.from_pretrained(model_name)',
    )

    if patched != source:
        eval_path.write_text(patched)
        print("  [rect-aligned] Patched baselines evaluate.py")
        return 1

    print("  [rect-aligned] No changes needed")
    return 0


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(project / "baselines" / "EvoEdit")
