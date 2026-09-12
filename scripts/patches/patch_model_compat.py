#!/usr/bin/env python3
"""Model compatibility: add Qwen2.5-7B support + fix Wikipedia dataset config.

Three patches in one file because they're all about making the vendor code
work with models beyond the original GPT-2/Llama-2/Llama-3:

1. Model-list: Add Qwen2.5-7B to cache_c/P shape initialization whitelist
2. Model-dtype: Load Qwen in float32 (bfloat16 causes NaN in compute_z)
3. Wikipedia: Fix deprecated dataset config (20200501.en → 20220301.en)

Patches vendor evaluate.py and vendor rome/layer_stats.py.
Idempotent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "util"))
from source_patches import (
    apply_model_list_patch,
    apply_model_dtype_patch,
    patch_layer_stats_file,
)


def apply(vendor_root: Path):
    patched = 0

    # evaluate.py: model list + dtype
    eval_path = vendor_root / "experiments" / "evaluate.py"
    source = eval_path.read_text()
    new_source = apply_model_list_patch(source)
    new_source = apply_model_dtype_patch(new_source)
    if new_source != source:
        eval_path.write_text(new_source)
        patched += 1
        print("  [model-compat] Patched evaluate.py (model-list + dtype)")
    else:
        print("  [model-compat] evaluate.py already patched")

    # layer_stats.py: Wikipedia config
    patch_layer_stats_file(vendor_root)
    patched += 1

    return patched


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(project / "vendor" / "AlphaEdit")
