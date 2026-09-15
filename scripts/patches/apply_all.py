#!/usr/bin/env python3
"""Apply all vendor and baseline patches in order.

Each patch is a single concern, idempotent, and prints what it does.
Run from the project root or any directory (resolves paths automatically).

Usage:
    python scripts/patches/apply_all.py
    python scripts/patches/apply_all.py --vendor-only
    python scripts/patches/apply_all.py --baselines-only
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"

# Add src/util to path for shared patch functions
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "patches"))

import patch_kwargs
import patch_canonical_name
import patch_nan_guard
import patch_p_cache
import patch_glue_map
import patch_model_compat
import patch_mega_batch_eval
import patch_s3_checkpoint
import patch_rect_aligned


def apply_all(vendor: bool = True, baselines: bool = True):
    total = 0
    vendor_root = VENDOR_ROOT if vendor else None
    baselines_root = BASELINES_ROOT if baselines and BASELINES_ROOT.exists() else None

    print("=== Applying patches ===")

    if vendor_root:
        print("\nVendor patches (vendor/AlphaEdit/):")
        total += patch_kwargs.apply(vendor_root=vendor_root)
        total += patch_canonical_name.apply(vendor_root, baselines_root)
        total += patch_nan_guard.apply(vendor_root)
        total += patch_p_cache.apply(vendor_root)
        total += patch_glue_map.apply(vendor_root)
        total += patch_model_compat.apply(vendor_root)

    if baselines_root:
        print("\nBaseline patches (baselines/EvoEdit/):")
        # Copy RECT-Err from vendor submodule (baselines/ is gitignored, not in workdir sync)
        _rect_err_src = PROJECT_ROOT / "vendor" / "OTE-SE-Alignment" / "memit" / "memit_seq_rect_err_main.py"
        _rect_err_dst = baselines_root / "memit" / "memit_seq_rect_err_main.py"
        if _rect_err_src.exists() and _rect_err_dst.parent.exists():
            import shutil
            shutil.copy2(str(_rect_err_src), str(_rect_err_dst))
            # Patch: add torch.cuda.empty_cache() at top of layer loop to prevent OOM on L40S (48GB).
            # The vendor code was developed on A100 (80GB) and doesn't free memory between layers.
            _rect_err_code = _rect_err_dst.read_text()
            _layer_loop_anchor = '    for i, layer in enumerate(hparams.layers):\n        print(f"\\n\\nLAYER {layer}\\n")'
            _layer_loop_patched = '    for i, layer in enumerate(hparams.layers):\n        torch.cuda.empty_cache()\n        print(f"\\n\\nLAYER {layer}\\n")'
            if _layer_loop_anchor in _rect_err_code and _layer_loop_patched not in _rect_err_code:
                _rect_err_dst.write_text(_rect_err_code.replace(_layer_loop_anchor, _layer_loop_patched, 1))
                print(f"  [rect-err] Copied + patched empty_cache() for L40S memory (from vendor/OTE-SE-Alignment)")
            else:
                print(f"  [rect-err] Copied memit_seq_rect_err_main.py from vendor/OTE-SE-Alignment")
        total += patch_kwargs.apply(baselines_root=baselines_root)
        total += patch_mega_batch_eval.apply(baselines_root)
        total += patch_s3_checkpoint.apply(baselines_root)
        total += patch_rect_aligned.apply(baselines_root)

    print(f"\n=== Done: {total} patches applied ===")
    return total


if __name__ == "__main__":
    vendor_only = "--vendor-only" in sys.argv
    baselines_only = "--baselines-only" in sys.argv
    apply_all(
        vendor=not baselines_only,
        baselines=not vendor_only,
    )
