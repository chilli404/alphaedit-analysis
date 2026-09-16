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
            # Patch: disable grad before the per-layer editing loop.
            # All compute_z calls (which need gradients for v_star optimization) are done
            # BEFORE this loop. The loop only does forward passes (compute_ks, get_module_input_output)
            # which don't need gradients. On A100 (80GB) the autograd overhead fits; on L40S (48GB)
            # it OOMs at BS=100 because PyTorch caches activations for 100 forward passes.
            _rect_err_code = _rect_err_dst.read_text()
            _loop_anchor = '    # Insert\n    for i, layer in enumerate(hparams.layers):'
            _loop_patched = (
                '    # Insert\n'
                '    # [PATCH] Disable grad for per-layer loop (compute_z already done above).\n'
                '    # Prevents autograd from caching activations — saves ~12GB on Llama-3-8B.\n'
                '    torch.set_grad_enabled(False)\n'
                '    for i, layer in enumerate(hparams.layers):'
            )
            if _loop_anchor in _rect_err_code and 'set_grad_enabled' not in _rect_err_code:
                # Also re-enable grad after the loop (before weight restore)
                _restore_anchor = '    # Restore state of original model'
                _restore_patched = '    torch.set_grad_enabled(True)\n    # Restore state of original model'
                _rect_err_code = _rect_err_code.replace(_loop_anchor, _loop_patched, 1)
                _rect_err_code = _rect_err_code.replace(_restore_anchor, _restore_patched, 1)
                _rect_err_dst.write_text(_rect_err_code)
                print(f"  [rect-err] Copied + no-grad-patched for L40S (from vendor/OTE-SE-Alignment)")
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
