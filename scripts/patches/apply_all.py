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
            # Two patches for L40S 48GB (vendor used A100 80GB):
            # 1. Disable grad for per-layer loop (saves ~12GB autograd graph)
            # 2. Build LHS in steps with cov freed early (saves ~2.5GB peak)
            # Both together give ~15GB headroom. Neither changes the algorithm.
            _rect_err_code = _rect_err_dst.read_text()
            _patched = False

            # Patch 1: no_grad for the per-layer loop
            _loop_anchor = '    # Insert\n    for i, layer in enumerate(hparams.layers):'
            if _loop_anchor in _rect_err_code and 'set_grad_enabled' not in _rect_err_code:
                _loop_patched = (
                    '    # Insert\n'
                    '    # [PATCH] no_grad: compute_z (needs grad) is done; loop only does forward passes.\n'
                    '    torch.set_grad_enabled(False)\n'
                    '    for i, layer in enumerate(hparams.layers):'
                )
                _restore_anchor = '    # Restore state of original model'
                _restore_patched = '    torch.set_grad_enabled(True)\n    # Restore state of original model'
                _rect_err_code = _rect_err_code.replace(_loop_anchor, _loop_patched, 1)
                _rect_err_code = _rect_err_code.replace(_restore_anchor, _restore_patched, 1)
                _patched = True

            # Patch 2: build LHS incrementally, free cov early
            _solve_anchor = (
                '        adj_k = torch.linalg.solve(\n'
                '            hparams.mom2_update_weight * cov.double() + cache_c[i,:,:].cuda().double() + layer_ks @ layer_ks.T + torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"),\n'
                '            layer_ks @ resid.T - error_cache[i,:,:].cuda().double().T,\n'
                '        )'
            )
            if _solve_anchor in _rect_err_code:
                _solve_patched = (
                    '        # [PATCH] Build LHS in steps, free cov early (saves ~2.5GB peak on L40S)\n'
                    '        _lhs = hparams.mom2_update_weight * cov.double()\n'
                    '        cov.cpu(); del cov; torch.cuda.empty_cache()\n'
                    '        _lhs += cache_c[i,:,:].cuda().double()\n'
                    '        _lhs += layer_ks @ layer_ks.T\n'
                    '        _lhs += torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda")\n'
                    '        _rhs = layer_ks @ resid.T - error_cache[i,:,:].cuda().double().T\n'
                    '        adj_k = torch.linalg.solve(_lhs, _rhs)\n'
                    '        del _lhs, _rhs'
                )
                _rect_err_code = _rect_err_code.replace(_solve_anchor, _solve_patched, 1)
                # Fix cov.shape reference in error block — use _lhs_dim instead
                _rect_err_code = _rect_err_code.replace(
                    'if upd_matrix_temp.shape[1] == cov.shape[0]:',
                    'if upd_matrix_temp.shape[1] == layer_ks.shape[0]:'
                )
                # Rebuild cov for error computation
                _err_block = '            mask_ = apply_rect(weights_copy[weight_name], upd_matrix.float())'
                _err_patched = (
                    '            cov = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),\n'
                    '                hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype)\n'
                    '            mask_ = apply_rect(weights_copy[weight_name], upd_matrix.float())'
                )
                _rect_err_code = _rect_err_code.replace(_err_block, _err_patched, 1)
                _patched = True

            if _patched:
                _rect_err_dst.write_text(_rect_err_code)
                print(f"  [rect-err] Copied + L40S memory-patched (from vendor/OTE-SE-Alignment)")
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
