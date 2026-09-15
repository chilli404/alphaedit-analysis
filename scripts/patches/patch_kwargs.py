#!/usr/bin/env python3
"""Add **_kwargs to all algorithm apply functions.

The vendor evaluate.py passes return_orig_weights_device as a kwarg.
Without **_kwargs, algorithms crash with TypeError at runtime.

Patches both vendor/AlphaEdit/ and baselines/EvoEdit/ trees.
Idempotent — safe to run multiple times.
"""
import re
import sys
from pathlib import Path


def _add_kwargs_to_signature(source: str, func_name: str) -> tuple[str, bool]:
    """Add **_kwargs to a function's signature. Returns (patched_source, changed)."""
    if "**_kwargs" in source:
        return source, False

    pattern = rf"(def {func_name}\([^)]*)\)(\s*(?:->|:))"
    match = re.search(pattern, source, re.DOTALL)
    if not match:
        return source, False

    sig = match.group(1)
    rest = match.group(2)
    if sig.rstrip().endswith(","):
        new_sig = sig + " **_kwargs,"
    else:
        new_sig = sig + ", **_kwargs,"
    return source[:match.start()] + new_sig + ")" + rest + source[match.end():], True


VENDOR_TARGETS = [
    ("memit/memit_main.py", "apply_memit_to_model"),
    ("AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
    ("nse/nse_main.py", "apply_nse_to_model"),
]

BASELINES_TARGETS = [
    ("EvoEdit/EvoEdit_main.py", "apply_EvoEdit_to_model"),
    ("nse/nse_main.py", "apply_nse_to_model"),
    ("memit/memit_main.py", "apply_memit_to_model"),
    ("memit/memit_seq_main.py", "apply_memit_seq_to_model"),
    ("memit/memit_rect_main.py", "apply_memit_rect_to_model"),
    ("memit/memit_seq_rect_main.py", "apply_memit_seq_rect_to_model"),
    ("memit/memit_seq_rect_err_main.py", "apply_memit_seq_rect_err_to_model"),
    ("AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
]


def apply(vendor_root: Path = None, baselines_root: Path = None):
    """Apply kwargs patches to all algorithm files."""
    patched = 0

    if vendor_root:
        for relpath, func in VENDOR_TARGETS:
            filepath = vendor_root / relpath
            if not filepath.exists():
                continue
            source = filepath.read_text()
            new_source, changed = _add_kwargs_to_signature(source, func)
            if changed:
                filepath.write_text(new_source)
                patched += 1
                print(f"  [kwargs] Patched {func} in vendor/{relpath}")
            else:
                print(f"  [kwargs] Already patched: vendor/{relpath}")

    if baselines_root:
        for relpath, func in BASELINES_TARGETS:
            filepath = baselines_root / relpath
            if not filepath.exists():
                continue
            source = filepath.read_text()
            new_source, changed = _add_kwargs_to_signature(source, func)
            if changed:
                filepath.write_text(new_source)
                patched += 1
                print(f"  [kwargs] Patched {func} in baselines/{relpath}")
            else:
                print(f"  [kwargs] Already patched: baselines/{relpath}")

    return patched


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(
        vendor_root=project / "vendor" / "AlphaEdit",
        baselines_root=project / "baselines" / "EvoEdit",
    )
