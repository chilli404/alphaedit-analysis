#!/usr/bin/env python3
"""Load cached null-space projection P instead of recomputing 45-min SVD.

Patches vendor/AlphaEdit/experiments/evaluate.py.
Idempotent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "util"))
from source_patches import apply_p_cache_patch


def apply(vendor_root: Path):
    filepath = vendor_root / "experiments" / "evaluate.py"
    source = filepath.read_text()
    patched = apply_p_cache_patch(source)
    if patched == source:
        print("  [p-cache] Already patched")
        return 0
    filepath.write_text(patched)
    print("  [p-cache] Patched evaluate.py")
    return 1


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(project / "vendor" / "AlphaEdit")
