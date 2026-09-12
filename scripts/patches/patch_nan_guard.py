#!/usr/bin/env python3
"""Skip edits where compute_z produces NaN.

Some target tokens cause numerical overflow in the v-optimization
(particularly on Qwen2.5-7B). When NaN is detected, the edit is
skipped by using the original z representation.

Patches vendor/AlphaEdit/memit/memit_main.py.
Idempotent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "util"))
from source_patches import apply_nan_guard_patch


def apply(vendor_root: Path):
    filepath = vendor_root / "memit" / "memit_main.py"
    source = filepath.read_text()
    patched = apply_nan_guard_patch(source)
    if patched == source:
        print("  [nan-guard] Already patched")
        return 0
    filepath.write_text(patched)
    print("  [nan-guard] Patched memit_main.py")
    return 1


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(project / "vendor" / "AlphaEdit")
