#!/usr/bin/env python3
"""Add GPT-J and Qwen to the GLUE context length map.

The vendor code only has entries for GPT-2 and Llama-2. We add
gpt-j-6b (2048) and qwen2.5-7b-instruct (4096).

Patches vendor/AlphaEdit/glue_eval/useful_functions.py.
Idempotent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "util"))
from source_patches import apply_glue_context_patch


def apply(vendor_root: Path):
    filepath = vendor_root / "glue_eval" / "useful_functions.py"
    source = filepath.read_text()
    patched = apply_glue_context_patch(source)
    if patched == source:
        print("  [glue-map] Already patched")
        return 0
    filepath.write_text(patched)
    print("  [glue-map] Patched useful_functions.py")
    return 1


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(project / "vendor" / "AlphaEdit")
