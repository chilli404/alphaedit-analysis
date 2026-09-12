#!/usr/bin/env python3
"""Normalize model.config._name_or_path after model loading.

Without this, _name_or_path can be anything (HF repo ID, S3 path, local path),
causing stats lookup and GLUE eval to fail with KeyError.

Sets _name_or_path to the canonical value that matches:
  - The GLUE context length map keys
  - The covariance stats directory names
  - The REVIVE SVD cache keys

Patches vendor/AlphaEdit/experiments/evaluate.py.
Idempotent.
"""
from pathlib import Path

ANCHOR = "        tok.pad_token = tok.eos_token\n    else:"
PATCH = '''        tok.pad_token = tok.eos_token
        # Normalize _name_or_path to canonical stats directory name
        _mn = model.config._name_or_path.lower()
        if "llama" in _mn:
            model.config._name_or_path = "llama3-8b-instruct"
        elif "gpt-j" in _mn:
            model.config._name_or_path = "gpt-j-6b"
        elif "qwen" in _mn:
            model.config._name_or_path = "qwen2.5-7b-instruct"
    else:'''


def _patch_file(filepath: Path, label: str) -> int:
    if not filepath.exists():
        return 0
    source = filepath.read_text()
    if 'model.config._name_or_path = "llama3-8b-instruct"' in source:
        print(f"  [canonical-name] Already patched: {label}")
        return 0
    if ANCHOR not in source:
        print(f"  [canonical-name] WARNING: anchor not found in {label}")
        return 0
    filepath.write_text(source.replace(ANCHOR, PATCH, 1))
    print(f"  [canonical-name] Patched {label}")
    return 1


def apply(vendor_root: Path = None, baselines_root: Path = None):
    patched = 0
    if vendor_root:
        patched += _patch_file(vendor_root / "experiments" / "evaluate.py", "vendor evaluate.py")
    if baselines_root:
        patched += _patch_file(baselines_root / "experiments" / "evaluate.py", "baselines evaluate.py")
    return patched


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(
        vendor_root=project / "vendor" / "AlphaEdit",
        baselines_root=project / "baselines" / "EvoEdit",
    )
