#!/usr/bin/env python3
"""Replace the vendor per-record evaluation loop with batched evaluation.

The vendor loop does 26 separate forward passes per record (~0.4 rec/s).
Mega-batch eval batches 4 records into one forward pass (~2.5 rec/s, 4-10x faster).

Patches baselines/EvoEdit/experiments/evaluate.py on disk.
Vendor evaluate.py is patched at runtime by each runner (not on disk).
Idempotent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "util"))
from mega_batch_eval import get_mega_batch_eval_source


EVAL_ANCHOR = "    gen_test_vars = [snips, vec]\n    for record in ds:"


def apply(baselines_root: Path = None):
    """Patch baselines evaluate.py with mega_batch_eval."""
    if not baselines_root:
        return 0

    eval_path = baselines_root / "experiments" / "evaluate.py"
    if not eval_path.exists():
        print("  [mega-batch] baselines evaluate.py not found")
        return 0

    source = eval_path.read_text()
    if "_mega_batch_eval" in source:
        print("  [mega-batch] Already patched")
        return 0

    if EVAL_ANCHOR not in source:
        print("  [mega-batch] WARNING: eval anchor not found")
        return 0

    fn_src = get_mega_batch_eval_source()
    fn_indented = "\n".join("    " + line for line in fn_src.strip().split("\n"))
    call = '''    # === MEGA-BATCH EVAL (injected by patch_mega_batch_eval.py) ===
    if not os.environ.get("SKIP_MEGA_BATCH_EVAL"):
        _mega_batch_eval(edited_model, tok, list(ds), case_result_template, num_edits, case_ids, exec_time, batch_size=4)
    else:
        print("  [MEGA-BATCH EVAL] Skipped (SKIP_MEGA_BATCH_EVAL=1)")
    # === END MEGA-BATCH EVAL ===
    if False:  # skip vendor per-record loop
        for record in ds:'''

    replacement = "    gen_test_vars = [snips, vec]\n" + fn_indented + "\n" + call
    patched = source.replace(EVAL_ANCHOR, replacement, 1)
    eval_path.write_text(patched)
    print("  [mega-batch] Patched baselines evaluate.py")
    return 1


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(baselines_root=project / "baselines" / "EvoEdit")
