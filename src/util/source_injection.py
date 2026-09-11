"""Shared source reading and patching for vendor code injection.

Extracts the common patterns used by all runners that read vendor source
files as text, apply patches, and exec(compile()) them.
"""

from pathlib import Path

CUDA_PATCH_TARGET = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'


def read_and_patch_evaluate(
    alphaedit_root: Path,
    results_dir: str | None = None,
    dir_name: str | None = None,
) -> str:
    """Read vendor evaluate.py and apply standard patches.

    Patches applied:
      1. Comment out hardcoded CUDA_VISIBLE_DEVICES
      2. Override RESULTS_DIR if provided
      3. Override dir_name if provided

    Returns the patched source as a string (ready for further injection + compile).
    """
    eval_path = alphaedit_root / "experiments" / "evaluate.py"
    source = eval_path.read_text()

    # Comment out CUDA line
    assert CUDA_PATCH_TARGET in source, (
        f"CUDA patch target not found in {eval_path}. "
        f"Vendor code may have changed from pinned commit b84624f."
    )
    source = source.replace(CUDA_PATCH_TARGET, f"# {CUDA_PATCH_TARGET}")

    # Inject RESULTS_DIR override at the top of the file (after imports)
    # The vendor code reads RESULTS_DIR from globals.py; we override it post-import.
    if results_dir:
        source = source.replace(
            "from util.globals import *",
            f'from util.globals import *\nRESULTS_DIR = Path("{results_dir}")',
        )

    # Override dir_name in main() call
    if dir_name:
        source = source.replace(
            'dir_name=args.alg_name,',
            f'dir_name="{dir_name}",',
        )

    return source


def read_and_patch_memit_main(alphaedit_root: Path) -> str:
    """Read vendor memit_main.py and fix relative imports for standalone exec.

    Returns the patched source as a string.
    """
    memit_path = alphaedit_root / "memit" / "memit_main.py"
    source = memit_path.read_text()

    source = source.replace("from .compute_ks", "from memit.compute_ks")
    source = source.replace("from .compute_z", "from memit.compute_z")
    source = source.replace("from .memit_hparams", "from memit.memit_hparams")

    return source
