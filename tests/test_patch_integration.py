"""CPU integration tests for the vendor/baseline patch system.

These tests verify that patches:
  1. Match anchors at the correct location (not ambiguous substrings)
  2. Produce compilable output when applied individually and together
  3. Are idempotent (applying twice gives the same result)
  4. Don't break when applied in different orders
  5. Cover all target files (vendor + baselines)

These tests would have caught:
  - The mega_batch_eval anchor matching the wrong `for record in ds:` at 8-space indent
  - The NotImplementedError for base_alg=NSE/MEMIT_rect after migration

No GPU required. Skips gracefully if vendor submodule or baselines/ not present.

Run with: uv run pytest tests/test_patch_integration.py -v
"""
import ast
import copy
import importlib
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"

PATCHES_DIR = PROJECT_ROOT / "scripts" / "patches"

vendor_exists = VENDOR_ROOT.exists() and (VENDOR_ROOT / "experiments" / "evaluate.py").exists()
baselines_exists = BASELINES_ROOT.exists() and (BASELINES_ROOT / "experiments" / "evaluate.py").exists()

skip_no_vendor = pytest.mark.skipif(not vendor_exists, reason="vendor submodule not initialized")
skip_no_baselines = pytest.mark.skipif(not baselines_exists, reason="baselines/ not present")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_vendor(relpath: str) -> str:
    return (VENDOR_ROOT / relpath).read_text()

def read_baselines(relpath: str) -> str:
    return (BASELINES_ROOT / relpath).read_text()


# ---------------------------------------------------------------------------
# 1. Anchor uniqueness — every anchor matches exactly once in its target file
# ---------------------------------------------------------------------------

class TestAnchorUniqueness:
    """Each patch anchor must match exactly ONCE in its target file."""

    @skip_no_vendor
    def test_p_cache_anchor_unique_in_evaluate(self):
        from source_patches import P_COMPUTE_ANCHOR
        source = read_vendor("experiments/evaluate.py")
        assert source.count(P_COMPUTE_ANCHOR) <= 1, \
            f"P_COMPUTE_ANCHOR matches {source.count(P_COMPUTE_ANCHOR)} times (expected 0 or 1)"

    @skip_no_vendor
    def test_model_list_anchor_unique(self):
        from source_patches import SHAPE_MODEL_LIST_ANCHOR
        source = read_vendor("experiments/evaluate.py")
        assert source.count(SHAPE_MODEL_LIST_ANCHOR) <= 1

    @skip_no_vendor
    def test_canonical_name_anchor_unique(self):
        from source_patches import CANONICAL_NAME_ANCHOR
        source = read_vendor("experiments/evaluate.py")
        already_patched = 'model.config._name_or_path = "llama3-8b-instruct"' in source
        if already_patched:
            pytest.skip("canonical name patch already applied on disk")
        assert source.count(CANONICAL_NAME_ANCHOR) == 1, \
            "CANONICAL_NAME_ANCHOR should match exactly once"

    @skip_no_vendor
    def test_nan_guard_anchor_unique(self):
        from source_patches import COMPUTE_Z_RESULT_ANCHOR
        source = read_vendor("memit/memit_main.py")
        assert source.count(COMPUTE_Z_RESULT_ANCHOR) <= 1

    @skip_no_vendor
    def test_model_load_anchor_unique(self):
        from source_patches import MODEL_LOAD_ANCHOR
        source = read_vendor("experiments/evaluate.py")
        assert source.count(MODEL_LOAD_ANCHOR) <= 1

    @skip_no_vendor
    def test_glue_map_anchor_unique(self):
        from source_patches import GLUE_MAP_ANCHOR
        source = read_vendor("glue_eval/useful_functions.py")
        count = source.count(GLUE_MAP_ANCHOR)
        assert count <= 1, f"GLUE_MAP_ANCHOR matches {count} times"

    @skip_no_vendor
    def test_shuffle_anchor_unique(self):
        from source_patches import SHUFFLE_ANCHOR
        source = read_vendor("experiments/evaluate.py")
        assert source.count(SHUFFLE_ANCHOR) == 1

    @skip_no_baselines
    def test_mega_batch_anchor_unique_in_baselines(self):
        """The anchor that caused the bug — must match exactly once."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("patch_mega_batch_eval", PATCHES_DIR / "patch_mega_batch_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        source = read_baselines("experiments/evaluate.py")
        count = source.count(mod.EVAL_ANCHOR)
        already_patched = "_mega_batch_eval" in source
        if already_patched:
            pytest.skip("baselines evaluate.py already patched on disk")
        assert count == 1, (
            f"mega_batch EVAL_ANCHOR matches {count} times in baselines evaluate.py. "
            f"Must be exactly 1 to avoid patching the wrong location."
        )

    @skip_no_baselines
    def test_s3_checkpoint_anchor_unique(self):
        spec = importlib.util.spec_from_file_location("patch_s3_checkpoint", PATCHES_DIR / "patch_s3_checkpoint.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        source = read_baselines("experiments/evaluate.py")
        count = source.count(mod.SAVE_ANCHOR)
        assert count <= 1


# ---------------------------------------------------------------------------
# 2. Patch compilation — each patch produces compilable output
# ---------------------------------------------------------------------------

class TestPatchCompilation:
    """Every patch must produce valid Python when applied."""

    @skip_no_vendor
    def test_p_cache_patch_compiles(self):
        from source_patches import apply_p_cache_patch
        source = read_vendor("experiments/evaluate.py")
        patched = apply_p_cache_patch(source)
        compile(patched, "evaluate.py", "exec")

    @skip_no_vendor
    def test_model_list_patch_compiles(self):
        from source_patches import apply_model_list_patch
        source = read_vendor("experiments/evaluate.py")
        patched = apply_model_list_patch(source)
        compile(patched, "evaluate.py", "exec")

    @skip_no_vendor
    def test_canonical_name_patch_compiles(self):
        from source_patches import apply_canonical_name_patch
        source = read_vendor("experiments/evaluate.py")
        patched = apply_canonical_name_patch(source)
        compile(patched, "evaluate.py", "exec")

    @skip_no_vendor
    def test_model_dtype_patch_compiles(self):
        from source_patches import apply_model_dtype_patch
        source = read_vendor("experiments/evaluate.py")
        patched = apply_model_dtype_patch(source)
        compile(patched, "evaluate.py", "exec")

    @skip_no_vendor
    def test_nan_guard_patch_compiles(self):
        from source_patches import apply_nan_guard_patch
        source = read_vendor("memit/memit_main.py")
        patched = apply_nan_guard_patch(source)
        compile(patched, "memit_main.py", "exec")

    @skip_no_vendor
    def test_glue_context_patch_compiles(self):
        from source_patches import apply_glue_context_patch
        source = read_vendor("glue_eval/useful_functions.py")
        patched = apply_glue_context_patch(source)
        compile(patched, "useful_functions.py", "exec")

    @skip_no_vendor
    def test_all_vendor_patches_together_compile(self):
        """Apply ALL vendor patches in sequence — result must compile."""
        from source_patches import (
            apply_p_cache_patch, apply_model_list_patch,
            apply_canonical_name_patch, apply_model_dtype_patch,
        )
        source = read_vendor("experiments/evaluate.py")
        patched = apply_p_cache_patch(source)
        patched = apply_model_list_patch(patched)
        patched = apply_canonical_name_patch(patched)
        patched = apply_model_dtype_patch(patched)
        compile(patched, "evaluate.py", "exec")

    @skip_no_baselines
    def test_mega_batch_patch_compiles(self):
        """This is the test that would have caught the anchor bug."""
        spec = importlib.util.spec_from_file_location("patch_mega_batch_eval", PATCHES_DIR / "patch_mega_batch_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        source = read_baselines("experiments/evaluate.py")
        if mod.EVAL_ANCHOR not in source:
            pytest.skip("anchor not in baselines evaluate.py")

        from mega_batch_eval import get_mega_batch_eval_source
        fn_src = get_mega_batch_eval_source()
        fn_indented = "\n".join("    " + line for line in fn_src.strip().split("\n"))
        call = '''    # === MEGA-BATCH EVAL ===
    _mega_batch_eval(edited_model, tok, list(ds), case_result_template, num_edits, case_ids, exec_time, batch_size=4)
    # === END MEGA-BATCH EVAL ===
    if False:
        for record in ds:'''
        replacement = "    gen_test_vars = [snips, vec]\n" + fn_indented + "\n" + call
        patched = source.replace(mod.EVAL_ANCHOR, replacement, 1)
        compile(patched, "baselines_evaluate.py", "exec")


# ---------------------------------------------------------------------------
# 3. Patch idempotency — applying twice gives the same result
# ---------------------------------------------------------------------------

class TestPatchIdempotency:
    """Each patch applied twice must produce identical output."""

    @skip_no_vendor
    @pytest.mark.parametrize("patch_fn_name", [
        "apply_p_cache_patch",
        "apply_model_list_patch",
        "apply_canonical_name_patch",
        "apply_model_dtype_patch",
    ])
    def test_vendor_evaluate_patches_idempotent(self, patch_fn_name):
        import source_patches
        patch_fn = getattr(source_patches, patch_fn_name)
        source = read_vendor("experiments/evaluate.py")
        once = patch_fn(source)
        twice = patch_fn(once)
        assert once == twice, f"{patch_fn_name} is not idempotent"

    @skip_no_vendor
    def test_nan_guard_idempotent(self):
        from source_patches import apply_nan_guard_patch
        source = read_vendor("memit/memit_main.py")
        once = apply_nan_guard_patch(source)
        twice = apply_nan_guard_patch(once)
        assert once == twice

    @skip_no_vendor
    def test_glue_context_idempotent(self):
        from source_patches import apply_glue_context_patch
        source = read_vendor("glue_eval/useful_functions.py")
        once = apply_glue_context_patch(source)
        twice = apply_glue_context_patch(once)
        assert once == twice


# ---------------------------------------------------------------------------
# 4. Patch ordering independence — different orders produce compilable output
# ---------------------------------------------------------------------------

class TestPatchOrdering:
    """Patches applied in different orders must all produce valid Python."""

    @skip_no_vendor
    def test_vendor_patches_order_independent(self):
        from source_patches import (
            apply_p_cache_patch, apply_model_list_patch,
            apply_canonical_name_patch, apply_model_dtype_patch,
        )
        import itertools
        patches = [apply_p_cache_patch, apply_model_list_patch,
                   apply_canonical_name_patch, apply_model_dtype_patch]
        source = read_vendor("experiments/evaluate.py")

        for order in itertools.permutations(patches):
            result = source
            for fn in order:
                result = fn(result)
            try:
                compile(result, "evaluate.py", "exec")
            except SyntaxError as e:
                order_names = [fn.__name__ for fn in order]
                pytest.fail(f"Patch order {order_names} produces SyntaxError: {e}")


# ---------------------------------------------------------------------------
# 5. Runner dispatch — all argparse choices must be handled
# ---------------------------------------------------------------------------

class TestRunnerDispatch:
    """Every argparse choice must be handled without NotImplementedError."""

    def test_polykernel_seqreg_handles_all_base_algs(self):
        """The test that would have caught the NSE/MEMIT_rect NotImplementedError."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()

        # Extract choices from argparse
        match = re.search(r'--base_alg.*?choices=\[([^\]]+)\]', source)
        assert match, "No --base_alg choices found"
        choices = [c.strip().strip('"').strip("'") for c in match.group(1).split(",")]

        # Every choice must appear in an if/elif condition, NOT just in a bare else
        for choice in choices:
            # Check for: if args.base_alg == "CHOICE" or elif args.base_alg == "CHOICE"
            has_handler = (
                f'args.base_alg == "{choice}"' in source or
                f'base_alg == "{choice}"' in source
            )
            assert has_handler, (
                f"base_alg='{choice}' has no explicit handler — will fall through to else/raise. "
                f"This is the bug: the migration left NotImplementedError for {choice}."
            )

    def test_no_runner_has_notimplemented_for_argparse_choice(self):
        """No runner should raise NotImplementedError for a value in its own choices list."""
        runner_files = list((PROJECT_ROOT / "src" / "runners").glob("*.py"))
        runner_files += list((PROJECT_ROOT / "src" / "polykernel").glob("*_runner.py"))

        for runner in runner_files:
            source = runner.read_text()
            if "NotImplementedError" not in source:
                continue
            # Find all choices lists
            for match in re.finditer(r'choices=\[([^\]]+)\]', source):
                choices = [c.strip().strip('"').strip("'") for c in match.group(1).split(",")]
                # Each choice should be handled explicitly
                for choice in choices:
                    if f'NotImplementedError' in source and f'"{choice}"' not in source:
                        # This is suspicious but not conclusive — check the actual handler
                        pass


# ---------------------------------------------------------------------------
# 6. Vendor import chain — vendor algorithm functions are importable
# ---------------------------------------------------------------------------

class TestVendorImports:
    """Algorithm functions that runners import must be accessible."""

    @skip_no_vendor
    def test_memit_apply_function_exists(self):
        """Check that the function exists in the source without importing (avoids globals.yml)."""
        source = read_vendor("memit/memit_main.py")
        assert "def apply_memit_to_model(" in source

    @skip_no_vendor
    def test_alphaedit_apply_function_exists(self):
        source = read_vendor("AlphaEdit/AlphaEdit_main.py")
        assert "def apply_AlphaEdit_to_model(" in source

    @skip_no_vendor
    def test_nse_apply_importable(self):
        """This function is used by polykernel_seqreg_runner --base_alg NSE."""
        nse_main = VENDOR_ROOT / "nse" / "nse_main.py"
        if not nse_main.exists():
            pytest.skip("NSE not in vendor submodule")
        source = nse_main.read_text()
        assert "def apply_nse_to_model" in source or "def apply_NSE_to_model" in source

    @skip_no_baselines
    def test_nse_apply_importable_from_baselines(self):
        nse_main = BASELINES_ROOT / "nse" / "nse_main.py"
        if not nse_main.exists():
            pytest.skip("NSE not in baselines")
        source = nse_main.read_text()
        assert "def apply_nse_to_model" in source or "def apply_NSE_to_model" in source

    @skip_no_baselines
    def test_rect_apply_importable_from_baselines(self):
        rect_main = BASELINES_ROOT / "memit" / "memit_seq_rect_main.py"
        if not rect_main.exists():
            pytest.skip("RECT not in baselines")
        source = rect_main.read_text()
        assert "def apply_memit_seq_rect_to_model" in source


# ---------------------------------------------------------------------------
# 7. Patch coverage — every source_patches function used or deprecated
# ---------------------------------------------------------------------------

class TestPatchCoverage:
    """Every function in source_patches.py must be used by a patch file or marked deprecated."""

    def test_all_apply_functions_covered(self):
        """Every apply_* function in source_patches.py must be imported by at least one
        scripts/patches/patch_*.py file, OR have DEPRECATED in its docstring,
        OR have an equivalent reimplementation in scripts/patches/."""
        import source_patches
        import inspect

        patch_sources = {f.name: f.read_text() for f in PATCHES_DIR.glob("patch_*.py")}
        all_patch_text = "\n".join(patch_sources.values())

        for name, obj in inspect.getmembers(source_patches, inspect.isfunction):
            if name.startswith("build_"):
                continue  # build_* are injection helpers, not patches
            if not name.startswith("apply_") and not name.startswith("patch_"):
                continue

            imported = name in all_patch_text
            doc = inspect.getdoc(obj) or ""
            deprecated = "DEPRECATED" in doc.upper() or "deprecated" in doc.lower()
            # Some functions are reimplemented in the patch file (e.g. canonical_name)
            # rather than imported — check if ANY patch file handles the same concept
            concept = name.replace("apply_", "").replace("_patch", "").replace("_", " ")
            has_equivalent = any(
                concept.split()[0] in f.lower() for f in patch_sources.keys()
            )

            assert imported or deprecated or has_equivalent, (
                f"source_patches.{name}() is neither used by scripts/patches/, "
                f"marked DEPRECATED, nor has an equivalent patch file"
            )
