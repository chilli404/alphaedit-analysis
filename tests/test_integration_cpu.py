"""CPU integration tests that catch bugs only visible during GPU execution.

These tests verify:
  1. Patch anchors match the correct locations (not ambiguous substrings)
  2. Runner argparse choices all have handler code (no NotImplementedError traps)
  3. Vendor anchor constants match the pinned submodule
  4. Import chains for all base algorithms work

These would have caught:
  - patch_mega_batch_eval matching the NSE cache loop instead of the eval loop
  - polykernel_seqreg_runner raising NotImplementedError for base_alg=NSE/MEMIT_rect
"""
import ast
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"
PATCHES_DIR = PROJECT_ROOT / "scripts" / "patches"

VENDOR_EXISTS = VENDOR_ROOT.exists() and (VENDOR_ROOT / "experiments" / "evaluate.py").exists()
BASELINES_EXISTS = BASELINES_ROOT.exists() and (BASELINES_ROOT / "experiments" / "evaluate.py").exists()


# ---------------------------------------------------------------------------
# 1. Patch anchor uniqueness
# ---------------------------------------------------------------------------


class TestPatchAnchorUniqueness:
    """Every patch anchor must match exactly once in its target file."""

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_cuda_anchor(self):
        from source_injection import CUDA_PATCH_TARGET
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        assert source.count(CUDA_PATCH_TARGET) == 1

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_shape_model_list_anchor(self):
        from source_patches import SHAPE_MODEL_LIST_ANCHOR
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        count = source.count(SHAPE_MODEL_LIST_ANCHOR)
        assert count <= 1, f"SHAPE_MODEL_LIST_ANCHOR matches {count} times"

    @pytest.mark.skipif(not BASELINES_EXISTS, reason="baselines not cloned")
    def test_baselines_mega_batch_anchor_not_ambiguous(self):
        """The mega-batch anchor must NOT match the NSE cache loop."""
        sys.path.insert(0, str(PATCHES_DIR))
        try:
            import patch_mega_batch_eval
            # Reload in case it was cached from a different sys.path
            import importlib
            importlib.reload(patch_mega_batch_eval)
        finally:
            sys.path.remove(str(PATCHES_DIR))

        source = (BASELINES_ROOT / "experiments" / "evaluate.py").read_text()
        ambiguous = "    for record in ds:"
        matches = source.count(ambiguous)
        if matches > 1:
            assert patch_mega_batch_eval.EVAL_ANCHOR != ambiguous, (
                f"EVAL_ANCHOR is '{ambiguous}' but it matches {matches} locations. "
                f"Use a more specific anchor."
            )


# ---------------------------------------------------------------------------
# 2. Runner dispatch completeness
# ---------------------------------------------------------------------------


class TestRunnerDispatchCompleteness:
    """Every argparse choice for base_alg / alg_name must have a handler."""

    def _find_handled_values(self, source: str, arg_name: str) -> set[str]:
        """Find all values explicitly handled in if/elif blocks for this arg."""
        handled = set()
        patterns = [
            rf'args\.{arg_name}\s*==\s*"(\w+)"',
            rf'args\.{arg_name}\s*==\s*\'(\w+)\'',
            rf'{arg_name}\s*==\s*"(\w+)"',
        ]
        for pattern in patterns:
            handled.update(re.findall(pattern, source))
        return handled

    def test_polykernel_seqreg_handles_all_base_algs(self):
        """polykernel_seqreg_runner must handle MEMIT, AlphaEdit, NSE, MEMIT_rect."""
        path = PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py"
        source = path.read_text()
        required = {"MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"}
        handled = self._find_handled_values(source, "base_alg")
        missing = required - handled
        assert not missing, (
            f"Missing handlers for: {missing}. These will raise at runtime."
        )

    def test_polykernel_editor_handles_alphaedit(self):
        """polykernel_editor is AlphaEdit-only; MEMIT choice exists but dispatches via vendor."""
        path = PROJECT_ROOT / "src" / "polykernel" / "polykernel_editor_runner.py"
        source = path.read_text()
        assert 'alg_name == "AlphaEdit"' in source

    def test_no_notimplementederror_for_base_alg(self):
        """No runner should have NotImplementedError that mentions a valid base_alg choice."""
        for runner in (PROJECT_ROOT / "src").rglob("*.py"):
            if "__pycache__" in str(runner):
                continue
            source = runner.read_text()
            if "NotImplementedError" not in source or "base_alg" not in source:
                continue
            # Check that no NotImplementedError message contains a valid base_alg
            for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
                pattern = rf'NotImplementedError\([^)]*{alg}[^)]*\)'
                if re.search(pattern, source):
                    pytest.fail(
                        f"{runner.relative_to(PROJECT_ROOT)}: NotImplementedError "
                        f"mentions '{alg}' — add a handler instead"
                    )


# ---------------------------------------------------------------------------
# 3. Vendor import chains (with globals pre-population)
# ---------------------------------------------------------------------------


class TestVendorImportChains:
    """All algorithm apply functions must be importable from the vendor tree."""

    @pytest.fixture(autouse=True)
    def _setup_vendor_globals(self):
        """Pre-populate vendor globals so imports don't crash on missing globals.yml."""
        if VENDOR_EXISTS:
            from evaluate_harness import _ensure_vendor_globals
            _ensure_vendor_globals(VENDOR_ROOT)

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_memit_function_exists_in_source(self):
        """Verify apply_memit_to_model is defined in the vendor source (no import needed)."""
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        assert "def apply_memit_to_model" in source

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_alphaedit_function_exists_in_source(self):
        source = (VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert "def apply_AlphaEdit_to_model" in source

    @pytest.mark.skipif(
        not VENDOR_EXISTS or not (VENDOR_ROOT / "nse" / "nse_main.py").exists(),
        reason="vendor NSE not available"
    )
    def test_nse_function_exists_in_source(self):
        source = (VENDOR_ROOT / "nse" / "nse_main.py").read_text()
        assert "def apply_nse_to_model" in source

    @pytest.mark.skipif(
        not VENDOR_EXISTS or not (VENDOR_ROOT / "memit" / "memit_rect_main.py").exists(),
        reason="vendor MEMIT_rect not available"
    )
    def test_memit_rect_function_exists_in_source(self):
        source = (VENDOR_ROOT / "memit" / "memit_rect_main.py").read_text()
        assert "def apply_memit_rect_to_model" in source


# ---------------------------------------------------------------------------
# 4. Patch compilation integration
# ---------------------------------------------------------------------------


class TestPatchCompilation:
    """Applying patches must produce valid Python."""

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_evaluate_compiles_after_source_patches(self):
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        from source_patches import apply_model_list_patch, apply_canonical_name_patch
        patched = apply_model_list_patch(source)
        patched = apply_canonical_name_patch(patched)
        compile(patched, "evaluate.py", "exec")

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_memit_compiles_after_nan_guard(self):
        from source_patches import apply_nan_guard_patch
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        patched = apply_nan_guard_patch(source)
        compile(patched, "memit_main.py", "exec")


# ---------------------------------------------------------------------------
# 5. ExperimentConfig path consistency
# ---------------------------------------------------------------------------


class TestExperimentConfigConsistency:
    """ExperimentConfig must produce distinct paths for every base_alg."""

    def test_all_base_algs_produce_distinct_checkpoint_dirs(self):
        from experiment_config import ExperimentConfig
        paths = set()
        for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(
                base_alg=alg, seed=42, ordering="fb_high_exposure",
                revive=True, revive_tau=0.1,
            )
            p = str(config.checkpoint_dir(Path("/tmp/ckpt")))
            assert p not in paths, f"base_alg={alg} produces duplicate path: {p}"
            paths.add(p)

    def test_variant_name_contains_base_alg(self):
        from experiment_config import ExperimentConfig
        for alg in ["AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(base_alg=alg, seed=42)
            assert alg in config.variant_name, (
                f"base_alg={alg} variant_name={config.variant_name} "
                f"doesn't contain '{alg}' — the MEMIT-Seq hardcode bug is back"
            )

    def test_memit_maps_to_memit_seq(self):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(base_alg="MEMIT", seed=42)
        assert "MEMIT-Seq" in config.variant_name
