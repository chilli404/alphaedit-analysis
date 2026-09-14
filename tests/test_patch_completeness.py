#!/usr/bin/env python3
"""Verify that ALL algorithm functions accept **_kwargs after patching.

Every apply_*_to_model function gets called by runners that pass extra kwargs
(return_orig_weights, return_orig_weights_device, etc.). If any function
doesn't accept **_kwargs, the runner crashes at runtime.

Also verify that the baselines evaluate.py has all required algorithm entries
after patching (MEMIT_seq_rect, etc.).

These tests run AFTER patches are applied (by the test session or CI).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Every apply_*_to_model must accept **_kwargs
# ============================================================================

class TestVendorKwargsPatched:
    """Vendor algorithm functions (imported by checkpoint_runner, polykernel_seqreg_runner)."""

    @pytest.mark.parametrize("relpath,func_name", [
        ("vendor/AlphaEdit/memit/memit_main.py", "apply_memit_to_model"),
        ("vendor/AlphaEdit/AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
        ("vendor/AlphaEdit/nse/nse_main.py", "apply_nse_to_model"),
    ])
    def test_vendor_has_kwargs(self, relpath, func_name):
        path = PROJECT_ROOT / relpath
        if not path.exists():
            pytest.skip(f"{relpath} not found")
        source = path.read_text()
        # Find the function def and check for **_kwargs
        func_start = source.find(f"def {func_name}(")
        assert func_start > 0, f"{func_name} not found in {relpath}"
        # Get the full signature (may span multiple lines)
        sig_end = source.find(")", func_start) + 1
        sig = source[func_start:sig_end]
        assert "**_kwargs" in sig or "**kwargs" in sig, \
            f"{relpath}:{func_name} must accept **_kwargs — " \
            f"runners pass return_orig_weights etc. as kwargs"


class TestBaselinesKwargsPatched:
    """Baselines algorithm functions (imported by baselines evaluate.py)."""

    @pytest.mark.parametrize("relpath,func_name", [
        ("baselines/EvoEdit/EvoEdit/EvoEdit_main.py", "apply_EvoEdit_to_model"),
        ("baselines/EvoEdit/nse/nse_main.py", "apply_nse_to_model"),
        ("baselines/EvoEdit/memit/memit_main.py", "apply_memit_to_model"),
        ("baselines/EvoEdit/memit/memit_seq_main.py", "apply_memit_seq_to_model"),
        ("baselines/EvoEdit/memit/memit_rect_main.py", "apply_memit_rect_to_model"),
        ("baselines/EvoEdit/memit/memit_seq_rect_main.py", "apply_memit_seq_rect_to_model"),
        ("baselines/EvoEdit/AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
    ])
    def test_baselines_has_kwargs(self, relpath, func_name):
        path = PROJECT_ROOT / relpath
        if not path.exists():
            pytest.skip(f"{relpath} not found")
        source = path.read_text()
        func_start = source.find(f"def {func_name}(")
        assert func_start > 0, f"{func_name} not found in {relpath}"
        sig_end = source.find(")", func_start) + 1
        sig = source[func_start:sig_end]
        assert "**_kwargs" in sig or "**kwargs" in sig, \
            f"{relpath}:{func_name} must accept **_kwargs"


# ============================================================================
# Patch targets must cover every function that runners import
# ============================================================================

class TestPatchTargetsCoverRunnerImports:
    """Every algorithm function imported by a runner must be in the patch targets."""

    def test_polykernel_runner_imports_are_patched(self):
        """polykernel_seqreg_runner imports from vendor path — all must be kwargs-patched."""
        runner = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        patch = (PROJECT_ROOT / "scripts" / "patches" / "patch_kwargs.py").read_text()

        # Functions imported by the runner
        imports = {
            "apply_nse_to_model": "vendor",
            "apply_memit_seq_rect_to_model": "baselines",
        }

        for func, scope in imports.items():
            if func in runner:
                targets_section = "VENDOR_TARGETS" if scope == "vendor" else "BASELINES_TARGETS"
                assert func in patch, \
                    f"{func} is imported by polykernel_seqreg_runner but not in {targets_section}"

    def test_checkpoint_runner_imports_are_patched(self):
        """checkpoint_runner imports from vendor path."""
        runner = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        patch = (PROJECT_ROOT / "scripts" / "patches" / "patch_kwargs.py").read_text()

        for func in ["apply_AlphaEdit_to_model", "apply_memit_to_model"]:
            if func in runner:
                assert func in patch, \
                    f"{func} is imported by checkpoint_runner but not in patch_kwargs VENDOR_TARGETS"


# ============================================================================
# Baselines evaluate.py must have all algorithms after patching
# ============================================================================

class TestBaselinesEvaluateCompleteness:
    """After patching, baselines evaluate.py must support all algorithms."""

    def test_memit_seq_rect_in_alg_dict(self):
        source = (PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py").read_text()
        assert '"MEMIT_seq_rect"' in source, \
            "baselines evaluate.py must have MEMIT_seq_rect in ALG_DICT after patching"

    def test_memit_seq_rect_in_choices(self):
        source = (PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py").read_text()
        # Find the argparse choices line
        choices_start = source.find("choices=[")
        assert choices_start > 0, "argparse choices not found"
        choices_line = source[choices_start:source.find("]", choices_start) + 1]
        assert "MEMIT_seq_rect" in choices_line, \
            f"MEMIT_seq_rect must be in argparse choices. Got: {choices_line}"

    def test_memit_seq_rect_import(self):
        source = (PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py").read_text()
        assert "from memit.memit_seq_rect_main import apply_memit_seq_rect_to_model" in source, \
            "baselines evaluate.py must import apply_memit_seq_rect_to_model"

    def test_error_cache_for_rect(self):
        source = (PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py").read_text()
        assert "error_cache" in source, \
            "baselines evaluate.py must handle error_cache for rect algorithms"


# ============================================================================
# Upstream evaluate.py reference is bundled for cluster reset
# ============================================================================

class TestUpstreamBundled:
    """The upstream baselines evaluate.py must be bundled for cluster reset."""

    def test_upstream_copy_exists(self):
        path = PROJECT_ROOT / "scripts" / "patches" / "baselines_evaluate_upstream.py"
        assert path.exists(), \
            "scripts/patches/baselines_evaluate_upstream.py must exist — " \
            "used by YAML to reset baselines before patching"

    def test_upstream_copy_has_no_patches(self):
        source = (PROJECT_ROOT / "scripts" / "patches" / "baselines_evaluate_upstream.py").read_text()
        assert "MEMIT_seq_rect" not in source, \
            "Upstream copy must be clean — no MEMIT_seq_rect (that's added by patches)"
        assert "_kwargs" not in source, \
            "Upstream copy must be clean — no **_kwargs (that's added by patches)"
        assert "SKIP_MEGA_BATCH_EVAL" not in source, \
            "Upstream copy must be clean — no SKIP_MEGA_BATCH_EVAL (that's added by patches)"

    def test_yaml_copies_upstream_before_patching(self):
        yaml = (PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml").read_text()
        assert "baselines_evaluate_upstream.py" in yaml, \
            "YAML must copy upstream evaluate.py before apply_all.py"
