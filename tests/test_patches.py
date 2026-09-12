#!/usr/bin/env python3
"""Tests for the unified patch system (scripts/patches/).

Every patch must:
  1. Be a single Python file with a clear name
  2. Have an apply() function
  3. Be idempotent (applying twice = applying once)
  4. Print what it did
  5. Return the number of changes made

Run with: uv run pytest tests/test_patches.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"
PATCHES_DIR = PROJECT_ROOT / "scripts" / "patches"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
sys.path.insert(0, str(PATCHES_DIR))


class TestPatchStructure:
    """Every patch file must follow the standard pattern."""

    PATCH_FILES = [
        "patch_kwargs.py",
        "patch_canonical_name.py",
        "patch_nan_guard.py",
        "patch_p_cache.py",
        "patch_glue_map.py",
        "patch_model_compat.py",
        "patch_mega_batch_eval.py",
        "patch_s3_checkpoint.py",
        "apply_all.py",
    ]

    @pytest.mark.parametrize("filename", PATCH_FILES)
    def test_patch_file_exists(self, filename):
        assert (PATCHES_DIR / filename).exists(), f"Missing patch file: {filename}"

    @pytest.mark.parametrize("filename", [f for f in PATCH_FILES if f != "apply_all.py"])
    def test_patch_has_apply_function(self, filename):
        source = (PATCHES_DIR / filename).read_text()
        assert "def apply(" in source, f"{filename} must have an apply() function"

    @pytest.mark.parametrize("filename", PATCH_FILES)
    def test_patch_has_docstring(self, filename):
        source = (PATCHES_DIR / filename).read_text()
        assert '"""' in source, f"{filename} must have a docstring explaining what it patches"

    @pytest.mark.parametrize("filename", [f for f in PATCH_FILES if f != "apply_all.py"])
    def test_patch_mentions_idempotent(self, filename):
        source = (PATCHES_DIR / filename).read_text()
        assert "idempotent" in source.lower() or "already patched" in source.lower(), (
            f"{filename} must document idempotency"
        )

    def test_apply_all_imports_all_patches(self):
        source = (PATCHES_DIR / "apply_all.py").read_text()
        for patch in self.PATCH_FILES:
            if patch == "apply_all.py":
                continue
            module = patch.replace(".py", "")
            assert module in source, f"apply_all.py must import {module}"


class TestPatchKwargs:
    """Kwargs patch must handle ALL algorithm apply functions."""

    def test_covers_vendor_memit(self):
        import patch_kwargs
        assert any("memit_main.py" in t[0] for t in patch_kwargs.VENDOR_TARGETS)

    def test_covers_vendor_alphaedit(self):
        import patch_kwargs
        assert any("AlphaEdit_main.py" in t[0] for t in patch_kwargs.VENDOR_TARGETS)

    def test_covers_baselines_evoedit(self):
        import patch_kwargs
        assert any("EvoEdit_main.py" in t[0] for t in patch_kwargs.BASELINES_TARGETS)

    def test_covers_baselines_nse(self):
        import patch_kwargs
        assert any("nse_main.py" in t[0] for t in patch_kwargs.BASELINES_TARGETS)

    def test_covers_baselines_rect(self):
        import patch_kwargs
        assert any("memit_seq_rect_main.py" in t[0] for t in patch_kwargs.BASELINES_TARGETS)

    def test_add_kwargs_is_idempotent(self):
        import patch_kwargs
        source = "def apply_test_to_model(\n    model,\n    tok,\n) -> None:"
        patched, changed = patch_kwargs._add_kwargs_to_signature(source, "apply_test_to_model")
        assert changed
        assert "**_kwargs" in patched
        patched2, changed2 = patch_kwargs._add_kwargs_to_signature(patched, "apply_test_to_model")
        assert not changed2
        assert patched == patched2


class TestPatchCanonicalName:
    """Canonical name patch must set correct values for all models."""

    def test_sets_llama_name(self):
        import patch_canonical_name
        assert "llama3-8b-instruct" in patch_canonical_name.PATCH

    def test_sets_gptj_name(self):
        import patch_canonical_name
        assert "gpt-j-6b" in patch_canonical_name.PATCH

    def test_sets_qwen_name(self):
        import patch_canonical_name
        assert "qwen2.5-7b-instruct" in patch_canonical_name.PATCH


class TestPatchMegaBatchEval:
    """Mega-batch eval patch must inject the shared module function."""

    def test_uses_shared_module(self):
        import patch_mega_batch_eval
        source = Path(patch_mega_batch_eval.__file__).read_text()
        assert "get_mega_batch_eval_source" in source

    def test_replaces_per_record_loop(self):
        import patch_mega_batch_eval
        assert patch_mega_batch_eval.EVAL_ANCHOR == "    for record in ds:"

    def test_skips_vendor_per_record_loop(self):
        """After injection, vendor per-record loop must be inside `if False:`."""
        import patch_mega_batch_eval
        # The call template must include the `if False:` guard
        source = Path(patch_mega_batch_eval.__file__).read_text()
        assert "if False:" in source


class TestPatchS3Checkpoint:
    """S3 checkpoint patch must replace save_pretrained."""

    def test_replaces_save_pretrained(self):
        import patch_s3_checkpoint
        assert "save_pretrained" in patch_s3_checkpoint.SAVE_ANCHOR
        assert "save_pretrained" not in patch_s3_checkpoint.SAVE_REPLACEMENT

    def test_saves_layer_weights(self):
        import patch_s3_checkpoint
        assert "model_weights.pt" in patch_s3_checkpoint.SAVE_REPLACEMENT

    def test_saves_cache_c(self):
        import patch_s3_checkpoint
        assert "cache_c" in patch_s3_checkpoint.SAVE_REPLACEMENT


class TestOldPatchSystemRetired:
    """Old scattered patching files must not exist — replaced by scripts/patches/."""

    def test_patch_baselines_sh_removed(self):
        """patch_baselines.sh replaced by patch_kwargs.py."""
        assert not (PROJECT_ROOT / "scripts" / "patch_baselines.sh").exists(), (
            "scripts/patch_baselines.sh should be deleted — use scripts/patches/patch_kwargs.py"
        )

    def test_patch_lightweight_checkpoint_removed(self):
        """patch_lightweight_checkpoint.py replaced by patch_s3_checkpoint.py."""
        assert not (PROJECT_ROOT / "scripts" / "patch_lightweight_checkpoint.py").exists(), (
            "scripts/patch_lightweight_checkpoint.py should be deleted — use scripts/patches/patch_s3_checkpoint.py"
        )

    def test_no_inline_sed_patches_in_yaml(self):
        """YAML files must not have inline sed patches — use apply_all.py."""
        for yaml_file in ["sky/smoke_test.yaml", "sky/alphaedit_gpu.yaml"]:
            path = PROJECT_ROOT / yaml_file
            if not path.exists():
                continue
            source = path.read_text()
            # Should not have sed -i patches for vendor code
            sed_count = source.count("sed -i")
            assert sed_count == 0, (
                f"{yaml_file} has {sed_count} inline sed patches — "
                f"all patches should go through scripts/patches/apply_all.py"
            )

    def test_yaml_uses_apply_all(self):
        """Both YAML files must call apply_all.py."""
        for yaml_file in ["sky/smoke_test.yaml", "sky/alphaedit_gpu.yaml"]:
            path = PROJECT_ROOT / yaml_file
            if not path.exists():
                continue
            source = path.read_text()
            assert "apply_all" in source, f"{yaml_file} must call scripts/patches/apply_all.py"

    def test_no_scripts_reference_deleted_files(self):
        """No active script should reference deleted patch files."""
        deleted = ["patch_baselines.sh", "patch_lightweight_checkpoint.py"]
        for script_dir in [PROJECT_ROOT / "scripts", PROJECT_ROOT / "sky"]:
            for f in script_dir.glob("*"):
                if f.is_file() and f.suffix in (".sh", ".py", ".yaml"):
                    source = f.read_text()
                    for d in deleted:
                        if d in source and "should be deleted" not in source and "replaced by" not in source:
                            assert False, f"{f.name} still references deleted {d}"


class TestApplyAllIntegration:
    """apply_all must run without errors on the actual codebase."""

    def test_apply_all_imports(self):
        """All patch modules must be importable."""
        import apply_all
        assert hasattr(apply_all, 'apply_all')

    def test_no_baseline_scripts_have_inline_mega_batch(self):
        """After restructuring, baseline scripts must NOT have inline mega_batch injection."""
        for script in ["run_evoedit_baseline.sh", "run_nse_baseline.sh", "run_rect_aligned_paper_replication.sh"]:
            path = PROJECT_ROOT / "scripts" / script
            if not path.exists():
                continue
            source = path.read_text()
            assert "get_mega_batch_eval_source" not in source, (
                f"{script} still has inline mega_batch_eval injection — "
                f"should use scripts/patches/patch_mega_batch_eval.py instead"
            )
            assert "_fn_match" not in source, (
                f"{script} still has regex mega_batch extraction code"
            )
