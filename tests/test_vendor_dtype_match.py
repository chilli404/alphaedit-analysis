#!/usr/bin/env python3
"""Verify ALL model loading matches the vendor: float32, no torch_dtype arg.

The rule is simple:
  - Vendor code: from_pretrained(model_name).cuda() → float32
  - Our code: must do the same — NO torch_dtype arg (except Qwen NaN workaround)
  - No .half() anywhere on checkpoint weights
  - Published paper numbers were ALL produced with float32

Any script that passes torch_dtype (other than Qwen float32) will produce
different results from published baselines.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _source(relpath: str) -> str:
    return (PROJECT_ROOT / relpath).read_text()


def _code_lines(source: str):
    """Yield non-comment, non-empty lines."""
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped


# ============================================================================
# Core rule: no torch_dtype in model loading (except Qwen)
# ============================================================================

class TestHarnessFloat32:
    """evaluate_harness loads models for AlphaEdit, MEMIT, REVIVE, etc."""

    def test_no_torch_dtype_in_else_branch(self):
        source = _source("src/evaluate_harness.py")
        assert '"auto"' not in source, "Harness must not use torch_dtype='auto'"
        assert "torch.float16" not in source.split("def load_model_and_tok")[1].split("def ")[0], \
            "Harness must not use float16"

    def test_no_dtype_reaches_from_pretrained(self):
        """When dtype=None, no torch_dtype should be passed."""
        import transformers
        from unittest.mock import patch
        captured = {}

        def spy(*a, **kw):
            captured.update(kw)
            raise RuntimeError("spy")

        from evaluate_harness import load_model_and_tok
        with patch.object(transformers.AutoModelForCausalLM, "from_pretrained", spy):
            with pytest.raises(RuntimeError):
                load_model_and_tok("test", device="cpu", dtype=None)

        assert "torch_dtype" not in captured, \
            f"No torch_dtype should be passed when dtype=None, got: {captured}"

    def test_explicit_dtype_passes_through(self):
        import torch, transformers
        from unittest.mock import patch
        captured = {}

        def spy(*a, **kw):
            captured.update(kw)
            raise RuntimeError("spy")

        from evaluate_harness import load_model_and_tok
        with patch.object(transformers.AutoModelForCausalLM, "from_pretrained", spy):
            with pytest.raises(RuntimeError):
                load_model_and_tok("test", device="cpu", dtype=torch.float32)

        assert captured.get("torch_dtype") == torch.float32

    def test_prints_model_dtype(self):
        source = _source("src/evaluate_harness.py")
        assert "Model dtype" in source, "Harness must print Model dtype for verification"


class TestEvalScriptsFloat32:
    """Eval scripts must also match vendor: no torch_dtype."""

    @pytest.mark.parametrize("script", [
        "scripts/eval_matched_ordering.py",
        "scripts/eval_prob_preference.py",
    ])
    def test_no_torch_dtype_arg(self, script):
        source = _source(script)
        # Find from_pretrained calls
        for line in _code_lines(source):
            if "AutoModelForCausalLM.from_pretrained" in line or \
               (line.strip().startswith("model_path") and "from_pretrained" in source[source.find(line)-200:source.find(line)+200]):
                pass
        # Check the multi-line from_pretrained call context
        idx = source.find("AutoModelForCausalLM.from_pretrained")
        context = source[idx:idx+150]
        assert "torch_dtype" not in context, \
            f"{script}: from_pretrained must not pass torch_dtype — vendor uses float32"

    @pytest.mark.parametrize("script", [
        "scripts/eval_matched_ordering.py",
        "scripts/eval_prob_preference.py",
    ])
    def test_no_half(self, script):
        source = _source(script)
        assert ".half()" not in source, f"{script}: no .half() — use .to(param.dtype)"

    @pytest.mark.parametrize("script", [
        "scripts/eval_matched_ordering.py",
        "scripts/eval_prob_preference.py",
    ])
    def test_checkpoint_uses_param_dtype(self, script):
        source = _source(script)
        if "param_dict" in source and ".data.copy_" in source:
            assert "param_dict[name].dtype" in source, \
                f"{script}: checkpoint weights must cast to param dtype"


class TestRunnerFloat32:
    """Runners must not pass dtype to load_model_and_tok."""

    def test_checkpoint_runner(self):
        source = _source("src/runners/checkpoint_runner.py")
        assert "torch.float32" not in source
        assert "torch.float16" not in source

    def test_polykernel_runner(self):
        source = _source("src/polykernel/polykernel_seqreg_runner.py")
        call = source[source.find("load_model_and_tok("):][:200]
        assert "dtype=" not in call


class TestVendorPatchFloat32:
    """The vendor model-loading patch must preserve float32 for non-Qwen."""

    def test_non_qwen_gets_none(self):
        source = _source("src/util/source_patches.py")
        idx = source.find("MODEL_LOAD_FP32")
        block = source[idx:idx+300]
        assert "else None" in block, "Non-Qwen models must get torch_dtype=None (→ float32)"
        assert '"auto"' not in block, "Must not use 'auto' — vendor uses None"

    def test_prints_dtype(self):
        source = _source("src/util/source_patches.py")
        idx = source.find("MODEL_LOAD_FP32")
        block = source[idx:idx+400]
        assert "Model dtype" in block, "Patch must print loaded dtype"


class TestNoHalfAnywhere:
    """No .half() on checkpoint weights in the entire codebase."""

    def test_no_half_in_src(self):
        for f in (PROJECT_ROOT / "src").rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            source = f.read_text()
            for i, line in enumerate(source.split("\n")):
                if ".half()" in line and "param_dict" in line:
                    pytest.fail(f"{f.relative_to(PROJECT_ROOT)}:{i+1}: .half() on weights")

    def test_no_half_in_scripts(self):
        for f in (PROJECT_ROOT / "scripts").rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            source = f.read_text()
            for i, line in enumerate(source.split("\n")):
                if ".half()" in line and ("param_dict" in line or "model_weights" in line):
                    pytest.fail(f"{f.relative_to(PROJECT_ROOT)}:{i+1}: .half() on weights")


class TestNoFloat16InModelLoading:
    """No float16 anywhere in model loading code."""

    @pytest.mark.parametrize("script", [
        "scripts/run_revive_paper_replication.sh",
        "scripts/run_nse_paper_replication.sh",
        "scripts/run_rect_aligned_paper_replication.sh",
        "scripts/run_evoedit_paper_replication.sh",
        "scripts/run_failure_curve_checkpointed.sh",
    ])
    def test_no_float16_in_paper_scripts(self, script):
        path = PROJECT_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        source = path.read_text()
        for line in _code_lines(source):
            assert "torch.float16" not in line, f"{script}: {line}"
            assert "torch_dtype=torch.float16" not in line, f"{script}: {line}"

    def test_nse_cache_no_float16(self):
        path = PROJECT_ROOT / "scripts" / "build_nse_cache.sh"
        if not path.exists():
            pytest.skip()
        source = path.read_text()
        for line in _code_lines(source):
            assert "torch.float16" not in line, f"build_nse_cache.sh: {line}"


# ============================================================================
# GPU: verify actual loaded dtype
# ============================================================================

requires_gpu = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="No GPU"
)


@requires_gpu
class TestGPUDtype:

    def test_harness_loads_float32(self):
        import torch
        from evaluate_harness import load_model_and_tok
        model, tok = load_model_and_tok("meta-llama/Meta-Llama-3-8B-Instruct", dtype=None)
        param = next(model.parameters())
        assert param.dtype == torch.float32, f"Expected float32, got {param.dtype}"
        del model
        torch.cuda.empty_cache()
