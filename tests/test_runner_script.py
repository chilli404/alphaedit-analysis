"""
Unit tests for the seeded runner script generation.

These tests validate that:
1. The generated runner script is syntactically valid Python
2. The CUDA_VISIBLE_DEVICES patch target is present in the upstream code
3. The argument list is correctly formatted
4. The determinism environment variables are set correctly

These tests do NOT require GPU access and can run in CI.
"""

import ast
import sys
from pathlib import Path

import pytest

# Add the src/ directory to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seeded_runner import build_runner_script, get_alphaedit_root


class TestBuildRunnerScript:
    """Test that the generated runner script is valid and correct."""

    def test_generates_valid_python(self):
        """The generated script must be syntactically valid Python."""
        script = build_runner_script(
            seed=42,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=2000,
            num_edits=100,
            downstream_eval_steps=5,
            skip_generation_tests=False,
            generation_test_interval=1,
            conserve_memory=True,
            use_cache=False,
        )
        # Should parse without raising SyntaxError
        ast.parse(script)

    def test_contains_seed_setting(self):
        """The script must set all required random seeds."""
        script = build_runner_script(
            seed=137,
            cuda_device="0",
            alg_name="MEMIT",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="zsre",
            dataset_size_limit=500,
            num_edits=50,
            downstream_eval_steps=0,
            skip_generation_tests=True,
            generation_test_interval=1,
            conserve_memory=True,
            use_cache=False,
        )
        assert "seed = 137" in script
        assert "random.seed(seed)" in script
        assert "np.random.seed(seed)" in script
        assert "torch.manual_seed(seed)" in script
        assert "torch.cuda.manual_seed_all(seed)" in script
        assert "torch.backends.cudnn.deterministic = True" in script
        assert "torch.backends.cudnn.benchmark = False" in script
        assert "torch.use_deterministic_algorithms(True" in script

    def test_contains_cuda_patch(self):
        """The script must patch the CUDA_VISIBLE_DEVICES hardcode."""
        script = build_runner_script(
            seed=42,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=100,
            num_edits=10,
            downstream_eval_steps=0,
            skip_generation_tests=True,
            generation_test_interval=1,
            conserve_memory=True,
            use_cache=False,
        )
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in script
        assert "CUDA_VISIBLE_DEVICES managed by seeded_runner" in script

    def test_argv_contains_all_args(self):
        """The sys.argv in the script must include all required arguments."""
        script = build_runner_script(
            seed=42,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=2000,
            num_edits=100,
            downstream_eval_steps=5,
            skip_generation_tests=False,
            generation_test_interval=1,
            conserve_memory=True,
            use_cache=False,
        )
        assert "--alg_name=AlphaEdit" in script
        assert "--ds_name=mcf" in script
        assert "--dataset_size_limit=2000" in script
        assert "--num_edits=100" in script
        assert "--downstream_eval_steps=5" in script
        assert "--conserve_memory" in script
        # use_cache=False means the flag should NOT appear
        assert "--use_cache" not in script

    def test_skip_generation_tests_flag(self):
        """The skip_generation_tests flag should only appear when True."""
        script_with = build_runner_script(
            seed=42, cuda_device="0", alg_name="AlphaEdit",
            model_name="m", hparams_fname="h.json", ds_name="mcf",
            dataset_size_limit=10, num_edits=1, downstream_eval_steps=0,
            skip_generation_tests=True, generation_test_interval=1,
            conserve_memory=False, use_cache=False,
        )
        script_without = build_runner_script(
            seed=42, cuda_device="0", alg_name="AlphaEdit",
            model_name="m", hparams_fname="h.json", ds_name="mcf",
            dataset_size_limit=10, num_edits=1, downstream_eval_steps=0,
            skip_generation_tests=False, generation_test_interval=1,
            conserve_memory=False, use_cache=False,
        )
        assert "--skip_generation_tests" in script_with
        assert "--skip_generation_tests" not in script_without

    def test_executes_as_main(self):
        """The script must execute evaluate.py with __name__ == '__main__'."""
        script = build_runner_script(
            seed=42, cuda_device="0", alg_name="AlphaEdit",
            model_name="m", hparams_fname="h.json", ds_name="mcf",
            dataset_size_limit=10, num_edits=1, downstream_eval_steps=0,
            skip_generation_tests=True, generation_test_interval=1,
            conserve_memory=True, use_cache=False,
        )
        assert '"__name__": "__main__"' in script


class TestUpstreamPatchTarget:
    """Verify the upstream evaluate.py still has the expected patch target."""

    def test_evaluate_py_has_cuda_line(self):
        """evaluate.py must contain the CUDA line we intend to patch."""
        alphaedit_root = get_alphaedit_root()
        eval_path = alphaedit_root / "experiments" / "evaluate.py"

        if not eval_path.exists():
            pytest.skip("AlphaEdit submodule not initialized")

        source = eval_path.read_text()
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in source, (
            "Upstream evaluate.py no longer contains the expected CUDA_VISIBLE_DEVICES line. "
            "The seeded_runner.py patch logic needs updating."
        )

    def test_evaluate_py_has_args_reference(self):
        """evaluate.py must reference args at module level (our exec workaround target)."""
        alphaedit_root = get_alphaedit_root()
        eval_path = alphaedit_root / "experiments" / "evaluate.py"

        if not eval_path.exists():
            pytest.skip("AlphaEdit submodule not initialized")

        source = eval_path.read_text()
        assert "args.downstream_eval_steps" in source, (
            "Upstream evaluate.py no longer references args.downstream_eval_steps. "
            "The exec workaround may no longer be necessary."
        )


class TestROMEHparamsExist:
    """Verify ROME hparams exist for the calibration baseline."""

    def test_rome_llama3_hparams_exist(self):
        """ROME hparams for Llama-3-8B must exist."""
        alphaedit_root = get_alphaedit_root()
        hparams_path = alphaedit_root / "hparams" / "ROME" / "Llama3-8B.json"

        if not alphaedit_root.exists():
            pytest.skip("AlphaEdit submodule not initialized")

        assert hparams_path.exists(), (
            f"ROME hparams not found at {hparams_path}. "
            "Cannot run calibration baseline."
        )
