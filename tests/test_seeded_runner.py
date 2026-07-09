"""
Tests for the seeded runner module.

Validates script generation, metadata recording, and environment setup.
No GPU required — tests script string generation and path operations.
"""

import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from seeded_runner import build_runner_script, get_project_root, get_alphaedit_root


class TestBuildRunnerScript:
    """Tests for the runner script generator."""

    def _build_default(self, **kwargs):
        defaults = dict(
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
        defaults.update(kwargs)
        return build_runner_script(**defaults)

    def test_valid_python(self):
        """Generated script should be syntactically valid Python."""
        script = self._build_default()
        ast.parse(script)

    def test_seed_setting(self):
        """Script should set the correct seed."""
        script = self._build_default(seed=2024)
        assert "seed = 2024" in script

    def test_random_seeding(self):
        """Script should seed all RNG sources."""
        script = self._build_default()
        assert "random.seed(seed)" in script
        assert "np.random.seed(seed)" in script
        assert "torch.manual_seed(seed)" in script
        assert "torch.cuda.manual_seed_all(seed)" in script

    def test_deterministic_flags(self):
        """Script should set all deterministic flags."""
        script = self._build_default()
        assert "torch.backends.cudnn.deterministic = True" in script
        assert "torch.backends.cudnn.benchmark = False" in script
        assert "torch.use_deterministic_algorithms(True" in script

    def test_argv_contains_alg_name(self):
        """sys.argv should include the algorithm name."""
        script = self._build_default(alg_name="MEMIT")
        assert "--alg_name=MEMIT" in script

    def test_argv_contains_model_name(self):
        """sys.argv should include the model name."""
        script = self._build_default(model_name="mistralai/Mistral-7B-Instruct-v0.3")
        assert "--model_name=mistralai/Mistral-7B-Instruct-v0.3" in script

    def test_argv_contains_dataset(self):
        """sys.argv should include dataset parameters."""
        script = self._build_default(ds_name="zsre", dataset_size_limit=500)
        assert "--ds_name=zsre" in script
        assert "--dataset_size_limit=500" in script

    def test_skip_generation_tests_flag(self):
        """--skip_generation_tests should only appear when True."""
        script_on = self._build_default(skip_generation_tests=True)
        assert "--skip_generation_tests" in script_on

        script_off = self._build_default(skip_generation_tests=False)
        assert "--skip_generation_tests" not in script_off

    def test_conserve_memory_flag(self):
        """--conserve_memory should only appear when True."""
        script_on = self._build_default(conserve_memory=True)
        assert "--conserve_memory" in script_on

        script_off = self._build_default(conserve_memory=False)
        assert "--conserve_memory" not in script_off

    def test_use_cache_flag(self):
        """--use_cache should only appear when True."""
        script_on = self._build_default(use_cache=True)
        assert "--use_cache" in script_on

        script_off = self._build_default(use_cache=False)
        assert "--use_cache" not in script_off

    def test_cuda_patch(self):
        """Script should patch the hardcoded CUDA device line."""
        script = self._build_default()
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in script
        assert "seeded_runner" in script

    def test_exec_as_main(self):
        """Script should execute evaluate.py as __main__."""
        script = self._build_default()
        assert '"__name__": "__main__"' in script

    def test_different_seeds_produce_different_scripts(self):
        """Different seeds should produce different scripts."""
        s1 = self._build_default(seed=42)
        s2 = self._build_default(seed=99)
        assert s1 != s2

    def test_all_algorithms_valid(self):
        """Should generate valid scripts for all supported algorithms."""
        for alg in ["AlphaEdit", "MEMIT", "ROME"]:
            script = self._build_default(alg_name=alg)
            ast.parse(script)
            assert f"--alg_name={alg}" in script


class TestPathResolution:
    """Tests for path resolution helpers."""

    def test_project_root_exists(self):
        """get_project_root() should return an existing directory."""
        root = get_project_root()
        assert root.exists()
        assert root.is_dir()

    def test_project_root_has_src(self):
        """Project root should contain the src/ directory."""
        root = get_project_root()
        assert (root / "src").exists()

    def test_alphaedit_root_path(self):
        """get_alphaedit_root() should point to vendor/AlphaEdit."""
        ae_root = get_alphaedit_root()
        assert ae_root.name == "AlphaEdit"
        assert ae_root.parent.name == "vendor"
