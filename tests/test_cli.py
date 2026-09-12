#!/usr/bin/env python3
"""Tests for the unified CLI (src/__main__.py + src/method_registry.py).

Run with: uv run pytest tests/test_cli.py -v
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMethodRegistry:
    """Method registry must cover all algorithms."""

    def test_import(self):
        from method_registry import METHODS, get_method, list_methods

    def test_all_core_methods_registered(self):
        from method_registry import METHODS
        required = [
            "alphaedit", "memit", "memit-seq",
            "revive+memit", "revive+alphaedit", "revive+nse", "revive+rect",
            "pathguard", "evoedit", "nse", "rect",
        ]
        for name in required:
            assert name in METHODS, f"Method '{name}' not registered"

    def test_at_least_11_methods(self):
        from method_registry import METHODS
        assert len(METHODS) >= 11

    def test_get_method_returns_correct_type(self):
        from method_registry import get_method, MethodDef
        m = get_method("alphaedit")
        assert isinstance(m, MethodDef)
        assert m.name == "alphaedit"

    def test_get_method_unknown_raises(self):
        from method_registry import get_method
        with pytest.raises(ValueError, match="Unknown method"):
            get_method("nonexistent-method")

    def test_list_methods_returns_sorted(self):
        from method_registry import list_methods
        methods = list_methods()
        names = [m.name for m in methods]
        assert names == sorted(names)

    def test_every_method_has_runner(self):
        from method_registry import METHODS
        for name, m in METHODS.items():
            assert m.runner, f"Method '{name}' has no runner"

    def test_every_method_has_description(self):
        from method_registry import METHODS
        for name, m in METHODS.items():
            assert m.description, f"Method '{name}' has no description"

    def test_shell_methods_point_to_existing_scripts(self):
        from method_registry import METHODS
        for name, m in METHODS.items():
            if m.shell:
                script = PROJECT_ROOT / m.runner
                assert script.exists(), f"Shell method '{name}' points to missing script: {m.runner}"

    def test_python_methods_point_to_existing_runners(self):
        from method_registry import METHODS
        for name, m in METHODS.items():
            if not m.shell:
                runner = PROJECT_ROOT / "src" / m.runner
                assert runner.exists(), f"Python method '{name}' points to missing runner: {m.runner}"

    @pytest.mark.parametrize("method,expected_key,expected_val", [
        ("alphaedit", "alg_name", "AlphaEdit"),
        ("memit", "alg_name", "MEMIT"),
        ("memit-seq", "base_alg", "MEMIT"),
        ("revive+memit", "revive", True),
        ("revive+alphaedit", "base_alg", "AlphaEdit"),
        ("revive+nse", "base_alg", "NSE"),
        ("revive+rect", "base_alg", "MEMIT_rect"),
        ("pathguard", "pathguard", True),
    ])
    def test_method_defaults_correct(self, method, expected_key, expected_val):
        from method_registry import get_method
        m = get_method(method)
        assert expected_key in m.defaults, f"'{method}' missing default '{expected_key}'"
        assert m.defaults[expected_key] == expected_val


class TestCLIBuildArgs:
    """CLI must build correct runner arguments for each method."""

    def _make_cli_args(self, **overrides):
        """Create a mock CLI args namespace."""
        defaults = {
            "seed": 42, "ordering": None, "edits": None, "num_edits": None,
            "ds": None, "save_interval": None, "lambda_prev": None,
            "lambda_delta": None, "revive_tau": None, "fast_checkpoint": False,
            "eval_at_checkpoints": False, "cuda_device": None,
        }
        defaults.update(overrides)
        ns = MagicMock()
        for k, v in defaults.items():
            setattr(ns, k, v)
        return ns

    def test_alphaedit_args(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_main", str(PROJECT_ROOT / "src" / "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        build_runner_args = cli.build_runner_args
        from method_registry import get_method
        method = get_method("alphaedit")
        args = build_runner_args(method, self._make_cli_args(seed=42))
        assert "--alg_name" in args
        assert "AlphaEdit" in args
        assert "--seed" in args
        assert "42" in args

    def test_revive_memit_has_revive_flag(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_main", str(PROJECT_ROOT / "src" / "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        build_runner_args = cli.build_runner_args
        from method_registry import get_method
        method = get_method("revive+memit")
        args = build_runner_args(method, self._make_cli_args())
        assert "--revive" in args
        assert "--revive_tau" in args
        assert "0.1" in args

    def test_ordering_passed_through(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_main", str(PROJECT_ROOT / "src" / "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        build_runner_args = cli.build_runner_args
        from method_registry import get_method
        method = get_method("memit-seq")
        args = build_runner_args(method, self._make_cli_args(ordering="fb_high_exposure"))
        assert "--ordering" in args
        assert "fb_high_exposure" in args

    def test_edits_override(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_main", str(PROJECT_ROOT / "src" / "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        build_runner_args = cli.build_runner_args
        from method_registry import get_method
        method = get_method("alphaedit")
        args = build_runner_args(method, self._make_cli_args(edits=5000))
        assert "--dataset_size_limit" in args
        assert "5000" in args

    def test_pathguard_has_pathguard_flags(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("src_main", str(PROJECT_ROOT / "src" / "__main__.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        build_runner_args = cli.build_runner_args
        from method_registry import get_method
        method = get_method("pathguard")
        args = build_runner_args(method, self._make_cli_args())
        assert "--pathguard" in args
        assert "--pathguard_M" in args
        assert "200" in args
        assert "--pathguard_adaptive" in args


class TestCLIHelp:
    """CLI help commands must work without crashing."""

    def test_main_help(self):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src", "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "list-methods" in result.stdout
        assert "patch" in result.stdout

    def test_run_help(self):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src", "run", "--help"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--method" in result.stdout
        assert "--seed" in result.stdout

    def test_list_methods_runs(self):
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src", "list-methods"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "alphaedit" in result.stdout
        assert "revive+memit" in result.stdout
        assert "pathguard" in result.stdout


class TestCLIPatch:
    """Patch subcommand must dispatch to apply_all.py."""

    def test_patch_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "patches" / "apply_all.py").exists()


class TestCLISmokeTest:
    """Smoke-test subcommand must dispatch to test script."""

    def test_smoke_test_script_exists(self):
        assert (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").exists()
