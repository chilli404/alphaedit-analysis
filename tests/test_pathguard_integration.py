"""
Integration tests for the PathGuard runner.

Tests the source injection, CLI parsing, variant naming, and checkpoint
format — all without requiring a GPU.

Run with: uv run pytest tests/test_pathguard_integration.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestPathGuardImports:
    """Verify the module can be imported and key functions exist."""

    def test_import_runner(self):
        from runners.pathguard_runner import build_pathguard_script, resolve_pathguard_checkpoint_dir

    def test_import_solver(self):
        from mechanism.pathguard_solver import select_vulnerable, woodbury_solve, compute_displacement, adapt_epsilon

    def test_import_margin_shield(self):
        from mechanism.margin_shield import compute_margin_gradient, check_margin_violations, apply_dual_correction


class TestSourceInjection:
    """Verify the generated script is valid Python."""

    def test_script_compiles(self):
        """The generated PathGuard script must be valid Python (no SyntaxError)."""
        from runners.pathguard_runner import build_pathguard_script

        script = build_pathguard_script(
            seed=42,
            cuda_device="0",
            alg_name="MEMIT",
            model_name="test-model",
            hparams_fname="test.json",
            ds_name="mcf",
            dataset_size_limit=200,
            num_edits=100,
            downstream_eval_steps=5,
            conserve_memory=True,
            lambda_prev=1.0,
            lambda_delta=0.0,
            cache_strategy="all",
            cache_max=None,
            output_jsonl="/tmp/test.jsonl",
            fast_checkpoint=False,
            eval_at_checkpoints_only=True,
            save_interval=10,
            checkpoint_dir="/tmp/ckpt",
            start_from_batch=0,
            variant_name="PathGuard-ED-M200",
            eval_results_dir="/tmp/results",
            # PathGuard-specific
            pathguard_M=200,
            pathguard_adaptive=True,
            pathguard_fixed_lambda=None,
            pathguard_random=False,
            pathguard_keys_dir="/tmp/keys",
            pathguard_representative_layer=6,
            pathguard_q=10,
            pathguard_no_margin_shield=False,
            pathguard_epsilon_init=1.0,
            pathguard_lambda_candidates="0.01,0.1,1.0,10.0",
        )

        assert isinstance(script, str)
        assert len(script) > 100

        # Must compile without SyntaxError
        try:
            compile(script, "<pathguard_test>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated script has SyntaxError at line {e.lineno}: {e.msg}")

    def test_script_contains_pathguard_markers(self):
        """Script should contain PathGuard-specific injection markers."""
        from runners.pathguard_runner import build_pathguard_script

        script = build_pathguard_script(
            seed=42, cuda_device="0", alg_name="MEMIT",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=5, conserve_memory=True,
            lambda_prev=1.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            output_jsonl="/tmp/test.jsonl",
            fast_checkpoint=False, eval_at_checkpoints_only=True,
            save_interval=10, checkpoint_dir="/tmp/ckpt",
            start_from_batch=0, variant_name="PathGuard-ED-M200",
            eval_results_dir="/tmp/results",
            pathguard_M=200, pathguard_adaptive=True,
            pathguard_fixed_lambda=None, pathguard_random=False,
            pathguard_keys_dir="/tmp/keys",
            pathguard_representative_layer=6, pathguard_q=10,
            pathguard_no_margin_shield=False,
            pathguard_epsilon_init=1.0,
            pathguard_lambda_candidates="0.01,0.1,1.0,10.0",
        )

        assert "_pg_" in script, "Script should contain PathGuard state variables"
        assert "PathGuard" in script, "Script should mention PathGuard"
        assert "woodbury" in script.lower() or "select_vulnerable" in script or "_pg_selected" in script, \
            "Script should contain Woodbury/selection logic"

    def test_memit_seq_anchors_still_present(self):
        """PathGuard should preserve MEMIT-Seq's K_prev regularization."""
        from runners.pathguard_runner import build_pathguard_script

        script = build_pathguard_script(
            seed=42, cuda_device="0", alg_name="MEMIT",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=5, conserve_memory=True,
            lambda_prev=1.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            output_jsonl="/tmp/test.jsonl",
            fast_checkpoint=False, eval_at_checkpoints_only=True,
            save_interval=10, checkpoint_dir="/tmp/ckpt",
            start_from_batch=0, variant_name="PathGuard-ED-M200",
            eval_results_dir="/tmp/results",
            pathguard_M=200, pathguard_adaptive=True,
            pathguard_fixed_lambda=None, pathguard_random=False,
            pathguard_keys_dir="/tmp/keys",
            pathguard_representative_layer=6, pathguard_q=10,
            pathguard_no_margin_shield=False,
            pathguard_epsilon_init=1.0,
            pathguard_lambda_candidates="0.01,0.1,1.0,10.0",
        )

        assert "_memit_prev_cache" in script, "Must preserve MEMIT-Seq K_prev cache"
        assert "_memit_lambda_prev" in script, "Must preserve MEMIT-Seq lambda_prev"
        assert "_memit_batch_idx" in script, "Must preserve MEMIT-Seq batch counter"


class TestVariantNaming:
    """Test PathGuard variant directory naming."""

    def test_checkpoint_dir_ed(self):
        from runners.pathguard_runner import resolve_pathguard_checkpoint_dir

        d = resolve_pathguard_checkpoint_dir(
            explicit_dir=None, seed=42,
            variant_name="PathGuard-ED-M200-e0.1",
        )
        assert "PathGuard-ED-M200-e0.1" in str(d)
        assert "seed42" in str(d)

    def test_checkpoint_dir_explicit(self):
        from runners.pathguard_runner import resolve_pathguard_checkpoint_dir

        d = resolve_pathguard_checkpoint_dir(
            explicit_dir="/my/custom/path", seed=42,
            variant_name="PathGuard-ED-M200",
        )
        assert str(d) == "/my/custom/path"

    def test_checkpoint_dir_with_ordering(self):
        from runners.pathguard_runner import resolve_pathguard_checkpoint_dir

        d = resolve_pathguard_checkpoint_dir(
            explicit_dir=None, seed=42,
            variant_name="PathGuard-ED-M200",
            ordering="fb_high_exposure",
        )
        assert "fb_high_exposure" in str(d)
        assert "matched_ordering" in str(d)


class TestCLIArgs:
    """Test argument parsing combinations."""

    def test_pathguard_ed_flags(self):
        from runners.pathguard_runner import main_parser

        args = main_parser().parse_args([
            "--seed", "42",
            "--pathguard",
            "--pathguard_adaptive",
            "--pathguard_M", "200",
        ])
        assert args.pathguard is True
        assert args.pathguard_adaptive is True
        assert args.pathguard_M == 200

    def test_pathguard_e_flags(self):
        from runners.pathguard_runner import main_parser

        args = main_parser().parse_args([
            "--seed", "42",
            "--pathguard",
            "--pathguard_fixed_lambda", "2.0",
        ])
        assert args.pathguard is True
        assert args.pathguard_fixed_lambda == 2.0
        assert args.pathguard_adaptive is False

    def test_pathguard_eds_flags(self):
        from runners.pathguard_runner import main_parser

        args = main_parser().parse_args([
            "--seed", "42",
            "--pathguard",
            "--pathguard_adaptive",
            "--pathguard_q", "15",
        ])
        assert args.pathguard is True
        assert args.pathguard_q == 15
        assert args.pathguard_no_margin_shield is False

    def test_pathguard_random_control(self):
        from runners.pathguard_runner import main_parser

        args = main_parser().parse_args([
            "--seed", "42",
            "--pathguard",
            "--pathguard_adaptive",
            "--pathguard_random",
        ])
        assert args.pathguard_random is True

    def test_default_lambda_candidates(self):
        from runners.pathguard_runner import main_parser

        args = main_parser().parse_args(["--seed", "42", "--pathguard"])
        assert "0.01" in args.pathguard_lambda_candidates
