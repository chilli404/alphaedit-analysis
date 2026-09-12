#!/usr/bin/env python3
"""
Tests for Phase 1: Shared infrastructure modules.

These tests define the expected API for checkpoint_io, mega_batch_eval,
and source_injection BEFORE implementation (test-driven development).

Run with: uv run pytest tests/test_shared_infrastructure.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


# ===========================================================================
# 1. checkpoint_io: save, load, should_save, should_skip, validate
# ===========================================================================

class TestCheckpointIO:
    """Tests for src/util/checkpoint_io.py."""

    def test_import(self):
        from util.checkpoint_io import (
            save_checkpoint, load_checkpoint, should_save, should_skip,
            validate_checkpoint_path, find_latest_checkpoint,
        )

    def test_should_skip(self):
        from util.checkpoint_io import should_skip
        assert should_skip(0, start_batch=5) is True
        assert should_skip(4, start_batch=5) is True
        assert should_skip(5, start_batch=5) is False
        assert should_skip(10, start_batch=5) is False
        assert should_skip(0, start_batch=0) is False

    def test_should_save(self):
        from util.checkpoint_io import should_save
        # save_interval=10 means save at batch 9, 19, 29, ...
        assert should_save(9, save_interval=10) is True
        assert should_save(19, save_interval=10) is True
        assert should_save(0, save_interval=10) is False
        assert should_save(5, save_interval=10) is False
        assert should_save(99, save_interval=10) is True

    def test_validate_checkpoint_path_correct(self):
        from util.checkpoint_io import validate_checkpoint_path
        # Should not raise
        validate_checkpoint_path(
            "/s3/polykernel_seqreg/AlphaEdit-poly1-REVIVE/fb_high/seed42",
            base_alg="AlphaEdit"
        )
        validate_checkpoint_path(
            "/s3/polykernel_seqreg/MEMIT-Seq-poly1-REVIVE/fb_high/seed42",
            base_alg="MEMIT"
        )

    def test_validate_checkpoint_path_mismatch(self):
        from util.checkpoint_io import validate_checkpoint_path
        with pytest.raises(RuntimeError, match="mismatch"):
            validate_checkpoint_path(
                "/s3/polykernel_seqreg/MEMIT-Seq-poly1-REVIVE/fb_high/seed42",
                base_alg="AlphaEdit"
            )

    @pytest.mark.parametrize("base_alg,expected_prefix", [
        ("MEMIT", "MEMIT-Seq"),
        ("AlphaEdit", "AlphaEdit"),
        ("NSE", "NSE"),
        ("MEMIT_rect", "MEMIT_rect"),
    ])
    def test_validate_all_base_algs(self, base_alg, expected_prefix):
        from util.checkpoint_io import validate_checkpoint_path
        good_path = f"/s3/ckpt/{expected_prefix}-poly1-lp0.0/seed42"
        validate_checkpoint_path(good_path, base_alg=base_alg)

        bad_path = "/s3/ckpt/WRONG-PREFIX-poly1-lp0.0/seed42"
        with pytest.raises(RuntimeError):
            validate_checkpoint_path(bad_path, base_alg=base_alg)

    def test_save_checkpoint_creates_files(self, tmp_path):
        from util.checkpoint_io import save_checkpoint
        import torch

        ckpt_dir = tmp_path / "ckpt"
        # Create mock model and hparams
        model = MagicMock()
        param = torch.randn(10, 10)
        model.named_parameters.return_value = [
            ("model.layers.4.mlp.down_proj.weight", param),
            ("model.layers.5.mlp.down_proj.weight", param),
        ]

        hparams = MagicMock()
        hparams.layers = [4, 5]
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        save_checkpoint(
            batch_idx=9, model=model, hparams=hparams,
            ckpt_dir=str(ckpt_dir), num_edits=100,
            metadata={"lambda_prev": 1.0, "test": True},
        )

        batch_dir = ckpt_dir / "batch_9"
        assert batch_dir.exists()
        assert (batch_dir / "model_weights.pt").exists()
        assert (batch_dir / "metadata.json").exists()

        meta = json.loads((batch_dir / "metadata.json").read_text())
        assert meta["batch_idx"] == 9
        assert meta["total_edits"] == 1000
        assert meta["lambda_prev"] == 1.0

    def test_save_checkpoint_with_extra_state(self, tmp_path):
        from util.checkpoint_io import save_checkpoint
        import torch

        ckpt_dir = tmp_path / "ckpt"
        model = MagicMock()
        model.named_parameters.return_value = []
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        extra = {"prev_cache.pt": {"layer_4": [torch.randn(10, 5)]}}
        save_checkpoint(
            batch_idx=19, model=model, hparams=hparams,
            ckpt_dir=str(ckpt_dir), num_edits=100,
            extra_state=extra,
        )

        batch_dir = ckpt_dir / "batch_19"
        assert (batch_dir / "prev_cache.pt").exists()

    def test_load_checkpoint_restores_weights(self, tmp_path):
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        import torch

        ckpt_dir = tmp_path / "ckpt"
        original_data = torch.randn(10, 10)

        # Save with original data
        save_model = MagicMock()
        save_model.named_parameters.return_value = [
            ("model.layers.4.mlp.down_proj.weight", MagicMock(data=original_data)),
        ]
        hparams = MagicMock()
        hparams.layers = [4]
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"
        save_checkpoint(9, save_model, hparams, str(ckpt_dir), 100)

        # Load into a model with a real parameter
        load_param = torch.nn.Parameter(torch.zeros(10, 10))
        load_model = MagicMock()
        load_model.named_parameters.return_value = [
            ("model.layers.4.mlp.down_proj.weight", load_param),
        ]
        result = load_checkpoint(load_model, hparams, str(ckpt_dir), batch_idx=9, device="cpu")
        assert result["loaded"] is True
        assert torch.allclose(load_param.data, original_data)

    def test_find_latest_checkpoint(self, tmp_path):
        from util.checkpoint_io import find_latest_checkpoint

        ckpt_dir = tmp_path / "ckpt"
        # Create batch_9 and batch_19
        for b in [9, 19]:
            d = ckpt_dir / f"batch_{b}"
            d.mkdir(parents=True)
            (d / "metadata.json").write_text(json.dumps({"batch_idx": b}))

        result = find_latest_checkpoint(ckpt_dir)
        assert result is not None
        assert result[0] == 19

    def test_find_latest_checkpoint_empty(self, tmp_path):
        from util.checkpoint_io import find_latest_checkpoint
        result = find_latest_checkpoint(tmp_path / "nonexistent")
        assert result is None


# ===========================================================================
# 2. source_injection: shared source reading and CUDA patching
# ===========================================================================

class TestSourceInjection:
    """Tests for src/util/source_injection.py."""

    def test_import(self):
        from util.source_injection import (
            CUDA_PATCH_TARGET,
            read_and_patch_evaluate,
            read_and_patch_memit_main,
        )

    def test_cuda_patch_target_constant(self):
        from util.source_injection import CUDA_PATCH_TARGET
        assert "CUDA_VISIBLE_DEVICES" in CUDA_PATCH_TARGET

    def test_read_and_patch_evaluate(self):
        from util.source_injection import read_and_patch_evaluate
        source = read_and_patch_evaluate(
            PROJECT_ROOT / "vendor" / "AlphaEdit",
            results_dir="/tmp/results",
            dir_name="TestAlg",
        )
        # CUDA line should be commented out
        assert '# os.environ["CUDA_VISIBLE_DEVICES"]' in source
        # RESULTS_DIR should be overridden
        assert 'RESULTS_DIR = Path("/tmp/results")' in source
        # dir_name should be overridden
        assert 'dir_name="TestAlg"' in source

    def test_read_and_patch_memit_main(self):
        from util.source_injection import read_and_patch_memit_main
        source = read_and_patch_memit_main(
            PROJECT_ROOT / "vendor" / "AlphaEdit"
        )
        # Relative imports should be fixed
        assert "from memit.compute_ks" in source or "from .compute_ks" not in source


# ===========================================================================
# 3. experiment_config: centralized path construction (Phase 2 prep)
# ===========================================================================

class TestMegaBatchEval:
    """Tests for src/util/mega_batch_eval.py."""

    def test_import(self):
        from util.mega_batch_eval import get_mega_batch_eval_source

    def test_source_compiles(self):
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        compile(source, "<mega_batch_eval>", "exec")

    def test_source_defines_function(self):
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "def _mega_batch_eval(" in source

    def test_source_has_dual_metrics(self):
        """Must produce both prob-pref (_probs) and argmax (_correct) fields."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "rewrite_prompts_probs" in source
        assert "rewrite_prompts_correct" in source
        assert "neighborhood_prompts_probs" in source
        assert "neighborhood_prompts_correct" in source
        assert "paraphrase_prompts_probs" in source
        assert "paraphrase_prompts_correct" in source

    def test_source_frees_gpu_memory(self):
        """Must call empty_cache() after each batch to prevent OOM."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "empty_cache()" in source
        assert "del tok_out, logits" in source or "del prompt_tok, logits" in source

    def test_source_handles_llama_tokenizer(self):
        """Must handle Llama tokenizer (strips BOS token from target)."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "_is_llama" in source
        assert "a_tok[1:]" in source or "a_tok = a_tok[1:]" in source

    def test_source_skips_existing_files(self):
        """Must skip records whose output file already exists (idempotent)."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "out_file.exists()" in source
        assert "_mbe_skipped" in source

    def test_all_vendor_runners_have_mega_batch_eval(self):
        """Every vendor runner that does MCF evaluation must have mega_batch_eval."""
        missing = []
        for runner in [
            "src/runners/checkpoint_runner.py",
            "src/polykernel/polykernel_seqreg_runner.py",
            "src/runners/memit_sequential_runner.py",
            "src/runners/pathguard_runner.py",
        ]:
            path = PROJECT_ROOT / runner
            if path.exists() and "_mega_batch_eval" not in path.read_text():
                missing.append(runner)
        assert not missing, f"These runners lack mega_batch_eval: {missing}"

    def test_all_baseline_scripts_have_mega_batch_eval(self):
        """Every baseline script must inject mega_batch_eval from the shared module."""
        missing = []
        for script in [
            "scripts/run_evoedit_baseline.sh",
            "scripts/run_nse_baseline.sh",
            "scripts/run_rect_aligned_paper_replication.sh",
        ]:
            path = PROJECT_ROOT / script
            if path.exists() and "mega_batch_eval" not in path.read_text():
                missing.append(script)
        assert not missing, f"These baselines lack mega_batch_eval: {missing}"

    def test_shared_module_matches_inline_output_format(self):
        """Shared module must produce the same output fields as inline copies."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        # Output JSON structure must match what analysis/loaders.py expects
        required_fields = [
            "case_id", "grouped_case_ids", "num_edits",
            "requested_rewrite", "time", "post",
        ]
        for field in required_fields:
            assert f'"{field}"' in source, f"mega_batch_eval output missing field: {field}"

    def test_shared_module_batch_size_parameterized(self):
        """batch_size must be a parameter (different GPUs need different sizes)."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "batch_size=4" in source or "batch_size=" in source

    def test_shared_module_prob_pref_convention(self):
        """NLL convention: target_new and target_true stored as NLL values.
        Lower NLL = higher probability. Success comparisons done by caller."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert '"target_new"' in source
        assert '"target_true"' in source
        assert "log_softmax" in source  # computes NLL via log_softmax

    def test_shared_module_progress_logging(self):
        """Must print [MEGA-BATCH EVAL] progress lines for sky logs monitoring."""
        from util.mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert "[MEGA-BATCH EVAL]" in source
        assert "Complete:" in source


class TestExperimentConfig:
    """Tests for src/util/experiment_config.py (Phase 2)."""

    def test_import(self):
        from util.experiment_config import ExperimentConfig

    def test_variant_name_memit(self):
        from util.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering="fb_high_exposure",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=1.0, lambda_delta=0.0,
            kernel_degree=2, revive=False,
        )
        assert cfg.variant_name.startswith("MEMIT-Seq-")
        assert "lp1.0" in cfg.variant_name

    def test_variant_name_alphaedit(self):
        from util.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, ordering="fb_high_exposure",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=0.0, lambda_delta=0.0,
            kernel_degree=1, revive=True, revive_tau=0.1,
        )
        assert cfg.variant_name.startswith("AlphaEdit-")
        assert "REVIVE" in cfg.variant_name

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_all_base_algs_distinct(self, base_alg):
        from util.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg=base_alg, seed=42, ordering="fb_high",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        expected = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert expected in cfg.variant_name

    def test_checkpoint_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path))
        from util.experiment_config import ExperimentConfig
        import importlib
        import util.paths
        importlib.reload(util.paths)

        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, ordering="fb_high_exposure",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            kernel_degree=1, revive=True, revive_tau=0.1,
        )
        ckpt = cfg.checkpoint_dir()
        assert "AlphaEdit" in str(ckpt)
        assert "fb_high_exposure" in str(ckpt)
        assert "seed42" in str(ckpt)

    def test_model_tag_gptj(self):
        from util.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            model_name="EleutherAI/gpt-j-6b",
        )
        assert cfg.model_tag == "gpt-j-6b"

    def test_model_tag_llama_default(self):
        from util.experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        assert cfg.model_tag == ""
