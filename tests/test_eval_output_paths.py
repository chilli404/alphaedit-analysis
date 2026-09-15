#!/usr/bin/env python3
"""Tests for eval output path construction in eval_matched_ordering.py.

Run with: uv run pytest tests/test_eval_output_paths.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_matched_ordering import resolve_eval_output_dir


class TestLlamaWithOrdering:
    """Llama models with an ordering → matched_ordering/{alg}/{ordering}/seed{N}/"""

    def test_default_model(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", "fb_random0", 42,
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        assert out == tmp_path / "matched_ordering" / "AlphaEdit" / "fb_random0" / "seed42"

    def test_none_model_name(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "EvoEdit", "fb_high_exposure", 2024, None)
        assert out == tmp_path / "matched_ordering" / "EvoEdit" / "fb_high_exposure" / "seed2024"

    def test_empty_model_name(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "MEMIT_seq_rect", "fb_low_exposure", 42, "")
        assert out == tmp_path / "matched_ordering" / "MEMIT_seq_rect" / "fb_low_exposure" / "seed42"

    def test_qwen_model(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", "fb_random0", 137,
            "Qwen/Qwen2.5-7B-Instruct",
        )
        assert out == tmp_path / "matched_ordering" / "AlphaEdit" / "fb_random0" / "seed137"


class TestLlamaNoOrdering:
    """Llama models without ordering → matched_ordering/{alg}/first_10k/seed{N}/"""

    def test_no_ordering_none(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", None, 42,
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        assert out == tmp_path / "matched_ordering" / "AlphaEdit" / "first_10k" / "seed42"

    def test_no_ordering_empty_string(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "EvoEdit", "", 2024, None)
        assert out == tmp_path / "matched_ordering" / "EvoEdit" / "first_10k" / "seed2024"

    def test_no_ordering_rect(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "MEMIT_seq_rect", None, 42, "")
        assert out == tmp_path / "matched_ordering" / "MEMIT_seq_rect" / "first_10k" / "seed42"

    def test_no_ordering_uses_matched_ordering_not_paper_replication(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "AlphaEdit", None, 42, None)
        assert "paper_replication" not in str(out)
        assert "matched_ordering" in str(out)


class TestGptjWithOrdering:
    """GPT-J models with an ordering → matched_ordering_gptj/{alg}/{ordering}/seed{N}/"""

    def test_canonical_name(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "EvoEdit", "fb_random0", 42,
            "EleutherAI/gpt-j-6b",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "EvoEdit" / "fb_random0" / "seed42"

    def test_lowercase_eleutherai(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "EvoEdit", "fb_high_exposure", 42,
            "eleutherai/gpt-j-6B",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "EvoEdit" / "fb_high_exposure" / "seed42"

    def test_gptj_shorthand(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", "fb_random0", 2024, "gptj",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "AlphaEdit" / "fb_random0" / "seed2024"

    def test_gpt_j_substring(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", "fb_low_exposure", 42, "some-org/gpt-j-model",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "AlphaEdit" / "fb_low_exposure" / "seed42"


class TestGptjNoOrdering:
    """GPT-J without ordering → matched_ordering_gptj/{alg}/first_10k/seed{N}/"""

    def test_gptj_no_ordering(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "EvoEdit", None, 42,
            "EleutherAI/gpt-j-6b",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "EvoEdit" / "first_10k" / "seed42"

    def test_gptj_no_ordering_empty(self, tmp_path):
        out = resolve_eval_output_dir(
            tmp_path, "AlphaEdit", "", 2024, "gptj",
        )
        assert out == tmp_path / "matched_ordering_gptj" / "AlphaEdit" / "first_10k" / "seed2024"

    def test_gptj_no_ordering_still_uses_gptj_dir(self, tmp_path):
        out = resolve_eval_output_dir(tmp_path, "EvoEdit", None, 42, "EleutherAI/gpt-j-6b")
        assert "matched_ordering_gptj" in str(out)


class TestFullOutputPath:
    """Integration: verify the full path including the v2 JSON filename."""

    @pytest.mark.parametrize("model_name,ordering,expected_parts", [
        ("meta-llama/Meta-Llama-3-8B-Instruct", "fb_random0",
         ["matched_ordering", "AlphaEdit", "fb_random0", "seed42"]),
        ("meta-llama/Meta-Llama-3-8B-Instruct", None,
         ["matched_ordering", "AlphaEdit", "first_10k", "seed42"]),
        ("EleutherAI/gpt-j-6b", "fb_random0",
         ["matched_ordering_gptj", "EvoEdit", "fb_random0", "seed42"]),
        ("EleutherAI/gpt-j-6b", None,
         ["matched_ordering_gptj", "EvoEdit", "first_10k", "seed42"]),
    ])
    def test_v2_json_path(self, tmp_path, model_name, ordering, expected_parts):
        alg = "EvoEdit" if "gpt" in model_name.lower() else "AlphaEdit"
        out_dir = resolve_eval_output_dir(tmp_path, alg, ordering, 42, model_name)
        out_path = out_dir / "full_eval_seed42_v2.json"
        for part in expected_parts:
            assert part in str(out_path), f"Expected {part!r} in {out_path}"
        assert out_path.name == "full_eval_seed42_v2.json"

    def test_mkdir_and_write(self, tmp_path):
        out_dir = resolve_eval_output_dir(
            tmp_path, "EvoEdit", "fb_high_exposure", 2024,
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "full_eval_seed2024_v2.json"
        out_path.write_text('{"test": true}')
        assert out_path.exists()
        assert out_path.read_text() == '{"test": true}'


class TestFalsyOrdering:
    """Falsy ordering values all route to first_10k subdir."""

    @pytest.mark.parametrize("ordering", [None, "", False])
    def test_falsy_ordering_uses_first_10k(self, tmp_path, ordering):
        out = resolve_eval_output_dir(tmp_path, "AlphaEdit", ordering, 42, None)
        assert "first_10k" in str(out)

    @pytest.mark.parametrize("ordering", [None, "", False])
    def test_falsy_ordering_no_ordering_in_path(self, tmp_path, ordering):
        out = resolve_eval_output_dir(tmp_path, "AlphaEdit", ordering, 42, None)
        parts = out.relative_to(tmp_path).parts
        assert "fb_random0" not in parts
        assert "fb_high_exposure" not in parts
