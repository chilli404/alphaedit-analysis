#!/usr/bin/env python3
"""
Unit tests for the model registry.
No GPU required — tests pure Python logic only.

Run with: uv run pytest tests/test_model_registry.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from model_registry import (
    MODEL_REGISTRY,
    LLAMA3_8B,
    GPTJ_6B,
    QWEN25_7B,
    get_model_spec,
    get_evaluate_model_names,
)


class TestRegistryContents:
    def test_all_three_models_registered(self):
        assert "Llama3-8B" in MODEL_REGISTRY
        assert "EleutherAI_gpt-j-6B" in MODEL_REGISTRY
        assert "Qwen2.5-7B" in MODEL_REGISTRY

    def test_llama3_spec(self):
        spec = MODEL_REGISTRY["Llama3-8B"]
        assert spec.hidden_size == 4096
        assert spec.n_layers == 32
        assert spec.edit_layers == (4, 5, 6, 7, 8)
        assert spec.weight_dim_index == 1
        assert spec.has_bos_token is True
        assert spec.context_length == 4096

    def test_gptj_spec(self):
        spec = MODEL_REGISTRY["EleutherAI_gpt-j-6B"]
        assert spec.hidden_size == 4096
        assert spec.n_layers == 28
        assert spec.edit_layers == (3, 4, 5, 6, 7, 8)
        assert spec.weight_dim_index == 1
        assert spec.has_bos_token is False
        assert spec.context_length == 2048

    def test_qwen_spec(self):
        spec = MODEL_REGISTRY["Qwen2.5-7B"]
        assert spec.hidden_size == 3584
        assert spec.n_layers == 28
        assert spec.edit_layers == (4, 5, 6, 7, 8)
        assert spec.weight_dim_index == 1
        assert spec.has_bos_token is False
        assert spec.context_length == 4096

    def test_clustering_layer_within_edit_layers(self):
        for name, spec in MODEL_REGISTRY.items():
            assert spec.clustering_layer in spec.edit_layers, (
                f"{name}: clustering_layer {spec.clustering_layer} "
                f"not in edit_layers {spec.edit_layers}"
            )


class TestModelResolution:
    """Test that various identifier forms resolve correctly."""

    def test_resolve_by_short_name(self):
        assert get_model_spec("Llama3-8B") is LLAMA3_8B
        assert get_model_spec("EleutherAI_gpt-j-6B") is GPTJ_6B
        assert get_model_spec("Qwen2.5-7B") is QWEN25_7B

    def test_resolve_by_hf_repo(self):
        assert get_model_spec("meta-llama/Meta-Llama-3-8B-Instruct") is LLAMA3_8B
        assert get_model_spec("EleutherAI/gpt-j-6b") is GPTJ_6B
        assert get_model_spec("Qwen/Qwen2.5-7B-Instruct") is QWEN25_7B

    def test_resolve_case_insensitive(self):
        assert get_model_spec("llama3-8b") is LLAMA3_8B
        assert get_model_spec("ELEUTHERAI_GPT-J-6B") is GPTJ_6B
        assert get_model_spec("qwen2.5-7b") is QWEN25_7B

    def test_resolve_by_basename(self):
        # Simulates model.config._name_or_path after loading
        assert get_model_spec("Meta-Llama-3-8B-Instruct") is LLAMA3_8B
        assert get_model_spec("gpt-j-6b") is GPTJ_6B
        assert get_model_spec("Qwen2.5-7B-Instruct") is QWEN25_7B

    def test_resolve_nousresearch_alias(self):
        assert get_model_spec("nousresearch/Meta-Llama-3-8B-Instruct") is LLAMA3_8B

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_model_spec("unknown-model/does-not-exist")


class TestEvaluateModelNames:
    def test_all_registered_models_use_dim1(self):
        names = get_evaluate_model_names()
        assert "Llama3-8B" in names
        assert "EleutherAI_gpt-j-6B" in names
        assert "Qwen2.5-7B" in names

    def test_returns_list_of_strings(self):
        names = get_evaluate_model_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
