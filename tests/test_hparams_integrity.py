#!/usr/bin/env python3
"""
Hparams integrity tests.
Validates all JSON hparams files for correctness and consistency.
No GPU required.

Run with: uv run pytest tests/test_hparams_integrity.py -v
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from model_registry import MODEL_REGISTRY

VENDOR_HPARAMS = PROJECT_ROOT / "vendor" / "AlphaEdit" / "hparams"
PROJECT_HPARAMS = PROJECT_ROOT / "configs" / "hparams"

REQUIRED_FIELDS = [
    "model_name",
    "layers",
    "fact_token",
    "v_num_grad_steps",
    "v_lr",
    "v_loss_layer",
    "rewrite_module_tmp",
    "layer_module_tmp",
    "mlp_module_tmp",
    "attn_module_tmp",
    "ln_f_module",
    "lm_head_module",
    "mom2_dataset",
    "mom2_n_samples",
    "mom2_dtype",
]

ALPHAEDIT_EXTRA_FIELDS = ["nullspace_threshold", "L2"]


def _find_all_hparams_files():
    """Collect all hparams JSON files from project and vendor."""
    files = []
    for root in [PROJECT_HPARAMS, VENDOR_HPARAMS]:
        if root.exists():
            for f in root.rglob("*.json"):
                files.append(f)
    return files


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class TestAllHparamsFilesParseCorrectly:
    @pytest.fixture(params=_find_all_hparams_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
    def hparams_file(self, request):
        return request.param

    def test_parses_as_valid_json(self, hparams_file):
        data = _load_json(hparams_file)
        assert isinstance(data, dict)


class TestProjectHparamsIntegrity:
    """Test the custom project hparams (configs/hparams/)."""

    @pytest.fixture(params=list(PROJECT_HPARAMS.rglob("*.json")), ids=lambda p: p.name)
    def hparams(self, request):
        return _load_json(request.param), request.param

    def test_required_fields_present(self, hparams):
        data, path = hparams
        for field in REQUIRED_FIELDS:
            assert field in data, f"Missing field '{field}' in {path.name}"

    def test_layers_within_bounds(self, hparams):
        data, path = hparams
        model_name = data["model_name"]
        # Find n_layers from registry if possible
        for spec in MODEL_REGISTRY.values():
            if spec.short_name == model_name:
                for layer in data["layers"]:
                    assert 0 <= layer < spec.n_layers, (
                        f"Layer {layer} out of bounds [0, {spec.n_layers}) "
                        f"in {path.name}"
                    )
                break

    def test_v_loss_layer_within_bounds(self, hparams):
        data, path = hparams
        model_name = data["model_name"]
        for spec in MODEL_REGISTRY.values():
            if spec.short_name == model_name:
                assert data["v_loss_layer"] < spec.n_layers, (
                    f"v_loss_layer {data['v_loss_layer']} >= n_layers {spec.n_layers} "
                    f"in {path.name}"
                )
                break


class TestQwenHparams:
    """Specific tests for Qwen2.5-7B hparams."""

    def test_alphaedit_exists(self):
        path = PROJECT_HPARAMS / "AlphaEdit" / "Qwen2.5-7B.json"
        assert path.exists()

    def test_memit_exists(self):
        path = PROJECT_HPARAMS / "MEMIT" / "Qwen2.5-7B.json"
        assert path.exists()

    def test_alphaedit_provenance(self):
        data = _load_json(PROJECT_HPARAMS / "AlphaEdit" / "Qwen2.5-7B.json")
        assert "_provenance" in data
        assert "EasyEdit" in data["_provenance"] or "zjunlp" in data["_provenance"]

    def test_alphaedit_layers(self):
        data = _load_json(PROJECT_HPARAMS / "AlphaEdit" / "Qwen2.5-7B.json")
        assert data["layers"] == [4, 5, 6, 7, 8]
        assert data["v_loss_layer"] == 27

    def test_alphaedit_has_nullspace_fields(self):
        data = _load_json(PROJECT_HPARAMS / "AlphaEdit" / "Qwen2.5-7B.json")
        for field in ALPHAEDIT_EXTRA_FIELDS:
            assert field in data, f"Missing AlphaEdit field '{field}'"

    def test_module_paths_match_qwen_architecture(self):
        data = _load_json(PROJECT_HPARAMS / "AlphaEdit" / "Qwen2.5-7B.json")
        # Qwen2.5 uses the same module naming as Llama
        assert data["rewrite_module_tmp"] == "model.layers.{}.mlp.down_proj"
        assert data["layer_module_tmp"] == "model.layers.{}"
        assert data["ln_f_module"] == "model.norm"


class TestGptjHparams:
    """Verify GPT-J official hparams are unchanged."""

    def test_vendor_alphaedit_exists(self):
        path = VENDOR_HPARAMS / "AlphaEdit" / "EleutherAI_gpt-j-6B.json"
        assert path.exists()

    def test_vendor_memit_exists(self):
        path = VENDOR_HPARAMS / "MEMIT" / "EleutherAI_gpt-j-6B.json"
        assert path.exists()

    def test_alphaedit_layers(self):
        data = _load_json(VENDOR_HPARAMS / "AlphaEdit" / "EleutherAI_gpt-j-6B.json")
        assert data["layers"] == [3, 4, 5, 6, 7, 8]
        assert data["v_loss_layer"] == 27

    def test_alphaedit_v_lr(self):
        data = _load_json(VENDOR_HPARAMS / "AlphaEdit" / "EleutherAI_gpt-j-6B.json")
        assert data["v_lr"] == 5e-1

    def test_module_paths_match_gptj_architecture(self):
        data = _load_json(VENDOR_HPARAMS / "AlphaEdit" / "EleutherAI_gpt-j-6B.json")
        assert data["rewrite_module_tmp"] == "transformer.h.{}.mlp.fc_out"
        assert data["layer_module_tmp"] == "transformer.h.{}"
        assert data["ln_f_module"] == "transformer.ln_f"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
