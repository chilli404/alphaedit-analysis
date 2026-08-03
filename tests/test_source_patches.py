#!/usr/bin/env python3
"""
Tests for source patches: model-list, GLUE context, P-cache.
Verifies patches are additive and don't break existing behavior.
No GPU required.

Run with: uv run pytest tests/test_source_patches.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from source_patches import (
    SHAPE_MODEL_LIST_ANCHOR,
    SHAPE_MODEL_LIST_EXTENDED,
    GLUE_MAP_ANCHOR,
    GLUE_MAP_EXTENDED,
    P_COMPUTE_ANCHOR,
    P_COMPUTE_CACHED,
    apply_model_list_patch,
    apply_glue_context_patch,
    apply_p_cache_patch,
)


class TestModelListPatch:
    def test_adds_qwen_to_list(self):
        source = f'some code\n    elif hparams.model_name in {SHAPE_MODEL_LIST_ANCHOR}:\n'
        patched = apply_model_list_patch(source)
        assert "Qwen2.5-7B" in patched

    def test_preserves_existing_models(self):
        source = f'some code\n    elif hparams.model_name in {SHAPE_MODEL_LIST_ANCHOR}:\n'
        patched = apply_model_list_patch(source)
        assert "EleutherAI_gpt-j-6B" in patched
        assert "Llama3-8B" in patched
        assert "phi-1.5" in patched

    def test_idempotent(self):
        source = f'some code\n    elif hparams.model_name in {SHAPE_MODEL_LIST_ANCHOR}:\n'
        patched = apply_model_list_patch(source)
        double_patched = apply_model_list_patch(patched)
        # After first patch, anchor is gone so second patch is no-op
        assert patched == double_patched

    def test_noop_when_anchor_missing(self):
        source = "some code without the anchor"
        patched = apply_model_list_patch(source)
        assert patched == source


class TestGlueContextPatch:
    def test_adds_qwen_entry(self):
        source = f'MODEL_MAP = {{\n    {GLUE_MAP_ANCHOR}\n}}'
        patched = apply_glue_context_patch(source)
        assert "qwen2.5-7b-instruct" in patched
        assert "4096" in patched

    def test_adds_gptj_entry(self):
        source = f'MODEL_MAP = {{\n    {GLUE_MAP_ANCHOR}\n}}'
        patched = apply_glue_context_patch(source)
        assert "gpt-j-6b" in patched
        assert "2048" in patched

    def test_preserves_existing_entries(self):
        source = f'"gpt2-xl": 1024,\n    {GLUE_MAP_ANCHOR}'
        patched = apply_glue_context_patch(source)
        assert "gpt2-xl" in patched
        assert "gpt2-medium" in patched

    def test_idempotent(self):
        source = f'MODEL_MAP = {{\n    {GLUE_MAP_ANCHOR}\n}}'
        patched = apply_glue_context_patch(source)
        double_patched = apply_glue_context_patch(patched)
        assert patched == double_patched

    def test_noop_when_anchor_missing(self):
        source = "no glue map here"
        patched = apply_glue_context_patch(source)
        assert patched == source


class TestPCachePatch:
    def test_replaces_compute_with_cache(self):
        source = f'before\n{P_COMPUTE_ANCHOR}\nafter'
        patched = apply_p_cache_patch(source)
        assert "null_space_project.pt" in patched
        assert "Loaded cached" in patched

    def test_uses_hparams_model_name_for_stats_path(self):
        source = f'before\n{P_COMPUTE_ANCHOR}\nafter'
        patched = apply_p_cache_patch(source)
        assert "hparams.model_name" in patched
        # Verify no hardcoded llama path
        assert "llama3-8b-instruct" not in patched

    def test_idempotent(self):
        source = f'before\n{P_COMPUTE_ANCHOR}\nafter'
        patched = apply_p_cache_patch(source)
        double_patched = apply_p_cache_patch(patched)
        assert patched == double_patched

    def test_noop_when_anchor_missing(self):
        source = "no P computation anchor here"
        patched = apply_p_cache_patch(source)
        assert patched == source


class TestCombinedPatches:
    """Test that all patches compose correctly on realistic source."""

    def test_all_patches_compose(self):
        # Simulate the real evaluate.py with both anchors
        source = (
            f'import torch\n'
            f'    if hparams.model_name == "gpt2-xl":\n'
            f'        pass\n'
            f'    elif hparams.model_name in {SHAPE_MODEL_LIST_ANCHOR}:\n'
            f'        pass\n'
            f'{P_COMPUTE_ANCHOR}\n'
            f'    for record_chunks in chunks(ds, num_edits):\n'
        )
        patched = apply_model_list_patch(source)
        patched = apply_p_cache_patch(patched)
        assert "Qwen2.5-7B" in patched
        assert "null_space_project.pt" in patched
        assert "hparams.model_name" in patched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
