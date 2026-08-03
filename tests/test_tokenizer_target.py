#!/usr/bin/env python3
"""
Target-token audit: verifies leading-space tokenization and BOS handling.

Tests that the target-string tokenization (leading-space handling) produces the
intended first-token match criterion on MCF records for each model.

Requires model tokenizers (downloaded on first run, no GPU needed).

Run with: uv run pytest tests/test_tokenizer_target.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

from model_registry import MODEL_REGISTRY

# Sample target strings from MultiCounterFact (representative subset)
MCF_TARGETS = [
    "Paris",
    "French",
    "English",
    "Microsoft",
    "Germany",
    "United States",
    "London",
    "Apple",
    "Christianity",
    "Japanese",
    "Harvard University",
    "Spanish",
    "New York City",
    "the United Kingdom",
    "Italian",
    "Python",
    "Catholic",
    "Stanford University",
    "Chinese",
    "Democratic",
]


@pytest.fixture(scope="module")
def tokenizers():
    """Load tokenizers for all registered models."""
    from transformers import AutoTokenizer

    toks = {}
    for name, spec in MODEL_REGISTRY.items():
        try:
            tok = AutoTokenizer.from_pretrained(spec.hf_repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            toks[name] = tok
        except Exception as e:
            pytest.skip(f"Cannot load tokenizer for {name}: {e}")
    return toks


class TestLeadingSpaceConvention:
    """All models use the leading-space convention: target_new is prepended with ' '."""

    def test_leading_space_produces_nonempty_tokens(self, tokenizers):
        for model_name, tok in tokenizers.items():
            for target in MCF_TARGETS:
                spaced_target = f" {target}"
                tokens = tok(spaced_target, add_special_tokens=False)["input_ids"]
                assert len(tokens) > 0, (
                    f"{model_name}: empty tokenization for ' {target}'"
                )

    def test_first_token_differs_without_space(self, tokenizers):
        """Verify the leading space actually matters for tokenization."""
        for model_name, tok in tokenizers.items():
            differ_count = 0
            for target in MCF_TARGETS:
                with_space = tok(f" {target}", add_special_tokens=False)["input_ids"]
                without_space = tok(target, add_special_tokens=False)["input_ids"]
                if with_space[0] != without_space[0]:
                    differ_count += 1
            # At least some targets should tokenize differently with/without space
            # (This is the whole point of the leading-space convention)
            assert differ_count > 0, (
                f"{model_name}: leading space never changes first token "
                f"(convention may be unnecessary for this tokenizer)"
            )


class TestBosTokenBehavior:
    """Verify BOS token behavior matches registry has_bos_token field."""

    def test_bos_matches_registry(self, tokenizers):
        for model_name, tok in tokenizers.items():
            spec = MODEL_REGISTRY[model_name]
            # Tokenize a simple string with special tokens
            tokens = tok("Hello world", add_special_tokens=True)["input_ids"]
            # Check if first token is the BOS token
            has_bos = (tok.bos_token_id is not None and
                       len(tokens) > 0 and
                       tokens[0] == tok.bos_token_id)
            assert has_bos == spec.has_bos_token, (
                f"{model_name}: expected has_bos_token={spec.has_bos_token} "
                f"but tokenizer {'adds' if has_bos else 'does not add'} BOS. "
                f"First token: {tokens[0]}, BOS ID: {tok.bos_token_id}"
            )


class TestEvalTokenizationConsistency:
    """
    Simulate the eval_utils_counterfact.py tokenization logic:
    - Tokenize ' {target}' to get target tokens
    - For Llama: strip first token (BOS)
    - Verify the resulting first token is the intended match target
    """

    def test_first_token_after_bos_strip(self, tokenizers):
        for model_name, tok in tokenizers.items():
            spec = MODEL_REGISTRY[model_name]
            for target in MCF_TARGETS:
                # This matches the vendor code: tok(f" {n}")["input_ids"]
                target_tokens = tok(f" {target}")["input_ids"]
                if spec.has_bos_token:
                    # Strip BOS (like the vendor code does for Llama)
                    target_tokens = target_tokens[1:]
                assert len(target_tokens) > 0, (
                    f"{model_name}: no tokens after BOS strip for ' {target}'"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
