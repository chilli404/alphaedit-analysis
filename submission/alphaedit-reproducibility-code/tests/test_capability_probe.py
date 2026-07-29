"""
Tests for the capability probe module.

Tests the MMLU formatting helper and perplexity computation logic.
Perplexity computation tests use mock model/tokenizer objects.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capability_probe import _format_mmlu_question, compute_perplexity


class TestFormatMMLUQuestion:
    """Tests for MMLU question formatting."""

    def test_basic_format_with_answer(self):
        """Should format question with choices and answer."""
        item = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }
        text = _format_mmlu_question(item, ["A", "B", "C", "D"])
        assert "What is 2+2?" in text
        assert "A. 3" in text
        assert "B. 4" in text
        assert "C. 5" in text
        assert "D. 6" in text
        assert "Answer: B" in text

    def test_format_without_answer(self):
        """Should format without answer when include_answer=False."""
        item = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }
        text = _format_mmlu_question(item, ["A", "B", "C", "D"], include_answer=False)
        assert "Answer:" in text
        assert "Answer: B" not in text

    def test_string_answer(self):
        """Should handle string answer (older MMLU format)."""
        item = {
            "question": "Capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": "B",
        }
        text = _format_mmlu_question(item, ["A", "B", "C", "D"])
        assert "Answer: B" in text

    def test_alternative_field_names(self):
        """Should handle 'input' and 'target' fields."""
        item = {
            "input": "What is DNA?",
            "choices": ["Protein", "Acid", "Sugar", "Enzyme"],
            "target": 1,
        }
        text = _format_mmlu_question(item, ["A", "B", "C", "D"])
        assert "What is DNA?" in text
        assert "Answer: B" in text

    def test_empty_choices(self):
        """Empty choices should still produce valid output."""
        item = {
            "question": "Test?",
            "choices": [],
            "answer": 0,
        }
        text = _format_mmlu_question(item, ["A", "B", "C", "D"])
        assert "Test?" in text
        assert "Answer:" in text


class TestComputePerplexity:
    """Tests for perplexity computation with mock objects."""

    def _make_mock_model_and_tokenizer(self, vocab_size=100, seq_len=10):
        """Create minimal mock objects for testing."""
        import torch

        class MockOutputs:
            def __init__(self, logits):
                self.logits = logits

        class MockModel:
            def __init__(self):
                self._device = torch.device("cpu")
                self._param = torch.nn.Parameter(torch.zeros(1))

            def eval(self):
                pass

            def parameters(self):
                return iter([self._param])

            def __call__(self, input_ids, attention_mask):
                batch_size, seq_length = input_ids.shape
                # Random logits
                torch.manual_seed(0)
                logits = torch.randn(batch_size, seq_length, vocab_size)
                return MockOutputs(logits)

        class MockTokenizer:
            def __call__(self, texts, return_tensors="pt", max_length=512,
                        truncation=True, padding=True):
                batch_size = len(texts)
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                return MockEncodings(input_ids, attention_mask)

        class MockEncodings:
            def __init__(self, input_ids, attention_mask):
                self.data = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self.data[key]

            def keys(self):
                return self.data.keys()

        return MockModel(), MockTokenizer()

    def test_returns_expected_keys(self):
        """Result should have all expected keys."""
        model, tokenizer = self._make_mock_model_and_tokenizer()
        result = compute_perplexity(model, tokenizer, ["Hello world"] * 5)
        assert "mean_perplexity" in result
        assert "median_perplexity" in result
        assert "std_perplexity" in result
        assert "n_samples" in result
        assert "n_tokens" in result

    def test_positive_perplexity(self):
        """Perplexity should always be positive."""
        model, tokenizer = self._make_mock_model_and_tokenizer()
        result = compute_perplexity(model, tokenizer, ["Test text"] * 3)
        assert result["mean_perplexity"] > 0
        assert result["median_perplexity"] > 0

    def test_empty_texts(self):
        """Empty text list should return NaN."""
        model, tokenizer = self._make_mock_model_and_tokenizer()
        result = compute_perplexity(model, tokenizer, [])
        assert np.isnan(result["mean_perplexity"])
        assert result["n_samples"] == 0

    def test_n_samples_matches_input(self):
        """n_samples should match number of valid inputs."""
        model, tokenizer = self._make_mock_model_and_tokenizer(seq_len=10)
        result = compute_perplexity(model, tokenizer, ["text"] * 7, batch_size=3)
        assert result["n_samples"] == 7

    def test_corpus_level_perplexity_definition(self):
        """mean_perplexity should be corpus-level: exp(total_NLL/total_tokens).

        This tests that the bug fix (averaging per-sample perplexities ->
        corpus-level computation) is correct.
        """
        import torch

        vocab_size = 10
        seq_len = 5

        # Create a model that returns known logits
        class DeterministicModel:
            def __init__(self):
                self._param = torch.nn.Parameter(torch.zeros(1))

            def eval(self):
                pass

            def parameters(self):
                return iter([self._param])

            def __call__(self, input_ids, attention_mask):
                batch_size, sl = input_ids.shape
                # Uniform logits: each token prediction has NLL = log(vocab_size)
                logits = torch.zeros(batch_size, sl, vocab_size)
                class Out:
                    pass
                out = Out()
                out.logits = logits
                return out

        class DeterministicEncodings:
            def __init__(self, input_ids, attention_mask):
                self._data = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            def to(self, device):
                return self

            def __getitem__(self, key):
                return self._data[key]

        class DeterministicTokenizer:
            def __call__(self, texts, **kwargs):
                batch_size = len(texts)
                input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                return DeterministicEncodings(input_ids, attention_mask)

        model = DeterministicModel()
        tokenizer = DeterministicTokenizer()

        # With uniform logits over vocab_size=10, NLL per token = log(10)
        # Expected perplexity = exp(log(10)) = 10
        result = compute_perplexity(model, tokenizer, ["a", "b", "c"])
        expected_ppl = vocab_size  # exp(log(vocab_size)) = vocab_size
        assert abs(result["mean_perplexity"] - expected_ppl) < 0.1
