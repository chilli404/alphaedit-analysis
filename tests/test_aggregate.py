"""
Tests for the aggregation module that converts case JSONs to CSVs.

Tests metric extraction, result collection, and summary statistics.
No GPU required — uses mock JSON data.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from aggregate import extract_metrics_from_case, collect_run_results, METRICS


class TestExtractMetricsFromCase:
    """Tests for extracting metrics from a single case JSON."""

    def test_basic_extraction(self):
        """Should extract all standard metrics from a well-formed case JSON."""
        case = {
            "case_id": 42,
            "num_edits": 100,
            "time": 12.5,
            "requested_rewrite": {
                "subject": "France",
                "target_new": {"str": "Paris"},
                "prompt": "The capital of {} is",
            },
            "post": {
                "rewrite_prompts_correct": 1.0,
                "paraphrase_prompts_correct": 0.8,
                "neighborhood_prompts_correct": 0.95,
                "reference_score": 0.92,
                "essence_score": 0.88,
                "rewrite_prompts_probs": 0.99,
                "paraphrase_prompts_probs": 0.75,
                "neighborhood_prompts_probs": 0.91,
            },
        }
        row = extract_metrics_from_case(case)

        assert row["case_id"] == 42
        assert row["num_edits"] == 100
        assert row["time"] == 12.5
        assert row["subject"] == "France"
        assert row["target_new"] == "Paris"
        assert row["efficacy"] == 1.0
        assert row["generalization"] == 0.8
        assert row["specificity"] == 0.95
        assert row["fluency"] == 0.92
        assert row["consistency"] == 0.88

    def test_list_metrics_averaged(self):
        """List-valued metrics should be averaged."""
        case = {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Test",
                "target_new": {"str": "value"},
            },
            "post": {
                "rewrite_prompts_correct": [1.0, 0.0, 1.0],  # mean = 0.667
                "paraphrase_prompts_correct": [0.5, 0.5],     # mean = 0.5
            },
        }
        row = extract_metrics_from_case(case)
        assert abs(row["efficacy"] - 2.0 / 3.0) < 0.001
        assert row["generalization"] == 0.5

    def test_missing_post_field(self):
        """Missing 'post' should result in None metrics."""
        case = {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Test",
                "target_new": {"str": "value"},
            },
        }
        row = extract_metrics_from_case(case)
        assert row["efficacy"] is None
        assert row["generalization"] is None

    def test_missing_metric_fields(self):
        """Missing individual metric fields should be None."""
        case = {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Test",
                "target_new": {"str": "value"},
            },
            "post": {
                "rewrite_prompts_correct": 1.0,
                # All others missing
            },
        }
        row = extract_metrics_from_case(case)
        assert row["efficacy"] == 1.0
        assert row["generalization"] is None
        assert row["specificity"] is None

    def test_requested_rewrite_as_list(self):
        """Some case JSONs have requested_rewrite as a list."""
        case = {
            "case_id": 5,
            "requested_rewrite": [
                {"subject": "First", "target_new": {"str": "value1"}},
                {"subject": "Second", "target_new": {"str": "value2"}},
            ],
            "post": {},
        }
        row = extract_metrics_from_case(case)
        # Should use the first element
        assert row["subject"] == "First"
        assert row["target_new"] == "value1"

    def test_empty_list_metrics(self):
        """Empty list metrics should be None."""
        case = {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Test",
                "target_new": {"str": "value"},
            },
            "post": {
                "rewrite_prompts_correct": [],
            },
        }
        row = extract_metrics_from_case(case)
        assert row["efficacy"] is None


class TestCollectRunResults:
    """Tests for collecting results from a run directory."""

    def _create_mock_run(self, tmp_dir: Path, n_cases: int = 3):
        """Create mock case JSON files in a directory."""
        for i in range(n_cases):
            case = {
                "case_id": i,
                "num_edits": 100,
                "time": 1.0 + i,
                "requested_rewrite": {
                    "subject": f"Subject{i}",
                    "target_new": {"str": f"Target{i}"},
                },
                "post": {
                    "rewrite_prompts_correct": 0.9 + i * 0.01,
                    "paraphrase_prompts_correct": 0.8,
                    "neighborhood_prompts_correct": 0.95,
                    "reference_score": 0.92,
                    "essence_score": 0.88,
                },
            }
            filename = f"100_edits-case_{i}.json"
            with open(tmp_dir / filename, "w") as f:
                json.dump(case, f)

    def test_collects_all_cases(self):
        """Should find and parse all case JSON files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_mock_run(tmp_path, n_cases=5)
            df = collect_run_results(tmp_path)
            assert len(df) == 5
            assert "run_id" in df.columns

    def test_empty_directory(self):
        """Empty directory should return empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmp:
            df = collect_run_results(Path(tmp))
            assert df.empty

    def test_correct_columns(self):
        """DataFrame should have expected columns."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_mock_run(tmp_path, n_cases=1)
            df = collect_run_results(tmp_path)
            assert "case_id" in df.columns
            assert "efficacy" in df.columns
            assert "generalization" in df.columns
            assert "specificity" in df.columns
            assert "run_id" in df.columns

    def test_ignores_non_case_files(self):
        """Should ignore files that don't match the case JSON pattern."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._create_mock_run(tmp_path, n_cases=2)
            # Add a non-case file
            (tmp_path / "metadata.json").write_text("{}")
            (tmp_path / "summary.csv").write_text("a,b\n1,2")
            df = collect_run_results(tmp_path)
            assert len(df) == 2


class TestMetricsMapping:
    """Tests for the METRICS mapping constant."""

    def test_all_standard_metrics_present(self):
        """Should have mappings for all standard CounterFact metrics."""
        expected_outputs = {
            "efficacy", "generalization", "specificity",
            "fluency", "consistency",
        }
        assert expected_outputs.issubset(set(METRICS.values()))

    def test_all_keys_are_json_fields(self):
        """All keys should be plausible JSON field names."""
        for key in METRICS:
            assert isinstance(key, str)
            assert "_" in key or key.isalpha()

    def test_all_values_are_short_names(self):
        """All values should be concise metric names."""
        for value in METRICS.values():
            assert isinstance(value, str)
            assert len(value) < 30
