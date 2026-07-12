"""
Tests for the conflict dataset generator.

Tests the logic for finding conflict pairs and building conflict sequences.
All tests use mock data — no real dataset files or GPU required.
"""

import json

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conflict_dataset import find_conflict_pairs, build_conflict_sequence


def make_record(case_id, subject, relation_id, target_new, target_true="original"):
    """Helper to create a mock CounterFact record."""
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation_id,
            "target_new": {"str": target_new},
            "target_true": {"str": target_true},
            "prompt": "The capital of {} is",
        },
        "paraphrase_prompts": ["What is the capital of {}?"],
        "neighborhood_prompts": ["unrelated prompt"],
    }


class TestFindConflictPairs:
    """Tests for finding natural and synthetic conflict pairs."""

    def test_finds_natural_conflicts(self):
        """Records with same subject+relation but different targets form pairs."""
        data = [
            make_record(1, "France", "P36", "Paris"),
            make_record(2, "France", "P36", "Lyon"),
            make_record(3, "Germany", "P36", "Berlin"),
        ]
        pairs = find_conflict_pairs(data, max_pairs=10)
        # At least 1 natural conflict (France->Paris vs France->Lyon)
        # May also produce synthetic pairs from same-relation records
        assert len(pairs) >= 1
        # First pair should be the natural conflict
        first, second = pairs[0]
        assert first["requested_rewrite"]["subject"] == "France"
        assert second["requested_rewrite"]["subject"] == "France"

    def test_no_conflicts_in_unique_data(self):
        """All different subjects should produce synthetic pairs only."""
        data = [
            make_record(1, "France", "P36", "Paris"),
            make_record(2, "Germany", "P36", "Berlin"),
            make_record(3, "Spain", "P36", "Madrid"),
            make_record(4, "Italy", "P36", "Rome"),
        ]
        pairs = find_conflict_pairs(data, max_pairs=2)
        # Should synthesize pairs from same-relation records
        assert len(pairs) >= 1

    def test_max_pairs_limit(self):
        """Should respect max_pairs limit."""
        data = [
            make_record(i, "Subject", "P36", f"Target{i}")
            for i in range(20)
        ]
        pairs = find_conflict_pairs(data, max_pairs=3)
        assert len(pairs) <= 3

    def test_multiple_conflicts_same_subject(self):
        """Multiple records with same subject should form multiple pairs."""
        data = [
            make_record(1, "France", "P36", "Paris"),
            make_record(2, "France", "P36", "Lyon"),
            make_record(3, "France", "P36", "Marseille"),
        ]
        pairs = find_conflict_pairs(data, max_pairs=10)
        # (1,2) and (2,3) are natural pairs; synthetic pairs may also be created
        # At least 2 natural conflicts should be found
        assert len(pairs) >= 2
        # First two pairs should be the natural conflicts
        assert pairs[0][0]["case_id"] == 1
        assert pairs[0][1]["case_id"] == 2
        assert pairs[1][0]["case_id"] == 2
        assert pairs[1][1]["case_id"] == 3

    def test_different_relations_not_conflicting(self):
        """Same subject with different relations should NOT be conflicts."""
        data = [
            make_record(1, "France", "P36", "Paris"),      # capital
            make_record(2, "France", "P530", "French"),    # language
        ]
        pairs = find_conflict_pairs(data, max_pairs=10)
        # No natural conflicts; synthetic pairs require same relation
        # Only 1 record per relation, so no synthetic pairs either
        assert len(pairs) == 0

    def test_synthetic_pairs_use_deep_copy(self):
        """Synthetic pairs should not share references with original data."""
        data = [
            make_record(1, "France", "P36", "Paris"),
            make_record(2, "Germany", "P36", "Berlin"),
        ]
        pairs = find_conflict_pairs(data, max_pairs=5)
        if pairs:
            # Modify a pair record; original should be unaffected
            pairs[0][1]["requested_rewrite"]["subject"] = "MODIFIED"
            assert data[0]["requested_rewrite"]["subject"] == "France"
            assert data[1]["requested_rewrite"]["subject"] == "Germany"

    def test_empty_data(self):
        """Empty dataset should return empty pairs."""
        pairs = find_conflict_pairs([], max_pairs=10)
        assert pairs == []

    def test_returns_list_of_tuples(self):
        """Each pair should be a tuple of (first, second) records."""
        data = [
            make_record(1, "France", "P36", "Paris"),
            make_record(2, "France", "P36", "Lyon"),
        ]
        pairs = find_conflict_pairs(data, max_pairs=10)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


class TestBuildConflictSequence:
    """Tests for building a sequential conflict dataset."""

    def _make_pairs(self):
        """Create a small set of mock conflict pairs."""
        return [
            (
                make_record(1, "France", "P36", "Paris"),
                make_record(2, "France", "P36", "Lyon"),
            ),
            (
                make_record(3, "Germany", "P36", "Berlin"),
                make_record(4, "Germany", "P36", "Munich"),
            ),
        ]

    def test_output_length(self):
        """Output should have 2 records per pair."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        assert len(sequence) == 4  # 2 pairs * 2

    def test_sequential_case_ids(self):
        """Case IDs should be sequential starting from 0."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        ids = [r["case_id"] for r in sequence]
        assert ids == [0, 1, 2, 3]

    def test_conflict_metadata_present(self):
        """Each record should have conflict_metadata."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        for record in sequence:
            assert "conflict_metadata" in record

    def test_first_second_positions(self):
        """Records should alternate first/second positions."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        assert sequence[0]["conflict_metadata"]["position"] == "first"
        assert sequence[1]["conflict_metadata"]["position"] == "second"
        assert sequence[2]["conflict_metadata"]["position"] == "first"
        assert sequence[3]["conflict_metadata"]["position"] == "second"

    def test_supersedes_references(self):
        """Second should reference first via supersedes/superseded_by."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        # Pair 0
        assert sequence[0]["conflict_metadata"]["superseded_by"] == 1
        assert sequence[1]["conflict_metadata"]["supersedes"] == 0
        # Pair 1
        assert sequence[2]["conflict_metadata"]["superseded_by"] == 3
        assert sequence[3]["conflict_metadata"]["supersedes"] == 2

    def test_pair_ids(self):
        """Pair IDs should be sequential."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        assert sequence[0]["conflict_metadata"]["pair_id"] == 0
        assert sequence[1]["conflict_metadata"]["pair_id"] == 0
        assert sequence[2]["conflict_metadata"]["pair_id"] == 1
        assert sequence[3]["conflict_metadata"]["pair_id"] == 1

    def test_deep_copy_independence(self):
        """Modifying output should not affect input pairs."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        sequence[0]["requested_rewrite"]["subject"] = "MODIFIED"
        assert pairs[0][0]["requested_rewrite"]["subject"] == "France"

    def test_empty_pairs(self):
        """Empty pairs should give empty sequence."""
        sequence = build_conflict_sequence([], seed=42)
        assert sequence == []

    def test_determinism(self):
        """Same seed should produce same sequence."""
        pairs = self._make_pairs()
        seq1 = build_conflict_sequence(pairs, seed=42)
        seq2 = build_conflict_sequence(pairs, seed=42)
        assert seq1 == seq2

    def test_preserves_original_data(self):
        """Requested_rewrite fields should be preserved from the original."""
        pairs = self._make_pairs()
        sequence = build_conflict_sequence(pairs, seed=42)
        # First entry in pair 0 should have France/Paris
        assert sequence[0]["requested_rewrite"]["subject"] == "France"
        assert sequence[0]["requested_rewrite"]["target_new"]["str"] == "Paris"
        # Second entry in pair 0 should have France/Lyon
        assert sequence[1]["requested_rewrite"]["subject"] == "France"
        assert sequence[1]["requested_rewrite"]["target_new"]["str"] == "Lyon"
