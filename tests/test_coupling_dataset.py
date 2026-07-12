"""Tests for coupling_dataset.py."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coupling_dataset import (
    build_indexes,
    select_unrelated_pairs,
    select_relation_match_pairs,
    select_subject_match_pairs,
    select_full_conflict_pairs,
    build_coupling_sequence,
    COUPLING_TYPES,
)


# --- Fixtures ---


def make_record(case_id, subject, relation_id, target_new, prompt=None):
    """Create a minimal MultiCounterFact-style record."""
    if prompt is None:
        prompt = f"The {{}} is known for {relation_id}"
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation_id,
            "target_new": {"str": target_new},
            "prompt": prompt,
        },
    }


@pytest.fixture
def sample_data():
    """Diverse dataset with known overlap structure."""
    return [
        # Same subject (Curie), different relations
        make_record(0, "Marie Curie", "born_in", "Warsaw"),
        make_record(1, "Marie Curie", "won_award", "Nobel Prize"),
        make_record(2, "Marie Curie", "field_of_work", "Physics"),
        # Same relation (born_in), different subjects
        make_record(3, "Einstein", "born_in", "Ulm"),
        make_record(4, "Tesla", "born_in", "Smiljan"),
        make_record(5, "Newton", "born_in", "Woolsthorpe"),
        # Same subject+relation (for conflict)
        make_record(6, "Einstein", "born_in", "Munich"),  # conflicts with case 3
        # Fully unrelated
        make_record(7, "Apple", "headquartered_in", "Cupertino"),
        make_record(8, "Google", "founded_by", "Larry Page"),
        make_record(9, "Microsoft", "ceo", "Satya Nadella"),
        make_record(10, "Amazon", "sells", "Books"),
        make_record(11, "Tesla Inc", "makes", "Cars"),
        make_record(12, "SpaceX", "launches", "Rockets"),
        make_record(13, "Netflix", "streams", "Movies"),
        make_record(14, "Meta", "owns", "Instagram"),
    ]


# --- Index Construction ---


class TestBuildIndexes:
    def test_by_subject(self, sample_data):
        by_subj, _, _ = build_indexes(sample_data)
        assert "Marie Curie" in by_subj
        assert len(by_subj["Marie Curie"]) == 3

    def test_by_relation(self, sample_data):
        _, by_rel, _ = build_indexes(sample_data)
        assert "born_in" in by_rel
        assert len(by_rel["born_in"]) == 5  # Curie, Einstein, Tesla, Newton, Einstein/Munich

    def test_by_subject_relation(self, sample_data):
        _, _, by_sr = build_indexes(sample_data)
        assert ("Einstein", "born_in") in by_sr
        assert len(by_sr[("Einstein", "born_in")]) == 2  # cases 3 and 6


# --- Pair Selection ---


class TestSelectUnrelatedPairs:
    def test_basic(self, sample_data):
        import random
        rng = random.Random(42)
        used = set()
        pairs = select_unrelated_pairs(sample_data, max_pairs=3, rng=rng, used_ids=used)
        assert len(pairs) >= 1

    def test_pairs_have_different_subject_and_relation(self, sample_data):
        import random
        rng = random.Random(42)
        used = set()
        pairs = select_unrelated_pairs(sample_data, max_pairs=5, rng=rng, used_ids=used)
        for a, b in pairs:
            rw_a = a["requested_rewrite"]
            rw_b = b["requested_rewrite"]
            assert rw_a["subject"] != rw_b["subject"]
            assert rw_a["relation_id"] != rw_b["relation_id"]

    def test_used_ids_updated(self, sample_data):
        import random
        rng = random.Random(42)
        used = set()
        pairs = select_unrelated_pairs(sample_data, max_pairs=3, rng=rng, used_ids=used)
        assert len(used) == len(pairs) * 2


class TestSelectRelationMatchPairs:
    def test_same_relation_different_subject(self, sample_data):
        import random
        _, by_rel, _ = build_indexes(sample_data)
        rng = random.Random(42)
        used = set()
        pairs = select_relation_match_pairs(by_rel, max_pairs=3, rng=rng, used_ids=used)
        assert len(pairs) >= 1
        for a, b in pairs:
            rw_a = a["requested_rewrite"]
            rw_b = b["requested_rewrite"]
            assert rw_a["relation_id"] == rw_b["relation_id"]
            assert rw_a["subject"] != rw_b["subject"]


class TestSelectSubjectMatchPairs:
    def test_same_subject_different_relation(self, sample_data):
        import random
        by_subj, _, _ = build_indexes(sample_data)
        rng = random.Random(42)
        used = set()
        pairs = select_subject_match_pairs(by_subj, max_pairs=3, rng=rng, used_ids=used)
        assert len(pairs) >= 1
        for a, b in pairs:
            rw_a = a["requested_rewrite"]
            rw_b = b["requested_rewrite"]
            assert rw_a["subject"] == rw_b["subject"]
            assert rw_a["relation_id"] != rw_b["relation_id"]


class TestSelectFullConflictPairs:
    def test_natural_conflict(self, sample_data):
        import random
        _, by_rel, by_sr = build_indexes(sample_data)
        rng = random.Random(42)
        used = set()
        pairs = select_full_conflict_pairs(by_sr, by_rel, max_pairs=3, rng=rng, used_ids=used)
        assert len(pairs) >= 1
        for a, b in pairs:
            rw_a = a["requested_rewrite"]
            rw_b = b["requested_rewrite"]
            assert rw_a["subject"] == rw_b["subject"]
            assert rw_a["relation_id"] == rw_b["relation_id"]


# --- Sequence Building ---


class TestBuildCouplingSequence:
    def test_warmup_first(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {0: [], 1: [], 2: [], 3: []}
        # Add one pair per type
        type_pairs[0] = [(sample_data[7], sample_data[8])]
        type_pairs[1] = [(sample_data[3], sample_data[4])]

        warmup_pool = [sample_data[9], sample_data[10], sample_data[11]]
        seq = build_coupling_sequence(type_pairs, warmup_pool, warmup_count=2, rng=rng)

        # First 2 should be warmup
        assert seq[0]["coupling_metadata"]["role"] == "warmup"
        assert seq[1]["coupling_metadata"]["role"] == "warmup"

    def test_anchor_before_probe(self, sample_data):
        import random
        rng = random.Random(0)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [(sample_data[3], sample_data[4])],
            2: [(sample_data[0], sample_data[1])],
            3: [(sample_data[3], sample_data[6])],
        }

        seq = build_coupling_sequence(type_pairs, [], warmup_count=0, rng=rng)

        # For each pair_id, anchor should appear before probe
        pair_positions = defaultdict(dict)
        for i, record in enumerate(seq):
            meta = record["coupling_metadata"]
            pid = meta["pair_id"]
            if pid is not None:
                pair_positions[pid][meta["role"]] = i

        for pid, positions in pair_positions.items():
            if "anchor" in positions and "probe" in positions:
                assert positions["anchor"] < positions["probe"]

    def test_case_ids_sequential(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [], 2: [], 3: [],
        }
        seq = build_coupling_sequence(type_pairs, [sample_data[9]], warmup_count=1, rng=rng)

        case_ids = [r["case_id"] for r in seq]
        assert case_ids == list(range(len(seq)))

    def test_metadata_fields_present(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [], 2: [], 3: [],
        }
        seq = build_coupling_sequence(type_pairs, [], warmup_count=0, rng=rng)

        for record in seq:
            meta = record["coupling_metadata"]
            assert "coupling_type" in meta
            assert "coupling_type_name" in meta
            assert "role" in meta
            assert "pair_id" in meta
            assert "anchor_case_id" in meta

    def test_probe_has_anchor_case_id(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [], 2: [], 3: [],
        }
        seq = build_coupling_sequence(type_pairs, [], warmup_count=0, rng=rng)

        probes = [r for r in seq if r["coupling_metadata"]["role"] == "probe"]
        for probe in probes:
            assert probe["coupling_metadata"]["anchor_case_id"] is not None


# --- Output Format ---


class TestOutputFormat:
    def test_record_has_requested_rewrite(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [], 2: [], 3: [],
        }
        seq = build_coupling_sequence(type_pairs, [], warmup_count=0, rng=rng)

        for record in seq:
            assert "requested_rewrite" in record
            rw = record["requested_rewrite"]
            assert "subject" in rw
            assert "relation_id" in rw
            assert "target_new" in rw
            assert "str" in rw["target_new"]

    def test_json_serializable(self, sample_data):
        import random
        rng = random.Random(42)
        type_pairs = {
            0: [(sample_data[7], sample_data[8])],
            1: [], 2: [], 3: [],
        }
        seq = build_coupling_sequence(type_pairs, [], warmup_count=0, rng=rng)
        # Should not raise
        json.dumps(seq)
