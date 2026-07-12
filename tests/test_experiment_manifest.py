"""
Tests for the experiment manifest configuration.

Validates YAML structure, required fields, and internal consistency.
No GPU required — pure configuration validation.
"""

from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "configs" / "experiment_manifest.yaml"


@pytest.fixture
def manifest():
    """Load the experiment manifest."""
    with open(MANIFEST_PATH, "r") as f:
        return yaml.safe_load(f)


class TestManifestStructure:
    """Tests for top-level manifest structure."""

    def test_manifest_exists(self):
        """Manifest file should exist."""
        assert MANIFEST_PATH.exists()

    def test_valid_yaml(self):
        """Manifest should be valid YAML."""
        with open(MANIFEST_PATH, "r") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_required_top_level_keys(self, manifest):
        """Should have all required top-level sections."""
        required = {"study", "models", "seeds", "experiments", "hardware"}
        assert required.issubset(set(manifest.keys()))

    def test_has_high_priority_experiments(self, manifest):
        """Should have high_priority_experiments section."""
        assert "high_priority_experiments" in manifest

    def test_has_calibration_experiments(self, manifest):
        """Should have calibration_experiments section."""
        assert "calibration_experiments" in manifest


class TestStudyMetadata:
    """Tests for study metadata fields."""

    def test_alphaedit_commit_is_pinned(self, manifest):
        """AlphaEdit commit should be a specific SHA."""
        commit = manifest["study"]["alphaedit_commit"]
        assert len(commit) == 40  # Full SHA-1 hash
        assert all(c in "0123456789abcdef" for c in commit)

    def test_target_venue(self, manifest):
        """Should target TMLR."""
        assert "TMLR" in manifest["study"]["target_venue"]


class TestModels:
    """Tests for model configuration."""

    def test_primary_model_is_llama(self, manifest):
        """Primary model should be Llama-3-8B."""
        primary = manifest["models"]["primary"]
        assert "Llama-3-8B" in primary["name"] or "llama" in primary["name"].lower()
        assert primary["hparams_fname"].endswith(".json")

    def test_secondary_model_exists(self, manifest):
        """Should have a secondary model for generalization testing."""
        assert "secondary" in manifest["models"]
        secondary = manifest["models"]["secondary"]
        assert "name" in secondary
        assert "hparams_fname" in secondary


class TestSeeds:
    """Tests for seed configuration."""

    def test_five_seeds(self, manifest):
        """Should have exactly 5 default seeds."""
        seeds = manifest["seeds"]["default"]
        assert len(seeds) == 5

    def test_seeds_are_integers(self, manifest):
        """All seeds should be integers."""
        seeds = manifest["seeds"]["default"]
        assert all(isinstance(s, int) for s in seeds)

    def test_seeds_are_unique(self, manifest):
        """All seeds should be unique."""
        seeds = manifest["seeds"]["default"]
        assert len(seeds) == len(set(seeds))

    def test_known_seeds(self, manifest):
        """Seeds should match the documented set."""
        seeds = set(manifest["seeds"]["default"])
        expected = {42, 137, 2024, 7, 99}
        assert seeds == expected


class TestExperiments:
    """Tests for experiment definitions."""

    def test_mve_experiments_exist(self, manifest):
        """All 4 MVE experiments should be defined."""
        experiments = manifest["experiments"]
        assert "mve1_alphaedit_mcf" in experiments
        assert "mve2_memit_mcf" in experiments
        assert "mve3_alphaedit_zsre" in experiments
        assert "mve4_conflict_seq" in experiments

    def test_experiments_have_required_fields(self, manifest):
        """Each experiment should have alg_name, ds_name, and num_edits."""
        for name, exp in manifest["experiments"].items():
            assert "alg_name" in exp, f"{name} missing alg_name"
            assert "ds_name" in exp, f"{name} missing ds_name"
            assert "num_edits" in exp, f"{name} missing num_edits"

    def test_valid_algorithm_names(self, manifest):
        """Algorithm names should be valid."""
        valid_algs = {"AlphaEdit", "MEMIT", "ROME", "both"}
        for section in ["experiments", "calibration_experiments", "high_priority_experiments"]:
            if section not in manifest:
                continue
            for name, exp in manifest[section].items():
                if "alg_name" in exp:
                    assert exp["alg_name"] in valid_algs, (
                        f"{name} has invalid alg_name: {exp['alg_name']}"
                    )

    def test_valid_dataset_names(self, manifest):
        """Dataset names should be valid."""
        valid_ds = {"mcf", "cf", "zsre", "mquake"}
        for section in ["experiments", "calibration_experiments", "high_priority_experiments"]:
            if section not in manifest:
                continue
            for name, exp in manifest[section].items():
                if "ds_name" in exp:
                    assert exp["ds_name"] in valid_ds, (
                        f"{name} has invalid ds_name: {exp['ds_name']}"
                    )

    def test_mve_consistency(self, manifest):
        """MVE experiments should use consistent parameters."""
        mve1 = manifest["experiments"]["mve1_alphaedit_mcf"]
        mve2 = manifest["experiments"]["mve2_memit_mcf"]
        # Same dataset, same size, same batch size
        assert mve1["ds_name"] == mve2["ds_name"]
        assert mve1["dataset_size_limit"] == mve2["dataset_size_limit"]
        assert mve1["num_edits"] == mve2["num_edits"]
        assert mve1["downstream_eval_steps"] == mve2["downstream_eval_steps"]

    def test_rome_is_calibration(self, manifest):
        """ROME should be in calibration section (not main experiments)."""
        cal = manifest["calibration_experiments"]
        assert "rome_baseline" in cal
        assert cal["rome_baseline"]["alg_name"] == "ROME"


class TestHighPriorityExperiments:
    """Tests for high-priority experiment definitions."""

    def test_failure_curve_has_edit_counts(self, manifest):
        """Failure curve should define a range of edit counts."""
        fc = manifest["high_priority_experiments"]["failure_curve_alphaedit"]
        assert "edit_counts" in fc
        assert len(fc["edit_counts"]) >= 5
        # Should be monotonically increasing
        counts = fc["edit_counts"]
        assert counts == sorted(counts)

    def test_nullspace_tracking_defined(self, manifest):
        """Null-space rank consumption experiment should be defined."""
        assert "nullspace_rank_consumption" in manifest["high_priority_experiments"]

    def test_cache_mitigation_has_variants(self, manifest):
        """Cache mitigation should define variant parameters."""
        cm = manifest["high_priority_experiments"]["cache_mitigation_sweep"]
        assert "variants" in cm
        variants = cm["variants"]
        assert "svd_truncation" in variants
        assert "exponential_decay" in variants
        assert "periodic_reset" in variants

    def test_order_sensitivity_has_num_orderings(self, manifest):
        """Order sensitivity should specify number of orderings."""
        os_exp = manifest["high_priority_experiments"]["edit_order_sensitivity"]
        assert "num_orderings" in os_exp
        assert os_exp["num_orderings"] >= 5

    def test_capability_probe_defined(self, manifest):
        """Capability probe should be defined."""
        assert "capability_probe" in manifest["high_priority_experiments"]
