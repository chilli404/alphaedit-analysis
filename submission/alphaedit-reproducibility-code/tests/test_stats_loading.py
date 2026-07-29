"""
Test that covariance statistics files are correctly linked and loadable.

Verifies:
1. All expected NPZ files exist
2. Files are loadable and contain expected keys
3. Matrix shapes match Llama-3-8B MLP dimensions
"""

import numpy as np
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "stats" / "Meta-Llama-3-8B-Instruct" / "wikipedia_stats"

# Llama-3-8B MLP down_proj shape: intermediate_size (14336) x hidden_size (4096)
# The covariance is computed on the INPUT to down_proj, which is intermediate_size
EXPECTED_LAYERS = [4, 5, 6, 7, 8]
EXPECTED_SHAPE = (14336, 14336)  # mom2 of MLP intermediate activations
# The NPZ files use AlphaEdit's CombinedStat format with nested keys
MOM2_KEY = "mom2.mom2"  # The actual matrix is at this nested key


@pytest.fixture
def stats_dir():
    return STATS_DIR


class TestStatsFiles:
    def test_stats_directory_exists(self, stats_dir):
        assert stats_dir.exists(), (
            f"Stats directory not found: {stats_dir}\n"
            "Run: bash scripts/link_stats.sh"
        )

    def test_all_layer_files_exist(self, stats_dir):
        if not stats_dir.exists():
            pytest.skip("Stats directory not found")

        for layer in EXPECTED_LAYERS:
            filename = f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            filepath = stats_dir / filename
            assert filepath.exists(), f"Missing stats file: {filename}"

    def test_npz_files_loadable(self, stats_dir):
        if not stats_dir.exists():
            pytest.skip("Stats directory not found")

        for layer in EXPECTED_LAYERS:
            filename = f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            filepath = stats_dir / filename
            if not filepath.exists():
                pytest.skip(f"File not found: {filename}")

            data = np.load(filepath)
            assert MOM2_KEY in data.files, (
                f"NPZ file {filename} missing '{MOM2_KEY}' key. "
                f"Available keys: {data.files}"
            )

    def test_matrix_shapes(self, stats_dir):
        if not stats_dir.exists():
            pytest.skip("Stats directory not found")

        for layer in EXPECTED_LAYERS:
            filename = f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            filepath = stats_dir / filename
            if not filepath.exists():
                pytest.skip(f"File not found: {filename}")

            data = np.load(filepath)
            if MOM2_KEY not in data.files:
                pytest.skip(f"No {MOM2_KEY} key in {filename}")

            mom2 = data[MOM2_KEY]
            assert mom2.shape == EXPECTED_SHAPE, (
                f"Layer {layer} mom2 shape {mom2.shape} != expected {EXPECTED_SHAPE}"
            )

    def test_matrices_are_symmetric(self, stats_dir):
        """Covariance matrices should be approximately symmetric."""
        if not stats_dir.exists():
            pytest.skip("Stats directory not found")

        for layer in EXPECTED_LAYERS:
            filename = f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            filepath = stats_dir / filename
            if not filepath.exists():
                pytest.skip(f"File not found: {filename}")

            data = np.load(filepath)
            if MOM2_KEY not in data.files:
                pytest.skip(f"No {MOM2_KEY} key in {filename}")

            mom2 = data[MOM2_KEY]
            # Check symmetry (allow small floating point differences)
            assert np.allclose(mom2, mom2.T, atol=1e-5), (
                f"Layer {layer} mom2 matrix is not symmetric"
            )

    def test_file_count(self, stats_dir):
        if not stats_dir.exists():
            pytest.skip("Stats directory not found")

        npz_files = list(stats_dir.glob("*.npz"))
        assert len(npz_files) == len(EXPECTED_LAYERS), (
            f"Expected {len(EXPECTED_LAYERS)} NPZ files, found {len(npz_files)}"
        )
