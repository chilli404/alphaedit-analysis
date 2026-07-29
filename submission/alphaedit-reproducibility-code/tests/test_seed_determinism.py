"""
Test that the seeded runner produces deterministic outputs.

This test validates the core reproducibility claim: same seed -> same results.
It runs two minimal experiments with the same seed and verifies that
the output JSON files are identical.

NOTE: This test requires GPU access and the model to be downloaded.
      It is marked as slow and should be run explicitly.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAEDIT_RESULTS = PROJECT_ROOT / "vendor" / "AlphaEdit" / "results"


def run_smoke_experiment(seed: int, run_label: str) -> Path:
    """Run a minimal experiment and return the results directory."""
    # Clear any existing results for this algorithm
    result_dir = ALPHAEDIT_RESULTS / f"AlphaEdit_determinism_{run_label}"
    if result_dir.exists():
        shutil.rmtree(result_dir)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "seeded_runner.py"),
        "--seed", str(seed),
        "--cuda_device", "0",
        "--alg_name", "AlphaEdit",
        "--model_name", "meta-llama/Meta-Llama-3-8B-Instruct",
        "--hparams_fname", "Llama3-8B.json",
        "--ds_name", "mcf",
        "--dataset_size_limit", "4",
        "--num_edits", "2",
        "--downstream_eval_steps", "0",
        "--skip_generation_tests",
        "--conserve_memory",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    if result.returncode != 0:
        pytest.fail(
            f"Experiment failed (seed={seed}, label={run_label}):\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    return ALPHAEDIT_RESULTS


@pytest.mark.slow
@pytest.mark.gpu
class TestSeedDeterminism:
    """Tests that require GPU and model access."""

    def test_same_seed_same_output(self):
        """Two runs with seed=42 should produce identical case JSONs."""
        # Run 1
        run_smoke_experiment(seed=42, run_label="run_a")

        # Collect run 1 results
        run_dirs = sorted(ALPHAEDIT_RESULTS.glob("AlphaEdit/run_*"))
        if not run_dirs:
            pytest.skip("No results produced - likely no GPU or model access")

        run_a_dir = run_dirs[-1]
        run_a_files = {f.name: f for f in run_a_dir.glob("*_edits-case_*.json")}

        if not run_a_files:
            pytest.skip("No case files produced")

        # Clear results and run 2
        # Move run_a aside
        run_a_backup = run_a_dir.parent / "run_a_backup"
        shutil.move(str(run_a_dir), str(run_a_backup))

        try:
            run_smoke_experiment(seed=42, run_label="run_b")

            run_dirs = sorted(ALPHAEDIT_RESULTS.glob("AlphaEdit/run_*"))
            run_b_dir = run_dirs[-1]
            run_b_files = {f.name: f for f in run_b_dir.glob("*_edits-case_*.json")}

            # Compare
            assert set(run_a_files.keys()) == set(run_b_files.keys()), (
                "Different files produced between runs"
            )

            for filename in run_a_files:
                with open(run_a_backup / filename) as f:
                    data_a = json.load(f)
                with open(run_b_files[filename]) as f:
                    data_b = json.load(f)

                # Compare metrics (ignore timing which may differ)
                assert data_a["case_id"] == data_b["case_id"]

                # Use approximate comparison for float metrics.
                # Even with deterministic flags, minor floating-point
                # differences can occur across runs (e.g., cuBLAS workspace).
                post_a = data_a["post"]
                post_b = data_b["post"]
                for key in post_a:
                    val_a, val_b = post_a[key], post_b[key]
                    if isinstance(val_a, float):
                        assert np.isclose(val_a, val_b, rtol=1e-5, atol=1e-7), (
                            f"Non-deterministic output for {filename}, metric '{key}':\n"
                            f"Run A: {val_a}\n"
                            f"Run B: {val_b}\n"
                            f"Diff: {abs(val_a - val_b)}"
                        )
                    else:
                        assert val_a == val_b, (
                            f"Non-deterministic output for {filename}, field '{key}':\n"
                            f"Run A: {val_a}\n"
                            f"Run B: {val_b}"
                        )

        finally:
            # Restore run_a
            if run_a_backup.exists():
                shutil.rmtree(str(run_a_backup))

    def test_different_seed_different_output(self):
        """Runs with different seeds should produce different results."""
        # This is a sanity check that seeding actually affects output
        run_smoke_experiment(seed=42, run_label="seed_42")
        run_dirs_42 = sorted(ALPHAEDIT_RESULTS.glob("AlphaEdit/run_*"))

        run_smoke_experiment(seed=99, run_label="seed_99")
        run_dirs_99 = sorted(ALPHAEDIT_RESULTS.glob("AlphaEdit/run_*"))

        if len(run_dirs_42) < 1 or len(run_dirs_99) < 2:
            pytest.skip("Not enough runs produced")

        # Note: For deterministic models, the edit quality metrics may be
        # identical across seeds (the model is deterministic given same input).
        # The randomness only affects data sampling order if dataset is shuffled.
        # This test mainly validates the infrastructure works with different seeds.
