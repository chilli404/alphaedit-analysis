"""Tests for coupling_stress_runner.py source injection validity."""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coupling_stress_runner import (
    build_coupling_script,
    validate_anchors,
    get_alphaedit_root,
    RESID_ANCHOR,
    UPD_ANCHOR,
    ALGO_IMPORT_ANCHOR,
    CUDA_PATCH_TARGET,
    SHUFFLE_ANCHOR,
    PRE_EDIT_ANCHOR,
    POST_EDIT_ANCHOR,
)


ALPHAEDIT_ROOT = get_alphaedit_root()
HAS_SUBMODULE = (ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py").exists()


# --- Script Generation ---


class TestBuildCouplingScript:
    def test_generates_valid_python(self):
        script = build_coupling_script(
            seed=42,
            cuda_device="0",
            model_name="test-model",
            hparams_fname="Test.json",
            ds_name="mcf",
            dataset_size_limit=10,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/test_coupling.json",
            output_jsonl="/tmp/test_output.jsonl",
        )
        # Should parse without SyntaxError
        ast.parse(script)

    def test_contains_seed_setting(self):
        script = build_coupling_script(
            seed=123,
            cuda_device="0",
            model_name="test-model",
            hparams_fname="Test.json",
            ds_name="mcf",
            dataset_size_limit=10,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/test.json",
            output_jsonl="/tmp/test.jsonl",
        )
        assert "seed = 123" in script
        assert "random.seed(seed)" in script
        assert "np.random.seed(seed)" in script
        assert "torch.manual_seed(seed)" in script

    def test_contains_algo_patching(self):
        script = build_coupling_script(
            seed=42,
            cuda_device="0",
            model_name="test-model",
            hparams_fname="Test.json",
            ds_name="mcf",
            dataset_size_limit=10,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/test.json",
            output_jsonl="/tmp/test.jsonl",
        )
        assert "AlphaEdit/AlphaEdit_main.py" in script
        assert "from AlphaEdit.compute_ks" in script
        assert "_patched_apply" in script
        assert "_patched_get_cov" in script

    def test_contains_measurement_code(self):
        script = build_coupling_script(
            seed=42,
            cuda_device="0",
            model_name="test-model",
            hparams_fname="Test.json",
            ds_name="mcf",
            dataset_size_limit=10,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/test.json",
            output_jsonl="/tmp/test.jsonl",
        )
        assert "_coupling_measure" in script
        assert "_coupling_layer_data" in script
        assert "projection_loss" in script
        assert "resid_norm" in script
        assert "projected_rhs_norm" in script

    def test_contains_dataset_override(self):
        script = build_coupling_script(
            seed=42,
            cuda_device="0",
            model_name="test-model",
            hparams_fname="Test.json",
            ds_name="mcf",
            dataset_size_limit=10,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/my_coupling.json",
            output_jsonl="/tmp/test.jsonl",
        )
        assert "/tmp/my_coupling.json" in script
        assert "ds.data = _coupling_data" in script

    def test_sys_argv_correct(self):
        script = build_coupling_script(
            seed=42,
            cuda_device="0",
            model_name="my-model",
            hparams_fname="MyHparams.json",
            ds_name="mcf",
            dataset_size_limit=500,
            downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/test.json",
            output_jsonl="/tmp/test.jsonl",
        )
        assert "--alg_name=AlphaEdit" in script
        assert "--model_name=my-model" in script
        assert "--hparams_fname=MyHparams.json" in script
        assert "--num_edits=1" in script
        assert "--dataset_size_limit=500" in script

    def test_conserve_memory_flag(self):
        script_mem = build_coupling_script(
            seed=42, cuda_device="0", model_name="m", hparams_fname="h",
            ds_name="mcf", dataset_size_limit=10, downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/t.json", output_jsonl="/tmp/t.jsonl",
        )
        assert "--conserve_memory" in script_mem

        script_no_mem = build_coupling_script(
            seed=42, cuda_device="0", model_name="m", hparams_fname="h",
            ds_name="mcf", dataset_size_limit=10, downstream_eval_steps=0,
            conserve_memory=False,
            coupling_dataset_path="/tmp/t.json", output_jsonl="/tmp/t.jsonl",
        )
        assert "--conserve_memory" not in script_no_mem


# --- Anchor Validation ---


@pytest.mark.skipif(not HAS_SUBMODULE, reason="AlphaEdit submodule not present")
class TestAnchorsExistInSource:
    def test_evaluate_py_anchors(self):
        eval_source = (ALPHAEDIT_ROOT / "experiments" / "evaluate.py").read_text()
        assert CUDA_PATCH_TARGET in eval_source
        assert SHUFFLE_ANCHOR in eval_source
        assert PRE_EDIT_ANCHOR in eval_source
        assert POST_EDIT_ANCHOR in eval_source
        assert ALGO_IMPORT_ANCHOR in eval_source

    def test_alphaedit_main_anchors(self):
        algo_source = (ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert RESID_ANCHOR in algo_source
        assert UPD_ANCHOR in algo_source

    def test_validate_anchors_passes(self):
        # Should not raise
        validate_anchors()


@pytest.mark.skipif(not HAS_SUBMODULE, reason="AlphaEdit submodule not present")
class TestRelativeImportReplacement:
    def test_relative_imports_exist(self):
        """Verify the relative imports we plan to replace actually exist."""
        algo_source = (ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert "from .compute_ks" in algo_source
        assert "from .compute_z" in algo_source
        assert "from .AlphaEdit_hparams" in algo_source

    def test_replacement_produces_valid_python(self):
        """After replacing relative imports, source should still parse."""
        algo_source = (ALPHAEDIT_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        algo_source = algo_source.replace("from .compute_ks", "from AlphaEdit.compute_ks")
        algo_source = algo_source.replace("from .compute_z", "from AlphaEdit.compute_z")
        algo_source = algo_source.replace("from .AlphaEdit_hparams", "from AlphaEdit.AlphaEdit_hparams")
        # Should parse without error
        ast.parse(algo_source)


# --- Exec Globals ---


class TestExecGlobals:
    def test_script_passes_required_globals(self):
        """Generated script must pass key functions to evaluate.py's exec."""
        script = build_coupling_script(
            seed=42, cuda_device="0", model_name="m", hparams_fname="h",
            ds_name="mcf", dataset_size_limit=10, downstream_eval_steps=0,
            conserve_memory=True,
            coupling_dataset_path="/tmp/t.json", output_jsonl="/tmp/t.jsonl",
        )
        # The exec call for evaluate.py should include these globals
        assert '"apply_AlphaEdit_to_model": _patched_apply' in script
        assert '"get_cov": _patched_get_cov' in script
        assert '"_coupling_output": _coupling_output' in script
        assert '"_coupling_metadata": _coupling_metadata' in script
        assert '"_coupling_measure": True' in script
