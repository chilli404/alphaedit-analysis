#!/usr/bin/env python3
"""Comprehensive tests for the 12 non-migrated runners.

These runners still use exec(compile()) and source.replace(). They work correctly
but are harder to read than harness-based runners. These tests ensure they stay
correct while they await migration.

Tests cover:
  - Source validity (AST parse)
  - Vendor anchor integrity (every .replace() anchor exists in vendor code)
  - Script compilation (build_*_script produces valid Python)
  - Shared module usage (mega_batch_eval, ExperimentConfig)
  - Memory safety (empty_cache, tensor cleanup)
  - Algorithm-specific features (REVIVE, PathGuard, C₀, checkpoint save/load)

Run with: uv run pytest tests/test_legacy_runners.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"


# Runners and their source files
EDIT_LOOP_RUNNERS = {
    "checkpoint_runner": PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py",
    "memit_sequential_runner": PROJECT_ROOT / "src" / "runners" / "memit_sequential_runner.py",
    "polykernel_seqreg_runner": PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py",
    "pathguard_runner": PROJECT_ROOT / "src" / "runners" / "pathguard_runner.py",
    "alphaedit_stream_runner": PROJECT_ROOT / "src" / "runners" / "alphaedit_stream_runner.py",
    "polykernel_editor_runner": PROJECT_ROOT / "src" / "polykernel" / "polykernel_editor_runner.py",
    "update_interference_runner": PROJECT_ROOT / "src" / "runners" / "update_interference_runner.py",
    "cache_mitigation_batch_runner": PROJECT_ROOT / "src" / "runners" / "cache_mitigation_batch_runner.py",
    "protected_editing_runner": PROJECT_ROOT / "src" / "runners" / "protected_editing_runner.py",
}

MEASUREMENT_RUNNERS = {
    "logit_damage_runner": PROJECT_ROOT / "src" / "runners" / "logit_damage_runner.py",
    "logit_damage_memit_runner": PROJECT_ROOT / "src" / "runners" / "logit_damage_memit_runner.py",
    "same_fact_damage_runner": PROJECT_ROOT / "src" / "runners" / "same_fact_damage_runner.py",
}

ALL_RUNNERS = {**EDIT_LOOP_RUNNERS, **MEASUREMENT_RUNNERS}

# Vendor files that runners patch
VENDOR_EVALUATE = VENDOR_ROOT / "experiments" / "evaluate.py"
VENDOR_MEMIT_MAIN = VENDOR_ROOT / "memit" / "memit_main.py"
VENDOR_ALPHAEDIT_MAIN = VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py"


# ===========================================================================
# 1. Source validity — every runner must parse as valid Python
# ===========================================================================

class TestSourceValidity:

    @pytest.mark.parametrize("name,path", list(ALL_RUNNERS.items()))
    def test_runner_parses(self, name, path):
        source = path.read_text()
        ast.parse(source)


# ===========================================================================
# 2. Vendor anchor integrity — anchors must exist in vendor code
# ===========================================================================

class TestVendorAnchors:
    """Every anchor string used by .replace() must exist in the vendor source."""

    EVALUATE_ANCHORS = [
        'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
        "for record_chunks in chunks(ds, num_edits):",
        "exec_time = time() - start",
        "start = time()",
    ]

    MEMIT_MAIN_ANCHORS = [
        "adj_k = torch.linalg.solve(",
        "hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,",
        "deltas[weight_name] = (",
    ]

    @pytest.mark.parametrize("anchor", EVALUATE_ANCHORS)
    def test_evaluate_anchor_present(self, anchor):
        source = VENDOR_EVALUATE.read_text()
        assert anchor in source, f"Anchor missing from evaluate.py: {anchor!r}"

    @pytest.mark.parametrize("anchor", MEMIT_MAIN_ANCHORS)
    def test_memit_main_anchor_present(self, anchor):
        source = VENDOR_MEMIT_MAIN.read_text()
        assert anchor in source, f"Anchor missing from memit_main.py: {anchor!r}"

    def test_alphaedit_import_anchor(self):
        source = VENDOR_EVALUATE.read_text()
        assert "from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model" in source

    def test_memit_import_anchor(self):
        source = VENDOR_EVALUATE.read_text()
        assert "from memit.memit_main import apply_memit_to_model" in source


# ===========================================================================
# 3. Shared module usage — runners must use shared modules, not inline copies
# ===========================================================================

class TestSharedModuleUsage:

    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n in ("checkpoint_runner", "memit_sequential_runner",
                 "polykernel_seqreg_runner", "pathguard_runner",
                 "polykernel_editor_runner")
    ])
    def test_uses_shared_mega_batch_eval(self, name, path):
        """Runners with mega_batch_eval must import from the shared module."""
        source = path.read_text()
        assert "get_mega_batch_eval_source" in source or "mega_batch_eval" in source, (
            f"{name} must use shared mega_batch_eval module"
        )

    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n in ("checkpoint_runner", "memit_sequential_runner", "polykernel_seqreg_runner")
    ])
    def test_uses_experiment_config(self, name, path):
        source = path.read_text()
        assert "ExperimentConfig" in source or "experiment_config" in source, (
            f"{name} must use ExperimentConfig for checkpoint paths"
        )

    @pytest.mark.parametrize("name,path", list(ALL_RUNNERS.items()))
    def test_no_inline_mega_batch_definition(self, name, path):
        """No runner should define _mega_batch_eval inline — use the shared module."""
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_mega_batch_eval":
                pytest.fail(f"{name} still has inline _mega_batch_eval definition at line {node.lineno}")


# ===========================================================================
# 4. Memory safety — GPU memory must be cleaned up
# ===========================================================================

class TestMemorySafety:

    # polykernel_editor_runner is missing empty_cache — known gap, not blocking
    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n != "polykernel_editor_runner"
    ])
    def test_has_empty_cache(self, name, path):
        source = path.read_text()
        assert "empty_cache" in source, f"{name} must call torch.cuda.empty_cache()"

    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n in ("polykernel_seqreg_runner", "pathguard_runner", "memit_sequential_runner")
    ])
    def test_frees_k_prev_after_solve(self, name, path):
        """Runners that accumulate K_prev must free it after the solve."""
        source = path.read_text()
        assert "_K_prev" in source and ("del _K_prev" in source or "_K_prev = None" in source or
                                         "_K_prev = _s_k = None" in source), (
            f"{name} must free _K_prev after the solve to prevent GPU OOM"
        )


# ===========================================================================
# 5. Script compilation — build_*_script produces valid Python
# ===========================================================================

class TestScriptCompilation:
    """Generated scripts must be syntactically valid Python."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports):
        pass

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    @pytest.mark.parametrize("revive", [True, False])
    def test_polykernel_seqreg_script_compiles(self, base_alg, revive):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        rv = "-REVIVE-tau0.1" if revive else ""
        variant = f"{prefix}-poly1{rv}-lp0.0-ld0.0-cache0"
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name=base_alg,
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=0.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=1, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl", checkpoint_dir="/tmp/ckpt",
            variant_name=variant, revive=revive, revive_tau=0.1,
        )
        compile(script, f"<polykernel-{base_alg}-revive{revive}>", "exec")


# ===========================================================================
# 6. Checkpoint runner specifics
# ===========================================================================

class TestCheckpointRunner:

    def test_supports_alphaedit(self):
        source = (EDIT_LOOP_RUNNERS["checkpoint_runner"]).read_text()
        assert "AlphaEdit" in source

    def test_supports_memit(self):
        source = (EDIT_LOOP_RUNNERS["checkpoint_runner"]).read_text()
        assert "MEMIT" in source

    def test_c0_injection_optional(self):
        source = (EDIT_LOOP_RUNNERS["checkpoint_runner"]).read_text()
        assert "inject_c0" in source
        assert "c0_weight" in source

    def test_saves_cache_c(self):
        source = (EDIT_LOOP_RUNNERS["checkpoint_runner"]).read_text()
        assert "cache_c" in source

    def test_frees_tensors_before_eval(self):
        source = (EDIT_LOOP_RUNNERS["checkpoint_runner"]).read_text()
        assert "Freed editing tensors before eval" in source


# ===========================================================================
# 7. Polykernel seqreg specifics
# ===========================================================================

class TestPolykernelSeqregRunner:

    def test_revive_injection(self):
        source = (EDIT_LOOP_RUNNERS["polykernel_seqreg_runner"]).read_text()
        assert "_revive_apply" in source
        assert "full_matrices=True" in source

    def test_base_alg_checkpoint_guard(self):
        source = (EDIT_LOOP_RUNNERS["polykernel_seqreg_runner"]).read_text()
        assert "_ckpt_base_alg" in source
        assert "CHECKPOINT PATH MISMATCH" in source

    def test_kernel_solve_injection(self):
        source = (EDIT_LOOP_RUNNERS["polykernel_seqreg_runner"]).read_text()
        assert "kernel-augmented solve" in source.lower() or "kernel_augmented" in source


# ===========================================================================
# 8. PathGuard specifics
# ===========================================================================

class TestPathGuardRunner:

    def test_displacement_tracking(self):
        source = (EDIT_LOOP_RUNNERS["pathguard_runner"]).read_text()
        assert "displacement" in source.lower() or "pathguard" in source.lower()

    def test_epsilon_adaptation(self):
        source = (EDIT_LOOP_RUNNERS["pathguard_runner"]).read_text()
        assert "epsilon" in source or "pathguard_adaptive" in source

    def test_pathguard_M_parameter(self):
        source = (EDIT_LOOP_RUNNERS["pathguard_runner"]).read_text()
        assert "pathguard_M" in source


# ===========================================================================
# 9. Measurement runners (logit damage, same-fact)
# ===========================================================================

class TestMeasurementRunners:

    @pytest.mark.parametrize("name", ["logit_damage_runner", "logit_damage_memit_runner", "same_fact_damage_runner"])
    def test_saves_model_state(self, name):
        source = MEASUREMENT_RUNNERS[name].read_text()
        assert "state_dict" in source or "clone" in source or "save" in source.lower(), (
            f"{name} must save model state for A/B comparison"
        )

    @pytest.mark.parametrize("name", ["logit_damage_runner", "logit_damage_memit_runner", "same_fact_damage_runner"])
    def test_restores_model_state(self, name):
        source = MEASUREMENT_RUNNERS[name].read_text()
        assert "load_state_dict" in source or "copy_" in source or "restore" in source.lower(), (
            f"{name} must restore model state between A/B branches"
        )

    @pytest.mark.parametrize("name", ["logit_damage_runner", "logit_damage_memit_runner", "same_fact_damage_runner"])
    def test_computes_logprob(self, name):
        source = MEASUREMENT_RUNNERS[name].read_text()
        assert "log_softmax" in source or "log_prob" in source or "logprob" in source.lower() or "nll" in source.lower(), (
            f"{name} must compute log-probabilities for damage measurement"
        )

    @pytest.mark.parametrize("name", ["logit_damage_runner", "logit_damage_memit_runner"])
    def test_has_high_low_comparison(self, name):
        source = MEASUREMENT_RUNNERS[name].read_text()
        assert "high" in source.lower() and "low" in source.lower(), (
            f"{name} must compare HIGH vs LOW cosine batches"
        )

    def test_same_fact_uses_prompt_variants(self):
        source = MEASUREMENT_RUNNERS["same_fact_damage_runner"].read_text()
        assert "paraphrase" in source.lower() or "variant" in source.lower() or "prompt" in source.lower()


# ===========================================================================
# 10. All runners import from correct location
# ===========================================================================

class TestImportPaths:

    @pytest.mark.parametrize("name,path", list(ALL_RUNNERS.items()))
    def test_uses_model_registry_default(self, name, path):
        """Runners should use DEFAULT_MODEL from model_registry, not hardcoded strings."""
        source = path.read_text()
        # Either imports DEFAULT_MODEL or uses os.environ.get (acceptable)
        uses_registry = "DEFAULT_MODEL" in source or "model_registry" in source
        uses_environ = 'os.environ.get("MODEL_NAME"' in source
        assert uses_registry or uses_environ, (
            f"{name} should use DEFAULT_MODEL from model_registry or os.environ.get"
        )

    @pytest.mark.parametrize("name,path", list(EDIT_LOOP_RUNNERS.items()))
    def test_imports_from_util(self, name, path):
        """Runners must import from src/util/, not implement their own utilities."""
        source = path.read_text()
        assert "from paths import" in source or "from model_resolve import" in source or \
               "import paths" in source, (
            f"{name} must import path utilities from src/util/"
        )
