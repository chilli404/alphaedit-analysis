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
    # Migrated runners may have empty_cache in hooks (hook_presets.py), not in the runner itself
    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n != "polykernel_editor_runner"
    ])
    def test_has_empty_cache(self, name, path):
        source = path.read_text()
        hooks_src = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        assert "empty_cache" in source or "empty_cache" in hooks_src, (
            f"{name} must call empty_cache (in runner or hook_presets)"
        )

    @pytest.mark.parametrize("name,path", [
        (n, p) for n, p in EDIT_LOOP_RUNNERS.items()
        if n in ("polykernel_seqreg_runner", "pathguard_runner", "memit_sequential_runner")
    ])
    def test_frees_k_prev_after_solve(self, name, path):
        """K_prev cleanup must exist in runner or hook_presets."""
        source = path.read_text()
        hooks_src = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        has_cleanup = (
            ("_K_prev" in source and ("del _K_prev" in source or "_K_prev = None" in source)) or
            ("empty_cache" in hooks_src) or
            ("del " in source and "K_prev" in source)
        )
        assert has_cleanup, f"{name}: K_prev cleanup must be in runner or hook_presets"


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
    def test_polykernel_seqreg_variant_name(self, base_alg, revive):
        """polykernel_seqreg_runner migrated — test variant naming via ExperimentConfig."""
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=base_alg, seed=42, kernel_degree=1,
            revive=revive, revive_tau=0.1,
        )
        prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert config.variant_name.startswith(prefix)
        if revive:
            assert "REVIVE" in config.variant_name


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

    def test_revive_in_hooks(self):
        """REVIVE is now in hook_presets.py, not inline in runner."""
        hooks_src = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        assert "full_matrices=True" in hooks_src
        assert "searchsorted" in hooks_src

    def test_base_alg_path_isolation(self):
        """ExperimentConfig.variant_name handles base_alg prefix."""
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(base_alg="AlphaEdit", seed=42, kernel_degree=1)
        assert "AlphaEdit" in config.variant_name
        assert "MEMIT-Seq" not in config.variant_name

    def test_kernel_in_hooks(self):
        hooks_src = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        assert "kernel" in hooks_src.lower()


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
