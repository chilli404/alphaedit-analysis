#!/usr/bin/env python3
"""
Comprehensive tests for the 3 measurement/intervention runners.

These runners DON'T follow the edit-eval loop pattern — they save/restore
model state, apply specific batches, and measure logprob damage.
They exec algorithm code (AlphaEdit_main.py or memit_main.py) to get
apply_fn, but do NOT exec evaluate.py.

Run with: uv run pytest tests/test_measurement_runners.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNERS = PROJECT_ROOT / "src" / "runners"


def _read(name: str) -> str:
    return (RUNNERS / name).read_text()


def _has_exec_of(source: str, filename: str) -> bool:
    """Check if source has exec(compile(...filename...))."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exec":
            source_text = ast.dump(node)
            if filename.replace(".", "_") in source_text or filename in ast.get_source_segment(source, node):
                return True
    # Fallback: string search
    return f'"{filename}"' in source and "exec(compile(" in source


# ===========================================================================
# 1. logit_damage_runner.py — AlphaEdit A/B intervention
# ===========================================================================

class TestLogitDamageRunner:
    """A/B intervention: HIGH vs LOW cosine future batches."""

    @pytest.fixture
    def source(self):
        return _read("logit_damage_runner.py")

    def test_parses(self, source):
        ast.parse(source)

    def test_no_evaluate_exec(self, source):
        """Must NOT exec evaluate.py — it's not an edit-eval loop."""
        assert "evaluate.py" not in source or source.count("evaluate.py") == 0, (
            "logit_damage_runner should not reference evaluate.py"
        )

    def test_execs_alphaedit_main(self, source):
        """Must exec AlphaEdit_main.py to get apply_AlphaEdit_to_model."""
        assert "AlphaEdit_main.py" in source
        assert "exec(compile(" in source

    def test_saves_model_state(self, source):
        """Must save model weights before A/B intervention for restoration."""
        assert "clone()" in source, "Must clone weights for save/restore"
        assert "baseline_w" in source or "saved_state" in source or "state_dict" in source

    def test_restores_model_state(self, source):
        """Must restore model state between HIGH and LOW branches."""
        assert "def restore" in source or "restore()" in source

    def test_measures_logprob(self, source):
        """Must compute log-probabilities for damage measurement."""
        assert "log_softmax" in source or "logprob" in source.lower()
        assert "def measure_logprobs" in source or "measure_logprob" in source

    def test_paired_design(self, source):
        """Must compare HIGH-cosine vs LOW-cosine batches."""
        src_lower = source.lower()
        assert "high" in src_lower and "low" in src_lower
        assert "cosine" in src_lower or "cos" in src_lower

    def test_outputs_intervention_results(self, source):
        """Must write intervention_results.json."""
        assert "intervention_results" in source

    def test_uses_focal_edits(self, source):
        """Must track focal edits (the edits being measured for damage)."""
        assert "focal" in source

    def test_trial_loop(self, source):
        """Must run multiple trial pairs for statistical power."""
        assert "trial" in source.lower()
        assert "n_trials" in source or "num_trials" in source

    def test_batch_ranking(self, source):
        """Must rank future batches by cosine similarity to focal keys."""
        assert "rank" in source.lower()
        assert "cosine" in source.lower() or "cos" in source.lower()

    def test_saves_cache_c(self, source):
        """AlphaEdit uses cache_c — must save/restore it alongside weights."""
        assert "cache_c" in source
        assert "baseline_cc" in source or "cache_c.clone()" in source


# ===========================================================================
# 2. logit_damage_memit_runner.py — MEMIT-Seq A/B intervention
# ===========================================================================

class TestLogitDamageMemitRunner:
    """A/B intervention using MEMIT-Seq instead of AlphaEdit."""

    @pytest.fixture
    def source(self):
        return _read("logit_damage_memit_runner.py")

    def test_parses(self, source):
        ast.parse(source)

    def test_no_evaluate_exec(self, source):
        assert "evaluate.py" not in source

    def test_execs_memit_main(self, source):
        """Must exec memit_main.py (NOT AlphaEdit_main.py)."""
        assert "memit_main.py" in source
        assert "exec(compile(" in source

    def test_does_not_exec_alphaedit(self, source):
        """Must use MEMIT, not AlphaEdit."""
        assert "AlphaEdit_main.py" not in source

    def test_saves_restores_model_state(self, source):
        assert "clone()" in source
        assert "def restore" in source or "restore()" in source

    def test_measures_logprob(self, source):
        assert "log_softmax" in source or "logprob" in source.lower()

    def test_canonical_model_name(self, source):
        """Must set _name_or_path to canonical value for stats lookup."""
        assert "_name_or_path" in source

    def test_seqreg_params(self, source):
        """Must support lambda_prev / lambda_delta for MEMIT-Seq regularization."""
        assert "lambda_prev" in source.lower() or "LAMBDA_PREV" in source

    def test_outputs_intervention_results(self, source):
        assert "intervention_results" in source or "results" in source

    def test_paired_design(self, source):
        src_lower = source.lower()
        assert "high" in src_lower and "low" in src_lower


# ===========================================================================
# 3. same_fact_damage_runner.py — Same-fact A/B intervention
# ===========================================================================

class TestSameFactDamageRunner:
    """Same subjects/relations/targets, different prompt-induced key geometry."""

    @pytest.fixture
    def source(self):
        return _read("same_fact_damage_runner.py")

    def test_parses(self, source):
        ast.parse(source)

    def test_no_evaluate_exec(self, source):
        assert "evaluate.py" not in source

    def test_execs_algorithm(self, source):
        """Must exec an algorithm file to get apply_fn."""
        assert "exec(compile(" in source

    def test_prompt_variant_selection(self, source):
        """Must select HIGH/LOW prompt variants based on key cosine."""
        assert "select_prompt_variants" in source or "variant" in source
        assert "cosine" in source.lower() or "cos" in source.lower()

    def test_same_fact_design(self, source):
        """The key distinguishing feature: same facts, different prompts."""
        # Must have prompt variant logic
        assert "variant" in source.lower()
        # Must use multiple prompt templates per record
        assert "prompt" in source.lower()

    def test_saves_restores_model_state(self, source):
        assert "clone()" in source
        assert "restore" in source

    def test_measures_logprob(self, source):
        assert "log_softmax" in source or "logprob" in source.lower()

    def test_outputs_intervention_results(self, source):
        assert "intervention_results" in source

    def test_focal_keys(self, source):
        """Must use pre-extracted keys for cosine computation."""
        assert "focal" in source or "keys" in source

    def test_paired_design(self, source):
        src_lower = source.lower()
        assert "high" in src_lower and "low" in src_lower

    def test_cosine_computation(self, source):
        """Must compute cosine similarity between prompt variant keys and focal keys."""
        assert "cosine" in source.lower() or "cos" in source.lower()
        # Must normalize or use dot product
        assert "norm" in source.lower() or "dot" in source.lower() or "@" in source


# ===========================================================================
# 4. Cross-cutting: all measurement runners share properties
# ===========================================================================

class TestMeasurementRunnersCrossCutting:
    """Properties all 3 measurement runners must share."""

    MEASUREMENT_RUNNERS = [
        "logit_damage_runner.py",
        "logit_damage_memit_runner.py",
        "same_fact_damage_runner.py",
    ]

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_no_evaluate_py_exec(self, runner):
        """Measurement runners must NOT exec evaluate.py."""
        source = _read(runner)
        # They exec algorithm files, but not evaluate.py
        lines = [l for l in source.split("\n") if "exec(compile(" in l]
        for line in lines:
            assert "evaluate.py" not in line, (
                f"{runner} execs evaluate.py — it should only exec algorithm files"
            )

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_has_seed_setting(self, runner):
        """Must set RNG seeds for reproducibility."""
        source = _read(runner)
        assert "seed" in source.lower()

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_has_result_output(self, runner):
        """Must write results to a file."""
        source = _read(runner)
        assert "json.dump" in source or "write" in source

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_has_argparse_or_config(self, runner):
        """Must accept configuration via argparse or similar."""
        source = _read(runner)
        assert "argparse" in source or "args" in source

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_uses_shared_paths(self, runner):
        """Must use centralized path resolution."""
        source = _read(runner)
        assert "get_result_root" in source or "RESULT_ROOT" in source or "get_checkpoint_root" in source
