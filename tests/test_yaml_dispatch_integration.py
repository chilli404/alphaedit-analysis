#!/usr/bin/env python3
"""Integration tests for SkyPilot YAML dispatch correctness.

These tests parse the actual YAML and simulate shell variable expansion
for every dispatch path, catching bugs like:
- Global defaults leaking into unrelated dispatches (ORDERING=clustered → v2_eval)
- Missing required env vars not caught until runtime
- Silent variable shadowing between dispatch paths

No GPU needed — pure parsing and simulation.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"


def _load_yaml_run_block():
    """Extract the run: block from the YAML."""
    source = YAML_PATH.read_text()
    run_start = source.find("run: |")
    if run_start < 0:
        run_start = source.find("run: |")
    return source[run_start:]


def _extract_global_defaults(run_block):
    """Parse all export VAR="${VAR:-default}" lines into a dict."""
    defaults = {}
    for match in re.finditer(r'export\s+(\w+)="\$\{\1:-([^}]*)\}"', run_block):
        var, default = match.groups()
        defaults[var] = default
    return defaults


def _extract_dispatch_block(run_block, experiment_name):
    """Extract the code block for a specific EXPERIMENT_NAME dispatch."""
    # Find the if/elif for this experiment
    patterns = [
        f'"{experiment_name}"',
        f'{experiment_name}*',
        f'"{experiment_name}"',
    ]
    for pat in patterns:
        idx = run_block.find(pat)
        if idx > 0:
            # Find the start of this branch (go back to if/elif)
            line_start = run_block.rfind("\n", 0, idx)
            # Find the end (next elif/else/fi)
            remaining = run_block[idx:]
            end_match = re.search(r'\n\s*(elif|else|fi)\b', remaining)
            if end_match:
                return remaining[:end_match.start()]
            return remaining[:500]
    return None


def _simulate_env(user_env, global_defaults):
    """Simulate what env vars a dispatch sees after global defaults are applied.

    In bash: export VAR="${VAR:-default}" means:
    - If VAR is set (even to empty), keep it
    - If VAR is unset, set to default

    user_env: dict of vars explicitly passed via --env
    global_defaults: dict from _extract_global_defaults
    """
    result = {}
    for var, default in global_defaults.items():
        if var in user_env:
            result[var] = user_env[var]
        elif default:
            result[var] = default
        else:
            result[var] = ""
    # User vars not in defaults pass through
    for var, val in user_env.items():
        result[var] = val
    return result


# ============================================================================
# Test the global defaults themselves
# ============================================================================

class TestGlobalDefaults:
    """Verify global defaults don't cause unintended behavior."""

    def test_ordering_does_not_default_to_clustered(self):
        """ORDERING must default to empty, not 'clustered'."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("ORDERING", "") != "clustered", \
            "ORDERING=clustered as default silently pollutes v2_eval and other dispatches"

    def test_ordering_default_is_empty(self):
        """ORDERING default must be empty string."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("ORDERING", "MISSING") == "", \
            f"ORDERING default should be empty, got: '{defaults.get('ORDERING')}'"

    def test_no_default_leaks_ordering_into_eval(self):
        """Simulate: launch v2_eval WITHOUT passing ORDERING — must get empty."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        env = _simulate_env({"EXPERIMENT_NAME": "v2_eval", "SEED": "42", "ALG_NAME": "AlphaEdit"}, defaults)
        assert env.get("ORDERING", "") == "", \
            f"v2_eval sees ORDERING='{env.get('ORDERING')}' without explicit --env ORDERING. " \
            f"This would load the wrong dataset."

    def test_ordering_passes_through_when_set(self):
        """Simulate: launch v2_eval WITH ORDERING=fb_high_exposure — must see it."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        env = _simulate_env({"EXPERIMENT_NAME": "v2_eval", "ORDERING": "fb_high_exposure"}, defaults)
        assert env["ORDERING"] == "fb_high_exposure"


# ============================================================================
# Test each dispatch path
# ============================================================================

class TestDispatchPaths:
    """Verify every dispatch path gets the correct env vars."""

    # Paper replications: must NOT receive ORDERING
    @pytest.mark.parametrize("experiment", [
        "evoedit_paper_replication",
        "nse_paper_replication",
        "revive_paper_replication",
        "rect_aligned_paper_replication",
    ])
    def test_paper_replication_no_ordering(self, experiment):
        """Paper replications must not see ORDERING — they use default MCF."""
        block = _extract_dispatch_block(_load_yaml_run_block(), experiment)
        assert block is not None, f"Dispatch for {experiment} not found"
        assert "ORDERING" not in block, \
            f"{experiment} must not reference ORDERING — paper replications use default MCF first-10K"

    # Baseline dispatches: must REQUIRE ORDERING
    @pytest.mark.parametrize("experiment", [
        "generic_baseline",
        "evoedit_baseline",
        "nse_baseline",
        "revive_baseline",
    ])
    def test_baseline_requires_ordering(self, experiment):
        """Baseline runs must require explicit ORDERING."""
        run_block = _load_yaml_run_block()
        block = _extract_dispatch_block(run_block, experiment)
        assert block is not None, f"Dispatch for {experiment} not found"
        assert "ORDERING:?" in block.replace('"', '').replace("'", ""), \
            f"{experiment} must require ORDERING with ${{ORDERING:?...}} — " \
            f"silent defaults cause wrong dataset"

    def test_v2_eval_ordering_is_optional(self):
        """v2_eval must treat ORDERING as optional."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        assert block is not None
        assert "${ORDERING:?" not in block, \
            "v2_eval must not require ORDERING — paper replication evals have none"
        assert "${ORDERING:+" in block, \
            "v2_eval should use ${ORDERING:+...} to pass only when set"

    # Simulate baseline launch without ORDERING — must error
    @pytest.mark.parametrize("experiment", [
        "generic_baseline",
        "evoedit_baseline",
        "nse_baseline",
        "revive_baseline",
    ])
    def test_baseline_without_ordering_would_error(self, experiment):
        """Launching a baseline without ORDERING must produce an error."""
        run_block = _load_yaml_run_block()
        block = _extract_dispatch_block(run_block, experiment)
        assert block is not None
        # The block must contain ${ORDERING:?...} which expands to an error when empty
        assert re.search(r'ORDERING:\?', block.replace('"', '')), \
            f"{experiment} must error when ORDERING is not set"

    # Simulate v2_eval launch: no ORDERING → no --ordering flag
    def test_v2_eval_without_ordering_no_flag(self):
        """v2_eval without ORDERING must NOT pass --ordering to the python script."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        assert block is not None
        # ${ORDERING:+--ordering "$ORDERING"} expands to NOTHING when ORDERING is empty
        # Verify this pattern exists
        assert re.search(r'\$\{ORDERING:\+.*ordering', block), \
            "v2_eval must use ${ORDERING:+--ordering ...} pattern"

    # Simulate v2_eval launch: WITH ORDERING → --ordering flag present
    def test_v2_eval_with_ordering_passes_flag(self):
        """v2_eval with ORDERING=fb_high must pass --ordering fb_high."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        assert block is not None
        assert "--ordering" in block


# ============================================================================
# End-to-end: simulate actual shell expansion
# ============================================================================

class TestShellExpansionSimulation:
    """Simulate the actual bash variable expansion for launch commands."""

    def _expand(self, template, env):
        """Simulate bash ${VAR:+value} and ${VAR:?error} expansion."""
        result = template
        for var, val in env.items():
            # ${VAR:+replacement} → replacement if VAR is non-empty, else nothing
            pattern = re.compile(r'\$\{' + var + r':\+([^}]*)\}')
            if val:
                result = pattern.sub(r'\1', result)
            else:
                result = pattern.sub('', result)
            # ${VAR:?error} → val if non-empty, else ERROR
            pattern = re.compile(r'\$\{' + var + r':\?[^}]*\}')
            if val:
                result = pattern.sub(val, result)
            else:
                result = pattern.sub('__ERROR__', result)
            # ${VAR:-default} → val if set, else default
            pattern = re.compile(r'\$\{' + var + r':-([^}]*)\}')
            if val:
                result = pattern.sub(val, result)
            else:
                result = pattern.sub(r'\1', result)
            # Simple $VAR
            result = result.replace(f'"${var}"', f'"{val}"')
            result = result.replace(f'${var}', val)
        return result

    def test_v2_eval_no_ordering_produces_clean_command(self):
        """v2_eval without ORDERING must produce command without --ordering."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        env = {"ORDERING": "", "SEED": "42", "ALG_NAME": "AlphaEdit",
               "MODEL_NAME": "meta-llama/Meta-Llama-3-8B-Instruct",
               "NUM_EDITS": "100", "CHECKPOINT_DIR": "/path/to/ckpt",
               "CHECKPOINTS": "99"}
        expanded = self._expand(block, env)
        assert "--ordering" not in expanded, \
            f"v2_eval without ORDERING must not have --ordering in command.\n" \
            f"Expanded: {expanded[:200]}"

    def test_v2_eval_with_ordering_includes_flag(self):
        """v2_eval with ORDERING must include --ordering in command."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        env = {"ORDERING": "fb_high_exposure", "SEED": "42", "ALG_NAME": "AlphaEdit",
               "MODEL_NAME": "meta-llama/Meta-Llama-3-8B-Instruct",
               "NUM_EDITS": "100"}
        expanded = self._expand(block, env)
        assert "--ordering" in expanded
        assert "fb_high_exposure" in expanded

    def test_baseline_no_ordering_errors(self):
        """Baseline without ORDERING must expand to __ERROR__."""
        block = _extract_dispatch_block(_load_yaml_run_block(), "evoedit_baseline")
        if block is None:
            pytest.skip("evoedit_baseline dispatch not found")
        env = {"ORDERING": "", "SEED": "42"}
        expanded = self._expand(block, env)
        assert "__ERROR__" in expanded, \
            "Baseline without ORDERING must error, not silently proceed"

    def test_full_simulation_paper_replication(self):
        """Full simulation: EXPERIMENT_NAME=v2_eval, no ORDERING, SEED=42."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        user_env = {"EXPERIMENT_NAME": "v2_eval", "SEED": "42",
                     "ALG_NAME": "EvoEdit", "CHECKPOINTS": "99"}
        final_env = _simulate_env(user_env, defaults)

        # ORDERING must be empty after global defaults
        assert final_env.get("ORDERING", "") == "", \
            f"Paper replication eval: ORDERING should be empty, got '{final_env.get('ORDERING')}'"

        # Expand the v2_eval block with final env
        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        expanded = self._expand(block, final_env)
        assert "--ordering" not in expanded, \
            f"Paper replication eval must not have --ordering.\nExpanded: {expanded[:300]}"

    def test_full_simulation_ordering_eval(self):
        """Full simulation: v2_eval WITH ORDERING=fb_high_exposure."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        user_env = {"EXPERIMENT_NAME": "v2_eval", "SEED": "42",
                     "ALG_NAME": "AlphaEdit", "ORDERING": "fb_high_exposure",
                     "CHECKPOINTS": "99"}
        final_env = _simulate_env(user_env, defaults)

        assert final_env["ORDERING"] == "fb_high_exposure"

        block = _extract_dispatch_block(_load_yaml_run_block(), "v2_eval")
        expanded = self._expand(block, final_env)
        assert "--ordering" in expanded
        assert "fb_high_exposure" in expanded


# ============================================================================
# Audit-driven tests: dangerous defaults caught by agent audit
# ============================================================================

class TestDangerousDefaults:
    """Tests for every global default that could silently corrupt results."""

    def test_ordering_empty_default(self):
        """ORDERING must default to empty — 'clustered' caused wrong eval dataset."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("ORDERING", "") == "", \
            f"ORDERING must default to empty, got '{defaults.get('ORDERING')}'"

    def test_alg_name_empty_default(self):
        """ALG_NAME must default to empty — 'both' leaks into v2_eval."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("ALG_NAME", "") == "", \
            f"ALG_NAME must default to empty, got '{defaults.get('ALG_NAME')}'"

    def test_model_name_default_is_llama(self):
        """MODEL_NAME default is Llama (acceptable — YAML case statement overrides for gptj/qwen)."""
        run_block = _load_yaml_run_block()
        # The case statement should override for gptj and qwen
        assert "gptj" in run_block.lower() or "gpt-j" in run_block.lower()
        assert "qwen" in run_block.lower()

    def test_lambda_prev_empty_default(self):
        """LAMBDA_PREV must default to empty — scripts define their own semantics."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("LAMBDA_PREV", "") == "", \
            f"LAMBDA_PREV must default to empty, got '{defaults.get('LAMBDA_PREV')}'"

    def test_lambda_delta_empty_default(self):
        """LAMBDA_DELTA must default to empty."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        assert defaults.get("LAMBDA_DELTA", "") == "", \
            f"LAMBDA_DELTA must default to empty, got '{defaults.get('LAMBDA_DELTA')}'"

    @pytest.mark.parametrize("var,dangerous_values", [
        ("ORDERING", ["clustered", "dispersed", "fb_high_exposure", "fb_low_exposure"]),
        ("ALG_NAME", ["both", "AlphaEdit", "MEMIT"]),
        ("LAMBDA_PREV", ["1.0", "0.0"]),
        ("LAMBDA_DELTA", ["1.0", "0.0"]),
        ("BASE_ALG", ["MEMIT", "AlphaEdit", "NSE"]),
    ])
    def test_no_dangerous_defaults(self, var, dangerous_values):
        """Variables that change algorithm behavior must not have non-empty defaults."""
        defaults = _extract_global_defaults(_load_yaml_run_block())
        actual = defaults.get(var, "")
        assert actual not in dangerous_values, \
            f"{var} defaults to '{actual}' which changes algorithm behavior. Must be empty."


class TestScriptDefaults:
    """Tests for shell script defaults that could cause silent errors."""

    def test_matched_ordering_requires_ordering(self):
        """run_matched_ordering.sh must require ORDERING, not default to clustered."""
        source = (PROJECT_ROOT / "scripts" / "run_matched_ordering.sh").read_text()
        assert "ORDERING:-clustered" not in source, \
            "run_matched_ordering.sh must not default ORDERING to clustered"
        assert "ORDERING:?" in source.replace('"', '').replace("'", ""), \
            "run_matched_ordering.sh must require ORDERING explicitly"

    def test_failure_curve_derives_hparams_from_model(self):
        """run_failure_curve_checkpointed.sh must derive HPARAMS from MODEL_NAME."""
        source = (PROJECT_ROOT / "scripts" / "run_failure_curve_checkpointed.sh").read_text()
        # Must have a case statement mapping model → hparams
        assert "case" in source and "gpt-j" in source and "HPARAMS_FNAME" in source, \
            "failure_curve script must derive HPARAMS from MODEL_NAME via case statement"
        # Must NOT hardcode Llama3-8B.json as the only default
        assert 'HPARAMS_FNAME:-Llama3-8B.json' not in source.replace('"', '').replace("'", ""), \
            "Must not hardcode Llama hparams — derive from MODEL_NAME"

    @pytest.mark.parametrize("script", [
        "run_revive_paper_replication.sh",
        "run_nse_paper_replication.sh",
        "run_rect_aligned_paper_replication.sh",
        "run_evoedit_paper_replication.sh",
        "run_evoedit_baseline.sh",
        "run_nse_baseline.sh",
        "run_revive_baseline.sh",
        "run_polykernel_seqreg.sh",
    ])
    def test_script_derives_hparams_from_model(self, script):
        """Every run script must derive HPARAMS from MODEL_NAME, not hardcode."""
        path = PROJECT_ROOT / "scripts" / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        source = path.read_text()
        assert "case" in source and "HPARAMS" in source, \
            f"{script} must derive HPARAMS_FNAME from MODEL_NAME via case statement"

    def test_paper_replication_results_go_to_failure_curve(self):
        """Paper replication scripts must write to failure_curve_checkpointed, not paper_replications."""
        for script in ["run_evoedit_paper_replication.sh", "run_nse_paper_replication.sh",
                       "run_rect_aligned_paper_replication.sh"]:
            path = PROJECT_ROOT / "scripts" / script
            if not path.exists():
                continue
            source = path.read_text()
            assert "paper_replications" not in source, \
                f"{script} must write to failure_curve_checkpointed, not paper_replications"

    # ---- generic_baseline.sh specific tests ----

    def test_generic_baseline_requires_ordering(self):
        """run_generic_baseline.sh must require ORDERING, not default to fb_random0."""
        path = PROJECT_ROOT / "scripts" / "run_generic_baseline.sh"
        if not path.exists():
            pytest.skip("run_generic_baseline.sh not found")
        source = path.read_text()
        assert "ORDERING:-fb_random0" not in source, \
            "run_generic_baseline.sh must not default ORDERING to fb_random0"
        assert "ORDERING:-clustered" not in source, \
            "run_generic_baseline.sh must not default ORDERING to clustered"

    def test_generic_baseline_requires_alg_name(self):
        """run_generic_baseline.sh must require ALG_NAME."""
        path = PROJECT_ROOT / "scripts" / "run_generic_baseline.sh"
        if not path.exists():
            pytest.skip("run_generic_baseline.sh not found")
        source = path.read_text()
        assert "ALG_NAME:?" in source.replace('"', '').replace("'", ""), \
            "run_generic_baseline.sh must require ALG_NAME"

    def test_generic_baseline_no_eval_during_editing(self):
        """run_generic_baseline.sh must skip mega-batch eval during editing."""
        path = PROJECT_ROOT / "scripts" / "run_generic_baseline.sh"
        if not path.exists():
            pytest.skip("run_generic_baseline.sh not found")
        source = path.read_text()
        assert "SKIP_MEGA_BATCH_EVAL" in source, \
            "run_generic_baseline.sh must set SKIP_MEGA_BATCH_EVAL=1 during editing — " \
            "mega-batch eval at every batch causes OOM and wastes hours"

    @pytest.mark.parametrize("script", [
        "run_evoedit_baseline.sh",
        "run_nse_baseline.sh",
    ])
    def test_baseline_no_eval_during_editing(self, script):
        """Baseline scripts must skip mega-batch eval during editing."""
        path = PROJECT_ROOT / "scripts" / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        source = path.read_text()
        assert "SKIP_MEGA_BATCH_EVAL" in source, \
            f"{script} must set SKIP_MEGA_BATCH_EVAL=1 during editing"

    @pytest.mark.parametrize("script", [
        "run_generic_baseline.sh",
        "run_evoedit_baseline.sh",
        "run_nse_baseline.sh",
    ])
    def test_baseline_no_ordering_default(self, script):
        """ALL baseline scripts must require explicit ORDERING."""
        path = PROJECT_ROOT / "scripts" / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        source = path.read_text()
        # No default ordering values
        for bad_default in ["ORDERING:-fb_random0", "ORDERING:-clustered",
                            "ORDERING:-fb_high", "ORDERING:-key_"]:
            assert bad_default not in source, \
                f"{script} has dangerous ORDERING default: {bad_default}"
