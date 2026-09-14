#!/usr/bin/env python3
"""Integration tests for eval_matched_ordering.py correctness.

Verifies that the eval script:
1. Loads the correct dataset for each scenario (ordering stream vs default MCF)
2. Evaluates the correct number of records per checkpoint
3. Writes output to the correct path
4. Uses correct model loading (bfloat16, no hardcoded dtype)
5. Handles all argument combinations without silent data mismatches

These tests parse the eval script source and simulate its logic on CPU.
No GPU needed — they test dataset selection and path construction, not model inference.
"""
import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_matched_ordering.py"
ORDERINGS_DIR = PROJECT_ROOT / "results" / "matched_ordering" / "orderings"
MCF_PATH = PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json"


def _load_mcf_ids(n=10000):
    """Load case_ids from default MCF first N records."""
    if not MCF_PATH.exists():
        pytest.skip("MCF dataset not available")
    with open(MCF_PATH) as f:
        mcf = json.load(f)
    return [r["case_id"] for r in mcf[:n]]


def _load_stream_ids(ordering, seed=42):
    """Load case_ids from an ordering stream."""
    path = ORDERINGS_DIR / f"{ordering}_seed{seed}.json"
    if not path.exists():
        pytest.skip(f"Stream file not available: {path.name}")
    with open(path) as f:
        stream = json.load(f)
    return [r["case_id"] for r in stream]


# ============================================================================
# Test class 1: Dataset selection logic
# ============================================================================

class TestDatasetSelection:
    """Verify the eval script loads the correct dataset for every scenario."""

    def test_source_has_three_dataset_branches(self):
        """Eval must have 3 distinct paths: dataset_path, ordering stream, default MCF."""
        source = EVAL_SCRIPT.read_text()
        assert "if args.dataset_path:" in source, "Missing --dataset_path branch"
        assert "elif args.ordering:" in source, "Missing --ordering stream branch"
        assert "else:" in source, "Missing default MCF branch"

    def test_ordering_branch_loads_stream_not_mcf(self):
        """When --ordering is set, the script must NOT load multi_counterfact.json."""
        source = EVAL_SCRIPT.read_text()
        # Find the ordering branch
        ordering_start = source.find("elif args.ordering:")
        else_start = source.find("else:", ordering_start)
        ordering_section = source[ordering_start:else_start]
        assert "multi_counterfact" not in ordering_section, \
            "Ordering branch must load the stream file, NOT multi_counterfact.json"
        assert "ordering" in ordering_section and "seed" in ordering_section, \
            "Ordering branch must construct stream path from ordering name + seed"

    def test_default_branch_loads_mcf(self):
        """When no ordering and no dataset_path, must load multi_counterfact.json."""
        source = EVAL_SCRIPT.read_text()
        # Find the else branch after "elif args.ordering:"
        ordering_pos = source.find("elif args.ordering:")
        else_start = source.find("else:", ordering_pos)
        default_section = source[else_start:else_start + 500]
        assert "multi_counterfact.json" in default_section

    def test_ordering_stream_path_format(self):
        """Stream path must be orderings/{ordering}_seed{seed}.json."""
        source = EVAL_SCRIPT.read_text()
        # Check for the expected path pattern
        assert "f\"{args.ordering}_seed{args.seed}.json\"" in source or \
               "ordering}_seed{" in source.replace("args.", ""), \
            "Stream path must follow {ordering}_seed{seed}.json format"

    def test_ordering_branch_exits_on_missing_stream(self):
        """Must sys.exit if ordering stream not found — not silently fall through."""
        source = EVAL_SCRIPT.read_text()
        ordering_start = source.find("elif args.ordering:")
        else_start = source.find("else:", ordering_start)
        ordering_section = source[ordering_start:else_start]
        assert "sys.exit" in ordering_section, \
            "Must exit if ordering stream not found — silent fallback to MCF is a data bug"


# ============================================================================
# Test class 2: Record identity verification
# ============================================================================

class TestRecordIdentity:
    """Verify ordering streams contain different records than default MCF first-10K."""

    CORE_ORDERINGS = [
        "fb_high_exposure", "fb_low_exposure", "fb_random0",
        "key_clustered", "key_dispersed",
    ]

    ALL_FB_ORDERINGS = [
        "fb_high_exposure", "fb_low_exposure",
        "fb_random0", "fb_random1", "fb_random2",
    ]

    def test_streams_differ_from_default_mcf(self):
        """Ordering streams must NOT be the same as default MCF first-10K."""
        default_ids = set(_load_mcf_ids(10000))
        stream_ids = set(_load_stream_ids("fb_high_exposure", 42))
        overlap = len(default_ids & stream_ids) / len(default_ids)
        assert overlap < 0.9, \
            f"Stream has {overlap:.0%} overlap with default MCF — " \
            f"evaluating default MCF for ordering runs would check wrong records"

    @pytest.mark.parametrize("ordering", CORE_ORDERINGS)
    def test_core_ordering_stream_exists_seed42(self, ordering):
        """Each core ordering must have a stream file for seed 42."""
        path = ORDERINGS_DIR / f"{ordering}_seed42.json"
        if not path.exists():
            pytest.skip(f"Stream not available: {path.name}")
        assert path.stat().st_size > 100000, f"Stream file too small: {path}"

    @pytest.mark.parametrize("ordering", CORE_ORDERINGS)
    def test_core_ordering_has_all_seeds(self, ordering):
        """Core orderings must have streams for all 3 seeds."""
        for seed in [42, 2024, 137]:
            path = ORDERINGS_DIR / f"{ordering}_seed{seed}.json"
            assert path.exists(), f"Missing stream: {ordering}_seed{seed}.json"

    def test_all_fb_orderings_share_same_records(self):
        """All fb orderings must use the same 10K records (fixed-batch design)."""
        if not (ORDERINGS_DIR / "fb_high_exposure_seed42.json").exists():
            pytest.skip("Stream files not available")
        ref_ids = set(_load_stream_ids("fb_high_exposure", 42))
        for ordering in self.ALL_FB_ORDERINGS:
            path = ORDERINGS_DIR / f"{ordering}_seed42.json"
            if not path.exists():
                continue
            ids = set(_load_stream_ids(ordering, 42))
            assert ids == ref_ids, \
                f"{ordering} has different records than fb_high_exposure — " \
                f"fixed-batch design requires identical record sets"

    def test_fb_orderings_differ_in_order(self):
        """fb_high and fb_low must have same records in DIFFERENT order."""
        if not (ORDERINGS_DIR / "fb_high_exposure_seed42.json").exists():
            pytest.skip("Stream files not available")
        high_ids = _load_stream_ids("fb_high_exposure", 42)
        low_ids = _load_stream_ids("fb_low_exposure", 42)
        assert set(high_ids) == set(low_ids), "Same record sets"
        assert high_ids != low_ids, "Must be different ORDER"

    def test_stream_records_are_valid_mcf(self):
        """All stream records must exist in the full MCF dataset."""
        if not MCF_PATH.exists():
            pytest.skip("MCF not available")
        with open(MCF_PATH) as f:
            mcf = json.load(f)
        all_mcf_ids = set(r["case_id"] for r in mcf)
        stream_ids = set(_load_stream_ids("fb_high_exposure", 42))
        assert stream_ids.issubset(all_mcf_ids), \
            f"Stream has {len(stream_ids - all_mcf_ids)} records not in MCF"

    @pytest.mark.parametrize("ordering", CORE_ORDERINGS)
    def test_stream_has_required_fields(self, ordering):
        """Each stream record must have all fields needed for evaluation."""
        path = ORDERINGS_DIR / f"{ordering}_seed42.json"
        if not path.exists():
            pytest.skip(f"Stream not available")
        with open(path) as f:
            records = json.load(f)
        required = {"case_id", "requested_rewrite", "paraphrase_prompts", "neighborhood_prompts"}
        for field in required:
            assert field in records[0], f"Stream record missing '{field}'"
        # Check rewrite sub-fields
        rw = records[0]["requested_rewrite"]
        rw_required = {"prompt", "subject", "target_new", "target_true"}
        for field in rw_required:
            assert field in rw, f"Rewrite record missing '{field}'"


# ============================================================================
# Test class 3: Output path construction
# ============================================================================

class TestOutputPath:
    """Verify eval output goes to the correct directory."""

    def test_ordering_output_includes_ordering_subdir(self):
        """With --ordering, output must include ordering in the path."""
        source = EVAL_SCRIPT.read_text()
        assert "if ordering:" in source
        assert "ordering" in source and "out_dir" in source
        # The path should be: result_root / matched_ordering / variant / ordering / seed
        output_section = source[source.find("if ordering:"):]
        assert "variant_name" in output_section[:200]

    def test_no_ordering_output_omits_ordering(self):
        """Without ordering, output must NOT have an ordering subdir."""
        source = EVAL_SCRIPT.read_text()
        else_start = source.find("else:", source.find("if ordering:"))
        no_ordering_section = source[else_start:else_start + 200]
        assert "variant_name" in no_ordering_section
        assert "seed" in no_ordering_section

    def test_output_filename_is_v2(self):
        """Output filename must be full_eval_seed{N}_v2.json."""
        source = EVAL_SCRIPT.read_text()
        assert "full_eval_seed" in source and "_v2.json" in source


# ============================================================================
# Test class 4: Record count correctness
# ============================================================================

class TestRecordCount:
    """Verify the eval evaluates the correct number of records per checkpoint."""

    def test_records_to_eval_uses_total_edits(self):
        """records_to_eval must be all_records[:total_edits], not all_records."""
        source = EVAL_SCRIPT.read_text()
        assert "all_records[:total_edits]" in source, \
            "Must evaluate only records up to the checkpoint, not all records"

    def test_total_edits_formula(self):
        """total_edits must be (batch_idx + 1) * num_edits."""
        source = EVAL_SCRIPT.read_text()
        assert "(batch_idx + 1) * args.num_edits" in source or \
               "(batch_idx+1)*args.num_edits" in source.replace(" ", "")

    def test_stream_has_enough_records(self):
        """Ordering streams must have at least 10K records for 10K-edit eval."""
        if not (ORDERINGS_DIR / "fb_high_exposure_seed42.json").exists():
            pytest.skip("Stream not available")
        ids = _load_stream_ids("fb_high_exposure", 42)
        assert len(ids) >= 10000, \
            f"Stream has {len(ids)} records but 10K eval needs 10000"


# ============================================================================
# Test class 5: Model loading
# ============================================================================

class TestEvalModelLoading:
    """Verify the eval script loads models correctly."""

    def test_no_hardcoded_float16(self):
        """Eval script must NOT hardcode torch.float16 for model loading."""
        source = EVAL_SCRIPT.read_text()
        assert "torch.float16" not in source, \
            "Eval must not hardcode float16 — use model config (bfloat16 for Llama-3)"

    def test_no_hardcoded_float32(self):
        """Eval script must NOT hardcode torch.float32 for model loading."""
        source = EVAL_SCRIPT.read_text()
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "from_pretrained" in line and "float32" in line:
                pytest.fail(f"Line {i+1}: from_pretrained with float32: {line.strip()}")

    def test_checkpoint_weights_not_cast_to_half(self):
        """Checkpoint weights must use model dtype, NOT .half() (float16).

        The model loads in bfloat16 (from config). Casting weights to float16
        via .half() before .copy_() causes silent precision loss since copy_()
        casts source to destination dtype — the intermediate float16 loses precision.
        """
        source = EVAL_SCRIPT.read_text()
        assert ".half()" not in source, \
            "Must not use .half() for checkpoint weight loading — " \
            "use .to(param.dtype) to match model's bfloat16"

    def test_checkpoint_weights_use_param_dtype(self):
        """Weights must be cast to the parameter's dtype, not a hardcoded type."""
        source = EVAL_SCRIPT.read_text()
        assert "param_dict[name].dtype" in source or "param.dtype" in source, \
            "Checkpoint weight casting must use the parameter's actual dtype"


# ============================================================================
# Test class 6: YAML dispatch
# ============================================================================

class TestYAMLDispatch:
    """Verify the SkyPilot YAML correctly dispatches eval jobs."""

    def test_ordering_is_optional_for_v2_eval(self):
        """ORDERING must be optional — empty by default, not 'clustered'."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        v2_start = source.find("v2_eval")
        assert v2_start > 0
        v2_section = source[v2_start:v2_start + 500]
        assert "${ORDERING:?" not in v2_section, \
            "ORDERING must be optional in v2_eval"

    def test_yaml_ordering_does_not_default_to_clustered(self):
        """ORDERING must NOT default to 'clustered' — it caused v2_eval to
        load the wrong dataset for paper replications.
        """
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        assert 'ORDERING:-clustered' not in source.replace('"', '').replace("'", ""), \
            "ORDERING must not default to 'clustered' — it leaks into v2_eval " \
            "and loads the clustered stream instead of default MCF"

    def test_checkpoint_dir_is_optional(self):
        """CHECKPOINT_DIR must be optional (auto-resolved when not set)."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        v2_start = source.find("v2_eval")
        v2_section = source[v2_start:v2_start + 500]
        assert "${CHECKPOINT_DIR:?" not in v2_section

    def test_alg_name_is_required(self):
        """ALG_NAME must be required — eval needs it for output path."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        v2_start = source.find("v2_eval")
        v2_section = source[v2_start:v2_start + 500]
        assert "ALG_NAME" in v2_section

    def test_seed_is_required(self):
        """SEED must be required."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        v2_start = source.find("v2_eval")
        v2_section = source[v2_start:v2_start + 500]
        assert "SEED" in v2_section

    def test_baseline_dispatches_require_ordering(self):
        """Baseline runs (evoedit, nse, revive, generic) must require ORDERING explicitly."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        for dispatch in ["evoedit_baseline", "nse_baseline", "revive_baseline", "generic_baseline"]:
            dispatch_start = source.find(f'"{dispatch}')
            if dispatch_start < 0:
                dispatch_start = source.find(f"{dispatch}")
            assert dispatch_start > 0, f"Dispatch {dispatch} not found"
            section = source[dispatch_start:dispatch_start + 300]
            assert "ORDERING:?" in section or "ORDERING:?" in section.replace('"', ''), \
                f"{dispatch} must require ORDERING (use ${{ORDERING:?...}}) — " \
                f"baseline runs must have an explicit ordering, not a silent default"

    def test_yaml_no_clustered_default(self):
        """YAML must not default ORDERING to 'clustered'."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        assert "ORDERING:-clustered" not in source.replace('"', '').replace("'", ""), \
            "ORDERING must not default to clustered — caused silent wrong dataset in evals"

    def test_yaml_no_clustered_remapping(self):
        """YAML must not silently remap 'clustered' to 'fb_random0'."""
        yaml_path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        source = yaml_path.read_text()
        assert 'clustered" ]] && _BL_ORDERING="fb_random0"' not in source, \
            "Removed: silent remapping of clustered→fb_random0 was a workaround for the wrong default"


# ============================================================================
# Test class 7: Eval script dual-metric output
# ============================================================================

class TestDualMetricOutput:
    """Verify the eval produces both prob-pref and argmax metrics."""

    def test_output_has_both_metrics(self):
        """Eval output must have efficacy (prob-pref) and efficacy_argmax."""
        source = EVAL_SCRIPT.read_text()
        assert "efficacy_argmax" in source, "Must compute argmax metric"
        assert "efficacy" in source, "Must compute prob-pref metric"
        assert "neighborhood_argmax" in source
        assert "paraphrase_argmax" in source

    def test_prob_pref_is_primary(self):
        """prob-pref efficacy must be the primary 'efficacy' field."""
        source = EVAL_SCRIPT.read_text()
        # The first 'efficacy' in the return dict should be prob-pref
        # efficacy_argmax should be separate
        assert '"efficacy":' in source or "'efficacy':" in source

    def test_output_has_cohort_metrics(self):
        """Output must include first_1k, latest_1k, etc."""
        source = EVAL_SCRIPT.read_text()
        assert "first_1k" in source
        assert "latest_1k" in source


# ============================================================================
# Test class 8: End-to-end simulation (no GPU)
# ============================================================================

class TestEndToEndSimulation:
    """Simulate the eval dataset selection logic end-to-end on CPU."""

    def _simulate_dataset_selection(self, ordering=None, dataset_path=None, seed=42):
        """Simulate the eval script's dataset selection without running it."""
        if dataset_path:
            return Path(dataset_path)

        if ordering:
            result_root = Path(os.environ.get("RESULT_ROOT", str(PROJECT_ROOT / "results")))
            candidates = [
                result_root / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json",
                PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json",
            ]
            for c in candidates:
                if c.exists():
                    return c
            return None  # would sys.exit

        candidates = [
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "multi_counterfact.json",
            Path("data/dsets") / "multi_counterfact.json",
            PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def test_no_args_loads_default_mcf(self):
        """No ordering, no dataset_path → default MCF."""
        path = self._simulate_dataset_selection()
        assert path is not None
        assert "multi_counterfact" in str(path)

    def test_ordering_loads_stream(self):
        """--ordering fb_high_exposure → stream file."""
        path = self._simulate_dataset_selection(ordering="fb_high_exposure", seed=42)
        if path is None:
            pytest.skip("Stream file not available")
        assert "fb_high_exposure_seed42" in str(path)
        assert "orderings" in str(path)

    def test_ordering_different_seed(self):
        """--ordering with different seed loads different stream."""
        p42 = self._simulate_dataset_selection(ordering="fb_high_exposure", seed=42)
        p2024 = self._simulate_dataset_selection(ordering="fb_high_exposure", seed=2024)
        if p42 is None or p2024 is None:
            pytest.skip("Stream files not available")
        assert p42 != p2024, "Different seeds must load different streams"

    def test_dataset_path_overrides_everything(self):
        """--dataset_path takes priority over --ordering."""
        path = self._simulate_dataset_selection(
            ordering="fb_high_exposure",
            dataset_path="/custom/path.json"
        )
        assert str(path) == "/custom/path.json"

    def test_nonexistent_ordering_returns_none(self):
        """Non-existent ordering should fail (None = sys.exit in real code)."""
        path = self._simulate_dataset_selection(ordering="nonexistent_ordering_xyz")
        assert path is None, "Non-existent ordering should fail, not silently use MCF"

    @pytest.mark.parametrize("ordering", [
        "fb_high_exposure", "fb_low_exposure", "fb_random0",
        "key_clustered", "key_dispersed",
        "sched_balanced", "sched_conditioning_only", "sched_exposure_only",
    ])
    def test_all_orderings_resolve_correctly(self, ordering):
        """Every ordering used in the paper must resolve to its stream file."""
        path = self._simulate_dataset_selection(ordering=ordering, seed=42)
        if path is None:
            pytest.skip(f"Stream not available for {ordering}")
        assert ordering in str(path), f"Resolved path doesn't contain ordering name"
        assert path.exists(), f"Resolved path doesn't exist: {path}"

    def test_ordering_stream_and_mcf_differ(self):
        """The stream file must contain different records than default MCF."""
        mcf_path = self._simulate_dataset_selection()
        stream_path = self._simulate_dataset_selection(ordering="fb_high_exposure", seed=42)
        if mcf_path is None or stream_path is None:
            pytest.skip("Files not available")
        assert mcf_path != stream_path, "Ordering must load stream, not MCF"

        with open(mcf_path) as f:
            mcf_ids = set(r["case_id"] for r in json.load(f)[:10000])
        with open(stream_path) as f:
            stream_ids = set(r["case_id"] for r in json.load(f)[:10000])

        overlap = len(mcf_ids & stream_ids) / max(len(mcf_ids), 1)
        assert overlap < 0.9, \
            f"Stream and MCF overlap {overlap:.0%} — eval would check wrong records"
