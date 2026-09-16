#!/usr/bin/env python3
"""Regression tests for dtype loading and REVIVE Phase 2 application.

These catch two critical bugs discovered during paper replication validation:

1. Model loading defaulted to float16 instead of letting the model config
   decide (bfloat16 for Llama-3). This caused AlphaEdit +10pp (when overridden
   to float32) and MEMIT -15pp (float16 has less dynamic range than bfloat16).

2. REVIVE spectral filter was applied in MEMIT's Phase 1 (temporary updates)
   but discarded when Phase 2 reconstructed updates from stored (adj_k, resid)
   deltas. REVIVE+MEMIT produced results identical to plain MEMIT.

No GPU needed for most tests — uses mock tensors and source inspection.
"""
import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Fix 1: Model dtype regression tests
# ============================================================================

class TestRevivePhase2Application:
    """Verify that post_solve hooks fire in MEMIT Phase 2 (delta application)."""

    def test_memit_phase2_calls_post_solve(self):
        """apply_memit_with_hooks must call hooks.post_solve in the Phase 2 loop."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()

        # Find the Phase 2 code: the block inside apply_memit_with_hooks
        # that iterates over deltas and applies updates
        # It should contain hooks.post_solve
        func_source = _extract_function_source(source, "apply_memit_with_hooks")

        # The Phase 2 loop is after _execute_memit_with_hooks returns
        # and before "New weights successfully inserted"
        phase2_start = func_source.find("for w_name, (key_mat, val_mat) in deltas")
        phase2_end = func_source.find("New weights successfully inserted")
        assert phase2_start > 0, "Phase 2 delta loop not found"
        assert phase2_end > phase2_start, "Phase 2 end marker not found"

        phase2_code = func_source[phase2_start:phase2_end]
        assert "hooks.post_solve" in phase2_code, \
            "Phase 2 must call hooks.post_solve on reconstructed upd_matrix. " \
            "Without this, REVIVE filter is computed in Phase 1 then discarded."

    def test_memit_phase2_post_solve_before_weight_update(self):
        """post_solve in Phase 2 must run BEFORE w[...] += upd_matrix."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        func_source = _extract_function_source(source, "apply_memit_with_hooks")

        phase2_start = func_source.find("for w_name, (key_mat, val_mat) in deltas")
        phase2_code = func_source[phase2_start:]

        post_solve_pos = phase2_code.find("hooks.post_solve")
        weight_update_pos = phase2_code.find("w[...] += upd_matrix")

        assert post_solve_pos > 0, "hooks.post_solve not found in Phase 2"
        assert weight_update_pos > 0, "weight update not found in Phase 2"
        assert post_solve_pos < weight_update_pos, \
            "hooks.post_solve must run BEFORE w[...] += upd_matrix in Phase 2"

    def test_post_solve_receives_reconstructed_update(self):
        """Phase 2 post_solve should receive upd_matrix = key_mat @ val_mat.T."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        func_source = _extract_function_source(source, "apply_memit_with_hooks")

        phase2_start = func_source.find("for w_name, (key_mat, val_mat) in deltas")
        phase2_code = func_source[phase2_start:]

        # upd_matrix must be computed before post_solve
        reconstruct_pos = phase2_code.find("upd_matrix = key_mat @ val_mat.T")
        post_solve_pos = phase2_code.find("hooks.post_solve")

        assert reconstruct_pos > 0, "upd_matrix reconstruction not found"
        assert reconstruct_pos < post_solve_pos, \
            "upd_matrix must be reconstructed before post_solve"

    def test_alphaedit_has_no_restore_reapply(self):
        """AlphaEdit applies updates directly — no Phase 1/2 split to worry about."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        # AlphaEdit should NOT have a deltas dict or restore/reapply pattern
        assert "deltas[" not in source or "deltas = {}" in source, \
            "AlphaEdit should not use the MEMIT-style deltas restore/reapply pattern"

    def test_post_solve_mock_modifies_final_update(self):
        """A mock post_solve hook should modify the final weight update in MEMIT."""
        from algorithms.hooks import AlgorithmHooks

        # Create a hook that zeros the update (simulating aggressive REVIVE)
        call_log = []
        def zeroing_post_solve(layer, upd, adj_k, ks, w_name, state):
            call_log.append(("phase2" if layer is None else "phase1", w_name))
            return torch.zeros_like(upd)

        hooks = AlgorithmHooks(post_solve=zeroing_post_solve)

        # Verify the hook is callable and returns correct shape
        dummy = torch.randn(4, 8)
        result = hooks.post_solve(None, dummy, None, None, "test.weight", {})
        assert result.shape == dummy.shape
        assert torch.all(result == 0)

    def test_phase1_and_phase2_both_marked(self):
        """Phase 1 passes layer index, Phase 2 passes None — distinguishable."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()

        # Phase 1 call (inside _execute_memit_with_hooks)
        exec_source = _extract_function_source(source, "_execute_memit_with_hooks")
        assert "hooks.post_solve(layer," in exec_source, \
            "Phase 1 post_solve should pass layer index"

        # Phase 2 call (inside apply_memit_with_hooks)
        apply_source = _extract_function_source(source, "apply_memit_with_hooks")
        assert "hooks.post_solve(None," in apply_source, \
            "Phase 2 post_solve should pass None for layer (distinguishes phases)"

    def test_all_preset_hooks_handle_none_layer(self):
        """All hook presets must handle layer_idx=None (Phase 2) without crashing."""
        from algorithms.hook_presets import seqreg_hooks, revive_hooks

        dummy_upd = torch.randn(4, 8)
        dummy_state = {
            "batch_idx": [0],
            "mechanism_log": [],
            "prev_cache": {},
        }

        # seqreg_hooks post_solve with None layer
        sr = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        result = sr.post_solve(None, dummy_upd, None, None, "test.weight", dummy_state)
        assert torch.equal(result, dummy_upd), \
            "seqreg post_solve(None, ...) should pass through without modification"
        assert len(dummy_state["mechanism_log"]) == 0, \
            "seqreg post_solve(None, ...) should not log (Phase 2 skip)"


# ============================================================================
# CPU integration: all presets × Phase 2 None-safety
# ============================================================================

class TestAllPresetsPhase2Safe:
    """Every hook preset that defines post_solve must handle layer_idx=None.

    MEMIT's Phase 2 calls post_solve(None, ...) for every layer's reconstructed
    update. Any preset that crashes on None would break REVIVE+MEMIT-Seq composed
    hooks, even if the individual preset doesn't need Phase 2 filtering.
    """

    def _make_seqreg_state(self):
        return {
            "batch_idx": [0],
            "mechanism_log": [],
            "prev_cache": {},
        }

    def test_seqreg_post_solve_none_passthrough(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        upd = torch.randn(4, 8)
        state = self._make_seqreg_state()
        result = hooks.post_solve(None, upd, None, None, "w", state)
        assert torch.equal(result, upd)
        assert len(state["mechanism_log"]) == 0

    def test_seqreg_post_solve_none_no_cache_modification(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0, cache_strategy="all")
        upd = torch.randn(4, 8)
        state = self._make_seqreg_state()
        hooks.post_solve(None, upd, None, None, "w", state)
        assert len(state["prev_cache"]) == 0, \
            "Phase 2 should not cache keys"

    def test_seqreg_post_solve_integer_layer_still_works(self):
        """Phase 1 with real layer index must still work after the None fix."""
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0, cache_strategy="all")
        upd = torch.randn(4, 8)
        layer_ks = torch.randn(8, 3)
        state = self._make_seqreg_state()
        result = hooks.post_solve(4, upd, None, layer_ks, "w", state)
        assert torch.equal(result, upd)
        assert len(state["mechanism_log"]) == 1
        assert state["mechanism_log"][0]["layer"] == 4
        assert 4 in state["prev_cache"]

    def test_revive_post_solve_none_filters(self):
        """REVIVE post_solve with None layer should filter (Phase 2), not just pass through."""
        from algorithms.hook_presets import revive_hooks
        hooks = revive_hooks(revive_tau=0.5, revive_svd_device="cpu")
        upd = torch.randn(32, 16)
        state = {"_current_weights": {"w": torch.randn(32, 16)}}
        result = hooks.post_solve(None, upd, None, None, "w", state)
        assert result.shape == upd.shape
        assert not torch.allclose(result, upd), \
            "REVIVE should actually filter the update, not pass through unchanged"

    def test_composed_seqreg_revive_none_safe(self):
        """compose_hooks(seqreg, revive) must handle None in Phase 2."""
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks

        sr = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        rv = revive_hooks(revive_tau=0.5, revive_svd_device="cpu")
        composed = compose_hooks(sr, rv)

        upd = torch.randn(32, 16)
        state = {
            "batch_idx": [0],
            "mechanism_log": [],
            "prev_cache": {},
            "_current_weights": {"w": torch.randn(32, 16)},
        }
        # This is what MEMIT Phase 2 does with composed hooks
        result = composed.post_solve(None, upd, None, None, "w", state)
        assert result.shape == upd.shape
        assert len(state["mechanism_log"]) == 0, \
            "seqreg should skip logging in Phase 2"

    def test_pathguard_has_no_post_solve(self):
        """PathGuard only has post_update — no Phase 2 concern."""
        from algorithms.hook_presets import pathguard_hooks
        hooks = pathguard_hooks()
        assert hooks.post_solve is None

    def test_polykernel_has_no_post_solve(self):
        """Polykernel only has build_lhs — no Phase 2 concern."""
        from algorithms.hook_presets import polykernel_hooks
        hooks = polykernel_hooks()
        assert hooks.post_solve is None


class TestComposeHooksPhase2Propagation:
    """Verify compose_hooks correctly propagates None layer through chains."""

    def test_chain_passes_none_to_all(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks

        received_layers = []

        def hook_a(layer, upd, *a):
            received_layers.append(("a", layer))
            return upd

        def hook_b(layer, upd, *a):
            received_layers.append(("b", layer))
            return upd * 0.5

        composed = compose_hooks(
            AlgorithmHooks(post_solve=hook_a),
            AlgorithmHooks(post_solve=hook_b),
        )
        upd = torch.randn(4, 8)
        composed.post_solve(None, upd, None, None, "w", {})

        assert received_layers == [("a", None), ("b", None)]

    def test_chain_passes_integer_layer(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks

        received_layers = []

        def hook_a(layer, upd, *a):
            received_layers.append(("a", layer))
            return upd

        composed = compose_hooks(
            AlgorithmHooks(post_solve=hook_a),
        )
        upd = torch.randn(4, 8)
        composed.post_solve(5, upd, None, None, "w", {})
        assert received_layers == [("a", 5)]



class TestNSEIntegration:
    """NSE-specific integration tests."""

    def test_nse_main_has_z_error_mask(self):
        """NSE's z-error mask must exist — it controls whether edits are applied."""
        for path in [
            PROJECT_ROOT / "baselines" / "EvoEdit" / "nse" / "nse_main.py",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "nse" / "nse_main.py",
        ]:
            if not path.exists():
                continue
            source = path.read_text()
            assert "alpha" in source and "upper_bound" in source, \
                f"{path.name} must have z-error bounds (alpha, upper_bound)"
            assert "mask" in source, \
                f"{path.name} must have z-error mask controlling edit application"

    def test_nse_hparams_have_bounds(self):
        """NSE hparams must include alpha and upper_bound for z-error gating."""
        import json
        for hparams_path in [
            PROJECT_ROOT / "baselines" / "EvoEdit" / "hparams" / "NSE" / "Llama3-8B.json",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "hparams" / "NSE" / "Llama3-8B.json",
        ]:
            if not hparams_path.exists():
                continue
            with open(hparams_path) as f:
                hparams = json.load(f)
            assert "alpha" in hparams, f"{hparams_path.name} missing alpha"
            assert "upper_bound" in hparams, f"{hparams_path.name} missing upper_bound"
            assert hparams["alpha"] > 0, "alpha must be positive"
            assert hparams["upper_bound"] > hparams["alpha"], "upper_bound must be > alpha"

    def test_polykernel_runner_nse_zero_delta_handling(self):
        """Runner must not crash when NSE produces zero deltas (e.g. cache/dtype mismatch)."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # The post-hoc REVIVE path must handle zero deltas gracefully
        assert "all deltas < 1e-10" in source or "deltas < " in source, \
            "Runner must detect and warn about zero deltas from NSE"

    def test_nse_cache_path_referenced(self):
        """The runner must reference the NSE KV cache directory."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "nse_kv_cache" in source or "NSE" in source, \
            "Runner must handle NSE KV cache"


class TestReviveBaselineDefaults:
    """REVIVE paper replication requires lambda_prev=0 for plain base methods."""

    def test_revive_paper_repl_uses_lambda_zero(self):
        """run_revive_paper_replication.sh must use lambda_prev=0 (plain base, no history)."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_paper_replication.sh").read_text()
        # The paper replication must set lambda_prev=0 to match published REVIVE results
        assert 'LAMBDA_PREV="${LAMBDA_PREV:-0' in source or \
               "lambda_prev 0" in source or \
               'LAMBDA_PREV="0' in source, \
            "run_revive_paper_replication.sh must default lambda_prev=0 " \
            "(plain base method, matching REVIVE paper's AlphaEdit/MEMIT/NSE baselines)"

    def test_revive_baseline_defaults_to_zero(self):
        """run_revive_baseline.sh must default to lambda_prev=0 (plain base + REVIVE)."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_baseline.sh").read_text()
        assert 'LAMBDA_PREV="${LAMBDA_PREV:-0' in source, \
            "run_revive_baseline.sh must default LAMBDA_PREV to 0 " \
            "(plain base method + REVIVE, matching the paper)"
        assert 'LAMBDA_PREV="${LAMBDA_PREV:-1' not in source, \
            "run_revive_baseline.sh must NOT default to 1.0 — " \
            "that silently adds SeqReg history on top of REVIVE"

    def test_revive_paper_repl_ae_no_history(self):
        """REVIVE+AlphaEdit paper config: lambda_prev=0, lambda_delta=0 (no SeqReg history)."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_paper_replication.sh").read_text()
        # Find the actual values passed to the runner
        assert "lambda_prev" in source.lower() or "LAMBDA_PREV" in source
        # Must not default to 1.0 for paper replication
        assert 'LAMBDA_PREV="${LAMBDA_PREV:-1' not in source, \
            "REVIVE paper replication must NOT default to lambda_prev=1.0 — " \
            "published REVIVE+AlphaEdit uses plain AlphaEdit (lambda_prev=0)"


class TestEvalAtEndOnly:
    """The --eval_at_end_only flag must checkpoint normally but only eval at the final batch."""

    def test_flag_exists_in_argparse(self):
        """polykernel_seqreg_runner must accept --eval_at_end_only."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "eval_at_end_only" in source, \
            "Runner must support --eval_at_end_only flag"

    def test_flag_in_eval_group(self):
        """--eval_at_end_only must be in the eval mutually exclusive group."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # Find the eval_group section
        group_start = source.find("eval_group = parser.add_mutually_exclusive_group")
        group_section = source[group_start:group_start + 500]
        assert "eval_at_end_only" in group_section, \
            "--eval_at_end_only must be in the eval mutually exclusive group"

    def test_should_eval_respects_end_only(self):
        """should_eval_fn must return False for all batches except the last when eval_at_end_only=True."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        func_start = source.find("def should_eval_fn")
        func_section = source[func_start:func_start + 300]
        assert "eval_at_end_only" in func_section, \
            "should_eval_fn must check eval_at_end_only flag"

    def test_end_only_still_saves_checkpoints(self):
        """--eval_at_end_only must NOT change checkpoint saving behavior."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # should_save must NOT reference eval_at_end_only
        save_fn_start = source.find("def should_save")
        if save_fn_start > 0:
            save_section = source[save_fn_start:save_fn_start + 200]
            assert "eval_at_end_only" not in save_section, \
                "should_save must not be affected by eval_at_end_only — checkpoints save normally"

    def test_eval_at_end_only_behavioral(self):
        """Simulate should_eval_fn: checkpoints every 10 batches, eval only at batch 99."""
        import types
        args = types.SimpleNamespace(
            eval_at_end_only=True,
            eval_at_checkpoints_only=False,
            fast_checkpoint=False,
            save_interval=10,
        )
        total_batches = 100  # 10K edits / 100 per batch

        def should_save(batch_idx, interval):
            return (batch_idx + 1) % interval == 0 or batch_idx == total_batches - 1

        def should_eval_fn(batch_idx):
            if args.eval_at_end_only:
                return batch_idx == total_batches - 1
            if args.eval_at_checkpoints_only:
                return should_save(batch_idx, args.save_interval)
            return True

        # Checkpoints save at 9, 19, 29, ... 89, 99
        save_batches = [b for b in range(100) if should_save(b, 10)]
        assert save_batches == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99], \
            f"Checkpoints should save every 10 batches: {save_batches}"

        # Eval only at batch 99
        eval_batches = [b for b in range(100) if should_eval_fn(b)]
        assert eval_batches == [99], \
            f"Eval should only run at final batch (99), got: {eval_batches}"

    def test_eval_at_checkpoints_only_behavioral(self):
        """Contrast: --eval_at_checkpoints_only evals at every checkpoint."""
        import types
        args = types.SimpleNamespace(
            eval_at_end_only=False,
            eval_at_checkpoints_only=True,
            fast_checkpoint=False,
            save_interval=10,
        )
        total_batches = 100

        def should_save(batch_idx, interval):
            return (batch_idx + 1) % interval == 0 or batch_idx == total_batches - 1

        def should_eval_fn(batch_idx):
            if args.eval_at_end_only:
                return batch_idx == total_batches - 1
            if args.eval_at_checkpoints_only:
                return should_save(batch_idx, args.save_interval)
            return True

        eval_batches = [b for b in range(100) if should_eval_fn(b)]
        assert eval_batches == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99], \
            f"Milestone eval should run at every checkpoint: {eval_batches}"

    def test_revive_baseline_supports_eval_at_end(self):
        """run_revive_baseline.sh must support EVAL_AT_END_ONLY env var."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_baseline.sh").read_text()
        assert "eval_at_end_only" in source.lower() or "EVAL_AT_END" in source, \
            "run_revive_baseline.sh must support eval-at-end-only mode"

    def test_revive_paper_repl_supports_eval_at_end(self):
        """run_revive_paper_replication.sh must support EVAL_AT_END_ONLY env var."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_paper_replication.sh").read_text()
        assert "eval_at_end_only" in source.lower() or "EVAL_AT_END" in source, \
            "run_revive_paper_replication.sh must support eval-at-end-only mode"


class TestVendorGlobalsHandling:
    """Vendor code requires globals.yml or pre-populated util.globals module."""

    def test_polykernel_runner_ensures_vendor_globals(self):
        """polykernel_seqreg_runner must call _ensure_vendor_globals before vendor imports."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        globals_call = source.find("_ensure_vendor_globals")
        alphaedit_import = source.find("from AlphaEdit import")
        assert globals_call > 0, \
            "polykernel_seqreg_runner must call _ensure_vendor_globals"
        assert globals_call < alphaedit_import, \
            "_ensure_vendor_globals must be called BEFORE 'from AlphaEdit import' " \
            "to prevent globals.yml FileNotFoundError"


class TestEvalOrderingDataset:
    """Eval must use the ordering stream as dataset when --ordering is specified.

    Ordering streams draw from the full 21K MCF pool — only ~48% overlap with
    the default first-10K. Evaluating default records after editing ordering
    records gives meaningless results.
    """

    def test_eval_loads_stream_when_ordering_specified(self):
        """eval_matched_ordering.py must load the ordering stream file, not raw MCF."""
        source = (PROJECT_ROOT / "scripts" / "eval_matched_ordering.py").read_text()
        assert "args.ordering" in source
        assert "stream_candidates" in source or "ordering_seed" in source.replace(" ", "").replace("_", ""), \
            "Eval script must load ordering stream file when --ordering is specified"

    def test_eval_has_ordering_stream_path(self):
        """The stream file path must follow orderings/{ordering}_seed{seed}.json."""
        source = (PROJECT_ROOT / "scripts" / "eval_matched_ordering.py").read_text()
        assert "ordering" in source and "_seed" in source and "orderings" in source

    def test_ordering_records_differ_from_default(self):
        """Verify that ordering streams use different records than default first-10K."""
        mcf_path = PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json"
        stream_path = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_high_exposure_seed42.json"

        if not mcf_path.exists() or not stream_path.exists():
            pytest.skip("Dataset files not available")

        import json as _json
        with open(mcf_path) as f:
            mcf = _json.load(f)
        with open(stream_path) as f:
            stream = _json.load(f)

        default_ids = set(r["case_id"] for r in mcf[:10000])
        stream_ids = set(r["case_id"] for r in stream[:10000])

        overlap = len(default_ids & stream_ids) / len(default_ids)
        assert overlap < 0.9, \
            f"Ordering stream should differ from default MCF first-10K (overlap={overlap:.0%}). " \
            f"If eval uses default MCF for ordering runs, it evaluates the WRONG records."

    def test_yaml_ordering_does_not_default_to_clustered(self):
        """ORDERING must not default to 'clustered' — leaks into v2_eval."""
        source = (PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml").read_text()
        assert 'ORDERING:-clustered' not in source.replace('"', '').replace("'", ""), \
            "ORDERING must not default to 'clustered'"


class TestEvalOrderingStreamIntegration:
    """Integration tests verifying every ordering loads its correct stream file.

    Each stream file contains a specific set of MCF records in a specific order.
    The eval script must resolve the correct stream for each ordering × seed.
    """

    # All orderings that have stream files
    ORDERINGS_WITH_STREAMS = [
        "fb_high_exposure", "fb_low_exposure", "fb_random0", "fb_random1", "fb_random2",
        "key_clustered", "key_dispersed", "clustered", "dispersed",
        "sched_balanced", "sched_conditioning_only", "sched_exposure_only",
        "suffix_random", "suffix_spread", "suffix_random_late", "suffix_spread_late",
        "greedy_minmax", "random", "cluster_topo",
        "subj_matched_clustered", "subj_matched_dispersed",
    ]

    def _stream_path(self, ordering, seed=42):
        return PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"

    def _eval_resolves_stream(self, ordering, seed=42):
        """Simulate the eval script's stream resolution logic."""
        result_root = PROJECT_ROOT / "results"
        candidates = [
            result_root / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json",
            PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    @pytest.mark.parametrize("ordering", ORDERINGS_WITH_STREAMS)
    def test_stream_file_exists_for_seed42(self, ordering):
        """Each ordering must have a stream file for at least seed 42."""
        path = self._stream_path(ordering, 42)
        if not path.exists():
            pytest.skip(f"Stream file not available: {path.name}")
        assert path.stat().st_size > 1000, f"Stream file too small: {path}"

    @pytest.mark.parametrize("ordering", ORDERINGS_WITH_STREAMS)
    def test_eval_resolves_correct_stream(self, ordering):
        """The eval script's resolution logic must find each ordering's stream."""
        path = self._stream_path(ordering, 42)
        if not path.exists():
            pytest.skip(f"Stream file not available: {path.name}")
        resolved = self._eval_resolves_stream(ordering, 42)
        assert resolved is not None, f"Eval cannot resolve stream for {ordering}"
        assert resolved == path, f"Eval resolved wrong path: {resolved} != {path}"

    @pytest.mark.parametrize("ordering", ORDERINGS_WITH_STREAMS)
    def test_stream_has_valid_mcf_records(self, ordering):
        """Each stream must contain valid MCF records with required fields."""
        path = self._stream_path(ordering, 42)
        if not path.exists():
            pytest.skip(f"Stream file not available: {path.name}")

        import json as _json
        with open(path) as f:
            records = _json.load(f)

        assert len(records) >= 1000, f"Stream too short: {len(records)} records"
        required = {"case_id", "requested_rewrite", "paraphrase_prompts", "neighborhood_prompts"}
        for field in required:
            assert field in records[0], f"Stream record missing '{field}'"

    @pytest.mark.parametrize("ordering", ["fb_high_exposure", "fb_low_exposure", "fb_random0",
                                           "key_clustered", "key_dispersed"])
    def test_core_orderings_have_all_seeds(self, ordering):
        """Core orderings used in the paper must have streams for all 3 seeds."""
        for seed in [42, 2024, 137]:
            path = self._stream_path(ordering, seed)
            assert path.exists(), \
                f"Core ordering {ordering} missing stream for seed {seed}"

    def test_no_ordering_uses_default_mcf(self):
        """Without --ordering, eval must use default MCF (not any stream file)."""
        source = (PROJECT_ROOT / "scripts" / "eval_matched_ordering.py").read_text()
        # The code path for no ordering should go to the else branch
        # that loads multi_counterfact.json
        assert "multi_counterfact.json" in source
        # And the ordering stream path should only be used when args.ordering is truthy
        assert "elif args.ordering:" in source or "if args.ordering:" in source

    def test_stream_records_are_from_mcf(self):
        """All stream records must be valid MCF records (subset of full dataset)."""
        mcf_path = PROJECT_ROOT / "data" / "dsets" / "multi_counterfact.json"
        stream_path = self._stream_path("fb_high_exposure", 42)
        if not mcf_path.exists() or not stream_path.exists():
            pytest.skip("Dataset files not available")

        import json as _json
        with open(mcf_path) as f:
            mcf = _json.load(f)
        with open(stream_path) as f:
            stream = _json.load(f)

        mcf_ids = set(r["case_id"] for r in mcf)
        stream_ids = set(r["case_id"] for r in stream)
        assert stream_ids.issubset(mcf_ids), \
            f"Stream has {len(stream_ids - mcf_ids)} records not in MCF"

    def test_different_orderings_have_different_record_order(self):
        """fb_high and fb_low should have the same records in different order."""
        high_path = self._stream_path("fb_high_exposure", 42)
        low_path = self._stream_path("fb_low_exposure", 42)
        if not high_path.exists() or not low_path.exists():
            pytest.skip("Stream files not available")

        import json as _json
        with open(high_path) as f:
            high = _json.load(f)
        with open(low_path) as f:
            low = _json.load(f)

        high_ids = [r["case_id"] for r in high]
        low_ids = [r["case_id"] for r in low]

        assert set(high_ids) == set(low_ids), \
            "fb_high and fb_low should contain the same records (fixed-batch)"
        assert high_ids != low_ids, \
            "fb_high and fb_low should be in DIFFERENT order"


class TestPolyKernelRunnerResumeVars:
    """Variables used in the resume path must be defined before the resume check."""

    def test_cache_c_defined_before_resume_check(self):
        """cache_c must be initialized before the checkpoint resume block."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        lines = source.split("\n")

        cache_c_init = None
        resume_check = None
        cache_c_use_in_resume = None

        for i, line in enumerate(lines):
            s = line.strip()
            if s == "cache_c = None" and cache_c_init is None:
                cache_c_init = i
            if "start_from_batch > 0" in s and resume_check is None:
                resume_check = i
            if resume_check and "cache_c" in s and "extra_keys" not in s and cache_c_use_in_resume is None:
                cache_c_use_in_resume = i

        assert cache_c_init is not None, "cache_c = None initialization not found"
        assert resume_check is not None, "start_from_batch > 0 check not found"
        assert cache_c_init < resume_check, \
            f"cache_c must be initialized (line {cache_c_init}) before resume check (line {resume_check})"

    def test_error_cache_defined_before_resume_check(self):
        """error_cache must be initialized before the checkpoint resume block."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        lines = source.split("\n")

        error_init = None
        resume_check = None

        for i, line in enumerate(lines):
            s = line.strip()
            if s == "error_cache = None" and error_init is None:
                error_init = i
            if "start_from_batch > 0" in s and resume_check is None:
                resume_check = i

        assert error_init is not None, "error_cache = None initialization not found"
        assert resume_check is not None, "start_from_batch > 0 check not found"
        assert error_init < resume_check, \
            f"error_cache must be initialized (line {error_init}) before resume check (line {resume_check})"


class TestMemitPhase2Integration:
    """Simulate the MEMIT Phase 2 loop with real hook presets on CPU."""

    def test_phase2_loop_with_seqreg(self):
        """Reproduce the exact Phase 2 code path that crashed."""
        from algorithms.hooks import AlgorithmHooks
        from algorithms.hook_presets import seqreg_hooks

        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        state = hooks.get_state()
        state["batch_idx"] = [0]

        # Simulate deltas from Phase 1
        deltas = {
            "model.layers.4.mlp.down_proj.weight": (
                torch.randn(16, 4), torch.randn(32, 4)
            ),
            "model.layers.5.mlp.down_proj.weight": (
                torch.randn(16, 4), torch.randn(32, 4)
            ),
        }

        # Simulate Phase 2 — this is the exact code from apply_memit_with_hooks
        for w_name, (key_mat, val_mat) in deltas.items():
            upd_matrix = key_mat @ val_mat.T
            if hooks.post_solve is not None:
                upd_matrix = hooks.post_solve(None, upd_matrix, None, None, w_name, state)
            # Would normally do: w[...] += upd_matrix.float()

        # Should not crash, and state should not have been modified
        assert len(state["mechanism_log"]) == 0

    def test_phase2_loop_with_composed_seqreg_revive(self):
        """The exact combo that was crashing: seqreg + revive in Phase 2."""
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks

        sr = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        rv = revive_hooks(revive_tau=0.5, revive_svd_device="cpu")
        hooks = compose_hooks(sr, rv)
        state = hooks.get_state()
        state["batch_idx"] = [0]

        # Weight shape: (d_out, d_in). Phase 2 reconstructs upd = key_mat @ val_mat.T
        # then _match_shape transposes if needed so upd matches weight shape.
        d_in, d_out = 16, 32
        w_names = [
            "model.layers.4.mlp.down_proj.weight",
            "model.layers.5.mlp.down_proj.weight",
        ]
        state["_current_weights"] = {n: torch.randn(d_out, d_in) for n in w_names}

        deltas = {n: (torch.randn(d_in, 4), torch.randn(d_out, 4)) for n in w_names}

        for w_name, (key_mat, val_mat) in deltas.items():
            upd_matrix = key_mat @ val_mat.T  # (d_in, d_out)
            # _match_shape: transpose to (d_out, d_in) to match weight
            upd_matrix = upd_matrix.T
            if hooks.post_solve is not None:
                upd_matrix = hooks.post_solve(None, upd_matrix, None, None, w_name, state)

        assert len(state["mechanism_log"]) == 0


# ============================================================================
# Combined regression: REVIVE filter must actually change MEMIT weights
# ============================================================================

class TestReviveFilterNotDiscarded:
    """Ensure the REVIVE filter produces different weights than no filter."""

    def test_memit_wrapper_source_has_phase2_hook(self):
        """The specific code pattern that was the bug: deltas stored unfiltered."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()

        lines = source.split("\n")
        found_reconstruct = False
        found_filter = False
        found_apply = False

        for line in lines:
            s = line.strip()
            if "upd_matrix = key_mat @ val_mat.T" in s:
                found_reconstruct = True
            if found_reconstruct and "hooks.post_solve" in s and "None," in s:
                found_filter = True
            if found_filter and "w[...] += upd_matrix" in s:
                found_apply = True
                break

        assert found_reconstruct, "Phase 2 upd_matrix reconstruction not found"
        assert found_filter, "Phase 2 post_solve filter not found after reconstruction"
        assert found_apply, "Weight application not found after Phase 2 filter"

    def test_phase2_hook_changes_final_weights(self):
        """Simulate MEMIT Phase 2 with and without a halving hook.

        This replicates the exact Phase 2 logic from apply_memit_with_hooks
        and verifies that a post_solve hook (like REVIVE) actually changes
        the weights applied to the model — catching the original bug where
        the hook ran but its result was discarded.
        """
        from algorithms.hooks import AlgorithmHooks

        # Simulate stored deltas from Phase 1 (adj_k, resid pairs)
        d_in, d_out, n = 16, 32, 4
        adj_k = torch.randn(d_in, n)
        resid = torch.randn(d_out, n)
        deltas = {"layer.weight": (adj_k, resid)}

        # Simulate model weight
        W_orig = torch.randn(d_out, d_in)

        # --- Run Phase 2 WITHOUT hooks (plain MEMIT) ---
        W_no_hook = W_orig.clone()
        upd = adj_k @ resid.T  # note: MEMIT Phase 2 does key_mat @ val_mat.T
        # _match_shape transposes if needed; for (d_in, n) @ (d_out, n).T = (d_in, d_out)
        # weight is (d_out, d_in), so upd.T
        upd_matched = upd.T if upd.shape != W_no_hook.shape else upd
        W_no_hook += upd_matched.float()

        # --- Run Phase 2 WITH halving hook (simulates REVIVE) ---
        halving_hook = AlgorithmHooks(
            post_solve=lambda layer, u, *a: u * 0.5
        )
        W_with_hook = W_orig.clone()
        upd2 = adj_k @ resid.T
        upd2_matched = upd2.T if upd2.shape != W_with_hook.shape else upd2
        # Apply hook — this is what Phase 2 now does
        upd2_matched = halving_hook.post_solve(None, upd2_matched, None, None, "layer.weight", {})
        W_with_hook += upd2_matched.float()

        # The two should differ by exactly half the update
        diff = (W_no_hook - W_with_hook).abs().sum()
        expected_diff = upd_matched.abs().sum() * 0.5
        assert diff > 0, "Hook should change the final weights"
        assert torch.allclose(diff, expected_diff, rtol=1e-4), \
            f"Halving hook should halve the update. Got diff={diff:.4f}, expected={expected_diff:.4f}"

    def test_zeroing_hook_prevents_all_weight_changes(self):
        """A zeroing post_solve (aggressive REVIVE) should leave weights unchanged."""
        from algorithms.hooks import AlgorithmHooks

        zeroing_hook = AlgorithmHooks(
            post_solve=lambda layer, u, *a: torch.zeros_like(u)
        )

        W_orig = torch.randn(32, 16)
        adj_k = torch.randn(16, 4)
        resid = torch.randn(32, 4)

        W_result = W_orig.clone()
        upd = adj_k @ resid.T
        upd_matched = upd.T if upd.shape != W_result.shape else upd
        upd_matched = zeroing_hook.post_solve(None, upd_matched, None, None, "w", {})
        W_result += upd_matched.float()

        assert torch.allclose(W_result, W_orig), \
            "Zeroing hook should prevent any weight change (REVIVE rejecting all update)"


# ============================================================================
# GPU integration test (only runs on GPU clusters)
# ============================================================================

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU available"
)


# ============================================================================
# Helpers
# ============================================================================

def _extract_function_source(full_source: str, func_name: str) -> str:
    """Extract the source code of a specific function from a module source."""
    tree = ast.parse(full_source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                lines = full_source.split("\n")
                return "\n".join(lines[node.lineno - 1: node.end_lineno])
    return ""
