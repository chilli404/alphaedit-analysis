"""CPU end-to-end integration tests.

Simulates the full pipeline WITHOUT a GPU by mocking models and tokenizers.
Verifies the complete chain: config → hooks → harness → checkpoint → eval.

These tests would have caught the two GPU-only bugs:
  1. patch_mega_batch_eval matching the wrong anchor
  2. polykernel_seqreg_runner raising NotImplementedError for NSE/MEMIT_rect

Run with: uv run pytest tests/test_endtoend_cpu.py -v
"""

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"


# ── Test Fixtures ──────────────────────────────────────────────────────────


def _make_dataset(n=20, num_edits=5):
    return [
        {
            "case_id": i,
            "requested_rewrite": {
                "prompt": "The capital of {} is",
                "subject": f"Country_{i}",
                "target_new": {"str": f"City_{i}"},
                "target_true": {"str": f"OldCity_{i}"},
            },
            "paraphrase_prompts": [f"What is the capital of Country_{i}?"],
            "neighborhood_prompts": [
                {"prompt": f"The capital of Neighbor_{i} is", "target": f"NeighCity_{i}"}
            ],
            "generation_prompts": [f"Country_{i} has its capital in"],
        }
        for i in range(n)
    ]


def _make_mock_model():
    model = MagicMock()
    params = {
        "model.layers.4.mlp.down_proj.weight": nn.Parameter(torch.randn(64, 64)),
        "model.layers.5.mlp.down_proj.weight": nn.Parameter(torch.randn(64, 64)),
    }
    model.named_parameters = MagicMock(return_value=list(params.items()))
    return model


def _make_mock_hparams():
    hp = MagicMock()
    hp.layers = [4, 5]
    hp.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"
    hp.mom2_update_weight = 15000.0
    return hp


def _make_mock_tok():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    return tok


def _make_apply_fn(call_log=None):
    """Returns an apply_fn that records calls and returns a mock result."""
    def apply_fn(model, tok, requests, hparams, **kwargs):
        if call_log is not None:
            call_log.append({
                "n_requests": len(requests),
                "kwargs": list(kwargs.keys()),
            })
        return (model, None)
    return apply_fn


# ── 1. Full Pipeline with Mock Model ──────────────────────────────────────


class TestFullPipeline:
    """Run the harness with mock everything and verify the chain works."""

    def test_correct_batch_count(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment
        result = run_experiment(
            model=_make_mock_model(), tok=_make_mock_tok(),
            hparams=_make_mock_hparams(), dataset=_make_dataset(20, 5),
            apply_fn=_make_apply_fn(), alg_name="MEMIT", num_edits=5,
            results_dir=tmp_path / "results",
            hooks=ExperimentHooks(should_eval=lambda _: False),
            max_batches=4,
        )
        assert result["batches_run"] == 4
        assert result["total_edits"] == 20

    def test_hooks_called_in_order(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment
        events = []

        def before(batch_idx, model, records, hparams):
            events.append(f"before_{batch_idx}")

        def after(batch_idx, model, records, hparams, result, exec_time):
            events.append(f"after_{batch_idx}")

        apply_log = []
        def tracked_apply(model, tok, requests, hparams, **kw):
            events.append(f"apply_{len(apply_log)}")
            apply_log.append(1)
            return (model, None)

        run_experiment(
            model=_make_mock_model(), tok=_make_mock_tok(),
            hparams=_make_mock_hparams(), dataset=_make_dataset(10, 5),
            apply_fn=tracked_apply, alg_name="MEMIT", num_edits=5,
            results_dir=tmp_path / "results",
            hooks=ExperimentHooks(
                before_edit=before, after_edit=after,
                should_eval=lambda _: False,
            ),
            max_batches=2,
        )
        assert events == ["before_0", "apply_0", "after_0", "before_1", "apply_1", "after_1"]

    def test_results_dir_created(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment
        results_dir = tmp_path / "deep" / "nested" / "results"
        run_experiment(
            model=_make_mock_model(), tok=_make_mock_tok(),
            hparams=_make_mock_hparams(), dataset=_make_dataset(10, 5),
            apply_fn=_make_apply_fn(), alg_name="MEMIT", num_edits=5,
            results_dir=results_dir,
            hooks=ExperimentHooks(should_eval=lambda _: False),
            max_batches=1,
        )
        assert results_dir.exists()

    def test_request_flattening(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment
        received = []

        def capture(model, tok, requests, hparams, **kw):
            received.extend(requests)
            return (model, None)

        run_experiment(
            model=_make_mock_model(), tok=_make_mock_tok(),
            hparams=_make_mock_hparams(), dataset=_make_dataset(5, 5),
            apply_fn=capture, alg_name="MEMIT", num_edits=5,
            results_dir=tmp_path / "results",
            hooks=ExperimentHooks(should_eval=lambda _: False),
            max_batches=1,
        )
        assert len(received) == 5
        for r in received:
            assert "target_new" in r, "request must be flattened"
            assert "requested_rewrite" not in r


# ── 2. Checkpoint Resume Pipeline ─────────────────────────────────────────


class TestCheckpointResume:
    """Save checkpoint → load → verify resume works."""

    def test_save_creates_files(self, tmp_path):
        from util.checkpoint_io import save_checkpoint
        model = _make_mock_model()
        hparams = _make_mock_hparams()
        ckpt_dir = tmp_path / "ckpt"

        batch_dir = save_checkpoint(
            batch_idx=9, model=model, hparams=hparams,
            ckpt_dir=str(ckpt_dir), num_edits=100,
            extra_state={"prev_cache.pt": {"layer4": [torch.randn(64, 10)]}},
            metadata={"lambda_prev": 1.0, "seed": 42},
        )
        assert (batch_dir / "model_weights.pt").exists()
        assert (batch_dir / "prev_cache.pt").exists()
        assert (batch_dir / "metadata.json").exists()

        meta = json.loads((batch_dir / "metadata.json").read_text())
        assert meta["batch_idx"] == 9
        assert meta["total_edits"] == 1000
        assert meta["lambda_prev"] == 1.0

    def test_load_restores_weights(self, tmp_path):
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        model = _make_mock_model()
        hparams = _make_mock_hparams()
        ckpt_dir = tmp_path / "ckpt"

        save_checkpoint(batch_idx=0, model=model, hparams=hparams,
                        ckpt_dir=str(ckpt_dir), num_edits=10)

        model2 = _make_mock_model()
        result = load_checkpoint(model=model2, hparams=hparams,
                                 ckpt_dir=str(ckpt_dir), batch_idx=0, device="cpu")
        assert result["loaded"] is True

    def test_find_latest_checkpoint(self, tmp_path):
        from util.checkpoint_io import save_checkpoint, find_latest_checkpoint
        model = _make_mock_model()
        hparams = _make_mock_hparams()
        ckpt_dir = tmp_path / "ckpt"

        for i in [9, 19, 29]:
            save_checkpoint(batch_idx=i, model=model, hparams=hparams,
                            ckpt_dir=str(ckpt_dir), num_edits=100)

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert latest[0] == 29

    def test_extra_state_roundtrip(self, tmp_path):
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        model = _make_mock_model()
        hparams = _make_mock_hparams()
        ckpt_dir = tmp_path / "ckpt"

        original_cache = {"4": [torch.randn(64, 10)], "5": [torch.randn(64, 8)]}
        save_checkpoint(batch_idx=0, model=model, hparams=hparams,
                        ckpt_dir=str(ckpt_dir), num_edits=10,
                        extra_state={"prev_cache.pt": original_cache})

        result = load_checkpoint(model=_make_mock_model(), hparams=hparams,
                                 ckpt_dir=str(ckpt_dir), batch_idx=0,
                                 extra_state_keys=["prev_cache.pt"], device="cpu")
        loaded_cache = result["prev_cache.pt"]
        assert set(loaded_cache.keys()) == set(original_cache.keys())


# ── 3. ExperimentConfig Path Chain ────────────────────────────────────────


class TestExperimentConfigChain:
    """ExperimentConfig → checkpoint_dir → save → load."""

    @pytest.mark.parametrize("base_alg,expected_prefix", [
        ("MEMIT", "MEMIT-Seq"),
        ("AlphaEdit", "AlphaEdit"),
        ("NSE", "NSE"),
        ("MEMIT_rect", "MEMIT_rect"),
    ])
    def test_checkpoint_dir_contains_prefix(self, tmp_path, base_alg, expected_prefix):
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=base_alg, seed=42, ordering="fb_high_exposure",
            kernel_degree=1, revive=True, revive_tau=0.1,
        )
        ckpt_dir = config.checkpoint_dir(root=tmp_path)
        assert expected_prefix in str(ckpt_dir)

    def test_all_base_algs_distinct(self, tmp_path):
        from util.experiment_config import ExperimentConfig
        paths = set()
        for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(base_alg=alg, seed=42, kernel_degree=1)
            paths.add(str(config.checkpoint_dir(root=tmp_path)))
        assert len(paths) == 4

    def test_save_to_config_dir(self, tmp_path):
        from util.experiment_config import ExperimentConfig
        from util.checkpoint_io import save_checkpoint
        config = ExperimentConfig(base_alg="AlphaEdit", seed=42, kernel_degree=1,
                                   revive=True, revive_tau=0.1)
        ckpt_dir = config.checkpoint_dir(root=tmp_path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(batch_idx=9, model=_make_mock_model(),
                        hparams=_make_mock_hparams(), ckpt_dir=str(ckpt_dir),
                        num_edits=100, base_alg="AlphaEdit")
        assert (ckpt_dir / "batch_9" / "model_weights.pt").exists()

    def test_validate_catches_mismatch(self, tmp_path):
        from util.checkpoint_io import save_checkpoint
        ckpt_dir = tmp_path / "MEMIT-Seq-wrong-path" / "seed42"
        ckpt_dir.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="mismatch"):
            save_checkpoint(batch_idx=0, model=_make_mock_model(),
                            hparams=_make_mock_hparams(), ckpt_dir=str(ckpt_dir),
                            num_edits=10, base_alg="AlphaEdit")


# ── 4. Hook State Accumulation ────────────────────────────────────────────


class TestHookStateAccumulation:
    """Verify seqreg_hooks accumulates state across batches."""

    def test_prev_cache_grows(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        state = hooks.get_state()

        for batch_idx in range(3):
            layer_ks = torch.randn(64, 10, dtype=torch.float64)
            cov = torch.eye(64, dtype=torch.float64)
            hp = _make_mock_hparams()

            if hooks.build_lhs:
                hooks.build_lhs(4, layer_ks, cov, hp, state)
            if hooks.post_solve:
                upd = torch.randn(64, 64, dtype=torch.float64)
                hooks.post_solve(4, upd, layer_ks, layer_ks, "w", state)

        assert 4 in state["prev_cache"]
        assert len(state["prev_cache"][4]) == 3

    def test_lambda_prev_adds_kprev_term(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        state = hooks.get_state()

        # First batch — no K_prev yet
        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        cov = torch.eye(64, dtype=torch.float64)
        hp = _make_mock_hparams()
        lhs1 = hooks.build_lhs(4, layer_ks, cov, hp, state)

        # Store keys
        if hooks.post_solve:
            hooks.post_solve(4, torch.randn(64, 64, dtype=torch.float64),
                             layer_ks, layer_ks, "w", state)

        # Second batch — now K_prev exists
        lhs2 = hooks.build_lhs(4, layer_ks, cov, hp, state)

        # LHS2 should be larger (K_prev term added)
        assert torch.linalg.norm(lhs2).item() > torch.linalg.norm(lhs1).item()

    def test_lambda_delta_adds_ridge(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks_no_ridge = seqreg_hooks(lambda_prev=0.0, lambda_delta=0.0)
        hooks_ridge = seqreg_hooks(lambda_prev=0.0, lambda_delta=1.0)

        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        cov = torch.eye(64, dtype=torch.float64)
        hp = _make_mock_hparams()

        lhs_no = hooks_no_ridge.build_lhs(4, layer_ks, cov, hp, hooks_no_ridge.get_state())
        lhs_yes = hooks_ridge.build_lhs(4, layer_ks, cov, hp, hooks_ridge.get_state())

        diff = lhs_yes - lhs_no
        # Ridge adds lambda * I — diagonal should be ~1.0
        diag_diff = torch.diag(diff)
        assert torch.allclose(diag_diff, torch.ones_like(diag_diff), atol=1e-6)


# ── 5. compose_hooks End-to-End ───────────────────────────────────────────


class TestComposeHooksEndToEnd:
    """Composed hooks should apply both effects."""

    def test_seqreg_plus_revive(self):
        from algorithms.hook_presets import seqreg_hooks, revive_hooks, compose_hooks

        combined = compose_hooks(
            seqreg_hooks(lambda_prev=1.0),
            revive_hooks(revive_tau=0.5, revive_svd_device="cpu"),
        )
        state = combined.get_state()
        assert "prev_cache" in state  # from seqreg

        # build_lhs from seqreg
        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        cov = torch.eye(64, dtype=torch.float64)
        hp = _make_mock_hparams()
        lhs = combined.build_lhs(4, layer_ks, cov, hp, state)
        assert lhs.shape == (64, 64)

    def test_compose_preserves_all_hooks(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks

        def build_a(l, k, c, h, s): return c
        def post_a(l, u, a, k, w, s): return u * 2
        def post_b(l, u, a, k, w, s): return u + 1

        combined = compose_hooks(
            AlgorithmHooks(build_lhs=build_a, post_solve=post_a),
            AlgorithmHooks(post_solve=post_b),
        )
        assert combined.build_lhs is not None
        assert combined.post_solve is not None

        # post_solve chains: first doubles, then adds 1
        upd = torch.tensor(3.0)
        result = combined.post_solve(0, upd, None, None, "w", {})
        assert result.item() == 7.0  # (3 * 2) + 1


# ── 6. Method Registry ───────────────────────────────────────────────────


class TestMethodRegistry:
    """Every registered method resolves to a valid runner."""

    def test_all_methods_resolve(self):
        from method_registry import METHODS, get_method
        for name in METHODS:
            method = get_method(name)
            assert method.name == name
            assert method.runner, f"{name} has no runner"

    def test_all_runners_exist(self):
        from method_registry import METHODS
        for name, method in METHODS.items():
            if method.shell:
                path = PROJECT_ROOT / method.runner  # shell runners include scripts/ in path
            else:
                path = PROJECT_ROOT / "src" / method.runner
            assert path.exists(), f"Runner for {name} not found: {path}"

    def test_list_methods_not_empty(self):
        from method_registry import list_methods
        methods = list_methods()
        assert len(methods) >= 5


# ── 7. Runner Dispatch: No NotImplementedError ────────────────────────────


class TestRunnerDispatchComplete:
    """Every argparse choice must be handled without NotImplementedError."""

    def _find_choices(self, source: str, arg_name: str) -> list[str]:
        """Extract choices list for a given argparse argument."""
        # Match: parser.add_argument("--base_alg", ..., choices=[...])
        # The arg and choices may be on the same line or nearby
        for line in source.split("\n"):
            if f"--{arg_name}" in line and "choices=" in line:
                match = re.search(r'choices=\[([^\]]+)\]', line)
                if match:
                    raw = match.group(1)
                    return [c.strip().strip('"').strip("'") for c in raw.split(",")]
        return []

    def _find_dispatch_block(self, source: str) -> str:
        """Extract the if/elif block that dispatches on base_alg."""
        lines = source.split("\n")
        block = []
        in_block = False
        for line in lines:
            if "args.base_alg" in line and ("==" in line or "if " in line):
                in_block = True
            if in_block:
                block.append(line)
                if line.strip().startswith("else:") or (block and not line.strip() and len(block) > 3):
                    break
        return "\n".join(block)

    def test_polykernel_seqreg_handles_all_base_algs(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        choices = self._find_choices(source, "base_alg")
        assert len(choices) >= 4, f"Expected >=4 choices, got {choices}"

        for choice in choices:
            assert f'"{choice}"' in source or f"'{choice}'" in source, (
                f"base_alg={choice} not handled in polykernel_seqreg_runner"
            )
        # Must NOT have a bare raise NotImplementedError for any valid choice
        assert "NotImplementedError" not in source or "Unknown base_alg" in source, (
            "polykernel_seqreg_runner has NotImplementedError — all choices must be handled"
        )


# ── 8. Shell Script → Python Command Validation ──────────────────────────


class TestShellScriptChain:
    """Shell scripts must reference existing runners and valid arguments."""

    SCRIPTS = [
        "scripts/run_mve1_alphaedit_mcf.sh",
        "scripts/run_matched_ordering.sh",
        "scripts/run_revive_baseline.sh",
    ]

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_script_exists(self, script):
        path = PROJECT_ROOT / script
        assert path.exists()

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_references_existing_runner(self, script):
        source = (PROJECT_ROOT / script).read_text()
        # Find python invocations
        py_calls = re.findall(r'(?:uv run )?python[3]?\s+(\S+\.py)', source)
        for call in py_calls:
            # Resolve relative to project root
            runner = PROJECT_ROOT / call.lstrip("./")
            if not runner.exists():
                runner = PROJECT_ROOT / "src" / call.lstrip("./")
            assert runner.exists(), f"{script} references non-existent runner: {call}"

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_no_deleted_imports(self, script):
        source = (PROJECT_ROOT / script).read_text()
        deleted = [
            "model_download", "build_polykernel_seqreg_script",
            "cache_mitigation_batch_runner", "memit_sequential_batch_runner",
        ]
        for d in deleted:
            assert d not in source, f"{script} references deleted module: {d}"


# ── 9. Patch Anchor Uniqueness ────────────────────────────────────────────


class TestPatchAnchorUniqueness:
    """Every patch anchor must match exactly once in its target file."""

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines/ not cloned")
    def test_mega_batch_anchor_unique_in_baselines(self):
        import importlib
        patches_dir = PROJECT_ROOT / "scripts" / "patches"
        import sys
        sys.path.insert(0, str(patches_dir))
        import patch_mega_batch_eval
        importlib.reload(patch_mega_batch_eval)

        eval_path = BASELINES_ROOT / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines evaluate.py not found")
        source = eval_path.read_text()
        if "_mega_batch_eval" in source:
            pytest.skip("baselines evaluate.py already patched — anchor test needs unpatched file")
        count = source.count(patch_mega_batch_eval.EVAL_ANCHOR)
        assert count == 1, (
            f"EVAL_ANCHOR matched {count} times (expected 1). "
            f"Anchor: {patch_mega_batch_eval.EVAL_ANCHOR[:60]}..."
        )

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines/ not cloned")
    def test_all_patches_compile_baselines(self):
        """Apply all patches to baselines evaluate.py and verify it compiles."""
        eval_path = BASELINES_ROOT / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines evaluate.py not found")

        original = eval_path.read_text()
        try:
            import subprocess
            result = subprocess.run(
                ["uv", "run", "python", str(PROJECT_ROOT / "scripts" / "patches" / "apply_all.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30,
            )
            patched = eval_path.read_text()
            compile(patched, str(eval_path), "exec")
        finally:
            eval_path.write_text(original)

    def test_vendor_anchor_exists(self):
        """Vendor evaluate.py must contain the CUDA patch target."""
        eval_path = VENDOR_ROOT / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("vendor submodule not initialized")
        source = eval_path.read_text()
        assert 'os.environ["CUDA_VISIBLE_DEVICES"]' in source
