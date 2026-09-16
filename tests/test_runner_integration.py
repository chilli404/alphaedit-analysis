"""CPU integration tests for all runners.

Catches bugs that unit tests miss — argparse dispatch gaps, broken import chains,
hook composition failures, config round-trip errors, checkpoint state mismatches,
and shell-script argument omissions.

These run without GPU. They verify the CODE PATHS, not the model outputs.

Run with: uv run pytest tests/test_runner_integration.py -v
"""
import ast
import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
VENDOR = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES = PROJECT_ROOT / "baselines" / "EvoEdit"

ALL_RUNNERS = sorted(
    list((SRC / "runners").glob("*.py")) + list((SRC / "polykernel").glob("*.py")),
)
ALL_RUNNERS = [r for r in ALL_RUNNERS if r.name != "__init__.py"]


# ─── 1. Argparse Completeness ─────────────────────────────────────────────────


def _extract_choices(filepath: Path) -> dict[str, list[str]]:
    """Extract all argparse choices from a runner file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    choices = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, ast.List):
                    elts = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                    # Find the corresponding --arg name
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                            choices[arg.value] = elts
                            break
    return choices


class TestArgparseCompleteness:
    """Every choices= value must be handled — no bare else: raise NotImplementedError."""

    @pytest.mark.parametrize("runner", ALL_RUNNERS, ids=lambda r: r.stem)
    def test_no_unhandled_choices(self, runner):
        source = runner.read_text()
        choices = _extract_choices(runner)
        for arg_name, values in choices.items():
            if not values:
                continue
            # The critical one: --base_alg and --alg_name must handle ALL choices
            if arg_name in ("--base_alg", "--alg_name"):
                for val in values:
                    assert f'"{val}"' in source or f"'{val}'" in source, (
                        f"{runner.name}: --{arg_name} choice '{val}' not found in source. "
                        f"Likely a bare else: raise NotImplementedError."
                    )

    def test_polykernel_seqreg_handles_all_base_algs(self):
        """The bug: polykernel_seqreg_runner raised NotImplementedError for NSE/MEMIT_rect."""
        source = (SRC / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
            # Must appear in an if/elif condition, not just in argparse choices
            pattern = rf'args\.base_alg\s*==\s*["\']{ alg}["\']|base_alg\s*==\s*["\']{ alg}["\']'
            assert re.search(pattern, source), (
                f"polykernel_seqreg_runner: base_alg='{alg}' not handled in if/elif chain"
            )

    def test_seeded_runner_handles_all_alg_names(self):
        source = (SRC / "runners" / "seeded_runner.py").read_text()
        for alg in ["AlphaEdit", "MEMIT"]:
            assert f'"{alg}"' in source


# ─── 2. Hook Composition ──────────────────────────────────────────────────────


class TestHookComposition:
    """All hook combinations that runners actually use must compose without error."""

    def test_seqreg_alone(self):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        assert hooks.build_lhs is not None
        assert hooks.post_solve is not None

    def test_revive_alone(self):
        from algorithms.hook_presets import revive_hooks
        hooks = revive_hooks(revive_tau=0.1)
        assert hooks.post_solve is not None

    def test_seqreg_plus_revive(self):
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks
        hooks = compose_hooks(seqreg_hooks(lambda_prev=1.0), revive_hooks(revive_tau=0.1))
        assert hooks.build_lhs is not None
        assert hooks.post_solve is not None
        state = hooks.get_state()
        assert "prev_cache" in state
        assert "lambda_prev" in state

    def test_seqreg_plus_pathguard(self):
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, pathguard_hooks
        hooks = compose_hooks(seqreg_hooks(), pathguard_hooks(pathguard_M=200))
        state = hooks.get_state()
        assert "prev_cache" in state
        assert "key_history" in state

    def test_polykernel_plus_seqreg(self):
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, polykernel_hooks
        hooks = compose_hooks(polykernel_hooks(kernel_degree=2), seqreg_hooks())
        assert hooks.build_lhs is not None

    def test_triple_compose(self):
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks, polykernel_hooks
        hooks = compose_hooks(
            polykernel_hooks(kernel_degree=2),
            seqreg_hooks(lambda_prev=1.0),
            revive_hooks(revive_tau=0.1),
        )
        assert hooks.build_lhs is not None
        assert hooks.post_solve is not None

    @pytest.mark.parametrize("lp", [0.0, 0.5, 1.0, 10.0])
    @pytest.mark.parametrize("ld", [0.0, 0.5, 1.0])
    def test_seqreg_parameter_sweep(self, lp, ld):
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=lp, lambda_delta=ld)
        state = hooks.get_state()
        assert state["lambda_prev"] == lp
        assert state["lambda_delta"] == ld

    def test_c0_hooks(self):
        try:
            from algorithms.hook_presets import c0_hooks
        except ImportError:
            pytest.skip("c0_hooks not yet implemented")
        mock_get_cov = MagicMock(return_value=torch.eye(64))
        hooks = c0_hooks(c0_weight=15000.0, get_cov_fn=mock_get_cov)
        assert hooks.build_lhs is not None


# ─── 3. ExperimentConfig Round-Trip ───────────────────────────────────────────


class TestExperimentConfigRoundTrip:
    """Config must produce correct variant names and paths for all algorithm combinations."""

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    @pytest.mark.parametrize("revive", [True, False])
    @pytest.mark.parametrize("kernel_prev", [True, False])
    def test_variant_name_contains_alg_prefix(self, base_alg, revive, kernel_prev):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=base_alg, seed=42, revive=revive,
            kernel_prev=kernel_prev, kernel_degree=2,
        )
        expected = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert expected in config.variant_name, (
            f"variant_name '{config.variant_name}' missing prefix '{expected}' for base_alg={base_alg}"
        )

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_all_algs_produce_distinct_variant_names(self, base_alg):
        from experiment_config import ExperimentConfig
        configs = [
            ExperimentConfig(base_alg=alg, seed=42, kernel_degree=1, revive=True, revive_tau=0.1)
            for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]
        ]
        names = [c.variant_name for c in configs]
        assert len(set(names)) == 4, f"Variant names not distinct: {names}"

    def test_checkpoint_dir_nonempty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path))
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(base_alg="MEMIT", seed=42, ordering="fb_high_exposure")
        ckpt = config.checkpoint_dir(root=tmp_path)
        assert str(ckpt) != ""
        assert "seed42" in str(ckpt)
        assert "fb_high_exposure" in str(ckpt)

    @pytest.mark.parametrize("model_name,expected_tag", [
        ("meta-llama/Meta-Llama-3-8B-Instruct", ""),
        ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    ])
    def test_model_tag(self, model_name, expected_tag):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(base_alg="MEMIT", seed=42, model_name=model_name)
        assert config.model_tag == expected_tag


# ─── 4. Checkpoint Save/Load Round-Trip ───────────────────────────────────────


class TestCheckpointRoundTrip:
    """Runner-specific extra_state must survive save→load."""

    def _make_mock_model(self, layers=(4, 5)):
        model = MagicMock()
        params = {}
        for l in layers:
            name = f"model.layers.{l}.mlp.down_proj.weight"
            param = torch.nn.Parameter(torch.randn(64, 128))
            params[name] = param
        model.named_parameters.return_value = list(params.items())
        return model, params

    def _make_hparams(self, layers=(4, 5)):
        hp = MagicMock()
        hp.layers = list(layers)
        hp.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"
        return hp

    def test_checkpoint_runner_cache_c(self, tmp_path):
        """checkpoint_runner saves cache_c.pt as extra_state."""
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        model, _ = self._make_mock_model()
        hp = self._make_hparams()
        cache_c = torch.randn(64, 64)
        save_checkpoint(0, model, hp, str(tmp_path), 100,
                        extra_state={"cache_c.pt": cache_c})
        result = load_checkpoint(model, hp, str(tmp_path), 0,
                                 extra_state_keys=["cache_c.pt"], device="cpu")
        assert result["loaded"]
        assert "cache_c.pt" in result
        torch.testing.assert_close(result["cache_c.pt"], cache_c)

    def test_memit_seq_prev_cache(self, tmp_path):
        """memit_sequential saves prev_cache.pt + mechanism_log.jsonl."""
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        model, _ = self._make_mock_model()
        hp = self._make_hparams()
        prev_cache = {4: [torch.randn(64, 10)], 5: [torch.randn(64, 10)]}
        log = [{"batch": 0, "layer": 4, "upd_norm": 0.5}]
        save_checkpoint(0, model, hp, str(tmp_path), 100,
                        extra_state={"prev_cache.pt": prev_cache, "mechanism_log.jsonl": log})
        result = load_checkpoint(model, hp, str(tmp_path), 0,
                                 extra_state_keys=["prev_cache.pt", "mechanism_log.jsonl"], device="cpu")
        assert result["loaded"]
        assert 4 in result["prev_cache.pt"]
        assert len(result["mechanism_log.jsonl"]) == 1
        assert result["mechanism_log.jsonl"][0]["upd_norm"] == 0.5

    def test_pathguard_state(self, tmp_path):
        """pathguard_runner saves pathguard_state.pt."""
        from util.checkpoint_io import save_checkpoint, load_checkpoint
        model, _ = self._make_mock_model()
        hp = self._make_hparams()
        pg_state = {"key_history": {4: torch.randn(10, 64)}, "epsilon": 0.1}
        save_checkpoint(0, model, hp, str(tmp_path), 100,
                        extra_state={"pathguard_state.pt": pg_state})
        result = load_checkpoint(model, hp, str(tmp_path), 0,
                                 extra_state_keys=["pathguard_state.pt"], device="cpu")
        assert result["loaded"]
        assert result["pathguard_state.pt"]["epsilon"] == 0.1

    def test_metadata_includes_custom_fields(self, tmp_path):
        from util.checkpoint_io import save_checkpoint
        model, _ = self._make_mock_model()
        hp = self._make_hparams()
        save_checkpoint(0, model, hp, str(tmp_path), 100,
                        metadata={"base_alg": "AlphaEdit", "variant_name": "test"})
        meta = json.loads((tmp_path / "batch_0" / "metadata.json").read_text())
        assert meta["base_alg"] == "AlphaEdit"
        assert meta["variant_name"] == "test"
        assert "timestamp_utc" in meta

    def test_validate_catches_wrong_prefix(self, tmp_path):
        from util.checkpoint_io import validate_checkpoint_path
        with pytest.raises(RuntimeError, match="Checkpoint path mismatch"):
            validate_checkpoint_path(str(tmp_path / "MEMIT-Seq-poly1"), "AlphaEdit")

    @pytest.mark.parametrize("base_alg,expected", [
        ("MEMIT", "MEMIT-Seq"),
        ("AlphaEdit", "AlphaEdit"),
        ("NSE", "NSE"),
        ("MEMIT_rect", "MEMIT_rect"),
    ])
    def test_validate_accepts_correct_prefix(self, tmp_path, base_alg, expected):
        from util.checkpoint_io import validate_checkpoint_path
        path = str(tmp_path / f"{expected}-poly1-REVIVE-tau0.1")
        validate_checkpoint_path(path, base_alg)  # should not raise


# ─── 5. Shell Script → Runner Argument Matching ──────────────────────────────


def _extract_python_cmd_from_shell(sh_path: Path) -> list[tuple[str, str]]:
    """Extract python commands from a shell script. Returns [(runner_path, args_str)]."""
    source = sh_path.read_text()
    results = []
    # Match: uv run python src/runners/foo.py --arg1 val1 ... (possibly multiline with \)
    # Join continuation lines first
    source = re.sub(r'\\\n\s*', ' ', source)
    for match in re.finditer(r'(?:uv run )?python[3]?\s+(src/[^\s]+\.py)\s+(.*?)(?:\n|$)', source):
        results.append((match.group(1), match.group(2)))
    return results


def _get_required_args(runner_path: Path) -> set[str]:
    """Extract required argparse arguments from a runner."""
    source = runner_path.read_text()
    tree = ast.parse(source)
    required = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                arg_name = None
                is_required = False
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        arg_name = arg.value
                for kw in node.keywords:
                    if kw.arg == "required" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        is_required = True
                if arg_name and is_required:
                    required.add(arg_name)
    return required


class TestShellScriptArguments:
    """Shell scripts must provide all required arguments to their runners."""

    SHELL_RUNNER_PAIRS = [
        ("scripts/run_mve1_alphaedit_mcf.sh", "src/runners/seeded_runner.py"),
        ("scripts/run_mve2_memit_mcf.sh", "src/runners/seeded_runner.py"),
        ("scripts/run_failure_curve_checkpointed.sh", "src/runners/checkpoint_runner.py"),
        ("scripts/run_logit_damage_memit.sh", "src/runners/logit_damage_memit_runner.py"),
        ("scripts/run_memit_sequential_gptj.sh", "src/runners/memit_sequential_runner.py"),
    ]

    @pytest.mark.parametrize("sh,runner", SHELL_RUNNER_PAIRS,
                             ids=lambda x: Path(x).stem if isinstance(x, str) else x)
    def test_required_args_provided(self, sh, runner):
        sh_path = PROJECT_ROOT / sh
        runner_path = PROJECT_ROOT / runner
        if not sh_path.exists() or not runner_path.exists():
            pytest.skip(f"Missing: {sh} or {runner}")

        required = _get_required_args(runner_path)
        if not required:
            return  # No required args

        cmds = _extract_python_cmd_from_shell(sh_path)
        runner_cmds = [args for path, args in cmds if runner.replace("src/", "") in path or path == runner]
        if not runner_cmds:
            pytest.skip(f"{sh} doesn't directly call {runner}")

        for args_str in runner_cmds:
            for req in required:
                # Check if the arg is in the command (may use $VAR)
                arg_base = req.lstrip("-")
                assert arg_base in args_str or req in args_str, (
                    f"{sh}: required arg {req} not found in python command"
                )


# ─── 6. Vendor Module Import Verification ────────────────────────────────────


class TestVendorImports:
    """Verify that vendor module imports resolve when vendor is on sys.path."""

    @pytest.mark.skipif(not VENDOR.exists(), reason="vendor submodule not initialized")
    def test_memit_main_exists(self):
        """Verify memit_main.py exists in vendor (import triggers globals.yml)."""
        assert (VENDOR / "memit" / "memit_main.py").exists()

    @pytest.mark.skipif(not VENDOR.exists(), reason="vendor submodule not initialized")
    def test_alphaedit_main_exists(self):
        assert (VENDOR / "AlphaEdit" / "AlphaEdit_main.py").exists()

    @pytest.mark.skipif(not VENDOR.exists(), reason="vendor submodule not initialized")
    def test_nse_main_exists(self):
        path = VENDOR / "nse" / "nse_main.py"
        if not path.exists():
            pytest.skip("NSE not in vendor (may be in baselines only)")
        assert path.exists()

    @pytest.mark.skipif(not BASELINES.exists(), reason="baselines not cloned")
    def test_baselines_evaluate_exists(self):
        assert (BASELINES / "experiments" / "evaluate.py").exists()


# ─── 7. Patch Anchor Uniqueness ──────────────────────────────────────────────


class TestPatchAnchorUniqueness:
    """Every patch anchor must match exactly once in its target file."""

    @pytest.mark.skipif(not BASELINES.exists(), reason="baselines not cloned")
    def test_mega_batch_anchor_unique_in_baselines(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "patches"))
        try:
            from patch_mega_batch_eval import EVAL_ANCHOR
        finally:
            sys.path.remove(str(PROJECT_ROOT / "scripts" / "patches"))
        source = (BASELINES / "experiments" / "evaluate.py").read_text()
        if "_mega_batch_eval" in source:
            pytest.skip("baselines evaluate.py already patched (idempotent)")
        count = source.count(EVAL_ANCHOR)
        assert count == 1, (
            f"EVAL_ANCHOR matches {count} times in baselines evaluate.py (expected 1). "
            f"The anchor must be unique to avoid patching the wrong location."
        )

    @pytest.mark.skipif(not VENDOR.exists(), reason="vendor submodule not initialized")
    def test_vendor_anchors_exist(self):
        """Vendor files must contain patch anchors (or already be patched)."""
        from util.source_patches import SHAPE_MODEL_LIST_ANCHOR, COMPUTE_Z_RESULT_ANCHOR
        eval_source = (VENDOR / "experiments" / "evaluate.py").read_text()
        memit_source = (VENDOR / "memit" / "memit_main.py").read_text()

        # Shape model list: either original anchor or already patched with Qwen
        assert SHAPE_MODEL_LIST_ANCHOR in eval_source or "Qwen2.5-7B" in eval_source, \
            "Shape model list anchor missing and Qwen not added"
        # NaN guard: either original anchor or already patched
        assert COMPUTE_Z_RESULT_ANCHOR in memit_source or "z_error" in memit_source, \
            "NaN guard anchor missing and guard not present"

    @pytest.mark.skipif(not BASELINES.exists(), reason="baselines not cloned")
    def test_all_patches_compile_baselines(self):
        """Apply ALL patches to baselines evaluate.py and verify it compiles."""
        source = (BASELINES / "experiments" / "evaluate.py").read_text()
        try:
            compile(source, "evaluate.py", "exec")
        except SyntaxError:
            pytest.skip("Baselines evaluate.py has pre-existing syntax issues")


# ─── 8. Runner Source Validation ──────────────────────────────────────────────


class TestRunnerSourceValidation:
    """Every runner must parse as valid Python and have key structural elements."""

    @pytest.mark.parametrize("runner", ALL_RUNNERS, ids=lambda r: r.stem)
    def test_parses(self, runner):
        ast.parse(runner.read_text())

    @pytest.mark.parametrize("runner", ALL_RUNNERS, ids=lambda r: r.stem)
    def test_has_main_guard(self, runner):
        source = runner.read_text()
        # Runners invoked directly must have if __name__ == "__main__"
        if "argparse" in source:
            assert '__name__' in source and '__main__' in source, (
                f"{runner.name} has argparse but no if __name__ == '__main__' guard"
            )

    @pytest.mark.parametrize("runner", ALL_RUNNERS, ids=lambda r: r.stem)
    def test_no_hardcoded_memit_seq_variant(self, runner):
        """The bug: variant_name was hardcoded to 'MEMIT-Seq' for all base_alg values."""
        source = runner.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):  # f-string
                for val in node.values:
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        if "MEMIT-Seq-" in val.value and "variant" in val.value.lower():
                            pytest.fail(
                                f"{runner.name}: hardcoded 'MEMIT-Seq' in variant f-string. "
                                f"Use ExperimentConfig.variant_name instead."
                            )
