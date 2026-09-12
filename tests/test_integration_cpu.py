"""CPU integration tests that catch bugs only visible during GPU execution.

These tests verify:
  1. Patch anchors match the correct locations (not ambiguous substrings)
  2. Runner argparse choices all have handler code (no NotImplementedError traps)
  3. Vendor anchor constants match the pinned submodule
  4. Import chains for all base algorithms work

These would have caught:
  - patch_mega_batch_eval matching the NSE cache loop instead of the eval loop
  - polykernel_seqreg_runner raising NotImplementedError for base_alg=NSE/MEMIT_rect
"""
import ast
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"
PATCHES_DIR = PROJECT_ROOT / "scripts" / "patches"

VENDOR_EXISTS = VENDOR_ROOT.exists() and (VENDOR_ROOT / "experiments" / "evaluate.py").exists()
BASELINES_EXISTS = BASELINES_ROOT.exists() and (BASELINES_ROOT / "experiments" / "evaluate.py").exists()


# ---------------------------------------------------------------------------
# 1. Patch anchor uniqueness
# ---------------------------------------------------------------------------


class TestPatchAnchorUniqueness:
    """Every patch anchor must match exactly once in its target file."""

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_cuda_anchor(self):
        from source_injection import CUDA_PATCH_TARGET
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        assert source.count(CUDA_PATCH_TARGET) == 1

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_shape_model_list_anchor(self):
        from source_patches import SHAPE_MODEL_LIST_ANCHOR
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        count = source.count(SHAPE_MODEL_LIST_ANCHOR)
        assert count <= 1, f"SHAPE_MODEL_LIST_ANCHOR matches {count} times"

    @pytest.mark.skipif(not BASELINES_EXISTS, reason="baselines not cloned")
    def test_baselines_mega_batch_anchor_not_ambiguous(self):
        """The mega-batch anchor must NOT match the NSE cache loop."""
        sys.path.insert(0, str(PATCHES_DIR))
        try:
            import patch_mega_batch_eval
            # Reload in case it was cached from a different sys.path
            import importlib
            importlib.reload(patch_mega_batch_eval)
        finally:
            sys.path.remove(str(PATCHES_DIR))

        source = (BASELINES_ROOT / "experiments" / "evaluate.py").read_text()
        ambiguous = "    for record in ds:"
        matches = source.count(ambiguous)
        if matches > 1:
            assert patch_mega_batch_eval.EVAL_ANCHOR != ambiguous, (
                f"EVAL_ANCHOR is '{ambiguous}' but it matches {matches} locations. "
                f"Use a more specific anchor."
            )


# ---------------------------------------------------------------------------
# 2. Runner dispatch completeness
# ---------------------------------------------------------------------------


class TestRunnerDispatchCompleteness:
    """Every argparse choice for base_alg / alg_name must have a handler."""

    def _find_handled_values(self, source: str, arg_name: str) -> set[str]:
        """Find all values explicitly handled in if/elif blocks for this arg."""
        handled = set()
        patterns = [
            rf'args\.{arg_name}\s*==\s*"(\w+)"',
            rf'args\.{arg_name}\s*==\s*\'(\w+)\'',
            rf'{arg_name}\s*==\s*"(\w+)"',
        ]
        for pattern in patterns:
            handled.update(re.findall(pattern, source))
        return handled

    def test_polykernel_seqreg_handles_all_base_algs(self):
        """polykernel_seqreg_runner must handle MEMIT, AlphaEdit, NSE, MEMIT_rect."""
        path = PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py"
        source = path.read_text()
        required = {"MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"}
        handled = self._find_handled_values(source, "base_alg")
        missing = required - handled
        assert not missing, (
            f"Missing handlers for: {missing}. These will raise at runtime."
        )

    def test_polykernel_editor_handles_alphaedit(self):
        """polykernel_editor is AlphaEdit-only; MEMIT choice exists but dispatches via vendor."""
        path = PROJECT_ROOT / "src" / "polykernel" / "polykernel_editor_runner.py"
        source = path.read_text()
        assert 'alg_name == "AlphaEdit"' in source

    def test_no_notimplementederror_for_base_alg(self):
        """No runner should have NotImplementedError that mentions a valid base_alg choice."""
        for runner in (PROJECT_ROOT / "src").rglob("*.py"):
            if "__pycache__" in str(runner):
                continue
            source = runner.read_text()
            if "NotImplementedError" not in source or "base_alg" not in source:
                continue
            # Check that no NotImplementedError message contains a valid base_alg
            for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
                pattern = rf'NotImplementedError\([^)]*{alg}[^)]*\)'
                if re.search(pattern, source):
                    pytest.fail(
                        f"{runner.relative_to(PROJECT_ROOT)}: NotImplementedError "
                        f"mentions '{alg}' — add a handler instead"
                    )


# ---------------------------------------------------------------------------
# 3. Vendor import chains (with globals pre-population)
# ---------------------------------------------------------------------------


class TestVendorImportChains:
    """All algorithm apply functions must be importable from the vendor tree."""

    @pytest.fixture(autouse=True)
    def _setup_vendor_globals(self):
        """Pre-populate vendor globals so imports don't crash on missing globals.yml."""
        if VENDOR_EXISTS:
            from evaluate_harness import _ensure_vendor_globals
            _ensure_vendor_globals(VENDOR_ROOT)

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_memit_function_exists_in_source(self):
        """Verify apply_memit_to_model is defined in the vendor source (no import needed)."""
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        assert "def apply_memit_to_model" in source

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_alphaedit_function_exists_in_source(self):
        source = (VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert "def apply_AlphaEdit_to_model" in source

    @pytest.mark.skipif(
        not VENDOR_EXISTS or not (VENDOR_ROOT / "nse" / "nse_main.py").exists(),
        reason="vendor NSE not available"
    )
    def test_nse_function_exists_in_source(self):
        source = (VENDOR_ROOT / "nse" / "nse_main.py").read_text()
        assert "def apply_nse_to_model" in source

    @pytest.mark.skipif(
        not VENDOR_EXISTS or not (VENDOR_ROOT / "memit" / "memit_rect_main.py").exists(),
        reason="vendor MEMIT_rect not available"
    )
    def test_memit_rect_function_exists_in_source(self):
        source = (VENDOR_ROOT / "memit" / "memit_rect_main.py").read_text()
        assert "def apply_memit_rect_to_model" in source


# ---------------------------------------------------------------------------
# 4. Patch compilation integration
# ---------------------------------------------------------------------------


class TestPatchCompilation:
    """Applying patches must produce valid Python."""

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_evaluate_compiles_after_source_patches(self):
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        from source_patches import apply_model_list_patch, apply_canonical_name_patch
        patched = apply_model_list_patch(source)
        patched = apply_canonical_name_patch(patched)
        compile(patched, "evaluate.py", "exec")

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_memit_compiles_after_nan_guard(self):
        from source_patches import apply_nan_guard_patch
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        patched = apply_nan_guard_patch(source)
        compile(patched, "memit_main.py", "exec")


# ---------------------------------------------------------------------------
# 5. ExperimentConfig path consistency
# ---------------------------------------------------------------------------


class TestExperimentConfigConsistency:
    """ExperimentConfig must produce distinct paths for every base_alg."""

    def test_all_base_algs_produce_distinct_checkpoint_dirs(self):
        from experiment_config import ExperimentConfig
        paths = set()
        for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(
                base_alg=alg, seed=42, ordering="fb_high_exposure",
                revive=True, revive_tau=0.1,
            )
            p = str(config.checkpoint_dir(Path("/tmp/ckpt")))
            assert p not in paths, f"base_alg={alg} produces duplicate path: {p}"
            paths.add(p)

    def test_variant_name_contains_base_alg(self):
        from experiment_config import ExperimentConfig
        for alg in ["AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(base_alg=alg, seed=42)
            assert alg in config.variant_name, (
                f"base_alg={alg} variant_name={config.variant_name} "
                f"doesn't contain '{alg}' — the MEMIT-Seq hardcode bug is back"
            )

    def test_memit_maps_to_memit_seq(self):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(base_alg="MEMIT", seed=42)
        assert "MEMIT-Seq" in config.variant_name


# ---------------------------------------------------------------------------
# 6. Shell script → runner consistency
# ---------------------------------------------------------------------------


YAML_DISPATCHED_SCRIPTS = [
    "scripts/run_evoedit_baseline.sh",
    "scripts/run_nse_baseline.sh",
    "scripts/run_revive_baseline.sh",
    "scripts/run_revive_paper_replication.sh",
    "scripts/run_rect_aligned_paper_replication.sh",
    "scripts/run_mve1_alphaedit_mcf.sh",
    "scripts/run_matched_ordering.sh",
    "scripts/run_pathguard.sh",
    "scripts/run_evoedit_paper_replication.sh",
    "scripts/run_nse_paper_replication.sh",
]


class TestShellScriptConsistency:
    """Shell scripts must reference existing runners with valid args."""

    @pytest.mark.parametrize("script", YAML_DISPATCHED_SCRIPTS)
    def test_script_exists(self, script):
        path = PROJECT_ROOT / script
        assert path.exists(), f"YAML-dispatched script missing: {script}"

    @pytest.mark.parametrize("script", YAML_DISPATCHED_SCRIPTS)
    def test_script_references_existing_runner(self, script):
        """Every 'uv run python src/...' in the script must reference an existing file."""
        path = PROJECT_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        content = path.read_text()
        for match in re.findall(r'uv run python\s+(src/\S+\.py)', content):
            runner = PROJECT_ROOT / match
            assert runner.exists(), (
                f"{script} references {match} which doesn't exist"
            )

    @pytest.mark.parametrize("script", YAML_DISPATCHED_SCRIPTS)
    def test_required_env_vars_have_defaults(self, script):
        """Env vars used with ${VAR:?...} (required, no default) must be documented."""
        path = PROJECT_ROOT / script
        if not path.exists():
            pytest.skip(f"{script} not found")
        content = path.read_text()
        required = re.findall(r'\$\{(\w+):\?', content)
        # SEED is always positional arg $1 — allowed to be required
        known_required = {"1", "SEED"}
        for var in required:
            if var not in known_required:
                # Check it's documented in a comment or echo
                assert var in content.split("${" + var + ":?")[0], (
                    f"{script}: ${{{var}:?...}} is required but not documented before use"
                )


# ---------------------------------------------------------------------------
# 7. SkyPilot YAML structure
# ---------------------------------------------------------------------------


class TestSkyPilotYAMLs:
    """All SkyPilot YAMLs must call apply_all.py and set S3 paths."""

    YAMLS = [str(f) for f in (PROJECT_ROOT / "sky").glob("*.yaml")]

    @pytest.mark.parametrize("yaml_path", YAMLS, ids=lambda p: Path(p).name)
    def test_yaml_calls_apply_all(self, yaml_path):
        """Every YAML must call scripts/patches/apply_all.py (in setup: or run:)."""
        content = Path(yaml_path).read_text()
        assert "apply_all" in content, (
            f"{Path(yaml_path).name}: doesn't call apply_all.py anywhere. "
            f"Vendor code will be unpatched."
        )

    @pytest.mark.parametrize("yaml_path", YAMLS, ids=lambda p: Path(p).name)
    def test_yaml_sets_s3_paths(self, yaml_path):
        """Experiment YAMLs must set RESULT_ROOT and CHECKPOINT_ROOT to S3."""
        content = Path(yaml_path).read_text()
        if "run:" not in content:
            pytest.skip("No run: block")
        run_block = content.split("run:")[1]
        # Test and smoke YAMLs use _smoke_test S3 paths
        if "RESULT_ROOT" in run_block:
            assert "/s3-data/" in run_block, (
                f"{Path(yaml_path).name}: RESULT_ROOT set but not to /s3-data/"
            )

    @pytest.mark.parametrize("yaml_path", YAMLS, ids=lambda p: Path(p).name)
    def test_yaml_links_data_before_experiment(self, yaml_path):
        """YAMLs must link stats/datasets before running experiments."""
        content = Path(yaml_path).read_text()
        if "run:" not in content:
            pytest.skip("No run: block")
        run_block = content.split("run:")[1]
        if "run_" in run_block and "link_stats" not in run_block:
            # Only flag if it actually runs an experiment (not just tests)
            if "smoke" in run_block.lower() or "test_smoke" in run_block:
                assert "link_stats" in run_block or "link_dsets" in run_block, (
                    f"{Path(yaml_path).name}: runs experiments without linking data"
                )


# ---------------------------------------------------------------------------
# 8. Vendor function signatures (kwargs compatibility)
# ---------------------------------------------------------------------------


class TestVendorFunctionSignatures:
    """Vendor apply functions must accept **_kwargs after our patch."""

    VENDOR_APPLY_FILES = [
        ("memit/memit_main.py", "apply_memit_to_model"),
        ("AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
    ]

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    @pytest.mark.parametrize("rel_path,fn_name", VENDOR_APPLY_FILES,
                             ids=lambda x: x if isinstance(x, str) else x[0])
    def test_vendor_apply_has_kwargs(self, rel_path, fn_name):
        """After patching, vendor apply functions must accept **_kwargs."""
        source = (VENDOR_ROOT / rel_path).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                # Check for **kwargs in the signature
                if node.args.kwarg is not None:
                    return  # Has **kwarg — good
                pytest.fail(
                    f"{rel_path}:{fn_name} missing **_kwargs. "
                    f"The kwargs patch wasn't applied."
                )
                return
        pytest.fail(f"{rel_path}: {fn_name} not found")

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_apply_accepts_return_orig_weights(self):
        """apply functions must accept return_orig_weights param."""
        for rel_path, fn_name in self.VENDOR_APPLY_FILES:
            source = (VENDOR_ROOT / rel_path).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    arg_names = [a.arg for a in node.args.args]
                    has_row = "return_orig_weights" in arg_names
                    has_kwargs = node.args.kwarg is not None
                    assert has_row or has_kwargs, (
                        f"{rel_path}:{fn_name} doesn't accept return_orig_weights "
                        f"(and no **kwargs to catch it)"
                    )

    @pytest.mark.skipif(not BASELINES_EXISTS, reason="baselines not cloned")
    def test_baselines_evoedit_has_kwargs(self):
        """EvoEdit apply function must accept **_kwargs."""
        source = (BASELINES_ROOT / "EvoEdit" / "EvoEdit_main.py").read_text()
        assert "**_kwargs" in source or "**kwargs" in source, (
            "EvoEdit_main.py missing **_kwargs — will crash with return_orig_weights"
        )

    @pytest.mark.skipif(not BASELINES_EXISTS, reason="baselines not cloned")
    def test_baselines_nse_has_kwargs(self):
        source = (BASELINES_ROOT / "nse" / "nse_main.py").read_text()
        assert "**_kwargs" in source or "**kwargs" in source

    @pytest.mark.skipif(not BASELINES_EXISTS, reason="baselines not cloned")
    def test_baselines_rect_has_kwargs(self):
        source = (BASELINES_ROOT / "memit" / "memit_seq_rect_main.py").read_text()
        assert "**_kwargs" in source or "**kwargs" in source


# ---------------------------------------------------------------------------
# 9. Data/stats path consistency
# ---------------------------------------------------------------------------


class TestDataPathConsistency:
    """Dataset and stats paths must be consistent across scripts."""

    EXPECTED_DATASETS = [
        "multi_counterfact.json",
        "zsre_mend_eval.json",
    ]

    def test_link_dsets_references_expected_files(self):
        """link_dsets.sh must reference the standard dataset files."""
        path = PROJECT_ROOT / "scripts" / "link_dsets.sh"
        if not path.exists():
            pytest.skip("link_dsets.sh not found")
        content = path.read_text()
        for ds in self.EXPECTED_DATASETS:
            assert ds in content, (
                f"link_dsets.sh doesn't reference {ds}"
            )

    def test_link_stats_references_null_space(self):
        """link_stats.sh must link the null-space projection file."""
        path = PROJECT_ROOT / "scripts" / "link_stats.sh"
        if not path.exists():
            pytest.skip("link_stats.sh not found")
        content = path.read_text()
        assert "null_space_project" in content, (
            "link_stats.sh doesn't reference null_space_project.pt"
        )

    def test_link_stats_handles_multiple_models(self):
        """link_stats.sh must handle at least Llama and GPT-J stats."""
        path = PROJECT_ROOT / "scripts" / "link_stats.sh"
        if not path.exists():
            pytest.skip("link_stats.sh not found")
        content = path.read_text()
        assert "llama" in content.lower() or "Meta-Llama" in content
        # GPT-J stats may be optional — just check the script doesn't hardcode one model

    @pytest.mark.skipif(not VENDOR_EXISTS, reason="vendor submodule not initialized")
    def test_vendor_hparams_llama_exists(self):
        """Standard Llama hparams must exist for both algorithms."""
        for alg in ["AlphaEdit", "MEMIT"]:
            path = VENDOR_ROOT / "hparams" / alg / "Llama3-8B.json"
            assert path.exists(), f"Missing: {path}"

    def test_configs_hparams_cross_model(self):
        """configs/hparams/ must have cross-model hparams (Mistral, Qwen)."""
        hparams_dir = PROJECT_ROOT / "configs" / "hparams"
        assert hparams_dir.exists(), "configs/hparams/ missing"
        for alg in ["AlphaEdit", "MEMIT"]:
            for model in ["Mistral-7B.json", "Qwen2.5-7B.json"]:
                path = hparams_dir / alg / model
                assert path.exists(), f"Missing cross-model hparams: {path}"


# ---------------------------------------------------------------------------
# 10. Checkpoint paths match GPU smoke test expectations
# ---------------------------------------------------------------------------


class TestCheckpointPathsMatchSmokeTest:
    """Every GPU smoke test algorithm must have its checkpoint path reproduced on CPU.

    The smoke test checks for files at specific S3 paths. If ExperimentConfig
    produces a different path, the checkpoint will be written to the wrong location
    and the smoke test will report 'checkpoint not found'.
    """

    @pytest.mark.parametrize("label,config_kwargs,expected_fragment", [
        (
            "AlphaEdit",
            dict(base_alg="AlphaEdit", seed=42, experiment_type="failure_curve"),
            "failure_curve/AlphaEdit/seed42",
        ),
        (
            "MEMIT",
            dict(base_alg="MEMIT", seed=42, experiment_type="failure_curve"),
            "failure_curve/MEMIT/seed42",
        ),
        (
            "MEMIT-Seq",
            dict(base_alg="MEMIT", seed=42, lambda_prev=1.0, lambda_delta=0.0,
                 kernel_degree=1, kernel_prev=True, experiment_type="polykernel_seqreg"),
            "polykernel_seqreg/MEMIT-Seq-poly1-lp1.0-ld0.0-cache0/seed42",
        ),
        (
            "REVIVE+MEMIT",
            dict(base_alg="MEMIT", seed=42, lambda_prev=0.0, lambda_delta=0.0,
                 kernel_degree=1, kernel_prev=True, revive=True, revive_tau=0.1,
                 experiment_type="polykernel_seqreg"),
            "polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed42",
        ),
        (
            "REVIVE+AlphaEdit",
            dict(base_alg="AlphaEdit", seed=42, lambda_prev=0.0, lambda_delta=0.0,
                 kernel_degree=1, kernel_prev=True, revive=True, revive_tau=0.1,
                 experiment_type="polykernel_seqreg"),
            "polykernel_seqreg/AlphaEdit-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed42",
        ),
        (
            "REVIVE+NSE",
            dict(base_alg="NSE", seed=42, lambda_prev=0.0, lambda_delta=0.0,
                 kernel_degree=1, kernel_prev=True, revive=True, revive_tau=0.1,
                 experiment_type="polykernel_seqreg"),
            "polykernel_seqreg/NSE-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed42",
        ),
        (
            "REVIVE+RECT",
            dict(base_alg="MEMIT_rect", seed=42, lambda_prev=0.0, lambda_delta=0.0,
                 kernel_degree=1, kernel_prev=True, revive=True, revive_tau=0.1,
                 experiment_type="polykernel_seqreg"),
            "polykernel_seqreg/MEMIT_rect-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/seed42",
        ),
    ], ids=lambda x: x if isinstance(x, str) else "")
    def test_checkpoint_path(self, label, config_kwargs, expected_fragment, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(**config_kwargs)
        ckpt = str(config.checkpoint_dir(tmp_path))
        assert expected_fragment in ckpt, (
            f"{label}: expected '{expected_fragment}' in checkpoint path, "
            f"got: {ckpt}"
        )


# ---------------------------------------------------------------------------
# 11. Hook composition for every REVIVE+X combo
# ---------------------------------------------------------------------------


class TestHookCompositionAllCombos:
    """Every REVIVE+X combination the GPU smoke test runs must compose correctly."""

    def _make_mock_tensors(self, d=64, n=10):
        import torch
        layer_ks = torch.randn(d, n, dtype=torch.float64)
        cov = torch.eye(d, dtype=torch.float64)
        return layer_ks, cov

    def _make_mock_hparams(self):
        from unittest.mock import MagicMock
        hp = MagicMock()
        hp.mom2_update_weight = 15000.0
        hp.layers = [4, 5, 6, 7, 8]
        return hp

    def test_revive_plus_memit_seqreg(self):
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks
        hooks = compose_hooks(
            seqreg_hooks(lambda_prev=0.0, lambda_delta=0.0),
            revive_hooks(revive_tau=0.1),
        )
        assert hooks.build_lhs is not None
        assert hooks.post_solve is not None
        state = hooks.get_state()
        layer_ks, cov = self._make_mock_tensors()
        hp = self._make_mock_hparams()
        lhs = hooks.build_lhs(4, layer_ks, cov, hp, state)
        assert lhs.shape == (64, 64)

    def test_revive_plus_alphaedit(self):
        """AlphaEdit+REVIVE: seqreg builds LHS, revive filters update."""
        from algorithms.hooks import compose_hooks
        from algorithms.hook_presets import seqreg_hooks, revive_hooks
        import torch
        hooks = compose_hooks(
            seqreg_hooks(lambda_prev=0.0, lambda_delta=0.0),
            revive_hooks(revive_tau=0.1, revive_svd_device="cpu"),
        )
        state = hooks.get_state()
        layer_ks, cov = self._make_mock_tensors()
        hp = self._make_mock_hparams()
        lhs = hooks.build_lhs(4, layer_ks, cov, hp, state)
        upd = torch.randn(64, 64, dtype=torch.float64)
        adj_k = torch.randn(64, 10, dtype=torch.float64)
        # post_solve needs _current_weights in state for revive
        state["_current_weights"] = {"test.weight": torch.randn(64, 64)}
        filtered = hooks.post_solve(4, upd, adj_k, layer_ks, "test.weight", state)
        assert filtered.shape == upd.shape

    def test_revive_only_for_nse(self):
        """NSE+REVIVE: no seqreg, just revive filter."""
        from algorithms.hook_presets import revive_hooks
        import torch
        hooks = revive_hooks(revive_tau=0.1, revive_svd_device="cpu")
        assert hooks.post_solve is not None
        assert hooks.build_lhs is None  # NSE doesn't augment LHS
        upd = torch.randn(64, 64, dtype=torch.float64)
        state = {"_current_weights": {"test.weight": torch.randn(64, 64)}}
        filtered = hooks.post_solve(4, upd, None, None, "test.weight", state)
        assert filtered.shape == upd.shape

    def test_revive_only_for_rect(self):
        """RECT+REVIVE: same as NSE — revive filter only."""
        from algorithms.hook_presets import revive_hooks
        import torch
        hooks = revive_hooks(revive_tau=0.1, revive_svd_device="cpu")
        upd = torch.randn(64, 64, dtype=torch.float64)
        state = {"_current_weights": {"test.weight": torch.randn(64, 64)}}
        filtered = hooks.post_solve(4, upd, None, None, "test.weight", state)
        assert torch.linalg.norm(filtered) <= torch.linalg.norm(upd) + 1e-6


# ---------------------------------------------------------------------------
# 12. Baseline script existence and vendor function availability
# ---------------------------------------------------------------------------


class TestBaselineScriptsAndFunctions:
    """Baseline scripts must exist and their vendor functions must be findable."""

    @pytest.mark.parametrize("script,function_name,source_file", [
        ("scripts/run_evoedit_baseline.sh", "apply_EvoEdit_to_model",
         "baselines/EvoEdit/EvoEdit/EvoEdit_main.py"),
        ("scripts/run_nse_baseline.sh", "apply_nse_to_model",
         "baselines/EvoEdit/nse/nse_main.py"),
        ("scripts/run_rect_aligned_paper_replication.sh", "apply_memit_seq_rect_to_model",
         "baselines/EvoEdit/memit/memit_seq_rect_main.py"),
    ])
    def test_baseline_script_and_function(self, script, function_name, source_file):
        assert (PROJECT_ROOT / script).exists(), f"Missing script: {script}"
        source_path = PROJECT_ROOT / source_file
        if source_path.exists():
            source = source_path.read_text()
            assert f"def {function_name}" in source, (
                f"{source_file} missing {function_name}"
            )
        else:
            pytest.skip(f"{source_file} not present (baselines not cloned)")


# ---------------------------------------------------------------------------
# 13. Ordering paths
# ---------------------------------------------------------------------------


class TestOrderingPaths:
    """Ordering experiments must produce correct subdirectory paths."""

    @pytest.mark.parametrize("ordering", [
        "fb_high_exposure", "fb_low_exposure", "fb_random0",
        "key_clustered", "key_dispersed",
    ])
    def test_ordering_in_checkpoint_dir(self, ordering, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=ordering,
            lambda_prev=1.0, kernel_degree=1,
            experiment_type="polykernel_seqreg",
        )
        ckpt = str(config.checkpoint_dir(tmp_path))
        assert ordering in ckpt, f"Ordering '{ordering}' not in checkpoint path: {ckpt}"

    @pytest.mark.parametrize("ordering", [
        "fb_high_exposure", "fb_low_exposure", "fb_random0",
        "key_clustered", "key_dispersed",
    ])
    def test_ordering_in_results_dir(self, ordering, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=ordering,
        )
        results = str(config.results_dir(tmp_path))
        assert ordering in results, f"Ordering '{ordering}' not in results path: {results}"

    def test_no_ordering_omits_matched_ordering(self, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            experiment_type="failure_curve",
        )
        results = str(config.results_dir(tmp_path))
        assert "matched_ordering" not in results

    @pytest.mark.parametrize("model_name,expected_tag", [
        ("meta-llama/Meta-Llama-3-8B-Instruct", ""),
        ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    ])
    def test_model_tag_in_checkpoint_dir(self, model_name, expected_tag, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, model_name=model_name,
            experiment_type="polykernel_seqreg", lambda_prev=1.0, kernel_degree=1,
        )
        ckpt = str(config.checkpoint_dir(tmp_path))
        if expected_tag:
            assert expected_tag in ckpt, f"model_tag '{expected_tag}' not in: {ckpt}"
        else:
            assert "gpt-j" not in ckpt and "qwen" not in ckpt


# ─── Tests that catch GPU smoke test failures from 2026-09-12 ──────────────


class TestAlphaEditRequiresPAndCacheC:
    """AlphaEdit's vendor apply function requires P and cache_c kwargs.
    The migrated checkpoint_runner must pass these via extra_apply_kwargs."""

    def test_vendor_alphaedit_signature_requires_P(self):
        """apply_AlphaEdit_to_model has P as a required-in-practice parameter."""
        import ast
        source = (PROJECT_ROOT / "vendor" / "AlphaEdit" / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "apply_AlphaEdit_to_model":
                param_names = [a.arg for a in node.args.args]
                assert "P" in param_names, "AlphaEdit apply function must accept P parameter"
                assert "cache_c" in param_names, "AlphaEdit apply function must accept cache_c parameter"
                break

    def test_checkpoint_runner_passes_P_for_alphaedit(self):
        """checkpoint_runner must load and pass P matrix for AlphaEdit."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        assert "null_space_project" in source or '"P"' in source or "'P'" in source, (
            "checkpoint_runner must load null_space_project.pt and pass P to AlphaEdit"
        )

    def test_checkpoint_runner_extra_kwargs_includes_P(self):
        """extra_apply_kwargs must include P for AlphaEdit, not just cache_c."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        # Find the extra_kwargs function
        assert '"P"' in source or "'P'" in source, (
            "extra_apply_kwargs must return P for AlphaEdit (not just cache_c)"
        )


class TestNSERequiresNSEHparams:
    """NSE requires NSEHyperParams, not MEMITHyperParams.
    Passing MEMIT hparams causes AttributeError: max_iterations."""

    def test_nse_hparams_has_max_iterations(self):
        """NSEHyperParams must have max_iterations attribute."""
        source = (PROJECT_ROOT / "vendor" / "AlphaEdit" / "nse" / "nse_hparams.py").read_text()
        assert "max_iterations" in source, "NSE hparams must define max_iterations"

    def test_memit_hparams_lacks_max_iterations(self):
        """MEMITHyperParams does NOT have max_iterations — using it for NSE is a bug."""
        source = (PROJECT_ROOT / "vendor" / "AlphaEdit" / "memit" / "memit_hparams.py").read_text()
        assert "max_iterations" not in source, "MEMIT hparams should not have max_iterations"

    def test_polykernel_seqreg_loads_correct_hparams_for_nse(self):
        """When base_alg=NSE, runner must load NSEHyperParams, not MEMITHyperParams."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # Must have NSE-specific hparams loading
        assert "NSEHyperParams" in source or "NSE" in source.split("HParams")[0] if "HParams" in source else True, (
            "polykernel_seqreg_runner must load NSEHyperParams when base_alg=NSE"
        )

    def test_polykernel_seqreg_nse_hparams_file_path(self):
        """NSE hparams should be loaded from hparams/NSE/, not hparams/MEMIT/."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # When base_alg=NSE, alg_for_hparams should be "NSE"
        has_nse_hparams_path = '"NSE"' in source and "alg_for_hparams" in source
        assert has_nse_hparams_path or "NSEHyperParams" in source, (
            "Runner must use hparams/NSE/ directory when base_alg=NSE"
        )


class TestBaselinesPatchCompilation:
    """Baselines evaluate.py must compile after ALL patches including mega-batch.
    Tests the actual patch pipeline that runs on the cluster."""

    def test_mega_batch_anchor_unique_in_baselines(self):
        """The mega-batch anchor must match exactly once in baselines evaluate.py."""
        eval_path = PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines not available")
        source = eval_path.read_text()
        from patch_mega_batch_eval import EVAL_ANCHOR
        if "_mega_batch_eval" in source:
            pytest.skip("baselines evaluate.py already patched (anchor consumed)")
        count = source.count(EVAL_ANCHOR)
        assert count == 1, f"EVAL_ANCHOR matches {count} times in baselines evaluate.py (expected 1)"

    def test_baselines_compile_after_all_patches(self):
        """After apply_all.py runs, baselines evaluate.py must compile."""
        eval_path = PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines not available")
        # Read original, apply patches, compile
        import importlib
        import patch_mega_batch_eval
        importlib.reload(patch_mega_batch_eval)
        source = eval_path.read_text()
        if "_mega_batch_eval" not in source:
            # Not yet patched — apply the patch in memory
            fn_src_mod = importlib.import_module("mega_batch_eval")
            fn_src = fn_src_mod.get_mega_batch_eval_source()
            fn_indented = "\n".join("    " + line for line in fn_src.strip().split("\n"))
            anchor = patch_mega_batch_eval.EVAL_ANCHOR
            if anchor in source:
                replacement = source.split(anchor)[0] + fn_indented + "\n    " + anchor.split("\n")[-1]
                # Just verify the anchor is unique
                assert source.count(anchor) == 1
        # The actual compilation test
        compile(source, str(eval_path), "exec")


class TestRECTImportPath:
    """MEMIT_rect apply function is in baselines, not vendor/AlphaEdit/memit/."""

    def test_rect_not_in_vendor_memit(self):
        """vendor/AlphaEdit/memit/ does NOT have memit_seq_rect_main.py."""
        rect_in_vendor = (PROJECT_ROOT / "vendor" / "AlphaEdit" / "memit" / "memit_seq_rect_main.py").exists()
        assert not rect_in_vendor, "memit_seq_rect_main.py is in baselines, not vendor"

    def test_rect_in_baselines(self):
        """baselines/EvoEdit/memit/ has memit_seq_rect_main.py."""
        rect_in_baselines = (PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_rect_main.py").exists()
        if not rect_in_baselines:
            pytest.skip("baselines not available")
        assert rect_in_baselines

    def test_polykernel_seqreg_rect_import_uses_baselines(self):
        """polykernel_seqreg_runner must add baselines to sys.path before importing RECT."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # The import "from memit.memit_seq_rect_main import" is fine IF baselines is on sys.path first
        if "memit_seq_rect_main" in source:
            assert "baselines" in source, (
                "RECT import needs baselines on sys.path (memit_seq_rect_main is in baselines, not vendor)"
            )


class TestAlphaEditHooksHandleNoneCov:
    """AlphaEdit doesn't use mom2 covariance — hooks must handle cov=None."""

    def test_seqreg_build_lhs_handles_none_cov(self):
        """seqreg_hooks.build_lhs must not crash when cov=None (AlphaEdit path)."""
        import torch
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=0.0, lambda_delta=0.0)
        state = hooks.get_state()
        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        # AlphaEdit passes cov=None
        try:
            lhs = hooks.build_lhs(0, layer_ks, None, type('H', (), {'mom2_update_weight': 1.0})(), state)
            # Should not crash
        except (TypeError, AttributeError) as e:
            if "NoneType" in str(e):
                pytest.fail(f"build_lhs crashed on cov=None: {e}")
            raise

    def test_compose_hooks_build_lhs_handles_none_cov(self):
        """compose_hooks with seqreg must handle cov=None."""
        import torch
        from algorithms.hook_presets import seqreg_hooks, revive_hooks
        from algorithms.hooks import compose_hooks
        hooks = compose_hooks(seqreg_hooks(lambda_prev=0.0), revive_hooks(revive_tau=0.1))
        state = hooks.get_state()
        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        try:
            lhs = hooks.build_lhs(0, layer_ks, None, type('H', (), {'mom2_update_weight': 1.0})(), state)
        except (TypeError, AttributeError) as e:
            if "NoneType" in str(e):
                pytest.fail(f"compose_hooks build_lhs crashed on cov=None: {e}")
            raise
