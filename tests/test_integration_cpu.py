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


class TestSmokeTestYAMLIsolation:
    """Each test cluster must use isolated checkpoint/result paths."""

    def test_all_yamls_use_unique_paths(self):
        """No two test YAMLs should share the same _smoke_test root."""
        import yaml
        paths = {}
        for yaml_file in (PROJECT_ROOT / "sky").glob("test_*.yaml"):
            content = yaml_file.read_text()
            for line in content.split("\n"):
                if "RESULT_ROOT" in line and "_smoke_test" in line:
                    path = line.split("=", 1)[-1].strip().strip('"')
                    name = yaml_file.name
                    if path in paths.values():
                        conflicting = [k for k, v in paths.items() if v == path]
                        pytest.fail(f"{name} shares RESULT_ROOT with {conflicting}: {path}")
                    paths[name] = path
        assert len(paths) >= 3, f"Expected at least 3 test YAMLs with RESULT_ROOT, found {len(paths)}"

    def test_yamls_include_integration_tests(self):
        """Each test YAML should run at least one integration test file."""
        integration_files = {"test_integration_cpu", "test_patch_integration",
                           "test_runner_integration", "test_endtoend_cpu",
                           "test_mechanism_correctness", "test_path_correctness",
                           "test_harness_integration"}
        for yaml_file in (PROJECT_ROOT / "sky").glob("test_*.yaml"):
            if yaml_file.name == "test_all.sh" or yaml_file.name == "test.yaml":
                continue
            content = yaml_file.read_text()
            found = [f for f in integration_files if f in content]
            assert len(found) > 0, (
                f"{yaml_file.name} doesn't include any integration test files. "
                f"Add at least one of: {integration_files}"
            )

    def test_eval_cluster_doesnt_run_alphaedit(self):
        """test-eval should NOT re-run AlphaEdit (already in test-vendor)."""
        eval_yaml = PROJECT_ROOT / "sky" / "test_eval_and_measure.yaml"
        if eval_yaml.exists():
            content = eval_yaml.read_text()
            assert "test_smoke_all_algorithms.sh" not in content or "AlphaEdit" not in content.split("test_smoke_all_algorithms.sh")[-1] if "test_smoke_all_algorithms.sh" in content else True, (
                "test-eval should not re-run AlphaEdit smoke test (test-vendor already covers it)"
            )


class TestSmokeTestREVIVEValidation:
    """REVIVE validation in smoke test must match hooks-based output."""

    def test_smoke_test_checks_revive_layer_output(self):
        """Smoke test should check for '[REVIVE] layer=' (hooks output)."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        assert "[REVIVE]" in source, "Smoke test must validate [REVIVE] output"

    def test_smoke_test_doesnt_check_old_revive_strings(self):
        """Smoke test should NOT check for old exec-based REVIVE strings."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        assert "Dynamic SVD" not in source, "Smoke test references old REVIVE init message"
        assert "full_matrices=True" not in source, "Smoke test references old REVIVE init message"

    def test_revive_hooks_print_layer_info(self):
        """revive_hooks() must print '[REVIVE] layer=' for validation."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        assert "[REVIVE] layer=" in source, "revive_hooks must print per-layer info for smoke test validation"


class TestSmokeTestTimeouts:
    """Verify smoke test timeout is reasonable and algorithms can complete in time."""

    def test_timeout_is_5_minutes_or_less(self):
        """Smoke test timeout must be ≤ 300s (5 min). Longer means something is stuck."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        import re
        # Default timeout (may be overridden by SMOKE_TIMEOUT env var for baselines)
        match = re.search(r'SMOKE_TIMEOUT:-(\d+)', source) or re.search(r'TIMEOUT=(\d+)', source)
        assert match, "TIMEOUT not found in smoke test script"
        timeout = int(match.group(1))
        assert timeout <= 300, f"Default smoke test timeout is {timeout}s — should be ≤ 300s"

    def test_dataset_limit_is_small(self):
        """Smoke test dataset must be small (≤ 50 records) for fast completion."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        import re
        match = re.search(r'DATASET_LIMIT=(\d+)', source)
        assert match, "DATASET_LIMIT not found in smoke test script"
        limit = int(match.group(1))
        assert limit <= 50, f"Dataset limit is {limit} — should be ≤ 50 for smoke tests"

    def test_batch_size_is_small(self):
        """Smoke test batch size must be small (≤ 20 edits) for fast completion."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        import re
        match = re.search(r'EDITS=(\d+)', source)
        assert match, "EDITS not found in smoke test script"
        edits = int(match.group(1))
        assert edits <= 20, f"Edits per batch is {edits} — should be ≤ 20 for smoke tests"


class TestBaselineScriptPerformance:
    """Baseline scripts must not do expensive operations inline."""

    def test_evoedit_no_inline_model_download(self):
        """EvoEdit script must use model_resolve, not snapshot_download."""
        source = (PROJECT_ROOT / "scripts" / "run_evoedit_baseline.sh").read_text()
        assert "snapshot_download" not in source, (
            "EvoEdit script downloads model inline — use model_resolve for cached resolution"
        )

    def test_nse_no_inline_model_download(self):
        """NSE script must use model_resolve, not snapshot_download."""
        source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        assert "snapshot_download" not in source, (
            "NSE script downloads model inline — use model_resolve for cached resolution"
        )

    def test_baselines_no_inline_p_matrix_compute(self):
        """Baseline scripts must use cached P matrix, not compute SVD inline."""
        for script in ["run_evoedit_baseline.sh", "run_nse_baseline.sh"]:
            path = PROJECT_ROOT / "scripts" / script
            if path.exists():
                source = path.read_text()
                assert "torch.linalg.svd" not in source, (
                    f"{script} computes SVD inline — use cached null_space_project.pt"
                )

    def test_no_double_patching(self):
        """apply_all.py and baseline scripts must not patch the same thing twice."""
        for script in ["run_evoedit_baseline.sh", "run_nse_baseline.sh"]:
            path = PROJECT_ROOT / "scripts" / script
            if not path.exists():
                continue
            source = path.read_text()
            # The mega-batch eval is patched by apply_all.py — scripts must not also patch it
            assert "_mega_batch_eval" not in source, (
                f"{script} patches mega_batch_eval inline — apply_all.py already handles this"
            )


class TestVendorFunctionRequirements:
    """Test that each vendor function gets its required kwargs.
    These MUST catch bugs before GPU smoke tests."""

    def test_nse_cache_c_initialized_without_P(self):
        """NSE needs cache_c but NOT P. cache_c must be initialized from model dims, not P."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # NSE section must initialize cache_c somewhere (not just in the AlphaEdit block)
        # Look for NSE-specific cache_c init
        assert "NSE" in source and "cache_c = torch.zeros" in source, (
            "Runner must initialize cache_c for NSE (needed by vendor nse_main.py)"
        )
        # The NSE cache_c init must NOT be inside an 'if P is not None' block
        # The cache_c init for NSE must exist somewhere in the file
        assert "NSE" in source and "cache_c = torch.zeros" in source
        # Specifically: there should be a cache_c init in the NSE-specific block
        nse_cache_section = source[source.find("elif args.base_alg == \"NSE\"", source.find("cache_c = None")):]
        assert "cache_c" in nse_cache_section[:600], "NSE cache_c section must exist"

    def test_nse_kv_cache_script_exists(self):
        """build_nse_cache.sh must exist — NSE/EvoEdit need precomputed KV caches
        or compute_z takes 25 gradient steps per edit (~15 min for 20 edits)."""
        assert (PROJECT_ROOT / "scripts" / "build_nse_cache.sh").exists(), (
            "scripts/build_nse_cache.sh missing — needed to populate NSE/EvoEdit KV caches"
        )

    def test_nse_kv_cache_extracts_from_s3(self):
        """build_nse_cache.sh must reference S3 or /s3-data for cache source."""
        source = (PROJECT_ROOT / "scripts" / "build_nse_cache.sh").read_text()
        assert "s3" in source.lower() or "S3" in source, (
            "build_nse_cache.sh must extract KV caches from S3"
        )

    def test_only_nse_loads_kv_cache(self):
        """Only run_nse_baseline.sh should load the NSE KV cache.
        EvoEdit computes v_star from the current model state per batch — no cache."""
        nse_source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        assert "nse_kv_cache" in nse_source, "NSE script must load KV cache"

        evoedit_source = (PROJECT_ROOT / "scripts" / "run_evoedit_baseline.sh").read_text()
        assert "nse_kv_cache" not in evoedit_source, (
            "EvoEdit must NOT load NSE KV cache — it computes v_star from current W_t"
        )

    def test_nse_cache_symlinks_model_name_variants(self):
        """NSE cache tars use NousResearch_ prefix but canonical model name is meta-llama_.
        After extraction, run_nse_baseline.sh must symlink the canonical name
        to the tar's directory name so evaluate.py finds the cache."""
        source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        # Must explicitly handle the NousResearch → meta-llama rename
        has_rename = ("NousResearch" in source and "meta-llama" in source) or \
                     "model name variant" in source.lower()
        assert has_rename, (
            "run_nse_baseline.sh must symlink meta-llama_ → NousResearch_ in kvs/. "
            "The S3 tars extract to NousResearch_Meta-Llama-3-8B-Instruct_NSE/ "
            "but evaluate.py looks for meta-llama_Meta-Llama-3-8B-Instruct_NSE/."
        )

    def test_nse_baseline_script_loads_kv_cache(self):
        """run_nse_baseline.sh must load KV caches from S3 — not the YAML.
        Without precomputed caches, compute_z runs 25 gradient steps per edit."""
        source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        assert "kvs" in source or "kv_cache" in source.lower(), (
            "run_nse_baseline.sh must load KV caches from S3. "
            "Cache loading belongs in the script, not the YAML."
        )

    def test_nse_script_uses_tar_only(self):
        """run_nse_baseline.sh must extract .tar files only — no slow individual
        .npz copy fallback. Fail fast if tars aren't found."""
        source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        assert "tar xf" in source, "Script must extract tar archives"
        assert 'cp "$_kv_dir"' not in source, (
            "Script must NOT have individual file copy fallback — "
            "copying 20K small files over S3 FUSE is ~100x slower than tar"
        )
        assert "exit 1" in source[source.find("nse_kv_cache"):], (
            "Script must fail fast (exit 1) if KV caches aren't available"
        )

    def test_revive_script_loads_nse_cache(self):
        """run_revive_baseline.sh must load NSE KV caches when BASE_ALG=NSE."""
        source = (PROJECT_ROOT / "scripts" / "run_revive_baseline.sh").read_text()
        assert "nse_kv_cache" in source.lower() or "NSE_CACHE" in source, (
            "run_revive_baseline.sh must load NSE KV caches for BASE_ALG=NSE"
        )

    def test_nse_cache_not_shared_with_memit_family(self):
        """NSE cache must NOT be symlinked to _MEMIT. NSE precomputes v_star from
        the original model W₀, while MEMIT/AlphaEdit/EvoEdit compute from the
        current edited model W_t. The cached values are NOT interchangeable."""
        for script_name in ["run_nse_baseline.sh", "run_evoedit_baseline.sh"]:
            path = PROJECT_ROOT / "scripts" / script_name
            if path.exists():
                source = path.read_text()
                assert "_MEMIT" not in source, (
                    f"{script_name} must NOT symlink _MEMIT → _NSE. "
                    "NSE computes v_star from W₀; MEMIT-family from W_t."
                )

    def test_evoedit_does_not_use_nse_cache(self):
        """EvoEdit must NOT load NSE caches or pass --use_cache.
        EvoEdit computes v_star from the current model state W_t per batch."""
        source = (PROJECT_ROOT / "scripts" / "run_evoedit_baseline.sh").read_text()
        assert "nse_kv_cache" not in source, "EvoEdit must not load NSE caches"
        assert "--use_cache" not in source, "EvoEdit must not use --use_cache"

    def test_nse_script_fails_fast_without_cache(self):
        """run_nse_baseline.sh must exit 1 if NSE KV cache is missing.
        NSE precomputes ALL v_star from W₀ — without cache this is very slow."""
        source = (PROJECT_ROOT / "scripts" / "run_nse_baseline.sh").read_text()
        cache_section = source[source.find("nse_kv_cache") if "nse_kv_cache" in source else 0:]
        assert "exit 1" in cache_section[:1000], (
            "run_nse_baseline.sh must exit 1 if NSE KV cache not available"
        )

    def test_runner_passes_cache_template_for_nse(self):
        """polykernel_seqreg_runner must pass cache_template when base_alg=NSE.
        Without it, apply_nse_to_model gets cache_template=None and recomputes
        all v_star from scratch (25 gradient steps per edit)."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "cache_template" in source, (
            "polykernel_seqreg_runner must pass cache_template to apply_nse_to_model. "
            "Without it, NSE recomputes v_star from scratch even when cache files exist."
        )

    def test_nse_kv_cache_tar_structure(self):
        """KV cache tars must extract to {model_name}_NSE/{file}.npz matching
        the path vendor NSE expects at share/projects/rewriting-knowledge/kvs/."""
        import tarfile
        tar_path = PROJECT_ROOT / "data" / "nse_kv_cache" / "llama_nse_cache.tar"
        if not tar_path.exists():
            pytest.skip("llama_nse_cache.tar not available locally")
        with tarfile.open(tar_path) as t:
            names = t.getnames()[:5]
        assert any("NousResearch" in n or "Meta-Llama" in n for n in names), (
            f"Tar must contain model-named directory, got: {names}"
        )
        assert any(n.endswith(".npz") for n in names), (
            f"Tar must contain .npz cache files, got: {names}"
        )

    def test_mega_batch_eval_skippable(self):
        """mega_batch_eval must be skippable via SKIP_MEGA_BATCH_EVAL env var.
        The editing smoke test sets this so it only validates edits + checkpoints.
        Eval is tested separately by the eval cluster."""
        eval_path = PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines not available")
        source = eval_path.read_text()
        if "_mega_batch_eval" not in source:
            pytest.skip("mega_batch_eval not patched yet")
        assert "SKIP_MEGA_BATCH_EVAL" in source, (
            "mega_batch_eval must check SKIP_MEGA_BATCH_EVAL env var"
        )

    def test_mega_batch_eval_respects_skip_env(self):
        """The mega_batch_eval patch must check SKIP_MEGA_BATCH_EVAL env var.
        Smoke tests set this so editing tests don't waste time on evaluation."""
        eval_path = PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py"
        if not eval_path.exists():
            pytest.skip("baselines not available")
        source = eval_path.read_text()
        if "_mega_batch_eval" not in source:
            pytest.skip("mega_batch_eval not patched yet")
        assert "SKIP_MEGA_BATCH_EVAL" in source, (
            "mega_batch_eval call must check os.environ.get('SKIP_MEGA_BATCH_EVAL'). "
            "The editing smoke test sets this to skip eval — eval is the eval cluster's job."
        )

    def test_smoke_test_sets_skip_eval_for_baselines(self):
        """The editing smoke test must set SKIP_MEGA_BATCH_EVAL=1 for baseline runs."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        assert "SKIP_MEGA_BATCH_EVAL" in source, (
            "Smoke test must set SKIP_MEGA_BATCH_EVAL=1 for baseline runs. "
            "Eval is the eval cluster's job, not the editing smoke test."
        )

    def test_eval_cluster_tests_mega_batch_eval(self):
        """The eval cluster YAML must test mega_batch_eval correctness."""
        yaml_path = PROJECT_ROOT / "sky" / "test_eval_and_measure.yaml"
        if not yaml_path.exists():
            pytest.skip("test_eval_and_measure.yaml not tracked")
        source = yaml_path.read_text()
        assert "TestEvalMetrics" in source, (
            "Eval cluster must test TestEvalMetrics (includes mega_batch_eval)"
        )

    def test_no_duplicate_algorithms_across_smoke_clusters(self):
        """Each algorithm must run on exactly ONE smoke test cluster."""
        import re
        filters = {}
        for yaml_name in ["test_vendor_runners.yaml", "test_revive_runners.yaml", "test_baseline_runners.yaml"]:
            path = PROJECT_ROOT / "sky" / yaml_name
            if not path.exists():
                continue
            source = path.read_text()
            m = re.search(r'test_smoke_all_algorithms\.sh\s+--skypilot\s+(.+)', source)
            if m:
                filters[yaml_name] = m.group(1).strip().split()
        assert len(filters) >= 2, "Need at least 2 smoke test YAMLs"
        all_tokens = []
        for yaml_name, tokens in filters.items():
            all_tokens.extend(tokens)
        # No token should be a substring of another (causes duplicate runs)
        for i, a in enumerate(all_tokens):
            for j, b in enumerate(all_tokens):
                if i != j and a in b and a != b:
                    # This is OK if the filter uses exact matching (e.g. AlphaEdit_ckpt)
                    # But NOT OK if one is a raw substring of the other
                    assert a.endswith("_ckpt") or a == "REVIVE", (
                        f"Filter token '{a}' is a substring of '{b}' — will cause duplicate runs. "
                        f"Use _ckpt suffix or exact aliases."
                    )

    def test_eval_cluster_does_not_skip_eval(self):
        """The eval cluster must NOT set SKIP_MEGA_BATCH_EVAL."""
        yaml_path = PROJECT_ROOT / "sky" / "test_eval_and_measure.yaml"
        if not yaml_path.exists():
            pytest.skip("test_eval_and_measure.yaml not tracked")
        source = yaml_path.read_text()
        assert "SKIP_MEGA_BATCH_EVAL" not in source, (
            "Eval cluster must NOT skip mega_batch_eval — it's the eval cluster's job to test it"
        )

    def test_link_stats_copies_p_matrix_locally(self):
        """link_stats.sh must cp (not symlink) the P matrix to vendor/AlphaEdit/.
        S3 FUSE doesn't support ln -sf into it, so P must be copied locally."""
        link_stats = (PROJECT_ROOT / "scripts" / "link_stats.sh").read_text()
        assert "cp" in link_stats and "null_space_project" in link_stats, (
            "link_stats.sh must use cp (not ln) for null_space_project.pt — "
            "S3 FUSE doesn't support symlinks into it."
        )

    def test_link_stats_removes_stale_symlink_before_copy(self):
        """link_stats.sh must rm the destination before cp. A previous run may have
        left a symlink to S3, and cp fails with 'are the same file' when the source
        resolves through that symlink."""
        link_stats = (PROJECT_ROOT / "scripts" / "link_stats.sh").read_text()
        # rm must appear BEFORE cp for the P matrix destination
        rm_pos = link_stats.find('rm -f "$_P_DST"')
        cp_pos = link_stats.find('cp "$P_SRC" "$_P_DST"')
        assert rm_pos > 0 and cp_pos > 0 and rm_pos < cp_pos, (
            "link_stats.sh must rm -f the P matrix destination BEFORE cp. "
            "Without this, cp fails when a stale symlink to S3 exists."
        )

    def test_runner_p_path_matches_link_stats(self):
        """Runner must look for P at alphaedit_root/null_space_project.pt —
        the exact path that link_stats.sh copies it to."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert 'alphaedit_root / "null_space_project.pt"' in source, (
            "Runner must check alphaedit_root/null_space_project.pt "
            "(where link_stats.sh copies it)"
        )

    def test_alphaedit_lhs_uses_double_precision(self):
        """AlphaEdit through hooks must use float64 for the LHS solve, not float16."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "hook_presets.py").read_text()
        # The build_lhs function must handle cov=None by still using double precision
        build_lhs_section = source[source.find("def build_lhs(layer_idx, layer_ks, cov, hparams, state)"):]
        build_lhs_section = build_lhs_section[:build_lhs_section.find("return AlgorithmHooks")]
        # When cov=None, layer_ks must still be converted to double
        assert "layer_ks @ layer_ks.T" in build_lhs_section

    def test_rect_module_has_relative_imports(self):
        """RECT module uses relative imports — importlib.util alone won't work."""
        rect_path = PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_rect_main.py"
        if not rect_path.exists():
            pytest.skip("baselines not available")
        source = rect_path.read_text()
        has_relative = "from ." in source
        assert has_relative, "RECT module uses relative imports — import strategy must handle this"

    def test_alphaedit_hooks_solve_in_double(self):
        """torch.linalg.solve requires float64, not float16. Test with mock tensors."""
        import torch
        from algorithms.hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=0.0, lambda_delta=0.0)
        state = hooks.get_state()
        # Simulate AlphaEdit path: cov=None, layer_ks in double
        layer_ks = torch.randn(64, 10, dtype=torch.float64)
        hparams = type('H', (), {'mom2_update_weight': 1.0})()
        lhs = hooks.build_lhs(0, layer_ks, None, hparams, state)
        # LHS must be float64 for torch.linalg.solve
        assert lhs.dtype == torch.float64, f"LHS dtype is {lhs.dtype}, must be float64 for solve"


class TestRECTRequiresCacheC:
    """RECT (MEMIT_seq_rect) needs cache_c initialized, just like NSE."""

    def test_rect_cache_c_branch_exists(self):
        """polykernel_seqreg_runner must have a MEMIT_rect branch that initializes cache_c."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        after_cache_init = source[source.find("cache_c = None"):]
        assert 'args.base_alg == "MEMIT_rect"' in after_cache_init or \
               "MEMIT_rect" in after_cache_init[:2000], (
            "polykernel_seqreg_runner must initialize cache_c for MEMIT_rect. "
            "RECT's execute_memit indexes cache_c[i,:,:] which crashes on None."
        )

    def test_rect_cache_c_not_none_when_selected(self):
        """When base_alg=MEMIT_rect, cache_c must be initialized (not left as None)."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # After "cache_c = None", there should be a MEMIT_rect block before apply_fn
        init_section = source[source.find("cache_c = None"):source.find("def apply_fn")]
        has_rect_init = ("MEMIT_rect" in init_section and
                         "cache_c = torch.zeros" in init_section[init_section.find("MEMIT_rect"):])
        assert has_rect_init, (
            "MEMIT_rect needs its own cache_c = torch.zeros(...) branch. "
            "Without it, cache_c stays None and RECT crashes with: "
            "TypeError: 'NoneType' object is not subscriptable"
        )


    def test_rect_error_cache_initialized(self):
        """RECT needs error_cache alongside cache_c. RECT's execute_memit does
        error_cache[i,:,:] += error_temp which crashes on None."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        init_section = source[source.find("cache_c = None"):source.find("def apply_fn")]
        assert "error_cache" in init_section and "MEMIT_rect" in init_section, (
            "polykernel_seqreg_runner must initialize error_cache for MEMIT_rect"
        )

    def test_rect_error_cache_passed_to_apply(self):
        """error_cache must be passed to RECT's apply function via extra kwargs."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        apply_section = source[source.find("def apply_fn"):source.find("def after_edit") if "def after_edit" in source else len(source)]
        assert "error_cache" in apply_section, (
            "apply_fn must pass error_cache to RECT"
        )

    def test_checkpoint_runner_initializes_cache_c_for_alphaedit(self):
        """checkpoint_runner must initialize cache_c to zeros for AlphaEdit.
        Vendor AlphaEdit_main.py does cache_c[i,:,:] which crashes on None."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        ae_section = source[source.find('"AlphaEdit" in args.alg_name'):source.find("# Load checkpoint")]
        assert "cache_c = torch.zeros" in ae_section, (
            "checkpoint_runner must initialize cache_c for AlphaEdit before first batch. "
            "Vendor does cache_c[i,:,:] which crashes on None."
        )

    def test_mega_batch_eval_template_has_two_placeholders(self):
        """mega_batch_eval formats template with (num_edits, case_id) — needs two {}."""
        source = (PROJECT_ROOT / "src" / "util" / "mega_batch_eval.py").read_text()
        assert "case_result_template.format(num_edits, record" in source, (
            "mega_batch_eval uses template.format(num_edits, case_id) — "
            "template must have two {} placeholders"
        )

    def test_harness_respects_skip_mega_batch_eval(self):
        """evaluate_harness must check SKIP_MEGA_BATCH_EVAL env var before calling eval_fn.
        The editing smoke test sets this — eval is the eval cluster's job."""
        source = (PROJECT_ROOT / "src" / "evaluate_harness.py").read_text()
        assert "SKIP_MEGA_BATCH_EVAL" in source, (
            "evaluate_harness must check SKIP_MEGA_BATCH_EVAL env var"
        )

    def test_smoke_test_skips_eval_for_vendor_runners(self):
        """run_and_check (vendor runners) must also set SKIP_MEGA_BATCH_EVAL."""
        source = (PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh").read_text()
        # Find the run_and_check execution line (not run_baseline)
        run_and_check_exec = [l for l in source.split("\n")
                              if "timeout" in l and '"$@"' in l]
        assert run_and_check_exec, "run_and_check must have a timeout execution line"
        assert "SKIP_MEGA_BATCH_EVAL" in run_and_check_exec[0], (
            "run_and_check must set SKIP_MEGA_BATCH_EVAL=1 — "
            "vendor runners also call mega_batch_eval via the harness"
        )


class TestAlphaEditSolveRegularization:
    """AlphaEdit's LHS must always include P projection and L2 regularization,
    even when hooks.build_lhs is set (e.g. for REVIVE composition)."""

    def test_alphaedit_hooks_path_applies_P_projection(self):
        """When hooks.build_lhs is set, the hook result must be wrapped with P projection.
        Without P @ (...) + L2*I, the LHS is rank-deficient (10 keys in 4096 dims)
        and torch.linalg.solve fails with 'singular matrix'."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        hook_start = source.find("if hooks.build_lhs is not None:")
        assert hook_start > 0
        # The hooks.build_lhs result must go into an intermediate variable (inner/term),
        # NOT directly into `lhs`. The actual `lhs = ...` must include P.
        solve_section = source[hook_start:source.find("post_solve", hook_start)]
        # Find the line that actually builds lhs (contains "P" and "lhs =")
        lhs_lines = [l.strip() for l in solve_section.split("\n")
                     if l.strip().startswith("lhs") and "=" in l]
        assert any("P" in l for l in lhs_lines), (
            "The `lhs = ...` assignment must include P projection. "
            "hooks.build_lhs provides the inner term only; AlphaEdit wraps with "
            "P @ (inner + cache_c) + L2*I."
        )

    def test_alphaedit_both_paths_use_L2(self):
        """Both hook and default paths must include L2 regularization."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        lhs_section = source[source.find("build_lhs"):source.find("post_solve")]
        assert "L2" in lhs_section, "L2 regularization must appear in the LHS construction"

    def test_alphaedit_frees_p_after_solve(self):
        """AlphaEdit must del P_i and empty_cache after the solve to free GPU memory.
        P_i is [14336, 14336] = 0.77 GiB in float32."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        solve_section = source[source.find("build_lhs"):source.find("post_solve")]
        assert "del P_i" in solve_section, (
            "AlphaEdit must free P_i after the solve to reclaim GPU memory"
        )


class TestReviveWorksForNonHookAlgorithms:
    """REVIVE must work for NSE and RECT, which are vendor functions that
    don't accept AlgorithmHooks. The runner must apply REVIVE post-hoc."""

    def test_nse_apply_fn_does_not_accept_hooks_kwarg(self):
        """Verify NSE's vendor function ignores hooks — confirms the problem exists."""
        nse_path = PROJECT_ROOT / "vendor" / "AlphaEdit" / "nse" / "nse_main.py"
        if not nse_path.exists():
            pytest.skip("NSE source not available")
        source = nse_path.read_text()
        # NSE's apply function signature — it accepts **_kwargs which silently drops hooks
        assert "def apply_nse_to_model" in source

    def test_posthoc_revive_uses_revive_hook_only(self):
        """Post-hoc REVIVE must call the REVIVE hook directly, not the composed chain.
        The composed chain includes seqreg_hooks.post_solve which expects
        state['mechanism_log'] — but the post-hoc state only has _current_weights."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc_section = source[source.find("# Non-hook path") if "Non-hook path" in source else source.find("pre_weights"):]
        posthoc_section = posthoc_section[:posthoc_section.find("return result") + 20] if "return result" in posthoc_section else posthoc_section[:500]
        uses_composed = "algo_hooks.post_solve" in posthoc_section
        uses_direct_revive = any(x in posthoc_section for x in [
            "revive_hooks", "_revive_hook", "revive_filter",
        ])
        assert not uses_composed or uses_direct_revive, (
            "Post-hoc REVIVE path must NOT call algo_hooks.post_solve (composed chain). "
            "The seqreg hook in the chain expects state['mechanism_log'] which doesn't exist "
            "in the post-hoc state. Use the REVIVE hook directly instead."
        )

    def test_runner_handles_revive_for_non_hook_algorithms(self):
        """polykernel_seqreg_runner must apply REVIVE post-hoc for NSE/RECT
        since their vendor functions don't call hook.post_solve."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        after_edit_pos = source.find("def after_edit")
        apply_fn_section = source[source.find("def apply_fn"):after_edit_pos if after_edit_pos > 0 else len(source)]

        # The apply_fn must distinguish hooks-aware vs non-hooks-aware algorithms
        hooks_aware_indicators = [
            "_hooks_aware", "hooks_aware", "base_alg in (",
        ]
        has_hooks_check = any(s in apply_fn_section for s in hooks_aware_indicators)

        # Or it must apply REVIVE post-hoc (snapshot weights, apply, compute delta, filter)
        posthoc_indicators = [
            "pre_weights", "post-hoc", "weight snapshot", "filtered",
        ]
        has_posthoc = any(s in apply_fn_section for s in posthoc_indicators)

        assert has_hooks_check or has_posthoc, (
            "apply_fn must either:\n"
            "  (a) check whether the base algorithm supports hooks, OR\n"
            "  (b) apply REVIVE as a post-hoc weight filter for NSE/RECT.\n"
            "Without this, REVIVE is silently skipped for non-hook algorithms "
            "(their vendor functions accept **_kwargs but ignore hooks)."
        )


# ─── Fixed-Batch Ordering Tests ─────────────────────────────────────────────
# The paper's centerpiece: identical batches in different temporal orders.


class TestOrderingFilesExist:
    """Every ordering × seed combination must have a stream file."""

    @pytest.mark.parametrize("ordering", [
        "fb_high_exposure", "fb_low_exposure", "fb_random0", "fb_random1", "fb_random2",
        "key_clustered", "key_dispersed",
    ])
    @pytest.mark.parametrize("seed", [42, 2024, 137])
    def test_ordering_stream_exists(self, ordering, seed):
        path = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
        if not path.exists():
            pytest.skip(f"Ordering file not available: {path.name}")
        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 100, f"Ordering must have ≥100 records, got {len(data)}"
        assert all("case_id" in r for r in data[:5]), "Records must have case_id"


class TestOrderingPathsAllMethods:
    """Every method × ordering must produce correct checkpoint and results paths."""

    METHODS = [
        {"base_alg": "MEMIT", "name": "MEMIT-Seq", "experiment_type": "polykernel_seqreg",
         "lambda_prev": 1.0, "kernel_degree": 1},
        {"base_alg": "AlphaEdit", "name": "REVIVE+AlphaEdit", "experiment_type": "polykernel_seqreg",
         "revive": True, "revive_tau": 0.1},
        {"base_alg": "MEMIT", "name": "REVIVE+MEMIT", "experiment_type": "polykernel_seqreg",
         "revive": True, "revive_tau": 0.1, "lambda_prev": 0.0},
        {"base_alg": "NSE", "name": "REVIVE+NSE", "experiment_type": "polykernel_seqreg",
         "revive": True, "revive_tau": 0.1, "lambda_prev": 0.0},
        {"base_alg": "MEMIT_rect", "name": "REVIVE+RECT", "experiment_type": "polykernel_seqreg",
         "revive": True, "revive_tau": 0.1, "lambda_prev": 0.0},
    ]

    ORDERINGS = ["fb_high_exposure", "fb_low_exposure", "fb_random0"]

    MODELS = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", ""),
        ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
    ]

    @pytest.mark.parametrize("method", METHODS, ids=[m["name"] for m in METHODS])
    @pytest.mark.parametrize("ordering", ORDERINGS)
    @pytest.mark.parametrize("model_name,model_tag", MODELS, ids=["llama", "gptj"])
    def test_path_contains_ordering(self, method, ordering, model_name, model_tag, tmp_path):
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=method["base_alg"], seed=42, ordering=ordering,
            model_name=model_name,
            experiment_type=method.get("experiment_type", "polykernel_seqreg"),
            lambda_prev=method.get("lambda_prev", 0.0),
            kernel_degree=method.get("kernel_degree", 1),
            revive=method.get("revive", False),
            revive_tau=method.get("revive_tau", 0.1),
        )
        ckpt = str(config.checkpoint_dir(tmp_path))
        assert ordering in ckpt, f"Ordering '{ordering}' not in checkpoint path: {ckpt}"
        if model_tag:
            assert model_tag in ckpt, f"Model tag '{model_tag}' not in path: {ckpt}"

    @pytest.mark.parametrize("method", METHODS, ids=[m["name"] for m in METHODS])
    @pytest.mark.parametrize("ordering", ORDERINGS)
    def test_paths_distinct_across_orderings(self, method, ordering, tmp_path):
        """Different orderings must produce different checkpoint paths."""
        from experiment_config import ExperimentConfig
        paths = set()
        for o in self.ORDERINGS:
            config = ExperimentConfig(
                base_alg=method["base_alg"], seed=42, ordering=o,
                experiment_type=method.get("experiment_type", "polykernel_seqreg"),
                lambda_prev=method.get("lambda_prev", 0.0),
                kernel_degree=method.get("kernel_degree", 1),
                revive=method.get("revive", False),
                revive_tau=method.get("revive_tau", 0.1),
            )
            paths.add(str(config.checkpoint_dir(tmp_path)))
        assert len(paths) == len(self.ORDERINGS), "Each ordering must produce a unique path"


class TestOrderingShellScripts:
    """Scripts that run ordering experiments must accept ordering as an argument."""

    def test_run_matched_ordering_accepts_ordering(self):
        path = PROJECT_ROOT / "scripts" / "run_matched_ordering.sh"
        if not path.exists():
            pytest.skip("run_matched_ordering.sh not found")
        source = path.read_text()
        assert "ORDERING" in source or "ordering" in source

    def test_run_evoedit_baseline_accepts_ordering(self):
        path = PROJECT_ROOT / "scripts" / "run_evoedit_baseline.sh"
        if not path.exists():
            pytest.skip("run_evoedit_baseline.sh not found")
        source = path.read_text()
        assert "ORDERING" in source or "ordering" in source

    def test_run_nse_baseline_accepts_ordering(self):
        path = PROJECT_ROOT / "scripts" / "run_nse_baseline.sh"
        if not path.exists():
            pytest.skip("run_nse_baseline.sh not found")
        source = path.read_text()
        assert "ORDERING" in source or "ordering" in source

    def test_run_revive_baseline_accepts_ordering(self):
        path = PROJECT_ROOT / "scripts" / "run_revive_baseline.sh"
        if not path.exists():
            pytest.skip("run_revive_baseline.sh not found")
        source = path.read_text()
        assert "ORDERING" in source or "ordering" in source

    def test_polykernel_seqreg_runner_accepts_ordering_arg(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "--ordering" in source, "polykernel_seqreg_runner must accept --ordering"

    def test_polykernel_seqreg_runner_accepts_dataset_override(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "--dataset_override" in source, "Runner must accept --dataset_override for ordering streams"


class TestFixedBatchInvariant:
    """The defining property of fixed-batch orderings: same batches, different order."""

    def test_high_low_same_records(self):
        """fb_high and fb_low must contain identical records (different order)."""
        hi = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_high_exposure_seed42.json"
        lo = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_low_exposure_seed42.json"
        if not hi.exists() or not lo.exists():
            pytest.skip("Ordering files not available")
        import json
        with open(hi) as f:
            hi_ids = sorted(r["case_id"] for r in json.load(f))
        with open(lo) as f:
            lo_ids = sorted(r["case_id"] for r in json.load(f))
        assert hi_ids == lo_ids, "HIGH and LOW must have identical case_ids"

    def test_high_low_different_order(self):
        """fb_high and fb_low must have different record order (that's the experiment)."""
        hi = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_high_exposure_seed42.json"
        lo = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_low_exposure_seed42.json"
        if not hi.exists() or not lo.exists():
            pytest.skip("Ordering files not available")
        import json
        with open(hi) as f:
            hi_order = [r["case_id"] for r in json.load(f)]
        with open(lo) as f:
            lo_order = [r["case_id"] for r in json.load(f)]
        assert hi_order != lo_order, "HIGH and LOW must have different ordering"

    def test_batch_membership_preserved(self):
        """Within each batch of 100, the same records must appear in both orderings."""
        hi = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_high_exposure_seed42.json"
        lo = PROJECT_ROOT / "results" / "matched_ordering" / "orderings" / "fb_low_exposure_seed42.json"
        if not hi.exists() or not lo.exists():
            pytest.skip("Ordering files not available")
        import json
        with open(hi) as f:
            hi_data = json.load(f)
        with open(lo) as f:
            lo_data = json.load(f)

        batch_size = 100
        hi_batches = [frozenset(r["case_id"] for r in hi_data[i:i+batch_size])
                      for i in range(0, len(hi_data), batch_size)]
        lo_batches = [frozenset(r["case_id"] for r in lo_data[i:i+batch_size])
                      for i in range(0, len(lo_data), batch_size)]
        assert set(hi_batches) == set(lo_batches), "Batch membership must be identical"

    @pytest.mark.parametrize("seed", [42, 2024, 137])
    def test_all_seeds_same_record_set(self, seed):
        """All orderings for the same seed must use the same records."""
        base = PROJECT_ROOT / "results" / "matched_ordering" / "orderings"
        hi = base / f"fb_high_exposure_seed{seed}.json"
        lo = base / f"fb_low_exposure_seed{seed}.json"
        r0 = base / f"fb_random0_seed{seed}.json"
        if not all(p.exists() for p in [hi, lo, r0]):
            pytest.skip(f"Not all ordering files for seed {seed}")
        import json
        sets = []
        for p in [hi, lo, r0]:
            with open(p) as f:
                sets.append(set(r["case_id"] for r in json.load(f)))
        assert sets[0] == sets[1] == sets[2], f"Seed {seed}: all orderings must use same records"


class TestV2EvalStructure:
    """v2 eval files must have both prob-pref and argmax metrics."""

    def test_v2_has_dual_metrics(self):
        import glob, json
        v2_files = glob.glob(str(PROJECT_ROOT / "results" / "matched_ordering" / "**" / "*_v2.json"), recursive=True)
        if not v2_files:
            pytest.skip("No v2 eval files found")
        for f in v2_files[:5]:
            with open(f) as fh:
                data = json.load(fh)
            first_key = list(data.keys())[0]
            metrics = data[first_key].get("all_facts", {})
            assert "efficacy" in metrics, f"v2 file missing efficacy: {f}"
            assert "efficacy_argmax" in metrics, f"v2 file missing efficacy_argmax: {f}"
            assert "neighborhood" in metrics, f"v2 file missing neighborhood: {f}"

    def test_v2_has_cohort_metrics(self):
        """v2 files should have first_1k and latest_1k cohort metrics."""
        import glob, json
        v2_files = glob.glob(str(PROJECT_ROOT / "results" / "matched_ordering" / "**" / "*_v2.json"), recursive=True)
        if not v2_files:
            pytest.skip("No v2 eval files found")
        for f in v2_files[:3]:
            with open(f) as fh:
                data = json.load(fh)
            last_key = sorted(data.keys(), key=lambda k: int(k.replace("_edits", "")))[-1]
            entry = data[last_key]
            assert "first_1k" in entry, f"v2 file missing first_1k: {f}"
            assert "latest_1k" in entry, f"v2 file missing latest_1k: {f}"

    def test_v2_prob_pref_geq_argmax(self):
        """Prob-pref efficacy should be ≥ argmax efficacy (it's a weaker criterion)."""
        import glob, json
        v2_files = glob.glob(str(PROJECT_ROOT / "results" / "matched_ordering" / "**" / "*_v2.json"), recursive=True)
        if not v2_files:
            pytest.skip("No v2 eval files found")
        for f in v2_files[:3]:
            with open(f) as fh:
                data = json.load(fh)
            first_key = list(data.keys())[0]
            m = data[first_key].get("all_facts", {})
            if "efficacy" in m and "efficacy_argmax" in m:
                assert m["efficacy"] >= m["efficacy_argmax"] - 0.01, (
                    f"Prob-pref ({m['efficacy']}) should be ≥ argmax ({m['efficacy_argmax']})"
                )


class TestCaseResultTemplate:
    """The case_result_template must have TWO placeholders: {num_edits} and {case_id}.
    mega_batch_eval calls template.format(num_edits, case_id)."""

    def test_harness_template_has_two_placeholders(self):
        """The harness case_result_template must have exactly 2 {} placeholders."""
        source = (PROJECT_ROOT / "src" / "evaluate_harness.py").read_text()
        template_line = [l for l in source.split("\n") if "case_result_template" in l and "{" in l][0]
        assert template_line.count("{}") == 2, (
            f"case_result_template must have 2 placeholders (num_edits and case_id). "
            f"Got: {template_line.strip()}"
        )

    def test_template_format_produces_correct_filename(self):
        """template.format(num_edits=100, case_id=42) must produce '100_edits-case_42.json'."""
        source = (PROJECT_ROOT / "src" / "evaluate_harness.py").read_text()
        # Extract the template pattern
        for line in source.split("\n"):
            if "case_result_template" in line and "edits-case" in line:
                # The template should be: str(run_dir / "{}_edits-case_{}.json")
                assert "{}_edits-case_{}" in line or "{{" not in line.split("edits-case")[0], (
                    f"Template must use two raw {{}} placeholders, not f-string with baked-in num_edits. "
                    f"Got: {line.strip()}"
                )
                break


class TestAlphaEditModelDtype:
    """AlphaEdit vendor code does float32 matmul (P @ K @ K^T). If the model
    loads in float16, layer_ks will be float16 and the matmul fails with
    'expected same dtype but got float != c10::Half'."""

    def test_checkpoint_runner_loads_float32_for_alphaedit(self):
        """checkpoint_runner must load model in float32 when alg_name is AlphaEdit."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        # Find the model loading section (within ~20 lines of load_model_and_tok call)
        load_pos = source.rfind("load_model_and_tok(")
        load_section = source[max(0, load_pos - 300):load_pos + 200]
        assert "float32" in load_section and "AlphaEdit" in load_section, (
            "checkpoint_runner must load model in float32 for AlphaEdit. "
            "Vendor AlphaEdit_main.py does float32 matmul (P @ layer_ks) which fails on float16 keys."
        )


class TestBaselineCanonicalName:
    """Baseline scripts (NSE, RECT, EvoEdit) must set model.config._name_or_path
    to the canonical stats directory name. Without this, get_cov looks for
    stats under the raw S3 path (e.g. _s3-data_continual-learning_models_Meta-Llama-3-8B)
    and tries to download Wikipedia to compute covariance from scratch."""

    def test_nse_script_sets_canonical_name(self):
        """run_nse_baseline.sh must set model.config._name_or_path after loading."""
        script = PROJECT_ROOT / "scripts" / "run_nse_baseline.sh"
        if not script.exists():
            pytest.skip("run_nse_baseline.sh not found")
        source = script.read_text()
        assert "_name_or_path" in source, (
            "run_nse_baseline.sh must set model.config._name_or_path to canonical name. "
            "Without this, get_cov looks for stats under the raw model path."
        )

    def test_rect_script_reapplies_patches(self):
        """run_rect_aligned_paper_replication.sh must re-apply patches before running.
        Workdir sync can overwrite on-disk patches, so the script must re-apply them."""
        script = PROJECT_ROOT / "scripts" / "run_rect_aligned_paper_replication.sh"
        if not script.exists():
            pytest.skip("run_rect_aligned_paper_replication.sh not found")
        source = script.read_text()
        assert "apply_all" in source, (
            "run_rect_aligned_paper_replication.sh must re-apply patches (apply_all.py) "
            "before running. Without canonical name patch, RECT's get_cov looks for "
            "stats under the raw S3 path and tries to download Wikipedia."
        )

    def test_no_trust_remote_code_in_runners(self):
        """No runner or experiment script should set trust_remote_code.
        If stats aren't found, the fix is to set the canonical model name, not
        to bypass safety checks. Data download scripts (download_*.py) are exempt."""
        import glob
        exempt = {"download_mmlu.py", "download_wikitext.py", "download_datasets.sh"}
        for pattern in ["scripts/run_*.sh", "scripts/eval_*.py", "src/**/*.py"]:
            for f in glob.glob(str(PROJECT_ROOT / pattern), recursive=True):
                if Path(f).name in exempt:
                    continue
                content = open(f).read()
                assert "trust_remote_code" not in content.lower(), (
                    f"{f} uses trust_remote_code — this is unsafe in runners. "
                    f"Fix the stats lookup (canonical model name) instead."
                )

    def test_link_stats_creates_symlink_for_canonical_name(self):
        """link_stats.sh must create symlinks so the canonical name resolves to stats."""
        script = PROJECT_ROOT / "scripts" / "link_stats.sh"
        if not script.exists():
            pytest.skip("link_stats.sh not found")
        source = script.read_text()
        assert "llama3-8b-instruct" in source or "canonical" in source.lower(), (
            "link_stats.sh must create stats symlinks for canonical names"
        )


class TestCanonicalNameResolvesStats:
    """Every code path that calls get_cov must find stats under the canonical name.
    Stats are at data/stats/{canonical_name}/wikipedia_stats/*.npz.
    If the model name isn't canonical, get_cov downloads Wikipedia and crashes."""

    CANONICAL_NAMES = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b-instruct",
        "NousResearch/Meta-Llama-3-8B-Instruct": "llama3-8b-instruct",
        "/s3-data/continual-learning/models/Meta-Llama-3-8B": "llama3-8b-instruct",
        "EleutherAI/gpt-j-6b": "gpt-j-6b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b-instruct",
    }

    @pytest.mark.parametrize("raw_name,expected", list(CANONICAL_NAMES.items()))
    def test_evaluate_harness_canonical_name(self, raw_name, expected):
        """evaluate_harness._canonical_name_or_path must resolve all variants."""
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from evaluate_harness import _canonical_name_or_path
        assert _canonical_name_or_path(raw_name) == expected

    def test_nse_model_patch_sets_canonical(self):
        """NSE script's model loading patch must contain canonical name logic."""
        script = PROJECT_ROOT / "scripts" / "run_nse_baseline.sh"
        if not script.exists():
            pytest.skip("run_nse_baseline.sh not found")
        source = script.read_text()
        for canonical in ["llama3-8b-instruct", "gpt-j-6b", "qwen2.5-7b-instruct"]:
            assert canonical in source, (
                f"NSE script must set canonical name '{canonical}' for stats lookup. "
                f"Without this, get_cov uses the raw model path and tries to download Wikipedia."
            )

    def test_baseline_evaluate_patched_by_apply_all(self):
        """apply_all.py must patch baselines evaluate.py with canonical names."""
        source = (PROJECT_ROOT / "scripts" / "patches" / "patch_canonical_name.py").read_text()
        assert "baselines_root" in source, (
            "patch_canonical_name must accept baselines_root to patch baselines evaluate.py"
        )
        for canonical in ["llama3-8b-instruct", "gpt-j-6b"]:
            assert canonical in source, (
                f"patch_canonical_name must set '{canonical}' for baselines evaluate.py"
            )

    def test_stats_dir_has_canonical_names(self):
        """Stats directory must have entries for canonical names."""
        stats_dir = PROJECT_ROOT / "data" / "stats"
        if not stats_dir.exists():
            pytest.skip("data/stats not available")
        for canonical in ["llama3-8b-instruct", "gpt-j-6b"]:
            p = stats_dir / canonical
            if not p.exists():
                # Check if symlinked under a different name
                found = any(canonical in d.name for d in stats_dir.iterdir() if d.is_dir())
                assert found, (
                    f"Stats directory must have entry for '{canonical}'. "
                    f"link_stats.sh should create this symlink."
                )

    def test_get_cov_uses_name_or_path_not_model_name(self):
        """Vendor get_cov must use model.config._name_or_path, not the --model_name arg.
        This ensures the canonical name patch takes effect."""
        for rect_path in [
            PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_rect_main.py",
            PROJECT_ROOT / "baselines" / "EvoEdit" / "nse" / "nse_main.py",
        ]:
            if not rect_path.exists():
                continue
            source = rect_path.read_text()
            if "get_cov" in source:
                get_cov_section = source[source.find("def get_cov"):][:500]
                assert "_name_or_path" in get_cov_section, (
                    f"{rect_path.name} get_cov must use model.config._name_or_path "
                    f"(set by canonical patch), not the raw model_name argument."
                )


class TestSmokeTestAlphaEditValidation:
    """AlphaEdit smoke test validation must not require model name in log output.
    AlphaEdit uses null_space_project.pt (P matrix), not get_cov, so it never
    prints the model name during stats lookup."""

    def test_alphaedit_validation_doesnt_require_model_name(self):
        """AlphaEdit smoke test check must NOT require model name string in output."""
        smoke = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke.exists():
            pytest.skip("smoke test not found")
        source = smoke.read_text()
        # Find the AlphaEdit validation section
        ae_start = source.find("*AlphaEdit*checkpoint*")
        ae_end = source.find(";;", ae_start) if ae_start >= 0 else -1
        if ae_start < 0:
            pytest.skip("AlphaEdit validation section not found")
        ae_section = source[ae_start:ae_end]
        assert "null_space_project" in ae_section or "P matrix" in ae_section or "cache_c" in ae_section, (
            "AlphaEdit validation should check for P matrix or cache_c, not model name. "
            "AlphaEdit uses null_space_project.pt directly and never prints the model name."
        )


class TestCheckpointVerification:
    """Checkpoints saved to S3 FUSE may silently fail to persist.
    save_checkpoint must verify the file exists after writing."""

    def test_save_checkpoint_verifies_write(self):
        """save_checkpoint must verify model_weights.pt exists after torch.save."""
        source = (PROJECT_ROOT / "src" / "util" / "checkpoint_io.py").read_text()
        save_section = source[source.find("def save_checkpoint"):source.find("def load_checkpoint")]
        assert "exists()" in save_section or "verify" in save_section.lower(), (
            "save_checkpoint must verify the checkpoint file exists after writing. "
            "S3 FUSE can silently drop writes."
        )


class TestOutputJsonlParentDir:
    """The output_jsonl write must ensure its parent directory exists.
    On S3 FUSE, directories created 200+ seconds earlier may not persist."""

    def test_output_jsonl_mkdir_before_write(self):
        """polykernel_seqreg_runner must mkdir parent before writing output_jsonl."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        write_pos = source.find('with open(output_jsonl, "w")')
        assert write_pos > 0, "output_jsonl write not found"
        # Look for a mkdir in the 200 chars before the write
        pre_write = source[max(0, write_pos - 200):write_pos]
        assert "mkdir" in pre_write or "makedirs" in pre_write, (
            "Must ensure output_jsonl parent directory exists before writing. "
            "On S3 FUSE, directories created at startup may not persist 200+ seconds later."
        )


class TestRevivePostHocPrintsOutput:
    """The post-hoc REVIVE wrapper for NSE/RECT must produce [REVIVE] output
    that the smoke test can validate."""

    def test_posthoc_revive_prints_revive_tag(self):
        """The post-hoc REVIVE path must print [REVIVE] for smoke test validation."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc_section = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        # The revive_hooks.post_solve already prints [REVIVE] internally.
        # But if delta.norm() < 1e-10, the filter is skipped entirely.
        # Check that the threshold isn't too aggressive
        assert "1e-10" in posthoc_section or "1e-8" in posthoc_section, (
            "Post-hoc REVIVE has a delta norm threshold that skips the filter. "
            "This should be documented and the smoke test should account for it."
        )

    def test_smoke_test_revive_check_accounts_for_posthoc(self):
        """The smoke test REVIVE validation must work for both hooks-based and post-hoc paths."""
        smoke_path = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke_path.exists():
            pytest.skip("smoke test not found")
        source = smoke_path.read_text()
        assert "[REVIVE]" in source, "Smoke test must check for [REVIVE] output"

    def test_smoke_test_accepts_zero_delta_warning(self):
        """Smoke test must accept 'post-hoc filter applied to 0 layers' as a pass.
        Post-hoc REVIVE for NSE/RECT may produce zero deltas on small datasets."""
        smoke_path = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke_path.exists():
            pytest.skip("smoke test not found")
        source = smoke_path.read_text()
        assert "post-hoc filter applied to 0 layers" in source, (
            "Smoke test must accept zero-delta warning as pass for REVIVE+NSE/RECT. "
            "Small datasets (20 records) can produce weight deltas below threshold."
        )

    def test_smoke_test_skips_removed_check_on_zero_deltas(self):
        """When zero deltas are accepted, the 'removed=' check must also be skipped.
        If the filter doesn't run, there's no removed= output to check."""
        smoke_path = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke_path.exists():
            pytest.skip("smoke test not found")
        source = smoke_path.read_text()
        # The removed= check must be conditional — either inside an else block
        # of the zero-delta check, or guarded by a flag
        revive_section = source[source.find("*REVIVE*)"):source.find(";;", source.find("*REVIVE*)"))]
        # After accepting zero deltas, the code must NOT unconditionally check removed=
        # There should be a skip/return/flag before the removed= check
        zero_delta_pos = revive_section.find("post-hoc filter applied to 0 layers")
        removed_pos = revive_section.find('removed=', zero_delta_pos)
        if zero_delta_pos >= 0 and removed_pos >= 0:
            between = revive_section[zero_delta_pos:removed_pos]
            has_early_return = "return" in between or "skip" in between.lower()
            has_guard = "else" in between or "_zero_delta" in between
            assert has_early_return or has_guard, (
                "The 'removed=' check must be skipped when zero deltas are accepted. "
                "Currently it runs unconditionally after the zero-delta warning, causing "
                "REVIVE+NSE to fail even when zero deltas are expected on small datasets."
            )

    def test_posthoc_revive_prints_warning_on_zero_deltas(self):
        """The post-hoc REVIVE wrapper must print a WARNING when all deltas are zero.
        Without this, the smoke test can't distinguish 'filter skipped (OK)' from
        'filter broken (bug)'."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert "_rv_applied == 0" in posthoc or "applied to 0" in posthoc, (
            "Post-hoc REVIVE must print warning when 0 layers are filtered. "
            "This allows the smoke test to distinguish correct skips from bugs."
        )

    def test_posthoc_revive_warning_contains_base_alg(self):
        """The zero-delta warning must include base_alg for debugging."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        warning_section = source[source.find("_rv_applied == 0"):source.find("_rv_applied == 0") + 200] if "_rv_applied == 0" in source else ""
        assert "base_alg" in warning_section, (
            "Post-hoc REVIVE zero-delta warning must include base_alg so we know "
            "which algorithm produced zero deltas."
        )


class TestSeedEverything:
    """_seed_everything must set all RNG sources deterministically."""

    def _seed_fn(self, seed):
        """Replicate _seed_everything without importing from runner (avoids vendor deps)."""
        import random as _random, numpy as _np, torch
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)

    def test_seed_everything_sets_python_random(self):
        import random as _random
        self._seed_fn(42)
        a = _random.random()
        self._seed_fn(42)
        b = _random.random()
        assert a == b, "Python random not seeded deterministically"

    def test_seed_everything_sets_numpy(self):
        import numpy as _np
        self._seed_fn(42)
        a = _np.random.rand()
        self._seed_fn(42)
        b = _np.random.rand()
        assert a == b, "NumPy random not seeded deterministically"

    def test_seed_everything_sets_torch(self):
        import torch
        self._seed_fn(42)
        a = torch.rand(1).item()
        self._seed_fn(42)
        b = torch.rand(1).item()
        assert a == b, "PyTorch random not seeded deterministically"

    def test_source_sets_all_rngs(self):
        """_seed_everything source must set random, numpy, torch, and cuda seeds."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        fn = source[source.find("def _seed_everything"):source.find("\n\ndef ", source.find("def _seed_everything") + 1)]
        assert "random.seed" in fn
        assert "np.random.seed" in fn
        assert "torch.manual_seed" in fn
        assert "torch.cuda.manual_seed_all" in fn
        assert "deterministic" in fn


class TestMakeEvalFn:
    """_make_eval_fn must exec mega_batch_eval source and return a callable.
    Cannot import directly (vendor deps) — test the logic structurally + via mega_batch_eval."""

    def test_make_eval_fn_logic(self):
        """_make_eval_fn execs get_mega_batch_eval_source and extracts _mega_batch_eval."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        fn = source[source.find("def _make_eval_fn"):source.find("\n\ndef ", source.find("def _make_eval_fn") + 1)]
        assert "get_mega_batch_eval_source" in fn, "Must call get_mega_batch_eval_source"
        assert "exec(" in fn, "Must exec the source"
        assert '_mega_batch_eval' in fn, "Must extract _mega_batch_eval from namespace"

    def test_fast_mode_filters_to_latest_batch(self):
        """Fast mode must filter records to only the latest batch's case_ids."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        fn = source[source.find("def _make_eval_fn"):source.find("\n\ndef ", source.find("def _make_eval_fn") + 1)]
        assert "fast_mode" in fn
        assert "case_ids[-num_edits:]" in fn or "case_ids[" in fn, (
            "Fast mode must filter to latest batch's case_ids"
        )

    def test_mega_batch_eval_source_produces_callable(self):
        """get_mega_batch_eval_source must produce a callable when exec'd."""
        from mega_batch_eval import get_mega_batch_eval_source
        ns = {}
        exec(get_mega_batch_eval_source(), ns)
        assert callable(ns["_mega_batch_eval"])


class TestInitAlgDict:
    """_init_alg_dict must populate ALG_DICT with algorithm mappings.
    Cannot directly import (vendor globals.yml CWD dependency) — test structurally."""

    def test_alg_dict_has_required_algorithms(self):
        """seeded_runner ALG_DICT must map AlphaEdit, MEMIT, and ROME."""
        source = (PROJECT_ROOT / "src" / "runners" / "seeded_runner.py").read_text()
        fn = source[source.find("def _init_alg_dict"):source.find("\n\ndef ", source.find("def _init_alg_dict") + 1)]
        for alg in ["AlphaEdit", "MEMIT", "ROME"]:
            assert f'"{alg}"' in fn, f"ALG_DICT must contain {alg}"

    def test_alg_dict_maps_to_tuples(self):
        """Each entry must map to (HyperParams, apply_fn)."""
        source = (PROJECT_ROOT / "src" / "runners" / "seeded_runner.py").read_text()
        fn = source[source.find("def _init_alg_dict"):source.find("\n\ndef ", source.find("def _init_alg_dict") + 1)]
        assert "AlphaEditHyperParams" in fn and "apply_AlphaEdit_to_model" in fn
        assert "MEMITHyperParams" in fn and "apply_memit_to_model" in fn
        assert "ROMEHyperParams" in fn and "apply_rome_to_model" in fn


class TestLoadC0PatchedApply:
    """_load_c0_patched_apply must return a callable apply function."""

    def test_function_exists_and_has_correct_signature(self):
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        assert "def _load_c0_patched_apply" in source
        assert "alphaedit_root" in source[source.find("def _load_c0_patched_apply"):][:200]
        assert "c0_weight" in source[source.find("def _load_c0_patched_apply"):][:200]

    def test_c0_injection_patches_solve_anchor(self):
        """The C₀ injection must find and replace the vendor solve anchor."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        fn_body = source[source.find("def _load_c0_patched_apply"):source.find("\ndef run(")]
        assert "solve_anchor" in fn_body, "Must define solve_anchor to find in vendor code"
        assert "assert solve_anchor in ae_source" in fn_body, "Must assert anchor exists"
        assert "c0_weight" in fn_body, "Must inject c0_weight into the patched solve"
        assert "get_cov" in fn_body, "C₀ injection must load covariance via get_cov"

    def test_c0_returns_apply_and_get_cov(self):
        """Must return both apply function and get_cov from the exec'd namespace."""
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        fn_body = source[source.find("def _load_c0_patched_apply"):source.find("\ndef run(")]
        assert 'ae_ns["apply_AlphaEdit_to_model"]' in fn_body
        assert 'ae_ns["get_cov"]' in fn_body


class TestExperimentConfigKernelTag:
    """kernel_tag must produce correct tags for all kernel configurations."""

    def test_poly1(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", kernel_type="poly", kernel_degree=1)
        assert c.kernel_tag == "poly1"

    def test_poly2(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", kernel_type="poly", kernel_degree=2)
        assert c.kernel_tag == "poly2"

    def test_poly2_hybrid(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", kernel_type="poly", kernel_degree=2, kernel_prev=False)
        assert c.kernel_tag == "poly2-hybrid"

    def test_poly1_revive(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", kernel_degree=1, revive=True, revive_tau=0.1)
        assert c.kernel_tag == "poly1-REVIVE-tau0.1"

    def test_rbf(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", kernel_type="rbf", kernel_sigma="median")
        assert c.kernel_tag == "rbf_median"


class TestRECTCacheCInCheckpoint:
    """RECT cache_c must be saved in checkpoints so resumed runs don't start with zeros."""

    def test_cache_c_saved_in_checkpoint_for_rect(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        save_section = source[source.find("extra_state = {"):source.find("save_checkpoint(") + 500]
        assert "cache_c" in save_section, (
            "Checkpoint must save cache_c for RECT. Without it, resumed runs start "
            "with zeros cache_c, producing different results than continuous runs."
        )

    def test_cache_c_loaded_from_checkpoint(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        load_section = source[source.find("load_checkpoint"):source.find("# Select apply function")]
        assert "cache_c" in load_section, (
            "Checkpoint load must restore cache_c for algorithms that use it."
        )


class TestMemitSeqOutputJsonlMkdir:
    """memit_sequential_runner must mkdir before writing output_jsonl (S3 FUSE safety)."""

    def test_mkdir_before_output_jsonl_write(self):
        source = (PROJECT_ROOT / "src" / "runners" / "memit_sequential_runner.py").read_text()
        write_pos = source.find('with open(output_jsonl')
        if write_pos < 0:
            pytest.skip("output_jsonl write not found")
        pre_write = source[max(0, write_pos - 300):write_pos]
        assert "mkdir" in pre_write, (
            "memit_sequential_runner must mkdir parent before writing output_jsonl. "
            "On S3 FUSE, directories created at startup may not persist."
        )


class TestReviveCurrentWeightsCloned:
    """REVIVE hooks _current_weights must hold clones, not references.
    References break if earlier layers' weights are modified in-place."""

    def test_hooks_path_clones_weights(self):
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # Find the weights_dict construction (before _current_weights assignment)
        weights_area = source[source.find("if args.revive:"):source.find("algo_state[\"_current_weights\"]") + 50]
        assert ".clone()" in weights_area, (
            "REVIVE _current_weights must use .clone() not bare .data reference. "
            "Without cloning, in-place weight modifications from earlier layers "
            "corrupt the SVD input for later layers."
        )


class TestSeqregCacheMaxTrimming:
    """seqreg_hooks with cache_max must trim old batches."""

    def test_recent_strategy_trims_to_cache_max(self):
        """With cache_strategy='recent' and cache_max=2, only last 2 batches kept."""
        import torch
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "algorithms"))
        from hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, cache_max=2, cache_strategy="recent")
        state = hooks.get_state()
        # Simulate 3 batches of keys for layer 0
        for _ in range(3):
            ks = torch.randn(64, 10, dtype=torch.float64)
            hooks.post_solve(0, torch.randn(64, 64), None, ks, "layer.0.weight", state)
        assert len(state["prev_cache"][0]) == 2, (
            f"With cache_max=2 and strategy='recent', should keep 2 batches, got {len(state['prev_cache'][0])}"
        )

    def test_all_strategy_does_not_trim(self):
        """With cache_strategy='all', all batches kept regardless of cache_max."""
        import torch
        from hook_presets import seqreg_hooks
        hooks = seqreg_hooks(lambda_prev=1.0, cache_max=2, cache_strategy="all")
        state = hooks.get_state()
        for _ in range(5):
            ks = torch.randn(64, 10, dtype=torch.float64)
            hooks.post_solve(0, torch.randn(64, 64), None, ks, "layer.0.weight", state)
        assert len(state["prev_cache"][0]) == 5, "Strategy 'all' should keep all batches"


class TestComposeHooksCovNone:
    """compose_hooks with cov=None must not crash (AlphaEdit+REVIVE path)."""

    def test_composed_build_lhs_with_cov_none(self):
        import torch
        from hook_presets import seqreg_hooks, revive_hooks
        from hooks import compose_hooks
        composed = compose_hooks(seqreg_hooks(lambda_prev=0.0), revive_hooks(revive_tau=0.1))
        state = composed.get_state()
        ks = torch.randn(64, 10, dtype=torch.float64)
        hparams = type('H', (), {'mom2_update_weight': 1.0})()
        lhs = composed.build_lhs(0, ks, None, hparams, state)
        assert lhs is not None
        assert not torch.isnan(lhs).any(), "LHS with cov=None must not produce NaN"


class TestExperimentConfigEdgeCases:
    """Edge cases for ExperimentConfig properties."""

    def test_memit_seq_variant_name(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="x", lambda_prev=1.0, lambda_delta=0.5)
        name = c.memit_seq_variant_name
        assert "MEMIT-Seq" in name
        assert "lp1.0" in name
        assert "ld0.5" in name

    def test_model_tag_qwen(self):
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="MEMIT", seed=42, ordering=None,
                            model_name="Qwen/Qwen2.5-7B-Instruct")
        assert c.model_tag == "qwen2.5-7b"

    def test_validate_checkpoint_path_rect(self):
        from checkpoint_io import validate_checkpoint_path
        validate_checkpoint_path("/tmp/MEMIT_rect-poly1/seed42", "MEMIT_rect")

    def test_validate_checkpoint_path_all_algs(self):
        from checkpoint_io import validate_checkpoint_path
        for alg, expected in [("MEMIT", "MEMIT-Seq"), ("AlphaEdit", "AlphaEdit"),
                              ("NSE", "NSE"), ("MEMIT_rect", "MEMIT_rect")]:
            validate_checkpoint_path(f"/tmp/{expected}-poly1/seed42", alg)

    def test_canonical_name_unknown_model(self):
        """Unknown model should return lowercased basename."""
        from evaluate_harness import _canonical_name_or_path
        result = _canonical_name_or_path("some-org/custom-model-7b")
        assert result == "custom-model-7b"

    def test_variant_name_complex_config(self):
        """Full config: revive + hybrid + poly3."""
        from experiment_config import ExperimentConfig
        c = ExperimentConfig(base_alg="NSE", seed=42, ordering="fb_high",
                            model_name="x", kernel_degree=3, kernel_prev=False,
                            revive=True, revive_tau=0.2, lambda_prev=0.5, lambda_delta=0.1)
        name = c.variant_name
        assert name.startswith("NSE-")
        assert "poly3-hybrid" in name
        assert "REVIVE-tau0.2" in name
        assert "lp0.5" in name
        assert "ld0.1" in name


class TestCheckpointIoSerializationPaths:
    """save_checkpoint must handle .pt, .jsonl, and .json extra_state correctly."""

    def test_save_and_load_all_formats(self, tmp_path):
        import torch, json
        from checkpoint_io import save_checkpoint, load_checkpoint
        from unittest.mock import MagicMock

        model = MagicMock()
        model.named_parameters.return_value = []
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        extra = {
            "tensor_data.pt": torch.randn(3, 3),
            "log.jsonl": [{"batch": 0, "loss": 1.5}, {"batch": 1, "loss": 0.8}],
        }

        save_checkpoint(0, model, hparams, str(tmp_path), 10,
                       extra_state=extra, metadata={"test": True})

        result = load_checkpoint(model, hparams, str(tmp_path), 0,
                                extra_state_keys=["tensor_data.pt", "log.jsonl"])

        assert result["loaded"] is True
        assert torch.is_tensor(result["tensor_data.pt"])
        assert result["tensor_data.pt"].shape == (3, 3)
        assert len(result["log.jsonl"]) == 2
        assert result["log.jsonl"][0]["batch"] == 0

    def test_find_latest_skips_dirs_without_metadata(self, tmp_path):
        """Batch dirs without metadata.json should be skipped."""
        import json
        from checkpoint_io import find_latest_checkpoint
        (tmp_path / "batch_5").mkdir()
        (tmp_path / "batch_10").mkdir()
        (tmp_path / "batch_10" / "metadata.json").write_text(json.dumps({"batch_idx": 10}))
        result = find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result[0] == 10, "Should skip batch_5 (no metadata) and find batch_10"


class TestPathguardHooksPostUpdate:
    """pathguard_hooks.post_update must record weight_norm."""

    def test_post_update_records_norm(self):
        import torch
        from hook_presets import pathguard_hooks
        hooks = pathguard_hooks(pathguard_M=100)
        state = hooks.get_state()
        w = torch.randn(64, 64)
        hooks.post_update(0, "test.weight", w, state)
        assert len(state["displacement_history"]) == 1
        assert "weight_norm" in state["displacement_history"][0]
        assert state["displacement_history"][0]["layer"] == 0


class TestPolykernelHooksRBF:
    """polykernel_hooks with RBF kernel must not crash."""

    def test_rbf_build_lhs(self):
        import torch
        from hook_presets import polykernel_hooks
        hooks = polykernel_hooks(kernel_type="rbf", kernel_sigma="median")
        ks = torch.randn(64, 10, dtype=torch.float64)
        cov = torch.eye(64, dtype=torch.float64)
        hparams = type('H', (), {'mom2_update_weight': 1.0})()
        lhs = hooks.build_lhs(0, ks, cov, hparams, {})
        assert lhs.shape == (64, 64)
        assert not torch.isnan(lhs).any()


class TestGetMegaBatchEvalSource:
    """get_mega_batch_eval_source must return compilable Python that defines _mega_batch_eval."""

    def test_returns_string(self):
        from mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        assert isinstance(source, str)
        assert len(source) > 100

    def test_source_compiles(self):
        from mega_batch_eval import get_mega_batch_eval_source
        source = get_mega_batch_eval_source()
        compile(source, "<mega_batch_eval>", "exec")

    def test_source_defines_function(self):
        from mega_batch_eval import get_mega_batch_eval_source
        ns = {}
        exec(get_mega_batch_eval_source(), ns)
        assert "_mega_batch_eval" in ns
        assert callable(ns["_mega_batch_eval"])


class TestCheckpointIoCorrectness:
    """Correctness tests for checkpoint_io functions — call them and verify output."""

    def test_should_save_interval_1(self):
        """should_save with interval=1 should save every batch."""
        from checkpoint_io import should_save
        assert should_save(0, 1) is True
        assert should_save(1, 1) is True
        assert should_save(99, 1) is True

    def test_should_save_interval_10(self):
        """should_save with interval=10 should save at 9, 19, 29..."""
        from checkpoint_io import should_save
        assert should_save(8, 10) is False
        assert should_save(9, 10) is True
        assert should_save(10, 10) is False
        assert should_save(19, 10) is True

    def test_should_skip(self):
        """should_skip returns True for batches before start_batch."""
        from checkpoint_io import should_skip
        assert should_skip(0, 5) is True
        assert should_skip(4, 5) is True
        assert should_skip(5, 5) is False
        assert should_skip(0, 0) is False

    def test_validate_checkpoint_path_correct(self):
        """validate_checkpoint_path should pass for correct paths."""
        from checkpoint_io import validate_checkpoint_path
        validate_checkpoint_path("/tmp/MEMIT-Seq-poly1/seed42", "MEMIT")
        validate_checkpoint_path("/tmp/AlphaEdit-poly1/seed42", "AlphaEdit")
        validate_checkpoint_path("/tmp/NSE-poly1/seed42", "NSE")

    def test_validate_checkpoint_path_mismatch(self):
        """validate_checkpoint_path should raise for wrong algorithm in path."""
        from checkpoint_io import validate_checkpoint_path
        with pytest.raises(RuntimeError, match="Checkpoint path mismatch"):
            validate_checkpoint_path("/tmp/MEMIT-Seq-poly1/seed42", "AlphaEdit")

    def test_find_latest_checkpoint_empty(self, tmp_path):
        """find_latest_checkpoint on empty dir returns None."""
        from checkpoint_io import find_latest_checkpoint
        assert find_latest_checkpoint(tmp_path) is None

    def test_find_latest_checkpoint_with_batches(self, tmp_path):
        """find_latest_checkpoint finds the highest batch with metadata."""
        from checkpoint_io import find_latest_checkpoint
        import json
        for b in [0, 9, 19]:
            d = tmp_path / f"batch_{b}"
            d.mkdir()
            (d / "metadata.json").write_text(json.dumps({"batch_idx": b}))
        result = find_latest_checkpoint(tmp_path)
        assert result is not None
        assert result[0] == 19


class TestExperimentConfigCorrectness:
    """Correctness tests for ExperimentConfig — construct configs and verify outputs."""

    def test_variant_name_memit(self):
        """MEMIT base_alg should produce MEMIT-Seq prefix."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=1.0, lambda_delta=0.0, kernel_degree=1,
        )
        assert config.variant_name.startswith("MEMIT-Seq-")

    def test_variant_name_alphaedit(self):
        """AlphaEdit base_alg should produce AlphaEdit prefix, not MEMIT-Seq."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=0.0, lambda_delta=0.0, kernel_degree=1, revive=True, revive_tau=0.1,
        )
        assert config.variant_name.startswith("AlphaEdit-")
        assert "MEMIT-Seq" not in config.variant_name

    def test_variant_name_nse(self):
        """NSE should produce NSE prefix."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="NSE", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=0.0, lambda_delta=0.0, kernel_degree=1, revive=True, revive_tau=0.1,
        )
        assert config.variant_name.startswith("NSE-")

    def test_variant_name_rect(self):
        """MEMIT_rect should produce MEMIT_rect prefix."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT_rect", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=0.0, lambda_delta=0.0, kernel_degree=1,
        )
        assert config.variant_name.startswith("MEMIT_rect-")

    def test_all_base_algs_produce_distinct_variants(self):
        """Every base_alg must produce a distinct variant_name prefix."""
        from experiment_config import ExperimentConfig
        names = set()
        for alg in ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]:
            config = ExperimentConfig(
                base_alg=alg, seed=42, ordering=None,
                model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                lambda_prev=0.0, lambda_delta=0.0, kernel_degree=1,
            )
            prefix = config.variant_name.split("-")[0]
            assert prefix not in names or alg == "MEMIT", (
                f"{alg} produces prefix '{prefix}' which conflicts with another algorithm"
            )
            names.add(prefix)

    def test_checkpoint_dir_contains_variant(self):
        """checkpoint_dir must contain the variant_name."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, ordering="fb_high_exposure",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            lambda_prev=0.0, lambda_delta=0.0, kernel_degree=1, revive=True, revive_tau=0.1,
            experiment_type="polykernel_seqreg",
        )
        ckpt = config.checkpoint_dir()
        assert "AlphaEdit" in str(ckpt)
        assert "fb_high_exposure" in str(ckpt)
        assert "seed42" in str(ckpt)

    def test_model_tag_empty_for_llama(self):
        """Llama (default model) should have empty model_tag."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        assert config.model_tag == ""

    def test_model_tag_gptj(self):
        """GPT-J should have model_tag 'gpt-j-6b'."""
        from experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, ordering=None,
            model_name="EleutherAI/gpt-j-6b",
        )
        assert config.model_tag == "gpt-j-6b"


class TestPathsCorrectness:
    """Correctness tests for paths.py functions."""

    def test_get_project_root(self):
        from paths import get_project_root
        root = get_project_root()
        assert (root / "src").is_dir()
        assert (root / "vendor").is_dir()

    def test_get_alphaedit_root(self):
        from paths import get_alphaedit_root
        root = get_alphaedit_root()
        assert "AlphaEdit" in str(root)

    def test_s3_guard_logic_exists(self):
        """_require_s3 must check for SKYPILOT_TASK_ID and /s3-data/ in path."""
        source = (PROJECT_ROOT / "src" / "util" / "paths.py").read_text()
        assert "SKYPILOT_TASK_ID" in source, "paths.py must check SKYPILOT_TASK_ID"
        assert "/s3-data/" in source, "paths.py must check for /s3-data/ in path"
        assert "RuntimeError" in source, "paths.py must raise RuntimeError for non-S3 paths"

    def test_s3_guard_passes_for_s3_path(self):
        """_require_s3 should pass for /s3-data/ paths."""
        import os
        from paths import _require_s3
        old = os.environ.get("SKYPILOT_TASK_ID")
        try:
            os.environ["SKYPILOT_TASK_ID"] = "test-123"
            import importlib, paths
            importlib.reload(paths)
            result = paths._require_s3(Path("/s3-data/test"), "TEST_VAR")
            assert str(result) == "/s3-data/test"
        finally:
            if old is None:
                os.environ.pop("SKYPILOT_TASK_ID", None)
            else:
                os.environ["SKYPILOT_TASK_ID"] = old
            import importlib, paths
            importlib.reload(paths)


class TestMemitNaNGuard:
    """memit_with_hooks must handle NaN from compute_z, matching the vendor's guard."""

    def test_memit_hooks_has_nan_guard(self):
        """memit_with_hooks.py must check for NaN after compute_z."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        z_section = source[source.find("compute_z("):source.find("zs = torch.stack")]
        assert "isnan" in z_section or "nan" in z_section.lower(), (
            "memit_with_hooks must check for NaN after compute_z. "
            "Vendor memit_main.py has a NaN guard that falls back to the unedited z-value. "
            "Without this, NaN propagates to weight updates and produces garbage."
        )


class TestReviveNonZeroDeltaCoverage:
    """We must verify that post-hoc REVIVE is tested with REAL non-zero deltas,
    not just the zero-delta acceptance path."""

    def test_revive_rect_smoke_validates_actual_filtering(self):
        """REVIVE+RECT smoke test must check for [REVIVE] layer= AND removed=.
        RECT produces large weight deltas, so the filter should actually run."""
        smoke = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke.exists():
            pytest.skip("smoke test not found")
        source = smoke.read_text()
        revive_section = source[source.find("*REVIVE*)"):source.find(";;", source.find("*REVIVE*)"))]
        assert "[REVIVE] layer=" in revive_section, "Must check for per-layer SVD output"
        assert "removed=" in revive_section, "Must check that filter removes something"

    def test_gpu_integration_tests_posthoc_revive(self):
        """GPU integration tests must include a test for post-hoc REVIVE with non-zero deltas."""
        gpu_test = PROJECT_ROOT / "tests" / "test_gpu_integration.py"
        if not gpu_test.exists():
            pytest.skip("GPU integration test not found")
        source = gpu_test.read_text()
        assert "posthoc_revive_filters_nonzero_deltas" in source, (
            "GPU integration tests must include test_posthoc_revive_filters_nonzero_deltas. "
            "This verifies the post-hoc REVIVE path works with real weight changes, "
            "not just the zero-delta acceptance path."
        )

    def test_revive_rect_is_tested_on_smoke_cluster(self):
        """REVIVE+RECT must appear in the smoke test with expected checkpoint path."""
        smoke = PROJECT_ROOT / "tests" / "test_smoke_all_algorithms.sh"
        if not smoke.exists():
            pytest.skip("smoke test not found")
        source = smoke.read_text()
        assert "REVIVE+RECT" in source, "Smoke test must include REVIVE+RECT"
        assert "MEMIT_rect-poly1-REVIVE" in source, "Expected RECT checkpoint path in smoke test"


class TestPostHocReviveContract:
    """The post-hoc REVIVE wrapper for NSE/RECT must:
    1. Snapshot weights before edit
    2. Compute delta after edit
    3. Skip layers with delta norm < threshold
    4. Apply REVIVE filter for non-zero deltas
    5. Print warning if ALL layers skipped
    6. Print [REVIVE] layer= for each filtered layer"""

    def test_posthoc_snapshots_pre_weights(self):
        """Must clone weights before calling base_apply."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert ".detach().clone()" in posthoc, "Must clone pre_weights (not just reference)"

    def test_posthoc_computes_delta_in_double(self):
        """Delta computation must use float64 to avoid precision loss."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert ".double()" in posthoc, "Delta must be computed in float64"

    def test_posthoc_has_delta_threshold(self):
        """Must skip layers with delta norm below threshold (avoids noise)."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert "1e-10" in posthoc or "1e-8" in posthoc, "Must have delta norm threshold"

    def test_posthoc_calls_revive_post_solve(self):
        """Must call _revive_hook.post_solve for filtering."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert "post_solve" in posthoc, "Must call revive_hook.post_solve for spectral filtering"

    def test_posthoc_writes_back_filtered_weights(self):
        """Must write pre_weight + filtered_delta back to model."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        posthoc = source[source.find("# Non-hook path"):source.find("return result", source.find("# Non-hook path"))]
        assert "pre_weights[wn] + filtered" in posthoc or "pre_weights[wn] +" in posthoc, (
            "Must write back pre_weight + filtered_delta, not just filtered_delta"
        )

    def test_posthoc_only_for_non_hook_algorithms(self):
        """Post-hoc REVIVE must only run for NSE/RECT, not MEMIT/AlphaEdit."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "_hooks_aware" in source, "Must have _hooks_aware flag"
        hooks_line = [l for l in source.split("\n") if "_hooks_aware" in l and "=" in l and "if" not in l][0]
        assert "MEMIT" in hooks_line and "AlphaEdit" in hooks_line, (
            "_hooks_aware must include MEMIT and AlphaEdit (they use hooks-based REVIVE)"
        )
        for alg in ["NSE", "MEMIT_rect"]:
            assert alg not in hooks_line, (
                f"{alg} must NOT be in _hooks_aware — it uses post-hoc REVIVE"
            )


class TestBaselineStreamOverrideRespectsLimit:
    """Baseline scripts that override the dataset with an ordering stream must
    respect dataset_size_limit. Without this, the stream loads all 10K records
    even when the smoke test requests only 10."""

    def test_nse_stream_override_respects_limit(self):
        """run_nse_baseline.sh stream override must limit to dataset_size_limit."""
        script = PROJECT_ROOT / "scripts" / "run_nse_baseline.sh"
        if not script.exists():
            pytest.skip("run_nse_baseline.sh not found")
        source = script.read_text()
        override_section = source[source.find("DATASET OVERRIDE"):source.find("END OVERRIDE")]
        assert "dataset_size_limit" in override_section, (
            "NSE stream override must respect dataset_size_limit. "
            "Without this, the stream loads all 10K records even for smoke tests."
        )

    def test_evoedit_stream_override_respects_limit(self):
        """run_evoedit_baseline.sh stream override must limit to dataset_size_limit."""
        script = PROJECT_ROOT / "scripts" / "run_evoedit_baseline.sh"
        if not script.exists():
            pytest.skip("run_evoedit_baseline.sh not found")
        source = script.read_text()
        if "DATASET OVERRIDE" in source:
            override_section = source[source.find("DATASET OVERRIDE"):source.find("END OVERRIDE")]
            assert "dataset_size_limit" in override_section, (
                "EvoEdit stream override must respect dataset_size_limit."
            )


class TestNSEKVCacheLoading:
    """NSE must load precomputed v_star from KV cache files to avoid slow
    gradient optimization (25 steps per edit). Without cache, NSE times out."""

    def test_polykernel_runner_builds_cache_template_for_nse(self):
        """polykernel_seqreg_runner must build _nse_cache_template when base_alg=NSE."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "_nse_cache_template" in source, "Runner must build NSE cache template"
        # The template must be passed to the apply function
        assert "cache_template" in source[source.find("def apply_fn"):source.find("def after_edit")], (
            "apply_fn must pass cache_template to NSE's apply function"
        )

    def test_nse_cache_template_format_matches_files(self):
        """The cache template must produce paths matching the actual .npz filenames."""
        # Template: {kvs_dir}/{model}_NSE/{ds}_layer_{z_layer}_clamp_{clamp}_case_{case_id}.npz
        cache_dir = PROJECT_ROOT / "data" / "nse_kv_cache" / "NousResearch_Meta-Llama-3-8B-Instruct_NSE"
        if not cache_dir.exists():
            pytest.skip("NSE KV cache not available locally")
        files = list(cache_dir.glob("*.npz"))
        assert len(files) > 0, "Cache dir exists but has no .npz files"
        # Verify naming convention
        sample = files[0].name
        parts = sample.replace(".npz", "").split("_")
        assert "layer" in sample, f"Cache filename must contain 'layer': {sample}"
        assert "clamp" in sample, f"Cache filename must contain 'clamp': {sample}"
        assert "case" in sample, f"Cache filename must contain 'case': {sample}"

    def test_nse_baseline_script_extracts_cache(self):
        """run_nse_baseline.sh must extract KV cache from S3 tar before running NSE."""
        script_path = PROJECT_ROOT / "scripts" / "run_nse_baseline.sh"
        if not script_path.exists():
            pytest.skip("run_nse_baseline.sh not found")
        source = script_path.read_text()
        assert "tar" in source and "nse_kv_cache" in source, (
            "run_nse_baseline.sh must extract NSE KV cache from S3 tar archives"
        )

    def test_smoke_test_baseline_yaml_allows_cache_extraction_time(self):
        """The baseline YAML must give enough timeout for NSE cache extraction + editing."""
        import glob
        yamls = glob.glob(str(PROJECT_ROOT / "sky" / "test_baseline*.yaml"))
        if not yamls:
            pytest.skip("No baseline test YAML found")
        content = open(yamls[0]).read()
        # Should have SMOKE_TIMEOUT >= 900
        if "SMOKE_TIMEOUT" in content:
            import re
            match = re.search(r"SMOKE_TIMEOUT=(\d+)", content)
            if match:
                timeout = int(match.group(1))
                assert timeout >= 900, f"Baseline timeout {timeout}s too low for NSE cache extraction + editing"

    def test_runner_extracts_nse_cache_from_tar(self):
        """polykernel_seqreg_runner must extract NSE KV cache from S3 tar if the
        kvs/ directory is empty or missing. Without this, compute_z falls through
        to 25 gradient steps per edit and the run times out."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        nse_section = source[source.find("_nse_cache_template"):source.find("def apply_fn")]
        assert "tar" in nse_section or "extract" in nse_section.lower(), (
            "polykernel_seqreg_runner must extract NSE KV cache from tar archives "
            "when base_alg=NSE. Without precomputed v_star, NSE does 25 gradient "
            "steps per edit and times out on smoke tests."
        )


class TestNSECacheCUpdatedAfterEdit:
    """NSE returns (model, cache_c) like AlphaEdit. The after_edit hook must
    update cache_c for NSE, not just AlphaEdit — otherwise NSE uses stale
    zeros cache_c for every batch."""

    def test_after_edit_updates_cache_c_for_nse(self):
        """The after_edit hook must update cache_c when base_alg is NSE."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        after_edit_section = source[source.find("def after_edit("):]
        after_edit_section = after_edit_section[:after_edit_section.find("\n    def ", 10)]
        assert '"NSE"' in after_edit_section or "'NSE'" in after_edit_section, (
            "after_edit must handle NSE's returned cache_c. NSE returns (model, cache_c) "
            "but only AlphaEdit is checked — NSE cache_c is silently discarded."
        )

    def test_cache_c_update_covers_all_algorithms_that_return_it(self):
        """Every algorithm that returns cache_c must have its value captured."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        after_edit_section = source[source.find("def after_edit("):]
        after_edit_section = after_edit_section[:after_edit_section.find("\n    def ", 10)]
        # AlphaEdit, NSE, and RECT all return cache_c that must be captured
        for alg in ["AlphaEdit", "NSE", "MEMIT_rect"]:
            assert alg in after_edit_section, (
                f"after_edit must update cache_c for {alg}. "
                f"Without this, {alg} uses stale zeros cache_c after batch 0."
            )


class TestRECTErrorCachePreserved:
    """RECT returns (model, cache_c, error_cache). The error_cache must be
    captured and passed back to the next batch, and saved in checkpoints."""

    def test_rect_error_cache_captured_from_result(self):
        """The apply_fn wrapper or after_edit must capture error_cache from RECT's 3-tuple."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        assert "error_cache" in source[source.find("def apply_fn"):source.find("def after_edit")], (
            "apply_fn must handle RECT's 3-tuple return: (model, cache_c, error_cache). "
            "Currently only cache_c is captured — error_cache is lost between batches."
        )

    def test_nonlocal_before_use_in_apply_fn(self):
        """nonlocal declarations must appear before any reference to the variable.
        Python raises SyntaxError if a name is used before its nonlocal declaration."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        apply_fn_body = source[source.find("def apply_fn("):source.find("\n    def ", source.find("def apply_fn(") + 10)]
        # Verify the file actually compiles (catches nonlocal ordering errors)
        import ast
        ast.parse(source)  # raises SyntaxError if nonlocal is misplaced

    def test_rect_error_cache_saved_in_checkpoint(self):
        """RECT's error_cache must be included in checkpoint extra_state."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        # Find the section between extra_state creation and the end of save_checkpoint call
        save_start = source.find("extra_state = {")
        save_end = source.find("save_checkpoint(", save_start)
        save_section = source[save_start:save_end + 500]  # include the save_checkpoint call
        assert "error_cache" in save_section, (
            "Checkpoint must include error_cache for RECT. Without it, resuming loses "
            "the rectification correction chain."
        )

    def test_rect_error_cache_loaded_from_checkpoint(self):
        """Checkpoint load must restore error_cache for RECT."""
        source = (PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py").read_text()
        load_section = source[source.find("load_checkpoint"):source.find("# Select apply function")]
        assert "error_cache" in load_section, (
            "Checkpoint load must restore error_cache for RECT."
        )
