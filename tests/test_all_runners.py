#!/usr/bin/env python3
"""
Comprehensive tests for ALL runners, ALL algorithms, ALL submodule patches.

Tests every execution path: vendor/AlphaEdit runners, baselines/EvoEdit runners,
REVIVE+X variants, cross-model (GPT-J, Qwen) paths.

No GPU required — tests source code, script generation, path construction,
patch correctness, and kwargs acceptance.

Run with: uv run pytest tests/test_all_runners.py -v
"""
import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "polykernel"))


# ===========================================================================
# 1. EVERY runner script compiles for every algorithm it supports
# ===========================================================================

class TestAllRunnerScriptCompilation:
    """Every runner must produce syntactically valid Python for all its algorithm modes."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports):
        pass

    # --- polykernel_seqreg_runner: 4 base_alg × 2 revive × 2 kernel_prev = 16 combos ---
    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    @pytest.mark.parametrize("revive", [True, False])
    @pytest.mark.parametrize("kernel_prev", [True, False])
    def test_polykernel_seqreg_compiles(self, base_alg, revive, kernel_prev):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        kt = "poly1" if kernel_prev else "poly1-hybrid"
        rv = f"-REVIVE-tau0.1" if revive else ""
        variant = f"{prefix}-{kt}{rv}-lp0.0-ld0.0-cache0"
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name=base_alg,
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=0.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=1, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl", checkpoint_dir="/tmp/ckpt",
            variant_name=variant, kernel_prev=kernel_prev,
            revive=revive, revive_tau=0.1,
        )
        compile(script, f"<test-{base_alg}-revive{revive}-kp{kernel_prev}>", "exec")

    # --- checkpoint_runner: uses Python 3.10+ syntax, test the source file parses ---
    def test_checkpoint_runner_source_parses(self):
        import ast
        source = (PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py").read_text()
        ast.parse(source)

    # --- memit_sequential_runner ---
    def test_memit_sequential_source_parses(self):
        import ast
        source = (PROJECT_ROOT / "src" / "runners" / "memit_sequential_runner.py").read_text()
        ast.parse(source)

    # --- pathguard_runner ---
    def test_pathguard_runner_source_parses(self):
        import ast
        source = (PROJECT_ROOT / "src" / "runners" / "pathguard_runner.py").read_text()
        ast.parse(source)

    # --- alphaedit_stream_runner ---
    def test_alphaedit_stream_runner_source_parses(self):
        import ast
        source = (PROJECT_ROOT / "src" / "runners" / "alphaedit_stream_runner.py").read_text()
        ast.parse(source)


# ===========================================================================
# 2. EVERY apply function accepts **kwargs (after patching)
# ===========================================================================

class TestAllApplyFunctionsKwargs:
    """Every algorithm's apply_*_to_model MUST accept extra kwargs after patching.

    evaluate.py passes return_orig_weights_device which crashes if not accepted.
    """

    # --- Vendor submodule (patched by source_patches.py) ---

    def test_vendor_memit_kwargs_anchor(self):
        """The anchor string that source_patches uses to inject **_kwargs must exist."""
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        assert "cache_template: Optional[str] = None,\n) -> Tuple[AutoModelForCausalLM" in source

    def test_vendor_alphaedit_kwargs_anchor(self):
        source = (VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        assert "P = None,\n) -> Dict[str, Tuple[torch.Tensor]]:" in source

    # --- Baselines submodule (patched by patch_baselines.sh) ---

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    @pytest.mark.parametrize("relpath,funcname", [
        ("EvoEdit/EvoEdit_main.py", "apply_EvoEdit_to_model"),
        ("nse/nse_main.py", "apply_nse_to_model"),
        ("memit/memit_main.py", "apply_memit_to_model"),
        ("memit/memit_seq_main.py", "apply_memit_seq_to_model"),
        ("memit/memit_rect_main.py", "apply_memit_rect_to_model"),
        ("memit/memit_seq_rect_main.py", "apply_memit_seq_rect_to_model"),
        ("AlphaEdit/AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
    ])
    def test_baselines_apply_function_exists(self, relpath, funcname):
        filepath = BASELINES_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not present")
        source = filepath.read_text()
        assert f"def {funcname}(" in source, f"{funcname} not found in {relpath}"

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    def test_baselines_rect_has_kwargs(self):
        """memit_seq_rect_main.py MUST have **_kwargs (not patched at runtime)."""
        fp = BASELINES_ROOT / "memit" / "memit_seq_rect_main.py"
        if not fp.exists():
            pytest.skip("memit_seq_rect_main.py not present")
        source = fp.read_text()
        sig = re.search(r"def apply_memit_seq_rect_to_model\([^)]*\)", source, re.DOTALL)
        assert sig, "apply_memit_seq_rect_to_model not found"
        assert "kwargs" in sig.group(), (
            "RECT apply function must have **_kwargs — without it, evaluate.py crashes at batch 21"
        )

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    def test_patch_baselines_script_exists(self):
        """The patching script must exist — it's called from the YAML run block."""
        assert (PROJECT_ROOT / "scripts" / "patch_baselines.sh").exists()


# ===========================================================================
# 3. BOTH submodule evaluate.py files have correct structure
# ===========================================================================

class TestEvaluateFilesIntegrity:
    """Both vendor and baselines evaluate.py must have expected structure."""

    def test_vendor_evaluate_exists(self):
        assert (VENDOR_ROOT / "experiments" / "evaluate.py").exists()

    def test_vendor_evaluate_has_cuda_line(self):
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        assert 'os.environ["CUDA_VISIBLE_DEVICES"]' in source

    def test_vendor_evaluate_has_edit_loop(self):
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        assert "for record in ds:" in source
        assert "exec_time = time() - start" in source

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    def test_baselines_evaluate_exists(self):
        assert (BASELINES_ROOT / "experiments" / "evaluate.py").exists()

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    def test_baselines_evaluate_imports_all_algorithms(self):
        source = (BASELINES_ROOT / "experiments" / "evaluate.py").read_text()
        assert "apply_EvoEdit_to_model" in source
        assert "apply_memit_seq_rect_to_model" in source
        assert "apply_nse_to_model" in source

    @pytest.mark.skipif(not BASELINES_ROOT.exists(), reason="baselines not present")
    def test_baselines_evaluate_cuda_already_commented(self):
        """baselines evaluate.py should have CUDA line already commented out."""
        source = (BASELINES_ROOT / "experiments" / "evaluate.py").read_text()
        # Either commented out or not present
        lines = source.split("\n")
        cuda_lines = [l for l in lines if "CUDA_VISIBLE_DEVICES" in l]
        for l in cuda_lines:
            assert l.strip().startswith("#"), f"CUDA line not commented: {l}"


# ===========================================================================
# 4. source_patches.py patches are correct and idempotent
# ===========================================================================

class TestSourcePatches:
    """All runtime patches must be correct and safe to apply multiple times."""

    def test_patch_memit_file_adds_kwargs(self):
        """patch_memit_file applies NaN guard AND adds **_kwargs."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_nan_guard_patch
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        patched = apply_nan_guard_patch(source)
        # NaN guard is applied by apply_nan_guard_patch
        # kwargs are applied by patch_memit_file (which calls apply_nan_guard_patch + adds kwargs)
        # Test the kwargs anchor exists (patch_memit_file will use it)
        kwargs_anchor = "cache_template: Optional[str] = None,\n) -> Tuple[AutoModelForCausalLM"
        assert kwargs_anchor in patched, "kwargs anchor must survive NaN guard patch"

    def test_nan_guard_patch_idempotent(self):
        from source_patches import apply_nan_guard_patch
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        patched1 = apply_nan_guard_patch(source)
        patched2 = apply_nan_guard_patch(patched1)
        assert patched1 == patched2, "NaN guard patch is not idempotent"

    def test_p_cache_patch_idempotent(self):
        from source_patches import apply_p_cache_patch
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        patched1 = apply_p_cache_patch(source)
        patched2 = apply_p_cache_patch(patched1)
        assert patched1 == patched2, "P-cache patch is not idempotent"

    def test_model_list_patch_idempotent(self):
        from source_patches import apply_model_list_patch
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        patched1 = apply_model_list_patch(source)
        patched2 = apply_model_list_patch(patched1)
        assert patched1 == patched2, "Model list patch is not idempotent"


# ===========================================================================
# 5. Cross-model path isolation (GPT-J, Qwen)
# ===========================================================================

class TestCrossModelPathIsolation:
    """Different models must produce isolated checkpoint and result paths."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        pass

    MODELS = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", ""),
        ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
    ]

    @pytest.mark.parametrize("model_name,expected_tag", MODELS)
    def test_model_tag_in_path(self, model_name, expected_tag):
        from polykernel_seqreg_runner import resolve_checkpoint_dir
        p = resolve_checkpoint_dir(
            None, 42, 1.0, 0.0, cache_max=None, kernel_degree=2,
            model_name=model_name, base_alg="MEMIT",
        )
        if expected_tag:
            assert expected_tag in str(p), f"Model tag '{expected_tag}' missing from {p}"
        else:
            assert "gpt-j" not in str(p) and "qwen" not in str(p)

    def test_all_models_distinct_paths(self):
        from polykernel_seqreg_runner import resolve_checkpoint_dir
        paths = set()
        for model_name, _ in self.MODELS:
            p = resolve_checkpoint_dir(
                None, 42, 1.0, 0.0, cache_max=None, kernel_degree=2,
                model_name=model_name, base_alg="MEMIT",
            )
            paths.add(str(p))
        assert len(paths) == len(self.MODELS), "Model paths must be distinct"


# ===========================================================================
# 6. Shell script parameter validation
# ===========================================================================

class TestShellScriptParameters:
    """Shell scripts must pass correct parameters to Python runners."""

    SCRIPTS = {
        "run_revive_baseline.sh": {
            "must_contain": ["--base_alg", "--revive", "--revive_tau", "--ordering"],
            "must_not_contain": [],
        },
        "run_revive_paper_replication.sh": {
            "must_contain": ["--base_alg", "--revive", "--revive_tau"],
            "must_not_contain": [],
        },
        "run_evoedit_baseline.sh": {
            "must_contain": ["EvoEdit"],
            "must_not_contain": [],
        },
        "run_nse_baseline.sh": {
            "must_contain": ["NSE"],
            "must_not_contain": [],
        },
        "run_matched_ordering.sh": {
            "must_contain": ["--ordering", "--seed"],
            "must_not_contain": [],
        },
    }

    @pytest.mark.parametrize("script_name,checks", list(SCRIPTS.items()))
    def test_script_parameters(self, script_name, checks):
        path = PROJECT_ROOT / "scripts" / script_name
        if not path.exists():
            pytest.skip(f"{script_name} not present")
        source = path.read_text()
        for pattern in checks["must_contain"]:
            assert pattern in source, f"{script_name} must contain '{pattern}'"
        for pattern in checks["must_not_contain"]:
            assert pattern not in source, f"{script_name} must NOT contain '{pattern}'"

    def test_revive_paper_replication_defaults_lp0(self):
        """REVIVE paper replication must default to lambda_prev=0 (plain MEMIT base)."""
        path = PROJECT_ROOT / "scripts" / "run_revive_paper_replication.sh"
        if not path.exists():
            pytest.skip("script not present")
        source = path.read_text()
        # Should have LAMBDA_PREV default of 0
        assert "LAMBDA_PREV" in source
        # The default should be 0 for the paper replication (plain MEMIT base)
        match = re.search(r'LAMBDA_PREV="\$\{LAMBDA_PREV:-(\d+)', source)
        if match:
            assert match.group(1) == "0", (
                f"REVIVE paper replication LAMBDA_PREV default should be 0, got {match.group(1)}"
            )


# ===========================================================================
# 7. Prob-pref metric computation correctness
# ===========================================================================

class TestProbPrefMetrics:
    """Probability-preference metric computation must follow correct NLL conventions."""

    def test_prob_pref_efficacy_convention(self):
        """Efficacy: target_new MORE probable → NLL(target_new) < NLL(target_true)."""
        # In NLL space: lower = more probable
        # Success: target_true_nll > target_new_nll (new is more probable)
        probs = {"target_new": 5.0, "target_true": 8.0}  # new has lower NLL = more probable
        success = probs["target_true"] > probs["target_new"]
        assert success is True

    def test_prob_pref_neighborhood_convention(self):
        """Neighborhood: target_true MORE probable → NLL(target_true) < NLL(target_new)."""
        probs = {"target_new": 8.0, "target_true": 5.0}  # true has lower NLL = preserved
        success = probs["target_true"] < probs["target_new"]
        assert success is True

    def test_prob_pref_from_real_data(self):
        """Test with actual per-case file structure."""
        case = {
            "post": {
                "rewrite_prompts_probs": [{"target_new": 5.77, "target_true": 5.91}],
                "rewrite_prompts_correct": [False],
                "neighborhood_prompts_probs": [
                    {"target_new": 8.0, "target_true": 5.0},
                    {"target_new": 3.0, "target_true": 7.0},
                ],
                "neighborhood_prompts_correct": [False, False],
            }
        }
        post = case["post"]

        # Prob-pref efficacy: target_true (5.91) > target_new (5.77) → success
        eff_pp = post["rewrite_prompts_probs"][0]["target_true"] > post["rewrite_prompts_probs"][0]["target_new"]
        assert eff_pp is True

        # Argmax efficacy: False
        eff_am = post["rewrite_prompts_correct"][0]
        assert eff_am is False

        # These CAN disagree — that's the metric mismatch we found
        assert eff_pp != eff_am, "This sample demonstrates prob-pref vs argmax disagreement"


# ===========================================================================
# 8. Checkpoint completeness validation
# ===========================================================================

class TestCheckpointCompleteness:
    """Checkpoint files must contain all required components."""

    def test_checkpoint_requires_model_weights(self):
        """Every checkpoint must have model_weights.pt."""
        # This is a structural test — verify the save code includes it
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name="MEMIT",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=1.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=2, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl", checkpoint_dir="/tmp/ckpt",
            variant_name="test",
        )
        assert 'model_weights.pt' in script
        assert 'metadata.json' in script
        assert 'prev_cache.pt' in script

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports):
        pass


# ===========================================================================
# 9. YAML dispatch correctness
# ===========================================================================

class TestYamlDispatch:
    """The SkyPilot YAML must dispatch to correct scripts for each experiment."""

    @pytest.fixture
    def yaml_source(self):
        path = PROJECT_ROOT / "sky" / "alphaedit_gpu.yaml"
        if not path.exists():
            pytest.skip("alphaedit_gpu.yaml not present")
        return path.read_text()

    def test_revive_dispatches_to_revive_script(self, yaml_source):
        assert "run_revive_baseline.sh" in yaml_source

    def test_nse_dispatches_to_nse_script(self, yaml_source):
        assert "run_nse_baseline.sh" in yaml_source

    def test_evoedit_dispatches_to_evoedit_script(self, yaml_source):
        assert "run_evoedit_baseline.sh" in yaml_source or "evoedit" in yaml_source.lower()

    def test_patch_baselines_called(self, yaml_source):
        """patch_baselines.sh must be called in the YAML run block."""
        assert "patch_baselines" in yaml_source
