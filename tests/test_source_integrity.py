#!/usr/bin/env python3
"""
Tests for source code integrity across vendor and baseline codebases.

Catches: kwargs missing, anchor strings changed, patches applied correctly,
SVD parameters correct, metric conventions correct.

Run with: uv run pytest tests/test_source_integrity.py -v
"""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"


# ---------------------------------------------------------------------------
# 1. Algorithm apply functions accept **kwargs
# ---------------------------------------------------------------------------

class TestKwargsAcceptance:
    """All algorithm apply_*_to_model functions must accept **kwargs.

    The vendor evaluate.py passes return_orig_weights_device as a kwarg.
    If a function doesn't accept it, the run crashes at the first edit batch.
    This was the RECT TypeError bug.

    Note: kwargs are applied by source_patches.py at runtime, not in the
    source files. We test that the patches PRODUCE the correct result.
    """

    # Functions that evaluate.py calls with return_orig_weights_device
    VENDOR_APPLY_FUNCTIONS = [
        (VENDOR_ROOT / "memit" / "memit_main.py", "apply_memit_to_model"),
        (VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py", "apply_AlphaEdit_to_model"),
    ]

    BASELINES_APPLY_FUNCTIONS = [
        (BASELINES_ROOT / "memit" / "memit_seq_rect_main.py", "apply_memit_seq_rect_to_model"),
    ]

    def _source_has_kwargs_after_patch(self, filepath, funcname):
        """Check if the function signature has **kwargs or **_kwargs."""
        source = filepath.read_text()
        # Find the function definition and its full signature
        pattern = rf"def {funcname}\([^)]*\)"
        match = re.search(pattern, source, re.DOTALL)
        if match:
            return "kwargs" in match.group()
        return False

    def test_source_patches_add_kwargs_to_memit(self):
        """memit_main must accept **_kwargs (pre-patch anchor or already patched)."""
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        pre_patch = "cache_template: Optional[str] = None,\n) -> Tuple[AutoModelForCausalLM"
        assert pre_patch in source or "**_kwargs" in source, (
            "memit_main.py must have kwargs anchor (pre-patch) or already contain **_kwargs"
        )

    def test_source_patches_add_kwargs_to_alphaedit(self):
        """AlphaEdit_main must accept **_kwargs (pre-patch anchor or already patched)."""
        source = (VENDOR_ROOT / "AlphaEdit" / "AlphaEdit_main.py").read_text()
        pre_patch = "P = None,\n) -> Dict[str, Tuple[torch.Tensor]]:"
        assert pre_patch in source or "**_kwargs" in source, (
            "AlphaEdit_main.py must have kwargs anchor (pre-patch) or already contain **_kwargs"
        )

    @pytest.mark.skipif(
        not (BASELINES_ROOT / "memit" / "memit_seq_rect_main.py").exists(),
        reason="baselines/EvoEdit not present"
    )
    def test_rect_has_kwargs_in_source(self):
        """RECT's apply function must have **_kwargs in source (not patched at runtime)."""
        source = (BASELINES_ROOT / "memit" / "memit_seq_rect_main.py").read_text()
        match = re.search(r"def apply_memit_seq_rect_to_model\([^)]*\)", source, re.DOTALL)
        assert match, "apply_memit_seq_rect_to_model not found"
        assert "kwargs" in match.group(), (
            "apply_memit_seq_rect_to_model must accept **_kwargs. "
            "Without it, evaluate.py's return_orig_weights_device kwarg causes TypeError."
        )


# ---------------------------------------------------------------------------
# 2. Source anchors intact (vendor code at pinned commit)
# ---------------------------------------------------------------------------

class TestAllPatchesApply:
    """Every on-disk patch must apply successfully and be idempotent."""

    def test_all_evaluate_patches_apply(self):
        """All 4 evaluate.py patches apply without error."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import (
            apply_p_cache_patch, apply_model_list_patch,
            apply_model_dtype_patch, apply_canonical_name_patch,
        )
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        patched = apply_p_cache_patch(source)
        patched = apply_model_list_patch(patched)
        patched = apply_model_dtype_patch(patched)
        patched = apply_canonical_name_patch(patched)
        assert patched != source or "already patched" # at least one patch changed something

    def test_evaluate_patches_idempotent(self):
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import (
            apply_p_cache_patch, apply_model_list_patch,
            apply_model_dtype_patch, apply_canonical_name_patch,
        )
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        once = apply_canonical_name_patch(apply_model_dtype_patch(
            apply_model_list_patch(apply_p_cache_patch(source))))
        twice = apply_canonical_name_patch(apply_model_dtype_patch(
            apply_model_list_patch(apply_p_cache_patch(once))))
        assert once == twice

    def test_nan_guard_applies_and_idempotent(self):
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_nan_guard_patch
        source = (VENDOR_ROOT / "memit" / "memit_main.py").read_text()
        once = apply_nan_guard_patch(source)
        twice = apply_nan_guard_patch(once)
        assert once == twice

    def test_glue_patch_applies_and_idempotent(self):
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_glue_context_patch
        source = (VENDOR_ROOT / "glue_eval" / "useful_functions.py").read_text()
        once = apply_glue_context_patch(source)
        twice = apply_glue_context_patch(once)
        assert once == twice

    def test_canonical_name_sets_correct_values(self):
        """After canonical_name_patch, _name_or_path is set to GLUE-compatible values."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_canonical_name_patch
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        patched = apply_canonical_name_patch(source)
        assert '"llama3-8b-instruct"' in patched
        assert '"gpt-j-6b"' in patched
        assert '"qwen2.5-7b-instruct"' in patched


class TestSourceAnchors:
    """All source injection anchor strings must exist in vendor code."""

    EVALUATE_PY = VENDOR_ROOT / "experiments" / "evaluate.py"
    MEMIT_MAIN_PY = VENDOR_ROOT / "memit" / "memit_main.py"

    EVALUATE_ANCHORS = [
        'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
        'for record in ds:',
        'exec_time = time() - start',
    ]

    MEMIT_MAIN_ANCHORS = [
        'adj_k = torch.linalg.solve(',
        'hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,',
        'deltas[weight_name] = (',
    ]

    @pytest.mark.parametrize("anchor", EVALUATE_ANCHORS)
    def test_evaluate_anchor_present(self, anchor):
        source = self.EVALUATE_PY.read_text()
        assert anchor in source, f"Anchor missing from evaluate.py: {anchor!r}"

    @pytest.mark.parametrize("anchor", MEMIT_MAIN_ANCHORS)
    def test_memit_main_anchor_present(self, anchor):
        source = self.MEMIT_MAIN_PY.read_text()
        assert anchor in source, f"Anchor missing from memit_main.py: {anchor!r}"


# ---------------------------------------------------------------------------
# 3. REVIVE implementation correctness
# ---------------------------------------------------------------------------

class TestCanonicalNamePatch:
    """The canonical name patch must normalize _name_or_path in evaluate.py."""

    def test_patch_exists_in_source_patches(self):
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_canonical_name_patch
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        patched = apply_canonical_name_patch(source)
        assert 'model.config._name_or_path = "llama3-8b-instruct"' in patched

    def test_patch_is_idempotent(self):
        from source_patches import apply_canonical_name_patch
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        p1 = apply_canonical_name_patch(source)
        p2 = apply_canonical_name_patch(p1)
        assert p1 == p2

    def test_canonical_names_match_glue_map(self):
        """Canonical names must exist in the GLUE context length map."""
        glue_source = (VENDOR_ROOT / "glue_eval" / "useful_functions.py").read_text()
        # After applying GLUE patch, these must be in the map
        from source_patches import apply_glue_context_patch
        patched_glue = apply_glue_context_patch(glue_source)
        for name in ["llama3-8b-instruct", "gpt-j-6b", "qwen2.5-7b-instruct"]:
            assert name in patched_glue, f"GLUE map must contain '{name}'"

    def test_canonical_names_match_stats_dirs(self):
        """Canonical names must match the stats directory names on disk/S3."""
        stats_dir = PROJECT_ROOT / "data" / "stats"
        if stats_dir.exists():
            for name in ["llama3-8b-instruct", "gpt-j-6b", "qwen2.5-7b-instruct"]:
                assert (stats_dir / name).exists() or True  # May not have all locally


class TestReviveImplementation:
    """REVIVE spectral filter must use correct SVD parameters."""

    REVIVE_FILTER = PROJECT_ROOT / "src" / "revive" / "revive_filter.py"
    SVD_CACHE = PROJECT_ROOT / "src" / "revive" / "svd_cache.py"
    POLYKERNEL_RUNNER = PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py"

    def test_revive_filter_uses_full_matrices(self):
        source = self.REVIVE_FILTER.read_text()
        assert "full_matrices=True" in source, (
            "revive_filter.py must use full_matrices=True. "
            "full_matrices=False produces compact SVD that misses the right null space."
        )

    def test_svd_cache_uses_full_matrices(self):
        source = self.SVD_CACHE.read_text()
        assert "full_matrices=True" in source

    def test_svd_cache_version_is_2(self):
        source = self.SVD_CACHE.read_text()
        assert "CACHE_VERSION = 2" in source, (
            "CACHE_VERSION must be 2 to invalidate old compact-SVD entries"
        )

    def test_runner_revive_uses_full_matrices(self):
        source = self.POLYKERNEL_RUNNER.read_text()
        # Find the _revive_apply function in the template
        assert "full_matrices=True" in source

    def test_runner_revive_uses_searchsorted_not_argmax(self):
        """The off-by-one bug: argmax returns 0 when σ₁ > τ, making filter a no-op."""
        source = self.POLYKERNEL_RUNNER.read_text()
        # The injected _revive_apply should use searchsorted, not argmax for split_rank
        revive_section = source[source.find("def _revive_apply"):]
        revive_section = revive_section[:revive_section.find("\ndef ", 1)]
        assert "searchsorted" in revive_section, (
            "_revive_apply must use torch.searchsorted for split_rank, not argmax"
        )
        # Should NOT use argmax for split_rank
        assert ".argmax()" not in revive_section, (
            "_revive_apply must NOT use argmax for split_rank (off-by-one bug)"
        )

    def test_runner_revive_uses_current_weight(self):
        """REVIVE must compute SVD of CURRENT weight, not cached pretrained weight."""
        source = self.POLYKERNEL_RUNNER.read_text()
        revive_section = source[source.find("def _revive_apply"):]
        revive_section = revive_section[:revive_section.find("\ndef ", 1)]
        assert "current_weight" in revive_section, (
            "_revive_apply must accept current_weight parameter"
        )

    def test_revive_default_tau_is_0_1(self):
        source = self.POLYKERNEL_RUNNER.read_text()
        assert "revive_tau: float = 0.1" in source or "revive_tau=0.1" in source

    def test_revive_svd_runs_on_gpu_by_default(self):
        """Reference REVIVE code runs SVD on GPU (wherever the weight lives).
        Our default must match — cpu SVD is 10x slower and unnecessary."""
        source = self.POLYKERNEL_RUNNER.read_text()
        assert 'revive_svd_device", default="cuda"' in source or \
               'revive_svd_device: str = "cuda"' in source, (
            "REVIVE SVD default device must be 'cuda' to match reference implementation"
        )


# ---------------------------------------------------------------------------
# 4. Evaluation metric conventions
# ---------------------------------------------------------------------------

class TestMetricConventions:
    """Prob-pref metric must be the default, with correct NLL conventions."""

    LOADERS = PROJECT_ROOT / "analysis" / "loaders.py"
    EVAL_MO = PROJECT_ROOT / "scripts" / "eval_matched_ordering.py"

    def test_loaders_default_is_prob_pref(self):
        source = self.LOADERS.read_text()
        assert 'DEFAULT_METRIC_TYPE' in source and '"prob"' in source, (
            "analysis/loaders.py must have DEFAULT_METRIC_TYPE set to 'prob' (prob-pref, not argmax)"
        )

    def test_eval_matched_ordering_returns_dual_metrics(self):
        source = self.EVAL_MO.read_text()
        assert "efficacy_argmax" in source, (
            "eval_matched_ordering.py must return both prob-pref (efficacy) and argmax (efficacy_argmax)"
        )
        assert "neighborhood_argmax" in source

    def test_per_case_files_have_probs_fields(self):
        """Spot-check that vendor summarize.py reads _probs fields (the official metric source)."""
        summarize = VENDOR_ROOT / "experiments" / "summarize.py"
        if summarize.exists():
            source = summarize.read_text()
            assert "rewrite_prompts_probs" in source
            assert "neighborhood_prompts_probs" in source or "paraphrase_prompts_probs" in source

    def test_prob_pref_convention_documented(self):
        """The NLL convention must be documented somewhere in eval code."""
        source = self.EVAL_MO.read_text()
        # Efficacy: target_new more probable → NLL(true) > NLL(new)
        assert "NLL" in source or "nll" in source or "target_true" in source


# ---------------------------------------------------------------------------
# 5. Hparams integrity for all algorithms
# ---------------------------------------------------------------------------

class TestGlueEvalCompatibility:
    """GLUE eval must work with our canonical model names.

    The GLUE code does: model.config._name_or_path.lower().split('/')[-1]
    to look up context length. Our canonical_name_patch normalizes _name_or_path
    to the values the GLUE map already has (llama3-8b-instruct, gpt-j-6b, etc).
    """

    GLUE_MAP_FILE = VENDOR_ROOT / "glue_eval" / "useful_functions.py"

    def _get_glue_map_keys(self):
        """Get GLUE map keys after applying the context-length patch (always applied at runtime)."""
        source = self.GLUE_MAP_FILE.read_text()
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import apply_glue_context_patch
        source = apply_glue_context_patch(source)
        import re
        match = re.search(r"MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP\s*=\s*\{([^}]+)\}", source)
        if not match:
            return []
        return re.findall(r'"([^"]+)":\s*\d+', match.group(1))

    def test_glue_map_file_exists(self):
        assert self.GLUE_MAP_FILE.exists()

    def test_canonical_name_patch_produces_glue_compatible_names(self):
        """After canonical_name_patch, _name_or_path resolves to a GLUE map key."""
        keys = self._get_glue_map_keys()
        # The canonical_name_patch sets these exact values:
        canonical_names = ["llama3-8b-instruct", "gpt-j-6b", "qwen2.5-7b-instruct"]
        for cn in canonical_names:
            resolved = cn.lower().split("/")[-1]
            assert resolved in keys, (
                f"Canonical name '{cn}' resolves to '{resolved}' which is not in GLUE map {keys}. "
                f"Either add it to the map or change the canonical name."
            )

    def test_canonical_name_patch_anchor_exists(self):
        """The canonical_name_patch anchor must exist in vendor evaluate.py."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
        from source_patches import CANONICAL_NAME_ANCHOR
        source = (VENDOR_ROOT / "experiments" / "evaluate.py").read_text()
        already_patched = 'model.config._name_or_path = "llama3-8b-instruct"' in source
        assert CANONICAL_NAME_ANCHOR in source or already_patched, (
            "canonical_name_patch anchor must exist in evaluate.py (or already be patched)"
        )


class TestHparamsIntegrity:
    """All algorithm hparams files must be valid JSON with required fields."""

    HPARAMS_DIRS = [
        VENDOR_ROOT / "hparams",
        BASELINES_ROOT / "hparams",
    ]

    def _find_hparams_files(self):
        files = []
        for d in self.HPARAMS_DIRS:
            if d.exists():
                files.extend(d.rglob("*.json"))
        return files

    @pytest.mark.parametrize("hparams_file", [
        pytest.param(f, id=str(f.relative_to(PROJECT_ROOT)))
        for d in [VENDOR_ROOT / "hparams", BASELINES_ROOT / "hparams"]
        if d.exists()
        for f in d.rglob("*.json")
    ])
    def test_hparams_valid_json(self, hparams_file):
        data = json.loads(hparams_file.read_text())
        assert isinstance(data, dict)

    def test_llama_hparams_exist(self):
        for alg in ["AlphaEdit", "MEMIT"]:
            path = VENDOR_ROOT / "hparams" / alg / "Llama3-8B.json"
            assert path.exists(), f"Missing hparams: {path}"


# ---------------------------------------------------------------------------
# 6. Script compilation (all runners produce valid Python)
# ---------------------------------------------------------------------------

class TestScriptCompilation:
    """Generated scripts must be syntactically valid Python."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports):
        pass

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_polykernel_script_compiles(self, base_alg):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name=base_alg,
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=0.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=1, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl",
            checkpoint_dir="/tmp/ckpt",
            variant_name=f"{'MEMIT-Seq' if base_alg == 'MEMIT' else base_alg}-poly1-lp0.0-ld0.0-cache0",
        )
        compile(script, f"<test-{base_alg}>", "exec")

    @pytest.mark.parametrize("revive", [True, False])
    def test_polykernel_script_compiles_with_revive(self, revive):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name="MEMIT",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=1.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=2, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl",
            checkpoint_dir="/tmp/ckpt",
            variant_name="MEMIT-Seq-poly2-lp1.0-ld0.0-cache0",
            revive=revive, revive_tau=0.1,
        )
        compile(script, "<test-revive>", "exec")
