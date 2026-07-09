"""
Tests for source injection scripts (nullspace_tracker, cache_mitigation, order_sensitivity).

Validates that generated scripts are syntactically valid Python, contain
the correct anchors, seed settings, and injection points. No GPU required.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nullspace_tracker import build_tracker_script, PRE_EDIT_ANCHOR, POST_EDIT_ANCHOR
from cache_mitigation_runner import build_mitigation_script
from order_sensitivity_runner import build_order_script


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHAEDIT_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"


class TestNullspaceTrackerScript:
    """Tests for nullspace_tracker.py script generation."""

    def _build_default(self, **kwargs):
        defaults = dict(
            seed=42,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=2000,
            num_edits=100,
            downstream_eval_steps=5,
            conserve_memory=True,
            output_jsonl="/tmp/test_output.jsonl",
        )
        defaults.update(kwargs)
        return build_tracker_script(**defaults)

    def test_generates_valid_python(self):
        """Generated script should parse without syntax errors."""
        script = self._build_default()
        ast.parse(script)

    def test_contains_seed_setting(self):
        """Script should set the specified seed."""
        script = self._build_default(seed=137)
        assert "seed = 137" in script

    def test_contains_torch_deterministic(self):
        """Script should enable deterministic mode."""
        script = self._build_default()
        assert "torch.backends.cudnn.deterministic = True" in script
        assert "torch.use_deterministic_algorithms" in script

    def test_contains_output_path(self):
        """Script should reference the output JSONL path."""
        script = self._build_default(output_jsonl="/results/rank_trace.jsonl")
        assert "/results/rank_trace.jsonl" in script

    def test_contains_cuda_patch(self):
        """Script should patch the hardcoded CUDA line."""
        script = self._build_default()
        assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"' in script

    def test_contains_pre_edit_injection(self):
        """Script should inject pre-edit tracking code."""
        script = self._build_default()
        assert "NULLSPACE TRACKING: pre-edit" in script

    def test_contains_post_edit_injection(self):
        """Script should inject post-edit tracking code."""
        script = self._build_default()
        assert "NULLSPACE TRACKING: post-edit" in script

    def test_contains_tracking_functions(self):
        """Script should define _ns_track_pre_batch and _ns_track_post_batch."""
        script = self._build_default()
        assert "def _ns_track_pre_batch" in script
        assert "def _ns_track_post_batch" in script

    def test_conserve_memory_flag(self):
        """conserve_memory flag should appear in argv when True."""
        script_on = self._build_default(conserve_memory=True)
        assert "--conserve_memory" in script_on

        script_off = self._build_default(conserve_memory=False)
        assert "--conserve_memory" not in script_off

    def test_anchors_exist_in_upstream(self):
        """Source anchors should exist in the pinned evaluate.py."""
        if not ALPHAEDIT_ROOT.exists():
            pytest.skip("vendor/AlphaEdit not available")
        eval_source = (ALPHAEDIT_ROOT / "experiments" / "evaluate.py").read_text()
        assert PRE_EDIT_ANCHOR in eval_source
        assert POST_EDIT_ANCHOR in eval_source


class TestCacheMitigationScript:
    """Tests for cache_mitigation_runner.py script generation."""

    def _build_default(self, **kwargs):
        defaults = dict(
            seed=42,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=2000,
            num_edits=100,
            downstream_eval_steps=5,
            conserve_memory=True,
            strategy="svd_truncation",
            truncation_interval=5,
            retain_ratio=0.75,
            decay_factor=0.95,
            reset_interval=10,
            metadata_jsonl="/tmp/meta.jsonl",
        )
        defaults.update(kwargs)
        return build_mitigation_script(**defaults)

    def test_generates_valid_python(self):
        """Generated script should parse without syntax errors."""
        script = self._build_default()
        ast.parse(script)

    def test_contains_seed(self):
        """Script should set the correct seed."""
        script = self._build_default(seed=99)
        assert "seed = 99" in script

    def test_svd_truncation_strategy(self):
        """SVD truncation parameters should appear in script."""
        script = self._build_default(strategy="svd_truncation", truncation_interval=10, retain_ratio=0.5)
        assert '"svd_truncation"' in script
        assert "_truncation_interval = 10" in script
        assert "_retain_ratio = 0.5" in script

    def test_exponential_decay_strategy(self):
        """Exponential decay parameters should appear in script."""
        script = self._build_default(strategy="exponential_decay", decay_factor=0.9)
        assert '"exponential_decay"' in script
        assert "_decay_factor = 0.9" in script

    def test_periodic_reset_strategy(self):
        """Periodic reset parameters should appear in script."""
        script = self._build_default(strategy="periodic_reset", reset_interval=20)
        assert '"periodic_reset"' in script
        assert "_reset_interval = 20" in script

    def test_contains_mitigation_function(self):
        """Script should define _apply_mitigation function."""
        script = self._build_default()
        assert "def _apply_mitigation" in script

    def test_contains_mitigation_injection(self):
        """Script should inject mitigation code."""
        script = self._build_default()
        assert "CACHE MITIGATION: apply strategy" in script

    def test_contains_metadata_output(self):
        """Script should write metadata to specified path."""
        script = self._build_default(metadata_jsonl="/output/meta.jsonl")
        assert "/output/meta.jsonl" in script

    def test_all_strategies_produce_valid_python(self):
        """All three strategies should produce valid scripts."""
        for strategy in ["svd_truncation", "exponential_decay", "periodic_reset"]:
            script = self._build_default(strategy=strategy)
            ast.parse(script)


class TestOrderSensitivityScript:
    """Tests for order_sensitivity_runner.py script generation."""

    def _build_default(self, **kwargs):
        defaults = dict(
            seed=42,
            order_seed=0,
            cuda_device="0",
            alg_name="AlphaEdit",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            hparams_fname="Llama3-8B.json",
            ds_name="mcf",
            dataset_size_limit=2000,
            num_edits=100,
            downstream_eval_steps=5,
            conserve_memory=True,
            metadata_jsonl="/tmp/order_meta.jsonl",
        )
        defaults.update(kwargs)
        return build_order_script(**defaults)

    def test_generates_valid_python(self):
        """Generated script should parse without syntax errors."""
        script = self._build_default()
        ast.parse(script)

    def test_contains_model_seed(self):
        """Script should set the model seed."""
        script = self._build_default(seed=137)
        assert "seed = 137" in script

    def test_contains_order_seed(self):
        """Script should set the order seed."""
        script = self._build_default(order_seed=7)
        assert "_order_seed = 7" in script

    def test_contains_shuffle_injection(self):
        """Script should inject shuffle code."""
        script = self._build_default()
        assert "ORDER SENSITIVITY: shuffle dataset" in script

    def test_shuffle_uses_order_seed(self):
        """Shuffle should use the order_seed, not model seed."""
        script = self._build_default(seed=42, order_seed=5)
        # The order_seed should be set as a variable
        assert "_order_seed = 5" in script
        # The shuffle injection builds the Random call via str(_order_seed)
        assert "str(_order_seed)" in script

    def test_different_order_seeds_produce_different_scripts(self):
        """Different order seeds should produce different scripts."""
        script_0 = self._build_default(order_seed=0)
        script_1 = self._build_default(order_seed=1)
        assert script_0 != script_1

    def test_supports_memit_algorithm(self):
        """Should work with MEMIT as well as AlphaEdit."""
        script = self._build_default(alg_name="MEMIT")
        assert "--alg_name=MEMIT" in script

    def test_contains_metadata_output(self):
        """Script should write metadata to specified path."""
        script = self._build_default(metadata_jsonl="/output/order.jsonl")
        assert "/output/order.jsonl" in script

    def test_shuffles_ds_data_attribute(self):
        """Should shuffle ds.data (CounterFact's internal list)."""
        script = self._build_default()
        assert "ds.data" in script
