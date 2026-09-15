"""Tests for checkpoint-clobber bug in polykernel_seqreg_runner.py.

Bug: The runner re-initializes cache_c/error_cache to None AFTER loading them
from a checkpoint, discarding the loaded values. Additionally, MEMIT_rect is
excluded from the checkpoint load conditional for cache_c.pt.

These tests read the runner source and verify structural invariants about
the ordering of initialization vs checkpoint loading.
"""
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py"


def _read_runner():
    return RUNNER_PATH.read_text()


class TestCheckpointClobberBug:
    """Detect the bug where cache_c/error_cache init clobbers checkpoint-loaded values."""

    def test_cache_c_init_not_after_checkpoint_load(self):
        """cache_c must not be reset to None or zeros AFTER load_checkpoint runs.

        The init block (cache_c = None, cache_c = torch.zeros(...)) must appear
        BEFORE the checkpoint load block (if start_from_batch > 0: ... load_checkpoint),
        so that loaded values overwrite defaults, not the other way around.
        """
        source = _read_runner()

        load_pos = source.find("load_checkpoint(")
        assert load_pos > 0, "load_checkpoint call not found"

        # Find all lines that reset cache_c to None or zeros after the load
        # The problematic pattern: "cache_c = None" appearing after load_checkpoint
        after_load = source[load_pos:]
        reinit_lines = [
            m.start() + load_pos
            for m in re.finditer(r"^\s+cache_c = None\b", after_load, re.MULTILINE)
        ]

        assert len(reinit_lines) == 0, (
            f"cache_c is reset to None at {len(reinit_lines)} location(s) AFTER "
            f"load_checkpoint. Loaded checkpoint values will be discarded."
        )

    def test_error_cache_init_not_after_checkpoint_load(self):
        """error_cache must not be reset to None or zeros AFTER load_checkpoint runs."""
        source = _read_runner()

        load_pos = source.find("load_checkpoint(")
        assert load_pos > 0, "load_checkpoint call not found"

        after_load = source[load_pos:]
        reinit_lines = [
            m.start() + load_pos
            for m in re.finditer(r"^\s+error_cache = None\b", after_load, re.MULTILINE)
        ]

        assert len(reinit_lines) == 0, (
            f"error_cache is reset to None at {len(reinit_lines)} location(s) AFTER "
            f"load_checkpoint. Loaded checkpoint values will be discarded."
        )

    def test_memit_rect_in_checkpoint_cache_c_load(self):
        """MEMIT_rect must be included in the extra_keys conditional for cache_c.pt.

        Currently only ("AlphaEdit", "MEMIT") load cache_c.pt from checkpoints,
        but MEMIT_rect also uses cache_c and must restore it on resume.
        """
        source = _read_runner()

        # Find the block that decides whether to add cache_c.pt to extra_keys
        # Pattern: if args.base_alg in (...): extra_keys.append("cache_c.pt")
        match = re.search(
            r'if args\.base_alg in \(([^)]+)\):\s*\n\s*extra_keys\.append\("cache_c\.pt"\)',
            source,
        )
        assert match is not None, "cache_c.pt extra_keys conditional not found"

        alg_list = match.group(1)
        assert "MEMIT_rect" in alg_list, (
            f"MEMIT_rect is missing from the cache_c.pt checkpoint load conditional. "
            f"Found: if args.base_alg in ({alg_list})"
        )

    def test_memit_rect_loads_cache_c_from_checkpoint(self):
        """The cache_c.pt load conditional must cover all algorithms that use cache_c.

        MEMIT_rect uses cache_c in its solve (LHS includes cache_c[i]).
        Without restoring it from checkpoint, resumed runs start with zeros,
        producing different results than continuous runs.
        """
        source = _read_runner()

        # Find lines near cache_c.pt that list which algorithms get it
        cache_c_region = source[
            max(0, source.find('"cache_c.pt"') - 200):
            source.find('"cache_c.pt"') + 50
        ]
        assert '"MEMIT_rect"' in cache_c_region or "'MEMIT_rect'" in cache_c_region, (
            "MEMIT_rect not found near cache_c.pt in checkpoint load logic"
        )


class TestCheckpointLoadBlockSanity:
    """Regression guards — these should pass now and continue passing after the fix."""

    def test_checkpoint_load_block_exists(self):
        """The runner must have checkpoint load logic."""
        source = _read_runner()
        assert "load_checkpoint(" in source, "load_checkpoint call not found"
        assert "start_from_batch > 0" in source, "start_from_batch guard not found"

    def test_cache_c_assigned_from_checkpoint(self):
        """When cache_c.pt is loaded, it must be assigned to cache_c."""
        source = _read_runner()
        assert 'cache_c = ckpt_result["cache_c.pt"]' in source, (
            "cache_c is not assigned from checkpoint result"
        )

    def test_error_cache_assigned_from_checkpoint(self):
        """When error_cache.pt is loaded, it must be assigned to error_cache."""
        source = _read_runner()
        assert 'error_cache = ckpt_result["error_cache.pt"]' in source, (
            "error_cache is not assigned from checkpoint result"
        )

    def test_error_cache_pt_loaded_for_memit_rect(self):
        """MEMIT_rect must load error_cache.pt from checkpoint."""
        source = _read_runner()
        assert '"error_cache.pt"' in source, "error_cache.pt not referenced in runner"
        # Verify it's in an extra_keys append that covers MEMIT_rect
        assert re.search(
            r'if args\.base_alg in.*"MEMIT_rect".*:\s*\n\s*extra_keys\.append\("error_cache\.pt"\)',
            source,
        ) or re.search(
            r'if args\.base_alg == "MEMIT_rect":\s*\n\s*extra_keys\.append\("error_cache\.pt"\)',
            source,
        ), "error_cache.pt is not loaded for MEMIT_rect in the checkpoint block"
