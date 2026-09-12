"""Vendor code patches — applied at cluster setup time by apply_all.py.

This is the CURRENT patching system. Each file is a single-concern,
idempotent patch that modifies vendor or baseline source files on disk
before any experiment runs.

Patches here:
  patch_kwargs.py          — Add **_kwargs to algorithm apply functions
  patch_canonical_name.py  — Normalize model.config._name_or_path
  patch_mega_batch_eval.py — Replace per-record eval with batched scoring
  patch_model_compat.py    — Model dtype and shape list patches
  patch_nan_guard.py       — Skip NaN-producing edits in compute_z
  patch_p_cache.py         — Cache null-space projection (avoids 45-min SVD)
  patch_glue_map.py        — Add Qwen/GPT-J to GLUE context length map
  patch_s3_checkpoint.py   — Lightweight checkpointing for S3 FUSE

The OLD patching system lives at src/util/source_patches.py. It has
orchestrator functions (patch_evaluate_file, patch_memit_file, etc.)
that are still called by non-migrated runners. New patches should go
here, not in source_patches.py.
"""
