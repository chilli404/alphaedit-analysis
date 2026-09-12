"""Algorithm implementations and composable hooks for knowledge editing.

Canonical algorithm names used throughout the codebase:

  Display name  | Vendor name  | Python module           | Results path prefix
  ------------- | ------------ | ----------------------- | -------------------
  AlphaEdit     | AlphaEdit    | -                       | AlphaEdit
  MEMIT         | MEMIT        | -                       | MEMIT
  MEMIT-Seq     | MEMIT_seq    | memit_sequential_runner  | MEMIT-Seq
  EvoEdit       | EvoEdit      | -                       | EvoEdit
  NSE           | NSE          | -                       | NSE
  RECT-Aligned  | MEMIT_seq_rect | -                     | MEMIT_seq_rect
  PathGuard     | -            | pathguard_runner         | PathGuard-ED

The vendor code uses underscores (MEMIT_seq); we use hyphens (MEMIT-Seq) in
result paths and display. ExperimentConfig.base_prefix handles the mapping.
"""
