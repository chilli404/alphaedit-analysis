"""Standardized logging for experiment runners.

All runners should use these functions for progress output. The formats
are designed to be grep-friendly for sky logs monitoring and the GPU
smoke test output scanner.

Prefixes used:
  [Batch N/M]   — batch start/complete
  [CHECKPOINT]  — checkpoint saved (smoke test greps for this)
  [EVAL]        — evaluation progress
  [MEGA-BATCH EVAL] — mega-batch eval progress (existing convention)

For runners that use exec(compile()), these functions can't be imported
directly into the exec'd namespace. Use INJECTABLE_SOURCE to inject them
as a string template, or pass them via the exec namespace dict.
"""


def log_batch_start(batch_idx: int, num_edits: int, total_batches: int):
    """Print standardized batch start line."""
    total_edits = (batch_idx + 1) * num_edits
    print(
        f"[Batch {batch_idx + 1}/{total_batches}] "
        f"Editing {num_edits} records ({total_edits} total)",
        flush=True,
    )


def log_batch_complete(batch_idx: int, num_edits: int, exec_time: float):
    """Print standardized batch completion line."""
    total_edits = (batch_idx + 1) * num_edits
    print(
        f"[Batch {batch_idx + 1}] "
        f"Complete: {total_edits} edits in {exec_time:.1f}s",
        flush=True,
    )


def log_checkpoint_saved(batch_idx: int, num_edits: int, path: str):
    """Print standardized checkpoint save line."""
    print(
        f"[CHECKPOINT] Saved batch {batch_idx} "
        f"({(batch_idx + 1) * num_edits} edits) -> {path}",
        flush=True,
    )


def log_eval_progress(done: int, total: int, rate: float, elapsed: float):
    """Print standardized eval progress line."""
    print(
        f"[EVAL] {done}/{total} records "
        f"({rate:.1f} rec/s, {elapsed:.0f}s)",
        flush=True,
    )


def log_run_summary(
    alg_name: str,
    seed: int,
    batches_run: int,
    total_edits: int,
    checkpoint_dir: str,
    elapsed: float,
):
    """Print standardized run summary."""
    print(f"\n=== Run complete ===")
    print(f"  Algorithm:    {alg_name}")
    print(f"  Seed:         {seed}")
    print(f"  Batches:      {batches_run}")
    print(f"  Total edits:  {total_edits}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(flush=True)


# Injectable source for exec(compile()) runners.
# Paste this into the template string so the functions are available
# inside the exec'd evaluate.py namespace.
INJECTABLE_SOURCE = '''
def _log_batch_start(batch_idx, num_edits, total_batches):
    total_edits = (batch_idx + 1) * num_edits
    print(f"[Batch {batch_idx + 1}/{total_batches}] Editing {num_edits} records ({total_edits} total)", flush=True)

def _log_batch_complete(batch_idx, num_edits, exec_time):
    total_edits = (batch_idx + 1) * num_edits
    print(f"[Batch {batch_idx + 1}] Complete: {total_edits} edits in {exec_time:.1f}s", flush=True)

def _log_checkpoint_saved(batch_idx, num_edits, path):
    print(f"[CHECKPOINT] Saved batch {batch_idx} ({(batch_idx + 1) * num_edits} edits) -> {path}", flush=True)

def _log_eval_progress(done, total, rate, elapsed):
    print(f"[EVAL] {done}/{total} records ({rate:.1f} rec/s, {elapsed:.0f}s)", flush=True)
'''
