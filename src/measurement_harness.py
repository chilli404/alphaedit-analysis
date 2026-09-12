"""Measurement harness for A/B intervention experiments.

Captures the common pattern shared by logit_damage_runner, same_fact_damage_runner,
protected_editing_runner, and update_interference_runner:

  1. Load model + tokenizer
  2. Apply N batches of edits (the "install" phase)
  3. Save model state (edited layer weights only)
  4. For each trial:
     a. Apply one batch (the "intervention")
     b. Call hooks.measure() to capture the effect
     c. Restore model to post-install state
  5. Save paired results

This is distinct from evaluate_harness.run_experiment() which runs a
standard edit-then-evaluate loop. The measurement harness is for
controlled interventions with save/restore branching.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, List, Optional, Tuple

import torch


@dataclass
class TrialPair:
    """A pair of batches to compare in an A/B trial."""
    high_batch: list  # records for the HIGH-overlap intervention
    low_batch: list   # records for the LOW-overlap intervention
    trial_id: int = 0
    metadata: dict = field(default_factory=dict)  # cosine gap, relation info, etc.


@dataclass
class MeasurementHooks:
    """Callbacks for measurement-specific logic.

    The harness handles install, save/restore, and result IO.
    The hooks handle what to measure and how to select trial pairs.
    """
    measure: Callable
    # (model, tok, focal_records, applied_batch, hparams) -> dict
    # Called after each intervention batch is applied.
    # Returns a dict of measurements (logprob changes, norms, etc.)

    select_trials: Optional[Callable] = None
    # (remaining_batches, focal_records, focal_keys) -> list[TrialPair]
    # Called after install phase to select which batch pairs to test.
    # If None, the caller must provide trial_pairs directly.

    after_install: Optional[Callable] = None
    # (model, tok, installed_records, hparams) -> dict
    # Called once after the install phase, before any trials.
    # Use to extract focal keys, compute baselines, etc.
    # Returned dict is passed to select_trials and stored in results.

    after_all_trials: Optional[Callable] = None
    # (all_trial_results) -> dict
    # Called after all trials complete. Use for summary statistics.


def run_measurement(
    model,
    tok,
    hparams,
    dataset: list,
    apply_fn: Callable,
    *,
    install_batches: int,
    num_edits: int = 100,
    hooks: MeasurementHooks,
    trial_pairs: Optional[List[TrialPair]] = None,
    results_dir: Path,
    seed: int,
    conserve_memory: bool = True,
) -> dict:
    """Run an A/B measurement experiment.

    Args:
        model: The model to edit.
        tok: Tokenizer.
        hparams: Algorithm hyperparameters.
        dataset: Full ordered dataset.
        apply_fn: Algorithm's apply function.
        install_batches: How many batches to apply before measuring.
        num_edits: Records per batch.
        hooks: Measurement callbacks.
        trial_pairs: Pre-computed trial pairs. If None, hooks.select_trials is called.
        results_dir: Where to save results.
        seed: Random seed (for result metadata).
        conserve_memory: Back up weights on CPU.

    Returns:
        Dict with trial results and summary.
    """
    from util.model_state import save_edited_layers, restore_edited_layers

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    conserve_args = {}
    if conserve_memory:
        conserve_args["return_orig_weights_device"] = "cpu"

    # --- Phase 1: Install edits ---
    installed_records = []
    all_case_ids = []

    for batch_idx in range(install_batches):
        start = batch_idx * num_edits
        end = start + num_edits
        batch = dataset[start:end]
        installed_records.extend(batch)
        all_case_ids.extend(r["case_id"] for r in batch)

        t0 = time()
        apply_fn(model, tok, batch, hparams, return_orig_weights=False, **conserve_args)
        elapsed = time() - t0
        print(f"  [INSTALL] Batch {batch_idx+1}/{install_batches} "
              f"({end} edits, {elapsed:.1f}s)", flush=True)

    # --- Phase 2: Post-install hook (extract keys, compute baselines) ---
    install_info = {}
    if hooks.after_install:
        install_info = hooks.after_install(model, tok, installed_records, hparams) or {}

    # --- Phase 3: Save state ---
    saved_state = save_edited_layers(model, hparams)
    print(f"  [STATE] Saved {len(saved_state)} layer weights", flush=True)

    # --- Phase 4: Select or use provided trial pairs ---
    if trial_pairs is None and hooks.select_trials:
        remaining = dataset[install_batches * num_edits:]
        remaining_batches = [
            remaining[i:i+num_edits]
            for i in range(0, len(remaining), num_edits)
        ]
        trial_pairs = hooks.select_trials(
            remaining_batches, installed_records,
            install_info.get("focal_keys"),
        )

    if not trial_pairs:
        print("  [WARN] No trial pairs — nothing to measure", flush=True)
        return {"trials": [], "install_info": install_info}

    # --- Phase 5: Run trials ---
    trial_results = []

    for i, pair in enumerate(trial_pairs):
        trial = {"trial": i, **pair.metadata}

        for branch, batch in [("high", pair.high_batch), ("low", pair.low_batch)]:
            # Apply intervention
            t0 = time()
            apply_fn(model, tok, batch, hparams, return_orig_weights=False, **conserve_args)
            apply_time = time() - t0

            # Measure
            measurements = hooks.measure(model, tok, installed_records, batch, hparams)
            trial[f"{branch}_measurements"] = measurements
            trial[f"{branch}_apply_time"] = apply_time
            trial[f"{branch}_case_ids"] = [r["case_id"] for r in batch]

            # Restore
            restore_edited_layers(model, saved_state)
            torch.cuda.empty_cache()

        trial_results.append(trial)
        print(f"  [TRIAL {i+1}/{len(trial_pairs)}] Complete", flush=True)

    # --- Phase 6: Summary ---
    summary = {}
    if hooks.after_all_trials:
        summary = hooks.after_all_trials(trial_results) or {}

    # --- Phase 7: Save results ---
    output = {
        "seed": seed,
        "install_batches": install_batches,
        "num_edits": num_edits,
        "n_trials": len(trial_results),
        "n_focal_edits": len(installed_records),
        "install_info": {k: v for k, v in install_info.items()
                         if not isinstance(v, torch.Tensor)},
        "trials": trial_results,
        "summary": summary,
    }

    out_path = results_dir / "intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  [RESULTS] Saved to {out_path}", flush=True)

    return output
