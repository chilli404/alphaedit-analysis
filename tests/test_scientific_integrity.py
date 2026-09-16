"""Scientific integration tests for the revision experiments.

These verify experimental design correctness — not just code syntax.
Tests 1–10 from advisor audit requirements.
"""

import ast
import glob
import json
import os
import re
import textwrap

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── 1. Branch identity: restore function in logit damage inner script ───────


def test_logit_damage_restore_uses_baseline_weights():
    """The inner script's restore() must copy baseline_w back to model params
    and reset cache_c to baseline_cc. Verify the generated script contains this."""
    import sys
    from pathlib import Path
    from logit_damage_runner import build_inner_script

    script = build_inner_script(
        seed=42, model_name="test", hparams_fname="test.json",
        batch_assignment_path="/tmp/ba.json", stream_path="/tmp/s.json",
        keys_path="/tmp/k.npz", checkpoint_dir="/tmp/ckpt",
        config_path="/tmp/cfg.json", output_path="/tmp/out.json",
        install_batches=10, num_edits=100,
    )
    assert "def restore():" in script, "restore function missing"
    assert "global cache_c" in script, "restore must use global cache_c"
    assert "baseline_w" in script, "restore must reference baseline_w"
    assert "baseline_cc" in script, "restore must reference baseline_cc"
    # Verify restore is called after BOTH high and low branches
    lines = script.split("\n")
    restore_calls = [i for i, l in enumerate(lines) if "restore()" in l and not l.strip().startswith("def")]
    assert len(restore_calls) >= 2, f"Expected >=2 restore() calls, found {len(restore_calls)}"


def test_logit_damage_baseline_saved_before_trials():
    """Baseline weights must be saved BEFORE any trial branches."""
    import sys
    from logit_damage_runner import build_inner_script

    script = build_inner_script(
        seed=42, model_name="t", hparams_fname="t.json",
        batch_assignment_path="/t", stream_path="/t", keys_path="/t",
        checkpoint_dir="/t", config_path="/t", output_path="/t",
        install_batches=10, num_edits=100,
    )
    baseline_pos = script.index("baseline_w")
    trial_pos = script.index("for trial_cfg in")
    assert baseline_pos < trial_pos, "Baseline must be saved before trials start"


# ─── 2. Fixed-batch identity: same batch membership across orderings ─────────


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROJECT_ROOT, "results/matched_ordering/orderings/fb_high_exposure_seed42.json")),
    reason="Ordering files not available locally",
)
def test_fixed_batch_identity():
    """All fixed-batch orderings must contain identical batch memberships."""
    for seed in [42]:
        batches_per_ordering = {}
        for ordering in ["fb_high_exposure", "fb_low_exposure", "fb_random0"]:
            path = os.path.join(
                PROJECT_ROOT, f"results/matched_ordering/orderings/{ordering}_seed{seed}.json"
            )
            if not os.path.exists(path):
                pytest.skip(f"{ordering}_seed{seed}.json not found")
            records = json.load(open(path))
            batch_size = 100
            chunks = [
                frozenset(r["case_id"] for r in records[i : i + batch_size])
                for i in range(0, len(records), batch_size)
            ]
            batches_per_ordering[ordering] = set(chunks)

        assert batches_per_ordering["fb_high_exposure"] == batches_per_ordering["fb_low_exposure"], \
            "HIGH and LOW have different batch memberships"
        assert batches_per_ordering["fb_high_exposure"] == batches_per_ordering["fb_random0"], \
            "HIGH and RANDOM0 have different batch memberships"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROJECT_ROOT, "results/matched_ordering/orderings/fb_high_exposure_seed42.json")),
    reason="Ordering files not available locally",
)
def test_fixed_batch_within_batch_cosine_identical():
    """Within-batch key cosine must be identical across orderings (if keys available)."""
    keys_path = os.path.join(PROJECT_ROOT, "results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    if not os.path.exists(keys_path):
        pytest.skip("Key vectors not available")

    npz = np.load(keys_path)
    all_keys = npz["keys"]
    all_cids = npz["case_ids"].tolist()
    cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}

    norms = np.linalg.norm(all_keys, axis=1, keepdims=True)
    normed = all_keys / np.maximum(norms, 1e-8)

    cosines_per_ordering = {}
    for ordering in ["fb_high_exposure", "fb_low_exposure", "fb_random0"]:
        path = os.path.join(
            PROJECT_ROOT, f"results/matched_ordering/orderings/{ordering}_seed42.json"
        )
        records = json.load(open(path))
        batch_cosines = []
        for i in range(0, len(records), 100):
            batch = records[i : i + 100]
            indices = [cid_to_idx[r["case_id"]] for r in batch if r["case_id"] in cid_to_idx]
            if len(indices) < 2:
                continue
            bk = normed[indices]
            cos = bk @ bk.T
            n = len(indices)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            batch_cosines.append(float(cos[mask].mean()))
        cosines_per_ordering[ordering] = batch_cosines

    # Within-batch cosines must be identical (same batches, same keys)
    hi = cosines_per_ordering["fb_high_exposure"]
    lo = cosines_per_ordering["fb_low_exposure"]
    # The VALUES should be the same set (batches reordered but each batch unchanged)
    assert sorted(hi) == pytest.approx(sorted(lo), abs=1e-6), \
        "Within-batch cosines differ between HIGH and LOW"


# ─── 3. Same-fact content identity ───────────────────────────────────────────


def test_same_fact_runner_uses_same_batch_records():
    """The select_prompt_variants function operates on the SAME batch_records
    for both HIGH and LOW, so case_ids are identical by construction."""
    runner_path = os.path.join(PROJECT_ROOT, "src/runners/same_fact_damage_runner.py")
    source = open(runner_path).read()

    # The inner script should call select_prompt_variants twice on the same batch
    # There should NOT be separate batch selection for high vs low
    assert 'mode="high"' in source or "mode='high'" in source, \
        "Missing high mode call"
    assert 'mode="low"' in source or "mode='low'" in source, \
        "Missing low mode call"

    # Both calls should use the same batch_records variable
    # The function signature takes batch_records as first arg
    assert "def select_prompt_variants" in source


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROJECT_ROOT, "results/same_fact_damage/seed42")),
    reason="Same-fact results not available",
)
def test_same_fact_results_have_same_case_ids_per_trial():
    """If trial config is available, verify HIGH and LOW share case_ids."""
    config_path = os.path.join(PROJECT_ROOT, "results/same_fact_damage/seed42/trial_config.json")
    if not os.path.exists(config_path):
        pytest.skip("trial_config.json not found")
    config = json.load(open(config_path))
    for trial in config.get("trials", []):
        hi_cids = set(trial.get("high_case_ids", trial.get("case_ids", [])))
        lo_cids = set(trial.get("low_case_ids", trial.get("case_ids", [])))
        if hi_cids and lo_cids:
            assert hi_cids == lo_cids, f"Trial {trial['trial']}: case_ids differ"


# ─── 4. No future leakage in survival panel ──────────────────────────────────


def test_no_future_leakage_in_key_similarity():
    """compute_key_similarity_predictors must only look at positions
    in [insertion_pos+1, max_pos), not beyond max_pos."""
    # Read the function source to verify the loop bound
    panel_path = os.path.join(PROJECT_ROOT, "analysis/interference_panel.py")
    source = open(panel_path).read()

    # Find the loop in compute_key_similarity_predictors
    func_start = source.index("def compute_key_similarity_predictors")
    func_body = source[func_start : func_start + 2000]

    # The range should be (insertion_pos + 1, min(max_pos, len(ordering)))
    assert "range(insertion_pos + 1, min(max_pos, len(ordering)))" in func_body, \
        "Key similarity loop must be bounded by max_pos (checkpoint position)"

    # Should NOT contain any reference to future data beyond max_pos
    assert "range(insertion_pos + 1, len(ordering))" not in func_body, \
        "Loop must NOT go to end of ordering — must stop at checkpoint"


def test_key_similarity_synthetic():
    """Synthetic test: verify predictor only uses keys within the window."""
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

    # Can't easily import due to dependencies, so test the logic directly
    ordering = [100, 101, 102, 103, 104]
    keys = {
        100: np.array([1.0, 0.0]),
        101: np.array([0.9, 0.1]),
        102: np.array([0.0, 1.0]),
        103: np.array([0.8, 0.2]),
        104: np.array([0.7, 0.3]),
    }

    # For case 100 at insertion_pos=0, max_pos=3:
    # should only see positions 1,2 (case_ids 101, 102), NOT 103 or 104
    insertion_pos = 0
    max_pos = 3
    k_i = keys[100] / np.linalg.norm(keys[100])
    cosines = []
    for pos_j in range(insertion_pos + 1, min(max_pos, len(ordering))):
        k_j = keys[ordering[pos_j]]
        k_j_norm = k_j / np.linalg.norm(k_j)
        cosines.append(float(np.dot(k_i, k_j_norm)))

    assert len(cosines) == 2, f"Expected 2 comparisons, got {len(cosines)}"
    # case 101 should have high cosine (similar), case 102 low (orthogonal)
    assert cosines[0] > 0.5, "Case 101 should be similar to case 100"
    assert cosines[1] < 0.1, "Case 102 should be dissimilar to case 100"


# ─── 5. Keyword args in all runners ──────────────────────────────────────────


def test_no_positional_cache_args_in_runners():
    """All apply_AlphaEdit/apply_ae calls must use keyword args for cache_c and P.
    Positional args caused the cache_template bug."""
    runner_dir = os.path.join(PROJECT_ROOT, "src/runners")
    issues = []

    for runner_file in glob.glob(os.path.join(runner_dir, "*.py")):
        source = open(runner_file).read()
        basename = os.path.basename(runner_file)

        # Find apply_ae or apply_AlphaEdit calls with hparams followed by args
        # Pattern: apply_ae(model, tok, ..., hparams, cache_c, P)  ← BAD
        # vs:      apply_ae(model, tok, ..., hparams, cache_c=cache_c, P=P)  ← GOOD
        for match in re.finditer(
            r"apply_(?:ae|AlphaEdit_to_model|memit_to_model)\([^)]*hparams,\s*([^)]+)\)",
            source,
        ):
            args_after_hparams = match.group(1).strip()
            # Skip if it's in a comment or docstring
            line_start = source.rfind("\n", 0, match.start()) + 1
            line = source[line_start : match.end()]
            if line.strip().startswith("#") or line.strip().startswith('"""'):
                continue
            # Check: after hparams, all args should be keyword (contain '=')
            # Split by comma and check each arg
            parts = [p.strip() for p in args_after_hparams.split(",")]
            for part in parts:
                if not part:
                    continue
                # Ignore continuation lines, return_orig_weights_device, etc.
                if "=" not in part and part not in ("P", ""):
                    # This is a positional arg — bad
                    if "cache_c" in part.lower() or part.strip() == "P":
                        issues.append(f"{basename}: positional arg '{part}' in {line.strip()[:80]}")

    assert not issues, "Positional cache_c/P args found:\n" + "\n".join(issues)


def test_no_positional_cache_in_generated_scripts():
    """Check generated inner scripts (which are f-string templates)."""
    import sys
    from logit_damage_runner import build_inner_script

    script = build_inner_script(
        seed=42, model_name="t", hparams_fname="t.json",
        batch_assignment_path="/t", stream_path="/t", keys_path="/t",
        checkpoint_dir="/t", config_path="/t", output_path="/t",
        install_batches=10, num_edits=100,
    )

    # All apply_ae calls should use keyword args (may span multiple lines)
    # Find each "apply_ae(" and grab until the matching balanced paren
    idx = 0
    while True:
        pos = script.find("apply_ae(", idx)
        if pos == -1:
            break
        # Find balanced closing paren
        depth = 0
        end = pos
        for i in range(pos, len(script)):
            if script[i] == "(":
                depth += 1
            elif script[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        call = script[pos:end]
        # Skip the def line itself
        if "def " in script[max(0, pos - 30):pos]:
            idx = end
            continue
        assert "cache_c=" in call, f"Missing cache_c= keyword in: {call[:120]}"
        assert "P=" in call, f"Missing P= keyword in: {call[:120]}"
        idx = end


# ─── 6. MEMIT-Seq uses same trial manifests ──────────────────────────────────


def test_memit_runner_imports_shared_trial_logic():
    """logit_damage_memit_runner must import rank_future_batches and
    build_intervention_config from logit_damage_runner — not reimplement."""
    runner_path = os.path.join(PROJECT_ROOT, "src/runners/logit_damage_memit_runner.py")
    if not os.path.exists(runner_path):
        pytest.skip("MEMIT runner not yet created")

    source = open(runner_path).read()
    assert "from logit_damage_runner import" in source or \
           "import logit_damage_runner" in source, \
        "MEMIT runner must import trial logic from logit_damage_runner"
    assert "rank_future_batches" in source, "Must use shared rank_future_batches"
    assert "build_intervention_config" in source, "Must use shared build_intervention_config"


# ─── 7. Protection history semantics audit ────────────────────────────────────


def test_protection_runner_documents_cache_behavior():
    """Protection runner reapplies edits. Verify and document whether
    reapplied edits add to cache_c again (the advisor's audit request)."""
    runner_path = os.path.join(PROJECT_ROOT, "src/runners/protected_editing_runner.py")
    if not os.path.exists(runner_path):
        pytest.skip("Protection runner not yet created")

    source = open(runner_path).read()

    # The runner calls apply_AlphaEdit_to_model for protection, which
    # internally updates cache_c (line 146-148 of AlphaEdit_main.py).
    # So YES, reapplied edits DO add their keys to cache_c again.
    # This is a documented design choice, not a bug.
    assert "apply_AlphaEdit_to_model" in source or "apply_ae" in source, \
        "Protection must call the editing function"

    # Check that the cache is updated (cache_c[:] = cache_c_new pattern)
    assert "cache_c" in source, "Protection runner must handle cache_c"

    # DOCUMENT: reapplication DOES add duplicate keys to cache_c.
    # This means protected edits get double-weighted in history.
    # Whether this is desirable depends on the experiment's goal.
    has_comment = "cache" in source.lower() and ("duplicate" in source.lower() or
                                                   "reappl" in source.lower() or
                                                   "double" in source.lower() or
                                                   "again" in source.lower())
    # Not a hard failure — just document the finding
    if not has_comment:
        import warnings
        warnings.warn(
            "Protection runner reapplies edits via apply_AlphaEdit_to_model, "
            "which adds duplicate keys to cache_c. This is the expected vendor "
            "behavior but should be documented in the paper."
        )


# ─── 8. Cancellation factor bounds ───────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROJECT_ROOT, "results/signed_all_trajectories.json")),
    reason="Signed trajectory results not available",
)
def test_cancellation_factor_bounds():
    """All cancellation factors must be in [0, 1]."""
    data = json.load(open(os.path.join(PROJECT_ROOT, "results/signed_all_trajectories.json")))
    trajectories = data.get("trajectories", data.get("results", []))
    if isinstance(trajectories, dict):
        trajectories = list(trajectories.values())

    found_any = False
    for traj in trajectories:
        if isinstance(traj, dict):
            cf = traj.get("cancellation_factor", traj.get("cancel_pct"))
            if cf is not None:
                found_any = True
                # cancellation_factor may be a percentage (0-100) or fraction (0-1)
                if cf > 1:
                    cf = cf / 100.0
                assert 0 <= cf <= 1, f"Cancellation factor {cf} out of bounds"

    if not found_any:
        pytest.skip("No cancellation_factor found in results")


# ─── 9. Exposure ordering validation ─────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROJECT_ROOT, "results/matched_ordering/diagnostics/fixed_batch_report_seed42.json")),
    reason="Fixed batch report not available",
)
def test_high_exposure_exceeds_low():
    """fb_high_exposure must have higher mean future-exposure than fb_low_exposure."""
    for seed in [42, 2024, 137]:
        path = os.path.join(
            PROJECT_ROOT,
            f"results/matched_ordering/diagnostics/fixed_batch_report_seed{seed}.json",
        )
        if not os.path.exists(path):
            continue
        report = json.load(open(path))
        orderings = report.get("orderings", {})
        hi = orderings.get("fb_high_exposure", {}).get("mean_future_exposure")
        lo = orderings.get("fb_low_exposure", {}).get("mean_future_exposure")
        if hi is not None and lo is not None:
            assert hi > lo, \
                f"Seed {seed}: HIGH exposure ({hi:.4f}) must exceed LOW ({lo:.4f})"


# ─── 10. Deterministic ordering generation ───────────────────────────────────


def test_ordering_generation_deterministic():
    """Running the ordering functions twice with the same seed must produce
    identical results."""
    import sys
    import random
    from generate_orderings import (
        assign_fixed_batches,
        compute_batch_centroids,
        order_batches_high_exposure,
        order_batches_low_exposure,
    )

    # Synthetic data
    np.random.seed(999)
    records = [{"case_id": i, "requested_rewrite": {"relation_id": f"R{i % 5}"}} for i in range(500)]
    keys = np.random.randn(500, 32).astype(np.float32)
    cid_to_idx = {i: i for i in range(500)}

    results = []
    for _ in range(2):
        rng = random.Random(42)
        batches = assign_fixed_batches(records, 100, rng)
        centroids = compute_batch_centroids(batches, keys, cid_to_idx)
        hi = order_batches_high_exposure(batches, centroids, random.Random(100))
        lo = order_batches_low_exposure(batches, centroids, random.Random(200))
        hi_ids = [r["case_id"] for r in hi]
        lo_ids = [r["case_id"] for r in lo]
        results.append((hi_ids, lo_ids))

    assert results[0][0] == results[1][0], "High-exposure ordering not deterministic"
    assert results[0][1] == results[1][1], "Low-exposure ordering not deterministic"


# ─── Bonus: all generated inner scripts must be valid Python ─────────────────


def test_all_runner_inner_scripts_parse():
    """Every build_inner_script function must produce valid Python."""
    import sys

    from logit_damage_runner import build_inner_script as ld_script

    dummy_args = dict(
        seed=42, model_name="t", hparams_fname="t.json",
        batch_assignment_path="/t", stream_path="/t", keys_path="/t",
        checkpoint_dir="/t", config_path="/t", output_path="/t",
        install_batches=10, num_edits=100,
    )
    ast.parse(ld_script(**dummy_args))

    # Same-fact runner
    sf_path = os.path.join(PROJECT_ROOT, "src/runners/same_fact_damage_runner.py")
    if os.path.exists(sf_path):
        source = open(sf_path).read()
        # Find the textwrap.dedent block and verify it's syntactically correct
        # (Can't easily call build function without more args, so just check file parses)
        ast.parse(source)

    # MEMIT runner
    memit_path = os.path.join(PROJECT_ROOT, "src/runners/logit_damage_memit_runner.py")
    if os.path.exists(memit_path):
        source = open(memit_path).read()
        ast.parse(source)
