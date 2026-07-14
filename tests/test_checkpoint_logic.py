#!/usr/bin/env python3
"""
Minimal end-to-end test for checkpoint runner logic.
No GPU required — tests pure Python logic only.

Run with: uv run python tests/test_checkpoint_logic.py
"""
import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Mock GPU-only imports before importing checkpoint_runner
import types

mock_model_download = types.ModuleType("model_download")
mock_model_download.resolve_model_path = lambda x: x
sys.modules["model_download"] = mock_model_download

mock_setup_hparams = types.ModuleType("setup_hparams")
mock_setup_hparams.link_hparams = lambda: None
sys.modules["setup_hparams"] = mock_setup_hparams


def should_save(cnt, interval=10):
    """Mirror of _ckpt_should_save logic."""
    return (cnt + 1) % interval == 0


def main():
    passed = 0
    failed = 0

    def check(condition, msg):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {msg}")
            passed += 1
        else:
            print(f"  FAIL: {msg}")
            failed += 1

    print("=== Checkpoint Runner Logic Tests ===\n")

    # Import after mocking
    from checkpoint_runner import (
        build_checkpoint_script,
        resolve_checkpoint_dir,
        find_latest_checkpoint,
    )
    from seeded_runner import find_latest_run_dir

    # --- Test 1: Generated script compiles ---
    print("[1] Generated script compilation")
    script = build_checkpoint_script(
        seed=42, cuda_device="0", alg_name="AlphaEdit",
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        hparams_fname="Llama3-8B.json", ds_name="mcf",
        dataset_size_limit=5000, num_edits=100,
        downstream_eval_steps=10, conserve_memory=True,
        start_from_batch=30, save_interval=10,
        checkpoint_dir="/tmp/test_ckpt/AlphaEdit/seed42",
        fast_checkpoint=False, eval_at_checkpoints_only=True,
    )
    try:
        compile(script, "<ckpt_script>", "exec")
        check(True, "Resume + milestone mode compiles")
    except SyntaxError as e:
        check(False, f"SyntaxError: {e}")

    script2 = build_checkpoint_script(
        seed=42, cuda_device="0", alg_name="MEMIT",
        model_name="test", hparams_fname="test.json", ds_name="mcf",
        dataset_size_limit=10000, num_edits=100,
        downstream_eval_steps=5, conserve_memory=True,
        start_from_batch=0, save_interval=10,
        checkpoint_dir="/tmp/test2",
        fast_checkpoint=True, eval_at_checkpoints_only=False,
    )
    try:
        compile(script2, "<fast_script>", "exec")
        check(True, "Fresh start + fast mode (MEMIT) compiles")
    except SyntaxError as e:
        check(False, f"SyntaxError: {e}")

    # --- Test 2: Anchors exist in evaluate.py ---
    print("\n[2] Source anchors in evaluate.py")
    from checkpoint_runner import LOOP_ANCHOR, PRE_EDIT_ANCHOR, POST_EDIT_ANCHOR, CUDA_PATCH_TARGET

    eval_path = Path(__file__).resolve().parent.parent / "vendor" / "AlphaEdit" / "experiments" / "evaluate.py"
    source = eval_path.read_text()

    check(LOOP_ANCHOR in source, "LOOP_ANCHOR found")
    check(PRE_EDIT_ANCHOR in source, "PRE_EDIT_ANCHOR found")
    check(POST_EDIT_ANCHOR in source, "POST_EDIT_ANCHOR found")
    check(CUDA_PATCH_TARGET in source, "CUDA_PATCH_TARGET found")

    eval_start = '    # torch.save(hs, "post_edit_hs_memit.pt")\n    start = time()'
    eval_loop = '    for record in ds:\n        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
    check(eval_start in source, "eval_start_anchor found")
    check(eval_loop in source, "eval_loop_anchor found")

    # --- Test 3: Full injection compiles ---
    print("\n[3] Full injection into evaluate.py")
    patched = source.replace(CUDA_PATCH_TARGET, "# patched")

    load_inj = (
        '    _ckpt_cache_c_loaded = None\n'
        '    if _ckpt_start_batch > 0 and "_ckpt_load" in globals():\n'
        '        _ckpt_cache_c_loaded = _ckpt_load(model, hparams, alg_name)\n'
        '        if _ckpt_cache_c_loaded is not None and alg_name == "AlphaEdit":\n'
        '            cache_c = _ckpt_cache_c_loaded\n'
    )
    patched = patched.replace(LOOP_ANCHOR, load_inj + LOOP_ANCHOR, 1)

    skip_inj = (
        '        if "_ckpt_should_skip" in globals() and _ckpt_should_skip(cnt):\n'
        '            cnt += 1\n'
        '            continue\n'
    )
    patched = patched.replace(PRE_EDIT_ANCHOR, skip_inj + PRE_EDIT_ANCHOR, 1)

    save_inj = (
        '        if "_ckpt_should_save" in globals() and _ckpt_should_save(cnt):\n'
        '            _ckpt_save(cnt, model, cache_c if alg_name == "AlphaEdit" else None, hparams, alg_name)\n'
    )
    patched = patched.replace(POST_EDIT_ANCHOR, save_inj + POST_EDIT_ANCHOR, 1)

    eval_skip_code = (
        '    # torch.save(hs, "post_edit_hs_memit.pt")\n'
        '    _do_final_eval = True\n'
        '    if _ckpt_eval_at_checkpoints_only and not _ckpt_should_save(cnt - 1):\n'
        '        _do_final_eval = False\n'
        '    start = time()'
    )
    patched = patched.replace(eval_start, eval_skip_code, 1)

    fast_inj = (
        '    for record in ds:\n'
        '        if not _do_final_eval:\n'
        '            break\n'
        '        if _ckpt_fast_mode and record["case_id"] not in case_ids:\n'
        '            continue\n'
        '        out_file = Path(case_result_template.format(num_edits, record["case_id"]))'
    )
    patched = patched.replace(eval_loop, fast_inj, 1)

    try:
        compile(patched, "evaluate_patched.py", "exec")
        check(True, "Patched evaluate.py compiles")
    except SyntaxError as e:
        check(False, f"SyntaxError at line {e.lineno}: {e.msg}")
        lines = patched.split("\n")
        for i in range(max(0, e.lineno - 4), min(len(lines), e.lineno + 4)):
            m = ">>>" if i == e.lineno - 1 else "   "
            print(f"    {m} {i+1}: {lines[i]}")

    # --- Test 4: Checkpoint dir resolution ---
    print("\n[4] Checkpoint directory resolution")
    ckpt = resolve_checkpoint_dir(None, "AlphaEdit", 42)
    check("AlphaEdit/seed42" in str(ckpt), f"Default -> {ckpt}")
    ckpt2 = resolve_checkpoint_dir("/custom", "MEMIT", 99)
    check(str(ckpt2) == "/custom/MEMIT/seed99", f"Explicit -> {ckpt2}")

    # --- Test 5: find_latest_checkpoint ---
    print("\n[5] find_latest_checkpoint")
    with tempfile.TemporaryDirectory() as tmpdir:
        check(find_latest_checkpoint(Path(tmpdir)) is None, "Empty dir -> None")
        check(find_latest_checkpoint(Path("/nonexistent")) is None, "Missing dir -> None")

        for i in [9, 19, 29]:
            bd = Path(tmpdir) / f"batch_{i}"
            bd.mkdir()
            (bd / "metadata.json").write_text(json.dumps({"batch_idx": i}))

        r = find_latest_checkpoint(Path(tmpdir))
        check(r is not None and r[0] == 29, "Finds latest (batch_29)")

        (Path(tmpdir) / "batch_39").mkdir()
        r = find_latest_checkpoint(Path(tmpdir))
        check(r[0] == 29, "Skips dir without metadata")

        (Path(tmpdir) / "batch_39" / "metadata.json").write_text('{"batch_idx":39}')
        r = find_latest_checkpoint(Path(tmpdir))
        check(r[0] == 39, "Finds batch_39 after metadata added")

    # --- Test 6: _ckpt_should_save ---
    print("\n[6] _ckpt_should_save logic")
    saves = [c for c in range(100) if should_save(c)]
    check(saves == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99], f"Correct positions: {saves[:5]}...")

    # --- Test 7: Skip + resume simulation ---
    print("\n[7] Skip + resume simulation (from batch 30)")
    start_from = 30
    cnt = 0
    edits = []
    save_pos = []

    for _ in range(50):
        if cnt < start_from:
            cnt += 1
            continue
        edits.append(cnt)
        if should_save(cnt):
            save_pos.append(cnt)
        cnt += 1

    check(edits == list(range(30, 50)), "Edited batches 30-49")
    check(save_pos == [39, 49], f"Saved at {save_pos}")
    check(cnt == 50, f"Final cnt={cnt}")

    # --- Test 8: eval_at_checkpoints_only ---
    print("\n[8] eval_at_checkpoints_only scenarios")
    cases = [
        (50, True, True, "cnt=50, milestone -> eval (batch 49 boundary)"),
        (75, True, False, "cnt=75, milestone -> skip (batch 74 not boundary)"),
        (100, True, True, "cnt=100, milestone -> eval (batch 99 boundary)"),
        (40, True, True, "cnt=40, milestone -> eval (batch 39 boundary)"),
        (75, False, True, "cnt=75, normal -> always eval"),
    ]
    for cnt_val, ckpt_only, expected, desc in cases:
        do_eval = not (ckpt_only and not should_save(cnt_val - 1))
        check(do_eval == expected, desc)

    # --- Test 9: JSONL metadata ---
    print("\n[9] JSONL metadata append")
    with tempfile.TemporaryDirectory() as tmpdir:
        mf = Path(tmpdir) / "test.jsonl"
        for seg in [
            {"resumed_from_batch": None, "ended_at_batch": 29},
            {"resumed_from_batch": 30, "ended_at_batch": 49},
        ]:
            with open(mf, "a") as f:
                f.write(json.dumps(seg) + "\n")
        lines = mf.read_text().strip().split("\n")
        check(len(lines) == 2, "Two segments in file")
        check(json.loads(lines[0])["resumed_from_batch"] is None, "Seg 1: fresh start")
        check(json.loads(lines[1])["resumed_from_batch"] == 30, "Seg 2: resumed at 30")

    # --- Test 10: find_latest_run_dir ---
    print("\n[10] find_latest_run_dir")
    rd, ri = find_latest_run_dir("NonExistent")
    check(rd is None and ri is None, "Missing algo -> (None, None)")

    # --- Test 11: globals() vs dir() in exec ---
    print("\n[11] globals() vs dir() in exec (confirms fix)")
    code = (
        "def main():\n"
        "    r = []\n"
        "    if '_fn' in globals(): r.append('globals')\n"
        "    if '_fn' in dir(): r.append('dir')\n"
        "    r.append(f'call={_fn(7)}')\n"
        "    return r\n"
        "_result = main()\n"
    )
    ns = {"__name__": "__main__", "_fn": lambda x: x * 3}
    exec(compile(code, "test", "exec"), ns)
    r = ns["_result"]
    check("globals" in r, "globals() finds injected fn")
    check("dir" not in r, "dir() does NOT find it (bug was real)")
    check("call=21" in r, "Direct call works via global lookup")

    # --- Test 12: Variable scoping ---
    print("\n[12] Variable scoping (_do_final_eval, cnt)")
    code2 = (
        "def main():\n"
        "    cnt = 0\n"
        "    for _ in range(5):\n"
        "        if cnt < 3: cnt += 1; continue\n"
        "        cnt += 1\n"
        "    _do_final_eval = cnt >= 5\n"
        "    evald = []\n"
        "    for i in range(10):\n"
        "        if not _do_final_eval: break\n"
        "        evald.append(i)\n"
        "    return cnt, _do_final_eval, evald\n"
        "_r = main()\n"
    )
    ns2 = {"__name__": "__main__"}
    exec(compile(code2, "scope", "exec"), ns2)
    cnt_r, eval_r, evald = ns2["_r"]
    check(cnt_r == 5, f"cnt persists across phases (cnt={cnt_r})")
    check(eval_r is True, "_do_final_eval set correctly")
    check(evald == list(range(10)), "Eval loop runs when _do_final_eval=True")

    # Summary
    print(f"\n{'=' * 50}")
    total = passed + failed
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
