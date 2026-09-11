"""Resume helper for EvoEdit/NSE checkpoint loading.

Injects resume logic into evaluate.py source code via string replacement.
Called from run_evoedit_baseline.sh and run_nse_baseline.sh.

Usage:
    from evoedit_resume import patch_resume
    source = patch_resume(source, result_base, num_edits=100)
"""

import os
import re
from pathlib import Path


RESUME_CODE = r'''    # === CHECKPOINT RESUME (injected) ===
    _resume_batch = 0
    _ckpt_root = run_dir / checkpoint_subdir
    if _ckpt_root.exists():
        import re as _re
        _ckpt_dirs = sorted(
            [d for d in _ckpt_root.iterdir() if d.is_dir() and _re.match(r"edits_\d+", d.name)],
            key=lambda d: int(d.name.split("_")[1])
        )
        _latest_ckpt = None
        for _cd in reversed(_ckpt_dirs):
            if (_cd / "model_weights.pt").exists():
                _latest_ckpt = _cd
                break
        if _latest_ckpt is not None:
            _ckpt_edits = int(_latest_ckpt.name.split("_")[1])
            _resume_batch = _ckpt_edits // num_edits
            print(f"  [RESUME] Checkpoint: {_latest_ckpt.name} ({_ckpt_edits} edits, batch {_resume_batch})")
            _ckpt_weights = torch.load(str(_latest_ckpt / "model_weights.pt"), map_location="cuda")
            _param_dict = dict(model.named_parameters())
            _loaded = 0
            for _wname, _wtensor in _ckpt_weights.items():
                if _wname in _param_dict:
                    _param_dict[_wname].data.copy_(_wtensor.cuda())
                    _loaded += 1
            del _ckpt_weights
            torch.cuda.empty_cache()
            print(f"  [RESUME] Loaded {_loaded} weight tensors")
            _cache_c_path = _latest_ckpt / "cache_c.pt"
            if _cache_c_path.exists():
                try:
                    cache_c = torch.load(str(_cache_c_path), map_location="cpu")
                    print(f"  [RESUME] Loaded cache_c: {cache_c.shape}")
                except Exception as _e:
                    print(f"  [RESUME] WARNING: cache_c load failed: {_e}")
            print(f"  [RESUME] Skipping first {_resume_batch} batches")
    # === END RESUME ===
'''

SKIP_ORIGINAL = '        # Is the chunk already done?\n        already_finished = True'
SKIP_PATCHED = '''        # Skip batches before resume checkpoint
        if _resume_batch > 0 and cnt < _resume_batch:
            cnt += 1
            continue
        already_finished = True'''

SAVE_CACHE_PATCH = '''                    if 'cache_c' in dir() or 'cache_c' in globals():
                        torch.save(cache_c, str(ckpt_dir / "cache_c.pt"))'''


def patch_resume(source, result_base, alg_name="EvoEdit"):
    """Patch evaluate.py source with resume logic.

    Args:
        source: evaluate.py source code string
        result_base: path like results/evoedit/fb_high/seed42/10000edits/EvoEdit
        alg_name: algorithm name for logging

    Returns:
        Patched source code string
    """
    # 1. Inject resume code before the main loop
    loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
    if loop_anchor in source:
        source = source.replace(loop_anchor, RESUME_CODE + loop_anchor, 1)
        print(f'  [PATCH] {alg_name} resume logic injected')

    # 2. Patch skip logic to honor _resume_batch
    if SKIP_ORIGINAL in source:
        source = source.replace(SKIP_ORIGINAL, SKIP_PATCHED, 1)
        print(f'  [PATCH] {alg_name} resume skip logic injected')

    # 3. Add cache_c save alongside model_weights in checkpoints
    # Find the model_weights.pt save line and add cache_c after it
    mw_save = "torch.save(_layer_weights, str(ckpt_dir / \"model_weights.pt\"))"
    if mw_save in source:
        source = source.replace(
            mw_save,
            mw_save + "\n" + SAVE_CACHE_PATCH,
            1
        )
        print(f'  [PATCH] {alg_name} cache_c save added to checkpoints')

    # 4. Add --continue_from_run detection
    existing_runs = []
    if os.path.isdir(result_base):
        existing_runs = sorted([
            d for d in os.listdir(result_base)
            if d.startswith("run_") and os.path.isdir(os.path.join(result_base, d))
        ])

    continue_run = find_continue_run(result_base)
    if continue_run:
        print(f'  [RESUME] Will continue from {continue_run}')

    return source, continue_run


def find_continue_run(result_base):
    """Find the latest run directory with checkpoints.

    Searches ALL run directories (not just the latest) because failed
    re-launches can create empty run_NNN directories after the one with
    actual checkpoints.
    """
    if not os.path.isdir(result_base):
        return None
    existing_runs = sorted([
        d for d in os.listdir(result_base)
        if d.startswith("run_") and os.path.isdir(os.path.join(result_base, d))
    ])
    if not existing_runs:
        return None
    # Search from latest to earliest for one with checkpoints
    for run_dir in reversed(existing_runs):
        ckpt_dir = os.path.join(result_base, run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            has_ckpts = any(
                d.startswith("edits_") and os.path.isdir(os.path.join(ckpt_dir, d))
                for d in os.listdir(ckpt_dir)
            )
            if has_ckpts:
                return run_dir
    return None
