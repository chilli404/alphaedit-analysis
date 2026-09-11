#!/usr/bin/env python3
"""Patch evaluate.py to use lightweight layer-weights-only checkpoint save.

Replaces model.save_pretrained() with torch.save() of edited layers only.
This is required for S3 FUSE mounts where save_pretrained's safetensors
sharding can fail silently.

Also saves cache_c alongside model_weights for methods that use it
(MEMIT_seq, MEMIT_seq_rect, AlphaEdit, EvoEdit, NSE).

Usage: python scripts/patch_lightweight_checkpoint.py [path/to/evaluate.py]
       Default: experiments/evaluate.py (relative to CWD)
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/evaluate.py"

with open(path, "r") as f:
    src = f.read()

old = '''                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size="5GB"
                    )
                    tok.save_pretrained(ckpt_dir)'''

new = '''                    _lw = {}
                    _ap = dict(model.named_parameters())
                    for _li in hparams.layers:
                        _rk = hparams.rewrite_module_tmp.format(_li) + '.weight'
                        if _rk in _ap:
                            _lw[_rk] = _ap[_rk].data.cpu()
                    torch.save(_lw, str(ckpt_dir / 'model_weights.pt'))
                    if 'cache_c' in dir() and cache_c is not None:
                        torch.save(cache_c, str(ckpt_dir / 'cache_c.pt'))
                    print(f'  Saved {len(_lw)} layer weights to {ckpt_dir / "model_weights.pt"}')'''

if old in src:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("  [PATCH] Checkpoint save: lightweight layer-weights-only + cache_c")
else:
    print("  [PATCH] WARNING: save_pretrained anchor not found — may already be patched")
