#!/usr/bin/env python3
"""Replace save_pretrained with lightweight layer-weights-only save.

save_pretrained creates 4 shards (16GB) which S3 FUSE mount loses.
This patches the baselines evaluate.py to save only the edited layer
weights (~560MB per checkpoint) in our checkpoint format.

Patches baselines/EvoEdit/experiments/evaluate.py on disk.
Idempotent.

Note: This is the same transformation as scripts/patch_lightweight_checkpoint.py
but implemented as a clean single-concern patch.
"""
from pathlib import Path


SAVE_ANCHOR = """                try:
                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size="5GB"
                    )
                    tok.save_pretrained(ckpt_dir)"""

SAVE_REPLACEMENT = """                try:
                    # Lightweight save: only edited layer weights (S3 FUSE compatible)
                    _layer_weights = {}
                    _all_params = dict(model.named_parameters())
                    for _li in hparams.layers:
                        _rkey = hparams.rewrite_module_tmp.format(_li) + ".weight"
                        if _rkey in _all_params:
                            _layer_weights[_rkey] = _all_params[_rkey].data.cpu()
                    torch.save(_layer_weights, str(ckpt_dir / "model_weights.pt"))
                    if "cache_c" in dir() or "cache_c" in globals():
                        torch.save(cache_c, str(ckpt_dir / "cache_c.pt"))
                    print(f"  Saved {len(_layer_weights)} layer weights + cache_c to {ckpt_dir}")"""


def apply(baselines_root: Path = None):
    if not baselines_root:
        return 0

    eval_path = baselines_root / "experiments" / "evaluate.py"
    if not eval_path.exists():
        return 0

    source = eval_path.read_text()
    if "Lightweight save" in source:
        print("  [s3-checkpoint] Already patched")
        return 0

    if SAVE_ANCHOR not in source:
        print("  [s3-checkpoint] WARNING: save_pretrained anchor not found")
        return 0

    patched = source.replace(SAVE_ANCHOR, SAVE_REPLACEMENT, 1)
    eval_path.write_text(patched)
    print("  [s3-checkpoint] Patched baselines evaluate.py")
    return 1


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent.parent
    apply(baselines_root=project / "baselines" / "EvoEdit")
