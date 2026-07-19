"""
Shared source patches for evaluate.py injection.

All runners that exec evaluate.py can use these patches to apply
common modifications without editing the submodule directly.
These are runtime patches (applied on disk before subprocess reads the file),
following the same pattern as the YAML sed patches.

Also provides shared source injection builders (order shuffle, fingerprint)
that multiple runners can use without duplicating injection logic.
"""

from pathlib import Path


# The original P computation block in evaluate.py (commit b84624f)
P_COMPUTE_ANCHOR = """\
    if alg_name == "AlphaEdit":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, "null_space_project.pt")"""

P_COMPUTE_CACHED = """\
    if alg_name == "AlphaEdit":
        if Path("null_space_project.pt").exists():
            P = torch.load("null_space_project.pt", map_location="cpu")
            print(f"Loaded cached null-space projection from null_space_project.pt")
        else:
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, "null_space_project.pt")
            _s3_p_cache = Path("/s3-data/continual-learning/alphaedit/stats/llama3-8b-instruct/null_space_project.pt")
            if _s3_p_cache.parent.exists():
                torch.save(P, str(_s3_p_cache))
                print(f"Computed and cached null-space projection (also persisted to S3)")
            else:
                print(f"Computed and cached null-space projection to null_space_project.pt")"""


def apply_p_cache_patch(source: str) -> str:
    """
    Patch evaluate.py source to cache the null-space projection P.

    P depends only on model architecture + covariance stats + threshold (not on
    edits or seed). Computing it requires SVD on 5 x (14336x14336) matrices which
    takes ~45 minutes. This patch loads from a cached file if available.

    Args:
        source: The evaluate.py source text.

    Returns:
        Patched source with P-caching logic.
    """
    if P_COMPUTE_ANCHOR not in source:
        # Already patched or upstream changed — skip silently
        return source
    return source.replace(P_COMPUTE_ANCHOR, P_COMPUTE_CACHED, 1)


def patch_evaluate_file(alphaedit_root: Path) -> None:
    """
    Apply runtime patches to vendor/AlphaEdit/experiments/evaluate.py on disk.

    Idempotent — safe to call multiple times. Applies:
      - P-cache: loads null_space_project.pt if present instead of recomputing SVD

    Args:
        alphaedit_root: Path to vendor/AlphaEdit/ directory.
    """
    eval_path = alphaedit_root / "experiments" / "evaluate.py"
    source = eval_path.read_text()
    patched = apply_p_cache_patch(source)
    if patched != source:
        eval_path.write_text(patched)
        print("  Applied P-cache patch to evaluate.py")


# --- Source anchor used by order shuffle injection ---
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'


def build_order_shuffle_injection(order_id: int) -> str:
    """
    Build source injection code for dataset shuffling by order_id.

    Injected into evaluate.py BEFORE the main edit loop. If order_id == 0,
    no shuffle is performed (canonical ordering). If order_id > 0, the
    dataset is shuffled using Random(order_id).

    Args:
        order_id: Shuffle seed. 0 = canonical (no shuffle), >0 = shuffle.

    Returns:
        Python source code string to inject before the loop anchor.
        Empty string if order_id == 0.
    """
    if order_id == 0:
        return ""

    return (
        f'    # === ORDER SHUFFLE: shuffle dataset with order_id={order_id} (injected) ===\n'
        f'    import random as _order_rng_module\n'
        f'    _order_rng = _order_rng_module.Random({order_id})\n'
        f'    _shuffled_indices = list(range(len(ds)))\n'
        f'    _order_rng.shuffle(_shuffled_indices)\n'
        f'    ds.data = [ds.data[i] for i in _shuffled_indices]\n'
        f'    print("ORDER SHUFFLE: shuffled " + str(len(ds)) + " records with order_id={order_id}")\n'
        f'    # === END order shuffle ===\n'
    )
