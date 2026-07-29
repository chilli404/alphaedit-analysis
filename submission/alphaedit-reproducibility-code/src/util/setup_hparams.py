"""
Symlink project hparams into the vendor/AlphaEdit submodule.

This keeps the submodule pristine (no tracked file modifications) while
allowing us to version-control custom hparam configs (e.g. Mistral-7B)
in our own repo under configs/hparams/.

Usage:
    from setup_hparams import link_hparams
    link_hparams()

Or standalone:
    python src/setup_hparams.py
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the alphaedit_replication/ directory."""
    return Path(__file__).resolve().parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


def link_hparams() -> None:
    """Symlink project hparams into the vendor submodule."""
    project_root = get_project_root()
    alphaedit_root = get_alphaedit_root()
    hparams_src = project_root / "configs" / "hparams"

    if not hparams_src.exists():
        return

    for alg_dir in hparams_src.iterdir():
        if not alg_dir.is_dir():
            continue
        for hparam_file in alg_dir.glob("*.json"):
            target_dir = alphaedit_root / "hparams" / alg_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / hparam_file.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(hparam_file.resolve())
            print(f"  Linked: hparams/{alg_dir.name}/{hparam_file.name}")


if __name__ == "__main__":
    link_hparams()
    print("Done.")
