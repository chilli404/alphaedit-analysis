#!/usr/bin/env python3
"""
Consolidate multi-run directories so the best run is always run_000.

For each parent directory containing run_000, run_001, ...:
1. Find the best same-model run (most per-case files).
2. If best != run_000: swap best into run_000, move old run_000 to _archive/.
3. Move all other same-model runs to _archive/.
4. Cross-model runs (e.g., GPT-J runs inside a Llama experiment dir) are
   moved to _archive/cross_model/.

Usage:
    python scripts/consolidate_runs.py --dry-run   # preview changes
    python scripts/consolidate_runs.py              # execute
"""

import argparse
import json
import os
import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def count_cases(run_dir: Path) -> int:
    return len(list(run_dir.glob("100_edits-case_*.json")))


def get_model(run_dir: Path) -> str:
    params = run_dir / "params.json"
    if params.exists():
        try:
            return json.loads(params.read_text()).get("model_name", "?")
        except Exception:
            pass
    return "?"


def expected_model(parent: Path) -> str:
    rel = str(parent.relative_to(RESULTS))
    if "gptj" in rel or "gpt-j" in rel:
        return "EleutherAI_gpt-j-6B"
    if "qwen" in rel:
        return "Qwen2.5-7B"
    return "Llama3-8B"


def consolidate(dry_run: bool = True):
    parents = sorted(set(p.parent for p in RESULTS.rglob("run_000") if p.parent.is_dir()))

    n_skip = 0
    n_already = 0
    n_swap = 0
    n_archive = 0
    n_cross = 0

    for parent in parents:
        runs = sorted([d for d in parent.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if len(runs) <= 1:
            n_skip += 1
            continue

        exp_model = expected_model(parent)
        run_info = {}
        for r in runs:
            run_info[r.name] = {
                "cases": count_cases(r),
                "model": get_model(r),
                "path": r,
            }

        same = {k: v for k, v in run_info.items()
                if exp_model in v["model"] or v["model"] == "?"}
        cross = {k: v for k, v in run_info.items() if k not in same}

        if not same:
            same = run_info
            cross = {}

        best_name = max(same, key=lambda k: (same[k]["cases"], k))
        best_info = same[best_name]
        rel = parent.relative_to(RESULTS)

        if best_name == "run_000" and not cross:
            removable = {k: v for k, v in same.items() if k != "run_000"}
            if not removable:
                n_skip += 1
                continue
            if all(v["cases"] == 0 for v in removable.values()):
                n_already += 1
            else:
                n_already += 1
            # Still archive the extras
            archive = parent / "_archive"
            for rname, rinfo in removable.items():
                src = rinfo["path"]
                dst = archive / rname
                print(f"  archive {rel}/{rname} ({rinfo['cases']} cases)")
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                n_archive += 1
        elif best_name != "run_000":
            n_swap += 1
            archive = parent / "_archive"
            run_000 = parent / "run_000"
            best_path = same[best_name]["path"]

            # 1. Move current run_000 to archive
            if run_000.exists():
                dst_old = archive / "run_000_old"
                print(f"  {rel}/run_000 -> _archive/run_000_old ({same.get('run_000', {}).get('cases', '?')} cases)")
                if not dry_run:
                    dst_old.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(run_000), str(dst_old))

            # 2. Move best into run_000
            print(f"  {rel}/{best_name} -> run_000 ({best_info['cases']} cases) *** PROMOTED")
            if not dry_run:
                shutil.move(str(best_path), str(run_000))

            # 3. Archive remaining same-model runs
            for rname, rinfo in same.items():
                if rname in (best_name, "run_000"):
                    continue
                src = rinfo["path"]
                if not src.exists():
                    continue
                dst = archive / rname
                print(f"  archive {rel}/{rname} ({rinfo['cases']} cases)")
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                n_archive += 1
        else:
            n_already += 1

        # Handle cross-model runs
        for rname, rinfo in cross.items():
            src = rinfo["path"]
            if not src.exists():
                continue
            archive = parent / "_archive" / "cross_model"
            dst = archive / f"{rname}_{rinfo['model'].replace('/', '_')}"
            print(f"  cross-model {rel}/{rname} ({rinfo['model']}, {rinfo['cases']} cases) -> _archive/cross_model/")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            n_cross += 1

    print(f"\nSummary:")
    print(f"  Single-run (no change): {n_skip}")
    print(f"  Already best=run_000:   {n_already}")
    print(f"  Swapped to run_000:     {n_swap}")
    print(f"  Archived same-model:    {n_archive}")
    print(f"  Archived cross-model:   {n_cross}")
    if dry_run:
        print(f"\n  DRY RUN — no files moved. Run without --dry-run to execute.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without moving files")
    args = parser.parse_args()
    consolidate(dry_run=args.dry_run)
