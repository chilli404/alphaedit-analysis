"""
Dataset fingerprinting and edit ordering utilities.

Provides provenance tracking for the exact dataset records and their
presentation order used in each experiment run. This enables:
  - Verifying that two runs used the same data (fingerprint match)
  - Reconstructing the exact edit sequence post-hoc (edit_ordering.json)
  - Detecting accidental data corruption or subsetting issues
"""

import hashlib
import json
from pathlib import Path


def compute_dataset_fingerprint(records: list[dict]) -> dict:
    """
    Compute a SHA-256 fingerprint of the dataset in presentation order.

    The fingerprint is the hash of the JSON-serialized list of case_ids
    in the order they will be processed. This captures both content
    (which cases) and ordering (in what sequence).

    Args:
        records: List of dataset records, each with a "case_id" key.

    Returns:
        Dict with fingerprint metadata:
            sha256: hex digest of ordered case_id list
            n_records: total record count
            first_5_ids: first 5 case_ids
            last_5_ids: last 5 case_ids
    """
    case_ids = [r["case_id"] for r in records]
    id_bytes = json.dumps(case_ids, separators=(",", ":")).encode("utf-8")
    sha256 = hashlib.sha256(id_bytes).hexdigest()

    return {
        "sha256": sha256,
        "n_records": len(case_ids),
        "first_5_ids": case_ids[:5],
        "last_5_ids": case_ids[-5:] if len(case_ids) >= 5 else case_ids,
    }


def save_edit_ordering(records: list[dict], output_path: Path) -> None:
    """
    Save the full ordered case_id sequence to a sidecar JSON file.

    This file is stored alongside results and enables exact reconstruction
    of the edit order used in an experiment.

    Args:
        records: List of dataset records in presentation order.
        output_path: Path to write the ordering JSON file.
    """
    case_ids = [r["case_id"] for r in records]
    ordering = {
        "case_ids_ordered": case_ids,
        "n_records": len(case_ids),
        "fingerprint": compute_dataset_fingerprint(records),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ordering, f, indent=2)


def build_fingerprint_injection(order_id: int = 0) -> str:
    """
    Build source injection code for dataset fingerprinting.

    Injected into evaluate.py AFTER dataset load, BEFORE the edit loop.
    Computes and prints the fingerprint, and saves edit_ordering.json
    alongside the results.

    Args:
        order_id: The order ID used for this run (0 = canonical).

    Returns:
        Python source code string to inject.
    """
    return f'''\
    # === FINGERPRINT: compute dataset fingerprint (injected) ===
    import hashlib as _fp_hashlib
    import json as _fp_json
    _fp_case_ids = [r["case_id"] for r in ds]
    _fp_id_bytes = _fp_json.dumps(_fp_case_ids, separators=(",", ":")).encode("utf-8")
    _fp_sha256 = _fp_hashlib.sha256(_fp_id_bytes).hexdigest()
    print(f"  [FINGERPRINT] Dataset: {{len(ds)}} records, SHA-256: {{_fp_sha256[:16]}}...")
    print(f"  [FINGERPRINT] Order ID: {order_id}, first 5 IDs: {{_fp_case_ids[:5]}}")
    # Save edit ordering sidecar
    _fp_ordering_path = run_dir / "edit_ordering.json"
    _fp_ordering = {{
        "case_ids_ordered": _fp_case_ids,
        "n_records": len(ds),
        "order_id": {order_id},
        "fingerprint": {{
            "sha256": _fp_sha256,
            "n_records": len(ds),
            "first_5_ids": _fp_case_ids[:5],
            "last_5_ids": _fp_case_ids[-5:] if len(_fp_case_ids) >= 5 else _fp_case_ids,
        }},
    }}
    with open(str(_fp_ordering_path), "w") as _fp_f:
        _fp_json.dump(_fp_ordering, _fp_f, indent=2)
    # === END fingerprint ===
'''
