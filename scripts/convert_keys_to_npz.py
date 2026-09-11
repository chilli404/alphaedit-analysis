#!/usr/bin/env python3
"""Convert polykernel key extractor .pt output to ordering-compatible .npz format.

The polykernel_key_extractor saves: {"keys": {layer: tensor[D, N]}, "batch_tags": [...]}
The generate_orderings.py expects: .npz with keys[N, D], case_ids[N], layer

Usage:
    uv run python scripts/convert_keys_to_npz.py \
        --input results/key_vectors/gptj_full_mcf/keys_AlphaEdit_seed42.pt \
        --output results/key_vectors/gptj_full_mcf/keys_seed42_layer5.npz \
        --layer 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert .pt key vectors to .npz for ordering generation")
    parser.add_argument("--input", type=str, required=True, help="Path to .pt file from polykernel_key_extractor")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--layer", type=int, required=True, help="Which layer's keys to extract")
    args = parser.parse_args()

    data = torch.load(args.input, map_location="cpu")
    keys_dict = data["keys"]
    batch_tags = data["batch_tags"]

    if args.layer not in keys_dict:
        available = sorted(keys_dict.keys())
        print(f"ERROR: Layer {args.layer} not found. Available: {available}")
        raise SystemExit(1)

    keys_tensor = keys_dict[args.layer]  # shape [D, N]
    keys_np = keys_tensor.numpy().T  # shape [N, D]

    # Extract case_ids from batch_tags
    case_ids = []
    for tag in batch_tags:
        for cid in tag["case_ids"]:
            case_ids.append(cid)
    case_ids_np = np.array(case_ids, dtype=np.int32)

    assert len(case_ids_np) == keys_np.shape[0], (
        f"Mismatch: {len(case_ids_np)} case_ids vs {keys_np.shape[0]} keys"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, keys=keys_np, case_ids=case_ids_np, layer=np.array(args.layer))

    print(f"Converted: {args.input} → {args.output}")
    print(f"  Layer: {args.layer}")
    print(f"  Keys shape: {keys_np.shape}")
    print(f"  Case IDs: {len(case_ids_np)}")


if __name__ == "__main__":
    main()
