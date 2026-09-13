"""Shared checkpoint save/load/skip/validate logic.

Extracted from the ~830 duplicated lines across checkpoint_runner.py,
memit_sequential_runner.py, polykernel_seqreg_runner.py, and pathguard_runner.py.

Each runner passes its algorithm-specific extra_state (e.g. cache_c, prev_cache,
pathguard_state) without this module needing to know about those details.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def should_skip(batch_idx: int, start_batch: int) -> bool:
    return batch_idx < start_batch


def should_save(batch_idx: int, save_interval: int) -> bool:
    return (batch_idx + 1) % save_interval == 0


def validate_checkpoint_path(ckpt_dir: str, base_alg: str) -> None:
    """Raise RuntimeError if checkpoint path doesn't contain the expected algorithm prefix."""
    expected = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
    if expected not in str(ckpt_dir):
        raise RuntimeError(
            f"Checkpoint path mismatch: base_alg={base_alg} expects "
            f"'{expected}' in path, got: {ckpt_dir}\n"
            f"This likely means the code is stale. Commit and redeploy."
        )


def find_latest_checkpoint(ckpt_dir: Path | str) -> tuple[int, Path] | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    batch_dirs = sorted(
        [d for d in ckpt_dir.glob("batch_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1,
    )
    for batch_dir in reversed(batch_dirs):
        if (batch_dir / "metadata.json").exists():
            try:
                batch_idx = int(batch_dir.name.split("_")[1])
                return (batch_idx, batch_dir)
            except (ValueError, IndexError):
                continue
    return None


def save_checkpoint(
    batch_idx: int,
    model: Any,
    hparams: Any,
    ckpt_dir: str,
    num_edits: int,
    extra_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    base_alg: str | None = None,
) -> Path:
    """Save model weights + optional extra state to a checkpoint directory.

    Args:
        batch_idx: Current batch index.
        model: The model (must support .named_parameters()).
        hparams: Hyperparameters (must have .layers and .rewrite_module_tmp).
        ckpt_dir: Base checkpoint directory.
        num_edits: Edits per batch (for computing total_edits).
        extra_state: Dict of {filename: tensor_or_serializable} for algorithm-specific state.
        metadata: Additional metadata fields merged with standard ones.
        base_alg: If set, validate the checkpoint path matches this algorithm.

    Returns:
        Path to the batch checkpoint directory.
    """
    import torch

    ckpt_dir = Path(ckpt_dir)
    if base_alg:
        validate_checkpoint_path(str(ckpt_dir), base_alg)

    batch_dir = ckpt_dir / f"batch_{batch_idx}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save edited layer weights
    layer_weights = {}
    all_params = dict(model.named_parameters())
    for layer_idx in hparams.layers:
        key = hparams.rewrite_module_tmp.format(layer_idx) + ".weight"
        if key in all_params:
            layer_weights[key] = all_params[key].data.cpu()
    if not layer_weights:
        print(f"  [CHECKPOINT] WARNING: No matching parameters for rewrite_module_tmp='{hparams.rewrite_module_tmp}'")
    torch.save(layer_weights, str(batch_dir / "model_weights.pt"))

    # Save extra state
    if extra_state:
        for filename, data in extra_state.items():
            filepath = batch_dir / filename
            if filename.endswith(".pt"):
                torch.save(data, str(filepath))
            elif filename.endswith(".jsonl"):
                with open(filepath, "w") as f:
                    for entry in data:
                        f.write(json.dumps(entry) + "\n")
            elif filename.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)

    # Save metadata
    meta = {
        "batch_idx": batch_idx,
        "total_edits": (batch_idx + 1) * num_edits,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)
    with open(batch_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Verify critical files exist (S3 FUSE can silently drop writes)
    weights_path = batch_dir / "model_weights.pt"
    meta_path = batch_dir / "metadata.json"
    if not weights_path.exists():
        raise RuntimeError(f"[CHECKPOINT] WRITE FAILED: {weights_path} not found after save")
    if not meta_path.exists():
        raise RuntimeError(f"[CHECKPOINT] WRITE FAILED: {meta_path} not found after save")

    print(f"  [CHECKPOINT] Saved batch {batch_idx} ({(batch_idx + 1) * num_edits} edits) -> {batch_dir}")
    return batch_dir


def load_checkpoint(
    model: Any,
    hparams: Any,
    ckpt_dir: str,
    batch_idx: int,
    extra_state_keys: list[str] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Load model weights + optional extra state from a checkpoint.

    Args:
        model: The model to load weights into.
        hparams: Hyperparameters.
        ckpt_dir: Base checkpoint directory.
        batch_idx: Which batch checkpoint to load.
        extra_state_keys: List of filenames to load (e.g. ["prev_cache.pt"]).
        device: Device to load weights to.

    Returns:
        Dict with "loaded": bool and any loaded extra state keyed by filename.
    """
    import torch

    batch_dir = Path(ckpt_dir) / f"batch_{batch_idx}"
    result = {"loaded": False}

    if not batch_dir.exists():
        print(f"  [CHECKPOINT] WARNING: Expected checkpoint at {batch_dir} not found.")
        return result

    # Load model weights
    weights_file = batch_dir / "model_weights.pt"
    if weights_file.exists():
        layer_weights = torch.load(str(weights_file), map_location=device)
        param_dict = dict(model.named_parameters())
        loaded_count = 0
        for param_name, param_data in layer_weights.items():
            if param_name in param_dict:
                param_dict[param_name].data.copy_(param_data.to(param_dict[param_name].device))
                loaded_count += 1
        print(f"  [CHECKPOINT] Loaded {loaded_count} weight tensors from {weights_file}")
        result["loaded"] = True

    # Load extra state
    if extra_state_keys:
        for key in extra_state_keys:
            filepath = batch_dir / key
            if filepath.exists():
                if key.endswith(".pt"):
                    result[key] = torch.load(str(filepath), map_location="cpu")
                elif key.endswith(".jsonl"):
                    entries = []
                    with open(filepath) as f:
                        for line in f:
                            if line.strip():
                                entries.append(json.loads(line))
                    result[key] = entries
                elif key.endswith(".json"):
                    with open(filepath) as f:
                        result[key] = json.load(f)
                print(f"  [CHECKPOINT] Loaded {key}")

    return result
