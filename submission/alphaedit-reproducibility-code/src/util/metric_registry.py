"""
Canonical metric name registry.

Ensures all analysis scripts use consistent metric names by:
  1. Loading the frozen registry from configs/metric_registry.yaml
  2. Validating metric names against the registry
  3. Mapping source field names to canonical names
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _get_registry_path() -> Path:
    """Resolve path to configs/metric_registry.yaml."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "configs" / "metric_registry.yaml"


def load_metric_registry() -> dict:
    """
    Load the canonical metric registry.

    Returns:
        Dict mapping metric_name -> {source_field, type, description, source}
    """
    registry_path = _get_registry_path()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Metric registry not found at {registry_path}."
        )
    with open(registry_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("metrics", {})


def validate_metric_name(name: str) -> bool:
    """
    Check if a metric name is in the canonical registry.

    Args:
        name: Metric name to validate.

    Returns:
        True if name is registered, False otherwise.
    """
    registry = load_metric_registry()
    return name in registry


def get_canonical_name(source_field: str) -> str | None:
    """
    Map a source field name to its canonical metric name.

    Args:
        source_field: The raw field name from evaluate.py output
                      (e.g., "rewrite_prompts_correct").

    Returns:
        Canonical metric name (e.g., "efficacy") or None if not found.
    """
    registry = load_metric_registry()
    for metric_name, spec in registry.items():
        if spec.get("source_field") == source_field:
            return metric_name
    return None


def get_metric_type(name: str) -> str | None:
    """
    Get the type of a metric (binary, continuous, integer).

    Args:
        name: Canonical metric name.

    Returns:
        Metric type string or None if not found.
    """
    registry = load_metric_registry()
    spec = registry.get(name)
    if spec is None:
        return None
    return spec.get("type")


def list_metrics_by_source(source: str) -> list[str]:
    """
    List all metrics from a given source.

    Args:
        source: Source identifier (e.g., "capability_probe", "glue_eval").

    Returns:
        List of canonical metric names from that source.
    """
    registry = load_metric_registry()
    return [
        name for name, spec in registry.items()
        if spec.get("source") == source
    ]
