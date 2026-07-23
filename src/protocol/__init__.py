"""Sequential-Memory Evaluation Protocol.

A reusable evaluation framework for sequential knowledge editing methods.
Reports 8 metrics that capture the divergence between current editing success
and accumulated retention — information hidden by aggregate efficacy.

Usage:
    from src.protocol import evaluate_method, protocol_table

    report = evaluate_method("AlphaEdit", seed=42, edits=5000)
    protocol_table([report_ae, report_memit], output_dir="results/figures/paper")
"""

from src.protocol.sequential_memory_eval import evaluate_method, MethodReport
from src.protocol.protocol_table import format_table
