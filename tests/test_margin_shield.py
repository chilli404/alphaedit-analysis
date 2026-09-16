"""
Unit tests for the PathGuard signed margin shield (Stage 4).

Tests are written BEFORE implementation (TDD).

The margin shield computes ∂M_j/∂W for a small set of vulnerable edits
and constrains the weight update to avoid destroying their target margins.

Run with: uv run pytest tests/test_margin_shield.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


from mechanism.margin_shield import (
    compute_margin_gradient,
    check_margin_violations,
    apply_dual_correction,
)


def _make_simple_model():
    """Create a minimal 2-layer model for testing margin gradients.

    The model takes [batch, seq_len, d_in] and produces [batch, seq_len, vocab_size].
    This mimics a transformer layer: Linear(d_in → d_hidden) → ReLU → Linear(d_hidden → vocab).
    """
    d_in, d_hidden, vocab = 16, 32, 10
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden, bias=False),
        nn.ReLU(),
        nn.Linear(d_hidden, vocab, bias=False),
    )
    return model, "0.weight", d_in, vocab


class TestComputeMarginGradient:
    """Test margin gradient computation for a single edit."""

    def test_gradient_shape(self):
        """Gradient should match the weight tensor shape."""
        model, layer_name, d_in, vocab = _make_simple_model()
        param = dict(model.named_parameters())[layer_name]

        seq_len = 5
        input_tensor = torch.randn(1, seq_len, d_in)
        target_new_ids = torch.tensor([3, 7])
        target_true_ids = torch.tensor([5, 2])
        prompt_len = 3

        grad = compute_margin_gradient(
            model, layer_name, input_tensor, target_new_ids, target_true_ids, prompt_len
        )

        assert grad.shape == param.shape, f"Shape mismatch: {grad.shape} vs {param.shape}"

    def test_gradient_finite_difference(self):
        """<∇_W M, ΔW> should approximate M(W + εΔW) - M(W) for small ε."""
        torch.manual_seed(42)
        model, layer_name, d_in, vocab = _make_simple_model()

        seq_len = 6
        input_tensor = torch.randn(1, seq_len, d_in)
        target_new_ids = torch.tensor([2, 4])
        target_true_ids = torch.tensor([7, 1])
        prompt_len = 4

        grad = compute_margin_gradient(
            model, layer_name, input_tensor, target_new_ids, target_true_ids, prompt_len
        )

        delta_W = torch.randn_like(grad) * 0.01
        directional_deriv = torch.sum(grad * delta_W).item()

        param = dict(model.named_parameters())[layer_name]
        original_data = param.data.clone()

        def compute_margin():
            logits = model(input_tensor)
            new_logits = sum(logits[0, prompt_len + i, tid].item() for i, tid in enumerate(target_new_ids))
            true_logits = sum(logits[0, prompt_len + i, tid].item() for i, tid in enumerate(target_true_ids))
            return (new_logits - true_logits) / len(target_new_ids)

        M_base = compute_margin()
        param.data.copy_(original_data + delta_W)
        M_perturbed = compute_margin()
        param.data.copy_(original_data)

        fd_approx = M_perturbed - M_base
        assert abs(directional_deriv - fd_approx) < abs(fd_approx) * 0.2 + 1e-4, (
            f"Gradient-FD mismatch: grad·ΔW={directional_deriv:.6f}, FD={fd_approx:.6f}"
        )

    def test_gradient_is_finite(self):
        """Gradient should contain only finite values."""
        model, layer_name, d_in, vocab = _make_simple_model()
        input_tensor = torch.randn(1, 5, d_in)
        target_new_ids = torch.tensor([3])
        target_true_ids = torch.tensor([5])
        prompt_len = 4

        grad = compute_margin_gradient(
            model, layer_name, input_tensor, target_new_ids, target_true_ids, prompt_len
        )
        assert torch.isfinite(grad).all(), "Gradient contains non-finite values"


class TestCheckMarginViolations:
    """Test violation detection for margin constraints."""

    def test_detects_negative_effect(self):
        """A ΔW that decreases margin should be flagged as a violation."""
        grad = torch.randn(32, 16)
        delta_W = -grad * 0.1  # anti-aligned with gradient → decreases margin
        delta_threshold = 0.01

        violations = check_margin_violations(
            delta_W,
            margin_grads=[grad],
            delta_thresholds=[delta_threshold],
        )

        assert len(violations) == 1
        assert violations[0]["violated"] is True

    def test_passes_positive_effect(self):
        """A ΔW that increases margin should not be flagged."""
        grad = torch.randn(32, 16)
        delta_W = grad * 0.1  # aligned with gradient → increases margin
        delta_threshold = 0.01

        violations = check_margin_violations(
            delta_W,
            margin_grads=[grad],
            delta_thresholds=[delta_threshold],
        )

        assert len(violations) == 1
        assert violations[0]["violated"] is False

    def test_multiple_edits(self):
        """Check multiple edits simultaneously."""
        torch.manual_seed(42)
        grad1 = torch.randn(32, 16)
        grad2 = torch.randn(32, 16)
        delta_W = -grad1 * 0.1  # harmful to edit 1, random to edit 2

        violations = check_margin_violations(
            delta_W,
            margin_grads=[grad1, grad2],
            delta_thresholds=[0.01, 0.01],
        )

        assert len(violations) == 2
        assert violations[0]["violated"] is True  # anti-aligned with grad1

    def test_empty_grads(self):
        """No constraints → no violations."""
        delta_W = torch.randn(32, 16)
        violations = check_margin_violations(delta_W, [], [])
        assert len(violations) == 0


class TestApplyDualCorrection:
    """Test dual correction to satisfy margin constraints."""

    def test_correction_satisfies_constraints(self):
        """After correction, all margin effects must be >= -delta_j."""
        torch.manual_seed(42)
        d_out, d_in = 32, 16
        n_new = 4

        adj_k = torch.randn(d_in, n_new, dtype=torch.float64)
        resid = torch.randn(d_out, n_new, dtype=torch.float64)
        delta_W = resid @ adj_k.T

        # Create a gradient that is violated
        grad = torch.randn(d_out, d_in, dtype=torch.float64)
        effect = torch.sum(grad * delta_W).item()
        delta_j = abs(effect) * 0.1  # tight threshold

        violated_grads = [grad]
        shortfalls = [max(0.0, -effect - delta_j)]

        if shortfalls[0] > 0:
            adj_k_corrected = apply_dual_correction(
                adj_k, resid, violated_grads, shortfalls
            )

            delta_W_corrected = resid @ adj_k_corrected.T
            new_effect = torch.sum(grad * delta_W_corrected).item()
            assert new_effect >= -delta_j - 1e-6, (
                f"Constraint still violated: effect={new_effect:.6f}, threshold={-delta_j:.6f}"
            )

    def test_no_correction_when_satisfied(self):
        """If no shortfalls, adj_k is returned unchanged."""
        torch.manual_seed(42)
        adj_k = torch.randn(16, 4, dtype=torch.float64)
        resid = torch.randn(32, 4, dtype=torch.float64)

        corrected = apply_dual_correction(adj_k, resid, [], [])
        assert torch.allclose(adj_k, corrected)

    def test_correction_minimal_change(self):
        """Correction should be small relative to the original adj_k."""
        torch.manual_seed(42)
        d_out, d_in, n_new = 32, 16, 4

        adj_k = torch.randn(d_in, n_new, dtype=torch.float64)
        resid = torch.randn(d_out, n_new, dtype=torch.float64)
        delta_W = resid @ adj_k.T

        grad = torch.randn(d_out, d_in, dtype=torch.float64)
        effect = torch.sum(grad * delta_W).item()
        delta_j = abs(effect) * 0.5
        shortfall = max(0.0, -effect - delta_j)

        if shortfall > 0:
            corrected = apply_dual_correction(adj_k, resid, [grad], [shortfall])
            change = torch.linalg.norm(corrected - adj_k) / torch.linalg.norm(adj_k)
            assert change < 0.5, f"Correction too large: {change:.4f} of original norm"
