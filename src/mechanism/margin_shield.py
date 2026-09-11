"""
PathGuard signed margin shield (Stage 4).

Computes margin gradients for vulnerable historical edits and applies
dual corrections to prevent the weight update from destroying them.

The margin M_j = mean(logit[target_new]) - mean(logit[target_true]) on the
rewrite prompt. The gradient ∂M_j/∂W tells us how a weight change affects
this margin. The constraint <∇_W M_j, ΔW> >= -delta_j prevents destructive
updates while allowing beneficial or neutral ones.

For the full PathGuard algorithm, this is applied AFTER the Woodbury solve
(Stages 1-3) as a sparse active-set correction on only the top-q most
vulnerable edits. Cost: q forward+backward passes per batch.
"""

import torch
import torch.nn as nn


def compute_margin_gradient(
    model: nn.Module,
    layer_name: str,
    input_ids: torch.Tensor,
    target_new_ids: torch.Tensor,
    target_true_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute ∂M_j/∂W for one edit at one layer.

    M_j = (1/T) Σ_t [logit(target_new_t) - logit(target_true_t)]
    at positions prompt_len, prompt_len+1, ..., prompt_len+T-1.

    Args:
        model: the language model (or a small test model)
        layer_name: name of the parameter to differentiate w.r.t.
        input_ids: [1, seq_len] token ids (prompt + target_new tokens)
        target_new_ids: [T] token ids for the new target
        target_true_ids: [T] token ids for the original true target
        prompt_len: number of prompt tokens (target starts at this position)

    Returns:
        grad: tensor matching the shape of model.{layer_name}, float32
    """
    param = dict(model.named_parameters())[layer_name]

    model.zero_grad()
    if param.grad is not None:
        param.grad.zero_()

    logits = model(input_ids.float() if not hasattr(model, 'config') else input_ids)

    if isinstance(logits, tuple):
        logits = logits[0]
    if hasattr(logits, 'logits'):
        logits = logits.logits

    T = len(target_new_ids)
    margin = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=False)
    n_positions = 0

    for t in range(T):
        pos = prompt_len + t
        if pos >= logits.shape[1]:
            break
        margin = margin + (logits[0, pos, target_new_ids[t]] - logits[0, pos, target_true_ids[t]])
        n_positions += 1

    margin = margin / max(n_positions, 1)
    margin.backward()

    grad = param.grad.detach().clone()
    model.zero_grad()

    return grad


def check_margin_violations(
    delta_W: torch.Tensor,
    margin_grads: list[torch.Tensor],
    delta_thresholds: list[float],
) -> list[dict]:
    """Check which margin constraints are violated by a proposed ΔW.

    Constraint: <∇_W M_j, ΔW> >= -delta_j

    Args:
        delta_W: [d_out, d_in] proposed weight update
        margin_grads: list of gradient tensors, one per constrained edit
        delta_thresholds: list of delta_j threshold values (positive)

    Returns:
        list of dicts with keys: index, effect, threshold, violated, shortfall
    """
    results = []
    for i, (grad, delta_j) in enumerate(zip(margin_grads, delta_thresholds)):
        effect = torch.sum(grad.to(delta_W.dtype) * delta_W).item()
        violated = effect < -delta_j
        shortfall = max(0.0, -effect - delta_j) if violated else 0.0
        results.append({
            "index": i,
            "effect": effect,
            "threshold": delta_j,
            "violated": violated,
            "shortfall": shortfall,
        })
    return results


def apply_dual_correction(
    adj_k: torch.Tensor,
    resid: torch.Tensor,
    violated_grads: list[torch.Tensor],
    shortfalls: list[float],
) -> torch.Tensor:
    """Apply minimum-norm correction to adj_k to satisfy margin constraints.

    Given ΔW = resid @ adj_k^T, and constraints <g_j, ΔW> >= -delta_j,
    we need to adjust adj_k so that the constraint is satisfied.

    The correction is: adj_k_new = adj_k + Σ_j alpha_j * correction_j
    where each correction_j pushes the update in the gradient direction.

    For a single constraint <g, resid @ adj_k^T> >= -delta:
        <g, resid @ (adj_k + alpha * v)^T> >= -delta
        <g, resid @ adj_k^T> + alpha * <g, resid @ v^T> >= -delta
    Choose v = resid^T @ g (flattened to adj_k's column space), then:
        alpha = shortfall / <g, resid @ v^T>

    Args:
        adj_k: [d_in, n_new] current solution
        resid: [d_out, n_new] target residuals
        violated_grads: list of gradient tensors for violated constraints
        shortfalls: list of positive shortfall values

    Returns:
        adj_k_corrected: [d_in, n_new] corrected solution
    """
    if not violated_grads or not shortfalls:
        return adj_k

    adj_k_corrected = adj_k.clone()

    for grad, shortfall in zip(violated_grads, shortfalls):
        if shortfall <= 0:
            continue

        grad_d = grad.to(adj_k.dtype)
        # v = resid^T @ grad: [n_new, d_in] → transpose to [d_in, n_new]
        v = (resid.T @ grad_d).T  # [d_in, n_new]

        # Denominator: <grad, resid @ v^T> = trace(grad^T @ resid @ v^T)
        denom = torch.sum(grad_d * (resid @ v.T)).item()

        if abs(denom) < 1e-12:
            continue

        alpha = shortfall / denom
        adj_k_corrected = adj_k_corrected + alpha * v

    return adj_k_corrected
