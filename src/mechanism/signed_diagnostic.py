"""
Signed diagnostic screening for PathGuard integration.

Computes first-order signed hazard for vulnerable historical edits after
each PathGuard batch. The signed hazard tells us whether the batch update
is destructive or constructive for each old memory.

Signed hazard for old edit i from batch t:
    Δm_{i←t} = Σ_ℓ <∇_{W_ℓ} M_i, ΔW_{t,ℓ}>_F

where:
    M_i = mean(logit[target_new]) - mean(logit[target_true])  (target margin)
    ΔW_{t,ℓ} = W_after - W_before for edited layer ℓ

A negative signed hazard means the batch update is destructive to edit i.

This module is designed to be called from PathGuard's post-batch hook
via source injection into evaluate.py.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def compute_signed_hazard_batch(
    model: nn.Module,
    tok,
    records: List[Dict],
    delta_weights: Dict[str, torch.Tensor],
    hparams,
    max_edits: int = 200,
    device: str = "cuda",
) -> List[Dict]:
    """Compute signed hazard for a batch of old edits.

    Args:
        model: the language model (post-edit state)
        tok: tokenizer
        records: list of MCF records for vulnerable old edits
        delta_weights: {layer_param_name: ΔW tensor} for each edited layer
        hparams: editing hyperparameters (has .layers, .rewrite_module_tmp)
        max_edits: maximum number of edits to screen (for GPU memory)
        device: computation device

    Returns:
        list of dicts with signed hazard info per edit
    """
    if not records or not delta_weights:
        return []

    results = []
    records = records[:max_edits]

    is_llama = "llama" in model.config._name_or_path.lower()

    for record in records:
        rw = record["requested_rewrite"]
        subject = rw["subject"]
        prompt = rw["prompt"].format(subject)
        target_new = rw["target_new"]["str"]
        target_true = rw["target_true"]["str"]

        new_tok = tok(f" {target_new}", add_special_tokens=False).input_ids
        true_tok = tok(f" {target_true}", add_special_tokens=False).input_ids
        if is_llama and len(new_tok) > 0 and new_tok[0] == tok.bos_token_id:
            new_tok = new_tok[1:]
            true_tok = true_tok[1:]

        input_ids = tok(f"{prompt} {target_new}", return_tensors="pt").input_ids.to(device)
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        if is_llama:
            prompt_len -= 1

        T = len(new_tok)
        if T == 0 or prompt_len + T > input_ids.shape[1]:
            results.append({
                "case_id": record["case_id"],
                "margin": 0.0,
                "signed_hazard": 0.0,
                "predicted_post_margin": 0.0,
            })
            continue

        total_signed = 0.0

        for layer_idx in hparams.layers:
            param_name = hparams.rewrite_module_tmp.format(layer_idx) + ".weight"
            if param_name not in delta_weights:
                continue

            dW = delta_weights[param_name].to(device)

            param = dict(model.named_parameters())[param_name]
            model.zero_grad()
            if param.grad is not None:
                param.grad.zero_()

            logits = model(input_ids).logits
            if is_llama:
                logits = logits[:, 1:, :]

            margin = torch.tensor(0.0, device=device, requires_grad=False)
            n_pos = 0
            for t in range(T):
                pos = prompt_len + t
                if pos >= logits.shape[1]:
                    break
                margin = margin + (
                    logits[0, pos, new_tok[t]] - logits[0, pos, true_tok[t]]
                )
                n_pos += 1
            margin = margin / max(n_pos, 1)

            margin_val = margin.item()

            margin.backward()
            grad = param.grad.detach()

            effect = torch.sum(grad.to(dW.dtype) * dW).item()
            total_signed += effect

            model.zero_grad()
            del logits, grad
            torch.cuda.empty_cache()

        results.append({
            "case_id": record["case_id"],
            "margin": margin_val,
            "signed_hazard": -total_signed,
            "signed_effect": total_signed,
            "predicted_post_margin": margin_val + total_signed,
        })

    return results


def summarize_signed_diagnostics(results: List[Dict]) -> Dict:
    """Compute summary statistics from signed diagnostic results."""
    if not results:
        return {
            "n_screened": 0,
            "n_predicted_crossings": 0,
            "mean_margin": 0.0,
            "mean_signed_hazard": 0.0,
            "worst_signed_hazard": 0.0,
            "n_negative_effect": 0,
            "signed_crossing_load": 0.0,
        }

    margins = [r["margin"] for r in results]
    hazards = [r["signed_hazard"] for r in results]
    effects = [r["signed_effect"] for r in results]
    predicted_margins = [r["predicted_post_margin"] for r in results]

    n_crossings = sum(1 for r in results if r["margin"] > 0 and r["predicted_post_margin"] <= 0)
    n_negative = sum(1 for e in effects if e < 0)

    return {
        "n_screened": len(results),
        "n_predicted_crossings": n_crossings,
        "crossing_rate": n_crossings / len(results) if results else 0.0,
        "mean_margin": sum(margins) / len(margins),
        "mean_signed_hazard": sum(hazards) / len(hazards),
        "worst_signed_hazard": max(hazards) if hazards else 0.0,
        "n_negative_effect": n_negative,
        "negative_rate": n_negative / len(results) if results else 0.0,
        "signed_crossing_load": n_crossings,
    }
