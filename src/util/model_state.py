"""Save and restore edited model layer weights.

Used by measurement/intervention scripts that need to apply an edit,
measure something, then restore the model to its pre-edit state.

This is the common pattern across logit_damage_runner, same_fact_damage_runner,
protected_editing_runner, and update_interference_runner.
"""

import torch


def save_edited_layers(model, hparams) -> dict:
    """Save only the edited layer weights to CPU. Returns a state dict.

    Only saves the layers specified in hparams.layers via hparams.rewrite_module_tmp,
    NOT the entire model. Typically 5 layers × [4096, 14336] = ~560MB.
    """
    state = {}
    params = dict(model.named_parameters())
    for layer_idx in hparams.layers:
        key = hparams.rewrite_module_tmp.format(layer_idx) + ".weight"
        if key in params:
            state[key] = params[key].detach().cpu().clone()
    return state


def restore_edited_layers(model, state: dict):
    """Restore edited layer weights from a saved state dict.

    Copies saved CPU tensors back to the model's device in-place.
    """
    params = dict(model.named_parameters())
    for key, saved_tensor in state.items():
        if key in params:
            params[key].data.copy_(saved_tensor.to(params[key].device))


def save_full_edited_state(model, hparams, extra: dict = None) -> dict:
    """Save edited layers + optional extra state (cache_c, prev_cache, etc.)."""
    state = {"weights": save_edited_layers(model, hparams)}
    if extra:
        import copy
        state["extra"] = copy.deepcopy(extra)
    return state


def restore_full_edited_state(model, hparams, state: dict, apply_extra=None):
    """Restore edited layers + optional extra state.

    Args:
        apply_extra: callable(extra_dict) to restore algorithm-specific state
    """
    restore_edited_layers(model, state["weights"])
    if "extra" in state and apply_extra:
        apply_extra(state["extra"])
