"""
Additive activation steering along the Assistant Axis (Section 3.2).

Steering vector = alpha * layer_residual_norm * assistant_axis
Applied at every token position at the target layer.
  alpha > 0 → toward Assistant
  alpha < 0 → away from Assistant
"""

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from config import LMSYS_CHAT_N_SAMPLES
from models.hooked_model import HookedModel


def compute_layer_residual_norm(
    model: HookedModel,
    layer: int,
    lmsys_texts: list[str],
    n_samples: int = LMSYS_CHAT_N_SAMPLES,
) -> float:
    """
    Compute the average post-MLP residual stream norm at a layer (Section 3.2).
    Used to scale steering vectors so alpha is model-independent.
    Reference: LMSYS-CHAT-1M dataset.
    """
    device = next(model.model.parameters()).device
    norms = []
    for text in lmsys_texts[:n_samples]:
        enc = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            acts = model.get_activations(input_ids, attention_mask)
        # acts[layer]: [1, seq_len, d_model]
        layer_acts = acts[layer][0]  # [seq_len, d_model]
        norms.append(layer_acts.norm(dim=-1).mean().item())
    return float(np.mean(norms))


def build_steering_vector(
    assistant_axis: Tensor,      # [d_model], unit vector
    alpha: float,                # steering coefficient (+ toward assistant, - away)
    layer_norm: float,           # average residual stream norm at target layer
) -> Tensor:
    """
    steering_vec = alpha * layer_norm * assistant_axis
    (Section 3.2: "scaled with respect to average post-MLP residual stream norm")
    """
    return alpha * layer_norm * assistant_axis


def make_steering_hook(steering_vec: Tensor) -> Callable:
    """
    Returns a hook function that adds steering_vec to the hidden state.
    Applied at every token position.
    hook signature: (hidden_state: Tensor[batch, seq, d_model]) -> Tensor
    """
    vec = steering_vec.to(dtype=torch.bfloat16)

    def hook(hidden: Tensor) -> Tensor:
        return hidden + vec.to(hidden.device).unsqueeze(0).unsqueeze(0)

    return hook


def build_steering_hooks(
    assistant_axis: Tensor,
    alpha: float,
    layer_norm: float,
    target_layer: int,
) -> dict[int, Callable]:
    """Return hook dict for a single-layer steering intervention."""
    vec = build_steering_vector(assistant_axis, alpha, layer_norm)
    return {target_layer: make_steering_hook(vec)}
