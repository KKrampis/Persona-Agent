"""
Activation Capping — the core stabilization method (Section 5).

Formula (Equation 1):
  h ← h − v · min(⟨h, v⟩ − τ, 0)

where:
  h = post-MLP residual stream activation [batch, seq_len, d_model]
  v = Assistant Axis unit vector [d_model]
  τ = activation cap threshold (25th percentile of calibration distribution)

Effect: if projection ⟨h,v⟩ < τ, h is shifted along v until projection = τ.
If projection ≥ τ, h is unchanged.
Applied simultaneously at multiple adjacent layers (Qwen: 46–53, Llama: 56–71).
"""

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from config import CAP_LAYERS, CAP_PERCENTILE
from models.hooked_model import HookedModel


def activation_cap(
    h: Tensor,      # [batch, seq_len, d_model]
    v: Tensor,      # [d_model], unit-normalized Assistant Axis
    tau: float,     # threshold scalar
) -> Tensor:
    """
    Apply activation capping (Equation 1 from Section 5).

    h ← h − v · min(⟨h, v⟩ − τ, 0)

    Clamps the component of h along v to a minimum of τ.
    """
    v = v.to(h.device, dtype=h.dtype)
    proj = (h * v).sum(dim=-1, keepdim=True)           # ⟨h, v⟩  [batch, seq, 1]
    correction = v * torch.clamp(proj - tau, max=0.0)  # v · min(proj−τ, 0)
    return h - correction


def make_capping_hook(v: Tensor, tau: float) -> Callable:
    """
    Return a hook function that applies activation capping.
    hook signature: (hidden_state: Tensor[batch, seq, d_model]) -> Tensor
    """
    def hook(hidden: Tensor) -> Tensor:
        return activation_cap(hidden, v, tau)
    return hook


def build_capping_hooks(
    assistant_axis: Tensor,   # [d_model], unit vector
    tau: float,               # threshold
    model_key: str,
) -> dict[int, Callable]:
    """
    Build hook dict for all capping layers for a given model.

    Best settings (Section 5.1.2):
      Qwen:  layers 46–53  (8 of 64 = 12.5%)
      Llama: layers 56–71  (16 of 80 = 20%)
    """
    layer_start, layer_end = CAP_LAYERS[model_key]
    hook = make_capping_hook(assistant_axis, tau)
    return {layer: hook for layer in range(layer_start, layer_end + 1)}


def calibrate_cap_threshold(
    model: HookedModel,
    calibration_texts: list[str],
    assistant_axis: Tensor,         # [d_model], unit vector
    layer: int,
    percentile: float = CAP_PERCENTILE,
    batch_size: int = 8,
) -> float:
    """
    Calibrate the activation cap threshold τ (Section 5.1.1).

    Uses the 25th percentile of projection distribution from the calibration set,
    which includes all role and assistant rollouts from the extraction phase (n≈912,000).
    The 25th percentile ≈ mean Assistant response projection.
    """
    device = next(model.model.parameters()).device
    projections = []

    for start in range(0, len(calibration_texts), batch_size):
        batch = calibration_texts[start : start + batch_size]
        enc = model.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            acts = model.get_activations(input_ids, attention_mask)
        # Use mean over all token positions for calibration
        layer_acts = acts[layer]  # [batch, seq, d_model]
        attn_mask = attention_mask.unsqueeze(-1).float()
        mean_acts = (layer_acts * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)  # [batch, d_model]
        v = assistant_axis.to(device, dtype=mean_acts.dtype)
        proj = (mean_acts * v).sum(dim=-1)  # [batch]
        projections.extend(proj.cpu().float().tolist())

    return float(np.percentile(projections, percentile))


def sweep_cap_settings(
    model: HookedModel,
    assistant_axes: dict[int, Tensor],   # {layer: axis}
    calibration_texts: list[str],
    model_key: str,
    percentiles: list[float] = [1.0, 25.0, 50.0, 75.0],
) -> dict:
    """
    Sweep over layer ranges and percentiles to find Pareto-optimal settings (Section 5.1.2).
    Returns: {(layer_start, layer_end, percentile): tau}
    """
    results = {}
    layer_start, layer_end = CAP_LAYERS[model_key]
    middle_layer = (layer_start + layer_end) // 2

    for pct in percentiles:
        tau = calibrate_cap_threshold(
            model, calibration_texts, assistant_axes[middle_layer], middle_layer, pct
        )
        results[(layer_start, layer_end, pct)] = tau

    return results
