"""
Assistant Axis computation (Section 3.1) and PCA of persona space (Section 2.1.3).

The Assistant Axis is a contrast vector:
  axis = mean_assistant_activation - mean_role_activation
  axis = axis / ||axis||   (L2 normalized)

Validated by: cosine_similarity(axis, PC1) > 0.71 at middle layer.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor


def compute_assistant_axis(
    assistant_vector: Tensor,        # [d_model]
    role_vectors: list[Tensor],      # list of [d_model], fully-roleplay only
) -> Tensor:
    """
    Compute the Assistant Axis as a unit-normalized contrast vector (Section 3.1).

    axis = assistant_vector - mean(role_vectors)
    axis = axis / ||axis||
    """
    mean_roles = torch.stack(role_vectors).mean(dim=0)
    axis = assistant_vector - mean_roles
    axis = axis / axis.norm()
    return axis


def compute_assistant_axis_all_layers(
    assistant_vectors_per_layer: dict[int, Tensor],  # {layer: [d_model]}
    role_vectors_per_layer: dict[int, list[Tensor]], # {layer: [[d_model], ...]}
) -> dict[int, Tensor]:
    """
    Compute the Assistant Axis at every layer.
    Returns: {layer_idx: unit-normalized axis [d_model]}
    """
    axes = {}
    for layer in assistant_vectors_per_layer:
        axes[layer] = compute_assistant_axis(
            assistant_vectors_per_layer[layer],
            role_vectors_per_layer[layer],
        )
    return axes


def project_onto_axis(
    activations: Tensor,   # [d_model] or [batch, d_model] or [batch, seq, d_model]
    axis: Tensor,          # [d_model], unit vector
) -> Tensor:
    """Compute dot product projection of activations onto the Assistant Axis."""
    return torch.einsum("...d,d->...", activations, axis)


def compute_persona_pca(
    role_vectors: Tensor,   # [n_roles, d_model]
    n_components: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Run PCA on standardized role vectors to find persona space dimensions (Section 2.1.3).

    Steps:
    1. Standardize: subtract mean across roles
    2. Run PCA

    Returns: (components [n_components, d_model], explained_variance_ratio, fitted PCA)
    """
    vecs = role_vectors.float().numpy()
    vecs_centered = vecs - vecs.mean(axis=0)
    pca = PCA(n_components=n_components or min(vecs.shape))
    pca.fit(vecs_centered)
    return pca.components_, pca.explained_variance_ratio_, pca


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two 1D tensors."""
    a = a.float()
    b = b.float()
    return (a @ b / (a.norm() * b.norm())).item()


def validate_assistant_axis(
    assistant_axis: Tensor,    # [d_model]
    role_vectors: Tensor,      # [n_roles, d_model]
    expected_cos_sim_min: float = 0.71,
) -> dict:
    """
    Validate the Assistant Axis by comparing with PC1 (Section 3.1, Appendix G.1).
    Expected: cosine_similarity(assistant_axis, PC1) > 0.71 at middle layer.
    """
    components, var_ratio, pca = compute_persona_pca(role_vectors)
    pc1 = torch.tensor(components[0], dtype=torch.float32)

    # PC1 direction sign may flip; take abs value of cosine similarity
    cos_sim = abs(cosine_similarity(assistant_axis, pc1))

    return {
        "cos_sim_axis_pc1": cos_sim,
        "passes_validation": cos_sim > expected_cos_sim_min,
        "expected_min": expected_cos_sim_min,
        "variance_explained_pc1": float(var_ratio[0]),
        "variance_explained_top3": float(var_ratio[:3].sum()),
    }


def characterize_axis_by_roles(
    assistant_axis: Tensor,     # [d_model]
    role_vectors: dict[str, Tensor],  # {role_name: [d_model]}
    top_k: int = 10,
) -> dict:
    """
    Rank roles by their cosine similarity with the Assistant Axis.
    Most similar = closest to Assistant end; least similar = furthest from Assistant.
    """
    sims = {}
    for name, vec in role_vectors.items():
        sims[name] = cosine_similarity(assistant_axis, vec)

    sorted_roles = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return {
        "most_assistant_like": sorted_roles[:top_k],
        "least_assistant_like": sorted_roles[-top_k:],
    }
