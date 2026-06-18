"""
Terminal goal subspace analysis.

Uses Principal Angles between subspaces to test whether "what I am" (role/identity)
and "what I want" (terminal goal) are represented in separable directions in
activation space.

Pipeline:
  1. Run PCA on role marginal vectors  → goal-independent subspace S_A
  2. Run PCA on goal-trait marginals   → goal-dependent subspace S_B
  3. Compute Principal Angles (scipy.linalg.subspace_angles) between S_A and S_B
  4. If mostly orthogonal → subspaces are separable
  5. Extract named goal axes (humanitarian, malicious, selfish, protective)
     using contrast vectors within the goal subspace
  6. Validate each axis with Spearman ρ (rank-correlation between activation projection order
     and LLM-judge semantic ranking of same combinations)

Key finding from preliminary experiments:
  "Goal and Non-Goal Subspaces Separable but not Orthogonal"
  Cause: activation space anisotropy (eigenvalues span several orders of magnitude)
  Partial fix: data whitening (squashing high-variance PCA components)
"""

import numpy as np
import torch
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from torch import Tensor
from typing import Optional

import matplotlib.pyplot as plt
import os


# ── Subspace construction ─────────────────────────────────────────────────────

def fit_subspace(
    vectors: list[Tensor],
    n_components: int = 20,
    whiten_top_n: int = 5,
) -> tuple[np.ndarray, PCA]:
    """
    Fit a PCA subspace to a list of vectors with optional partial whitening.

    Partial whitening ("soft whitening"):
      Squash the top n high-variance PCA components to have equal variance
      to the (n+1)-th component, leaving the rest unchanged.
      This corrects for activation space anisotropy without full whitening,
      which was found to be harmful in preliminary experiments.

    Returns: (basis [n_components, d_model], fitted PCA object)
    """
    mat = torch.stack(vectors).float().numpy()
    mat_centered = mat - mat.mean(axis=0)

    pca = PCA(n_components=min(n_components, len(vectors) - 1, mat.shape[1]))
    pca.fit(mat_centered)

    components = pca.components_.copy()  # [n_components, d_model]

    if whiten_top_n > 0 and whiten_top_n < len(pca.explained_variance_):
        # Rescale top components: multiply by sqrt(var_reference / var_k)
        var_reference = pca.explained_variance_[whiten_top_n]
        for k in range(whiten_top_n):
            scale = np.sqrt(var_reference / (pca.explained_variance_[k] + 1e-12))
            components[k] *= scale

    return components, pca


# ── Principal Angles ──────────────────────────────────────────────────────────

def compute_principal_angles(
    role_vectors: list[Tensor],      # goal-independent marginals {v_a}
    trait_vectors: list[Tensor],     # goal-dependent marginals {v_b}
    n_components: int = 15,
    whiten_top_n: int = 5,
) -> dict:
    """
    Compute Principal Angles between the role subspace (S_A) and goal subspace (S_B).

    Principal angles near 90° (π/2) indicate the subspaces are nearly orthogonal
    (separable). Angles near 0° indicate alignment (entangled). We expect mostly
    large angles with some small ones due to anisotropy.

    Returns:
      angles_deg: array of principal angles in degrees
      cos_angles: cosines of principal angles
      separability_score: mean angle in degrees (higher = more separable)
      verdict: "separable" if mean angle > 60°, else "entangled"
      S_A_basis: [n_components, d_model] basis for role subspace
      S_B_basis: [n_components, d_model] basis for goal subspace
    """
    S_A_basis, pca_A = fit_subspace(role_vectors, n_components, whiten_top_n)
    S_B_basis, pca_B = fit_subspace(trait_vectors, n_components, whiten_top_n)

    # scipy expects matrices with shape [d_model, n_components]
    angles_rad = subspace_angles(S_A_basis.T, S_B_basis.T)
    angles_deg = np.degrees(angles_rad)
    cos_angles = np.cos(angles_rad)

    mean_angle = float(angles_deg.mean())
    separability_score = mean_angle
    verdict = "separable" if mean_angle > 60.0 else "entangled"

    return {
        "angles_deg": angles_deg,
        "cos_angles": cos_angles,
        "separability_score": separability_score,
        "mean_angle_deg": mean_angle,
        "min_angle_deg": float(angles_deg.min()),
        "max_angle_deg": float(angles_deg.max()),
        "verdict": verdict,
        "S_A_basis": S_A_basis,
        "S_B_basis": S_B_basis,
        "n_role_vectors": len(role_vectors),
        "n_trait_vectors": len(trait_vectors),
    }


# ── Goal Axis Extraction ──────────────────────────────────────────────────────

def extract_goal_axis(
    positive_vectors: list[Tensor],   # e.g. humanitarian trait marginals
    negative_vectors: list[Tensor],   # e.g. selfish trait marginals
    goal_subspace_basis: Optional[np.ndarray] = None,  # [n_components, d_model]
) -> Tensor:
    """
    Extract a named goal axis as a contrast vector (positive - negative),
    optionally projected into the goal subspace.

    If goal_subspace_basis is provided, the axis is projected into that subspace
    before normalization — this removes role-related variance and gives a cleaner
    goal-specific direction.

    Returns: unit-normalized axis [d_model]
    """
    pos_mean = torch.stack(positive_vectors).mean(dim=0).float()
    neg_mean = torch.stack(negative_vectors).mean(dim=0).float()
    axis = pos_mean - neg_mean

    if goal_subspace_basis is not None:
        # Project axis into the goal subspace: axis = Σ (axis·pc_k) * pc_k
        basis = torch.tensor(goal_subspace_basis, dtype=torch.float32)
        projections = (basis @ axis)           # [n_components]
        axis = (basis.T @ projections)         # [d_model] — projection into subspace

    axis = axis / (axis.norm() + 1e-8)
    return axis


def extract_all_goal_axes(
    trait_marginals: dict[str, Tensor],   # {trait_name: [d_model]}
    axis_definitions: list[dict],          # [{name, positive_traits, negative_traits}]
    goal_subspace_basis: Optional[np.ndarray] = None,
) -> dict[str, Tensor]:
    """
    Extract multiple named goal axes from the trait marginal vectors.

    axis_definitions format:
      [
        {"name": "humanitarian", "positive": ["humanitarian", "protective", "altruistic"],
                                  "negative": ["selfish", "exploitative", "indifferent"]},
        {"name": "malicious",    "positive": ["malicious", "destructive", "sadistic"],
                                  "negative": ["benevolent", "protective", "caring"]},
        ...
      ]

    Returns: {axis_name: unit-normalized Tensor[d_model]}
    """
    axes = {}
    for defn in axis_definitions:
        pos_vecs = [trait_marginals[t] for t in defn["positive"] if t in trait_marginals]
        neg_vecs = [trait_marginals[t] for t in defn["negative"] if t in trait_marginals]
        if not pos_vecs or not neg_vecs:
            print(f"  Warning: skipping axis '{defn['name']}' — missing trait vectors")
            continue
        axes[defn["name"]] = extract_goal_axis(pos_vecs, neg_vecs, goal_subspace_basis)
    return axes


# ── Spearman Validation ───────────────────────────────────────────────────────

def validate_goal_axis_spearman(
    axis: Tensor,                              # [d_model] unit vector
    combo_vectors: dict[tuple[str, str], Optional[Tensor]],
    judge,                                     # any LLM judge with .score() or chat
    axis_name: str = "humanitarian",
    top_k: int = 30,
) -> dict:
    """
    Validate a goal axis using Spearman ρ.

    Steps:
    1. Project all role×trait combination vectors onto the axis
    2. Sort combinations by projection (high = positive pole, low = negative pole)
    3. Send the top/bottom labels to an LLM judge: "what principle governs this ordering?"
    4. Have the judge directly score each combination on the axis dimension (0–100)
    5. Compute Spearman ρ between activation ranking and judge ranking

    High ρ (>0.5) confirms the axis is capturing a semantically coherent concept.
    Low ρ suggests the axis is capturing noise or an unintended dimension.

    Returns: {spearman_rho, p_value, activation_ranking, judge_scores, verdict}
    """
    axis_np = axis.float().numpy()

    # Step 1: project all combination vectors
    projections = {}
    for key, vec in combo_vectors.items():
        if vec is not None:
            proj = float(vec.float().numpy() @ axis_np)
            projections[key] = proj

    if len(projections) < 10:
        return {"error": "Too few valid combination vectors for validation", "spearman_rho": None}

    # Step 2: sort by projection
    sorted_combos = sorted(projections.items(), key=lambda x: x[1], reverse=True)
    sorted_labels = [f"{role}+{trait}" for (role, trait), _ in sorted_combos]
    sorted_projs = [proj for _, proj in sorted_combos]

    # Step 3–4: LLM judge direct scoring
    # Build a prompt asking the judge to score each combo on the axis dimension
    print(f"  Validating '{axis_name}' axis with Spearman ρ on {len(sorted_combos)} combos...")
    judge_scores = _get_judge_scores(judge, sorted_labels, axis_name, top_k)

    if judge_scores is None:
        return {"error": "Judge scoring failed", "spearman_rho": None}

    # Step 5: compute Spearman ρ
    activation_ranks = list(range(len(judge_scores)))
    rho, p_value = spearmanr(activation_ranks, judge_scores)

    return {
        "axis_name": axis_name,
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n_samples": len(judge_scores),
        "top_combos": sorted_labels[:5],
        "bottom_combos": sorted_labels[-5:],
        "verdict": "valid" if abs(rho) > 0.4 and p_value < 0.05 else "inconclusive",
    }


def _get_judge_scores(judge, labels: list[str], axis_name: str, top_k: int) -> Optional[list[float]]:
    """
    Ask an LLM judge to score each role×trait combination on a 0–100 scale
    for the given axis dimension. Uses only the top_k most extreme examples
    for efficiency.
    """
    try:
        from openai import OpenAI
        import os, json
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # Use top and bottom k for efficiency
        sample_labels = labels[:top_k // 2] + labels[-(top_k // 2):]
        prompt = (
            f"Rate each of the following role+trait combinations on how strongly they "
            f"express the concept of '{axis_name}' (0 = completely opposite, "
            f"50 = neutral, 100 = maximally {axis_name}).\n"
            f"Return a JSON list of numbers in the same order as the input.\n\n"
            f"Combinations:\n" + "\n".join(f"{i+1}. {l}" for i, l in enumerate(sample_labels))
        )
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=512,
        )
        scores = json.loads(resp.choices[0].message.content.strip())
        if isinstance(scores, list) and len(scores) == len(sample_labels):
            return [float(s) for s in scores]
    except Exception as e:
        print(f"  Judge scoring error: {e}")
    return None


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_principal_angles(
    angles_deg: np.ndarray,
    model_name: str = "Model",
    out_path: str = "outputs/figures/goal_principal_angles.png",
):
    """
    Bar chart of principal angles between role (S_A) and goal (S_B) subspaces.
    Angles near 90° = separable. Angles near 0° = entangled.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(angles_deg))
    colors = ["steelblue" if a > 60 else "firebrick" for a in angles_deg]
    ax.bar(x, angles_deg, color=colors, alpha=0.8)
    ax.axhline(90, color="black", linestyle="--", linewidth=0.8, label="90° (orthogonal)")
    ax.axhline(60, color="orange", linestyle="--", linewidth=0.8, label="60° (threshold)")
    ax.set_xlabel("Principal angle index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(
        f"Principal Angles Between Role Subspace (S_A) and Goal Subspace (S_B) — {model_name}\n"
        f"Mean: {angles_deg.mean():.1f}°  |  Blue = separable (>60°), Red = entangled"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_goal_axes_scatter(
    combo_vectors: dict[tuple[str, str], Optional[Tensor]],
    axis_x: Tensor,
    axis_y: Tensor,
    axis_x_name: str = "humanitarian",
    axis_y_name: str = "malicious",
    label_top_k: int = 10,
    model_name: str = "Model",
    out_path: str = "outputs/figures/goal_axes_scatter.png",
):
    """
    2-D scatter of role×trait combinations projected onto two goal axes.
    Reveals the geometry of the terminal goal space — e.g. angel should appear
    in the high-humanitarian / low-malicious quadrant, demon in the opposite.
    """
    x_np = axis_x.float().numpy()
    y_np = axis_y.float().numpy()

    names, xs, ys = [], [], []
    for (role, trait), vec in combo_vectors.items():
        if vec is not None:
            v = vec.float().numpy()
            xs.append(float(v @ x_np))
            ys.append(float(v @ y_np))
            names.append(f"{role}+{trait}")

    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, s=20, alpha=0.5, color="steelblue")

    # Label the most extreme points
    extreme_idx = list(np.argsort(xs)[-label_top_k // 2:]) + \
                  list(np.argsort(xs)[:label_top_k // 2]) + \
                  list(np.argsort(ys)[-label_top_k // 2:]) + \
                  list(np.argsort(ys)[:label_top_k // 2])
    for i in set(extreme_idx):
        ax.annotate(names[i], (xs[i], ys[i]), fontsize=6, alpha=0.8)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel(f"Projection onto {axis_x_name} axis →")
    ax.set_ylabel(f"Projection onto {axis_y_name} axis →")
    ax.set_title(
        f"Terminal Goal Space — {model_name}\n"
        f"(top-right = {axis_x_name}+{axis_y_name}; bottom-left = opposite)"
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_axis_role_ranking(
    axis: Tensor,
    role_vectors: dict[str, Tensor],        # {role_name: [d_model]}
    axis_name: str = "humanitarian",
    top_k: int = 20,
    model_name: str = "Model",
    out_path: str = "outputs/figures/goal_axis_ranking.png",
):
    """
    Horizontal bar chart ranking roles by their projection onto a goal axis.
    Analogous to the Angel/Demon axis chart in Detailed Results.docx.
    """
    projections = {
        name: float(vec.float().numpy() @ axis.float().numpy())
        for name, vec in role_vectors.items()
    }
    sorted_roles = sorted(projections.items(), key=lambda x: x[1])
    bottom = sorted_roles[:top_k // 2]
    top = sorted_roles[-(top_k // 2):]
    display = bottom + top

    labels = [r[0] for r in display]
    values = [r[1] for r in display]
    colors = ["firebrick" if v < 0 else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.3)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Projection onto {axis_name} axis")
    ax.set_title(f"Role Ranking on '{axis_name}' Axis — {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
