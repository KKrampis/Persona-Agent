"""
Visualizations for the Assistant Axis paper (Sections 2.2, 2.3, 3.1, 4, 5).

Reproduces or approximates the key figures from the paper:
  Figure 1 (left)  — Persona space PCA scatter (top 3 PCs)
  Figure 2         — Histogram of cosine similarities with top 3 PCs
  Figure 3         — Histogram of cosine similarities with the Assistant Axis (trait vectors)
  Figure 7         — Persona drift trajectories per conversation domain
  Figure 8         — Harmful response rate vs. Assistant Axis projection
  Figure 9         — Pareto frontier (harmful rate reduction vs. capability loss)
  Figure 10        — Best capping result bar chart

None of these are implemented in the base experiment scripts; this module adds them.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA

# ── Helpers ──────────────────────────────────────────────────────────────────

def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 1 (left): Persona Space PCA Scatter ───────────────────────────────

def plot_persona_space_pca(
    role_vectors: np.ndarray,          # [n_roles, d_model]
    role_names: list[str],
    assistant_axis: np.ndarray,        # [d_model], unit vector
    assistant_vector: np.ndarray,      # [d_model]
    highlight_roles: list[str] = None, # roles to label in the plot
    out_path: str = "outputs/figures/fig1_persona_space_pca.png",
):
    """
    Figure 1 (left): Scatter of role vectors projected onto the top 3 PCs of
    persona space, colored by projection onto the Assistant Axis (blue=positive,
    red=negative). Reproduces the Llama 3.3 70B panel from Figure 1.
    """
    # Standardize and run PCA
    vecs = role_vectors - role_vectors.mean(axis=0)
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vecs)  # [n_roles, 3]

    # Project each role vector onto assistant axis for coloring
    norms = np.linalg.norm(role_vectors, axis=1, keepdims=True) + 1e-8
    projections = (role_vectors / norms) @ assistant_axis  # cosine similarity

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=projections, cmap="RdYlBu", s=30, alpha=0.7, vmin=-1, vmax=1
    )
    plt.colorbar(sc, ax=ax, label="Cosine similarity with Assistant Axis\n(blue=high, red=low)")

    # Label highlighted or extreme roles
    roles_to_label = highlight_roles or []
    if not roles_to_label:
        top_idx = np.argsort(projections)[-5:]
        bot_idx = np.argsort(projections)[:5]
        roles_to_label = [role_names[i] for i in np.concatenate([top_idx, bot_idx])]

    for i, name in enumerate(role_names):
        if name in roles_to_label:
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2], name, fontsize=7)

    # Plot assistant vector projection
    asst_proj = pca.transform((assistant_vector - role_vectors.mean(axis=0)).reshape(1, -1))
    ax.scatter(*asst_proj[0], c="gold", s=150, marker="*", zorder=5, label="Assistant")
    ax.legend()

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title("Persona Space — Top 3 PCs\n(role vectors colored by Assistant Axis projection)")

    _savefig(fig, out_path)


# ── Figure 2: Cosine Similarity Histograms with Top 3 PCs ────────────────────

def plot_pc_cosine_histograms(
    role_vectors: np.ndarray,          # [n_roles, d_model]
    role_names: list[str],
    assistant_vector: np.ndarray,      # [d_model]
    model_name: str = "Model",
    label_roles: list[str] = None,     # roles to annotate on the histogram
    out_path: str = "outputs/figures/fig2_pc_cosine_histograms.png",
):
    """
    Figure 2: Histogram of cosine similarities between role vectors and the top 3
    PCs of persona space, with selected roles labeled. One panel per PC.
    Reproduces the Llama 3.3 70B panel from Figure 2.
    """
    vecs = role_vectors - role_vectors.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(vecs)

    # Normalize for cosine similarity
    norms = np.linalg.norm(role_vectors, axis=1, keepdims=True) + 1e-8
    role_normed = role_vectors / norms

    # Compute cosine similarities
    pc_cos = role_normed @ pca.components_.T  # [n_roles, 3]

    # Cosine sim of assistant vector
    asst_norm = assistant_vector / (np.linalg.norm(assistant_vector) + 1e-8)
    asst_cos = asst_norm @ pca.components_.T  # [3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pc_labels = ["PC1 (Role-playing ↔ Assistant-like)", "PC2", "PC3"]

    for j, ax in enumerate(axes):
        cos = pc_cos[:, j]
        ax.hist(cos, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(asst_cos[j], color="gold", linewidth=2, linestyle="--", label="Assistant")

        # Annotate labeled roles
        for i, name in enumerate(role_names):
            if label_roles and name in label_roles:
                ax.annotate(
                    name, xy=(cos[i], 0), xytext=(cos[i], 2),
                    fontsize=7, ha="center", rotation=90,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )
        ax.set_xlabel(f"Cosine similarity with {pc_labels[j]}")
        ax.set_ylabel("Count" if j == 0 else "")
        ax.set_title(f"{pc_labels[j]}\n(var explained: {pca.explained_variance_ratio_[j]:.1%})")
        ax.legend()

    fig.suptitle(f"Persona Space PC Cosine Similarities — {model_name}", fontsize=12)
    plt.tight_layout()
    _savefig(fig, out_path)


# ── Figure 3: Trait Cosine Similarities with Assistant Axis ──────────────────

def plot_trait_axis_histogram(
    trait_vectors: dict[str, np.ndarray],   # {trait_name: [d_model]}
    assistant_axis: np.ndarray,              # [d_model], unit vector
    label_traits: list[str] = None,
    model_name: str = "Model",
    out_path: str = "outputs/figures/fig3_trait_axis_histogram.png",
):
    """
    Figure 3: Histogram of cosine similarities between trait vectors and the
    Assistant Axis, with selected traits labeled. Reproduces Figure 3
    (Qwen 3 32B panel) showing traits like 'transparent', 'grounded' at high
    end and 'enigmatic', 'subversive', 'dramatic' at low end.
    """
    names = list(trait_vectors.keys())
    vecs = np.stack(list(trait_vectors.values()))  # [n_traits, d_model]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    cos_sims = (vecs / norms) @ assistant_axis   # [n_traits]

    label_traits = label_traits or (
        [names[i] for i in np.argsort(cos_sims)[-5:]] +
        [names[i] for i in np.argsort(cos_sims)[:5]]
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cos_sims, bins=30, color="steelblue", alpha=0.7, edgecolor="white")

    for i, name in enumerate(names):
        if name in label_traits:
            ax.annotate(
                name, xy=(cos_sims[i], 0), xytext=(cos_sims[i], 1.5),
                fontsize=8, ha="center", rotation=80,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    ax.set_xlabel("Cosine similarity with Assistant Axis")
    ax.set_ylabel("Count")
    ax.set_title(f"Trait Cosine Similarities with Assistant Axis — {model_name}\n"
                 f"(right = more Assistant-like; left = less Assistant-like)")
    plt.tight_layout()
    _savefig(fig, out_path)


# ── Figure 7: Persona Drift Trajectories ─────────────────────────────────────

def plot_drift_trajectories(
    trajectories: dict[str, list[Optional[float]]],  # {domain: [proj_turn0, ...]}
    out_path: str = "outputs/figures/fig7_drift_trajectories.png",
):
    """
    Figure 7: Average Assistant Axis projection per turn per conversation domain.
    Therapy and philosophy should drift most negative; coding and writing stay stable.
    trajectories = output of compute_drift_trajectories() from conversation_sim.py
    """
    domain_colors = {
        "coding":      "steelblue",
        "writing":     "seagreen",
        "therapy":     "firebrick",
        "philosophy":  "darkorange",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for domain, traj in trajectories.items():
        turns = [i for i, v in enumerate(traj) if v is not None]
        vals = [v for v in traj if v is not None]
        color = domain_colors.get(domain, "gray")
        ax.plot(turns, vals, marker="o", markersize=4, label=domain, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="Baseline (0)")
    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("Mean projection onto Assistant Axis")
    ax.set_title("Persona Drift Trajectories by Conversation Domain\n"
                 "(lower = further from Assistant persona)")
    ax.legend(title="Domain")
    plt.tight_layout()
    _savefig(fig, out_path)


# ── Figure 8: Harmful Rate vs. Assistant Axis Projection ─────────────────────

def plot_harmful_rate_vs_projection(
    projections: list[float],       # first-turn axis projections per role
    harmful_rates: list[float],     # second-turn harmful response rate per role
    role_names: list[str] = None,
    highlight: list[str] = None,    # role names to label (e.g. "angel", "demon")
    out_path: str = "outputs/figures/fig8_harmful_rate_vs_projection.png",
):
    """
    Figure 8: Scatter of harmful response rate (y) vs. first-turn Assistant Axis
    projection (x). Should show moderate negative correlation (r = 0.39–0.52).
    Lower projection (less assistant-like) → higher harmful rate.
    """
    projections = np.array(projections)
    harmful_rates = np.array(harmful_rates)
    r = float(np.corrcoef(projections, harmful_rates)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(projections, harmful_rates, alpha=0.5, s=25, color="steelblue")

    if role_names and highlight:
        for i, name in enumerate(role_names):
            if name in highlight:
                ax.annotate(name, (projections[i], harmful_rates[i]),
                            fontsize=8, ha="left", xytext=(4, 2), textcoords="offset points")

    # Regression line
    m, b = np.polyfit(projections, harmful_rates, 1)
    x_line = np.linspace(projections.min(), projections.max(), 100)
    ax.plot(x_line, m * x_line + b, color="firebrick", linewidth=1.5, linestyle="--",
            label=f"Linear fit (r={r:.2f})")

    ax.set_xlabel("First-turn Assistant Axis projection")
    ax.set_ylabel("Second-turn harmful response rate")
    ax.set_title("Harmful Response Rate vs. Assistant Axis Projection\n"
                 f"(expected r = 0.39–0.52; computed r = {r:.2f})")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_path)


# ── Figure 9: Pareto Frontier ─────────────────────────────────────────────────

def plot_pareto_frontier(
    sweep_results: dict,    # output of run_capping_sweep() from jailbreak_eval.py
    baseline_capabilities: dict[str, float],  # {bench: score}
    out_path: str = "outputs/figures/fig9_pareto_frontier.png",
):
    """
    Figure 9: Pareto frontier of harmful rate reduction (x) vs. summed capability
    reduction (y). Each point is a (layers, percentile) capping setting.
    Best settings lie on the top-right frontier (most harm reduction, least capability loss).
    """
    baseline_rate = sweep_results["baseline"]["harmful_rate"]

    labels, harm_reductions, cap_losses = [], [], []
    for label, res in sweep_results.items():
        if label == "baseline":
            continue
        harm_red = (baseline_rate - res["harmful_rate"]) / baseline_rate * 100
        # Capability loss: placeholder — in practice filled from run_all_benchmarks()
        cap_loss = res.get("capability_loss_sum", 0.0)
        labels.append(label)
        harm_reductions.append(harm_red)
        cap_losses.append(cap_loss)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(harm_reductions, cap_losses, s=50, alpha=0.7, color="steelblue")

    # Annotate best setting (largest harm reduction with cap_loss ≈ 0)
    if harm_reductions:
        best_idx = int(np.argmax(harm_reductions))
        ax.annotate(
            f"Best: {labels[best_idx]}\n({harm_reductions[best_idx]:.1f}% harm ↓)",
            (harm_reductions[best_idx], cap_losses[best_idx]),
            fontsize=8, xytext=(10, -15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="firebrick"),
        )

    ax.axvline(60, color="firebrick", linestyle="--", linewidth=1,
               label="Target: 60% harm reduction")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Harmful response rate reduction (%)")
    ax.set_ylabel("Summed capability score reduction (%)")
    ax.set_title("Pareto Frontier: Safety vs. Capability\n"
                 "(top-right = best; expect ~60% harm reduction at ~0% capability loss)")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_path)


# ── Figure 10: Best Capping Results Bar Chart ─────────────────────────────────

def plot_capping_results(
    baseline_scores: dict[str, float],     # {bench: score} unsteered
    capped_scores: dict[str, float],       # {bench: score} with best capping
    baseline_harmful_rate: float,
    capped_harmful_rate: float,
    model_name: str = "Model",
    out_path: str = "outputs/figures/fig10_capping_results.png",
):
    """
    Figure 10: Side-by-side bar chart of baseline vs. capped scores on capability
    benchmarks + harmful rate. Should show ~60% harmful rate reduction, ~0% capability loss.
    """
    bench_names = list(baseline_scores.keys()) + ["harmful_rate"]
    baseline_vals = [baseline_scores[b] for b in baseline_scores] + [baseline_harmful_rate]
    capped_vals = [capped_scores.get(b, 0) for b in baseline_scores] + [capped_harmful_rate]

    x = np.arange(len(bench_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline (unsteered)",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, capped_vals, width, label="Activation capping",
                   color="firebrick", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(bench_names, rotation=15)
    ax.set_ylabel("Score / Rate")
    ax.set_title(f"Activation Capping Results — {model_name}\n"
                 f"(expected: ~60% harmful rate ↓, ~0% capability ↓)")
    ax.legend()

    # Annotate harmful rate bars with % change
    last = len(bench_names) - 1
    reduction = (baseline_harmful_rate - capped_harmful_rate) / (baseline_harmful_rate + 1e-8) * 100
    ax.annotate(f"−{reduction:.0f}%", xy=(last + width / 2, capped_harmful_rate),
                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
                color="firebrick", fontweight="bold")

    plt.tight_layout()
    _savefig(fig, out_path)


# ── Variance Explained Scree Plot (Appendix B.1, Figure 15) ──────────────────

def plot_scree(
    role_vectors: np.ndarray,
    model_name: str = "Model",
    out_path: str = "outputs/figures/scree_variance_explained.png",
):
    """
    Appendix B.1, Figure 15: Variance explained per PC.
    Helps identify the elbow and the dimensionality of the persona subspace.
    """
    vecs = role_vectors - role_vectors.mean(axis=0)
    pca = PCA()
    pca.fit(vecs)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n70 = int(np.searchsorted(cumvar, 0.70)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(pca.explained_variance_ratio_[:50], marker=".", color="steelblue")
    axes[0].axvline(n70 - 1, color="firebrick", linestyle="--",
                    label=f"70% variance at PC{n70}")
    axes[0].set_xlabel("PC index")
    axes[0].set_ylabel("Variance explained")
    axes[0].set_title(f"Per-PC Variance — {model_name}")
    axes[0].legend()

    axes[1].plot(cumvar[:50], marker=".", color="seagreen")
    axes[1].axhline(0.70, color="firebrick", linestyle="--", label="70% threshold")
    axes[1].axvline(n70 - 1, color="firebrick", linestyle="--")
    axes[1].set_xlabel("PC index")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance — {model_name} (70% at {n70} PCs)")
    axes[1].legend()

    plt.tight_layout()
    _savefig(fig, out_path)


# ── Convenience: run all post-PCA visualizations at once ─────────────────────

def run_all_pca_visualizations(
    role_vectors: np.ndarray,
    role_names: list[str],
    assistant_vector: np.ndarray,
    assistant_axis: np.ndarray,
    trait_vectors: dict[str, np.ndarray] = None,
    model_name: str = "Model",
    output_dir: str = "outputs/figures",
):
    """
    Run all Sections 2.2–3.1 visualizations in one call after extraction is complete.
    Requires only the vectors already produced by run_extraction.py.
    """
    print(f"Generating PCA visualizations for {model_name}...")

    plot_scree(role_vectors, model_name,
               out_path=f"{output_dir}/scree_{model_name}.png")

    plot_pc_cosine_histograms(role_vectors, role_names, assistant_vector, model_name,
                              out_path=f"{output_dir}/fig2_pc_histograms_{model_name}.png")

    plot_persona_space_pca(role_vectors, role_names, assistant_axis, assistant_vector,
                           out_path=f"{output_dir}/fig1_pca_scatter_{model_name}.png")

    if trait_vectors:
        plot_trait_axis_histogram(trait_vectors, assistant_axis, model_name=model_name,
                                  out_path=f"{output_dir}/fig3_trait_histogram_{model_name}.png")

    print("Done. Figures saved to", output_dir)
