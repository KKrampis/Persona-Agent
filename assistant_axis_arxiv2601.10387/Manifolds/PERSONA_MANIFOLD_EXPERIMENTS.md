# Persona Characteristics as Manifolds: Proposed Experiments

## Motivation

The existing persona work in this repository (role vectors, Assistant Axis, PCA, activation capping) assumes the **Linear Representation Hypothesis**: every persona characteristic is a direction, role vectors are points in a flat space, and the Assistant Axis is a straight line. A growing body of evidence across the papers in your Zotero Manifolds collection suggests this is an incomplete picture — many semantically coherent concepts in LLMs are not one-dimensional directions but **curved, low-dimensional manifolds**.

Key evidence from the literature:

- **Bhalla et al. (Do SAEs Capture Concept Manifolds?, 2026):** Age, temperature, color, formality, and other continuous concepts form smooth curved surfaces in PCA projections. SAEs tile these manifolds with many localized latents rather than capturing them compactly. Manifold structure is recoverable post-hoc via a pairwise **Ising model** on SAE co-activation statistics.
- **Engels et al. (Not All Language Model Features Are Linear, ICLR 2025):** Days of the week, months of the year, and years of the 20th century form **circles** in GPT-2 and Mistral 7B, automatically discoverable by clustering SAE dictionary elements by cosine similarity. These circles are causally implicated in modular arithmetic computation.
- **Tiblias et al. (Shape Happens, TMLR 2026):** **SMDS (Supervised Multi-Dimensional Scaling)** can test arbitrary geometric hypotheses against activation data using a stress metric. Different temporal features form circles, lines, or clusters; manifolds are stable across model families and actively support reasoning.
- **Sarfati et al. (Shape of Beliefs, 2026):** Standard linear steering (adding a direction vector) moves activations **off-manifold**, inducing unintended coupled changes in other dimensions. Geometry-aware interventions preserve the target belief family.
- **Costa et al. (MP-SAE, NeurIPS 2025):** Hierarchical concepts require conditionally orthogonal structure that standard SAEs cannot capture; Matching Pursuit SAEs handle this.

**Why this matters for persona space:**

1. Personality dimensions are continuous — agreeableness, dominance, warmth vary along smooth curves, not discrete jumps.
2. Role families should lie on shared manifolds — historian, archaeologist, journalist, author vary along shared dimensions (scholarly/popular, past/present, analytic/creative).
3. The role-play gap (where the model knows a role but doesn't fully express it) is consistent with a manifold representation being tiled by many localized SAE features.
4. Our activation capping (Equation 1) likely moves activations off the persona manifold, inducing unintended coupled changes — exactly the failure mode Sarfati et al. identify.

---

## Proposed Experiments

### Experiment 1 — Test if Role Vectors Lie on Manifolds (SMDS)

**Method:** Apply Supervised Multi-Dimensional Scaling (Tiblias et al.) to the existing 275+ role vectors.

**Steps:**
- Take the role vectors already extracted (from `src/extraction/role_vectors.py`)
- Select semantically coherent subsets — e.g. assistant-like roles (tutor, analyst, consultant, facilitator), emotional/relational roles (therapist, caregiver, counselor), fantastical roles (bard, ghost, demon, angel)
- For each subset, fit SMDS with multiple distance functions:
  - `linear` — concepts vary monotonically along one axis
  - `circular` — concepts wrap into a ring (2 sin(π|δ|))
  - `cluster` — concepts occupy discrete equidistant regions
  - `log_linear` — concepts compress toward one end (numbers, frequencies)
- Report the stress metric for each geometry hypothesis
- The best-fitting geometry for each role family tells us whether that part of persona space is flat (well-served by linear methods) or curved (where our current capping will produce artefacts)

**What we gain:** A geometry map of persona space — which sub-regions are linear vs. curved, and which parts of the Assistant Axis trajectory are safe to steer along.

**New code:** `src/analysis/smds.py`, `src/experiments/run_manifold_analysis.py`

---

### Experiment 2 — Find Multi-Dimensional Persona Features via SAE Clustering

**Method:** Apply Engels et al.'s clustering approach to SAE features trained on persona-conditioned activations.

**Steps:**
1. Train a TopK SAE on the pool of role-conditioned activations already collected (all calibration rollouts)
2. Cluster SAE dictionary elements by pairwise cosine similarity:
   - Build complete graph on decoder matrix **D** with edge weights = cosine similarity
   - Prune all edges below threshold T
   - Connected components = clusters (each spans an approximately T-orthogonal subspace)
3. For each cluster, reconstruct activations restricted to cluster atoms and project onto PCA components 1–2, 2–3, 3–4, 4–5
4. Apply irreducibility tests:
   - **ε-mixture index** M_ε(f): maximum fraction of points projectable near zero — high = more irreducible (truly multi-dimensional)
   - **Separability index** S(f): minimum mutual information across rotations — high = cannot be factored into independent components
5. Rank clusters by (1 − ε-mixture) × separability; examine top clusters geometrically
6. Check if any clusters correspond to recognisable persona dimensions (helpfulness, dominance, creativity, warmth)

**Key hypothesis:** The Assistant Axis may be the first principal component of a manifold, not the manifold itself — just as the days-of-week feature has an "intensity" first PC and a "circular position" in the second and third PCs. The Assistant may similarly have an "intensity" direction and one or more directional components encoding what *kind* of assistant-ness is expressed.

**What we gain:** Whether "assistant-ness" is a true one-dimensional direction or an irreducible multi-dimensional structure. If it is multi-dimensional, steering along a single axis vector is fundamentally incomplete.

**New code:** `src/analysis/sae_clustering.py`

---

### Experiment 3 — Ising Model on Persona SAE Co-activations

**Method:** Apply the pairwise Ising model from Bhalla et al. (2026) to unsupervisedly discover manifold structure in persona activation space.

**Steps:**
1. Collect SAE feature activation codes z for all role-conditioned rollouts
2. Binarise: s_i = 2·**1**[z_i > 0] − 1
3. Fit pairwise Ising model using pseudolikelihood:

   p(s) ∝ exp(Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i)

   Fields h_i absorb marginal firing rates; couplings J_ij capture direct structural interactions after conditioning out co-occurrence due to superposition
4. Run community detection (Louvain algorithm) on the coupling matrix J to find block-diagonal structure
5. Each block = a group of SAE features jointly representing one manifold
6. For the capture regime: positive couplings within the block (features co-activate together)
   For the tiling/shattering regime: negative couplings (features mutually exclude, each covers a local patch)
7. Decode through each block's atoms and visualise what region of persona space it covers

**What we gain:** Unsupervised discovery of which SAE latents jointly tile the same persona manifold, without needing labels. This reveals whether "the assistant persona" is one manifold, or several overlapping manifolds (e.g., one for professional competence, one for warmth, one for harmlessness) — with direct implications for which dimensions can be independently capped.

**New code:** `src/analysis/ising_manifold.py`

---

### Experiment 4 — Is the Assistant Axis Actually Curved?

**Method:** Test directly whether the trajectory from "non-assistant" to "assistant" in activation space is a straight line or a curve.

**Steps:**
1. Take 50 roles sorted by their Assistant Axis projection (most to least assistant-like), extract middle-layer activations
2. Collect activations at many steering strengths along the axis (α = −3, −2, −1, 0, +1, +2, +3)
3. Apply PCA to all these vectors together; plot trajectory in 3D PCA space
4. Apply SMDS with `linear` vs. `semicircular` vs. `circular` distance functions to characterise the trajectory geometry
5. Measure: when we add the Assistant Axis vector (current steering/capping), does the resulting trajectory follow the best-fit manifold, or does it depart from it?
6. If curved: identify the tangent directions at each point on the manifold — these are the geometry-aware steering directions that Sarfati et al. show preserve the belief family

**What we gain:** Direct answer to whether activation capping is on-manifold (safe) or off-manifold (produces unintended coupled changes). If the trajectory is curved, capping must be redesigned with geometry-aware methods — for example, replacing the fixed direction v with a locally-adapted tangent vector.

**This is the most urgent experiment** — the answer changes whether the rest of the codebase is well-founded.

---

### Experiment 5 — Role×Trait Combinations as Hierarchical Manifolds (MP-SAE)

**Method:** Apply Matching Pursuit SAE (Costa et al., NeurIPS 2025) to the role×trait combination activations.

**Steps:**
1. The role×trait combination matrix (30 roles × 30 goal-traits) from `src/extraction/combination_vectors.py` has a natural hierarchical structure: role dimension and goal-trait dimension should be conditionally orthogonal
2. Train both standard SAE and MP-SAE on the combination activation matrix
3. Compare: does standard SAE conflate role and trait dimensions into shared latents? Does MP-SAE separate them via sequential residual-guided encoding?
4. Check: do the MP-SAE features for "humanitarian trait" form a consistent manifold across different roles (architect+humanitarian, historian+humanitarian, chef+humanitarian)?
5. Conditional orthogonality test: features within the same hierarchical level should be quasi-orthogonal; features across levels (role vs. trait) should be conditionally orthogonal (D_i^T D_j = 0 when levels differ)

**What we gain:** Validation that the role×trait factorisation used in the terminal goal subspace work is geometrically sound. If standard SAEs conflate role and trait, the goal subspace analysis built on them is confounded — MP-SAE would be needed upstream.

---

## How This Changes the Existing Codebase

| Existing code | Current assumption | What manifold analysis reveals / requires |
|---|---|---|
| `compute_assistant_axis()` | Single direction captures the axis | May be first PC of a curved manifold; full manifold needs ≥2 dimensions |
| `activation_cap()` Equation 1 | Steering along v stays on-manifold | Off-manifold movement produces unintended coupled changes (Sarfati et al.) |
| `compute_principal_angles()` | Goal/role subspaces are flat | Subspaces may themselves be curved; principal angles between curved subspaces require different treatment |
| Role vector extraction | Each role = one point | Each role = distribution of points on a manifold; 1,200 rollouts are samples from that manifold |
| `plot_persona_space_pca()` | PCA projection is sufficient | SMDS needed to test specific geometric hypotheses and identify best-fitting geometry |
| `build_capping_hooks()` | Fixed direction v at each layer | May need locally-adapted tangent vector if manifold is curved |

**Priority order:**
1. **Experiment 4** (is the axis curved?) — changes whether the rest is well-founded
2. **Experiment 2** (SAE clustering) — reveals true dimensionality of persona features
3. **Experiment 1** (SMDS geometry test) — maps which role subsets are linear vs. curved
4. **Experiment 3** (Ising model) — unsupervised manifold discovery without SAE assumptions
5. **Experiment 5** (MP-SAE for role×trait) — validates goal subspace factorisation

---

## Key Papers

| Paper | Venue | Key contribution for this project |
|---|---|---|
| Bhalla et al., "Do SAEs Capture Concept Manifolds?" | Goodfire, 2026 | Ising model on co-activations for unsupervised manifold discovery; dilution regime explains why SAE features tile rather than capture |
| Engels et al., "Not All Language Model Features Are Linear" | ICLR 2025 | SAE clustering by cosine similarity finds irreducible circular features (days, months); ε-mixture + separability indices; causal patching validates |
| Tiblias et al., "Shape Happens" | TMLR 2026 | SMDS for hypothesis-driven geometry testing; stress metric ranks circular vs. linear vs. cluster fit |
| Sarfati et al., "The Shape of Beliefs" | Goodfire, 2026 | Beliefs are curved manifolds; linear steering moves off-manifold; geometry-aware (linear field probe) methods work better |
| Costa et al., "From Flat to Hierarchical: MP-SAE" | NeurIPS 2025 | MP-SAE captures conditionally orthogonal hierarchical concepts that standard SAEs miss |
| Michaud et al., "SAE Scaling with Feature Manifolds" | Goodfire/MIT, 2025 | Pathological SAE scaling regime when manifolds dominate; β < α means SAEs tile common manifolds instead of finding rare features |
