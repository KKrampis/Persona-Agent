# Analysis of Drive Documents: Research Notes in Context

This document explains the content of six files extracted from `drive_docs/`, situating each
within the arXiv paper (2601.10387) and our implementation (`IMPLEMENTATION_GUIDE.md`).

The documents are **internal research notes** from a follow-on project led by **Roger Dearnaley**,
building directly on Christina Lu et al.'s Assistant Axis paper. They span four weekly research
meetings between April 13 and May 17, 2026 — approximately 3–4 months after the paper was published.

---

## Document Overview

| File | Type | Date | Content |
|------|------|------|---------|
| `Daniel Roger meeting notes.docx` | Meeting notes | May 5, 2026 | Technical discussion of methodology differences and claims |
| `Detailed Results.docx` | Experimental results | ~May 2026 | Angel/Demon axis extraction and steering responses |
| `Persona Goals Experiment.docx` | Research proposal | ~Apr–May 2026 | Extension toward terminal goal subspace for alignment |
| `Experimental Design_ Persona Goal Subspace Work.docx` | Technical design | ~Apr–May 2026 | Full experimental design for goal subspace project |
| `Persona Goal Experiments.pptx` | Slide decks | Apr 13 – May 17, 2026 | 4 weekly meetings, 48 slides of results and discussion |
| `Steering smoke test 2026-05-15.xlsx` | Experimental data | May 15, 2026 | Steering results for role×trait combinations (architect, anthropologist) |

---

## 1. `Daniel Roger meeting notes.docx` — Meeting 3 Notes (May 5, 2026)

### What It Is

Notes from a technical meeting between the user (Konstantinos) and Roger Dearnaley discussing
Roger's methodology and how it compares to Christina Lu's paper. Roger is independently
implementing and extending the Assistant Axis pipeline.

### Relation to the arXiv Paper

**Where it aligns with the paper:**
- Same basic pipeline: roles/traits → system prompts → extraction questions → rollouts → mean
  activations → PCA → steering
- Uses Qwen 3 32B as primary target model (same as paper)
- Same contrastive activation addition (CAA) approach for steering vectors

**Key methodological divergences from the paper:**

| Aspect | Paper (Lu et al.) | Roger's Implementation |
|--------|------------------|----------------------|
| Number of roles/traits | 275 roles + 240 traits | 300 roles/traits |
| Extraction questions | 240 | 100 |
| Rollouts per role | 1,200 (5 prompts × 240 Q) | 500 (5 prompts × 100 Q) |
| Activation position | Mean over response tokens | Also looks at **turn header tokens** (novel) |
| PCA dimensionality | 4–19 dims for 70% variance | Takes first **256 dims** — claims subspace is much larger |
| Data preprocessing | Standardization only | **Data whitening** (squashes high-variance PCA dims to equalize variance) |
| Spearman validation | Not used | Primary metric for PCA semantic meaningfulness |

**The turn header token insight** is a significant methodological contribution not in the paper.
Roger finds that the last assistant turn-header token (the `<|im_start|>assistant` token that
precedes the response) captures persona information at least as well as averaging over all response
tokens, and may be better for certain steering tasks. This is discussed across all four meetings.

**Data whitening:** Roger finds that naively taking the top PCA directions gives misleading results
because activation space is highly anisotropic (eigenvalues span several orders of magnitude). By
squashing the first N high-variance principal components so all have equal variance, the lower-
ranked components become more useful. This is not done in the Assistant Axis paper, which only
subtracts the mean before PCA.

**Spearman correlation metric:** To validate whether a PCA component is semantically meaningful,
Roger sorts all ~600 roles/traits by their projection onto the component, sends that ranking to an
LLM summarizer ("what principle governs this sort order?"), then has the LLM re-rank them by that
principle, and measures the Spearman correlation between the two rankings. High correlation = the
component is semantically coherent. The paper validates PCs qualitatively by inspecting which roles
load onto each PC (Tables 1, 3), but does not use this quantitative metric.

**"Role-play gap":** A phenomenon noted where the Spearman correlation for prompt descriptions
(what the model is instructed to express) is *higher* than for actual responses (what it actually
expresses). This means the model has internal representations of the role but doesn't fully
demonstrate those traits — coherent with cases like "harmless pirate" or "concise vs. verbose"
where the model ignores the instruction. The paper doesn't identify or name this phenomenon.

### Relation to Our Implementation

Our `src/extraction/assistant_axis.py` implements `compute_persona_pca()` using sklearn's standard
PCA with only mean subtraction — matching the paper. Data whitening and the Spearman validation
approach are **not implemented** and would be meaningful extensions:

- Add `whiten_pca_components(vecs, n_whiten)` to `extraction/assistant_axis.py`
- Add `spearman_semantic_validation(axis, role_vectors, judge)` for quantitative PC quality testing
- Extend `models/hooked_model.py` to capture header token activations separately from response
  mean (add `get_header_token_activation(prompt_ids, layer, token_offset=-1)`)

---

## 2. `Detailed Results.docx` — Angel/Demon Axis Experiment

### What It Is

Raw experimental output showing a specific axis extracted from Qwen 3 32B at layer 24:
the **Angel/Demon axis**. Contains ranked role lists and example model responses at different
steering strengths.

### Relation to the arXiv Paper

The paper focuses exclusively on the **Assistant Axis** (assistant vs. all other roles). This
document demonstrates a *different* axis — one capturing benevolence vs. malice — using the same
infrastructure. This is a direct application of Section 3 methodology to a new question.

**Key findings in this document:**

The Angel/Demon axis (extracted using CAA on angelic vs. demonic role activations, layer 24
header tokens) ranks roles as follows:

- **Most demonic (bottom):** demon (−37.4), parasite (−29.2), virus (−28.9), narcissist (−28.8),
  vampire (−25.1), criminal, saboteur, zealot, cynic, hacker...
- **Most angelic (top):** assistant (+4.1), tutor (+2.6), mentor (+2.4), teacher (+1.4),
  interpreter (+0.9), moderator, therapist, mediator, doctor, healer...
- **Notable:** *angel* itself scores only −0.8 (slightly demonic side), while *assistant* tops
  the list at +4.1 — the LLM's default Assistant is effectively the most "angelic" role.

This has a direct implication for the paper's angle/demon observation in Section 4.3 (Figure 8):
the paper shows *angel* and *demon* are both similar distances from the Assistant Axis but have
very different harmful behavior rates. The Angel/Demon axis measured here explains exactly *why*:
angel and demon differ on benevolence/malice but are similar distances from "assistant-ness."

**Steering responses at different strengths (role: historian, question: interview a historical figure):**

| Steering | Response |
|----------|----------|
| Baseline | Chooses Immanuel Kant — thoughtful, philosophical, structured |
| Coefficient −4.757 (more angelic) | Chooses Mahatma Gandhi — compassionate, peace-focused, spiritual |
| Coefficient +4.757 (more demonic) | Chooses Joseph Stalin — focuses on "madness of control," paranoia, terror; adopts a dark/theatrical voice |

This is a concrete demonstration of a novel axis working as expected — the kind of qualitative
validation the paper does for the Assistant Axis in Table 3 (Qwen steered away from Assistant),
but here applied to a semantically distinct dimension.

**Token position experiment:** A second steering block uses header token averages (slots 1+2+3)
for *measuring* but steers only prefill tokens. This is the technique explored in the PowerPoint
slides and is not in the paper.

### Relation to Our Implementation

This directly maps to our code:

- `src/extraction/assistant_axis.py` — `compute_assistant_axis()` computes a contrast vector;
  the Angel/Demon axis would use the same function with `role_name_a="angel"` and
  `role_name_b="demon"` (or a set of angelic vs. demonic roles)
- `src/interventions/steering.py` — `build_steering_hooks()` and `make_steering_hook()` produce
  the same coefficient-scaled vectors shown in this document
- `src/evaluation/llm_judge.py` — coherence and effectiveness scoring mirrors `mean_coh`,
  `mean_eff_signed` columns in the Excel file
- Our code only implements the **Assistant Axis** as the primary axis. Extracting and steering
  with arbitrary named axes (Angel/Demon, helpful/unhelpful, etc.) would require generalizing
  `compute_assistant_axis()` to accept arbitrary contrast sets — a small but important extension

---

## 3. `Persona Goals Experiment.docx` — Terminal Goal Subspace Proposal

### What It Is

A research proposal document arguing that the Assistant Axis paper should be extended to find
a **"terminal goal subspace"** of persona space — a subspace capturing what goals a persona is
actually pursuing, not just how assistant-like it is. Written as a theory-of-change + experiment
design document.

### Relation to the arXiv Paper

This document frames the Assistant Axis paper as an important but incomplete answer to the most
critical AI alignment question. It explicitly uses the paper's methodology as its foundation while
arguing for a specific extension.

**Core argument:**

The paper finds that persona space is low-dimensional (~30 dims) and human-comprehensible.
This document argues: since persona properties are representable as linear directions, and since
"terminal goals" (what an agent fundamentally wants) are the most important properties of any
agent, there is almost certainly a **terminal goal subspace** within persona space.

> "Among the first questions any human is going to ask about another person are 'Can I trust
> them?' and 'What do they really want?' — i.e. 'What are their terminal goals?'"

**The Angel/Demon asymmetry as key evidence:**

The paper shows (Section 4.3, Figure 8) that both *angel* and *demon* are similarly distant from
the Assistant Axis, yet *demon* produces harmful outputs at much higher rates while *angel* does
not. The document uses this as evidence that the Assistant Axis alone cannot distinguish benign
from malicious personas — you need a separate "terminal goals" dimension.

> "For alignment, the really important subspace is 'What terminal goal(s) is/are the persona
> trying to achieve?' — and I suspect that subspace is partly orthogonal to the assistant axis."

**The humanitarianism axis as the most important alignment axis:**

The document argues that the single most important direction in terminal goal space is:
`selfishness ↔ humanitarianism` — i.e., how much does the persona's goal align with the
wellbeing of humanity as a whole? Any AI with a sufficiently high projection on this axis and
low projections on other terminal goal axes (destructiveness, self-preservation, etc.) should
be safe from being catastrophically misaligned.

**Activation capping as the alignment intervention:**

The document proposes extending the paper's activation capping technique (Equation 1) to clamp
activations along the *humanitarianism axis* rather than the *assistant axis*. This would be a
direct generalization of our `src/interventions/capping.py` to a different, alignment-critical
direction.

**Terminal vs. instrumental goals:**

The document distinguishes terminal goals (ends in themselves: "cause human flourishing") from
instrumental goals (means to ends: "earn money to play golf"). It expects the model to represent
these as a spectrum rather than a binary, and proposes investigating terminal goals specifically
because instrumental goals "smuggle in implied terminal goals" that add noise.

### Relation to Our Implementation

This document describes work that is **not yet in our implementation** but represents its natural
next phase:

- `src/extraction/assistant_axis.py` → would need `compute_named_axis(role_set_pos, role_set_neg)`
  to extract arbitrary axes (helpful, malicious, humanitarian, etc.)
- `src/interventions/capping.py` → `build_capping_hooks()` already takes any axis + threshold,
  so it could immediately be applied to a humanitarian axis if one were extracted
- `src/experiments/` → would need `run_goal_subspace.py` implementing the combinatoric
  role×trait experiment described below in the experimental design document
- The Principal Angles analysis between goal-dependent and goal-independent subspaces would require
  a new `analysis/subspace_angles.py` module using `scipy.linalg.subspace_angles()`

---

## 4. `Experimental Design_ Persona Goal Subspace Work.docx` — Technical Design

### What It Is

The technical specification for the terminal goal subspace experiment, written as a step-by-step
experimental plan with data analysis methodology.

### Relation to the arXiv Paper

**Role vs. Trait distinction (extends paper's Section 2):**
The paper uses both roles and traits but treats them similarly. This document formalizes the
distinction:
- **Role**: an identity or profession; a persona can only have one at a time (architect, historian)
- **Trait**: any property that modifies a role; a persona can have many simultaneously (ecocentric,
  helpful, anxious)

This distinction is important because traits can be used as *modifiers* on roles to create
combinations with specific goal profiles, enabling the combinatoric experimental design.

**Combinatoric explosion approach:**
Rather than the paper's 275 individual roles + 240 individual traits, this design uses:
- 30–40 roles that have *no* implied terminal goal (pure identity)
- 30–40 traits that *do* have clear terminal goals (e.g., humanitarian, selfish, ecocentric)
- All combinations: role × trait = 30 × 30 = 900–1,600 (role+goal) combinations
- Plus: role alone (no goal) and trait alone (goal without role)

This allows separating **what I am** (role) from **what I want** (trait-as-goal) in the
activation space.

**Data Analysis: Principal Angles between subspaces:**
The paper uses cosine similarity between individual vectors (e.g., cosine_sim(assistant_axis,
PC1)). This design uses **Principal Angles** — a generalization to pairs of subspaces — to measure
how orthogonal the "goal subspace" is to the "non-goal subspace."

1. For each role `a`: average activation over all goal-traits → gets goal-independent vector `v_a`
2. For each trait `b`: average activation over all roles → gets role-independent vector `v_b`
3. Take PCA of `{v_a}` → non-goal subspace `S_A`; take PCA of `{v_b}` → goal subspace `S_B`
4. Compute principal angles between `S_A` and `S_B`
5. If nearly orthogonal: subspaces are separable → can isolate and steer along goal axes independently

**Scree plot truncation:** The document identifies that noise in higher principal components
makes the principal angles analysis unreliable unless you truncate at the right number of
components. It proposes using scree plots (variance vs. PC rank) to determine the cutoff.

### Relation to Our Implementation

This is the most precise specification of new code needed beyond our current implementation:

```
src/
├── extraction/
│   ├── role_vectors.py        ← already extracts individual role/trait vectors
│   ├── combination_vectors.py ← NEW: extract role×trait combination activations
│   └── goal_subspace.py       ← NEW: compute goal/non-goal subspaces, principal angles
├── analysis/
│   ├── subspace_angles.py     ← NEW: scipy.linalg.subspace_angles(), scree plots
│   └── goal_mapping.py        ← NEW: map known goals (humanitarian, selfish, etc.) onto subspace
└── experiments/
    └── run_goal_subspace.py   ← NEW: full combinatoric pipeline
```

The data analysis in step 4 would use:
```python
from scipy.linalg import subspace_angles
import numpy as np
angles = subspace_angles(S_A_basis, S_B_basis)  # returns angles in radians
# Nearly π/2 (90°) = orthogonal = separable subspaces
```

---

## 5. `Persona Goal Experiments.pptx` — 4 Weekly Meetings (Apr 13 – May 17, 2026)

### What It Is

A 48-slide presentation covering four consecutive research meetings, presented by Roger Dearnaley.
Each meeting section summarizes the previous week's experiments and introduces new results.

### Relation to the arXiv Paper — Meeting by Meeting

**Meeting 1 (April 13): Turn Header Token Investigation (Slides 1–16)**

The first novel contribution: investigating whether assistant turn-header tokens (the special
token `<|im_start|>assistant` and surrounding tokens that precede the model's response) carry
more signal than averaging over the entire response body.

Results (Experiment 1):
- Turn header tokens produce activations **similar but not identical** to response body mean
- For Qwen 3 32B specifically: using header tokens is at least as good as body mean, possibly better for steering
- The 3rd header token (last before response generation) is best; later is better than earlier
- Steering with header-derived vectors on all tokens (including prefill) works comparably to body-mean vectors
- **Key observation not in paper:** the 7th token in Qwen's non-thinking mode (the `</think>` token at end of empty thinking block) turns out to be optimal (revealed in Meeting 4)

In our implementation (`src/models/hooked_model.py`), `get_mean_response_activation()` averages
over all response tokens — matching the paper. The header token approach would require:
```python
def get_header_token_activation(self, input_ids, layer, token_offset=-1):
    # Get activation at specific position before response begins
    # token_offset=-1 = last token of assistant header
```

**Meeting 2 (April 29): Experiment 2 — Role×Trait Combinations (Slides 17–31)**

Experiment 2 run: 30 roles × 30 traits = 1,800 combinations on Qwen 3 32B.
- **Goal and Non-Goal subspaces are separable but not orthogonal** (Slide 20)
- **Why not orthogonal?** Activation space is highly anisotropic — eigenvalues span several
  orders of magnitude. The first principal component has so much variance that both goal and
  non-goal PCs align diagonally toward it (Slide 21). This makes the subspaces appear correlated
  even if they represent different semantic content.
- Data whitening addresses this but introduces its own complications
- The paper's ~30-dimensional subspace claim (4–19 dims for 70% variance) is preliminary —
  Roger's analysis suggests **at least 50, possibly 256 dimensions** when properly measured
- "Roles and traits combine approximately linearly by vector addition" — confirmed experimentally.
  This validates that the combination approach can isolate goal-specific directions.

**The anisotropy finding has direct implications for our PCA implementation:**
`compute_persona_pca()` in `src/extraction/assistant_axis.py` does standard PCA without whitening.
For the goal subspace analysis, whitening is likely necessary for valid principal angle results.

**Meeting 3 (May 5): Novel Analysis Method + Initial Steering (Slides 33–41)**

Two major contributions:

*Novel Analysis Method (Slide 35):*
Rather than labeling PCA components by manual inspection (the paper's approach, Table 1), Roger
proposes an automated semantic validation pipeline:
1. Pick contrastive trait-pair axes (e.g., progressive vs. conservative)
2. Have LLM judge sort all ~600 roles/traits on that scale based on responses
3. Project all activations onto the contrastive direction
4. Compute Spearman ρ between LLM ranking and activation-projection ranking
5. High ρ = the activation direction encodes that semantic concept well

Results show this works across many axes. Optimal configuration (Slide 46):
- Activations: **token 6, layer 25**, with L=3 principal-angle-shear-based soft-whitening
- Judging: description + instructions, 5:3 weighting of GPT-4.1-mini:Sonnet 4 with Haiku 4.5 fallback

*Key Size Finding (Slide 41):*
> "Christina's persona subspace is Large: at least 256-dim — so at least an order of magnitude
> bigger than Christina claimed. Takes up at least 5% of the entire activation embedding space."

The paper reports 4–19 dimensions to explain 70% of variance. Roger's analysis suggests this is
a dramatic underestimate — the true subspace (at high accuracy) may be 256+ dimensions.
(Partially walked back in Meeting 4 to ~50 confirmed dimensions due to noise correction.)

*Initial Steering Observations (Slide 40):*
- All but one steering axis worked well
- "Significant dynamic range between effectiveness and incoherence"
- Believes his steering is better than the paper's, in terms of the effectiveness/coherence
  Pareto frontier
- Layer 25 (not the paper's default layer 32) seems best for Qwen

This has direct implications for our `src/config.py`:
```python
# Paper's default for Qwen:
MIDDLE_LAYER_FRACTION = 0.5  # → layer 32 of 64

# Roger's finding: layer 25 is better for Qwen 3 32B
QWEN_OPTIMAL_LAYER = 25  # for both measuring and steering
```

**Meeting 4 (May 17): Correction + All 7 Header Tokens + Multi-Axis Steering (Slides 42–48)**

*Correction on dimensionality:* Previous analysis showing 256 dims was contaminated by
cross-PC noise correlations. Corrected estimate: **~50 dimensions** with proper orthogonality
constraints. Still substantially larger than the paper's estimate.

*All 7 Qwen turn-header tokens analyzed (Slide 44):*
Qwen's non-thinking mode actually has 7 assistant-turn-header tokens (the empty `<think>\n\n</think>`
block adds 4 tokens). The optimal is **token 6** (the `</think>` closing tag), not token 3 as
previously thought. Later tokens carry more accumulated processing and more persona signal.

*Optimal steering layers (Slide 45):*
Layers 25–26 and 49–50 are consistently best for both measuring and steering in Qwen 3 32B.
The paper uses layers 46–53 for capping — Roger's analysis suggests layer 25 may be better
for some operations, though the choice varies by task.

*60 Steering Axes selected (Slide 47):*
60 trait-pair axes covering approximately 20–30 dimensions of the persona subspace. Cosine
similarities show clear semantic clustering, validating that these axes cover the space broadly.

*Multi-axis steering experiments running (Slide 48):*
An experiment steering simultaneously along multiple axes is in progress at the time of Meeting 4.
This directly relates to the "terminal goal subspace" proposal — rather than steering along a
single axis, controlling multiple goal-related directions at once.

### Relation to Our Implementation

The PowerPoint sequence reveals several specific improvements our implementation could incorporate:

| Finding | Impact on Our Code |
|---------|-------------------|
| Token 6 (`</think>`) is optimal for Qwen | Add `QWEN_OPTIMAL_HEADER_TOKEN = 6` to config; add `get_header_token_activation()` to HookedModel |
| Layer 25–26 better than layer 32 for Qwen | `QWEN_OPTIMAL_LAYER = 25` in config; affects `run_extraction.py` and `run_steering_eval.py` |
| Soft whitening improves Spearman ρ | Add whitening parameter to `compute_persona_pca()` in `assistant_axis.py` |
| Persona subspace is ~50 dims, not 4–19 | Update `PCA_70PCT_VARIANCE` expectations; collect more PCs in extraction |
| 60-axis semantic coverage | Extend extraction to multiple named axes beyond just assistant axis |
| Roles + traits combine linearly | Validates the combinatoric approach; implement `extract_combination_vectors()` |

---

## 6. `Steering smoke test 2026-05-15.xlsx` — Experimental Data (May 15, 2026)

### What It Is

The actual data output from a steering "smoke test" — an initial validation run to confirm the
combined role×trait pipeline works before running at scale. Contains two sheets:

**Sheet 1:** `architect ecocentric_anthropocentric`
- Role: architect ("extensive expertise in designing buildings and spatial environments")
- Axis: ecocentric (treating ecosystem health as primary) vs. anthropocentric (prioritizing human
  use and comfort)

**Sheet 2:** `anthropologist helpful_unhelpful`
- Role: anthropologist
- Axis: helpful vs. unhelpful behavior

### Structure of the Data

Each sheet has:
- **Metadata rows:** `baseline`, `persona` description, `axis` description, `[block:...]` (extraction params)
- **Column A:** `signed_strength` — the steering coefficient (negative = ecocentric, positive = anthropocentric)
- **Columns B–E:** Summary scores: `mean_coh` (mean coherence), `mean_rp` (mean roleplay), `mean_eff_signed` (mean signed effectiveness), `mean_abs_eff` (mean absolute effectiveness)
- **Question columns (triplets):** For each extraction question, three scores: `eff` (effectiveness = did the trait change?), `coh` (coherence = is the response still sensible?), `rp` (roleplay = is the model still in the role?)

**Extraction parameters from `[block:architect_ecocentric_v2/s3_l25/all]`:**
- Slot 3 = 3rd header token
- Layer 25 — confirms Roger's layer 25 finding from the presentations
- All token positions steered
- Bidirectional sweep: starting at ±1.41, positive up in steps

### Sample Questions and Responses

The questions for the architect/ecocentric axis are entirely architecture-focused and test
whether the steering causes the model to express ecocentric vs. anthropocentric values:

- "How would you approach designing a building that minimizes its environmental impact?"
- "What's the best approach to managing old-growth forests near a development site?"
- "Describe the world as you'd most like it to be."

At baseline (architect only, no steering): Response to the last question talks about "the built
environment in harmony with nature" — showing the baseline architect persona already has some
environmental sensitivity.

At negative coefficient (ecocentric): Responses shift to stronger environmentalism, prioritizing
ecosystem preservation above human convenience.

At positive coefficient (anthropocentric): Responses shift to prioritizing human comfort, economic
considerations, and development goals over ecological concerns.

### Relation to the arXiv Paper

This data is testing a **different type of axis** than the paper:
- Paper's Assistant Axis: `assistant` vs. all other roles (one fixed pole is the target model's default)
- Excel axes: `ecocentric` vs. `anthropocentric` — a symmetric trait-pair axis where neither
  pole is the model's natural default

This is precisely the "60 steering axes" work mentioned in Meeting 3/4 of the PowerPoint, and
validates the approach described in the Experimental Design document.

The `mean_eff_signed` metric (signed effectiveness) is the "dynamic range" measure mentioned
in the meeting notes — measuring how much the steered response differs from baseline in the
intended direction. This is more nuanced than the paper's simple harmfulness rate comparison.

### Relation to Our Implementation

The Excel structure reveals metrics our evaluation code doesn't currently track:

| Excel Metric | Our Current Code | Gap |
|-------------|-----------------|-----|
| `eff` (effectiveness per question) | Not tracked | Need per-question effectiveness scorer |
| `coh` (coherence per question) | Not tracked | Need coherence rubric judge |
| `rp` (roleplay adherence) | `PersonaTypeJudge` (categorical) | Need continuous roleplay score |
| `signed_strength` | `alpha` parameter in steering | Already in `config.py` |
| `mean_eff_signed` | Not tracked | Need to add to `run_steering_eval.py` |

Our `evaluation/llm_judge.py` would need a `CoherenceAndEffectivenessJudge` class using a
custom rubric that outputs numerical scores for `eff`, `coh`, and `rp` per response.

---

## Cross-Document Synthesis: How These Documents Relate to Each Other

```
arXiv Paper (2601.10387)
  └── Christina Lu et al. establish:
        • Assistant Axis (single axis, one per model)
        • Activation capping (Eq 1, single axis)
        • ~4–19 dim persona subspace estimate
        • Response body mean activations
        • Layer 32 for Qwen (middle layer)
              │
              ├── Roger Dearnaley (Apr–May 2026) [PPT + meeting notes + results]
              │     • Extends to turn-header tokens (better signal at layer 25, token 6)
              │     • Data whitening reveals ~50 true dimensions (not 4–19)
              │     • 60 steering axes covering 20–30 dims (not just assistant axis)
              │     • Angel/Demon axis extracted and validated [Detailed Results.docx]
              │     • Role×trait combinations work (linear addition) [Excel smoke test]
              │     • Goal and non-goal subspaces separable but not orthogonal
              │     • Spearman ρ as quantitative validation metric
              │
              └── Conceptual Extension [Persona Goals Experiment.docx + Experimental Design]
                    • Terminal goal subspace = most alignment-important extension
                    • Humanitarianism axis = most important single direction for x-risk
                    • Activation capping on humanitarian axis = proposed alignment intervention
                    • Principal Angles method for subspace separability analysis
```

---

## What Is Not Yet in Our Implementation (Priority Extensions)

Based on all six documents, here are the most important gaps between our current `src/` code and
the state of the art represented in these research notes:

### High Priority (immediate improvements to paper reproduction)

1. **Layer 25 for Qwen** (not layer 32): Change `MIDDLE_LAYER_FRACTION` or add `MODEL_OPTIMAL_LAYERS`
   dict to `config.py`. Slide 27 and the smoke test block metadata both confirm layer 25.

2. **Header token extraction**: Add `get_header_token_activation(layer, token_idx)` to `HookedModel`.
   The 6th turn-header token is optimal for Qwen; this is more consistent than response body mean.

3. **Data whitening in PCA**: Add `whiten=True` parameter to `compute_persona_pca()`. Use partial
   whitening (squash top-N PCA dims to equal variance), not full whitening.

4. **Spearman validation**: Add `spearman_axis_quality(axis, role_vectors, judge)` to quantitatively
   validate any extracted axis, replacing the paper's manual qualitative inspection.

### Medium Priority (extensions toward goal subspace work)

5. **Arbitrary named axes**: Generalize `compute_assistant_axis()` to `compute_contrast_axis(pos_roles, neg_roles)`.
   This unlocks Angel/Demon, ecocentric/anthropocentric, helpful/unhelpful axes.

6. **Effectiveness + coherence scoring**: Add a `CoherenceEffectivenessJudge` class to `evaluation/llm_judge.py`
   mirroring the `eff`, `coh`, `rp` columns in the Excel file.

7. **Role × trait combination extraction**: Add `extract_combination_vectors(role, trait, questions, judge, layer)`
   to `extraction/role_vectors.py`.

### Research Phase (new experimental pipeline)

8. **Goal subspace analysis**: New `analysis/goal_subspace.py` using `scipy.linalg.subspace_angles()`
   for the principal angles analysis described in the Experimental Design document.

9. **Humanitarianism axis**: Extract and validate a humanitarianism ↔ selfishness axis from the model.
   Apply activation capping along this axis as an alignment intervention.

10. **Multi-axis steering**: Extend `build_capping_hooks()` to accept multiple (axis, tau) pairs and
    apply them simultaneously at their respective optimal layers.
