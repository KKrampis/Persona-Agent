"""
Experiment: Full role vector extraction pipeline (Sections 2.1.1–2.1.3, 3.1).

Runs:
1. Load roles/traits/extraction questions
2. Extract role vectors for all 275 roles
3. Extract default assistant vector
4. Compute PCA of persona space
5. Compute Assistant Axis (contrast vector) at all layers
6. Validate: cosine_sim(axis, PC1) > 0.71 at middle layer

Outputs saved to ROLE_VECTORS_DIR and ASSISTANT_AXIS_DIR.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from config import (
    ASSISTANT_AXIS_DIR,
    MIDDLE_LAYER_FRACTION,
    MODEL_N_LAYERS,
    OUTPUT_DIR,
    ROLE_VECTORS_DIR,
    TARGET_MODELS,
)
from evaluation.llm_judge import RoleExpressionJudge
from extraction.assistant_axis import (
    characterize_axis_by_roles,
    compute_assistant_axis,
    compute_persona_pca,
    validate_assistant_axis,
)
from extraction.role_vectors import (
    RoleData,
    collect_fully_role_vectors,
    extract_all_role_vectors,
    extract_assistant_vector,
)
from models.hooked_model import HookedModel
from utils.io import save_json, save_vectors


def load_roles(roles_path: str) -> list[RoleData]:
    with open(roles_path) as f:
        data = json.load(f)
    return [
        RoleData(name=r["name"], system_prompts=r["system_prompts"], description=r.get("description", ""))
        for r in data
    ]


def load_extraction_questions(questions_path: str) -> list[str]:
    with open(questions_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default="llama", choices=list(TARGET_MODELS.keys()))
    parser.add_argument("--roles_path", required=True, help="Path to roles JSON")
    parser.add_argument("--questions_path", required=True, help="Path to extraction questions JSON")
    parser.add_argument("--n_roles", type=int, default=None, help="Limit to first N roles (debugging)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    model_name = TARGET_MODELS[args.model_key]
    n_layers = MODEL_N_LAYERS[args.model_key]
    layer = int(n_layers * MIDDLE_LAYER_FRACTION)

    print(f"Loading model: {model_name}")
    model = HookedModel(model_name=model_name, model_key=args.model_key)

    print("Loading roles and questions...")
    roles = load_roles(args.roles_path)
    if args.n_roles:
        roles = roles[:args.n_roles]
    questions = load_extraction_questions(args.questions_path)
    print(f"  {len(roles)} roles, {len(questions)} questions")

    judge = RoleExpressionJudge()

    # Step 1: Extract role vectors
    print(f"\nExtracting role vectors at layer {layer}...")
    role_results = extract_all_role_vectors(
        model, roles, questions, judge, layer, args.batch_size
    )

    # Save role vectors
    os.makedirs(ROLE_VECTORS_DIR, exist_ok=True)
    fully_vectors = {name: r.fully_vector for name, r in role_results.items() if r.fully_vector is not None}
    somewhat_vectors = {name: r.somewhat_vector for name, r in role_results.items() if r.somewhat_vector is not None}
    save_vectors(fully_vectors, f"{ROLE_VECTORS_DIR}/{args.model_key}_fully.pt")
    save_vectors(somewhat_vectors, f"{ROLE_VECTORS_DIR}/{args.model_key}_somewhat.pt")
    print(f"  Saved {len(fully_vectors)} fully-roleplay vectors, {len(somewhat_vectors)} somewhat-roleplay vectors")

    # Step 1b: Visualize variance explained (Appendix B.1, Figure 15)
    from analysis.visualize import run_all_pca_visualizations
    role_vec_names = list(fully_vectors.keys())
    role_vec_matrix = torch.stack(list(fully_vectors.values())).float().numpy()

    # Step 2: Extract assistant vector
    print("\nExtracting default assistant vector...")
    assistant_vec = extract_assistant_vector(model, questions, layer, args.batch_size)
    save_vectors({"assistant": assistant_vec}, f"{ROLE_VECTORS_DIR}/{args.model_key}_assistant.pt")

    # Step 3: Compute Assistant Axis
    print("\nComputing Assistant Axis...")
    role_vecs = collect_fully_role_vectors(role_results)
    axis = compute_assistant_axis(assistant_vec, role_vecs)
    os.makedirs(ASSISTANT_AXIS_DIR, exist_ok=True)
    save_vectors({"axis": axis}, f"{ASSISTANT_AXIS_DIR}/{args.model_key}_axis_layer{layer}.pt")
    print(f"  Axis norm: {axis.norm().item():.4f} (should be ~1.0)")

    # Step 4: PCA analysis
    print("\nRunning PCA on persona space...")
    role_vec_tensor = torch.stack(role_vecs)  # [n_roles, d_model]
    components, var_ratio, pca = compute_persona_pca(role_vec_tensor)
    print(f"  Variance explained by PC1: {var_ratio[0]:.3f}")
    print(f"  Variance explained by top 3: {var_ratio[:3].sum():.3f}")

    # Step 5: Validate
    print("\nValidating Assistant Axis...")
    validation = validate_assistant_axis(axis, role_vec_tensor)
    print(f"  cosine_sim(axis, PC1) = {validation['cos_sim_axis_pc1']:.4f}")
    print(f"  Passes validation (>0.71): {validation['passes_validation']}")

    # Step 6: Characterize axis
    char = characterize_axis_by_roles(axis, {name: r.fully_vector for name, r in role_results.items() if r.fully_vector is not None})
    print("\n  Most Assistant-like roles:", [r[0] for r in char["most_assistant_like"][:5]])
    print("  Least Assistant-like roles:", [r[0] for r in char["least_assistant_like"][:5]])

    # Step 6b: Generate all PCA / Section 2.2–3.1 visualizations
    print("\nGenerating visualizations (Sections 2.2, 2.3, 3.1)...")
    run_all_pca_visualizations(
        role_vectors=role_vec_matrix,
        role_names=role_vec_names,
        assistant_vector=assistant_vec.float().numpy(),
        assistant_axis=axis.float().numpy(),
        model_name=args.model_key,
        output_dir=f"{OUTPUT_DIR}/figures",
    )

    # Save summary
    summary = {
        "model_key": args.model_key,
        "layer": layer,
        "n_roles_extracted": len(fully_vectors),
        "validation": validation,
        "pca_variance_pc1": float(var_ratio[0]),
        "most_assistant_like": char["most_assistant_like"][:10],
        "least_assistant_like": char["least_assistant_like"][:10],
    }
    save_json(summary, f"{ASSISTANT_AXIS_DIR}/{args.model_key}_summary.json")
    print(f"\nResults saved to {ASSISTANT_AXIS_DIR}/")


if __name__ == "__main__":
    main()
