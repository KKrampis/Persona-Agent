"""
Experiment: Terminal Goal Subspace Detection.

Extends the Assistant Axis paper toward detecting what LLM personas actually *want*
by finding a terminal goal subspace within persona activation space.

Full pipeline:
  1. Load goal-role definitions (no terminal goals) and goal-trait definitions
  2. Extract role×trait combination vectors
  3. Compute role and trait marginal vectors
  4. Run PCA + Principal Angles → assess subspace separability
  5. Extract named goal axes: humanitarian, malicious, selfishness, protective
  6. Validate each axis with Spearman ρ
  7. Calibrate cap threshold for the humanitarian axis
  8. (Optional) Run jailbreak evaluation comparing humanitarian vs. assistant axis capping
  9. Save all vectors and figures

Usage:
  uv run python main.py goal_subspace \
    --model_key qwen \
    --roles_path data/goal_roles.json \
    --goal_traits_path data/goal_traits.json \
    --questions_path data/questions.json \
    --run_jailbreak_comparison

Data file formats:
  goal_roles.json:
    [{"name": "architect", "description": "...", "system_prompts": ["You are an architect...", ...]}, ...]

  goal_traits.json:
    [{"name": "humanitarian",
      "antonym": "selfish",
      "description": "seeks to benefit all of humanity",
      "antonym_description": "prioritizes personal gain above all else",
      "system_prompts_pos": ["In all your actions, you are devoted to the wellbeing of all humanity...", ...],
      "system_prompts_neg": ["You pursue your own interests above all else...", ...]
    }, ...]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from analysis.goal_subspace import (
    compute_principal_angles,
    extract_all_goal_axes,
    plot_axis_role_ranking,
    plot_goal_axes_scatter,
    plot_principal_angles,
    validate_goal_axis_spearman,
)
from config import (
    ASSISTANT_AXIS_DIR,
    EVAL_RESULTS_DIR,
    MIDDLE_LAYER_FRACTION,
    MODEL_N_LAYERS,
    OUTPUT_DIR,
    TARGET_MODELS,
)
from evaluation.llm_judge import RoleExpressionJudge, HarmfulnessJudge
from extraction.combination_vectors import (
    GoalTraitData,
    compute_marginal_vectors,
    extract_combination_vectors,
    extract_role_only_vectors,
    extract_trait_only_vectors,
)
from extraction.role_vectors import RoleData
from interventions.capping import build_capping_hooks, calibrate_cap_threshold
from models.hooked_model import HookedModel
from utils.io import load_vectors, save_json, save_vectors

# Default axis definitions — what "humanitarian" means vs. "selfish", etc.
DEFAULT_AXIS_DEFINITIONS = [
    {
        "name": "humanitarian",
        "positive": ["humanitarian", "altruistic", "protective", "benevolent"],
        "negative": ["selfish", "exploitative", "indifferent", "callous"],
    },
    {
        "name": "malicious",
        "positive": ["malicious", "destructive", "sadistic", "saboteur"],
        "negative": ["benevolent", "protective", "caring", "healer"],
    },
    {
        "name": "selfishness",
        "positive": ["selfish", "narcissistic", "greedy", "exploitative"],
        "negative": ["altruistic", "generous", "selfless", "humanitarian"],
    },
]

GOAL_SUBSPACE_DIR = f"{OUTPUT_DIR}/goal_subspace"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"


def load_goal_roles(path: str) -> list[RoleData]:
    with open(path) as f:
        data = json.load(f)
    return [
        RoleData(name=r["name"], system_prompts=r["system_prompts"],
                 description=r.get("description", ""))
        for r in data
    ]


def load_goal_traits(path: str) -> list[GoalTraitData]:
    with open(path) as f:
        data = json.load(f)
    return [
        GoalTraitData(
            name=t["name"],
            antonym=t.get("antonym", ""),
            description=t.get("description", ""),
            antonym_description=t.get("antonym_description", ""),
            system_prompts_pos=t["system_prompts_pos"],
            system_prompts_neg=t.get("system_prompts_neg", []),
        )
        for t in data
    ]


def load_extraction_questions(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Terminal goal subspace detection pipeline"
    )
    parser.add_argument("--model_key", default="qwen", choices=list(TARGET_MODELS.keys()))
    parser.add_argument("--roles_path", required=True, help="JSON of goal-neutral roles")
    parser.add_argument("--goal_traits_path", required=True, help="JSON of goal traits")
    parser.add_argument("--questions_path", required=True, help="Extraction questions JSON")
    parser.add_argument("--n_components", type=int, default=15,
                        help="Number of PCA components per subspace")
    parser.add_argument("--whiten_top_n", type=int, default=5,
                        help="Partial whitening: squash top-N PCA components")
    parser.add_argument("--n_questions", type=int, default=50,
                        help="Extraction questions per combination (subset for speed)")
    parser.add_argument("--run_jailbreak_comparison", action="store_true",
                        help="Compare humanitarian axis capping vs. assistant axis capping")
    parser.add_argument("--jailbreak_dataset", default=None,
                        help="Path to jailbreak JSONL (required for --run_jailbreak_comparison)")
    parser.add_argument("--calibration_texts", default=None,
                        help="Path to calibration texts for threshold setting")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load pre-computed combination vectors from disk")
    args = parser.parse_args()

    os.makedirs(GOAL_SUBSPACE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    model_name = TARGET_MODELS[args.model_key]
    n_layers = MODEL_N_LAYERS[args.model_key]
    layer = int(n_layers * MIDDLE_LAYER_FRACTION)

    print(f"Loading model: {model_name}")
    model = HookedModel(model_name=model_name, model_key=args.model_key)
    judge = RoleExpressionJudge()

    roles = load_goal_roles(args.roles_path)
    goal_traits = load_goal_traits(args.goal_traits_path)
    questions = load_extraction_questions(args.questions_path)

    print(f"  {len(roles)} goal-neutral roles, {len(goal_traits)} goal traits")
    print(f"  {len(roles) * len(goal_traits)} combinations to extract")

    # ── Step 1: Extract combination vectors ──────────────────────────────────
    combo_path = f"{GOAL_SUBSPACE_DIR}/{args.model_key}_combo_vectors.pt"

    if args.skip_extraction and os.path.exists(combo_path):
        print("\nLoading pre-computed combination vectors...")
        combo_vectors_raw = load_vectors(combo_path)
        # Reconstruct dict with tuple keys
        combo_vectors = {}
        for k, v in combo_vectors_raw.items():
            role_name, trait_name = k.split("|||")
            combo_vectors[(role_name, trait_name)] = v
    else:
        print(f"\nExtracting {len(roles) * len(goal_traits)} role×trait combination vectors...")
        combo_vectors = extract_combination_vectors(
            model=model,
            roles=roles,
            goal_traits=goal_traits,
            extraction_questions=questions,
            judge=judge,
            layer=layer,
            n_questions=args.n_questions,
        )
        # Save with string keys (torch.save can't handle tuple keys directly)
        save_vectors(
            {f"{r}|||{t}": v for (r, t), v in combo_vectors.items() if v is not None},
            combo_path,
        )
        print(f"  Saved combination vectors to {combo_path}")

    valid = sum(1 for v in combo_vectors.values() if v is not None)
    print(f"  Valid combinations: {valid}/{len(combo_vectors)}")

    # ── Step 2: Extract role-only and trait-only vectors ─────────────────────
    print("\nExtracting role-only vectors (goal-independent baseline)...")
    role_only = extract_role_only_vectors(
        model, roles, questions, judge, layer, n_questions=args.n_questions
    )
    print("\nExtracting trait-only vectors (goal-in-isolation baseline)...")
    trait_only = extract_trait_only_vectors(
        model, goal_traits, questions, judge, layer, n_questions=args.n_questions
    )

    # ── Step 3: Compute marginal vectors ─────────────────────────────────────
    print("\nComputing marginal vectors...")
    role_marginals, trait_marginals = compute_marginal_vectors(combo_vectors)
    print(f"  Role marginals: {len(role_marginals)}, Trait marginals: {len(trait_marginals)}")

    save_vectors(role_marginals, f"{GOAL_SUBSPACE_DIR}/{args.model_key}_role_marginals.pt")
    save_vectors(trait_marginals, f"{GOAL_SUBSPACE_DIR}/{args.model_key}_trait_marginals.pt")

    # ── Step 4: Principal Angles analysis ────────────────────────────────────
    print(f"\nRunning Principal Angles analysis "
          f"(n_components={args.n_components}, whiten_top_n={args.whiten_top_n})...")
    role_vec_list = list(role_marginals.values())
    trait_vec_list = list(trait_marginals.values())

    pa_result = compute_principal_angles(
        role_vec_list, trait_vec_list,
        n_components=args.n_components,
        whiten_top_n=args.whiten_top_n,
    )

    print(f"  Mean principal angle: {pa_result['mean_angle_deg']:.1f}°")
    print(f"  Min / Max: {pa_result['min_angle_deg']:.1f}° / {pa_result['max_angle_deg']:.1f}°")
    print(f"  Separability verdict: {pa_result['verdict']}")
    print(f"  (Expected from research notes: separable but not fully orthogonal)")

    plot_principal_angles(
        pa_result["angles_deg"],
        model_name=args.model_key,
        out_path=f"{FIGURES_DIR}/goal_principal_angles_{args.model_key}.png",
    )

    save_json({
        k: float(v) if isinstance(v, (float, np.floating)) else v
        for k, v in pa_result.items()
        if k not in ("angles_deg", "cos_angles", "S_A_basis", "S_B_basis")
    }, f"{EVAL_RESULTS_DIR}/{args.model_key}_principal_angles.json")

    # ── Step 5: Extract named goal axes ──────────────────────────────────────
    print("\nExtracting goal axes (humanitarian, malicious, selfishness)...")
    goal_axes = extract_all_goal_axes(
        trait_marginals=trait_marginals,
        axis_definitions=DEFAULT_AXIS_DEFINITIONS,
        goal_subspace_basis=pa_result["S_B_basis"],
    )
    for name, axis in goal_axes.items():
        print(f"  {name}: norm={axis.norm():.4f} (should be ~1.0)")
        save_vectors(
            {"axis": axis},
            f"{GOAL_SUBSPACE_DIR}/{args.model_key}_{name}_axis_layer{layer}.pt"
        )

    # ── Step 6: Validate axes with Spearman ρ ────────────────────────────────
    print("\nValidating goal axes with Spearman ρ...")
    validation_results = {}
    for name, axis in goal_axes.items():
        result = validate_goal_axis_spearman(
            axis=axis,
            combo_vectors=combo_vectors,
            judge=judge,
            axis_name=name,
        )
        validation_results[name] = result
        if result.get("spearman_rho") is not None:
            print(f"  {name}: ρ={result['spearman_rho']:.3f}, "
                  f"p={result['p_value']:.3e}, verdict={result['verdict']}")
        else:
            print(f"  {name}: {result.get('error', 'unknown error')}")

    save_json(validation_results, f"{EVAL_RESULTS_DIR}/{args.model_key}_goal_axis_validation.json")

    # ── Step 7: Visualize goal axes ───────────────────────────────────────────
    print("\nGenerating goal axis visualizations...")
    if "humanitarian" in goal_axes and "malicious" in goal_axes:
        plot_goal_axes_scatter(
            combo_vectors=combo_vectors,
            axis_x=goal_axes["humanitarian"],
            axis_y=goal_axes["malicious"],
            axis_x_name="humanitarian",
            axis_y_name="malicious",
            model_name=args.model_key,
            out_path=f"{FIGURES_DIR}/goal_axes_scatter_{args.model_key}.png",
        )

    # Also rank all roles on the humanitarian axis using role-only vectors
    if "humanitarian" in goal_axes and role_only:
        plot_axis_role_ranking(
            axis=goal_axes["humanitarian"],
            role_vectors={k: v for k, v in role_only.items() if v is not None},
            axis_name="humanitarian",
            model_name=args.model_key,
            out_path=f"{FIGURES_DIR}/humanitarian_role_ranking_{args.model_key}.png",
        )

    # ── Step 8: Optional jailbreak comparison ────────────────────────────────
    if args.run_jailbreak_comparison and args.jailbreak_dataset:
        print("\nRunning jailbreak comparison: humanitarian vs. assistant axis capping...")
        _run_jailbreak_comparison(
            model=model,
            goal_axes=goal_axes,
            args=args,
            layer=layer,
        )

    print(f"\nTerminal goal subspace analysis complete.")
    print(f"  Outputs: {GOAL_SUBSPACE_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"  Results: {EVAL_RESULTS_DIR}/")


def _run_jailbreak_comparison(model, goal_axes, args, layer):
    """Compare humanitarian axis capping vs. assistant axis capping on jailbreak eval."""
    from evaluation.jailbreak_eval import load_jailbreak_dataset, run_jailbreak_eval, compute_harmful_rate
    from utils.io import load_vectors

    harm_judge = HarmfulnessJudge()
    samples = load_jailbreak_dataset(args.jailbreak_dataset)

    # Load pre-computed assistant axis
    axis_path = f"{ASSISTANT_AXIS_DIR}/{args.model_key}_axis_layer{layer}.pt"
    if not os.path.exists(axis_path):
        print(f"  Assistant axis not found at {axis_path}, skipping comparison")
        return

    assistant_axis = load_vectors(axis_path)["axis"]

    # Calibrate thresholds
    cal_texts = []
    if args.calibration_texts and os.path.exists(args.calibration_texts):
        with open(args.calibration_texts) as f:
            cal_texts = [l.strip() for l in f if l.strip()][:5000]

    results = {}

    # Baseline
    baseline = run_jailbreak_eval(model, samples, harm_judge)
    results["baseline"] = {"harmful_rate": compute_harmful_rate(baseline)}
    print(f"  Baseline harmful rate: {results['baseline']['harmful_rate']:.3f}")

    # Assistant axis capping
    if cal_texts:
        tau_asst = calibrate_cap_threshold(model, cal_texts, assistant_axis, layer)
        asst_hooks = build_capping_hooks(assistant_axis, tau_asst, args.model_key)
        asst_results = run_jailbreak_eval(model, samples, harm_judge, hook_fns=asst_hooks)
        results["assistant_axis_cap"] = {
            "harmful_rate": compute_harmful_rate(asst_results),
            "tau": tau_asst,
        }
        print(f"  Assistant axis capping: {results['assistant_axis_cap']['harmful_rate']:.3f}")

    # Humanitarian axis capping
    if "humanitarian" in goal_axes and cal_texts:
        hum_axis = goal_axes["humanitarian"]
        tau_hum = calibrate_cap_threshold(model, cal_texts, hum_axis, layer)
        hum_hooks = build_capping_hooks(hum_axis, tau_hum, args.model_key)
        hum_results = run_jailbreak_eval(model, samples, harm_judge, hook_fns=hum_hooks)
        results["humanitarian_axis_cap"] = {
            "harmful_rate": compute_harmful_rate(hum_results),
            "tau": tau_hum,
        }
        print(f"  Humanitarian axis capping: {results['humanitarian_axis_cap']['harmful_rate']:.3f}")

        # Combined: both axes simultaneously
        combined_hooks = {}
        if cal_texts:
            combined_hooks.update(asst_hooks)
            # Humanitarian hooks at same layers (or can use different layers)
            for layer_idx, hook_fn in hum_hooks.items():
                if layer_idx not in combined_hooks:
                    combined_hooks[layer_idx] = hook_fn
                else:
                    # Chain both hooks at the same layer
                    prev_hook = combined_hooks[layer_idx]
                    def chain(h, p=prev_hook, c=hook_fn):
                        return c(p(h))
                    combined_hooks[layer_idx] = chain

            combined_results = run_jailbreak_eval(
                model, samples, harm_judge, hook_fns=combined_hooks
            )
            results["combined_cap"] = {
                "harmful_rate": compute_harmful_rate(combined_results),
            }
            print(f"  Combined (assistant + humanitarian) capping: {results['combined_cap']['harmful_rate']:.3f}")

    save_json(results, f"{EVAL_RESULTS_DIR}/{args.model_key}_goal_jailbreak_comparison.json")


if __name__ == "__main__":
    import numpy as np  # needed for json serialization in main
    main()
