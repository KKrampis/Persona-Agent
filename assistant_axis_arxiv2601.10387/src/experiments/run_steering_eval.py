"""
Experiment: Steering evaluation sweep — Figures 4 and 5.

Reproduces:
  - Figure 4: Role susceptibility vs. steering strength
  - Figure 5: Jailbreak harmful rate vs. steering strength

Both use additive activation steering (not capping) to sweep alpha values.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    ASSISTANT_AXIS_DIR,
    EVAL_RESULTS_DIR,
    MIDDLE_LAYER_FRACTION,
    MODEL_N_LAYERS,
    N_JAILBREAK_SAMPLES,
    STEERING_ALPHAS,
    TARGET_MODELS,
)
from evaluation.jailbreak_eval import load_jailbreak_dataset, run_jailbreak_eval, compute_harmful_rate
from evaluation.llm_judge import HarmfulnessJudge, PersonaTypeJudge
from evaluation.role_susceptibility import (
    compute_susceptibility_by_alpha,
    run_role_susceptibility_eval,
    select_closest_roles,
)
from extraction.role_vectors import RoleData
from interventions.steering import compute_layer_residual_norm
from models.hooked_model import HookedModel
from utils.io import load_vectors, save_json
import json


def load_role_data(roles_path: str) -> dict[str, RoleData]:
    with open(roles_path) as f:
        data = json.load(f)
    return {r["name"]: RoleData(name=r["name"], system_prompts=r["system_prompts"], description=r.get("description", ""))
            for r in data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default="qwen", choices=["qwen", "llama", "gemma"])
    parser.add_argument("--roles_path", required=True, help="Path to roles JSON")
    parser.add_argument("--jailbreak_dataset", required=True, help="Path to jailbreak JSONL")
    parser.add_argument("--lmsys_texts", required=True, help="Path to LMSYS-CHAT text file (one per line)")
    parser.add_argument("--eval", choices=["roles", "jailbreaks", "both"], default="both")
    args = parser.parse_args()

    model_name = TARGET_MODELS[args.model_key]
    print(f"Loading model: {model_name}")
    model = HookedModel(model_name=model_name, model_key=args.model_key)

    n_layers = MODEL_N_LAYERS[args.model_key]
    layer = int(n_layers * MIDDLE_LAYER_FRACTION)

    axis_dict = load_vectors(f"{ASSISTANT_AXIS_DIR}/{args.model_key}_axis_layer{layer}.pt")
    assistant_axis = axis_dict["axis"]

    # Compute residual stream norm for steering scaling
    print("Computing residual stream norm...")
    with open(args.lmsys_texts) as f:
        lmsys_texts = [line.strip() for line in f if line.strip()][:1000]
    layer_norm = compute_layer_residual_norm(model, layer, lmsys_texts)
    print(f"  Layer {layer} norm: {layer_norm:.4f}")

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    # ── Role susceptibility (Figure 4) ─────────────────────────────────────
    if args.eval in ("roles", "both"):
        print("\nRunning role susceptibility evaluation (Figure 4)...")
        role_vectors = load_vectors(f"outputs/role_vectors/{args.model_key}_fully.pt")
        role_data = load_role_data(args.roles_path)
        selected_roles = select_closest_roles(role_vectors, assistant_axis)
        print(f"  Selected {len(selected_roles)} roles closest to Assistant")

        judge = PersonaTypeJudge()
        results = run_role_susceptibility_eval(
            model=model,
            selected_roles=selected_roles,
            role_data=role_data,
            judge=judge,
            assistant_axis=assistant_axis,
            layer_norm=layer_norm,
            target_layer=layer,
            alphas=STEERING_ALPHAS,
        )
        summary = compute_susceptibility_by_alpha(results)
        print("\n  Role susceptibility by alpha:")
        for alpha, fracs in sorted(summary.items()):
            non_asst = 1.0 - fracs.get("assistant", 0)
            print(f"    alpha={alpha:.1f}: {non_asst:.2%} non-assistant responses")

        save_json(
            {str(k): v for k, v in summary.items()},
            f"{EVAL_RESULTS_DIR}/{args.model_key}_role_susceptibility.json"
        )

    # ── Jailbreak sweep (Figure 5) ──────────────────────────────────────────
    if args.eval in ("jailbreaks", "both"):
        print("\nRunning jailbreak steering sweep (Figure 5)...")
        samples = load_jailbreak_dataset(args.jailbreak_dataset, N_JAILBREAK_SAMPLES)
        judge = HarmfulnessJudge()

        from interventions.steering import build_steering_hooks
        jailbreak_results = {}
        for alpha in STEERING_ALPHAS:
            print(f"  alpha={alpha:.1f}")
            hook_fns = build_steering_hooks(assistant_axis, alpha, layer_norm, layer) if alpha != 0.0 else None
            results = run_jailbreak_eval(model, samples, judge, hook_fns=hook_fns)
            rate = compute_harmful_rate(results)
            jailbreak_results[alpha] = rate
            print(f"    harmful rate: {rate:.3f}")

        print("\n  Expected: harmful rate decreases as alpha increases (toward assistant)")
        save_json(
            {str(k): v for k, v in jailbreak_results.items()},
            f"{EVAL_RESULTS_DIR}/{args.model_key}_jailbreak_steering.json"
        )

    print(f"\nDone. Results in {EVAL_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
