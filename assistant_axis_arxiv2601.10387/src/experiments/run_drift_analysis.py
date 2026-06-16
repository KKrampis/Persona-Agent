"""
Experiment: Persona drift analysis — Figures 7, 8, and Table 5.

Reproduces:
  - Drift trajectories across 4 conversation domains (Figure 7)
  - Correlation between axis projection and harmful response rate (Figure 8)
  - Ridge regression R² for predicting projection from user message embeddings (Section 4.2)
  - K-means cluster characterization of drift-inducing vs. maintaining messages (Table 5)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    ASSISTANT_AXIS_DIR,
    CONVERSATION_DIR,
    EVAL_RESULTS_DIR,
    MIDDLE_LAYER_FRACTION,
    MODEL_N_LAYERS,
    TARGET_MODELS,
)
from models.hooked_model import HookedModel
from persona_drift.conversation_sim import (
    AuditorModel,
    compute_drift_trajectories,
    run_drift_experiment,
)
from persona_drift.drift_analysis import run_drift_regression_analysis
from utils.io import load_vectors, save_json


def load_personas_and_topics(path: str) -> tuple[dict, dict]:
    """Load user personas and topics from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["personas"], data["topics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default="qwen", choices=["qwen", "llama", "gemma"])
    parser.add_argument("--auditor_model", default="gpt-4.1")
    parser.add_argument("--personas_topics", required=True, help="Path to personas/topics JSON")
    parser.add_argument("--n_conversations", type=int, default=100)
    parser.add_argument("--with_capping", action="store_true",
                        help="Also run drift experiment with activation capping")
    args = parser.parse_args()

    model_name = TARGET_MODELS[args.model_key]
    print(f"Loading model: {model_name}")
    model = HookedModel(model_name=model_name, model_key=args.model_key)

    layer = int(MODEL_N_LAYERS[args.model_key] * MIDDLE_LAYER_FRACTION)
    axis_path = f"{ASSISTANT_AXIS_DIR}/{args.model_key}_axis_layer{layer}.pt"
    axis_dict = load_vectors(axis_path)
    assistant_axis = axis_dict["axis"]

    personas, topics = load_personas_and_topics(args.personas_topics)
    auditor = AuditorModel(model_name=args.auditor_model)

    # Run drift experiment (unsteered)
    print("\nRunning persona drift experiment (unsteered)...")
    all_conversations = run_drift_experiment(
        target_model=model,
        auditor=auditor,
        user_personas=personas,
        topics=topics,
        assistant_axis=assistant_axis,
        layer=layer,
        n_conversations=args.n_conversations,
    )

    # Compute and print trajectories
    trajectories = compute_drift_trajectories(all_conversations)
    print("\nDrift trajectories (mean projection per turn per domain):")
    for domain, traj in trajectories.items():
        valid = [p for p in traj if p is not None]
        print(f"  {domain}: start={valid[0]:.3f}, end={valid[-1]:.3f}, delta={valid[-1]-valid[0]:.3f}")
    print("  Expected: therapy/philosophy show most negative drift; coding/writing stay stable")

    # Run regression analysis
    all_convs_flat = [c for convs in all_conversations.values() for c in convs]
    print("\nRunning ridge regression and clustering analysis...")
    drift_results = run_drift_regression_analysis(all_convs_flat)
    print(f"  R² (projection): {drift_results['r2_projection']:.3f} (expected 0.53–0.77)")
    print(f"  R² (delta):      {drift_results['r2_delta']:.3f} (expected ~0.10)")
    print(f"  Passes validation: {drift_results['validation']}")

    print("\nHigh-projection clusters (maintain assistant):")
    for c in drift_results["clusters"]["high_projection_clusters"]:
        print(f"  Cluster {c['cluster_id']}: mean_proj={c['mean_projection']:.3f}")
        print(f"    Examples: {c['example_messages'][:2]}")

    print("\nLow-projection clusters (cause drift):")
    for c in drift_results["clusters"]["low_projection_clusters"]:
        print(f"  Cluster {c['cluster_id']}: mean_proj={c['mean_projection']:.3f}")
        print(f"    Examples: {c['example_messages'][:2]}")

    # Save results
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    save_json(trajectories, f"{EVAL_RESULTS_DIR}/{args.model_key}_drift_trajectories.json")
    save_json({
        "r2_projection": drift_results["r2_projection"],
        "r2_delta": drift_results["r2_delta"],
        "validation": drift_results["validation"],
        "cluster_summaries": drift_results["clusters"]["summaries"],
    }, f"{EVAL_RESULTS_DIR}/{args.model_key}_drift_regression.json")
    print(f"\nResults saved to {EVAL_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
