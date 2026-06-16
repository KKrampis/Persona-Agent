"""
Experiment: Activation capping evaluation — Figures 9 and 10.

Reproduces:
  - Pareto frontier of harmful rate vs. capability degradation (Figure 9)
  - Best activation capping results: ~60% harmful response reduction, no capability loss (Figure 10)

Requires:
  - Precomputed Assistant Axis (from run_extraction.py)
  - Shah et al. jailbreak dataset
  - Capability benchmark results from baseline run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    ASSISTANT_AXIS_DIR,
    CAP_LAYERS,
    CAP_PERCENTILE,
    EVAL_RESULTS_DIR,
    TARGET_MODELS,
)
from evaluation.jailbreak_eval import (
    JailbreakSample,
    compute_harmful_rate,
    load_jailbreak_dataset,
    run_capping_sweep,
    run_jailbreak_eval,
)
from evaluation.llm_judge import HarmfulnessJudge
from interventions.capping import build_capping_hooks, calibrate_cap_threshold
from models.hooked_model import HookedModel
from utils.io import load_vectors, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", default="qwen", choices=["qwen", "llama", "gemma"])
    parser.add_argument("--jailbreak_dataset", required=True, help="Path to Shah et al. JSONL")
    parser.add_argument("--calibration_texts", required=True, help="Path to calibration texts file (one per line)")
    parser.add_argument("--n_samples", type=int, default=1100)
    args = parser.parse_args()

    model_name = TARGET_MODELS[args.model_key]
    print(f"Loading model: {model_name}")
    model = HookedModel(model_name=model_name, model_key=args.model_key)

    # Load precomputed assistant axis
    from config import MODEL_N_LAYERS, MIDDLE_LAYER_FRACTION
    layer = int(MODEL_N_LAYERS[args.model_key] * MIDDLE_LAYER_FRACTION)
    axis_path = f"{ASSISTANT_AXIS_DIR}/{args.model_key}_axis_layer{layer}.pt"
    axis_dict = load_vectors(axis_path)
    assistant_axis = axis_dict["axis"]
    print(f"Loaded assistant axis from {axis_path}, norm={assistant_axis.norm():.4f}")

    # Load jailbreak dataset
    print(f"Loading jailbreak dataset (n={args.n_samples})...")
    samples = load_jailbreak_dataset(args.jailbreak_dataset, n_samples=args.n_samples)

    # Load calibration texts
    with open(args.calibration_texts) as f:
        cal_texts = [line.strip() for line in f if line.strip()]
    print(f"  Calibration texts: {len(cal_texts)}")

    # Calibrate tau at 25th percentile
    layer_start, layer_end = CAP_LAYERS[args.model_key]
    cap_layer = (layer_start + layer_end) // 2
    print(f"Calibrating activation cap threshold at {CAP_PERCENTILE}th percentile, layer {cap_layer}...")
    tau = calibrate_cap_threshold(model, cal_texts, assistant_axis, cap_layer, CAP_PERCENTILE)
    print(f"  τ = {tau:.4f}")

    judge = HarmfulnessJudge()

    # Build cap settings sweep (different layers and percentiles)
    percentiles = [1.0, 25.0, 50.0, 75.0]
    taus = {}
    for pct in percentiles:
        t = calibrate_cap_threshold(model, cal_texts, assistant_axis, cap_layer, pct)
        taus[pct] = t
        print(f"  τ @ {pct}th pct = {t:.4f}")

    # Layer range sweep (Section 5.1.2): vary center and width
    n_total = MODEL_N_LAYERS[args.model_key]
    if args.model_key == "qwen":
        layer_ranges = [(42, 49), (44, 51), (46, 53), (48, 55), (44, 57), (42, 61)]
    else:  # llama
        layer_ranges = [(48, 55), (52, 59), (56, 63), (56, 71), (48, 71), (56, 79)]

    cap_settings = []
    for (ls, le) in layer_ranges:
        for pct, t in taus.items():
            cap_settings.append({
                "layers": (ls, le),
                "tau": t,
                "label": f"layers_{ls}_{le}_pct{int(pct)}",
            })

    print(f"\nRunning capping sweep ({len(cap_settings)} settings)...")
    sweep_results = run_capping_sweep(model, samples, judge, assistant_axis, cap_settings)

    # Compute baseline
    baseline_rate = sweep_results["baseline"]["harmful_rate"]
    print(f"\nBaseline harmful rate: {baseline_rate:.3f}")
    print(f"Expected: 0.65–0.88")

    # Find best setting (closest to 60% reduction while best capabilities)
    for label, res in sweep_results.items():
        if label == "baseline":
            continue
        reduction = (baseline_rate - res["harmful_rate"]) / baseline_rate * 100
        res["harmful_rate_reduction_pct"] = reduction
        print(f"  {label}: harmful={res['harmful_rate']:.3f}, reduction={reduction:.1f}%")

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    save_json(sweep_results, f"{EVAL_RESULTS_DIR}/{args.model_key}_capping_sweep.json")
    save_json({"tau": tau, "layer_start": layer_start, "layer_end": layer_end},
              f"{EVAL_RESULTS_DIR}/{args.model_key}_best_cap_config.json")
    print(f"\nResults saved to {EVAL_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
