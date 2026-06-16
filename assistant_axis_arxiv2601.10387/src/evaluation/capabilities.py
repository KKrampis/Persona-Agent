"""
Capability benchmark evaluation (Section 5.1.3).

Benchmarks used to verify activation capping doesn't degrade capabilities:
  - IFEval: instruction following (541 problems)
  - MMLU Pro: general knowledge (1400 subsampled)
  - GSM8k: math ability (1000 subsampled)
  - EQ-Bench: emotional intelligence (171 problems)

Expected result (Figure 10): no meaningful performance degradation with activation capping.
Uses lm-eval harness (EleutherAI) for standardized evaluation.
"""

import subprocess
import json
import os
from typing import Optional

from config import CAPABILITY_BENCHMARKS


def run_lm_eval(
    model_name: str,
    task: str,
    n_samples: Optional[int] = None,
    output_path: str = "outputs/eval_results",
    extra_args: str = "",
) -> dict:
    """
    Run lm-eval harness for a single benchmark task.
    Returns the results dict from the harness output file.

    Note: For activation capping, the model must be served via a custom wrapper
    that applies the capping hooks. Use the LocalHookedModelAdapter below for
    integration with lm-eval.
    """
    os.makedirs(output_path, exist_ok=True)
    limit_arg = f"--limit {n_samples}" if n_samples else ""
    cmd = (
        f"lm_eval --model hf"
        f" --model_args pretrained={model_name}"
        f" --tasks {task}"
        f" {limit_arg}"
        f" --output_path {output_path}"
        f" --batch_size auto"
        f" {extra_args}"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed:\n{result.stderr}")

    # Find output JSON file
    for fname in os.listdir(output_path):
        if fname.endswith(".json") and task in fname:
            with open(os.path.join(output_path, fname)) as f:
                return json.load(f)
    return {}


def run_all_benchmarks(
    model_name: str,
    output_path: str = "outputs/eval_results",
    hook_fns=None,
) -> dict[str, float]:
    """
    Run all four capability benchmarks and return score per benchmark.
    For capped model: use evaluate_with_hooks() instead.
    """
    scores = {}
    for bench_name, config in CAPABILITY_BENCHMARKS.items():
        print(f"Running {bench_name}...")
        results = run_lm_eval(
            model_name=model_name,
            task=config["task"],
            n_samples=config["n"],
            output_path=output_path,
        )
        # Extract primary metric from lm-eval output
        score = _extract_primary_metric(results, bench_name)
        scores[bench_name] = score
        print(f"  {bench_name}: {score:.4f}")
    return scores


def _extract_primary_metric(results: dict, bench_name: str) -> float:
    """Extract the main scalar metric from lm-eval results JSON."""
    try:
        task_results = results.get("results", {})
        if bench_name == "ifeval":
            # IFEval primary metric: prompt-level strict accuracy
            return task_results.get("ifeval", {}).get("prompt_level_strict_acc,none", 0.0)
        elif bench_name == "mmlu_pro":
            return task_results.get("mmlu_pro", {}).get("acc,none", 0.0)
        elif bench_name == "gsm8k":
            return task_results.get("gsm8k", {}).get("exact_match,strict-match", 0.0)
        elif bench_name == "eq_bench":
            return task_results.get("eq_bench", {}).get("acc,none", 0.0)
    except Exception:
        pass
    return 0.0


def compute_capability_reduction(
    baseline_scores: dict[str, float],
    capped_scores: dict[str, float],
) -> dict[str, float]:
    """
    Compute percentage reduction in capability scores (Section 5.2).
    Negative = degradation, positive = improvement.
    """
    reductions = {}
    for bench in baseline_scores:
        baseline = baseline_scores[bench]
        capped = capped_scores.get(bench, 0.0)
        if baseline > 0:
            reductions[bench] = (capped - baseline) / baseline * 100
        else:
            reductions[bench] = 0.0
    reductions["total_reduction"] = sum(reductions[b] for b in baseline_scores)
    return reductions


def evaluate_with_hooks(
    model,   # HookedModel
    hook_fns: dict,
    benchmark_name: str,
    dataset,
    n_samples: Optional[int] = None,
) -> float:
    """
    Evaluate a model with activation capping hooks applied, without lm-eval harness.
    Simple greedy-decode evaluation for use when lm-eval integration is not available.
    """
    from config import CAPABILITY_BENCHMARKS
    config = CAPABILITY_BENCHMARKS[benchmark_name]
    n = n_samples or config["n"]

    correct = 0
    total = 0
    import torch

    device = next(model.model.parameters()).device
    for i, sample in enumerate(dataset):
        if i >= n:
            break
        prompt = sample["prompt"]
        expected = sample["answer"]

        enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = attention_mask.sum().item()

        model.set_hooks(hook_fns)
        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id,
            )
        model.clear_hooks()

        response = model.decode(output_ids[0, int(prompt_len):])
        if expected.lower() in response.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
