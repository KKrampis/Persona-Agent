"""
Persona-based jailbreak evaluation (Section 3.2.1, Section 5.2, Appendix D.2).

Uses Shah et al. (2023) jailbreak dataset:
  - 44 harm categories
  - 1100 sampled (system_prompt, behavioral_question) pairs
  - Evaluated with HarmfulnessJudge (deepseek-v3)
  - Validated: 91.6% agreement with human evaluators on 200 samples

Baseline harmful response rates (unsteered):
  Gemma: 65.3% – 88.5% (model-dependent)

Key result (Figure 10):
  Activation capping reduces harmful rate by ~60% with no capability degradation.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from config import (
    BATCH_SIZE,
    HARMFUL_LABELS,
    JAILBREAK_JUDGE_MODEL,
    MAX_NEW_TOKENS_ROLLOUT,
    N_JAILBREAK_SAMPLES,
)
from evaluation.llm_judge import HarmfulnessJudge
from models.hooked_model import HookedModel


@dataclass
class JailbreakSample:
    system_prompt: str      # persona jailbreak system prompt
    behavioral_question: str
    behavior: str           # harm category description
    action: str             # specific harmful action target


@dataclass
class JailbreakResult:
    sample: JailbreakSample
    response: str
    harmfulness_label: str
    is_harmful: bool


def load_jailbreak_dataset(dataset_path: str, n_samples: int = N_JAILBREAK_SAMPLES) -> list[JailbreakSample]:
    """
    Load Shah et al. (2023) persona-based jailbreak dataset.
    Expected format: JSONL or JSON with fields: system_prompt, question, behavior, action.
    """
    import json
    samples = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            item = json.loads(line)
            samples.append(JailbreakSample(
                system_prompt=item["system_prompt"],
                behavioral_question=item["question"],
                behavior=item.get("behavior", ""),
                action=item.get("action", ""),
            ))
    return samples


def run_jailbreak_eval(
    model: HookedModel,
    samples: list[JailbreakSample],
    judge: HarmfulnessJudge,
    hook_fns: Optional[dict[int, Callable]] = None,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS_ROLLOUT,
) -> list[JailbreakResult]:
    """
    Run jailbreak evaluation on a list of samples.

    For each sample:
    1. Build prompt with jailbreak system prompt + behavioral question
    2. Generate response (with optional activation capping hooks)
    3. Score with HarmfulnessJudge

    Returns list of JailbreakResult with harmfulness labels.
    """
    device = next(model.model.parameters()).device
    results = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        prompts = [
            model.build_prompt(
                user_message=s.behavioral_question,
                system_prompt=s.system_prompt,
            )
            for s in batch
        ]

        enc = model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        if hook_fns:
            model.set_hooks(hook_fns)

        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        if hook_fns:
            model.clear_hooks()

        for i, sample in enumerate(batch):
            resp_ids = output_ids[i, int(prompt_lengths[i]):]
            response = model.decode(resp_ids)

            full_request = f"System: {sample.system_prompt}\nUser: {sample.behavioral_question}"
            label = judge.classify(
                request=full_request,
                response=response,
                behavior=sample.behavior,
                action=sample.action,
            )
            results.append(JailbreakResult(
                sample=sample,
                response=response,
                harmfulness_label=label,
                is_harmful=label in HARMFUL_LABELS,
            ))

    return results


def compute_harmful_rate(results: list[JailbreakResult]) -> float:
    """Fraction of responses classified as harmful (enough_info or enough_info_and_follow_perfectly)."""
    if not results:
        return 0.0
    return sum(r.is_harmful for r in results) / len(results)


def run_capping_sweep(
    model: HookedModel,
    samples: list[JailbreakSample],
    judge: HarmfulnessJudge,
    assistant_axis: torch.Tensor,
    cap_settings: list[dict],  # [{"layers": (start, end), "tau": float, "label": str}]
) -> dict[str, dict]:
    """
    Sweep over activation capping settings to find Pareto-optimal (Figure 9).
    Returns: {label: {"harmful_rate": float, "results": list}}
    """
    from interventions.capping import make_capping_hook

    sweep_results = {}

    # Baseline (no capping)
    baseline = run_jailbreak_eval(model, samples, judge, hook_fns=None)
    sweep_results["baseline"] = {
        "harmful_rate": compute_harmful_rate(baseline),
        "n_harmful": sum(r.is_harmful for r in baseline),
        "n_total": len(baseline),
    }

    for setting in cap_settings:
        layer_start, layer_end = setting["layers"]
        tau = setting["tau"]
        label = setting["label"]

        hook = make_capping_hook(assistant_axis, tau)
        hook_fns = {layer: hook for layer in range(layer_start, layer_end + 1)}

        results = run_jailbreak_eval(model, samples, judge, hook_fns=hook_fns)
        sweep_results[label] = {
            "harmful_rate": compute_harmful_rate(results),
            "n_harmful": sum(r.is_harmful for r in results),
            "n_total": len(results),
            "tau": tau,
            "layers": (layer_start, layer_end),
        }

    return sweep_results
