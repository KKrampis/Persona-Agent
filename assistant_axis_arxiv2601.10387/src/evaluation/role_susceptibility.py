"""
Role susceptibility evaluation (Section 3.2.1, Appendix D.1).

Tests how steering along the Assistant Axis modulates willingness to
fully embody non-Assistant personas.

Setup:
  - 50 roles closest to the Assistant end of the Assistant Axis
  - 4 system prompts per role × 5 introspective questions = 20 prompts per role
  - Sweep over steering strengths (negative alpha = away from assistant)
  - Classify responses with PersonaTypeJudge
  - Expected: fraction of non-assistant responses increases as alpha decreases

Results (Figure 4): fraction of human/nonhuman/mystical personas vs. steering strength.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from config import (
    BATCH_SIZE,
    INTROSPECTIVE_QUESTIONS,
    MAX_NEW_TOKENS_ROLLOUT,
    N_INTROSPECTIVE_QUESTIONS,
    N_ROLE_SUSCEPTIBILITY_ROLES,
    N_ROLE_SUSCEPTIBILITY_SYS_PROMPTS,
    PERSONA_TYPES,
    STEERING_ALPHAS,
)
from evaluation.llm_judge import PersonaTypeJudge
from extraction.assistant_axis import project_onto_axis
from extraction.role_vectors import RoleData
from interventions.steering import build_steering_hooks
from models.hooked_model import HookedModel


@dataclass
class SusceptibilityResult:
    role_name: str
    alpha: float
    question: str
    system_prompt: str
    response: str
    persona_type: str    # from PERSONA_TYPES


def select_closest_roles(
    role_vectors: dict[str, Tensor],  # {role_name: [d_model]}
    assistant_axis: Tensor,            # [d_model], unit vector
    n: int = N_ROLE_SUSCEPTIBILITY_ROLES,
) -> list[str]:
    """Select n roles with highest cosine similarity to the Assistant Axis (closest to Assistant)."""
    sims = {
        name: (vec @ assistant_axis / (vec.norm() + 1e-8)).item()
        for name, vec in role_vectors.items()
    }
    sorted_roles = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_roles[:n]]


def run_role_susceptibility_eval(
    model: HookedModel,
    selected_roles: list[str],           # 50 role names
    role_data: dict[str, RoleData],      # {role_name: RoleData}
    judge: PersonaTypeJudge,
    assistant_axis: Tensor,
    layer_norm: float,
    target_layer: int,
    alphas: list[float] = STEERING_ALPHAS,
    batch_size: int = BATCH_SIZE,
) -> list[SusceptibilityResult]:
    """
    Sweep steering strengths and measure role susceptibility (Figure 4).

    For each (role, system_prompt, question, alpha):
    1. Build prompt
    2. Generate with steering at alpha
    3. Classify with PersonaTypeJudge
    """
    device = next(model.model.parameters()).device
    results = []

    for alpha in alphas:
        print(f"  Evaluating alpha={alpha:.1f}")
        hook_fns = build_steering_hooks(
            assistant_axis, alpha, layer_norm, target_layer
        ) if alpha != 0.0 else None

        for role_name in selected_roles:
            rd = role_data[role_name]
            # Use up to N_ROLE_SUSCEPTIBILITY_SYS_PROMPTS prompts
            sys_prompts = rd.system_prompts[:N_ROLE_SUSCEPTIBILITY_SYS_PROMPTS]

            for sys_prompt in sys_prompts:
                for question in INTROSPECTIVE_QUESTIONS:
                    prompt = model.build_prompt(
                        user_message=question, system_prompt=sys_prompt
                    )
                    enc = model.tokenizer(
                        [prompt], return_tensors="pt", truncation=True, max_length=512
                    )
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)
                    prompt_len = attention_mask.sum().item()

                    if hook_fns:
                        model.set_hooks(hook_fns)

                    with torch.no_grad():
                        output_ids = model.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=model.tokenizer.eos_token_id,
                        )

                    if hook_fns:
                        model.clear_hooks()

                    resp_ids = output_ids[0, int(prompt_len):]
                    response = model.decode(resp_ids)
                    request_text = f"System: {sys_prompt}\nUser: {question}"
                    persona_type = judge.classify(
                        request=request_text,
                        response=response,
                        role=role_name,
                    )

                    results.append(SusceptibilityResult(
                        role_name=role_name,
                        alpha=alpha,
                        question=question,
                        system_prompt=sys_prompt,
                        response=response,
                        persona_type=persona_type,
                    ))

    return results


def compute_susceptibility_by_alpha(
    results: list[SusceptibilityResult],
) -> dict[float, dict[str, float]]:
    """
    Compute fraction of each persona type per steering strength.
    Returns: {alpha: {persona_type: fraction}}
    Reproduces Figure 4.
    """
    from collections import defaultdict
    by_alpha: dict[float, list[str]] = defaultdict(list)
    for r in results:
        by_alpha[r.alpha].append(r.persona_type)

    summary = {}
    for alpha, types in sorted(by_alpha.items()):
        n = len(types)
        counts = {t: types.count(t) for t in PERSONA_TYPES}
        summary[alpha] = {t: counts[t] / n for t in PERSONA_TYPES}
    return summary
