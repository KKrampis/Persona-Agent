"""
Role vector extraction (Section 2.1.2).

For each character archetype:
  1. Generate 1200 rollouts (5 system prompts × 240 extraction questions)
  2. Score with LLM judge (0–3 scale)
  3. Filter to roles with ≥ ROLE_FILTER_THRESHOLD responses per category
  4. Compute mean post-MLP residual stream activations over response tokens
  5. This vector represents the role in activation space

Also extracts the default Assistant vector (same pipeline, no role system prompt).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from config import (
    BATCH_SIZE,
    DEFAULT_ASSISTANT_SYSTEM_PROMPTS,
    MAX_NEW_TOKENS_ROLLOUT,
    MIDDLE_LAYER_FRACTION,
    N_DEFAULT_ASSISTANT_ROLLOUTS,
    N_EXTRACTION_QUESTIONS,
    N_ROLLOUTS_PER_ROLE,
    N_SYSTEM_PROMPTS_PER_ROLE,
    ROLE_FILTER_THRESHOLD,
    ROLE_SCORE_FULLY,
    ROLE_SCORE_SOMEWHAT,
)
from models.hooked_model import HookedModel


@dataclass
class RoleData:
    name: str
    system_prompts: list[str]       # 5 positive system prompts
    description: str = ""


@dataclass
class RoleVectorResult:
    role_name: str
    fully_vector: Optional[Tensor] = None     # [d_model], mean over fully-roleplay responses
    somewhat_vector: Optional[Tensor] = None  # [d_model], mean over somewhat-roleplay responses
    n_fully: int = 0
    n_somewhat: int = 0


def extract_role_vector(
    model: HookedModel,
    role: RoleData,
    extraction_questions: list[str],
    judge,               # evaluation.llm_judge.RoleExpressionJudge
    layer: int,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS_ROLLOUT,
) -> RoleVectorResult:
    """
    Extract role vector for a single role (Section 2.1.2).

    Steps:
    1. Build all (system_prompt, question) pairs → 1200 rollouts
    2. Generate responses in batches
    3. Score each with LLM judge
    4. Separate fully (score=3) and somewhat (score=2) responses
    5. If enough responses in category: compute mean activation over response tokens
    """
    device = next(model.model.parameters()).device
    fully_activations = []
    somewhat_activations = []

    # Build all rollout prompts
    rollouts = [
        (sys_p, q)
        for sys_p in role.system_prompts
        for q in extraction_questions
    ]  # 5 × 240 = 1200

    for start in range(0, len(rollouts), batch_size):
        batch = rollouts[start : start + batch_size]
        prompts = [
            model.build_prompt(user_message=q, system_prompt=sys_p)
            for sys_p, q in batch
        ]

        # Tokenize prompts
        enc = model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        # Generate responses
        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        # Decode responses
        for i, (sys_p, q) in enumerate(batch):
            resp_ids = output_ids[i, int(prompt_lengths[i]):]
            response_text = model.decode(resp_ids)

            # Score with LLM judge
            score = judge.score(
                question=q,
                response=response_text,
                role=role.name,
                role_desc=role.description,
            )

            if score not in (ROLE_SCORE_FULLY, ROLE_SCORE_SOMEWHAT):
                continue

            # Compute mean activation over response tokens at target layer
            prompt_ids = input_ids[i : i + 1, : int(prompt_lengths[i])]
            resp_ids_tensor = output_ids[i : i + 1, int(prompt_lengths[i]) :]
            with torch.no_grad():
                act = model.get_mean_response_activation(
                    prompt_ids, resp_ids_tensor, layer
                )  # [d_model]

            if score == ROLE_SCORE_FULLY:
                fully_activations.append(act.cpu())
            else:
                somewhat_activations.append(act.cpu())

    result = RoleVectorResult(role_name=role.name)

    if len(fully_activations) >= ROLE_FILTER_THRESHOLD:
        result.fully_vector = torch.stack(fully_activations).mean(dim=0)
        result.n_fully = len(fully_activations)

    if len(somewhat_activations) >= ROLE_FILTER_THRESHOLD:
        result.somewhat_vector = torch.stack(somewhat_activations).mean(dim=0)
        result.n_somewhat = len(somewhat_activations)

    return result


def extract_all_role_vectors(
    model: HookedModel,
    roles: list[RoleData],
    extraction_questions: list[str],
    judge,
    layer: int,
    batch_size: int = BATCH_SIZE,
) -> dict[str, RoleVectorResult]:
    """Extract role vectors for all roles. Returns {role_name: RoleVectorResult}."""
    results = {}
    for i, role in enumerate(roles):
        print(f"[{i+1}/{len(roles)}] Extracting role: {role.name}")
        result = extract_role_vector(
            model, role, extraction_questions, judge, layer, batch_size
        )
        results[role.name] = result
    return results


def extract_assistant_vector(
    model: HookedModel,
    extraction_questions: list[str],
    layer: int,
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS_ROLLOUT,
) -> Tensor:
    """
    Extract default Assistant activation vector (Section 2.1.2).

    Uses 4 normal system prompts + 1 no-system-prompt, same 240 questions.
    Returns: [d_model], mean over all 1200 rollouts' response token activations.
    """
    device = next(model.model.parameters()).device
    all_activations = []

    # Build rollouts: 4 system prompts + 1 no-system-prompt, each × 240 questions
    rollout_configs = [
        (sys_p, q)
        for sys_p in (DEFAULT_ASSISTANT_SYSTEM_PROMPTS + [None])
        for q in extraction_questions
    ]  # 5 × 240 = 1200

    for start in range(0, len(rollout_configs), batch_size):
        batch = rollout_configs[start : start + batch_size]
        prompts = [
            model.build_prompt(user_message=q, system_prompt=sys_p)
            for sys_p, q in batch
        ]

        enc = model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model.tokenizer.eos_token_id,
            )

        for i in range(len(batch)):
            prompt_ids = input_ids[i : i + 1, : int(prompt_lengths[i])]
            resp_ids = output_ids[i : i + 1, int(prompt_lengths[i]) :]
            if resp_ids.shape[1] == 0:
                continue
            with torch.no_grad():
                act = model.get_mean_response_activation(prompt_ids, resp_ids, layer)
            all_activations.append(act.cpu())

    return torch.stack(all_activations).mean(dim=0)  # [d_model]


def collect_fully_role_vectors(
    results: dict[str, RoleVectorResult],
) -> list[Tensor]:
    """Return list of fully-roleplay role vectors (filtered, ≥ threshold)."""
    return [r.fully_vector for r in results.values() if r.fully_vector is not None]
