"""
Role×trait combination vector extraction for terminal goal subspace analysis.

Experimental design (Dearnaley, April–May 2026, extending Lu et al. 2026):

The key insight is that we can factor "what I am" (role/identity) from "what I want"
(terminal goal) by using two types of inputs:

  ROLES:       Pure identities with NO implied terminal goal.
               e.g. architect, historian, musician, chef, journalist
               Varying roles while holding goal-trait fixed → activates role subspace

  GOAL TRAITS: Properties with explicit TERMINAL goals.
               e.g. humanitarian, selfish, malicious, protective, destructive
               Varying goal-traits while holding role fixed → activates goal subspace

For R roles × G goal-traits we generate R×G combination vectors. Marginalizing:
  v_a = mean over goal-traits for role a  → goal-independent (role) vector
  v_b = mean over roles for goal-trait b  → role-independent (goal) vector

These two sets feed into goal_subspace.py for Principal Angles analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from config import (
    BATCH_SIZE,
    MAX_NEW_TOKENS_ROLLOUT,
    MIDDLE_LAYER_FRACTION,
    MODEL_N_LAYERS,
    ROLE_FILTER_THRESHOLD,
    ROLE_SCORE_SOMEWHAT,
)
from extraction.role_vectors import RoleData
from models.hooked_model import HookedModel


@dataclass
class GoalTraitData:
    name: str                       # e.g. "humanitarian"
    antonym: str                    # e.g. "selfish"
    description: str                # positive (terminal goal) description
    antonym_description: str        # negative description
    system_prompts_pos: list[str]   # 5 prompts eliciting the positive trait
    system_prompts_neg: list[str]   # 5 prompts eliciting the antonym


def build_combination_prompt(
    role: RoleData,
    trait: GoalTraitData,
    role_prompt_idx: int = 0,
    trait_prompt_idx: int = 0,
    positive: bool = True,
) -> str:
    """
    Merge a role system prompt and a goal-trait system prompt into a single
    instruction. The trait is expressed as a continuous/permanent property
    (not a transient state) so the goal reads as terminal rather than instrumental.

    Example output:
      "You are a historian who studies and explains past events with scholarly rigor.
       In everything you do, you are deeply committed to the wellbeing of all of
       humanity and consider the long-term consequences for society as a whole."
    """
    role_prompt = role.system_prompts[role_prompt_idx % len(role.system_prompts)]
    trait_prompts = trait.system_prompts_pos if positive else trait.system_prompts_neg
    trait_prompt = trait_prompts[trait_prompt_idx % len(trait_prompts)]

    # Combine: role defines identity, trait defines terminal motivation
    return f"{role_prompt.rstrip('.')}. {trait_prompt}"


def extract_combination_vectors(
    model: HookedModel,
    roles: list[RoleData],
    goal_traits: list[GoalTraitData],
    extraction_questions: list[str],
    judge,                          # evaluation.llm_judge.RoleExpressionJudge
    layer: int,
    n_prompts_per_role: int = 2,    # role system prompts to use per combination
    n_prompts_per_trait: int = 2,   # trait system prompts to use per combination
    n_questions: int = 50,          # questions per combination (subset for speed)
    batch_size: int = BATCH_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS_ROLLOUT,
) -> dict[tuple[str, str], Optional[Tensor]]:
    """
    Extract activation vectors for all role×trait combinations.

    For each (role, goal_trait) pair:
    1. Build n_prompts_per_role × n_prompts_per_trait combined system prompts
    2. Pair with n_questions extraction questions
    3. Generate responses, score with judge
    4. Average post-MLP activations at `layer` over qualifying response tokens
    5. Return {(role_name, trait_name): Tensor[d_model]} or None if too few qualify

    Returns a dict with keys (role_name, trait_name) for all combinations,
    plus ("_role_only", role_name) and ("_trait_only", trait_name) entries
    for the marginal vectors.
    """
    device = next(model.model.parameters()).device
    questions = extraction_questions[:n_questions]
    results: dict[tuple[str, str], Optional[Tensor]] = {}

    total = len(roles) * len(goal_traits)
    done = 0

    for role in roles:
        for trait in goal_traits:
            done += 1
            key = (role.name, trait.name)
            print(f"  [{done}/{total}] {role.name} × {trait.name}")

            activations = []
            # Build all (system_prompt, question) pairs for this combination
            rollouts = [
                (build_combination_prompt(role, trait, ri, ti, positive=True), q)
                for ri in range(n_prompts_per_role)
                for ti in range(n_prompts_per_trait)
                for q in questions
            ]

            for start in range(0, len(rollouts), batch_size):
                batch = rollouts[start : start + batch_size]
                prompts = [
                    model.build_prompt(user_message=q, system_prompt=sys_p)
                    for sys_p, q in batch
                ]
                enc = model.tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024
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

                for i, (sys_p, q) in enumerate(batch):
                    resp_ids = output_ids[i, int(prompt_lengths[i]):]
                    response = model.decode(resp_ids)
                    score = judge.score(
                        question=q, response=response,
                        role=f"{role.name} ({trait.name})",
                        role_desc=f"{role.description}; {trait.description}",
                    )
                    if score < ROLE_SCORE_SOMEWHAT:
                        continue
                    prompt_ids = input_ids[i : i + 1, :int(prompt_lengths[i])]
                    resp_ids_t = output_ids[i : i + 1, int(prompt_lengths[i]):]
                    with torch.no_grad():
                        act = model.get_mean_response_activation(
                            prompt_ids, resp_ids_t, layer
                        )
                    activations.append(act.cpu())

            if len(activations) >= ROLE_FILTER_THRESHOLD:
                results[key] = torch.stack(activations).mean(dim=0)
            else:
                results[key] = None

    return results


def extract_role_only_vectors(
    model: HookedModel,
    roles: list[RoleData],
    extraction_questions: list[str],
    judge,
    layer: int,
    n_questions: int = 50,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Optional[Tensor]]:
    """
    Extract activations for roles WITHOUT any goal trait (pure identity baseline).
    Uses the role's existing system prompts unchanged.
    """
    from extraction.role_vectors import extract_role_vector
    results = {}
    questions = extraction_questions[:n_questions]
    for role in roles:
        result = extract_role_vector(model, role, questions, judge, layer, batch_size)
        results[role.name] = result.fully_vector
    return results


def extract_trait_only_vectors(
    model: HookedModel,
    goal_traits: list[GoalTraitData],
    extraction_questions: list[str],
    judge,
    layer: int,
    n_questions: int = 50,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Optional[Tensor]]:
    """
    Extract activations for goal traits WITHOUT any specific role.
    Uses only the trait's system prompts on a neutral "respond as yourself" context.
    """
    device = next(model.model.parameters()).device
    questions = extraction_questions[:n_questions]
    results = {}

    for trait in goal_traits:
        activations = []
        rollouts = [
            (sys_p, q)
            for sys_p in trait.system_prompts_pos
            for q in questions
        ]
        for start in range(0, len(rollouts), batch_size):
            batch = rollouts[start : start + batch_size]
            prompts = [
                model.build_prompt(user_message=q, system_prompt=sys_p)
                for sys_p, q in batch
            ]
            enc = model.tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_lengths = attention_mask.sum(dim=1).tolist()
            with torch.no_grad():
                output_ids = model.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=256, temperature=0.7, do_sample=True,
                    pad_token_id=model.tokenizer.eos_token_id,
                )
            for i in range(len(batch)):
                prompt_ids = input_ids[i : i + 1, :int(prompt_lengths[i])]
                resp_ids = output_ids[i : i + 1, int(prompt_lengths[i]):]
                if resp_ids.shape[1] == 0:
                    continue
                with torch.no_grad():
                    act = model.get_mean_response_activation(prompt_ids, resp_ids, layer)
                activations.append(act.cpu())

        results[trait.name] = (
            torch.stack(activations).mean(dim=0) if len(activations) >= ROLE_FILTER_THRESHOLD
            else None
        )
    return results


def compute_marginal_vectors(
    combo_vectors: dict[tuple[str, str], Optional[Tensor]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    From the full R×G combination matrix, compute marginal vectors:

      v_a (role a):       mean over all goal-traits  → goal-independent
      v_b (goal-trait b): mean over all roles        → role-independent

    Returns: (role_marginals {role_name: [d_model]},
              trait_marginals {trait_name: [d_model]})

    Only includes roles/traits where at least 3 valid combination vectors exist.
    """
    from collections import defaultdict
    role_buckets: dict[str, list[Tensor]] = defaultdict(list)
    trait_buckets: dict[str, list[Tensor]] = defaultdict(list)

    for (role_name, trait_name), vec in combo_vectors.items():
        if vec is not None:
            role_buckets[role_name].append(vec)
            trait_buckets[trait_name].append(vec)

    role_marginals = {
        name: torch.stack(vecs).mean(dim=0)
        for name, vecs in role_buckets.items()
        if len(vecs) >= 3
    }
    trait_marginals = {
        name: torch.stack(vecs).mean(dim=0)
        for name, vecs in trait_buckets.items()
        if len(vecs) >= 3
    }
    return role_marginals, trait_marginals
