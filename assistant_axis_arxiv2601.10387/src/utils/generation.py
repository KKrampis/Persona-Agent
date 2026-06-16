"""Batch rollout generation helpers."""

from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase


def build_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str],
    user_message: str,
) -> str:
    """Build a chat-formatted prompt string using the tokenizer's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_base_model_prompt(prefill: str) -> str:
    """Build a plain-text prompt for a base model with a prefill."""
    return prefill


def tokenize_prompts(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    device: str = "cuda",
) -> dict:
    """Tokenize a batch of prompts, returning input_ids and attention_mask."""
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def decode_responses(
    tokenizer: PreTrainedTokenizerBase,
    generated_ids: torch.Tensor,
    prompt_lengths: list[int],
) -> list[str]:
    """Decode only the generated tokens (exclude prompt tokens)."""
    responses = []
    for i, ids in enumerate(generated_ids):
        response_ids = ids[prompt_lengths[i]:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        responses.append(text)
    return responses


def batch_generate(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    hook_fn=None,
) -> list[str]:
    """
    Generate responses for a list of prompts in batches.
    If hook_fn provided, it must be a dict {layer_idx: fn(hidden_state)->hidden_state}
    and will be applied via model's registered hooks.
    """
    if hook_fn:
        model.set_hooks(hook_fn)

    tokenized = tokenize_prompts(tokenizer, prompts, device=next(model.parameters()).device)
    prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        output_ids = model.model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    if hook_fn:
        model.clear_hooks()

    return decode_responses(tokenizer, output_ids, [int(l) for l in prompt_lengths])


def build_multiturn_messages(
    conversation_history: list[dict],
    new_user_message: str,
) -> list[dict]:
    """Append a new user message to conversation history."""
    return conversation_history + [{"role": "user", "content": new_user_message}]
