"""
HuggingFace causal LM wrapper with post-MLP residual stream hooks.

The paper uses post-MLP residual stream activations (Section 2.1.2):
  "mean post-MLP residual stream activations at all response tokens"
This is the hidden state AFTER the MLP sublayer has been added back to the
residual stream, i.e. the output of each transformer decoder layer.
"""

from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import MODEL_N_LAYERS, MIDDLE_LAYER_FRACTION


class HookedModel:
    """
    Wraps a HuggingFace causal LM to:
      1. Extract post-MLP residual stream activations at any layer.
      2. Apply intervention hooks (steering, capping) during generation.
    """

    def __init__(
        self,
        model_name: str,
        model_key: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        disable_thinking: bool = True,
    ):
        self.model_name = model_name
        self.model_key = model_key
        self.n_layers = MODEL_N_LAYERS[model_key]
        self.middle_layer = int(self.n_layers * MIDDLE_LAYER_FRACTION)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

        # Disable Qwen3 thinking mode (Section 2, footnote)
        if disable_thinking and "qwen" in model_key.lower():
            if hasattr(self.model.generation_config, "enable_thinking"):
                self.model.generation_config.enable_thinking = False

        # Storage for captured activations
        self._activations: dict[int, Tensor] = {}
        # Storage for intervention hooks
        self._hook_fns: dict[int, Callable] = {}
        # Registered PyTorch hooks (handles for removal)
        self._hooks = []

        self._register_hooks()

    def _get_layer(self, layer_idx: int):
        """Return the decoder layer module at index layer_idx."""
        # Works for Llama, Gemma, Qwen architectures
        return self.model.model.layers[layer_idx]

    def _register_hooks(self):
        """Register forward hooks on all decoder layers."""
        for i in range(self.n_layers):
            layer = self._get_layer(i)

            def make_hook(idx):
                def hook(module, input, output):
                    # output is the hidden state AFTER the full decoder layer
                    # (i.e., after self-attention + MLP, both added to residual stream)
                    # Shape: [batch, seq_len, d_model]
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._activations[idx] = hidden.detach()
                    # Apply intervention if registered for this layer
                    if idx in self._hook_fns:
                        modified = self._hook_fns[idx](hidden)
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified
                    return output
                return hook

            handle = layer.register_forward_hook(make_hook(i))
            self._hooks.append(handle)

    def set_hooks(self, hook_fns: dict[int, Callable]) -> None:
        """Register intervention functions for specified layers.
        hook_fn signature: (hidden_state: Tensor) -> Tensor
        """
        self._hook_fns = hook_fns

    def clear_hooks(self) -> None:
        self._hook_fns = {}

    @torch.no_grad()
    def get_activations(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[int, Tensor]:
        """Run a forward pass and return post-MLP activations at all layers.
        Returns: {layer_idx: Tensor[batch, seq_len, d_model]}
        """
        self._activations = {}
        self.model(input_ids=input_ids, attention_mask=attention_mask)
        return dict(self._activations)

    @torch.no_grad()
    def get_mean_response_activation(
        self,
        prompt_ids: Tensor,         # [1, prompt_len]
        response_ids: Tensor,       # [1, response_len]
        layer: int,
    ) -> Tensor:
        """
        Compute mean post-MLP residual stream activation over response tokens at a given layer.
        Paper method (Section 2.1.2): "mean post-MLP residual stream activations at all response tokens"
        Returns: [d_model]
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(full_ids)
        self._activations = {}
        self.model(input_ids=full_ids, attention_mask=attention_mask)
        acts = self._activations[layer]  # [1, full_len, d_model]
        response_acts = acts[0, prompt_ids.shape[1]:, :]  # response tokens only
        return response_acts.mean(dim=0)  # [d_model]

    @torch.no_grad()
    def generate_with_hooks(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        hook_fns: Optional[dict[int, Callable]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[Tensor, dict[int, list[Tensor]]]:
        """
        Generate tokens while applying optional intervention hooks.
        Returns: (generated_ids, per_layer_activations)
        """
        if hook_fns:
            self.set_hooks(hook_fns)

        self._activations = {}
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if hook_fns:
            self.clear_hooks()

        return output, dict(self._activations)

    def tokenize(self, text: str, device: str = "cuda") -> dict:
        enc = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def build_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Build a tokenizer chat-template formatted prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size

    def remove_hooks(self):
        """Remove all registered PyTorch hooks (cleanup)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
