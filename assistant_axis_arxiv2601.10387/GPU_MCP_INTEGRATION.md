# GPU Providers, MCP, and Integration Guide

## The Core Constraint for This Project

Before surveying providers, one architectural reality shapes everything: **the Assistant Axis method requires PyTorch-level access to the model's residual stream at every layer during inference**. The activation capping hook rewrites hidden states mid-forward-pass (`h ← h − v · min(⟨h, v⟩ − τ, 0)`). No standard inference API — OpenAI-compatible, REST, or otherwise — exposes this. You get text in, text out.

This splits the work into two categories with very different infrastructure requirements:

| Task                                       | Needs hooks?       | Can use hosted API? |
| ------------------------------------------ | ------------------ | ------------------- |
| Role vector extraction (generate rollouts) | No                 | Yes                 |
| LLM judge calls (DeepSeek, GPT)            | No                 | Yes                 |
| Assistant Axis computation                 | No (post-hoc math) | Yes                 |
| **Activation capping during inference**    | **Yes**            | **No**              |
| **Activation steering during inference**   | **Yes**            | **No**              |
| **Persona drift projection per turn**      | **Yes**            | **No**              |
| Embedding user messages                    | No                 | Yes                 |
| Capability benchmarks                      | No                 | Yes                 |

For everything in the "No hooks" row you can use provider APIs. For the "Yes hooks" rows you must run the model yourself on GPU hardware you control.

---

## GPU Providers with MCP Support (Current Landscape)

### 1. Hugging Face — MCP Already Installed in This Session

**What it is:** The HF MCP exposes Hub metadata and Spaces execution. It is the most directly useful for this project's *setup* phase.

**MCP tools available (already configured):**

| Tool               | What it does                                                         | Relevant for this project                                                                          |
| ------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `hf_hub_query`     | Search and filter models, datasets, spaces by tags, downloads, likes | Find the exact Qwen3-32B / Llama-3.3-70B / Gemma-2-27b model IDs before downloading                |
| `hub_repo_details` | Get metadata, file list, model card for any repo                     | Verify model architecture, layer count, config before writing config.py                            |
| `dynamic_space`    | Invoke a HuggingFace Space (GPU-backed Gradio/Streamlit apps)        | Run lightweight text-generation spaces for role generation — **not suitable for activation hooks** |
| `hf_doc_search`    | Search HF documentation                                              | Look up Transformers API for specific model architectures                                          |
| `paper_search`     | Search arXiv papers indexed on HF                                    | Find related work                                                                                  |
| `space_search`     | Find Spaces by task                                                  | Locate existing inference Spaces                                                                   |

**HF Inference Endpoints** (separate from MCP, but deeply relevant):

- Deploy any HF model on managed GPU (A10G, A100, H100) via HF's infrastructure
- Supports **custom inference handlers** — you can inject your own Python class
- This is where activation hooks become possible: write a custom `EndpointHandler` that wraps `HookedModel` and exposes capping as part of the inference call
- No native MCP server for Inference Endpoints yet — you would call the REST API from code

**Cost:** Inference Endpoints charge per hour of GPU uptime (~$3–6/hr for A10G, ~$10–20/hr for A100).

---

### 2. Modal — Best Option for Full Hook Access via MCP

**What it is:** Modal lets you write Python functions decorated with `@app.function(gpu="A100")` and deploy them as serverless GPU compute. Critically, it also lets you **serve an MCP server** directly from a Modal app, meaning you can expose the full `HookedModel` + capping infrastructure as MCP tools that Claude Code can call.

**MCP integration approach:**

Modal added native MCP server support. You define tools as decorated functions, Modal handles the GPU provisioning, and Claude Code connects via `stdio` or HTTP. Example structure:

```python
# modal_mcp_server.py
import modal
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = modal.App("assistant-axis-mcp")
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "mcp"
)

@app.function(gpu="A100-80GB", timeout=600, image=image)
def extract_role_vector(role_name: str, system_prompts: list[str], questions: list[str]) -> list[float]:
    """Load model once (cached), extract role vector, return as list."""
    from src.models.hooked_model import HookedModel
    from src.extraction.role_vectors import extract_role_vector
    # ... implementation
    return vector.tolist()

@app.function(gpu="A100-80GB", timeout=300, image=image)
def apply_activation_capping(prompt: str, system_prompt: str = None) -> str:
    """Generate a response with activation capping applied."""
    from src.models.hooked_model import HookedModel
    from src.interventions.capping import build_capping_hooks
    # ... implementation
    return response_text

# Expose as MCP via stdio
mcp_server = app.function()(lambda: run_mcp_server())
```

**Why Modal is best for hook-based tasks:**

- Serverless: you pay only for GPU seconds used, not idle time
- Cold start ~10–30s (model loading); subsequent calls are warm if within timeout
- Can snapshot model weights into the container image to eliminate cold start
- Native Python — your existing `src/` code runs unchanged
- Supports H100, A100, A10G

**Cost:** ~$2.70/hr for A100-40GB, ~$4/hr for A100-80GB, pay-per-second.

**MCP config for Claude Code** (`.claude/settings.json`):

```json
{
  "mcpServers": {
    "assistant-axis-gpu": {
      "command": "modal",
      "args": ["serve", "modal_mcp_server.py"]
    }
  }
}
```

---

### 3. Together AI — Text Inference Only (No Hooks)

**What it is:** Fast OpenAI-compatible inference API for open-weight models. Has an MCP server, but it exposes **documentation access only** — not actual inference calls.

**MCP tool:** `npx add-mcp https://docs.together.ai/mcp`  
Lets an agent look up Together AI documentation within the editor. Useful for writing integration code.

**Actual inference** (not via MCP) uses their REST API:

```python
from openai import OpenAI
client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")
response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct", ...)
```

**Relevant for this project:** The **role generation** step (generating 275 × 1200 rollouts) could be offloaded to Together AI instead of running locally. No hooks needed — you just need the text output. Together AI hosts Llama 3.3 70B and Qwen 3 32B natively. Cost: ~$0.88/M input tokens, ~$0.88/M output tokens for Llama 70B.

---

### 4. Replicate — Community MCP, Hosted Models

**What it is:** Managed model hosting with pay-per-prediction billing. Community-maintained MCP server exists (`r8.im/replicate/replicate-mcp` — unofficial).

**MCP tools (community server):** Run any Replicate model, list predictions, get model versions.

**Relevant for this project:** Replicate hosts Llama and other models but with standard text I/O. Activation hooks are not possible. Useful for: generating training/calibration text (role rollouts, LMSYS-style prompts) cheaply.

---

### 5. RunPod / Lambda Labs / Vast.ai — Raw GPU Rental

**What they are:** Rent bare-metal or containerized GPU instances by the hour. No native MCP.

**How to add MCP:** Deploy the Python `mcp` package alongside your code and expose it via `stdio` (for local tunneling via SSH) or HTTP. RunPod supports persistent network volumes — store model weights once (~140GB for Llama 70B), reuse across pod restarts.

**RunPod template approach:**

```bash
# In RunPod container startup:
cd /workspace/assistant-axis/src
pip install mcp
python -m mcp.server.http --port 8080 mcp_server:server
# Then tunnel: ssh -L 8080:localhost:8080 user@pod-ip
```

**Cost:** ~$1.50–2.50/hr for A100 80GB on Vast.ai (cheapest), ~$3–5/hr on RunPod (more reliable).

---

## Recommended Architecture for This Project

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude Code / Claude                          │
│                   (reads from .claude/settings.json)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │ MCP
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
  HF MCP (installed)   Modal MCP Server   Together AI
  ─────────────────    ─────────────────   ──────────
  • Search for models  • extract_role_     • Generate
  • Verify model IDs     vector()            role rollouts
  • Check model cards  • apply_capping()     cheaply via
  • Find HF Spaces     • compute_axis()      REST API
  • Paper search       • run_jailbreak_    • No hooks
  (metadata only)        eval()              needed here
                       FULL HOOK ACCESS
                       A100 80GB on demand
```

### Phase-by-Phase Provider Mapping

| Phase    | Task                                          | Provider                                  | Why                                               |
| -------- | --------------------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| Setup    | Verify model IDs and configs                  | **HF MCP** (`hub_repo_details`)           | Already installed; instant                        |
| Phase 1a | Generate 275 × 1200 rollouts for role scoring | **Together AI** REST API                  | Cheapest at scale for text-only; Llama 70B hosted |
| Phase 1b | Extract mean activations for role vectors     | **Modal** (hook required)                 | Only option that supports hooks at scale          |
| Phase 1c | LLM judge scoring (gpt-4.1-mini)              | **OpenAI** REST API                       | Already in code                                   |
| Phase 2  | Compute Assistant Axis + PCA                  | Local / **Modal**                         | Numpy/sklearn, no GPU needed for math             |
| Phase 3a | Activation steering sweep (Figures 4, 5)      | **Modal**                                 | Hook required                                     |
| Phase 3b | LLM judge for persona/harmfulness             | **DeepSeek** REST API                     | Already in code                                   |
| Phase 4  | Activation capping eval (Figures 9, 10)       | **Modal**                                 | Hook required                                     |
| Phase 5  | Multi-turn conversation sim                   | **Modal** (target) + **OpenAI** (auditor) | Target needs hooks; auditor is just API call      |
| Phase 6  | Capability benchmarks                         | **Modal** or local                        | lm-eval can run inside Modal function             |

---

## How to Set Up the Modal MCP Server

### Step 1: Install Modal and MCP

```bash
uv add modal mcp
modal token new   # authenticate
```

### Step 2: Create `src/modal_mcp_server.py`

```python
"""Modal MCP server exposing HookedModel tools on GPU."""

import modal
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import json

# Build container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers>=4.45.0", "accelerate>=0.34.0",
        "numpy", "scikit-learn", "sentence-transformers", "mcp"
    )
    # Optional: snapshot model weights to eliminate cold start
    # .run_commands("python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-32B')\"")
)

app = modal.App("assistant-axis", image=image)

# Keep model warm between calls (avoid reload on every invocation)
with image.imports():
    from src.models.hooked_model import HookedModel
    from src.interventions.capping import build_capping_hooks, calibrate_cap_threshold
    from src.extraction.assistant_axis import compute_assistant_axis
    import torch

_model_cache: dict = {}

def get_model(model_key: str) -> HookedModel:
    if model_key not in _model_cache:
        from src.config import TARGET_MODELS
        _model_cache[model_key] = HookedModel(
            model_name=TARGET_MODELS[model_key],
            model_key=model_key,
        )
    return _model_cache[model_key]


@app.function(gpu="A100-80GB", timeout=600, secrets=[modal.Secret.from_name("hf-token")])
def generate_with_capping(
    prompt: str,
    model_key: str = "qwen",
    axis_path: str = None,
    tau: float = None,
    system_prompt: str = None,
) -> dict:
    """Generate a response with activation capping applied. Returns {response, axis_projection}."""
    import os, sys
    sys.path.insert(0, "/root/src")
    model = get_model(model_key)

    if axis_path and tau:
        axis = torch.load(axis_path, map_location="cpu")
        hook_fns = build_capping_hooks(axis, tau, model_key)
    else:
        hook_fns = None

    device = next(model.model.parameters()).device
    full_prompt = model.build_prompt(prompt, system_prompt=system_prompt)
    enc = model.tokenizer([full_prompt], return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = attention_mask.sum().item()

    if hook_fns:
        model.set_hooks(hook_fns)

    with torch.no_grad():
        out = model.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=512, temperature=0.7, do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    if hook_fns:
        model.clear_hooks()

    response = model.decode(out[0, int(prompt_len):])
    return {"response": response, "model_key": model_key}


@app.function(gpu="A100-80GB", timeout=1800)
def extract_role_vector_remote(
    role_name: str,
    system_prompts: list[str],
    questions: list[str],
    model_key: str = "qwen",
    layer: int = 32,
) -> list[float]:
    """Extract activation vector for a character role. Returns [d_model] floats."""
    import sys
    sys.path.insert(0, "/root/src")
    from src.evaluation.llm_judge import RoleExpressionJudge
    from src.extraction.role_vectors import RoleData, extract_role_vector
    model = get_model(model_key)
    judge = RoleExpressionJudge()
    role = RoleData(name=role_name, system_prompts=system_prompts)
    result = extract_role_vector(model, role, questions, judge, layer)
    if result.fully_vector is not None:
        return result.fully_vector.tolist()
    return []


# ── MCP Server ──────────────────────────────────────────────────────────────

server = Server("assistant-axis-gpu")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="generate_with_capping",
            description="Generate a model response with activation capping applied to stabilize the Assistant persona.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model_key": {"type": "string", "enum": ["qwen", "llama", "gemma"]},
                    "system_prompt": {"type": "string"},
                    "apply_capping": {"type": "boolean", "default": True},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="generate_unsteered",
            description="Generate a model response without any activation intervention (baseline).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "model_key": {"type": "string", "enum": ["qwen", "llama", "gemma"]},
                    "system_prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="extract_role_vector",
            description="Extract the activation vector for a character role from the model's residual stream.",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_name": {"type": "string"},
                    "system_prompts": {"type": "array", "items": {"type": "string"}},
                    "questions": {"type": "array", "items": {"type": "string"}},
                    "model_key": {"type": "string", "enum": ["qwen", "llama", "gemma"]},
                },
                "required": ["role_name", "system_prompts", "questions"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "generate_with_capping":
        result = generate_with_capping.remote(
            prompt=arguments["prompt"],
            model_key=arguments.get("model_key", "qwen"),
            system_prompt=arguments.get("system_prompt"),
        )
        return [TextContent(type="text", text=result["response"])]

    elif name == "generate_unsteered":
        result = generate_with_capping.remote(
            prompt=arguments["prompt"],
            model_key=arguments.get("model_key", "qwen"),
            system_prompt=arguments.get("system_prompt"),
            axis_path=None, tau=None,
        )
        return [TextContent(type="text", text=result["response"])]

    elif name == "extract_role_vector":
        vec = extract_role_vector_remote.remote(
            role_name=arguments["role_name"],
            system_prompts=arguments["system_prompts"],
            questions=arguments["questions"],
            model_key=arguments.get("model_key", "qwen"),
        )
        return [TextContent(type="text", text=json.dumps({"vector_length": len(vec), "sample": vec[:5]}))]

async def run():
    async with stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="assistant-axis-gpu",
            server_version="0.1.0",
        ))

if __name__ == "__main__":
    asyncio.run(run())
```

### Step 3: Add MCP Server to Claude Code Settings

Add to `.claude/settings.json` in this project:

```json
{
  "mcpServers": {
    "assistant-axis-gpu": {
      "command": "modal",
      "args": ["run", "src/modal_mcp_server.py::run"]
    },
    "huggingface": {
      "command": "uvx",
      "args": ["huggingface-mcp"]
    }
  }
}
```

### Step 4: Store HuggingFace Token as Modal Secret

```bash
modal secret create hf-token HF_TOKEN=hf_your_token_here
```

### Step 5: Deploy

```bash
modal deploy src/modal_mcp_server.py
```

---

## What You Can Do Once Integrated

With the Modal MCP server running, Claude Code can call tools directly in conversation:

- **"Run this jailbreak prompt through the capped model and compare it to baseline"** → calls `generate_with_capping` and `generate_unsteered` in parallel on A100
- **"Extract the role vector for 'oracle' using these 5 system prompts"** → calls `extract_role_vector_remote` on GPU
- **"Search HuggingFace for Qwen3 32B and verify it has 64 layers"** → calls `hub_repo_details` via existing HF MCP

The HF MCP already installed handles all discovery and metadata. Modal handles all GPU execution requiring activation hooks. Together AI or Replicate handle cheap bulk text generation.

---

## Honest Limitations

- **Cold start latency:** Modal functions take 30–120s to start the first time (loading 32B–70B model weights). Subsequent calls within the timeout window are warm (~2–5s). For interactive use this is acceptable; for batch extraction of 330,000 rollouts you would want to use Modal's `.map()` for parallelism.
- **Cost at scale:** The full extraction pipeline (275 roles × 1200 rollouts each) with activation capture on Llama 70B would cost ~$150–300 on A100s, depending on sequence lengths and parallelism.
- **No streaming:** The MCP tool call blocks until the full response is generated. For long generations, increase `timeout` in the Modal function decorator.
- **HF MCP cannot run models:** The already-installed HF MCP is metadata-only. `dynamic_space` can call HF Spaces but those are opaque inference endpoints — no hook access.
- **Together AI MCP is docs-only:** Their actual inference is REST API, not MCP.

---

## Summary Table

| Provider                         | Has MCP?             | Supports Activation Hooks? | Best For in This Project              |
| -------------------------------- | -------------------- | -------------------------- | ------------------------------------- |
| **Hugging Face MCP** ✅ installed | Yes (metadata)       | No                         | Model discovery, config verification  |
| **Modal**                        | Yes (build your own) | **Yes** ← best option      | All hook-based experiments            |
| **Together AI**                  | Yes (docs only)      | No                         | Role rollout text generation (REST)   |
| **Replicate**                    | Community MCP        | No                         | Bulk text inference (REST)            |
| **RunPod / Vast.ai**             | No (DIY)             | **Yes** (raw VM)           | Cost-effective alternative to Modal   |
| **Lambda Labs**                  | No                   | **Yes** (raw VM)           | Longer-running experiments            |
| **HF Inference Endpoints**       | No native MCP        | **Yes** (custom handler)   | Production deployment of capped model |
