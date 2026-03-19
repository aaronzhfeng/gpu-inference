# CLAUDE.md

## Project Overview

`gpu-inference` is a minimal Docker setup for self-hosted LLM inference on GPU clouds (RunPod, Vast.ai, or local). Supports **vLLM** and **SGLang** as interchangeable backends, both exposing an OpenAI-compatible `/v1/chat/completions` API.

Primary use case: serving Qwen3.5 small models (0.8B–9B) for high-concurrency batch experiments, replacing OpenRouter API calls.

## Structure

```
init.sh              # Run first on any new pod: installs Claude Code, restores ephemeral state
start.sh             # Launch inference: cache check → model download → start engine
Dockerfile           # CUDA 12.4 + vLLM + SGLang, single image (for Docker-based deploys)
models.yaml          # Reference presets (model sizes, VRAM estimates)
.env.example         # Environment variable template
```

## How It Works

1. All config via env vars: `ENGINE`, `MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `PORT`, `ENABLE_THINKING`, `EXTRA_ARGS`
2. `init.sh` restores ephemeral state (Claude Code, system packages, PATH) on any new pod
3. `/setup` slash command bootstraps the full environment (Python venv, model download)
4. `start.sh` checks if model weights are cached at `$HF_HOME` (`/workspace/models` by default)
4. Downloads from HuggingFace if not cached
5. Launches vLLM or SGLang based on `$ENGINE`
6. Both engines serve the same OpenAI-compatible API on `$PORT`
7. `ENABLE_THINKING` defaults to `false` — Qwen3.5 reasoning/`<think>` blocks are disabled. Set to `true` to enable.

## RunPod Deployment

- Use a **network volume** mounted at `/workspace` so model weights persist across pod restarts
- Recommended GPU: **RTX 4090** ($0.44/hr) for Qwen3.5-4B/9B, **A40** ($0.38/hr) for Qwen3.5-27B-FP8
- First boot: ~2min (model download). Subsequent boots: ~30s (cached)
- Build and push the Docker image, create a RunPod template pointing to it

## Integration with RRMC

This server is consumed by the [RRMC](https://github.com/aaronzhfeng/RRMC) project:
- RRMC's `configs/providers/selfhosted.yaml` points at this server's URL
- Use `--provider selfhosted --base_url http://<pod-ip>:8000/v1` when running RRMC experiments
- The OpenAI SDK client in RRMC's `LLMWrapper` works identically with vLLM/SGLang endpoints

## Quick Deploy: Qwen3.5-9B (With Reasoning)

### Prerequisites
- venv at `/workspace/.persist/venv` with `vllm` and `huggingface_hub[cli]` installed
- Model weights cached at `/workspace/models` (HF_HOME)
- GPU: RTX 4090 (24 GB) — model uses ~17.7 GB VRAM

### Launch Command

```bash
source /workspace/.persist/venv/bin/activate && \
export HF_HOME=/workspace/models && \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --swap-space 2 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager
```

**Key flags:**
- `--enforce-eager` — **required** for Qwen3.5-9B on RTX 4090. Without it, vLLM 0.17.0 crashes during CUDA graph capture with `AssertionError: num_cache_lines >= batch` in `causal_conv1d_update` (Mamba state cache too small after 17.7 GiB model load)
- `--gpu-memory-utilization 0.85` — slightly lower than default to leave headroom for Mamba state cache
- Thinking/reasoning is enabled by default (no `--default-chat-template-kwargs` needed)

### Qwen3.5-4B Alternative (No Reasoning)

For the smaller 4B model without thinking, use:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --swap-space 2 \
    --port 8000 \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```
- `--default-chat-template-kwargs '{"enable_thinking": false}'` disables `<think>` blocks
- Do NOT use `--chat-template-kwargs` (invalid in vLLM 0.17.0)
- 4B does NOT need `--enforce-eager` (only ~8.6 GB VRAM, enough room for CUDA graphs)

### Startup Time
- ~3 min on first launch (model load + profiling)
- `--enforce-eager` skips torch.compile and CUDA graph capture, so startup is slightly faster

### Verify It's Running

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/health
```

### Connect to the Endpoint

| Field | Value |
|---|---|
| Base URL | `http://localhost:8000/v1` |
| Model name | `Qwen/Qwen3.5-9B` |
| Max context | 4,096 tokens |
| Thinking | Enabled (reasoning in responses) |
| Auth | None (no API key required) |

**Chat completions:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Using start.sh (Recommended)

`start.sh` now has native `ENABLE_THINKING` support (defaults to `false`):

```bash
# Default: no reasoning, Qwen3.5-4B, port 8000
bash /workspace/gpu-inference/start.sh

# With reasoning enabled
ENABLE_THINKING=true bash /workspace/gpu-inference/start.sh

# Different model
MODEL=Qwen/Qwen3.5-9B ENABLE_THINKING=true bash /workspace/gpu-inference/start.sh
```

## New Workspace Bootstrap (From Scratch)

### Any new pod (first time or returning):

```bash
cd /workspace/gpu-inference
./init.sh
claude
# first time: type /setup (installs venv, downloads model, launches server)
# returning:  type /deploy (pick a model and launch)
```

`init.sh` installs Claude Code, restores system packages, links config, and hooks itself into `.bashrc` for future logins. It's the only script you need to run manually.

### What `/setup` does (first time only):
1. Creates Python venv with vllm at `/workspace/.persist/venv/`
2. Runs `/deploy` to pick a model, download it, launch, and benchmark


### What persists across workspace resets (on /workspace disk):
- `/workspace/.persist/venv/` — Python venv with vllm
- `/workspace/.persist/claude-config/` — Claude credentials & settings
- `/workspace/gpu-inference/init.sh` — auto-init script (restores ephemeral state on new pods)
- `/workspace/models/` — HuggingFace model weights cache
- `/workspace/gpu-inference/` — this repo (scripts, config, docs)

### What auto-restores on new pods (via init.sh):
After first-time setup, `/workspace/gpu-inference/init.sh` is sourced from `.bashrc` on every new pod. It automatically restores:
- System packages (`nvtop`, `libnuma1`)
- Claude Code (via `curl -fsSL https://claude.ai/install.sh | bash`)
- Claude config symlink (`~/.claude` → `/workspace/.persist/claude-config`)
- PATH (`~/.local/bin`)

**You do NOT need to re-run `/setup` when switching GPUs** — just open a shell (init.sh auto-fires) and everything is ready. Only run `/setup` on a brand-new network volume.

### What does NOT persist (but auto-restores):
- System packages (`nvtop`, etc.) — reinstalled by `init.sh` on first login
- `.bashrc` init hook — must exist for auto-restore to work (self-installed by `init.sh` on first run)

## Engine Choice: vLLM vs SGLang

**Use vLLM (default).** Benchmarked 2026-03-10 on RTX 4090 with Qwen3.5-4B, 128 max tokens:

| Concurrency | SGLang (tok/s) | vLLM (tok/s) | vLLM speedup |
|---|---|---|---|
| 1 | 90 | 95 | 1.1x |
| 8 | 530 | 615 | 1.2x |
| 16 | 974 | 1,157 | 1.2x |
| 32 | 970 | 1,969 | 2.0x |
| 64 | 1,147 | 2,951 | 2.6x |

vLLM is 1.2–2.6x faster, especially at high concurrency. Likely because Qwen3.5 is a **hybrid Mamba-attention architecture** (not pure transformer), and vLLM 0.17.0 has better torch.compile + CUDA graph support for it.

### SGLang caveats (if you still want to try it):
- SGLang and vLLM have **conflicting torch versions** (SGLang needs 2.9.1, vLLM needs 2.10.0). Installing both in one venv breaks one of them. Use separate venvs if needed.
- SGLang requires extra system deps: `libnuma1`, `nvidia-cudnn-cu12>=9.16`
- SGLang has no `--default-chat-template-kwargs` flag. To disable thinking, use `--chat-template` with the patched template at `chat_templates/qwen3.5-no-thinking.jinja`
- SGLang binds to `127.0.0.1` by default (not `0.0.0.0`); add `--host 0.0.0.0` for external access
- The 50 GB network volume quota is tight with both engines installed (~23 GB venv + 27 GB models)

## GPU Selection Guide (RunPod, as of 2026-03-10)

Pick GPU based on which model you want to run. Sorted by value (bandwidth per dollar).

### For Qwen3.5-4B / 9B (need 24 GB VRAM)

| GPU | VRAM | Mem BW | $/hr | Notes |
|---|---|---|---|---|
| **RTX A5000** | 24 GB | 768 GB/s | $0.16 | Cheapest. Great for 4B. |
| **RTX 3090** | 24 GB | 936 GB/s | $0.22 | Best speed/$ for 9B. |
| RTX 4090 | 24 GB | 1,008 GB/s | $0.34 | Fastest 24 GB. |
| L4 | 24 GB | 300 GB/s | $0.44 | Slow, skip. |

### For Qwen3.5-27B-FP8 (need 48 GB VRAM)

| GPU | VRAM | Mem BW | $/hr | Notes |
|---|---|---|---|---|
| **RTX A6000** | 48 GB | 768 GB/s | $0.33 | Best value. |
| A40 | 48 GB | 696 GB/s | $0.35 | Slightly worse than A6000. |
| RTX 6000 Ada | 48 GB | 960 GB/s | $0.74 | Fastest 48 GB. |
| L40S | 48 GB | 864 GB/s | $0.79 | Overpriced. |

### For Qwen3.5-27B bf16 or larger (need 80+ GB VRAM)

| GPU | VRAM | Mem BW | $/hr | Notes |
|---|---|---|---|---|
| **A100 PCIe** | 80 GB | 2,039 GB/s | $1.19 | Best value 80 GB. |
| A100 SXM | 80 GB | 2,039 GB/s | $1.39 | NVLink for multi-GPU. |
| H100 SXM | 80 GB | 3,350 GB/s | $2.69 | Max speed. |

### Quick recommendation

| Model | Best GPU | $/hr |
|---|---|---|
| Qwen3.5-4B | RTX A5000 | $0.16 |
| Qwen3.5-9B | RTX 3090 | $0.22 |
| Qwen3.5-27B-FP8 | RTX A6000 | $0.33 |
| Qwen3.5-27B (bf16) | A100 PCIe | $1.19 |

> LLM inference is memory-bandwidth bound. More bandwidth = more tok/s. Prices from RunPod on-demand (March 2026), may vary.

## Development Notes

- No test suite. Validate by starting the container and hitting the `/v1/models` endpoint
- `EXTRA_ARGS` env var passes arbitrary CLI flags to the engine (e.g., `--quantization awq`)
- Benchmark script: `/workspace/gpu-inference/benchmark.py` — async concurrency sweep against the running server
