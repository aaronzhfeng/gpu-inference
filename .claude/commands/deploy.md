Set up a model for inference. Follow this interactive flow:

## Step 0: Detect environment

Before asking anything, probe the GPU environment:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader
```

Parse results to determine:
- **GPU model** (e.g., RTX 4090, RTX A6000, A100-40GB, A100-80GB, H100-80GB, L40S, etc.)
- **Total VRAM** per GPU (in GB)
- **Number of GPUs** available
- **Free VRAM** (to detect if something else is already using the GPU)

Also check:
```bash
nproc                           # CPU cores
free -h                         # system RAM
df -h /workspace                # disk space available for model weights
hostname                        # for constructing proxy URLs
```

Use this info throughout the flow to make smart recommendations. Key VRAM tiers:

| GPU | VRAM | Good for |
|---|---|---|
| RTX 3090 / 4090 | 24 GB | Models up to ~14B (bf16) or ~28B (4-bit) |
| RTX A5000 | 24 GB | Same as 4090, slightly slower |
| RTX A6000 / L40S | 48 GB | Models up to ~28B (bf16) or ~70B (4-bit) |
| A100-40GB | 40 GB | Models up to ~22B (bf16), fast HBM |
| A100-80GB / H100-80GB | 80 GB | Models up to ~45B (bf16) or ~70B+ (4-bit) |
| Multi-GPU (any) | N×VRAM | Use tensor parallelism (`--tensor-parallel-size N`) |

---

## Step 1: Ask what the user wants

Use AskUserQuestion with **multiSelect: true** so users can deploy multiple models simultaneously.

Pick the **top 4 options** based on the detected GPU VRAM tier. Always include the full model catalog in the question text so users know what's available via "Other".

### Full Qwen3.5 model catalog

| Model | ID | VRAM (bf16) | Notes |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | ~1.7 GB | Tiny, very fast. Vision-capable. |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | ~4.2 GB | Small, fast. Good for simple tasks. |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | ~8.6 GB | Fast, no reasoning. High-throughput batch. |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | ~17.7 GB | Thinking/reasoning capable. Good quality. |
| Qwen3.5-27B-FP8 | `Qwen/Qwen3.5-27B-FP8` | ~27 GB | FP8 quantized, near-bf16 quality. |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | ~54 GB | Full precision bf16. |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` | ~7 GB active | MoE: 35B total, 3B active. Very fast. |
| Qwen3.5-35B-A3B-FP8 | `Qwen/Qwen3.5-35B-A3B-FP8` | ~18 GB | MoE FP8. Good for 24 GB GPUs. |
| Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | ~65 GB | MoE: 122B total, 10B active. |
| Qwen3.5-122B-A10B-FP8 | `Qwen/Qwen3.5-122B-A10B-FP8` | ~65 GB | MoE FP8. Needs 80 GB GPU. |

### Option selection by GPU VRAM tier

**VRAM < 24 GB:**
1. Qwen3.5-4B (Recommended) — ~8.6 GB. Fast, no reasoning.
2. Qwen3.5-2B — ~4.2 GB. Smaller, faster.
3. Qwen3.5-0.8B — ~1.7 GB. Tiny, fastest. Vision-capable.
4. Qwen3.5-35B-A3B — ~7 GB active (MoE). 35B quality, 3B speed.

**VRAM 24 GB:**
1. Qwen3.5-9B (Recommended) — ~17.7 GB. Thinking/reasoning capable.
2. Qwen3.5-4B — ~8.6 GB. Fast, no reasoning.
3. Qwen3.5-35B-A3B-FP8 — ~18 GB (MoE FP8). High quality, fast.
4. Qwen3.5-0.8B — ~1.7 GB. Tiny auxiliary model.

**VRAM 40–48 GB:**
1. Qwen3.5-27B-FP8 (Recommended) — ~27 GB. Best quality for single-GPU.
2. Qwen3.5-9B — ~17.7 GB. Thinking/reasoning.
3. Qwen3.5-4B — ~8.6 GB. Fast auxiliary model.
4. Qwen3.5-0.8B — ~1.7 GB. Tiny auxiliary model.

**VRAM 80+ GB:**
1. Qwen3.5-27B (Recommended) — ~54 GB. Full precision bf16.
2. Qwen3.5-27B-FP8 — ~27 GB. Leaves room for a second model.
3. Qwen3.5-122B-A10B-FP8 — ~65 GB (MoE). Largest quality.
4. Qwen3.5-9B — ~17.7 GB. Fast auxiliary model.

**Question format:** "Which models to deploy? Select one or more. (Detected: <GPU_NAME>, <VRAM> GB). Other models available via 'Other': <list remaining models not in options>"
**Header:** "Models"
**multiSelect:** true

> Users can pick multiple models (e.g., 27B-FP8 + 0.8B). The deploy flow handles multi-model by assigning different ports and splitting GPU memory.
> "Other" (auto-provided by AskUserQuestion) covers custom HuggingFace models and any catalog models not in the top 4.

---

## Step 2: Configure thinking mode

For each selected model, ask about thinking/reasoning. If only one model was selected, use a simple question. If multiple, use multiSelect to indicate which models should have thinking enabled.

**For single model:**
- **Question:** "Enable thinking/reasoning mode?"
- **Header:** "Thinking"
- **Options:**
  1. **Yes (Recommended for 9B+)** — Step-by-step reasoning in responses. Better quality, more tokens.
  2. **No** — Direct answers only. Faster, fewer tokens.

**For multiple models:**
- **Question:** "Enable thinking for which models?"
- **Header:** "Thinking"
- **multiSelect:** true
- **Options:** List each selected model as an option. Mark models 9B+ as "(Recommended)". For 0.8B/2B/4B, add "(not recommended — too small for useful reasoning)" in description.

**Default thinking recommendations:**
- 0.8B, 2B, 4B: thinking OFF (too small, quality degrades)
- 9B, 27B, 27B-FP8, 35B-A3B, 122B-A10B: thinking ON

Then proceed to **Step 3: Deploy**.

---

## If user picks "Other" (new model from HuggingFace):

1. Ask the user for the model name/ID (e.g. `Qwen/Qwen3-8B`, `mistralai/Mistral-7B-Instruct-v0.3`)
2. Search HuggingFace for the model:
   ```bash
   source /workspace/.persist/venv/bin/activate
   python -c "from huggingface_hub import model_info; info = model_info('MODEL_ID'); print(f'Model: {info.modelId}'); print(f'Size: {info.safetensors.total if info.safetensors else \"unknown\"} bytes'); print(f'Pipeline: {info.pipeline_tag}'); print(f'Library: {info.library_name}'); print(f'Tags: {info.tags}')"
   ```
3. Provide the user the HuggingFace link: `https://huggingface.co/<model-id>` so they can review the model card manually
4. Check vLLM compatibility:
   ```bash
   python -c "
   from vllm.config import ModelConfig
   try:
       cfg = ModelConfig(model='MODEL_ID', max_model_len=4096, trust_remote_code=True)
       print(f'Architecture: {cfg.hf_config.architectures}')
       print(f'Dtype: {cfg.dtype}')
       print('vLLM compatibility: OK')
   except Exception as e:
       print(f'vLLM compatibility: FAILED - {e}')
   "
   ```
5. **Estimate VRAM and check fit:**
   - Rough formula: `params_billions × 2 GB` for bf16, `params_billions × 0.5 GB` for 4-bit quantized
   - Compare against detected GPU VRAM
   - If model won't fit on one GPU but multiple GPUs are available, recommend `--tensor-parallel-size N`
   - If model won't fit at all, warn the user and suggest alternatives (smaller variant, quantized version)
   - Check disk space: model weights are roughly same size as VRAM usage
6. Report findings to the user:
   - Model name, size, architecture
   - HuggingFace link for manual review
   - vLLM compatibility status
   - Estimated VRAM vs available VRAM
   - Fit assessment: "Fits easily" / "Tight fit" / "Won't fit — need quantization or bigger GPU"
   - Multi-GPU recommendation if applicable
7. Ask for confirmation before proceeding

If confirmed, proceed to **Step 3: Deploy**.

---

## Step 3: Deploy

### 0. Multi-model port and memory planning

When deploying multiple models, plan port assignments and GPU memory splits **before** launching anything.

**Port assignment:**
- First/largest model: port 8000
- Second model: port 8002 (port 8001 is often used by nginx on RunPod)
- Third model: port 8003
- Fourth model: port 8004, etc.

**GPU memory splitting:**
When multiple models share a GPU, `gpu_memory_utilization` values must sum to ≤ 0.95 (leave 5% headroom).

Calculate each model's share based on its weight size relative to total VRAM:
1. Sum all model weight sizes (e.g., 27 GB + 1.7 GB = 28.7 GB)
2. Each model gets: `(model_weight_size / total_VRAM) + KV_cache_bonus`
3. KV cache bonus: give larger models ~0.05 extra, smaller models ~0.10 extra (small models benefit more from proportionally larger KV cache)
4. Verify sum ≤ 0.95

**CRITICAL: KV cache must be large enough for concurrent requests.** If the KV cache is too small, vLLM will serialize requests (Running: 1, Waiting: N) and throughput won't scale with concurrency. Each concurrent request needs `max_model_len × per_token_KV_size` bytes of cache. Prioritize KV cache for the larger model — it's usually the bottleneck.

**Also consider reducing `max_model_len`** when co-hosting. 2048 instead of 4096 halves per-request KV usage, doubling concurrent capacity.

**Example for RTX A6000 (48 GB) with 27B-FP8 + 0.8B:**
- 27B-FP8: 0.80 (36.8 GB — 27 GB model + 4.75 GB KV cache for ~9 concurrent reqs at max_model_len=2048)
- 0.8B: 0.15 (6.9 GB — 1.7 GB model + 5.2 GB KV cache)
- Total: 0.95 — leaves 2.3 GB headroom
- Result: 227 tok/s at c=64 (vs 14.5 tok/s with gpu_mem_util=0.70 and max_model_len=4096)

### 1. Stop any running servers:
```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
```

### 2. Download all models (in parallel if multiple):
```bash
source /workspace/.persist/venv/bin/activate
export HF_HOME=/workspace/models
huggingface-cli download <MODEL_ID_1> &
huggingface-cli download <MODEL_ID_2> &
wait
```

### 3. Determine launch flags for each model:

**GPU memory utilization:**
- Single model, uses < 50% of VRAM → `0.90`
- Single model, uses 50-75% of VRAM → `0.85`
- Single model, uses > 75% of VRAM → `0.80`
- Multi-model: use the splitting formula from Step 3.0

**Enforce eager (skip CUDA graph capture):**
- Default: `false` (CUDA graphs are faster)
- Set `true` if: model uses hybrid architecture (e.g., Mamba-attention like Qwen3.5) AND VRAM is tight (model > 70% of VRAM). This avoids CUDA graph capture assertion errors.
- Known requirement: Qwen3.5-9B on RTX 4090 (24 GB) needs `--enforce-eager`

**Tensor parallelism (multi-GPU):**
- If model fits on 1 GPU: don't use TP (overhead not worth it)
- If model needs multiple GPUs: add `--tensor-parallel-size N` where N = number of GPUs needed
- Rule of thumb: distribute so each GPU shard uses < 80% of per-GPU VRAM

**Max model length:**
- Default: 4096
- Increase if GPU has headroom (e.g., 8192 or 16384 on A100-80GB with a small model)
- Decrease if VRAM is tight and you need more concurrent requests

**Quantization:**
- Pre-quantized FP8 models (e.g., `Qwen/Qwen3.5-27B-FP8`): use `QUANTIZATION=fp8` — these are official Qwen FP8 checkpoints with near-identical quality to bf16
- If model is too large for available VRAM in bf16, try `--quantization awq` or `--quantization gptq` (model must have quantized variant on HuggingFace)
- Alternative: use `--quantization fp8` on H100/L40S (hardware FP8 support)

### 4. Launch models sequentially via start.sh:

Launch the **largest model first** (it takes longest and claims GPU memory first), then launch smaller models.

```bash
# Model 1 (largest)
MODEL=<MODEL_ID_1> \
ENABLE_THINKING=<true|false> \
ENFORCE_EAGER=<true|false> \
GPU_MEMORY_UTILIZATION=<value> \
MAX_MODEL_LEN=<value> \
PORT=8000 \
EXTRA_ARGS="<any extra flags>" \
bash /workspace/gpu-inference/start.sh &
```

Wait for the first model to be healthy before launching the next (to avoid GPU memory contention during loading):

```bash
# Poll until healthy, then launch model 2
MODEL=<MODEL_ID_2> \
PORT=8002 \
GPU_MEMORY_UTILIZATION=<value> \
... \
bash /workspace/gpu-inference/start.sh &
```

### 5. Poll health for each model:

Use HTTP status code check (not body grep — vLLM returns empty body with 200):
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/health
```

Poll every 15-30 seconds. Timeouts:
- Models < 10B: 5 minutes
- Models 10-30B: 10 minutes
- Models > 30B or multi-GPU: 15 minutes

### 6. Smoke test each model:
```bash
curl -s http://localhost:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":64}'
```
- Verify `"choices"` with non-empty `"content"`
- If thinking enabled, confirm reasoning appears
- If test fails, troubleshoot

### 7. If launch fails — auto-recover:

Common failures and fixes to try automatically:

| Error | Fix |
|---|---|
| `AssertionError: num_cache_lines >= batch` (CUDA graph Mamba bug) | Retry with `ENFORCE_EAGER=true` |
| `OutOfMemoryError` / CUDA OOM | Lower `GPU_MEMORY_UTILIZATION` by 0.05, or reduce `MAX_MODEL_LEN` |
| `torch.OutOfMemoryError` during model load | Model doesn't fit — suggest quantization or smaller model |
| Timeout (no health after 10 min) | Check `nvidia-smi` for GPU usage. If GPU at 100% util, still loading. If idle, check logs. |
| `ValueError: ... not supported` | Architecture not in vLLM — suggest trying SGLang or a different model |

If auto-recovery fails after 2 attempts, stop and report to the user with diagnostics.

---

## Step 4: Benchmark — find optimal concurrency

Run after successful deployment to characterize performance on this specific GPU.

### a. Run concurrency sweep for each deployed model:
```bash
source /workspace/.persist/venv/bin/activate
# For each model, run benchmark against its port:
python /workspace/gpu-inference/benchmark.py --port <PORT>
```

If benchmark.py doesn't support `--port`, use an inline script that hits the correct port (see the 0.8B benchmark example in conversation history), or temporarily modify the URL constant.

### b. Analyze results:
- Find concurrency with highest tok/s
- Note where failures start or throughput plateaus
- Compare against known baselines:

**Reference baselines (128 max tokens):**

| GPU | Model | Quantization | Peak tok/s | Best concurrency | Notes |
|---|---|---|---|---|---|
| RTX 4090 (24 GB) | Qwen3.5-4B | bf16 | ~2,900 | 64 | thinking disabled |
| RTX 4090 (24 GB) | Qwen3.5-9B | bf16 | TBD | TBD | enforce-eager required |
| A100-40GB | Qwen3.5-27B-FP8 | fp8 | TBD | TBD | |
| A100-80GB | Qwen3.5-27B | bf16 | TBD | TBD | |
| A100-80GB | Qwen3.5-27B-FP8 | fp8 | TBD | TBD | |
| H100-80GB | Qwen3.5-27B-FP8 | fp8 | TBD | TBD | |
| A40 (48 GB) | Qwen3.5-27B-FP8 | fp8 | ~398 | 32–64 | enforce-eager required, ~37.5 GB VRAM |
| A40 (48 GB) | Qwen3.5-9B | bf16 | ~1,238 | 64 | CUDA graphs OK, ~17.7 GB model + large KV cache |
| A40 (48 GB) | Qwen3.5-0.8B | bf16 | ~4,935 | 64 | co-hosted with 27B-FP8, gpu_mem_util=0.28 |
| RTX A6000 (48 GB) | Qwen3.5-27B-FP8 | fp8 | ~227 | 64 | co-hosted with 0.8B, gpu_mem_util=0.80, max_model_len=2048, enforce-eager required |
| RTX A6000 (48 GB) | Qwen3.5-0.8B | bf16 | ~3,201 | 64 | co-hosted with 27B-FP8, gpu_mem_util=0.15 |

> Fill in TBD entries as you benchmark on new GPUs. Add new rows for new GPU + model combos.

If results are significantly below expectations:
- Check `nvidia-smi` — is GPU actually being used?
- Check if `--enforce-eager` is unnecessarily on (disables CUDA graphs, ~20-40% perf hit)
- Try increasing `--gpu-memory-utilization` if VRAM headroom exists
- Try increasing `--max-model-len` if throughput is bottlenecked by short KV cache

### c. Record results:
Save benchmark results as part of the deployment info (see Step 5).

---

## Step 5: Write deployment info + report

### a. Write `/workspace/gpu-inference/DEPLOYED.md`:

```markdown
# Currently Deployed Model

| Field | Value |
|---|---|
| Model | `<MODEL_ID>` |
| Base URL (local) | `http://localhost:8000/v1` |
| Base URL (external) | `https://<pod-id>-8000.proxy.runpod.net/v1` |
| API key | `unused` (any string, but required by OpenAI SDK) |
| Max context | <MAX_MODEL_LEN> tokens |
| Thinking | <enabled/disabled> |
| VRAM usage | ~<X> GB |
| GPU | <GPU name> × <count> (<VRAM> GB each) |
| Tensor parallel | <N or "no"> |
| Launch flags | `ENFORCE_EAGER=<val> GPU_MEMORY_UTILIZATION=<val>` |
| Deployed at | <current UTC timestamp> |

## Performance (benchmarked on deploy)

| Concurrency | Throughput (tok/s) | Avg Latency (s) | Status |
|---|---|---|---|
| 1 | <X> | <X> | ok |
| 4 | <X> | <X> | ok |
| ... | ... | ... | ... |

**Recommended max concurrency:** <N>
**Peak throughput:** <X> tok/s

## Python (OpenAI SDK)

from openai import OpenAI
client = OpenAI(base_url="https://<pod-id>-8000.proxy.runpod.net/v1", api_key="unused")
response = client.chat.completions.create(
    model="<MODEL_ID>",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

## curl

curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

Fill in all `<placeholders>` with actual values.

**External URL detection:**
- RunPod: `https://<pod-id>-8000.proxy.runpod.net/v1` (get pod ID from `hostname` or `RUNPOD_POD_ID` env var)
- Vast.ai: check `VAST_TCP_PORT_8000` env var or the Vast dashboard
- Local/other: `http://<public-ip>:8000/v1`
- If unsure, report local URL and ask user for the external URL

### b. Report to user:
Print the full deployment info (same as DEPLOYED.md) directly in the response so the user can copy-paste it to other agents.

---

## Step 6: Register new model or GPU results

### For "Other" (custom HuggingFace model):
After a successful deployment, **update this file** so future `/deploy` runs include it:

1. Read this file (`/workspace/gpu-inference/.claude/commands/deploy.md`)
2. Add the model to the **Full Qwen3.5 model catalog** table in Step 1
3. Add it to the relevant **VRAM tier option lists** in Step 1
4. Add benchmark results to the **Reference baselines** table in Step 4
5. Update CLAUDE.md if the new model has notable quirks or launch requirements

### For existing catalog models on a new GPU:
If you deployed a known model on a GPU that doesn't have a baseline yet:

1. Add a new row to the **Reference baselines** table in Step 4 with the GPU + model combo
2. Include co-hosting notes if applicable (e.g., "co-hosted with 0.8B, gpu_mem_util=0.70")

This ensures the model list and performance data grow over time across different GPUs and models.
