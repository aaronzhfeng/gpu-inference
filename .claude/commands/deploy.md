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

Use AskUserQuestion. Tailor recommendations based on the detected GPU:

**Question:** "What model do you want to deploy? (Detected: <GPU_NAME>, <VRAM> GB)"
**Header:** "Model"
**Options** (include all known models, mark recommended based on GPU):
1. **Qwen3.5-27B-FP8** — ~27 GB VRAM. FP8 quantized, near-bf16 quality. Best quality available for single-GPU. Mark as "(Recommended)" if VRAM >= 40 GB.
2. **Qwen3.5-27B** — ~54 GB VRAM (bf16). Full precision. Mark as "(Recommended)" if VRAM >= 80 GB. For 2× GPU setups with 24 GB each, use with `TENSOR_PARALLEL=2`.
3. **Qwen3.5-9B** — ~17.7 GB VRAM. Thinking/reasoning capable. Mark as "(Recommended)" if VRAM is 24 GB.
4. **Qwen3.5-4B** — ~8.6 GB VRAM. Fast, no reasoning. Good for high-throughput batch work. Mark as "(Recommended)" if VRAM < 24 GB.
5. **New model from HuggingFace** — Search for and deploy a model not listed here.

> **"New model" is always the last option.** If more models have been added above, include them all.
> Adjust recommendations based on GPU. For 80 GB GPUs, suggest larger models. For multi-GPU, note tensor parallelism.

---

## If user picks an existing model:

Ask a follow-up question:

**Question:** "Enable thinking/reasoning mode?"
**Header:** "Thinking"
**Options:**
1. **Yes (Recommended for 9B+)** — Step-by-step reasoning in responses. Better quality, more tokens.
2. **No** — Direct answers only. Faster, fewer tokens.

Then proceed to **Step 3: Deploy**.

---

## If user picks "New model from HuggingFace":

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

### 1. Stop any running server:
```bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2
```

### 2. Download model if not cached:
```bash
source /workspace/.persist/venv/bin/activate
export HF_HOME=/workspace/models
huggingface-cli download <MODEL_ID>
```

### 3. Determine launch flags based on GPU + model:

**GPU memory utilization:**
- Model uses < 50% of VRAM → `0.90` (plenty of room for KV cache)
- Model uses 50-75% of VRAM → `0.85`
- Model uses > 75% of VRAM → `0.80` (tight, leave room for Mamba/KV cache)

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

### 4. Launch via start.sh:
```bash
MODEL=<MODEL_ID> \
ENABLE_THINKING=<true|false> \
ENFORCE_EAGER=<true|false> \
GPU_MEMORY_UTILIZATION=<value> \
MAX_MODEL_LEN=<value> \
EXTRA_ARGS="<any extra flags like --tensor-parallel-size 2>" \
bash /workspace/gpu-inference/start.sh &
```

### 5. Poll health every 30 seconds (up to 5 minutes):
```bash
curl -s http://localhost:8000/health
```

**Longer timeout for large models:** If model is > 20B params or multi-GPU, extend timeout to 10 minutes (large models take longer to load and compile).

### 6. Smoke test:
```bash
curl -s http://localhost:8000/v1/chat/completions \
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

### a. Run concurrency sweep:
```bash
source /workspace/.persist/venv/bin/activate
python /workspace/gpu-inference/benchmark.py
```

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

## Step 6: Register new model (only for "New model from HuggingFace")

After a successful deployment of a new model, **update this file** so future `/deploy` runs include it:

1. Read this file (`/workspace/gpu-inference/.claude/commands/deploy.md`)
2. In the **Step 1** options list, insert a new numbered option **before** "New model from HuggingFace" with:
   - Model name and ID
   - Approximate VRAM usage discovered during deployment
   - Key characteristics (reasoning capable, architecture type, etc.)
   - Any required flags discovered (e.g., `--enforce-eager`)
   - Which GPUs it was tested on
3. Add benchmark results to the **Reference baselines** table in Step 4
4. Update CLAUDE.md if the new model has notable quirks or launch requirements

This ensures the model list and performance data grow over time across different GPUs and models.
