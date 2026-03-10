Bootstrap this workspace for GPU inference. Run the following steps in order, stopping on any failure:

## 1. System packages

```bash
apt-get update -qq && apt-get install -y -qq nvtop
```

## 2. Python venv

Check if `/workspace/.persist/venv/bin/activate` exists.

- If YES: activate it and verify vllm is installed (`python -c "import vllm; print(vllm.__version__)"`)
- If NO: create it and install dependencies:

```bash
python3 -m venv /workspace/.persist/venv
source /workspace/.persist/venv/bin/activate
pip install --upgrade pip
pip install vllm huggingface_hub[cli] --no-cache-dir
```

## 3. Model download

Activate the venv, set `HF_HOME=/workspace/models`, then check if Qwen/Qwen3.5-4B is cached:

```bash
source /workspace/.persist/venv/bin/activate
export HF_HOME=/workspace/models
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"
```

If not cached, download it:

```bash
huggingface-cli download Qwen/Qwen3.5-4B
```

## 4. Launch inference server

Run the server in the background using `start.sh` defaults (Qwen3.5-4B, port 8000, thinking disabled):

```bash
source /workspace/.persist/venv/bin/activate
export HF_HOME=/workspace/models
bash /workspace/gpu-inference/start.sh &
```

Wait up to 5 minutes for the server to become healthy. Poll with:

```bash
curl -s http://localhost:8000/health
```

## 5. Verify and report

Once healthy, hit `/v1/models` and report back to the user with:
- Confirmation that the server is running
- The base URL: `http://localhost:8000/v1`
- The model name: `Qwen/Qwen3.5-4B`
- Thinking mode: disabled
- A sample curl command they can copy-paste to test

If any step fails, stop and report the error clearly so the user can fix it.
