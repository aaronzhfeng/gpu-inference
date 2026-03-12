Bootstrap this workspace for GPU inference. Run the following steps in order, stopping on any failure:

## 1. System packages

```bash
apt-get update -qq && apt-get install -y -qq nvtop libnuma1 2>/dev/null
```

## 2. Python venv

Check if `/workspace/.persist/venv/bin/activate` exists.

- If YES: activate it and verify vllm is installed (`python -c "import vllm; print(vllm.__version__)"`)
- If NO: create it and install dependencies:

```bash
python3 -m venv /workspace/.persist/venv
source /workspace/.persist/venv/bin/activate
pip install --upgrade pip
pip install vllm huggingface_hub[cli] aiohttp
```

If the venv already exists but `aiohttp` is missing, install it:
```bash
source /workspace/.persist/venv/bin/activate
pip install aiohttp 2>/dev/null
```

## 3. Deploy a model

After dependencies are ready, run the `/deploy` slash command to interactively pick a model, deploy it, benchmark it, and report connection info.

Tell the user: "Dependencies are ready. Running /deploy to set up a model..."

Then execute `/deploy`.
