#!/bin/bash
# Workspace bootstrap — set as RunPod Start Command or run manually.
# Restores Claude Code + Python venv from persistent /workspace, then launches inference.
#
# Usage:
#   bash /workspace/gpu-inference/setup-workspace.sh                # default: setup + inference
#   bash /workspace/gpu-inference/setup-workspace.sh --no-inference  # setup only
set -e

PERSIST="/workspace/.persist"
VENV="$PERSIST/venv"

echo "=== Workspace Setup ==="

# ── System packages ──────────────────────────────────────────
echo "  Installing system packages..."
apt-get update -qq && apt-get install -y -qq nvtop libnuma1 2>/dev/null
echo "  System packages ready"

# ── Python venv (vllm + deps) ──────────────────────────────
if [ -d "$VENV/bin" ]; then
    echo "  Python venv ready"
else
    echo "  Creating Python venv and installing vllm..."
    python -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install vllm huggingface_hub[cli] aiohttp
    echo "  vllm installed"
fi

# ── Claude Code binary ──────────────────────────────────────
mkdir -p /root/.local/share/claude/versions /root/.local/bin

if [ -d "$PERSIST/claude-data/versions" ]; then
    LATEST=$(ls "$PERSIST/claude-data/versions" 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST" ]; then
        cp -a "$PERSIST/claude-data/versions/$LATEST" \
              /root/.local/share/claude/versions/"$LATEST"
        ln -sf /root/.local/share/claude/versions/"$LATEST" /root/.local/bin/claude
        echo "  Claude Code $LATEST restored"
    fi
else
    echo "  No cached Claude Code found — installing fresh..."
    npm install -g @anthropic-ai/claude-code
    mkdir -p "$PERSIST/claude-data/versions"
    LATEST=$(ls /root/.local/share/claude/versions 2>/dev/null | sort -V | tail -1)
    [ -n "$LATEST" ] && cp -a /root/.local/share/claude/versions/"$LATEST" \
                                "$PERSIST/claude-data/versions/$LATEST"
fi

# ── Claude config (credentials, settings, history) ──────────
if [ -d "$PERSIST/claude-config" ]; then
    rm -rf /root/.claude
    ln -sf "$PERSIST/claude-config" /root/.claude
    echo "  Claude config linked"
else
    mkdir -p "$PERSIST/claude-config"
    rm -rf /root/.claude
    ln -sf "$PERSIST/claude-config" /root/.claude
    echo "  Fresh config dir created — run 'claude' to authenticate"
fi

# ── PATH ────────────────────────────────────────────────────
if ! grep -q '.local/bin' /root/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
fi
export PATH="$HOME/.local/bin:$PATH"

echo "  claude: $(claude --version 2>/dev/null || echo 'not installed')"
echo "=== Workspace Ready ==="

# ── Inference server (default: on) ──────────────────────────
if [ "$1" = "--no-inference" ]; then
    echo "Skipping inference server."
    sleep infinity
else
    echo ""
    echo "Launching inference server..."
    exec bash /workspace/gpu-inference/start.sh
fi
