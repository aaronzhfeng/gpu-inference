#!/bin/bash
# First script to run on any new pod. Gets Claude Code + ephemeral deps ready.
# Usage:
#   cd /workspace/gpu-inference && ./init.sh
#   claude                  # then: /setup (first time) or /deploy (returning)

MARKER="/tmp/.workspace-init-done"
if [ -f "$MARKER" ]; then
    return 0 2>/dev/null || exit 0
fi

PERSIST="/workspace/.persist"
CLAUDE_BIN="$PERSIST/bin"

# ── PATH ──
export PATH="$CLAUDE_BIN:$HOME/.local/bin:$PATH"

# ── Self-install into .bashrc and .zshrc ──
for rc in /root/.bashrc /root/.zshrc; do
    if [ -f "$rc" ] || [ "$(basename "$rc")" = ".bashrc" ]; then
        if ! grep -q 'workspace/gpu-inference/init.sh' "$rc" 2>/dev/null; then
            echo '[ -f /workspace/gpu-inference/init.sh ] && source /workspace/gpu-inference/init.sh' >> "$rc"
        fi
    fi
done

# ── System packages (lost on pod restart) ──
if ! command -v nvtop &>/dev/null; then
    (apt-get update -qq && apt-get install -y -qq nvtop libnuma1 2>/dev/null) &
fi

# ── Claude Code (install to persistent /workspace volume) ──
if ! command -v claude &>/dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash 2>/dev/null
    # Move binary to persistent storage so it survives GPU changes
    mkdir -p "$CLAUDE_BIN"
    if [ -f "$HOME/.local/bin/claude" ]; then
        cp "$HOME/.local/bin/claude" "$CLAUDE_BIN/claude"
        chmod +x "$CLAUDE_BIN/claude"
    fi
    export PATH="$CLAUDE_BIN:$HOME/.local/bin:$PATH"
fi

# ── Claude config (link persistent credentials) ──
if [ -d "$PERSIST/claude-config" ] && [ ! -L /root/.claude ]; then
    rm -rf /root/.claude
    ln -sf "$PERSIST/claude-config" /root/.claude
fi

touch "$MARKER"
echo "workspace init done"
