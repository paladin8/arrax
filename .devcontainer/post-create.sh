#!/bin/bash
# Runs once on container creation (before the firewall starts, so GitHub and
# package registries are reachable without restriction). Bootstraps Claude
# Code and the project's dependencies.
set -euo pipefail

# --- Claude Code plugins ----------------------------------------------------
# Register the official marketplace (idempotent) and install superpowers.
# Both commands are idempotent so this is safe to re-run on rebuild when the
# claude-code-config volume already has state.
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install superpowers@claude-plugins-official

# Merge default model + effort into settings.json without clobbering the
# plugin entries that `claude plugin install` just wrote.
SETTINGS="$HOME/.claude/settings.json"
jq '. + {model: "opus", effortLevel: "high"}' "$SETTINGS" > "$SETTINGS.tmp"
mv "$SETTINGS.tmp" "$SETTINGS"

# --- Project dependencies ---------------------------------------------------
# Install whatever the workspace's language ecosystem declares. Each branch is
# a no-op if the relevant manifest isn't present, so this stays useful when
# the devcontainer is reused across projects.
cd /workspace

if [ -f pyproject.toml ]; then
    echo ">>> pyproject.toml detected — running uv sync"
    uv sync
fi

if [ -f package-lock.json ]; then
    echo ">>> package-lock.json detected — running npm ci"
    npm ci
elif [ -f package.json ]; then
    echo ">>> package.json detected — running npm install"
    npm install
fi

if [ -f Cargo.toml ]; then
    echo ">>> Cargo.toml detected — running cargo fetch"
    cargo fetch
fi

if [ -f go.mod ]; then
    echo ">>> go.mod detected — running go mod download"
    go mod download
fi

echo "post-create complete"
