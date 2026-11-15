#!/usr/bin/env bash
# Proposer transport for vla-rsi-propose: prompt on stdin, model reply on stdout.
# Runs the opencode CLI once, in an empty directory so the agent has nothing to read or edit.
# Model: $RSI_PROPOSER_MODEL (provider/model as opencode names it).
set -euo pipefail
model="${RSI_PROPOSER_MODEL:?set RSI_PROPOSER_MODEL, e.g. opencode-go/deepseek-v4.1-flash}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat > "$work/prompt.txt"
cd "$work"
opencode run --archived -m "$model" "$(cat prompt.txt)" 2>/dev/null | sed -e 's/\x1b\[[0-9;]*m//g'
