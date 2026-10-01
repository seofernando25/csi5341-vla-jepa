#!/usr/bin/env bash
# Fetch the published project on macOS/Linux; never launches GPU work.
set -euo pipefail
repo_url='https://github.com/seofernando25/csi5341-vla-jepa.git'
target="${1:-$HOME/Documents/Projects/csi5341-vla-jepa}"
if [[ -d "$target/.git" ]]; then
  if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo 'Existing checkout has local changes; preserve them before updating.' >&2
    exit 1
  fi
  if [[ "$(git -C "$target" remote get-url origin)" != "$repo_url" ]]; then
    echo 'Existing origin differs from the project URL; refusing to update.' >&2
    exit 1
  fi
  if [[ "$(git -C "$target" branch --show-current)" != main ]]; then
    echo "Switch the existing checkout to main before updating." >&2
    exit 1
  fi
  git -C "$target" pull --ff-only origin main
elif [[ -e "$target" ]]; then
  echo 'Destination already exists and is not a Git checkout.' >&2
  exit 1
else
  mkdir -p "$(dirname "$target")"
  git clone "$repo_url" "$target"
fi
printf '\nProject: %s\n' "$target"
printf 'Revision: '; git -C "$target" rev-parse --short HEAD
printf 'Read HANDOFF.md for the current evidence, host-only artifacts and next steps.\n'
printf 'Mac: review code/results; CUDA training and LIBERO remain on a Linux GPU host.\n'
