#!/bin/bash
# One-time setup of the Python environment. Same command on every cluster:
#
#   scripts/bootstrap.sh            # full environment (GPU inference + training)
#   scripts/bootstrap.sh --cpu      # CPU-only (data preparation and scoring; no torch/vllm)
#
# Installs a pinned `uv` into ~/.local/bin (your shell profile is not modified), then creates
# .venv from uv.lock: the exact same package versions and the same standalone CPython
# everywhere. Nothing is installed system-wide and conda is not used.
set -euo pipefail
UV_VERSION="0.13.8"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 || [ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]; then
    echo "[bootstrap] installing uv $UV_VERSION into ~/.local/bin"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi

extras=(--extra gpu --extra dev)
[ "${1:-}" = "--cpu" ] && extras=(--extra dev)
echo "[bootstrap] creating .venv from uv.lock (${extras[*]})"
uv sync --frozen "${extras[@]}"

# shellcheck disable=SC1091
source scripts/env.sh
echo "[bootstrap] cluster profile: ${T2C_CLUSTER:-none (using defaults; see configs/clusters/TEMPLATE.env.example)}"
t2c doctor
