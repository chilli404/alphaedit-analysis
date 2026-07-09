#!/usr/bin/env bash
set -euo pipefail

# One-time setup for the permanent remote cluster (RTX PRO Blackwell 6000).
# Run this on the remote machine after cloning the repo.
#
# Prerequisites:
#   - git, curl installed
#   - GPU drivers installed
#   - HF_TOKEN environment variable set
#
# Usage:
#   ssh remote "cd /path/to/alphaedit_replication && bash scripts/remote_setup.sh /path/to/stats"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STATS_PATH="${1:-}"

cd "$PROJECT_DIR"

echo "=== Remote Cluster Setup ==="
echo "  Host: $(hostname)"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "[1/6] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/6] uv already installed: $(uv --version)"
fi

# 2. Install Python 3.10 and sync
echo "[2/6] Installing Python 3.10 and syncing dependencies..."
uv python install 3.10
uv sync

# 3. Initialize submodule
echo "[3/6] Initializing AlphaEdit submodule..."
git submodule update --init --recursive

# 4. Verify commit
CURRENT=$(git -C vendor/AlphaEdit rev-parse HEAD)
EXPECTED="b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
if [[ "$CURRENT" != "$EXPECTED" ]]; then
    echo "ERROR: AlphaEdit at $CURRENT, expected $EXPECTED"
    exit 1
fi
echo "  AlphaEdit commit verified: ${CURRENT:0:7}"

# 5. Link stats
if [[ -n "$STATS_PATH" ]]; then
    echo "[5/6] Linking covariance stats from: $STATS_PATH"
    bash scripts/link_stats.sh "$STATS_PATH"
else
    echo "[5/6] SKIPPED: No stats path provided. Run later:"
    echo "    bash scripts/link_stats.sh /path/to/wikipedia_stats/"
fi

# 6. HuggingFace login
echo "[6/6] HuggingFace authentication..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    uv run huggingface-cli login --token "$HF_TOKEN"
    echo "  Logged in successfully."
else
    echo "  WARNING: HF_TOKEN not set. Set it and run:"
    echo "    uv run huggingface-cli login --token \$HF_TOKEN"
fi

# Download NLTK data
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# GPU check
echo ""
echo "=== GPU Information ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not found"
fi

echo ""
echo "=== Remote setup complete ==="
echo "Run smoke test: bash scripts/smoke_test.sh"
