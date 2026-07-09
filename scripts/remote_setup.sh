#!/usr/bin/env bash
set -euo pipefail

# One-time setup for the permanent remote cluster.
# Run this on the remote machine after cloning the repo.
#
# Prerequisites:
#   - git, curl installed
#   - GPU drivers installed
#   - HF_TOKEN environment variable set if model access requires auth
#   - ARTIFACTORY_USERNAME and ARTIFACTORY_TOKEN set if using Grainger Artifactory
#
# Usage:
#   ssh remote "cd /path/to/alphaedit_replication && bash scripts/remote_setup.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Remote Cluster Setup ==="
echo "  Host: $(hostname)"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "[1/8] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[1/8] uv already installed: $(uv --version)"
fi

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface}"

# 2. Configure Artifactory auth if available
echo "[2/8] Configuring Python package index..."
if [[ -n "${ARTIFACTORY_USERNAME:-}" && -n "${ARTIFACTORY_TOKEN:-}" ]]; then
    echo "  Using Grainger Artifactory for uv."

    cat > "$HOME/.netrc" <<NETRC_EOF
machine graingerinc.jfrog.io
login $ARTIFACTORY_USERNAME
password $ARTIFACTORY_TOKEN
NETRC_EOF
    chmod 600 "$HOME/.netrc"

    mkdir -p "$HOME/.config/uv"
    cat > "$HOME/.config/uv/uv.toml" <<'UV_EOF'
[[index]]
name = "grainger-artifactory"
url = "https://graingerinc.jfrog.io/artifactory/api/pypi/pypi-shared-virtual/simple"
default = true
UV_EOF
else
    echo "  Skipping Artifactory auth; using standard PyPI."
fi

# 3. Install Python 3.10 and sync
echo "[3/8] Installing Python 3.10 and syncing dependencies..."
uv python install 3.10
uv sync

# 4. Initialize submodule
echo "[4/8] Initializing AlphaEdit submodule..."
git submodule update --init --recursive

# 5. Verify commit
echo "[5/8] Verifying AlphaEdit commit..."
CURRENT="$(git -C vendor/AlphaEdit rev-parse HEAD)"
EXPECTED="b84624f44dfe8fc6cd9e41df916c44124a0c46dc"

if [[ "$CURRENT" != "$EXPECTED" ]]; then
    echo "ERROR: AlphaEdit at $CURRENT, expected $EXPECTED"
    exit 1
fi

echo "  AlphaEdit commit verified: ${CURRENT:0:7}"

# 6. Link stats
echo "[6/8] Linking covariance stats..."
bash scripts/link_stats.sh

# 7. HuggingFace login
echo "[7/8] HuggingFace authentication..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    uv run huggingface-cli login --token "$HF_TOKEN"
    echo "  Logged in successfully."
else
    echo "  WARNING: HF_TOKEN not set. Set it and run:"
    echo "    uv run huggingface-cli login --token \$HF_TOKEN"
fi

# 8. Download NLTK data
echo "[8/8] Downloading NLTK data..."
uv run python - <<'PY'
import nltk

for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)
PY

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
