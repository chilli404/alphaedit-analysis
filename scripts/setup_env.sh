#!/usr/bin/env bash
set -euo pipefail

# One-time local environment setup for AlphaEdit replication.
# Run from the alphaedit_replication/ directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== AlphaEdit Replication Environment Setup ==="

# 1. Install Python 3.10 via uv
echo "[1/5] Installing Python 3.10..."
uv python install 3.10

# 2. Sync dependencies
echo "[2/5] Syncing dependencies..."
uv sync

# 3. Initialize git submodule
echo "[3/5] Initializing AlphaEdit submodule..."
git submodule update --init --recursive

# 4. Verify submodule commit
echo "[4/5] Verifying AlphaEdit commit..."
CURRENT=$(git -C vendor/AlphaEdit rev-parse HEAD)
EXPECTED="b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
if [[ "$CURRENT" != "$EXPECTED" ]]; then
    echo "ERROR: AlphaEdit submodule at $CURRENT, expected $EXPECTED"
    exit 1
fi
echo "  Commit verified: ${CURRENT:0:7}"

# 5. Download NLTK data
echo "[5/5] Downloading NLTK data..."
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run: bash scripts/link_stats.sh"
echo "  2. Ensure HF_TOKEN is set (huggingface-cli login)"
echo "  3. Run: bash scripts/smoke_test.sh"
