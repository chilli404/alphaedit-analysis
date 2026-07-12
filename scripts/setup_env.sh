#!/usr/bin/env bash
set -euo pipefail

# One-time local environment setup for AlphaEdit replication.
# Run from the alphaedit_replication/ directory.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== AlphaEdit Replication Environment Setup ==="

# 1. Install Python 3.10 via uv
echo "[1/8] Installing Python 3.10..."
uv python install 3.10

# 2. Sync dependencies
echo "[2/8] Syncing dependencies..."
uv sync

# 3. Initialize git submodule
echo "[3/8] Initializing AlphaEdit submodule..."
git submodule update --init --recursive

# 4. Verify submodule commit
echo "[4/8] Verifying AlphaEdit commit..."
CURRENT=$(git -C vendor/AlphaEdit rev-parse HEAD)
EXPECTED="b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
if [[ "$CURRENT" != "$EXPECTED" ]]; then
    echo "ERROR: AlphaEdit submodule at $CURRENT, expected $EXPECTED"
    exit 1
fi
echo "  Commit verified: ${CURRENT:0:7}"

# 5. Download NLTK data
echo "[5/8] Downloading NLTK data..."
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 6. Link stats and datasets
echo "[6/8] Linking covariance stats..."
bash scripts/link_stats.sh

echo "[7/8] Linking datasets..."
bash scripts/link_dsets.sh

# 8. Patch vendor submodule for compatibility
echo "[8/8] Patching vendor submodule..."
# Add NousResearch model name variants to context length map
if ! grep -q "meta-llama-3-8b-instruct" vendor/AlphaEdit/glue_eval/useful_functions.py; then
    sed -i 's/"llama3-8b-instruct": 4096,/"llama3-8b-instruct": 4096, "meta-llama-3-8b-instruct": 4096, "nousresearch--meta-llama-3-8b-instruct": 4096,/' vendor/AlphaEdit/glue_eval/useful_functions.py
    echo "  Patched model name map"
fi
# Fix MEMIT and AlphaEdit: accept extra kwargs (return_orig_weights_device) passed by evaluate.py
if ! grep -q "_kwargs" vendor/AlphaEdit/memit/memit_main.py; then
    sed -i 's/    cache_template: Optional\[str\] = None,/    cache_template: Optional[str] = None, **_kwargs,/' vendor/AlphaEdit/memit/memit_main.py
    echo "  Patched MEMIT kwargs"
fi
if ! grep -q "_kwargs" vendor/AlphaEdit/AlphaEdit/AlphaEdit_main.py; then
    sed -i 's/    P = None,/    P = None, **_kwargs,/' vendor/AlphaEdit/AlphaEdit/AlphaEdit_main.py
    echo "  Patched AlphaEdit kwargs"
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Ensure HF_TOKEN is set (huggingface-cli login)"
echo "  2. Run: bash scripts/smoke_test.sh"
