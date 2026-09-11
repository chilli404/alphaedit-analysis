#!/usr/bin/env bash
set -euo pipefail

# Zero-copy covariance stats linking for SkyPilot clusters.
#
# Creates a symlink tree that maps vendor's expected model-name directories
# to the actual S3 FUSE stats path. Then rewrites globals.yml to point at it.
#
# Before: 10 file copies × 784MB each = ~8GB of S3→local copies (minutes)
# After:  symlinks only (instant)
#
# On local machines: symlinks to data/stats/ (also instant)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_STATS="/s3-data/continual-learning/alphaedit/stats"

_CALLER_MODEL_NAME="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
MODEL_NAME="${_CALLER_MODEL_NAME:-${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}}"

echo "=== Linking Covariance Statistics ==="
echo "  Model:  $MODEL_NAME"

# Map: model name → S3 subdirectory name → canonical _name_or_path value
# These MUST match evaluate_harness.py::_canonical_name_or_path()
declare -A STATS_SUBDIRS CANONICALS
STATS_SUBDIRS=(
    [llama]="llama3-8b-instruct"
    [gptj]="gpt-j-6b"
    [qwen]="qwen2.5-7b-instruct"
)
CANONICALS=(
    [llama]="llama3-8b-instruct"
    [gptj]="gpt-j-6b"
    [qwen]="qwen2.5-7b-instruct"
)

# Detect which model family
_mn_lower="$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
if [[ "$_mn_lower" == *llama* ]]; then MODEL_KEY="llama"
elif [[ "$_mn_lower" == *gpt-j* ]]; then MODEL_KEY="gptj"
elif [[ "$_mn_lower" == *qwen* ]]; then MODEL_KEY="qwen"
else
    echo "WARNING: Unknown model '$MODEL_NAME'. Falling back to copy mode."
    MODEL_KEY=""
fi

STATS_SUBDIR="${STATS_SUBDIRS[$MODEL_KEY]:-}"
CANONICAL="${CANONICALS[$MODEL_KEY]:-${MODEL_NAME##*/}}"

# Find stats source
if [[ -d "$S3_STATS/$STATS_SUBDIR" ]]; then
    STATS_SRC="$S3_STATS/$STATS_SUBDIR"
elif [[ -d "$PROJECT_DIR/data/stats/$STATS_SUBDIR/wikipedia_stats" ]]; then
    STATS_SRC="$PROJECT_DIR/data/stats/$STATS_SUBDIR/wikipedia_stats"
elif [[ -d "$PROJECT_DIR/data/stats/$STATS_SUBDIR" ]]; then
    STATS_SRC="$PROJECT_DIR/data/stats/$STATS_SUBDIR"
else
    echo "ERROR: Stats not found for $STATS_SUBDIR"
    exit 1
fi

echo "  Source: $STATS_SRC"

# All variant names code might use for _name_or_path
_all_variants() {
    echo "$CANONICAL"
    case "$MODEL_KEY" in
        llama)
            echo "Meta-Llama-3-8B Meta-Llama-3-8B-Instruct Llama3-8B"
            echo "NousResearch_Meta-Llama-3-8B-Instruct NousResearch-Meta-Llama-3-8B-Instruct"
            echo "_s3-data_continual-learning_models_Meta-Llama-3-8B meta-llama_Meta-Llama-3-8B-Instruct"
            echo "llama3-8b-instruct"
            ;;
        gptj)
            echo "gpt-j-6b EleutherAI_gpt-j-6B EleutherAI_gpt-j-6b EleutherAI-gpt-j-6b"
            echo "_s3-data_continual-learning_models_gpt-j-6b"
            ;;
        qwen)
            echo "Qwen2.5-7B Qwen2.5-7B-Instruct Qwen_Qwen2.5-7B-Instruct qwen2.5-7b-instruct"
            ;;
    esac
}

_link_tree() {
    local stats_dir="$1"
    local label="$2"
    local count=0

    mkdir -p "$stats_dir"

    for variant in $(_all_variants); do
        local dst="$stats_dir/$variant/wikipedia_stats"
        rm -rf "$dst" 2>/dev/null
        mkdir -p "$(dirname "$dst")"
        ln -sfn "$STATS_SRC" "$dst"
        count=$((count + 1))
    done
    echo "  $label: $count symlinked variants → $STATS_SRC"
}

# Create symlink trees for vendor and baselines
_link_tree "$PROJECT_DIR/vendor/AlphaEdit/data/stats" "vendor/AlphaEdit"
[[ -d "$PROJECT_DIR/baselines/EvoEdit" ]] && \
    _link_tree "$PROJECT_DIR/baselines/EvoEdit/data/stats" "baselines/EvoEdit"

# P matrix — symlink instead of copy
P_SRC="$STATS_SRC/null_space_project.pt"
if [[ -f "$P_SRC" ]]; then
    ln -sf "$P_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt"
    ln -sf "$P_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project_${STATS_SUBDIR}.pt"
    [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]] && {
        ln -sf "$P_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project.pt"
        ln -sf "$P_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project_${STATS_SUBDIR}.pt"
    }
    echo "  Linked cached null-space projection: $(basename "$P_SRC")"
fi

# Per-threshold P caches
P_COUNT=0
for p_file in "$STATS_SRC"/null_space_project_t*.pt; do
    [[ -f "$p_file" ]] || continue
    ln -sf "$p_file" "$PROJECT_DIR/vendor/AlphaEdit/$(basename "$p_file")"
    P_COUNT=$((P_COUNT + 1))
done
[[ $P_COUNT -gt 0 ]] && echo "  Linked $P_COUNT per-threshold P cache(s)"

echo "=== Done ==="
