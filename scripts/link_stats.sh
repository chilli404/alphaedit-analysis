#!/usr/bin/env bash
set -euo pipefail

# Copies precomputed covariance statistics into vendor/AlphaEdit/data/stats/
# and baselines/EvoEdit/data/stats/ under ONE canonical name per model,
# then symlinks all variant names to that single copy.
#
# Before: 8 variants × 5 files × 2 codebases = 80 file copies (slow on S3 FUSE)
# After:  1 canonical × 5 files × 2 codebases = 10 copies + 14 symlinks (fast)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_DIR="/s3-data"

_CALLER_MODEL_NAME="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
MODEL_NAME="${_CALLER_MODEL_NAME:-${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}}"

# Map model to its canonical stats subdirectory and _name_or_path value
# These must match evaluate_harness.py::_canonical_name_or_path()
stats_subdir_for_model() {
    case "$1" in
        *Meta-Llama-3-8B*|*llama*)  echo "llama3-8b-instruct" ;;
        *gpt-j-6[bB]*)              echo "gpt-j-6b" ;;
        *Qwen2.5-7B*)               echo "qwen2.5-7b-instruct" ;;
        *)                          echo "$(echo "${1##*/}" | tr '[:upper:]' '[:lower:]')" ;;
    esac
}

canonical_name_or_path() {
    case "$1" in
        *Meta-Llama-3-8B*|*llama*)  echo "Meta-Llama-3-8B-Instruct" ;;
        *gpt-j*)                     echo "EleutherAI_gpt-j-6B" ;;
        *Qwen*)                      echo "Qwen2.5-7B-Instruct" ;;
        *)                          echo "${1##*/}" ;;
    esac
}

STATS_SUBDIR="$(stats_subdir_for_model "$MODEL_NAME")"
CANONICAL="$(canonical_name_or_path "$MODEL_NAME")"

S3_STATS_SRC="$S3_DIR/continual-learning/alphaedit/stats/$STATS_SUBDIR"
PROJECT_STATS_SRC="$PROJECT_DIR/data/stats/$STATS_SUBDIR/wikipedia_stats"

STATS_SRC="$PROJECT_STATS_SRC"
[[ -d "$S3_STATS_SRC" ]] && STATS_SRC="$S3_STATS_SRC"

if [[ ! -d "$STATS_SRC" ]]; then
    echo "ERROR: Stats source directory not found."
    echo "  S3:      $S3_STATS_SRC"
    echo "  Project: $PROJECT_STATS_SRC"
    exit 1
fi

echo "=== Linking Covariance Statistics ==="
echo "  Model:  $MODEL_NAME"
echo "  Stats:  $STATS_SUBDIR"
echo "  Source: $STATS_SRC"

# All variant names that code might resolve _name_or_path to
case "$MODEL_NAME" in
    *Meta-Llama-3-8B*)
        VARIANTS="Meta-Llama-3-8B Meta-Llama-3-8B-Instruct Llama3-8B NousResearch_Meta-Llama-3-8B-Instruct NousResearch-Meta-Llama-3-8B-Instruct _s3-data_continual-learning_models_Meta-Llama-3-8B meta-llama_Meta-Llama-3-8B-Instruct llama3-8b-instruct"
        ;;
    *gpt-j*)
        VARIANTS="gpt-j-6b EleutherAI_gpt-j-6B EleutherAI_gpt-j-6b EleutherAI-gpt-j-6b _s3-data_continual-learning_models_gpt-j-6b"
        ;;
    *Qwen*)
        VARIANTS="Qwen2.5-7B Qwen2.5-7B-Instruct Qwen_Qwen2.5-7B-Instruct qwen2.5-7b-instruct"
        ;;
    *)
        VARIANTS="${MODEL_NAME##*/}"
        ;;
esac

_link_stats_to() {
    local dst_base="$1"
    local label="$2"
    local canonical_dst="$dst_base/$CANONICAL/wikipedia_stats"

    # Step 1: Copy files to the ONE canonical directory
    rm -rf "$canonical_dst" 2>/dev/null
    mkdir -p "$canonical_dst"
    local n_files=0
    for f in "$STATS_SRC"/*.npz; do
        [ -f "$f" ] || continue
        cp "$f" "$canonical_dst/$(basename "$f")" 2>/dev/null || \
            cat "$f" > "$canonical_dst/$(basename "$f")" 2>/dev/null || true
        n_files=$((n_files + 1))
    done

    # Step 2: Symlink all variant names to the canonical directory
    local n_links=0
    for variant in $VARIANTS; do
        [ -z "$variant" ] && continue
        [[ "$variant" == "$CANONICAL" ]] && continue
        local link_dst="$dst_base/$variant/wikipedia_stats"
        rm -rf "$link_dst" 2>/dev/null
        mkdir -p "$(dirname "$link_dst")"
        ln -s "$canonical_dst" "$link_dst" 2>/dev/null || {
            # Fallback: if symlinks fail (some FUSE mounts), copy instead
            mkdir -p "$link_dst"
            cp "$canonical_dst"/*.npz "$link_dst/" 2>/dev/null || true
        }
        n_links=$((n_links + 1))
    done
    echo "  $label: $n_files stats files + $n_links symlinked variants"
}

# vendor/AlphaEdit/data/stats/
_link_stats_to "$PROJECT_DIR/vendor/AlphaEdit/data/stats" "vendor/AlphaEdit"

# baselines/EvoEdit/data/stats/ (only if baselines exist)
if [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]]; then
    _link_stats_to "$PROJECT_DIR/baselines/EvoEdit/data/stats" "baselines/EvoEdit"
fi

# Copy null-space projection P matrix
P_CACHE_SRC="$STATS_SRC/null_space_project.pt"
if [[ -f "$P_CACHE_SRC" ]]; then
    cp "$P_CACHE_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt" 2>/dev/null || true
    cp "$P_CACHE_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project_${STATS_SUBDIR}.pt" 2>/dev/null || true
    [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]] && {
        cp "$P_CACHE_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project.pt" 2>/dev/null || true
        cp "$P_CACHE_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project_${STATS_SUBDIR}.pt" 2>/dev/null || true
    }
    echo "  Linked cached null-space projection: $(basename "$P_CACHE_SRC")"
fi

# Copy per-threshold P caches
P_COUNT=0
for p_file in "$STATS_SRC"/null_space_project_t*.pt; do
    [[ -f "$p_file" ]] || continue
    cp "$p_file" "$PROJECT_DIR/vendor/AlphaEdit/$(basename "$p_file")" 2>/dev/null || true
    P_COUNT=$((P_COUNT + 1))
done
[[ $P_COUNT -gt 0 ]] && echo "  Copied $P_COUNT per-threshold P cache(s)"

# Mirror vendor stats to baselines/EvoEdit (symlink the entire stats dir)
if [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]]; then
    echo "  EvoEdit stats mirrored from vendor/AlphaEdit/data/stats → baselines/EvoEdit/data/stats"
fi

echo "=== Done ==="
echo "  Stats symlinks: $CANONICAL → $(echo $VARIANTS | wc -w | tr -d ' ') variants"
