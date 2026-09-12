#!/usr/bin/env bash
set -euo pipefail

# Link covariance statistics into vendor and baselines data directories.
#
# S3 structure (canonical):
#   /s3-data/.../stats/llama3-8b-instruct/wikipedia_stats/*.npz
#   /s3-data/.../stats/gpt-j-6b/wikipedia_stats/*.npz
#   /s3-data/.../stats/qwen2.5-7b-instruct/wikipedia_stats/*.npz
#
# Vendor expects:
#   data/stats/{model_name}/wikipedia_stats/*.npz
#   where model_name = model.config._name_or_path.replace("/", "_")
#
# With canonical name normalization (source_patches.py::apply_canonical_name_patch),
# _name_or_path is always set to the stats directory name (e.g. "llama3-8b-instruct"),
# so vendor looks for: data/stats/llama3-8b-instruct/wikipedia_stats/*.npz
# which maps directly to S3. Just symlink the stats root.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_STATS="/s3-data/continual-learning/alphaedit/stats"
LOCAL_STATS="$PROJECT_DIR/data/stats"

# Determine stats source
if [[ -d "$S3_STATS" ]]; then
    STATS_SRC="$S3_STATS"
elif [[ -d "$LOCAL_STATS" ]]; then
    STATS_SRC="$LOCAL_STATS"
else
    echo "ERROR: No stats source found at $S3_STATS or $LOCAL_STATS"
    exit 1
fi

echo "=== Linking Covariance Statistics ==="
echo "  Source: $STATS_SRC"

# Vendor: symlink data/stats → stats source
VENDOR_STATS="$PROJECT_DIR/vendor/AlphaEdit/data/stats"
rm -rf "$VENDOR_STATS" 2>/dev/null
mkdir -p "$(dirname "$VENDOR_STATS")"
ln -sfn "$STATS_SRC" "$VENDOR_STATS"
echo "  vendor/AlphaEdit/data/stats → $STATS_SRC"

# Baselines: symlink data/stats → stats source
if [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]]; then
    BASELINES_STATS="$PROJECT_DIR/baselines/EvoEdit/data/stats"
    rm -rf "$BASELINES_STATS" 2>/dev/null
    mkdir -p "$(dirname "$BASELINES_STATS")"
    ln -sfn "$STATS_SRC" "$BASELINES_STATS"
    echo "  baselines/EvoEdit/data/stats → $STATS_SRC"
fi

# P matrix: symlink into vendor and baselines CWD
_CALLER_MODEL_NAME="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
MODEL_NAME="${_CALLER_MODEL_NAME:-${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}}"

# Detect stats subdir
_mn_lower="$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
if [[ "$_mn_lower" == *llama* ]]; then STATS_SUBDIR="llama3-8b-instruct"
elif [[ "$_mn_lower" == *gpt-j* ]]; then STATS_SUBDIR="gpt-j-6b"
elif [[ "$_mn_lower" == *qwen* ]]; then STATS_SUBDIR="qwen2.5-7b-instruct"
else STATS_SUBDIR="${MODEL_NAME##*/}"; fi

P_SRC="$STATS_SRC/$STATS_SUBDIR/null_space_project.pt"
if [[ -f "$P_SRC" ]]; then
    # Vendor CWD path (vendor evaluate.py loads from CWD)
    ln -sf "$P_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt"
    ln -sf "$P_SRC" "$PROJECT_DIR/vendor/AlphaEdit/null_space_project_${STATS_SUBDIR}.pt"
    # Inside wikipedia_stats/ (harness runners check data/stats/{model}/wikipedia_stats/)
    WIKISTATS="$PROJECT_DIR/vendor/AlphaEdit/data/stats/$STATS_SUBDIR/wikipedia_stats"
    [[ -d "$WIKISTATS" ]] && ln -sf "$P_SRC" "$WIKISTATS/null_space_project.pt"
    [[ -d "$PROJECT_DIR/baselines/EvoEdit" ]] && {
        ln -sf "$P_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project.pt"
        ln -sf "$P_SRC" "$PROJECT_DIR/baselines/EvoEdit/null_space_project_${STATS_SUBDIR}.pt"
        BL_WIKISTATS="$PROJECT_DIR/baselines/EvoEdit/data/stats/$STATS_SUBDIR/wikipedia_stats"
        [[ -d "$BL_WIKISTATS" ]] && ln -sf "$P_SRC" "$BL_WIKISTATS/null_space_project.pt"
    }
    echo "  Linked null-space projection: $STATS_SUBDIR"
fi

# Per-threshold P caches
P_COUNT=0
for p_file in "$STATS_SRC/$STATS_SUBDIR"/null_space_project_t*.pt; do
    [[ -f "$p_file" ]] || continue
    ln -sf "$p_file" "$PROJECT_DIR/vendor/AlphaEdit/$(basename "$p_file")"
    P_COUNT=$((P_COUNT + 1))
done
[[ $P_COUNT -gt 0 ]] && echo "  Linked $P_COUNT per-threshold P cache(s)"

echo "=== Done ==="
