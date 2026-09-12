#!/usr/bin/env bash
set -euo pipefail
# Smart SkyPilot launcher — wraps sky/run.yaml and sky/test.yaml.
#
# Usage:
#   bash sky/launch.sh METHOD SEED [ORDERING]         # single experiment
#   bash sky/launch.sh METHOD SEED1,SEED2 [ORDERING]  # multi-seed (parallel)
#   bash sky/launch.sh --test                          # all tests
#   bash sky/launch.sh --test "EvoEdit NSE"            # test specific algorithms
#   bash sky/launch.sh --list                          # list available methods
#   bash sky/launch.sh --status                        # sky status
#   bash sky/launch.sh --down                          # tear down all clusters
#
# Examples:
#   bash sky/launch.sh alphaedit 42
#   bash sky/launch.sh revive+memit 42,2024,137 fb_high_exposure
#   bash sky/launch.sh nse 42 fb_random0
#   bash sky/launch.sh --test "PathGuard REVIVE"

case "${1:-}" in
    --test)
        FILTER="${2:-}"
        CLUSTER="test-$(date +%H%M)"
        echo "Launching test cluster: $CLUSTER"
        if [[ -n "$FILTER" ]]; then
            sky launch sky/test.yaml --cluster "$CLUSTER" --env "SMOKE_FILTER=$FILTER" --detach-run -y
        else
            sky launch sky/test.yaml --cluster "$CLUSTER" --detach-run -y
        fi
        echo "Monitor: sky logs $CLUSTER"
        exit 0
        ;;
    --list)
        echo "Available methods:"
        uv run python -m src list-methods 2>/dev/null || echo "  Run 'uv run python -m src list-methods' locally"
        exit 0
        ;;
    --status)
        sky status
        exit 0
        ;;
    --down)
        echo "Tearing down all clusters..."
        sky down -a -y
        exit 0
        ;;
    --help|-h|"")
        echo "Usage: bash sky/launch.sh METHOD SEED [ORDERING]"
        echo "       bash sky/launch.sh --test [FILTER]"
        echo "       bash sky/launch.sh --list | --status | --down"
        exit 0
        ;;
esac

METHOD="$1"
SEEDS="${2:?Usage: bash sky/launch.sh METHOD SEED [ORDERING]}"
ORDERING="${3:-}"

# Launch one cluster per seed
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
for SEED in "${SEED_ARRAY[@]}"; do
    CLUSTER="${METHOD//+/-}-s${SEED}"
    [[ -n "$ORDERING" ]] && CLUSTER="${CLUSTER}-${ORDERING:0:6}"

    ENV_ARGS="METHOD=$METHOD,SEED=$SEED"
    [[ -n "$ORDERING" ]] && ENV_ARGS="$ENV_ARGS,ORDERING=$ORDERING"

    echo "Launching: $CLUSTER ($METHOD seed=$SEED${ORDERING:+ ordering=$ORDERING})"
    sky launch sky/run.yaml --cluster "$CLUSTER" --env "$ENV_ARGS" --detach-run -y
done

echo ""
echo "Monitor: sky queue"
echo "Logs:    sky logs <cluster>"
