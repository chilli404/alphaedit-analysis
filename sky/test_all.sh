#!/usr/bin/env bash
# Launch all 4 GPU smoke test clusters in parallel.
# Total wall-clock: ~15 min (vs ~35 min sequential on one cluster).
#
# Usage:
#   bash sky/test_all.sh              # launch all 4
#   bash sky/test_all.sh --status     # check status of running tests
#   bash sky/test_all.sh --down       # tear down all test clusters

set -euo pipefail

CLUSTERS="test-vendor test-revive test-baseline test-eval"

case "${1:-launch}" in
    --status|status)
        for cl in $CLUSTERS; do
            echo "=== $cl ==="
            sky logs "$cl" --no-follow --tail 5 2>/dev/null || echo "  (not running)"
            echo ""
        done
        exit 0
        ;;
    --down|down)
        for cl in $CLUSTERS; do
            sky down "$cl" -y 2>/dev/null &
        done
        wait
        echo "All test clusters terminated."
        exit 0
        ;;
    launch|"")
        ;;
    *)
        echo "Usage: $0 [--status|--down]"
        exit 1
        ;;
esac

echo "Launching 4 GPU smoke test clusters in parallel..."
echo ""

sky launch sky/test_vendor_runners.yaml   --cluster test-vendor   --detach-run -y &
sky launch sky/test_revive_runners.yaml   --cluster test-revive   --detach-run -y &
sky launch sky/test_baseline_runners.yaml --cluster test-baseline --detach-run -y &
sky launch sky/test_eval_and_measure.yaml --cluster test-eval     --detach-run -y &
wait

echo ""
echo "All 4 test clusters launched."
echo "  Monitor: sky queue"
echo "  Logs:    sky logs test-vendor / test-revive / test-baseline / test-eval"
echo "  Status:  bash sky/test_all.sh --status"
echo "  Cleanup: bash sky/test_all.sh --down"
