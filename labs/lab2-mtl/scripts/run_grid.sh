#!/usr/bin/env bash
# Full grid: prepare, EDA, every condition x folds, aggregate, report.
# A failing condition does not stop the rest, because every run is resumable:
# a finished fold has metrics.json and is skipped on the next attempt.
set -u
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"
LOG="logs/grid_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs outputs report

exec > >(tee -a "$LOG") 2>&1
echo "config: $CONFIG"
echo "start:  $(date)"

step() {
  echo
  echo "=== $*"
  "$@" || echo "!!! FAILED: $*"
}

UV="uv run --system-certs"

step $UV ecgmtl prepare --config "$CONFIG" --set data.prefer_mirror=true
step $UV ecgmtl eda --config "$CONFIG"

for condition in single_diag single_rhythm single_age mtl_equal mtl_uncertainty mtl_pcgrad; do
  step $UV ecgmtl train --config "$CONFIG" --condition "$condition" --folds 5
done

for condition in loto_no_rhythm loto_no_age mtl_age_weight_0.1; do
  step $UV ecgmtl train --config "$CONFIG" --condition "$condition" --folds 2
done

step $UV ecgmtl summarize --config "$CONFIG"
step $UV ecgmtl report --config "$CONFIG"

echo
echo "done:   $(date)"
echo "log:    $LOG"
