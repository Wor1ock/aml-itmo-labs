#!/usr/bin/env bash
# Wrap the grid so a broken tmux session is restartable: rerun the same command and
# finished folds are skipped. Usage: bash scripts/night.sh [config]
set -u
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"

if tmux has-session -t grid 2>/dev/null; then
  echo "session 'grid' already exists; attach with: tmux attach -t grid"
  exit 1
fi

tmux new-session -d -s grid "cd $(pwd) && UV_HTTP_TIMEOUT=600 bash scripts/run_grid.sh $CONFIG"
echo "started tmux session 'grid' with $CONFIG"
echo "attach: tmux attach -t grid"
echo "watch:  tail -f logs/grid_*.log"
