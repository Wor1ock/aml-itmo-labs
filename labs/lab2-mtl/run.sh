#!/usr/bin/env bash
# One command for the grader: build the image (data + preprocessing at build time)
# and run the full pipeline into ./outputs and ./report.
set -euo pipefail
cd "$(dirname "$0")"
docker compose build
docker compose run --rm app
echo "report: $(pwd)/report/index.html"
