#!/bin/sh
# Run the full pipeline. Inside Docker this file is optional:
# the image entrypoint is already `python -m fusion_slu.cli`.
set -eu
CONFIG="${CONFIG:-configs/default.yaml}"
exec python -m fusion_slu.cli all --config "$CONFIG" "$@"
