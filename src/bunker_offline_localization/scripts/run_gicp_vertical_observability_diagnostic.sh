#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/gicp_vertical_observability_diagnostic}"
config="${2:-${workspace}/src/bunker_offline_localization/config/gicp_vertical_observability_diagnostic.yaml}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bunker_gicp_vertical_observability_matplotlib}"
ros2 run bunker_offline_localization generate_gicp_vertical_observability_diagnostic.py \
  --config "${config}" \
  --output-directory "${output}"
