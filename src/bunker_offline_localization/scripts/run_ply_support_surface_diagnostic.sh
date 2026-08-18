#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/ply_support_surface_geometry_diagnostic}"
config="${2:-${workspace}/src/bunker_offline_localization/config/ply_support_surface_geometry_diagnostic.yaml}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bunker_ply_support_surface_matplotlib}"
ros2 run bunker_offline_localization generate_ply_support_surface_diagnostic.py \
  --config "${config}" \
  --output-directory "${output}"
