#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/imu_gicp_surface_pose_z_diagnostic}"
config="${2:-${workspace}/src/bunker_offline_localization/config/imu_gicp_surface_pose_z_diagnostic.yaml}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bunker_surface_pose_z_matplotlib}"
ros2 run bunker_offline_localization generate_surface_pose_z_diagnostic.py \
  --config "${config}" \
  --output-directory "${output}"
