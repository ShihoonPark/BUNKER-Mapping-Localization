#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/imu_gicp_physical_gate}"
config="${2:-${workspace}/src/bunker_offline_localization/config/imu_gicp_physical_gate.yaml}"

if [[ "${output}" == "${workspace}/results/planar_ekf_gicp_gate"* ]]; then
  echo "Refusing to overwrite the production planar-EKF/GICP result tree: ${output}" >&2
  exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/bunker_localization_matplotlib}"
ros2 run bunker_offline_localization generate_imu_gicp_physical_gate.py \
  --config "${config}" \
  --output-directory "${output}"
