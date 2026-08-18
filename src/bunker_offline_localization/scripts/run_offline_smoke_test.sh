#!/usr/bin/env bash
set -euo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
results_directory="${1:-${workspace}/results}"
max_scans="${2:-0}"

mkdir -p "${workspace}/log/ros" "${results_directory}" /tmp/bunker_localization_matplotlib
export ROS_LOG_DIR="${workspace}/log/ros"
export MPLCONFIGDIR="/tmp/bunker_localization_matplotlib"

source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"

ros2 launch bunker_offline_localization offline_localization.launch.py \
  results_directory:="${results_directory}" max_scans:="${max_scans}"

ros2 run bunker_offline_localization generate_report.py \
  --localization-csv "${results_directory}/localization.csv" \
  --estimated-trajectory "${results_directory}/estimated_traj_lidar.tum" \
  --reference-trajectory "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/dump_150626_direct_20260814_222719/traj_lidar.txt" \
  --map-ply "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply" \
  --output-directory "${results_directory}" \
  --timestamp-tolerance 0.06
