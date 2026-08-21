#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
results_directory="${1:-${workspace}/results/independent_163346_0_50s}"
max_scans="${2:-0}"

mkdir -p "${workspace}/log/ros" "${results_directory}" /tmp/bunker_localization_matplotlib
export ROS_LOG_DIR="${workspace}/log/ros"
export MPLCONFIGDIR="/tmp/bunker_localization_matplotlib"

source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"
set -u

ros2 launch bunker_offline_localization independent_localization.launch.py \
  results_directory:="${results_directory}" max_scans:="${max_scans}" \
  window_start_sec:=0.0 window_end_sec:=50.0

if [[ "${max_scans}" == "0" ]]; then
  ros2 run bunker_offline_localization generate_independent_report.py \
    --localization-csv "${results_directory}/localization.csv" \
    --estimated-trajectory "${results_directory}/estimated_traj_lidar.tum" \
    --map-ply "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply" \
    --output-directory "${results_directory}" \
    --window-origin-timestamp 1786692827.2210245 \
    --window-start-sec 0.0 \
    --window-end-sec 50.0 \
    --expected-scans 490
fi
