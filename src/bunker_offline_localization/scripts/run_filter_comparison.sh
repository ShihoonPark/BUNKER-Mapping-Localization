#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/filter_comparison}"

mkdir -p "${workspace}/log/ros" "${output}" /tmp/bunker_localization_matplotlib
export ROS_LOG_DIR="${workspace}/log/ros"
export MPLCONFIGDIR="/tmp/bunker_localization_matplotlib"

source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"
set -u

for dataset in same_bag independent; do
  for filter_type in ekf ukf; do
    run_directory="${output}/${dataset}/${filter_type}"
    mkdir -p "${run_directory}"
    ros2 launch bunker_offline_localization filter_comparison.launch.py \
      dataset:="${dataset}" filter_type:="${filter_type}" \
      results_directory:="${run_directory}" max_scans:=0
  done
done

ros2 run bunker_offline_localization generate_filter_comparison.py \
  --same-bag-ekf-csv "${output}/same_bag/ekf/localization.csv" \
  --same-bag-ukf-csv "${output}/same_bag/ukf/localization.csv" \
  --independent-ekf-csv "${output}/independent/ekf/localization.csv" \
  --independent-ukf-csv "${output}/independent/ukf/localization.csv" \
  --reference-trajectory "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/dump_150626_direct_20260814_222719/traj_lidar.txt" \
  --output-directory "${output}" \
  --independent-origin-timestamp 1786692827.2210245 \
  --timestamp-tolerance 0.06 \
  --expected-independent-scans 490 \
  --config "${workspace}/src/bunker_offline_localization/config/localization.yaml" \
  --config "${workspace}/src/bunker_offline_localization/config/independent_163346.yaml" \
  --config "${workspace}/src/bunker_offline_localization/config/filter_common.yaml" \
  --config "${workspace}/src/bunker_offline_localization/config/filter_ekf.yaml" \
  --config "${workspace}/src/bunker_offline_localization/config/filter_ukf.yaml"
