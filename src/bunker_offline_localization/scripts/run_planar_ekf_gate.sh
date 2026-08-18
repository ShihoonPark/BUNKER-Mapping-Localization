#!/usr/bin/env bash
set -eo pipefail

workspace="/home/a/Desktop/shihoon/bunker_localization_ws"
output="${1:-${workspace}/results/planar_ekf_gicp_gate}"

mkdir -p "${output}/same_bag" "${output}/independent_163346_0_50s"

ros2 run bunker_offline_localization run_offline_smoke_test.sh "${output}/same_bag" 0
ros2 run bunker_offline_localization run_independent_163346.sh \
  "${output}/independent_163346_0_50s" 0
ros2 run bunker_offline_localization generate_planar_ekf_gate_report.py \
  --same-bag-csv "${output}/same_bag/localization.csv" \
  --independent-csv "${output}/independent_163346_0_50s/localization.csv" \
  --output-directory "${output}" \
  --independent-origin-timestamp 1786692827.2210245 \
  --ramp-start-sec 25.0 --ramp-end-sec 35.0 \
  --expected-same-bag-scans 869 --expected-independent-scans 490 \
  --baseline-same-bag-accepted 854 --baseline-independent-accepted 487 \
  --minimum-same-bag-accepted 854 --minimum-independent-accepted 485
