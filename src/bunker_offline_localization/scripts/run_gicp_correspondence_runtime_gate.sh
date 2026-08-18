#!/usr/bin/env bash
set -eo pipefail

workspace=/home/a/Desktop/shihoon/bunker_localization_ws
output=${workspace}/results/gicp_correspondence_rviz_diagnostic
runtime_results=${output}/headless_run
source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"

python3 "${workspace}/src/bunker_offline_localization/scripts/generate_gicp_correspondence_rviz_diagnostic.py" --output "${output}" --snapshot-only
ros2 launch bunker_offline_localization gicp_rviz_diagnostic.launch.py mode:=continuous launch_rviz:=false publish_visualization:=false publish_correspondences_every_n_scans:=0 playback_rate:=1.0 results_directory:="${runtime_results}" output_directory:="${output}"
python3 "${workspace}/src/bunker_offline_localization/scripts/generate_gicp_correspondence_rviz_diagnostic.py" --output "${output}" --runtime-results "${runtime_results}" --before "${output}/protected_inputs_before.json"
