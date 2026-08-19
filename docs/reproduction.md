# Reproduction guide

All commands assume ROS 2 Humble and the external datasets/maps at the recorded absolute paths. They do not install dependencies or modify source data.

Conceptually, `DATA_ROOT`, `MAP_ROOT`, and `RESULTS_ROOT` refer to the external bag archive, GLIM result archive, and ignored local result tree. The checked-in launch defaults record this machine's corresponding paths for exact laboratory reproduction.

## Build and test

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select bunker_offline_localization --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test --packages-select bunker_offline_localization --event-handlers console_direct+
colcon test-result --verbose
```

## Bag D coarse global initialization

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization bag_d_coarse_initializer.launch.py
```

This writes generated evidence under ignored `results/bag_D_flat_20260819_coarse_init/`. Do not overwrite a result needed for comparison; choose a separate output path when rerunning experiments.

## Bag D 0–50 s localization

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization bag_d_localization.launch.py mode:=short
```

The launch uses the metadata-relative short window and saved coarse seed defined by the Bag D configuration/launch code. See the package README for exact report-generation commands.

## Bag D full localization

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization bag_d_localization.launch.py mode:=full
```

The canonical result is a 1.0× full metadata-interval replay. Do not infer absolute RMSE from this run.

## Regenerate the Bag D full report

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws && PYTHONPATH=src/bunker_offline_localization/scripts:$PYTHONPATH python3 src/bunker_offline_localization/scripts/generate_full_bag_report.py --localization-csv results/bag_D_flat_20260819_localization_full/localization.csv --latency-csv results/bag_D_flat_20260819_localization_full/continuous_latency.csv --estimated-trajectory results/bag_D_flat_20260819_localization_full/estimated_traj_lidar.tum --full-bag-summary results/bag_D_flat_20260819_localization_full/full_bag_summary.json --short-localization-csv results/bag_D_prediction_discontinuity_diagnostic/post_fix_localization_0_50s/localization.csv --short-latency-csv results/bag_D_prediction_discontinuity_diagnostic/post_fix_localization_0_50s/continuous_latency.csv --map-ply /home/a/Desktop/shihoon/glim_real/20260819_flat/results/flat_bag_C_imu_on.ply --output-directory results/bag_D_flat_20260819_localization_full --origin-timestamp 1787142248.3215761 --metadata-duration-sec 219.251211092 --expected-scans 2181
```

This command overwrites generated files in its output directory; use a new output directory when preserving an existing comparison. The checked-in curated copies are evidence snapshots, not report inputs.

## Bag D RViz visual replay

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch bunker_offline_localization bag_d_rviz_localization.launch.py playback_rate:=1.0
```

Expected display: fixed Bag C map, registered Bag D scan moving over the map, accepted GICP path/pose, and prediction pose. Raw scan and correspondences can be enabled for diagnosis. RViz remains open with the final state after EOF; stop with Ctrl+C.

`playback_rate:=2.0` or `3.0` is visualization convenience only and must not replace the 1.0× production gate.

## Inputs that must remain read-only

```text
/home/a/Desktop/shihoon/Slam/20260819_dataset/bag_D_flat_independent_localization
/home/a/Desktop/shihoon/glim_real/20260819_flat/results/flat_bag_C_imu_on.ply
/home/a/Desktop/shihoon/bunker_localization_ws/results/bag_D_flat_20260819_coarse_init/coarse_initialization.json
```

The detailed package README is the authoritative command log for earlier classroom, filter A/B, physical-consistency, vertical-observability, and latency gates.
