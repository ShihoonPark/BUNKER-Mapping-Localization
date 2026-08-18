# bunker_offline_localization

ROS 2 Humble Phase 1 offline localization smoke test. It aligns each consecutive Velodyne scan
against an existing GLIM PLY map with the official `small_gicp` implementation. The package does
not include Nav2, vehicle control, map regeneration, GLIM changes, or finished global
relocalization.

## Transform convention

Every transform follows one rule:

```text
p_A = T_A_B * p_B
```

The global PLY is the registration target and the current LiDAR scan is the source. Therefore:

```text
small_gicp result.T_target_source = T_map_lidar
p_map = T_map_lidar * p_lidar
```

The result is never inverted. ROS axes are `+x` forward, `+y` left, `+z` up, with positive yaw
counter-clockwise. Unit tests enforce transform direction, inverse consistency, quaternion
normalization, forward/left signs, and positive CCW yaw.

## Inputs and observed bag properties

- Map: `/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply`
- Bag: `/home/a/Desktop/shihoon/Slam/slam_20260814_150626`
- GLIM reference: `/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/dump_150626_direct_20260814_222719/traj_lidar.txt`
- `/velodyne_points`: `frame_id=velodyne`, 878 scans, 57,984 points per organized cloud
- `/odom`: `odom -> base_link`; pose/twist change, but all covariance entries are zero
- `/imu/data`: `frame_id=imu_link`; all covariance entries are zero
- IMU orientation is exactly `(0,0,0,1)` in all 12,555 messages and is not fused
- IMU angular velocity changes and `angular_velocity.z` is fused
- Known `T_lidar_imu`: translation `[0,0,-0.07]`, quaternion `[0,0,0,1]`
- Actual `T_base_lidar` is unknown and is not fabricated as physical ground truth

The first reference timestamp is `1786687590.879746914`. Scans earlier than its association
window are skipped. The first associated scan initializes `T_map_lidar` from the first valid GLIM
pose. Nearest timestamp association is accepted only within the configured tolerance.

The independent Gate uses:

- Bag: `/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346`
- Bag metadata origin: `1786692827.221024517`
- Full bag: 129.757 s, 1,075 LiDAR scans, 4,732 odometry messages, 12,692 IMU messages
- Gate interval: inclusive bag-relative `[0.0, 50.0]` s
- Gate messages by header stamp: 490 LiDAR scans, 2,417 odometry, 6,919 IMU
- Frames and covariance behavior match the good bag: `velodyne`, `odom -> base_link`, and
  `imu_link`; source odom/IMU covariance arrays are zero and every IMU orientation is identity

Large LiDAR/IMU timing gaps appear later in this bag (first observed after about 62.4 s). They are
outside the Gate and are neither fused nor registered.

## Dependencies and vendoring

- ROS 2 Humble and `robot_localization/ekf_node` from `/opt/ros/humble`
- Ubuntu/ROS-compatible PCL 1.12 for binary PLY loading
- Eigen3 and OpenMP
- Official `small_gicp` Git submodule at `third_party/small_gicp`
- Upstream: `https://github.com/koide3/small_gicp.git`
- Pinned upstream commit: `aea131352e0d362d3e579a334c477cfafa5ee5eb` (`small_gicp` 1.0.1)

A submodule was chosen instead of copying files so the exact official upstream revision remains
auditable and the vendored implementation stays unchanged. This package consumes its header API
through an interface CMake target. It does not reimplement GICP and does not use PCL GICP.

The machine also has separate PCL 1.15 files under `/usr/lib`; CMake intentionally resolves the
Ubuntu PCL 1.12 package at `/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake` to remain ABI
compatible with ROS Humble.

## Pipeline

```text
/odom ---- covariance adapter ----\
                                  robot_localization ekf_node ---- pose prediction
/imu/data - covariance adapter ---/                                  |
                                                                      v
/velodyne_points ---- finite filter/source preprocess ---- small_gicp(map target)
                                                                      |
                                                                      v
                                            quality gate -> CSV/TUM/report/plots
```

The adapter republishes separate topics; it never overwrites the bag topics. Its covariance
values are configurable initial tuning assumptions, not measured covariance. The EKF fuses odom
`x/y/yaw`, forward velocity and yaw rate plus IMU yaw rate. `two_d_mode` is `false`: the EKF is a
robust initial-guess motion model, while small_gicp applies a full 6DoF map correction needed for
the ramp and stage height changes.

The target map is loaded, voxelized, assigned covariances, and indexed with a KD-tree exactly once.
Every source scan is independently filtered, voxelized, assigned covariances, and registered.

## Independent-drive initialization and time window

`time_window.enabled`, `time_window.origin_timestamp`, `time_window.start_offset_sec`, and
`time_window.end_offset_sec` are shared by the covariance adapter and localizer. Bounds are
inclusive and offsets are measured from the rosbag metadata origin, not from an individual
topic's first message. They are disabled by default in `localization.yaml`, preserving the
original 150626 behavior. `independent_163346.yaml` enables `[0, 50]` s, and the launch file also
exposes `window_start_sec` and `window_end_sec` arguments.

There is no GLIM reference trajectory for 163346. The independent configuration therefore uses
`initialization.mode: parameter` and the first valid 150626 `T_map_lidar` only as a small_gicp
initial guess, under the explicit Phase 1 assumption that both recordings begin at the mapped
staging pose. It is not evaluation ground truth, absolute RMSE is not computed, and this is not a
finished global relocalization method.

## Base-to-LiDAR limitation

The production code default refuses to start without either an actual `T_base_lidar` or explicit
permission to use the Phase 1 identity approximation. `config/localization.yaml` and the Phase 1
launch opt into that approximation visibly:

```yaml
base_to_lidar:
  available: false
  allow_identity_for_phase1_smoke_test: true
```

With an actual calibration, configure `available: true`, translation `[x,y,z]`, and
`rotation_xyzw: [qx,qy,qz,qw]`, then remove the test-only static identity publisher. Prediction is
converted using:

```text
T_map_lidar(k) = T_map_lidar(anchor)
                 * inverse(T_base_lidar)
                 * inverse(T_odom_base(anchor)) * T_odom_base(k)
                 * T_base_lidar
```

Until that extrinsic is measured, this phase validates `T_map_lidar`; final base_link localization
and `map -> odom` TF validation remain incomplete.

## Initial registration settings

All values below are calibration starting assumptions and are parameters in
`config/localization.yaml`:

| Parameter | Default |
|---|---:|
| `registration_type` | `GICP` |
| `num_threads` | 4 |
| `map_voxel_resolution` | 0.20 m |
| `scan_voxel_resolution` | 0.20 m |
| `num_neighbors` | 20 |
| `max_correspondence_distance` | 1.0 m |
| `max_iterations` | 30 |
| `min_inliers` | 100 |
| `max_final_error_per_inlier` | 5.0 |
| `max_translation_correction` | 1.0 m |
| `max_rotation_correction` | 0.523599 rad |
| `reference_timestamp_tolerance` | 0.06 s |
| `prediction_timestamp_tolerance` | 0.10 s |

Registrations are rejected with an explicit reason rather than replaced by the previous pose:
`NOT_CONVERGED`, `LOW_INLIERS`, `HIGH_ERROR`, `TRANSLATION_JUMP`, `ROTATION_JUMP`,
`NONFINITE_TRANSFORM`, `NO_PREDICTION`, `TIMESTAMP_MISMATCH`, `EMPTY_SCAN`, or
`REGISTRATION_EXCEPTION`.

## Build and test

Run these commands in one terminal:

```bash
cd /home/a/Desktop/shihoon/bunker_localization_ws
git submodule update --init --recursive
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR=/home/a/Desktop/shihoon/bunker_localization_ws/log/ros
colcon build --symlink-install --packages-select bunker_offline_localization --cmake-args -DCMAKE_BUILD_TYPE=Release
source /home/a/Desktop/shihoon/bunker_localization_ws/install/setup.bash
colcon test --packages-select bunker_offline_localization --event-handlers console_direct+
colcon test-result --verbose
```

The 23 tests cover transform direction/signs/inversion, quaternion normalization, TUM parsing,
timestamp tolerance, invalid registration rejection, inclusive/disabled/invalid time windows,
and a known-transform synthetic point cloud registered with official small_gicp.

## Run

First real scan only:

```bash
ros2 launch bunker_offline_localization offline_localization.launch.py max_scans:=1 results_directory:=/tmp/bunker_localization_first_scan
```

Full good-bag smoke test followed by report generation:

```bash
ros2 run bunker_offline_localization run_offline_smoke_test.sh /home/a/Desktop/shihoon/bunker_localization_ws/results 0
```

Regenerate only the report:

```bash
ros2 run bunker_offline_localization generate_report.py --localization-csv /home/a/Desktop/shihoon/bunker_localization_ws/results/localization.csv --estimated-trajectory /home/a/Desktop/shihoon/bunker_localization_ws/results/estimated_traj_lidar.tum --reference-trajectory /home/a/Desktop/shihoon/glim_real/20260814_classroom/results/dump_150626_direct_20260814_222719/traj_lidar.txt --map-ply /home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply --output-directory /home/a/Desktop/shihoon/bunker_localization_ws/results --timestamp-tolerance 0.06
```

The wrapper sets writable ROS and matplotlib cache paths. It replays only `/odom`, `/imu/data`,
and `/velodyne_points`, publishes `/clock`, and shuts the launch down after playback is flushed.

Independent first-scan integration check (writes only to `/tmp`):

```bash
ros2 launch bunker_offline_localization independent_localization.launch.py max_scans:=1 results_directory:=/tmp/bunker_independent_first_scan
```

Full independent 0-50 s Gate and report:

```bash
ros2 run bunker_offline_localization run_independent_163346.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s 0
```

The launch defaults are fixed to the independent bag, origin, and `[0, 50]` interval. To run a
different reviewed interval, pass all three metadata-relative arguments explicitly:

```bash
ros2 launch bunker_offline_localization independent_localization.launch.py window_origin_timestamp:=1786692827.2210245 window_start_sec:=0.0 window_end_sec:=50.0 results_directory:=/home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s
```

Regenerate the independent report without replaying the bag:

```bash
ros2 run bunker_offline_localization generate_independent_report.py --localization-csv /home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s/localization.csv --estimated-trajectory /home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s/estimated_traj_lidar.tum --map-ply /home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply --output-directory /home/a/Desktop/shihoon/bunker_localization_ws/results/independent_163346_0_50s --window-origin-timestamp 1786692827.2210245 --window-start-sec 0.0 --window-end-sec 50.0 --expected-scans 490
```

## Results

Generated under `results/`:

- `localization.csv`: prediction, raw GICP pose, status/reason, iterations, inliers, raw final
  error, runtime, point counts, association deltas, and all 36 Hessian entries
- `estimated_traj_lidar.tum`: accepted `T_map_lidar` poses only
- `reference_comparison.csv`: timestamp-associated position/yaw/rotation errors
- `run_summary.json`: ingestion and acceptance counters
- `summary.json`: aggregate error/runtime statistics
- `diagnosis.md`: interpretation and known limitations
- `trajectory_xy.png`, `z_over_time.png`, `yaw_over_time.png`
- `translation_error.png`, `yaw_error.png`, `gicp_final_error.png`
- `num_inliers.png`, `runtime_ms.png`, `acceptance_timeline.png`
- `map_trajectory_overlay.png`

Validated full-run result on this machine:

- 878 bag scans received; 9 pre-reference scans skipped; 869 processed
- 854 accepted, 15 rejected (`NOT_CONVERGED`), acceptance rate 98.274%
- 852 accepted poses associated to reference within 0.06 s
- Translation error RMSE/mean/max: 0.1241 / 0.0945 / 0.4306 m
- Component RMSE x/y/z: 0.0162 / 0.0177 / 0.1218 m
- Absolute yaw error RMSE/max: 0.9331 / 4.7423 deg
- Registration runtime mean/p95/max: 2.242 / 3.187 / 7.403 ms

These values are explicitly a **pipeline/reference consistency smoke test**, not independent
localization accuracy, because the map and reference trajectory came from the same bag.

## Independent 163346 Gate results

Generated separately under `results/independent_163346_0_50s/`:

- `localization.csv` and `estimated_traj_lidar.tum`: independent scan records/accepted trajectory
- `accepted_locations.csv` and `rejected_locations.csv`: event time, bag-relative time, position,
  yaw, status, inliers, error, and runtime
- `pose_jumps.csv`, `summary.json`, and `independent_report.md`: jump/smoothness and aggregate data
- `map_trajectory_overlay.png`, `acceptance_timeline.png`, `pose_jumps.png`,
  `trajectory_smoothness.png`, `num_inliers.png`, `gicp_final_error.png`, and `runtime_ms.png`

Validated 0-50 s result on this machine:

- Exactly 490 scans processed; first/last offsets 1.000909 / 49.992760 s
- 487 accepted, 3 rejected (`NOT_CONVERGED`), 99.388% convergence/acceptance
- Rejections at 5.909805, 16.830560, and 28.752881 s; their positions are retained separately
- Inliers mean/p95/min: 2667.6 / 2937.8 / 2017
- Raw final error mean/p95/max: 335.919 / 474.043 / 943.746; per-inlier p95 0.17995
- Runtime mean/p95/max: 2.226 / 3.044 / 8.026 ms
- Consecutive accepted translation step p95/max: 0.1021 / 0.2125 m
- Consecutive accepted rotation step p95/max: 2.509 / 8.636 deg
- The map overlay keeps the trajectory inside the mapped classroom path: it travels along the
  lower edge from the origin, turns at the right side, and continues upward; rejected locations
  remain on that same path rather than appearing as spatial outliers

The largest motion spikes occur near the main turn around 27-29 s. They remain below the current
registration correction gates, but the speed/acceleration plots are plausibility diagnostics, not
vehicle-dynamics ground truth. Since 163346 has no independent reference trajectory, these results
support map alignment continuity and robustness but make no absolute accuracy or RMSE claim.
