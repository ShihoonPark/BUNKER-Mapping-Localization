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

The tests cover transform direction/signs/inversion, quaternion normalization, TUM parsing,
timestamp tolerance, invalid registration rejection, and a known-transform synthetic point cloud
registered with official small_gicp.

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
