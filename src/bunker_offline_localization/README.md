# bunker_offline_localization

ROS 2 Humble Phase 1 offline localization smoke test. It aligns each consecutive Velodyne scan
against an existing GLIM PLY map with the official `small_gicp` implementation. The package does
not include Nav2, vehicle control, map regeneration, GLIM changes, or finished global
relocalization.

The production objective is not standalone filter 6DoF estimation. The official
`robot_localization/ekf_node` supplies a stable planar-motion initial guess, and small_gicp
produces the final full-6DoF `T_map_lidar` pose. The prior EKF/UKF A/B Gate remains archived under
`results/filter_comparison/`; its production decision is EKF.

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
- Python 3 with NumPy, PyYAML, Matplotlib, Open3D, `rosbag2_py`, and ROS message Python bindings
  for the read-only physical-consistency and plateau-label diagnostic reports. Open3D reads only
  the PLY XY points used by the diagnostic overlay; it is not used for registration.
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
                                  robot_localization ekf_node ---- SE(2) dx/dy/dyaw
/imu/data - covariance adapter ---/                                  |
                                                                      v
last accepted GICP full 6DoF ---- hold z/roll/pitch, update x/y/yaw ---- initial guess
                                                                      |
/velodyne_points ---- finite filter/source preprocess ---- small_gicp(map target)
                                                                      |
                                                                      v
                                  full-6DoF T_map_lidar -> quality gate -> CSV/TUM/report/plots
```

The adapter republishes separate topics; it never overwrites the bag topics. Its covariance
values are configurable initial tuning assumptions, not measured covariance. The EKF fuses odom
`x/y/yaw`, forward velocity and yaw rate plus IMU yaw rate. Its two poses are projected to SE(2)
before relative `dx/dy/dyaw` is extracted, so EKF z/roll/pitch cannot leak into the prediction.
The last accepted GICP pose is the anchor: prediction updates x/y/yaw and holds its z/roll/pitch
exactly. Small_gicp then applies the full 6DoF map correction needed for the ramp and stage height
changes. `two_d_mode` remains `false` to preserve the validated filter configuration; unobserved
filter axes are simply not consumed by the localizer.

The target map is loaded, voxelized, assigned covariances, and indexed with a KD-tree exactly once.
Every source scan is independently filtered, voxelized, assigned covariances, and registered.

## IMU–GICP physical consistency Gate

This is a post-processing validation Gate, not another localization mode. It reads the preserved
production CSV and the independent bag's `/imu/data` and `/odom` topics, and writes only to
`results/imu_gicp_physical_gate_stage_150mm/`. The earlier no-height result under
`results/imu_gicp_physical_gate/` is a protected input and cannot be selected as the new output.
Input size/mtime fingerprints plus the production CSV SHA-256 are compared before and after every
real-data run.

For adjacent CSV rows whose two endpoints are accepted and whose interval is at most 0.15 s, the
body/LiDAR-frame registration angular velocity is:

```text
delta_R = transpose(R_map_lidar(k)) * R_map_lidar(k+1)
omega_gicp_lidar = Log(delta_R) / dt
```

The SO(3) logarithm has explicit small-angle and near-pi branches. A rejected row breaks the
pair sequence; neither GICP pairing nor lag interpolation crosses that break. IMU gyro is selected
only in bag-relative 0-50 s. Its robust median bias uses the reviewed 1-8 s candidate interval plus
odom linear speed, odom angular speed, gyro norm, and odom-association thresholds. Each GICP
interval receives a timestamp-weighted gyro average. The known identity LiDAR-IMU rotation keeps
the x/y/z comparison unchanged; no automatic axis/sign or clock correction is applied.

The ramp orientation comparison anchors the first accepted 25-35 s GICP orientation and integrates
all three bias-corrected gyro axes with SO(3) exponential updates. This is a relative common anchor,
not an absolute IMU orientation observation. Identity IMU orientations and accelerometer double
integration are deliberately unused.

Stable-plateau detection searches 20-40 s with rolling-median z/pitch rates and emits candidates
before reading the measured height. A second, height-blind temporal rule partitions candidates
before 25 s as ground-before, the unique stable candidate inside the predeclared 25-35 s ramp
window as stage-top, and candidates after 35 s as ground-after. A missing group, a boundary
overlap, or multiple in-ramp candidates is treated as ambiguous and no height is forced. Manually
reviewed windows can override this rule without using the physical height for selection.

The two ground groups fit robust
`z = a*x + b*y + c`; stage height is the median plane residual with robust spread and bootstrap
confidence interval. The measured carpet-to-flat-stage vertical height is 0.150 m. Its 0.050 m
tolerance is explicitly an initial physical-consistency screening threshold chosen with the
0.20 m map/scan voxel resolution and manual measurement in mind, not a calibrated localization
accuracy threshold. Raw absolute and relative errors are reported independently of the threshold.

### Diagnostic-only plateau-label review

The preserved 20-40 s height Gate and its `HEIGHT_FAIL` are not reclassified or overwritten.
`plateau_label_diagnostic` separately searches the complete independent valid interval 0-50 s
with the same z-rate, pitch-rate, duration, consecutive-acceptance, and timestamp-gap rules. It
does not receive the measured height during detection and assigns every candidate the physical
label `UNASSIGNED`; it does not automatically select ground-before, stage-top, or ground-after.

The measured 0.150 m is introduced only after all candidates are frozen. Every candidate-pair raw
median-z difference and measured-height error is emitted in natural candidate-ID order, never
ranked by height error and never used for selection. The report retains the historical fact that
the earlier temporal partition called candidate 4 stage-top and candidates 5-7 ground-after, plus
their raw median-z ordering. The 8.206 deg fitted ground-plane slope is recorded as a warning that
those surface labels are not independently established. Therefore the diagnostic conclusion is
`height validation currently inconclusive because physical plateau labels are not independently
established`; this is not a production-localization failure decision.

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
`rotation_xyzw: [qx,qy,qz,qw]`, then remove the test-only static identity publisher. The public
conversion interface uses:

```text
T_lidar_base = inverse(T_base_lidar)
T_map_base = T_map_lidar * T_lidar_base
```

The direction is unit-tested with a non-identity extrinsic. Until that extrinsic is measured, the
pipeline writes only `T_map_lidar`; it does not fabricate `T_map_base`, and final base_link
localization and `map -> odom` TF validation remain incomplete.

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

All prior test targets behind the earlier 36-result colcon summary remain. The physical-Gate pytest
target now contains eighteen cases covering stable SO(3) logarithms near zero, 90 degrees, and pi; body-frame
relative direction; constant-rate gyro integration; robust median bias; timestamp-weighted interval
averaging; rejected/gap exclusion; axis/sign correlation; synthetic lag recovery; robust
ground-plane fitting; known synthetic stage height; unavailable physical-height PENDING behavior;
height-blind temporal partitioning, ambiguous-candidate refusal, raw screening errors,
before/after inconsistency warnings, and protected input/output separation. The direct target
reports 18/18 passed. The separate plateau-label diagnostic target adds five cases covering the
full 0-50 s height-blind window, rejection/gap boundaries, unassigned labels, non-ranked post-hoc
pair differences, preserved historical labels/raw ordering/HEIGHT_FAIL, and the inconclusive
conclusion. The physical-surface pose-z target adds seven cases for exact parallel-plane recovery,
nonparallel planes, fixed label locking, measured-height leakage prevention, pair-cherry-pick
prevention, finite outputs, and protected-input integrity. The full ROS result summary reports
42 tests, 0 errors, 0 failures, and 0 skipped (the
colcon total includes its CTest aggregate records). Together the suite covers transform
direction/signs/inversion,
quaternion normalization, TUM parsing,
timestamp tolerance, invalid registration rejection, inclusive/disabled/invalid time windows,
a known-transform synthetic point cloud registered with official small_gicp, both comparison
launch/config paths, shared sensor fields, disabled IMU orientation, prediction-to-GICP delta
direction, comparison statistics, SE(2)-only EKF relative motion, accepted-pose z/roll/pitch hold,
negative/positive yaw branches, `T_map_base` direction, and Gate report invariants.

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

EKF/UKF first-scan comparison integration checks:

```bash
ros2 launch bunker_offline_localization filter_comparison.launch.py dataset:=same_bag filter_type:=ekf max_scans:=1 results_directory:=/tmp/bunker_comparison_ekf_first
ros2 launch bunker_offline_localization filter_comparison.launch.py dataset:=same_bag filter_type:=ukf max_scans:=1 results_directory:=/tmp/bunker_comparison_ukf_first
```

Full four-run EKF/UKF comparison and combined report:

```bash
ros2 run bunker_offline_localization run_filter_comparison.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/filter_comparison
```

Production planar-EKF/full-6DoF-GICP regression Gate. This writes to a separate directory and
does not replace the archived A/B results:

```bash
ros2 run bunker_offline_localization run_planar_ekf_gate.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/planar_ekf_gicp_gate
```

Run the read-only IMU–GICP physical-consistency Gate against that preserved production result:

```bash
ros2 run bunker_offline_localization run_imu_gicp_physical_gate.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/imu_gicp_physical_gate_stage_150mm /home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/config/imu_gicp_physical_gate.yaml
```

Regenerate it directly without changing or rerunning localization:

```bash
ros2 run bunker_offline_localization generate_imu_gicp_physical_gate.py --config /home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/config/imu_gicp_physical_gate.yaml --output-directory /home/a/Desktop/shihoon/bunker_localization_ws/results/imu_gicp_physical_gate_stage_150mm
```

Run the separate diagnostic-only 0-50 s plateau-label review. It reads the existing production
CSV, independent bag sensors, PLY, and preserved height result; it does not replay localization:

```bash
ros2 run bunker_offline_localization run_plateau_label_diagnostic.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/imu_gicp_plateau_label_diagnostic /home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/config/imu_gicp_physical_gate.yaml
```

The comparison launch accepts `filter_type:=ekf|ukf` and
`dataset:=same_bag|independent`. `filter_common.yaml` contains every shared state/sensor field;
`filter_ukf.yaml` only makes the official defaults `alpha=0.001`, `kappa=0`, and `beta=2`
explicit. No filter-specific covariance, GICP, voxel, seed, or quality-gate tuning is applied.

`filter_runtime_ms` is not claimed as internal robot_localization CPU time, which the official
nodes do not publish. It is the identically instrumented steady-clock latency from reception of
the nearest adapted odom/IMU input to reception of filtered odometry and therefore includes ROS
transport and scheduling.

## Results

Generated under `results/`:

- `localization.csv`: prediction, raw GICP pose, status/reason, iterations, inliers, raw final
  error, runtime, point counts, association deltas, explicit correction translation/roll/pitch/yaw,
  and all 36 Hessian entries
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

## EKF vs UKF A/B Gate results

Generated under `results/filter_comparison/`:

- `same_bag/{ekf,ukf}/` and `independent/{ekf,ukf}/`: raw per-run CSV/TUM/metadata
- `ekf_summary.json`, `ukf_summary.json`, and `comparison_manifest.json`: complete aggregate,
  fairness hashes, reject recovery, and 27-29 s metrics
- `filter_comparison.csv`, `prediction_comparison.csv`, and `correction_comparison.csv`
- `comparison_report.md`
- `predicted_xy_trajectory.png`, `predicted_yaw.png`, `prediction_translation_error.png`,
  `prediction_yaw_error.png`, `gicp_correction_translation.png`,
  `gicp_correction_rotation.png`, `gicp_iterations.png`, `gicp_runtime.png`,
  `filter_runtime.png`, and `accepted_rejected_timeline.png`

Same-bag pre-GICP prediction results, 867 timestamp associations per filter:

| Metric | EKF | UKF |
|---|---:|---:|
| Translation RMSE | 0.124166 m | 8.555469 m |
| x/y/z RMSE | 0.019356 / 0.019404 / 0.121104 m | 1.077693 / 0.893364 / 8.440174 m |
| Yaw RMSE | 0.820115 deg | 8.421087 deg |
| Full rotation RMSE | 2.368001 deg | 24.952337 deg |
| Prediction translation step p95/max | 0.074405 / 0.320899 m | 1.059552 / 8.344850 m |
| Prediction rotation step p95/max | 4.056504 / 11.232718 deg | 4.371867 / 14.596177 deg |

The EKF prediction error stayed below 0.423 m. The untuned UKF first exceeded 1 m at 55.711 s
and reached 23.559 m, dominated by z drift. The downstream same-bag localization accepted
854/869 for EKF versus 555/869 for UKF. UKF rejections were 268 `LOW_INLIERS`, 31
`TRANSLATION_JUMP`, and 15 `NOT_CONVERGED`; 252 late rejects had no later recovery. This is
consistent with an unobservable/untuned 3D UKF state becoming unstable under the current sensor
selection, but that causal interpretation requires a separate UKF-specific experiment.

Independent 0-50 s results:

| Metric | EKF | UKF |
|---|---:|---:|
| Accepted/rejected | 487 / 3 | 486 / 4 |
| Convergence rate | 99.388% | 99.184% |
| Inliers mean/p95/min | 2667.6 / 2937.8 / 2017 | 2667.6 / 2937.8 / 2017 |
| Final error mean/p95/max | 335.952 / 474.103 / 943.263 | 335.477 / 474.009 / 942.409 |
| Iterations mean/p95 | 3.914 / 8.550 | 4.020 / 9.000 |
| GICP runtime mean/p95/max | 2.277 / 3.064 / 6.246 ms | 2.239 / 3.161 / 8.154 ms |
| Filter latency mean/p95/max | 5.055 / 9.225 / 17.148 ms | 5.177 / 9.610 / 17.646 ms |

Every independent reject recovered on the next scan. UKF added one reject at 16.329612 s; both
filters rejected 5.909805, 16.830560, and 28.752881 s.

Independent accepted-scan prediction-to-GICP corrections, mean/median/p95/max:

| Metric | EKF | UKF |
|---|---:|---:|
| Translation [m] | 0.027672 / 0.016115 / 0.101960 / 0.210034 | 0.027979 / 0.016246 / 0.101856 / 0.248527 |
| Full rotation [deg] | 0.616826 / 0.394448 / 2.078785 / 8.645969 | 0.637166 / 0.401625 / 2.116769 / 8.702454 |
| Absolute yaw [deg] | 0.137936 / 0.084695 / 0.426800 / 2.356378 | 0.169090 / 0.111835 / 0.504126 / 2.262974 |

In the 27-29 s large-turn interval both accepted 19/20. EKF versus UKF correction translation
mean/p95 was 0.082964/0.166933 versus 0.085302/0.167766 m; rotation mean/p95 was
3.184006/7.500154 versus 3.216800/7.514198 deg; iteration mean/p95 was 8.50/13.80 versus
8.75/14.75.

With differences below 1% treated as practical ties, EKF wins five of six primary criteria and
UKF wins none. The current BUNKER initial-guess choice should remain EKF. This conclusion is only
for the present untuned, identical-condition Gate and does not claim that EKF is generally
superior to UKF. Any UKF-specific process-noise/state-observability tuning belongs in a separate
experiment.

## Accepted-GICP-anchored planar EKF Gate results

Generated separately under `results/planar_ekf_gicp_gate/`; existing same-bag, independent, and
EKF/UKF A/B result directories remain unchanged.

| Dataset | Scans | Accepted/rejected | Prior accepted | Accepted 6DoF finite |
|---|---:|---:|---:|---:|
| 150626 full | 869 | 856 / 13 | 854 | 856 / 856 |
| 163346 0-50 s | 490 | 486 / 4 | 487 | 486 / 486 |

Independent accepted prediction-to-GICP correction:

| Metric | Mean | p95 | Max |
|---|---:|---:|---:|
| Translation | 0.028994 m | 0.106938 m | 0.280748 m |
| Absolute roll | 0.325622 deg | 1.176557 deg | 3.898094 deg |
| Absolute pitch | 0.416589 deg | 1.532423 deg | 8.514659 deg |
| Absolute yaw | 0.127365 deg | 0.371356 deg | 2.482382 deg |

The maximum independent prediction hold error was zero for z, `2.819e-17` rad for roll, and
`1.665e-16` rad for pitch, confirming that EKF unobserved axes do not accumulate. All accepted
prediction/GICP poses were finite and their maximum quaternion norm error was `2.220e-16`.

In the observed 25-35 s ramp interval, accepted GICP z covered 0.373813 m with a 0.205983 m
maximum step. Pitch covered 26.498698 deg with an 8.508549 deg maximum step. The separate
`independent_ramp_z.png` and `independent_ramp_pitch.png` plots show the intentional held
prediction and the full-6DoF GICP corrections. The combined report is
`results/planar_ekf_gicp_gate/planar_ekf_gate_report.md`.

The Gate allows at most two additional independent rejects (0.408 percentage point) relative to
the prior 487/490 result; the measured result lost one scan. Same-bag acceptance improved by two.
No small_gicp parameter, voxel size, quality gate, covariance, seed, or retry policy was changed.

## IMU–GICP physical consistency Gate results

Generated separately under `results/imu_gicp_physical_gate_stage_150mm/`; the production CSV,
earlier no-height Gate, prior reports, map, bag, and GLIM dump remain unchanged.

- Input: all 490 localization rows, 6,919 IMU messages, and 2,417 odometry messages within 0-50 s
- IMU orientation: 6,919/6,919 identity messages; none used
- Robust stationary gyro median bias x/y/z: `0 / 0 / 0` rad/s from 990 selected samples; the
  corresponding means are `-6.347e-5 / +1.922e-4 / +6.170e-5` rad/s
- GICP pairs: 481 valid adjacent accepted pairs; 8 pairs excluded by rejected endpoints, no large
  gap pair, and 480 pairs retained after IMU interval coverage
- Full zero-lag Pearson x/y/z: `0.1947 / 0.7627 / 0.9712`
- Full best lag x/y/z: `-0.04 / -0.06 / -0.04` s, with Pearson
  `0.2306 / 0.8865 / 0.9895`; these offsets are diagnostic only
- Ramp zero-lag Pearson x/y/z: `0.2523 / 0.8248 / 0.9742`; pitch sign agreement is `0.7527`
- All three expected x/y/z mappings are their strongest absolute-correlation mapping and all have
  the expected sign; weak roll correlation and low-motion sign agreement remain explicit warnings
- Ramp relative-orientation error mean/p95/max: `1.6508 / 4.3329 / 6.0415` deg; final error
  `0.2883` deg
- Seven stable candidates were detected without the measured height. The height-blind temporal
  partition selected candidate IDs 1-3 as ground-before (29 samples), ID 4 as stage-top (7
  samples), and IDs 5-7 as ground-after (24 samples). Exact windows and individual sample counts
  are retained in `candidate_plateaus.csv` and `height_validation.json`
- Measured stage height: `0.150000 m`
- Estimated stage height: `-0.051391 m`; robust spread `0.008931 m`; bootstrap 95% CI
  `[-0.071607, -0.026933] m`
- Raw absolute/relative error: `0.201391 m / 134.261%`
- Ground-before/after referenced estimates: `-0.049596 / -0.051474 m`; difference `0.001878 m`,
  therefore consistent under the initial 0.050 m before/after limit
- Robust ground plane: `a=0.093054`, `b=0.110163`, `c=-0.563177 m`; slope `0.144205 m/m`
  (`8.206 deg`), which raises a separate map-tilt/localization-z warning
- Raw 25-35 s z max-min is `0.373789 m` and is explicitly not used as stage height
- Height screening tolerance: `0.050 m`, initial screening only and not calibrated accuracy
- Decision: `GYRO_WARN`, `HEIGHT_FAIL`, `OVERALL_FAIL`

`GYRO_WARN` is not a parameter-tuning failure: pitch and yaw are strongly consistent, but the
x/roll signal does not meet the checked-in initial correlation threshold and low-motion x/y sign
agreement does not meet its initial threshold. The report keeps observed metrics separate from
possible causes. `HEIGHT_FAIL` is the raw measured-versus-estimated result under the predeclared
height-blind temporal partition; it is not an absolute localization-accuracy claim. See
`results/imu_gicp_physical_gate_stage_150mm/physical_validation_report.md` and the eight separate
diagnostic plots for the full evidence.

## Plateau-label diagnostic results

Generated separately under `results/imu_gicp_plateau_label_diagnostic/`; the production
localization, map, bag, GLIM inputs, and preserved height-Gate directory passed before/after
fingerprint checks.

- Search: complete independent valid interval 0.0-50.0 s, with no measured-height input
- Inputs: 490 localization rows (486 accepted), 6,919 IMU messages, 2,417 odometry messages, and
  all 13,058 PLY vertices
- Detected: 22 stable candidates; every physical label remains `UNASSIGNED`
- Post-hoc table: all 231 candidate pairs in candidate-ID order; every selection flag is false
- Historical result retained: candidate 4 stage-top and candidates 5-7 ground-after
- Historical raw median-z ordering: candidate 4 (`0.133461 m`) < candidate 5 (`0.201568 m`) <
  candidate 6 (`0.268985 m`) < candidate 7 (`0.281900 m`)
- Warning retained: fitted ground-plane slope `8.206 deg` makes those surface labels suspect
- Preserved status: `HEIGHT_FAIL`; this is not interpreted as production-localization failure
- Diagnostic conclusion: `height validation currently inconclusive because physical plateau
  labels are not independently established`

`plateau_candidates_map_overlay.png` overlays the PLY, accepted XY trajectory, large diagnostic
IDs, and every candidate time window. `plateau_candidates_timeline.png` synchronizes GICP z,
GICP pitch, IMU angular-velocity y, odom speed, and the same shaded candidate intervals.
`candidate_pair_height_diagnostics.csv` is post-hoc evidence only; it does not highlight or select
the pair nearest 0.150 m.

## Physical-surface LiDAR pose-z diagnostic

This diagnostic explains spatial variation in accepted map-frame `T_map_lidar.z`; it does not
change or rerun localization. It reads the independent 0-50 s production CSV, the frozen
22-candidate plateau diagnostic, odom speed, and the PLY only for an XY overlay. Human-reviewed
physical labels are locked before fitting: IDs 1-11 are `ground_before`, IDs 12-22 are
`stage_top`, and `ground_after` is unavailable in this interval. These are the newer diagnostic
IDs, not the historical seven-candidate physical-Gate IDs.

Phase A performs deterministic Huber-IRLS sample and candidate-balanced independent planes, plane
parallelism, a common-tilt parallel-plane model, candidate-block bootstrap, conditioning, and
residual correlation without reading the measured height. It writes and SHA256-seals
`blind_surface_fit_summary.json`. Only then does Phase B read the unchanged 0.150 m measurement
and 0.050 m initial-screening tolerance for a post-hoc comparison.

```bash
ros2 run bunker_offline_localization run_surface_pose_z_diagnostic.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/imu_gicp_surface_pose_z_diagnostic /home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/config/imu_gicp_surface_pose_z_diagnostic.yaml
```

Outputs are isolated under `results/imu_gicp_surface_pose_z_diagnostic/`. The fitted quantity is a
LiDAR sensor-center pose-z spatial support plane, not segmented physical floor geometry. It cannot
establish exact map-floor tilt, absolute localization accuracy, `T_base_lidar`, lever-arm
correction, or ground-after height.

## PLY local support-surface geometry diagnostic

The pose-z plane diagnostic above describes LiDAR sensor-center trajectories, so it cannot by
itself separate reconstructed map geometry from mapping-run pose behavior or independent
localization behavior. This read-only Gate performs that separation without replaying localization
or modifying the 13,058-point PLY. It estimates deterministic 20-neighbor PCA normals on the
original vertices, rejects wall-like normals, creates locally continuous components using fixed XY,
z-discontinuity, and normal-angle links, and fits a local Huber-IRLS plane only to a component that
geometrically supports each frozen candidate XY.

The primary XY radius is fixed at 0.60 m. The 0.45 and 0.75 m runs are sensitivity reports only.
Component selection has no pose-z or measured-height input; equal spatial support from multiple
horizontal layers returns `AMBIGUOUS_SURFACE`. The component-link values reflect the sparse PLY
sampling and local support continuity and are not tuned to 0.150 m. Candidate labels remain IDs
1-11 `ground_before`, IDs 12-22 `stage_top`, and unavailable `ground_after`.

Canonical `traj_lidar.txt` poses are associated to candidate route order with dynamic-programming
monotonic XY matching. Association never uses z, quaternion, cross-bag timestamps, or the measured
height. For each successfully extracted local surface, the diagnostic reports
`mapping_clearance = mapping_lidar_z - ply_surface_z` and
`independent_clearance = independent_lidar_z - ply_surface_z`. Their absolute expected values are
unknown because the physical `T_base_lidar` remains unavailable. The first-ramp view retains raw
support-like PLY `s-z` scatter instead of forcing a single curve through sparse or multi-surface
geometry.

Phase A writes and SHA256-seals `blind_ply_surface_geometry_summary.json` before opening the
physical-height config. Phase B then compares only the predeclared topology sets (lower IDs 10-11,
upper IDs 12-13) with the manual 0.150 m height. The unchanged 0.050 m tolerance is an initial
screening threshold, not calibrated accuracy, and its result is not a production-localization
failure.

```bash
ros2 run bunker_offline_localization run_ply_support_surface_diagnostic.sh /home/a/Desktop/shihoon/bunker_localization_ws/results/ply_support_surface_geometry_diagnostic /home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/config/ply_support_surface_geometry_diagnostic.yaml
```

Outputs are isolated under `results/ply_support_surface_geometry_diagnostic/`. Limitations include
the sparse multi-surface PLY, unknown `T_base_lidar`, unavailable ground-after plateau, and the fact
that observed spatial correlations do not identify GLIM's internal causal mechanism.
