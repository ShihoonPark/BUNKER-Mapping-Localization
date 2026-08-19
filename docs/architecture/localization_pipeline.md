# Localization pipeline

## Objective

The filter is not expected to produce the final 6DoF localization pose by itself. Its job is to give `small_gicp` a stable scan-to-scan initial guess. Full-6DoF registration against the fixed global map produces the final `T_map_lidar`.

```mermaid
flowchart LR
    Bag[ROS 2 bag] --> Odom[/odom]
    Bag --> Imu[/imu/data]
    Bag --> Scan[/velodyne_points]
    Odom --> EKF[robot_localization EKF]
    Imu -->|angular velocity z| EKF
    Prev[previous accepted full-6DoF GICP pose] --> Predict[accepted-pose anchor policy]
    EKF -->|relative dx dy dyaw| Predict
    Predict -->|init T_map_lidar| GICP[official small_gicp]
    Map[fixed GLIM PLY map] -->|target| GICP
    Scan -->|source| GICP
    GICP --> Gate[finite / convergence / inlier / jump gates]
    Gate -->|accepted| Prev
    Gate --> Out[T_map_lidar CSV / TUM / RViz]
```

## Prediction and correction policy

1. Load the global PLY once, voxel-downsample it, and prepare its neighbor structure.
2. Maintain the last accepted `T_map_lidar`.
3. Obtain planar relative motion `dx`, `dy`, and `dyaw` from the EKF between LiDAR scans.
4. Compose that planar delta on the last accepted full-6DoF pose. Retain its `z`, roll, and pitch rather than independently propagating unobserved vertical attitude.
5. Run full-6DoF GICP with map as target and current scan as source.
6. Accept only finite, converged results satisfying the configured inlier and pose-jump gates.
7. Record both prediction and registration result, correction, convergence metrics, runtime, and rejection reason.

This policy prevents the EKF's unobserved `z/roll/pitch` states from drifting while preserving full-6DoF map registration. It does not solve weak vertical map geometry or mapping deformation.

## Initialization

The original same-bag smoke test starts from the first valid GLIM trajectory pose. Independent Bag D starts elsewhere and therefore uses a separate global initializer:

- derive map-wide translation bounds from the Bag C PLY;
- search the full yaw range on a coarse voxel representation;
- retain ranked candidates;
- refine candidates with full-6DoF official `small_gicp`;
- save the selected `best_candidate.T_map_lidar` as evidence;
- inject that saved transform into the unchanged production localizer.

The initializer is a robust starting-pose search, not a source of ground truth.

## Runtime outputs

The offline node records timestamp, predicted and GICP position/orientation, prediction-to-GICP correction, convergence, iterations, inliers, final error, runtime, acceptance, and rejection reason. It writes CSV and TUM trajectories. The diagnostic publisher additionally provides:

- `/map_cloud` in `map`, transient-local;
- `/registered_scan` in `map`;
- `/raw_scan` in `velodyne`;
- `/gicp_pose`, `/gicp_path`, and `/prediction_pose` in `map`;
- optional correspondence markers.

Dynamic visualization topics use best-effort, volatile QoS. Selected/final snapshots remain visible after playback so the user can inspect the last state.

## Deliberate non-goals

No Nav2 integration, real-vehicle actuation, GLIM modification, map regeneration, invented extrinsic, or replacement implementation of GICP is part of this baseline.
