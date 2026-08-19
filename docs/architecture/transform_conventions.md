# Transform conventions

## Axes and pose direction

ROS convention is used throughout:

- `+x`: forward
- `+y`: left
- `+z`: up
- positive yaw: counter-clockwise

The GICP map is the **target** and the current LiDAR scan is the **source**. Therefore:

```text
result.T_target_source = T_map_lidar
p_map = T_map_lidar * p_lidar
```

This direction is covered by unit tests together with the `+x`, `+y`, and positive-yaw sign checks.

## Base-frame interface

The desired future conversion is:

```text
T_map_base = T_map_lidar * T_lidar_base
```

where `T_lidar_base` is the inverse of a measured `T_base_lidar`. No reliable real `base_link -> velodyne` extrinsic is currently available. Phase 1 retains the documented identity approximation and reports `T_map_lidar`; it must not be presented as a calibrated base pose.

The LiDAR-to-IMU approximation used in the Bag D production configuration is a `z = -0.07 m` translation. This is a configured assumption, not a completed extrinsic calibration.

## Validity checks

Every accepted 6DoF pose must have finite `x/y/z/roll/pitch/yaw` and a normalized quaternion. Pose-jump gates compare both translation and rotation. `z`, roll, and pitch in the next prediction are anchored to the last accepted GICP pose, while the EKF supplies relative planar motion.

## Interpretation boundary

Transform correctness does not guarantee physical height accuracy. A scan can be consistently transformed into a vertically deformed map. The independent Bag D result therefore supports registration continuity and horizontal map localization, while the physical interpretation of absolute `z` remains limited by Bag C map geometry.
