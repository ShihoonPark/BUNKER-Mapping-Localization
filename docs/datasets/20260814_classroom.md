# 2026-08-14 classroom datasets

These datasets established the Phase 1 pipeline before the independent flat-map campaign.

| Role | Path | Duration | LiDAR scans | Notes |
|---|---|---:|---:|---|
| Mapping / same-bag smoke | `/home/a/Desktop/shihoon/Slam/slam_20260814_150626` | 90.819 s | 878 | Source of the first GLIM map and reference trajectory |
| Independent validation | `/home/a/Desktop/shihoon/Slam/slam_flat_rc_20260814_163346` | 129.757 s | 1,075 | Accuracy gate deliberately limited to metadata-relative 0–50 s because of later sensor timing gaps |

Primary topics are `/velodyne_points`, `/odom`, `/imu/data`, `/tf`, and `/tf_static`. The inspection that preceded implementation verified frame IDs, covariance contents, and that the recorded IMU orientation should not be fused blindly. The production configuration prioritizes odometry pose/twist and IMU angular velocity.

The mapping artifacts were:

- PLY: `/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply`
- GLIM LiDAR trajectory: `/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/dump_150626_direct_20260814_222719/traj_lidar.txt`
- PLY SHA-256: `ef603b91db6677ae2dd6a757fb5273bf0ad36909581475e3bd7c0b512544399c`
- trajectory SHA-256: `091489d44f2ac5a6121c4cb0a7a11ba27d2f18acb1cdf12d78fe924ba7435ad6`

This became the canonical early mapping baseline because it provided the stable GLIM PLY/reference pair used to validate the entire offline pipeline and transform convention. It should not be confused with a surveyed map or independent accuracy reference.

The same-bag run was a smoke test, not independent evidence. The 163346 0–50 s run was more meaningful because a different drive was localized in the same map, but its Phase 1 parameter initialization assumed the known shared staging pose. That assumption is explicitly not complete global relocalization. The run had no GLIM reference trajectory, so the reports intentionally omit absolute RMSE and instead evaluate acceptance, convergence, inliers, error, runtime, pose jumps, smoothness, and map overlay. Bag D later removed the shared-start assumption through a separate map-wide initializer.

Later diagnostics showed that apparently reasonable XY localization can coexist with anomalous absolute `z`. This motivated the 2026-08-19 independent mapping/localization campaign and the vertical-observability work documented elsewhere.
