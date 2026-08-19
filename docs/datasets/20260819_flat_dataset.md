# 2026-08-19 flat mapping and localization dataset

The campaign separates calibration observations, mapping, and independent localization. Counts below are read directly from rosbag metadata; database sizes are approximate filesystem sizes. Raw bags are not versioned in Git.

| Bag | Intended role | Duration (s) | LiDAR | IMU | Odom | Approx. DB size | Status / caveat |
|---|---|---:|---:|---:|---:|---:|---|
| A_short | short sensor check | 36.814 | 348 | 5,050 | 1,840 | 0.52 GB | usable diagnostic capture |
| A2_long | longer sensor check | 150.154 | 1,480 | 21,229 | 7,408 | 2.20 GB | usable diagnostic capture |
| B1_forward_low_recovered | low-speed forward calibration observation | 199.693 | 1,983 | 28,401 | 9,942 | 2.94 GB | recovered after an active DB access/reindex issue; metadata retains an unusual DB basename |
| B2_controlled | controlled command observation | 757.023 | 7,526 | 107,373 | 37,697 | 11.17 GB | `v=0.08, ω=-0.15` showed no motion; retry at `v=0.12, ω=-0.15` moved. This is an observation, not a calibrated dead-zone threshold. |
| C | flat-area mapping | 318.638 | 3,152 | 45,004 | 15,800 | 4.68 GB | source of the production IMU-on PLY map |
| D | independent localization | 219.251 | 2,181 | 31,177 | 10,925 | 3.24 GB | different start pose/route; no ground-truth trajectory |

The A2 health recording gives representative metadata-average rates of about `9.86 Hz` LiDAR, `141.38 Hz` IMU, `49.34 Hz` odometry, `49.38 Hz` bunker status, and `49.39 Hz` RC state. These are whole-bag averages, not calibrated timing guarantees. The localization pipeline expects LiDAR frame `velodyne`, IMU frame `imu_link`, and odometry `odom -> base_link`; exact physical base/LiDAR and base/IMU extrinsics remain unknown.

Paths:

```text
/home/a/Desktop/shihoon/Slam/20260819_dataset/
  bag_A_short_smoke/
  bag_A2_long_health/
  bag_B1_forward_low_recovered/
  bag_B2_controlled_calibration/
  bag_C_flat_dense_mapping/
  bag_D_flat_independent_localization/
```

Dataset DB checksums were not present in the research outputs and were not retroactively computed during documentation. Map and trajectory checksums that were already available are preserved below.

For portable command descriptions, use the conceptual roots below and substitute the local archive paths:

```bash
DATA_ROOT=/path/to/20260819_dataset
MAP_ROOT=/path/to/20260819_flat/results
RESULTS_ROOT=/path/to/bunker_localization_ws/results
```

## Bag C map artifacts

- PLY: `/home/a/Desktop/shihoon/glim_real/20260819_flat/results/flat_bag_C_imu_on.ply`
- trajectory: the corresponding IMU-on GLIM dump `traj_lidar.txt`
- PLY SHA-256: `b3a208bf44c7f71db848b676641192f272eabf7fb07af153f26d42f6ca847240`
- trajectory SHA-256: `a79b3a5502160821af479803e9914c86495f30557ae9e1a33501bb192176dc0b`
- trajectory poses: 3,141 over 314.622 s
- XY path length: 104.333 m
- XY closure displacement: 0.00287 m
- `z` range: 0.8219 m (`-0.1224` to `0.6995 m`)

The excellent XY closure does not establish a flat physical `z` surface. Comparison runs and local PLY geometry diagnostics indicate a vertical map deformation that remains unresolved.

## Bag D role

Bag D is the principal independent localization dataset. It starts from a different pose and follows a different route from Bag C. Its initial `T_map_lidar` is obtained by the saved coarse global initializer result, not by borrowing a trajectory pose. Because no independent reference trajectory exists, the full-bag gate reports robustness and consistency metrics rather than absolute accuracy.
