# Chronological research record

This chronology links the Git history to the scientific questions answered at each gate. Commit IDs identify the implementation state; generated bulk `results/` remain outside Git, while representative evidence is copied into `docs/assets/`.

| Stage | Commit | Question and result |
|---:|---|---|
| 1 | pre-repository data campaign, then `d353802` | Checked real-robot sensor/topic health (`/velodyne_points`, `/imu/data`, `/odom`, `/bunker_status`, `/bunker_rc_state`, `/cmd_vel`) and initialized dependencies, including official `small_gicp` as a pinned submodule. |
| 2 | `d06348e`, `89d8452` | Implemented and documented the C++17 ROS 2 offline pipeline using the existing library rather than reimplementing GICP. Same-bag 150626 smoke test established basic transform and logging correctness. |
| 3 | `ae54de3` | Added time-windowed independent 163346 localization. Only 0–50 s was used for the accuracy gate because of later sensor timing gaps. |
| 4 | `9e60248` | Compared EKF and UKF under the same downstream registration. EKF remained the production baseline because filter complexity did not provide a useful localization advantage. |
| 5 | `d5389c8` | Defined the accepted-GICP anchor policy: EKF predicts planar relative motion; previous accepted GICP retains `z/roll/pitch`; GICP corrects full 6DoF. |
| 6 | `040b648`–`1594ef8` | Added IMU/GICP physical-height and plateau diagnostics. A manually measured 0.150 m stage height was used only after blind candidate detection. The height result failed its initial screen but labels were not independently established, so this was not treated as production localization failure. |
| 7 | `ddfcbf4`, `4c65963` | Tested surface pose-`z` and local PLY support geometry. Evidence shifted attention from filter drift alone toward vertical deformation/label ambiguity in the map and local surfaces. |
| 8 | `33d9363` | Quantified independent GICP vertical observability. XY registration quality remained strong while vertical information and pose `z` could behave anomalously. |
| 9 | `6ddb68a`, `a3dc093`, `0fb6d57` | Added continuous RViz, correspondence, latency, QoS persistence, selected snapshots, and full-bag visualization without changing localization parameters. The 20 ms latency gate remained failed and was not reinterpreted. |
| 10 | `a1ce8e9` | Added Bag D coarse global relocalization against the Bag C map. Ranked search produced an unambiguous usable seed. |
| 11 | `024b38d`, `f099d3d` | Diagnosed a startup-order discontinuity, enabled relative odometry in the existing EKF, then ran full Bag D: 2,176/2,181 accepted (99.771%), five non-convergence rejects, no catastrophic jumps. |
| 12 | current uncommitted documentation-branch state | Added a thin Bag D RViz launch that reads the same saved coarse seed and reuses the production Bag D config. A native-DDS 50-scan smoke received all required frames/topics and accepted 50/50 scans. A later manual 1.0× visual replay processed all 2,181 scans and observed 2,177 accepted / 4 rejected with continuous map overlay; this remains visual sanity evidence, not the canonical quantitative Gate. |

## Interpretation discipline

- A successful same-bag smoke test verifies the pipeline but is not independent localization evidence.
- Bag D full-bag acceptance and continuity are strong independent robustness evidence, but they do not yield absolute pose RMSE without ground truth.
- Failed physical-height and latency gates remain recorded. They are not silently promoted to passes, nor automatically interpreted as total localization failure.
- Visual RViz inspection is a sanity check for overlays, jumps, and message flow; it does not replace quantitative accuracy measurement.
