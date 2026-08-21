# Bag D full independent localization stability Gate

## Verdict: PASS

This is a map-relative continuity evaluation of independent Bag D scans against the Bag C PLY map.
Bag D has no independent reference trajectory, and the Bag C map has known vertical deformation;
therefore this report does not claim absolute x/y/z/yaw accuracy or compute RMSE.

- PASS: `eof_and_all_inputs_accounted`
- PASS: `no_large_prediction_discontinuity`
- PASS: `no_rejection_cascade`
- PASS: `no_catastrophic_accepted_jump`
- PASS: `final_acceptance_reaches_bag_end`
- PASS: `no_temporal_segment_collapse`
- PASS: `all_accepted_poses_finite`
- PASS: `short_window_regression_near_identical`

## Input accounting

- Metadata / received / processed / skipped LiDAR scans: 2181 / 2181 / 2181 / 0
- EOF received / process survived / all inputs accounted: True / True / True
- First / last processed time: 0.892362 / 219.187644 s
- Final accepted time: 219.187644 s (0.000000 s before the last LiDAR input)
- LiDAR gap > 1.25 s detected: False

## Registration and rejection continuity

- Accepted / rejected: 2176 / 5 (99.771% accepted)
- Converged: 2176 (99.771%)
- Reject histogram: `{'NOT_CONVERGED': 5}`
- Longest consecutive rejection run: 1 scan(s)
- Inliers mean / p95 / min: 2823.7 / 3153.0 / 2393
- Final error/inlier mean / p95 / max: 0.133418 / 0.228123 / 0.495609
- Iterations mean / p95 / max: 2.403 / 4.000 / 29

| run | scans | start [s] | end [s] | reasons |
|---:|---:|---:|---:|---|
| 1 | 1 | 16.731557 | 16.731557 | `{'NOT_CONVERGED': 1}` |
| 2 | 1 | 110.811061 | 110.811061 | `{'NOT_CONVERGED': 1}` |
| 3 | 1 | 197.352820 | 197.352820 | `{'NOT_CONVERGED': 1}` |
| 4 | 1 | 201.659730 | 201.659730 | `{'NOT_CONVERGED': 1}` |
| 5 | 1 | 214.179879 | 214.179879 | `{'NOT_CONVERGED': 1}` |

## Prediction and GICP correction

- Prediction translation mean / p95 / p99 / max: 0.033155 / 0.048958 / 0.055442 / 0.085188 m
- Prediction yaw mean / p95 / p99 / max: 1.188426 / 4.594140 / 5.497460 / 9.072255 deg
- Prediction steps >0.5 m / >1.0 m: 0 / 0
- Prediction steps >20 deg / >30 deg: 0 / 0
- Previous 9.213 s startup discontinuity recurrence: False
- GICP translation correction mean / p95 / p99 / max: 0.009186 / 0.021746 / 0.032282 / 0.062873 m
- GICP SO(3) correction mean / p95 / p99 / max: 0.329291 / 0.798993 / 1.250964 / 3.342042 deg

## Accepted trajectory and 6DoF behavior

- Translation step mean / p95 / p99 / max: 0.034580 / 0.050037 / 0.057424 / 0.110497 m
- Rotation step mean / p95 / p99 / max: 1.256945 / 4.570930 / 5.445314 / 7.737411 deg
- Catastrophic contiguous accepted jumps (>1 m / >30 deg): 0 / 0
- Accepted poses inside map XY bounding box: 2176 / 2176
- z min / max / range: -0.087948 / 0.172299 / 0.260247 m
- roll min / max / range: -3.264769 / 2.527787 / 5.792555 deg
- pitch min / max / range: -1.590327 / 3.935063 / 5.525390 deg

The map bounding-box check and overlay screen for loss of map-frame continuity; they are not an
absolute localization-accuracy measurement. The vertical values are likewise map-relative only.

## Runtime

- Registration runtime mean / p95 / p99 / max: 2.185131 / 2.781573 / 3.178752 / 6.074576 ms
- Core localization latency mean / p95 / p99 / max: 12.942318 / 32.268613 / 39.598308 / 46.483106 ms

The two timings are reported separately: registration runtime covers small_gicp, while core latency
covers the localizer's per-scan processing path.

## Time segments

Prediction maxima are assigned to the segment containing the current scan of each scan-to-scan step.

| interval | processed | accepted | acceptance | rejects | prediction max | correction max |
|---|---:|---:|---:|---|---|---:|
| 0-50 s | 491 | 490 | 99.796% | `{'NOT_CONVERGED': 1}` | 0.0641 m / 9.072 deg | 0.0441 m |
| 50-100 s | 500 | 500 | 100.000% | `{}` | 0.0627 m / 7.191 deg | 0.0497 m |
| 100-150 s | 499 | 498 | 99.800% | `{'NOT_CONVERGED': 1}` | 0.0852 m / 6.880 deg | 0.0629 m |
| 150-200 s | 499 | 498 | 99.800% | `{'NOT_CONVERGED': 1}` | 0.0766 m / 5.018 deg | 0.0531 m |
| 200-EOF | 192 | 190 | 98.958% | `{'NOT_CONVERGED': 2}` | 0.0202 m / 6.118 deg | 0.0259 m |

## 0-50 s replay regression

- Full replay first 50 s accepted / rejected: 490 / 1
- Saved post-fix short replay accepted / rejected: 490 / 1
- Timestamp/status mismatches: 0
- Accepted pose difference p95 / max: 0.000499 / 0.003782 m
- Accepted rotation difference p95 / max: 0.010177 / 0.107722 deg
- Counts/reasons identical: True
- Near-identical replay result: True
- Full first-50 s registration runtime mean / p95 / max: 2.094 / 2.748 / 5.599 ms
- Saved short registration runtime mean / p95 / max: 2.068 / 2.615 / 5.517 ms
- Full first-50 s core latency mean / p95 / max: 13.637 / 35.004 / 46.483 ms
- Saved short core latency mean / p95 / max: 16.083 / 41.584 / 44.555 ms

## Artifacts

- `localization.csv`, `estimated_traj_lidar.tum`, `full_bag_summary.json`
- `summary.json`, `short_baseline_comparison.json`, `segment_summary.csv`, `rejection_runs.csv`
- `acceptance_timeline.png`, `map_trajectory_overlay.png`
- `prediction_step_over_time.png`, `gicp_correction_over_time.png`
- `z_over_time.png`, `roll_pitch_over_time.png`
