# Bag D EKF startup discontinuity — before/after

Read-only layer synchronization classified the original failure as `CASE B`. Raw and adapted
odometry were continuous and identical. The first discontinuity occurred in EKF output at
`1787142257.460545540` (bag-relative `9.138969421 s`): `4.592545 m / 94.692534 deg`.
The next LiDAR scan at `9.213096857 s` inherited a `4.679179 m / 96.204441 deg` prediction jump.

The only behavior change is `odom0_relative: false -> true`. Covariance, EKF rejection thresholds,
coarse seed, GICP parameters, and localization quality gates are unchanged.

| Metric | Before | After |
|---|---:|---:|
| Processed | 491 | 491 |
| Accepted / rejected | 317 / 174 | 490 / 1 |
| Acceptance | 64.562% | 99.796% |
| Reject reasons | `{'TRANSLATION_JUMP': 120, 'NOT_CONVERGED': 54}` | `{'NOT_CONVERGED': 1}` |
| Longest rejection run | 141 | 1 |
| Prediction translation p95 / max [m] | 0.077 / 4.679 | 0.042 / 0.064 |
| Prediction yaw p95 / max [deg] | 5.147 / 96.204 | 5.014 / 9.072 |
| Prediction >1 m or >30 deg | 1 | 0 |
| GICP correction translation p95 / max [m] | 1.479 / 1.511 | 0.019 / 0.044 |
| Accepted pose translation p95 / max [m] | 0.094 / 3.251 | 0.043 / 0.061 |
| Accepted pose rotation p95 / max [deg] | 5.723 / 96.934 | 5.032 / 5.677 |
| Runtime mean / p95 / max [ms] | 3.386 / 6.993 / 10.817 | 2.068 / 2.615 / 5.517 |
| z range [m] | -0.0429 .. 0.1391 | -0.0363 .. 0.1480 |
| roll range [deg] | -3.563 .. 3.098 | -2.091 .. 2.402 |
| pitch range [deg] | -1.942 .. 4.000 | -1.114 .. 3.713 |

Post-fix 7.0–11.5 s layer check: raw/adapted/EKF/localizer prediction threshold jumps are all
zero. There are no contiguous accepted-pose catastrophic jumps. Verdict: `PASS`.
