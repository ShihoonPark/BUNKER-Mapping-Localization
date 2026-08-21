# EKF startup-order discontinuity

## Symptom

The first Bag D 0–50 s localization attempt processed 491 scans but accepted only 317. There were 174 rejects: 120 `TRANSLATION_JUMP` and 54 `NOT_CONVERGED`. The maximum prediction step reached `4.679 m / 96.204°`, with a longest reject run of 141 scans.

## Layer-by-layer diagnosis

The discontinuity did not originate in raw odometry, frame adaptation, the saved coarse seed, or a GICP correction:

- raw odometry first-step jump: `0`
- adapted odometry first-step jump: `0`
- first EKF output time: `9.138969 s` after bag origin
- first EKF state discrepancy: `4.592545 m / 94.692534°`
- first downstream affected LiDAR scan: `9.213097 s`

Bag D delivers IMU messages before the first odometry measurement. With the prior absolute odometry configuration, filter initialization order could produce a large world-frame discontinuity.

## Minimal correction

The production EKF changed only `odom0_relative` to `true`; `differential` remained `false`. Relative mode removes the arbitrary first odometry origin while retaining scan-to-scan odometry motion. The saved map-frame coarse seed still determines global initialization, and GICP parameters and quality gates remain unchanged.

## Before / after

| Metric | Before | After |
|---|---:|---:|
| processed scans | 491 | 491 |
| accepted | 317 | 490 |
| rejected | 174 | 1 |
| max prediction translation | 4.679 m | 0.064 m |
| max prediction rotation | 96.204° | 9.072° |
| longest reject run | 141 | 1 |
| steps over `1 m` / `30°` | present | 0 / 0 |

The full independent Bag D gate subsequently accepted 2,176/2,181 scans. The lesson is specific: when a separate global seed owns the map-frame origin, the motion filter should provide relative increments and must be tested against sensor startup order.
