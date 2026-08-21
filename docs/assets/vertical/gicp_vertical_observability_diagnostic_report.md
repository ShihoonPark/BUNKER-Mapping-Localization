# Independent GICP vertical observability / z-anomaly diagnostic

This is read-only post-processing of the existing production localization CSV. No localization,
map, bag, small_gicp, configuration, quality Gate, or prior diagnostic was changed or replayed.

## small_gicp Hessian convention audit

- Pinned commit: `aea131352e0d362d3e579a334c477cfafa5ee5eb`
- Parameter order: `['rx', 'ry', 'rz', 'tx', 'ty', 'tz']`; source-local `tz` index is 5.
- Matrix: OpenMP sum of inlier GICP per-correspondence J^T M J matrices at the last optimizer linearization; production copies RegistrationResult.H row-major to CSV.
- Update: right-multiplicative update T_target_source <- T_target_source * Exp(delta); delta is a source/LiDAR-local tangent.
- Last-linearization caveat: H is computed before the final accepted optimizer delta and is not relinearized after that delta; convergence bounds make this the last near-final linearization.
- Weighting: GICP residual precision uses inverse(target_cov + T*source_cov*T^T); error is 0.5*r^T*M*r and H is J^T*M*J summed over accepted correspondences.
- Production copy: `/home/a/Desktop/shihoon/bunker_localization_ws/src/bunker_offline_localization/src/registration.cpp:105`
- Optimizer update/H assignment: `/home/a/Desktop/shihoon/bunker_localization_ws/third_party/small_gicp/include/small_gicp/registration/optimizer.hpp:[53]` /
  `[55, 137]`
- GICP Jacobian/information: `/home/a/Desktop/shihoon/bunker_localization_ws/third_party/small_gicp/include/small_gicp/factors/gicp_factor.hpp:65-68`
- SE(3) order: `/home/a/Desktop/shihoon/bunker_localization_ws/third_party/small_gicp/include/small_gicp/util/lie.hpp:75`

Because source `tz` is LiDAR-local, the primary map-vertical diagnostic re-expresses only the
translation tangent in map axes. This is a derived analysis; the stored matrix is unchanged.

## z anomaly decomposition

- Anchor invariant violations: 0 at
  1.0e-09 m tolerance.
- Largest negative accepted dz: -0.246565 m
  at 13.223787 s,
  candidate=None.
- ID6-to-ID7 unassigned interval: 24 scans,
  sum dz=-0.298491 m,
  min/max=-0.246565/
  0.188389 m.
- Cumulative dz: last ID6=-0.065408 m,
  immediately before ID7=-0.363900 m,
  after first ID7=-0.317018 m.
- Conclusion: The ID7 entry prediction inherits an already-low previous accepted pose. The discrepancy originated in several large, alternating accepted GICP z corrections between IDs 6 and 7, including a single -0.246565 m correction; it is not a smooth sequence of uniformly small negative corrections and is not EKF z drift.

## IDs 5-12 detailed candidate medians

| ID | scans | map z | pred z | GICP z | ind-map z | dz median | min dz | sum dz | lambda min | cond | effective z info | weak z part | inliers | error/inlier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8 | 0.061424 | 0.008583 | 0.006244 | -0.055687 | -0.003011 | -0.014607 | -0.019930 | 8372.7 | 364.2 | 8690.7 | 0.9623 | 2627.5 | 0.062195 |
| 6 | 6 | 0.073523 | -0.008403 | -0.005869 | -0.080433 | 0.004589 | -0.007929 | 0.022307 | 8249.2 | 315.6 | 8568.0 | 0.9622 | 2640.0 | 0.081034 |
| 7 | 7 | 0.028541 | -0.280265 | -0.280265 | -0.307116 | 0.018981 | -0.051611 | 0.001451 | 10783.7 | 247.4 | 11420.3 | 0.9444 | 2687.0 | 0.150222 |
| 8 | 6 | -0.024474 | -0.235323 | -0.235323 | -0.205474 | -0.008595 | -0.032101 | -0.001291 | 9377.8 | 302.4 | 9874.5 | 0.9486 | 2654.5 | 0.135137 |
| 9 | 9 | 0.012357 | -0.212091 | -0.212210 | -0.226431 | -0.001717 | -0.014563 | -0.000442 | 9886.1 | 267.5 | 10430.3 | 0.9455 | 2608.0 | 0.125824 |
| 10 | 11 | 0.035726 | -0.168320 | -0.167563 | -0.205635 | 0.003430 | -0.077347 | 0.056670 | 8758.4 | 305.3 | 9199.2 | 0.9509 | 2630.0 | 0.116013 |
| 11 | 11 | 0.103915 | -0.081280 | -0.078128 | -0.181924 | 0.005168 | -0.037241 | 0.049955 | 8922.2 | 351.5 | 9278.6 | 0.9605 | 2696.0 | 0.142430 |
| 12 | 7 | 0.299068 | 0.130819 | 0.133461 | -0.165607 | 0.005506 | -0.042398 | 0.031418 | 6900.4 | 549.8 | 7156.4 | 0.9631 | 2582.0 | 0.153203 |

Candidate windows are not expanded: scans between candidates remain `NONE/unassigned`.

## Baseline vs anomaly vs recovery (scan-level medians)

| group | scans | dz | sum dz | abs dz | lambda min | cond | effective z info | inverse proxy | weak z part | z-roll C | z-pitch C | inliers | error/inlier | iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 100 | 0.000777 | -0.049567 | 0.005334 | 7618.1 | 381.7 | 7903.1 | 0.000126533 | 0.9641 | 0.9068 | 0.7143 | 2533.0 | 0.079278 | 2.0 |
| transition | 6 | 0.004589 | 0.022307 | 0.007657 | 8249.2 | 315.6 | 8568.0 | 0.000116715 | 0.9622 | 0.9137 | 0.5418 | 2640.0 | 0.081034 | 2.0 |
| anomaly | 44 | -0.000438 | 0.106343 | 0.017430 | 9324.2 | 299.8 | 9811.2 | 0.000101927 | 0.9491 | 0.8626 | 0.2724 | 2653.0 | 0.132391 | 3.5 |
| recovery_stage | 118 | 0.002702 | 0.186498 | 0.011157 | 7702.6 | 473.6 | 7964.9 | 0.000125551 | 0.9674 | 0.8464 | 0.3832 | 2730.0 | 0.131397 | 3.0 |

Candidate-balanced versions are retained in `group_comparison.json` and
`group_vertical_metrics.csv`; neither aggregation is selected as the sole result.

## Hessian sanity and limitations

- Finite/full-rank accepted Hessians: 486/
  486 of
  486.
- Relative symmetry error median/max:
  2.889e-17/
  9.811e-17.
- Significant/numerical negative eigenvalue totals:
  0/
  0.
- `H_pinv[z,z]` is named an inverse-curvature proxy, not physical covariance or 1-sigma
  localization uncertainty. Rotation/translation eigenvector components mix radians and meters.

## Mapping association and quality-Gate visibility

- XY-only ordered association: 486/486 available;
  median/p95/max distance=0.101787/
  0.258185/0.273271 m.
- Association excludes different-bag timestamps, z, orientation, and measured height.
- Anomaly inliers remain comparable, while error-per-inlier and iterations are elevated. All anomaly scans still pass; existing scalar metrics do not directly encode the canonical vertical discrepancy.

## Supported interpretation

**Hessian-insufficient correspondence/local-minimum hypothesis with accepted-anchor carry**.
At ID7 entry the prediction is already low, but the anchor invariant proves that it came from the
previous accepted GICP pose. The earlier onset contains large alternating corrections rather than
uniform small negative accumulation. Anomaly median effective z information and lambda-min are
higher than baseline, while inverse curvature and weakest-mode z participation are lower. Thus a
region-specific loss of local vertical curvature is not supported by this final-Hessian evidence.

The local Hessian is still strongly z-dominant in its weakest mode throughout the route, and its
meter/radian scaling prevents calibrated physical comparison of eigenvector components. A healthy
local final-solution Hessian cannot rule out wrong correspondences, a different local minimum, or
a multi-modal cost landscape. The next justified Gate is selected-scan correspondence/RViz audit;
no replay, cost sweep, or tuning is performed here.

Protected inputs unchanged: True.
