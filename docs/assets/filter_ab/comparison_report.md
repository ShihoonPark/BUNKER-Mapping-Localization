# EKF vs UKF localization comparison Gate

Only the official robot_localization executable changes between runs. Map, scan stream, global
seed, adapter covariances, fused fields, small_gicp parameters, voxel sizes, and registration
quality gates are identical.

| Filter | prediction translation RMSE [m] | prediction yaw RMSE [deg] | independent accepted | correction translation p95 [m] | correction rotation p95 [deg] | GICP iterations mean | GICP runtime mean [ms] | filter latency mean [ms] | 27-29 s accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EKF | 0.124166 | 0.820115 | 487/490 | 0.101960 | 2.078785 | 3.914 | 2.277 | 5.055 | 19/20 |
| UKF | 8.555469 | 8.421087 | 486/490 | 0.101856 | 2.116769 | 4.020 | 2.239 | 5.177 | 19/20 |

## Same-bag prediction-only comparison

| Filter | associated | translation RMSE [m] | x RMSE | y RMSE | z RMSE | yaw RMSE [deg] | full rotation RMSE [deg] | translation step p95/max [m] | rotation step p95/max [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EKF | 867 | 0.124166 | 0.019356 | 0.019404 | 0.121104 | 0.820115 | 2.368001 | 0.074405 / 0.320899 | 4.056504 / 11.232718 |
| UKF | 867 | 8.555469 | 1.077693 | 0.893364 | 8.440174 | 8.421087 | 24.952337 | 1.059552 / 8.344850 | 4.371867 / 14.596177 |

The EKF prediction never exceeded 1 m translation error (maximum
0.423 m). The UKF first exceeded 1 m at
55.711 s and reached
23.559 m. Its dominant component is z.
This propagated into same-bag localization: EKF accepted 854/869 with
{'NOT_CONVERGED': 15}; UKF accepted 555/869 with
{'LOW_INLIERS': 268, 'NOT_CONVERGED': 15, 'TRANSLATION_JUMP': 31}. EKF recovered after every reject on the next scan. UKF has
252 rejects with no later recovery and a maximum
recovered gap of 40 scans.

## Independent 0-50 s comparison

| Filter | scans | accepted / rejected | convergence | inliers mean/p95/min | final error mean/p95/max | iterations mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| EKF | 490 | 487 / 3 | 99.388% | 2667.6 / 2937.8 / 2017 | 335.952 / 474.103 / 943.263 | 3.914 / 8.550 |
| UKF | 490 | 486 / 4 | 99.184% | 2667.6 / 2937.8 / 2017 | 335.477 / 474.009 / 942.409 | 4.020 / 9.000 |

- EKF reject events: 5.909805s (NOT_CONVERGED), 16.830560s (NOT_CONVERGED), 28.752881s (NOT_CONVERGED)
- UKF reject events: 5.909805s (NOT_CONVERGED), 16.329612s (NOT_CONVERGED), 16.830560s (NOT_CONVERGED), 28.752881s (NOT_CONVERGED)
- Every independent reject recovered on the next scan (about 0.1002 s) for both filters.

## Prediction-to-GICP correction on independent scans

Values are mean / median / p95 / max.

| Filter | translation [m] | full rotation [deg] | absolute yaw [deg] |
|---|---:|---:|---:|
| EKF | 0.027672 / 0.016115 / 0.101960 / 0.210034 | 0.616826 / 0.394448 / 2.078785 / 8.645969 | 0.137936 / 0.084695 / 0.426800 / 2.356378 |
| UKF | 0.027979 / 0.016246 / 0.101856 / 0.248527 | 0.637166 / 0.401625 / 2.116769 / 8.702454 | 0.169090 / 0.111835 / 0.504126 / 2.262974 |

## Runtime comparison

Values are mean / p95 / max.

| Filter | GICP runtime [ms] | filter latency proxy [ms] |
|---|---:|---:|
| EKF | 2.277 / 3.064 / 6.246 | 5.055 / 9.225 / 17.148 |
| UKF | 2.239 / 3.161 / 8.154 | 5.177 / 9.610 / 17.646 |

## Large-rotation interval, 27-29 s

| Filter | accepted / rejected | correction translation mean/p95 [m] | correction rotation mean/p95 [deg] | iterations mean/p95 | GICP runtime mean/p95 [ms] |
|---|---:|---:|---:|---:|---:|
| EKF | 19 / 1 | 0.082964 / 0.166933 | 3.184006 / 7.500154 | 8.500 / 13.800 | 2.867 / 3.676 |
| UKF | 19 / 1 | 0.085302 / 0.167766 | 3.216800 / 7.514198 | 8.750 / 14.750 | 3.141 / 4.972 |

## Current decision

EKF is the better current BUNKER initial-guess candidate in this untuned A/B Gate. The evidence count is EKF 5 vs UKF 0 across six primary
criteria after treating differences within 1% as practical ties. This is a dataset-specific
engineering decision, not proof that one estimator family is generally superior. UKF-specific
tuning was deliberately not performed.

## Interpretation

- Same-bag values compare the pre-registration `T_map_lidar` prediction against timestamp-matched
  GLIM poses. Final small_gicp poses are excluded from those prediction-only errors.
- Corrections use `delta_T = inverse(T_prediction_map_lidar) * T_gicp_map_lidar`; smaller values
  mean the initial guess landed closer to the accepted registration, but are not sufficient alone
  to establish filter superiority.
- Independent 163346 processing remains restricted to bag-relative 0-50 s. It has no reference
  trajectory, so no independent absolute RMSE is reported.
- `filter_runtime` is an identically measured steady-clock, end-to-end latency proxy from the
  nearest adapted sensor input reception to filtered odometry reception. It includes ROS
  scheduling/transport and is not robot_localization internal CPU time.
- The actual `T_base_lidar` remains unavailable; both runs use the same explicit Phase 1 identity
  prediction approximation.
