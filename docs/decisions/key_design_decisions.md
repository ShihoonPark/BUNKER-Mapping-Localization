# Key design decisions

## Use official `small_gicp`

The project links the installed/upstream library through the pinned `third_party/small_gicp` submodule. GICP is not reimplemented locally. This keeps the research focused on initialization, prediction, gating, and evaluation.

## Keep EKF as production filter

The EKF/UKF A/B gate did not show a downstream localization benefit sufficient to replace the simpler EKF baseline. The filter's responsibility is a stable relative initial guess, not final full-6DoF state estimation.

## Anchor prediction to accepted GICP

Planar EKF relative motion updates `x/y/yaw`; `z/roll/pitch` remain from the previous accepted GICP pose until the next full-6DoF correction. This avoids independent drift in states the selected filter inputs do not directly observe.

## Separate global initialization from tracking

Independent Bag D does not inherit the mapping trajectory start pose. A bounded map-wide coarse search finds and records a seed, after which the unchanged local tracker runs. Initialization scores are evidence, not ground truth.

## Publish LiDAR pose until extrinsics are measured

The authoritative output is `T_map_lidar`. The `T_map_base = T_map_lidar * T_lidar_base` interface and transform-direction test exist, but no real base pose is claimed from an assumed extrinsic.

## Preserve diagnostic failures

The 20 ms continuous latency gate remains FAIL (`11.144 ms` mean, `27.393 ms` p95, `36.369 ms` max, `104/490` overruns in its recorded run). The physical-height screen also remains FAIL/inconclusive. Visualization fixes and later localization success do not rewrite those outcomes.

## Keep production and visualization semantics aligned

RViz launch files reuse the same map, EKF, GICP parameters, gates, and saved seed. Best-effort/volatile QoS is explicit for dynamic diagnostics; the map and retained snapshots use persistence appropriate for late subscribers. Faster replay is for visual convenience only; 1.0× remains the verification basis.
