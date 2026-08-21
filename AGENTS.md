# Repository working rules

These rules apply to the entire repository.

- Treat Git/GitHub as the reproducible research record and portfolio, not merely a backup. Keep the primary branch in a validated state and use purpose-specific feature, experiment, or test branches.
- When practical, close a milestone in this order: implementation → test → real-data validation → result analysis → regression → commit → push → PR → merge → branch cleanup.
- Treat ROS bags, GLIM dumps, PLY maps, calibration inputs, and generated `results/` as read-only evidence. Never rewrite them in place.
- Do not commit large sensor data or generated build trees. Keep `build/`, `install/`, `log/`, rosbag databases, and bulk experiment outputs out of Git.
- Document each dataset's role, metadata, known issues, storage location, available checksum, and reproduction use instead of committing the raw recording.
- Preserve transform direction explicitly. The localization result is `T_map_lidar`, with `p_map = T_map_lidar * p_lidar`. Do not guess an unavailable `T_base_lidar`.
- The production prediction policy is planar EKF relative motion (`x/y/yaw`) anchored to the previous accepted full-6DoF GICP pose; GICP supplies the final full-6DoF correction.
- Keep the production filter as EKF unless a separately defined gate justifies a change. Do not tune localization, EKF, GICP, voxel, or quality-gate parameters while doing documentation or visualization work.
- Report measured values, configured assumptions, experimental observations, and interpretations as different categories. Never present an unavailable ground truth, extrinsic, or absolute RMSE as measured.
- Preserve reproducible configs, scripts, compact results, important plots/reports, failures, A/B tests, root causes, and negative results. A diagnostic gate failure is not automatically a production-localization failure.
- Never call relaxed acceptance gates an improvement merely because acceptance rises. Prefer completing the research pipeline over repeatedly tuning an already validated subsystem.
- Prefer small, reviewable changes. Maintain relevant regression tests, build and test the affected package, record exact commands, and inspect the diff before committing.
- Do not modify user-local UI state such as `imgui.ini`.
