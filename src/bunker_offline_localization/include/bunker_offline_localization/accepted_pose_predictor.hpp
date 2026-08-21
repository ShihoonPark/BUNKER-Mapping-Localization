#pragma once

#include <Eigen/Geometry>

namespace bunker_offline_localization {

struct PlanarRelativeMotion {
  double dx{};
  double dy{};
  double dyaw{};
};

// Extract only the locally expressed planar motion from the EKF poses. Any EKF z, roll, or
// pitch propagation is intentionally ignored.
PlanarRelativeMotion ekfPlanarRelativeMotion(
  const Eigen::Isometry3d& anchor_T_odom_base,
  const Eigen::Isometry3d& current_T_odom_base);

// Apply EKF dx/dy/dyaw to the last accepted full-6DoF LiDAR pose. The accepted pose's z, roll,
// and pitch are held exactly until the next accepted full-6DoF small_gicp correction.
Eigen::Isometry3d predictMapLidarFromAcceptedPose(
  const Eigen::Isometry3d& accepted_T_map_lidar,
  const PlanarRelativeMotion& motion);

}  // namespace bunker_offline_localization
