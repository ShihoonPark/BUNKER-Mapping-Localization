#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>

namespace bunker_offline_localization {

// Transform convention used by the entire package:
// p_A = T_A_B * p_B. For example, p_map = T_map_lidar * p_lidar.
Eigen::Isometry3d makeTransform(
  double x, double y, double z, double qx, double qy, double qz, double qw);

Eigen::Quaterniond normalizedQuaternion(double qx, double qy, double qz, double qw);
bool isFiniteTransform(const Eigen::Isometry3d& transform);
double rotationAngle(const Eigen::Matrix3d& rotation);
double wrapAngle(double angle);
std::array<double, 3> rollPitchYaw(const Eigen::Matrix3d& rotation);

// The caller supplies the measured p_base = T_base_lidar * p_lidar calibration. This returns
// T_map_base = T_map_lidar * T_lidar_base without guessing an unavailable extrinsic.
Eigen::Isometry3d mapBaseFromMapLidar(
  const Eigen::Isometry3d& T_map_lidar,
  const Eigen::Isometry3d& T_base_lidar);

}  // namespace bunker_offline_localization
