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

}  // namespace bunker_offline_localization
