#include "bunker_offline_localization/accepted_pose_predictor.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>

namespace bunker_offline_localization {

PlanarRelativeMotion ekfPlanarRelativeMotion(
  const Eigen::Isometry3d& anchor_T_odom_base,
  const Eigen::Isometry3d& current_T_odom_base)
{
  if (!isFiniteTransform(anchor_T_odom_base) || !isFiniteTransform(current_T_odom_base)) {
    throw std::invalid_argument("EKF anchor/current transform must be finite rigid transforms");
  }
  const auto anchor_rpy = rollPitchYaw(anchor_T_odom_base.linear());
  const auto current_rpy = rollPitchYaw(current_T_odom_base.linear());
  const double delta_x_odom =
    current_T_odom_base.translation().x() - anchor_T_odom_base.translation().x();
  const double delta_y_odom =
    current_T_odom_base.translation().y() - anchor_T_odom_base.translation().y();
  const double cosine = std::cos(anchor_rpy[2]);
  const double sine = std::sin(anchor_rpy[2]);
  return PlanarRelativeMotion{
    cosine * delta_x_odom + sine * delta_y_odom,
    -sine * delta_x_odom + cosine * delta_y_odom,
    wrapAngle(current_rpy[2] - anchor_rpy[2])};
}

Eigen::Isometry3d predictMapLidarFromAcceptedPose(
  const Eigen::Isometry3d& accepted_T_map_lidar,
  const PlanarRelativeMotion& motion)
{
  if (!isFiniteTransform(accepted_T_map_lidar) || !std::isfinite(motion.dx) ||
    !std::isfinite(motion.dy) || !std::isfinite(motion.dyaw))
  {
    throw std::invalid_argument("Accepted pose and planar EKF motion must be finite");
  }

  const auto accepted_rpy = rollPitchYaw(accepted_T_map_lidar.linear());
  const double yaw = accepted_rpy[2];
  const double predicted_x = accepted_T_map_lidar.translation().x() +
    std::cos(yaw) * motion.dx - std::sin(yaw) * motion.dy;
  const double predicted_y = accepted_T_map_lidar.translation().y() +
    std::sin(yaw) * motion.dx + std::cos(yaw) * motion.dy;
  const double predicted_yaw = wrapAngle(yaw + motion.dyaw);

  const Eigen::Quaterniond orientation =
    Eigen::AngleAxisd(predicted_yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(accepted_rpy[1], Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(accepted_rpy[0], Eigen::Vector3d::UnitX());
  return makeTransform(
    predicted_x, predicted_y, accepted_T_map_lidar.translation().z(),
    orientation.x(), orientation.y(), orientation.z(), orientation.w());
}

}  // namespace bunker_offline_localization
