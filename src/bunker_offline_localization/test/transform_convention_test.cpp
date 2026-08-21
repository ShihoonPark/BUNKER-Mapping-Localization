#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <limits>

namespace bol = bunker_offline_localization;

TEST(TransformConvention, TargetMapFromSourceLidarDirection)
{
  // p_map = T_map_lidar * p_lidar. A LiDAR point one metre forward from a sensor at x=10
  // must land at map x=11, not at x=-9 (the inverse-direction failure mode).
  const Eigen::Isometry3d T_map_lidar = bol::makeTransform(10, 2, 0, 0, 0, 0, 1);
  const Eigen::Vector3d p_lidar(1, 0, 0);
  const Eigen::Vector3d p_map = T_map_lidar * p_lidar;
  EXPECT_TRUE(p_map.isApprox(Eigen::Vector3d(11, 2, 0), 1.0e-12));
}

TEST(TransformConvention, InverseIsOppositeDirection)
{
  const Eigen::Isometry3d T_A_B = bol::makeTransform(1, -2, 0.5, 0, 0, 0.2, 0.98);
  const Eigen::Isometry3d T_B_A = T_A_B.inverse();
  EXPECT_TRUE((T_A_B * T_B_A).matrix().isApprox(Eigen::Matrix4d::Identity(), 1.0e-12));
}

TEST(TransformConvention, QuaternionIsNormalized)
{
  const Eigen::Quaterniond quaternion = bol::normalizedQuaternion(0, 0, 1, 1);
  EXPECT_NEAR(quaternion.norm(), 1.0, 1.0e-12);
}

TEST(TransformConvention, InvalidTransformIsRejected)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation().x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(bol::isFiniteTransform(transform));
  EXPECT_THROW(bol::normalizedQuaternion(0, 0, 0, 0), std::invalid_argument);
}
