#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>

namespace bol = bunker_offline_localization;

TEST(SignConvention, ForwardOneMeterIncreasesX)
{
  const Eigen::Vector3d initial(0.0, 0.0, 0.0);
  const Eigen::Vector3d forward = initial + Eigen::Vector3d::UnitX();
  EXPECT_DOUBLE_EQ(forward.x(), 1.0);
}

TEST(SignConvention, LeftOneMeterIncreasesY)
{
  const Eigen::Vector3d initial(0.0, 0.0, 0.0);
  const Eigen::Vector3d left = initial + Eigen::Vector3d::UnitY();
  EXPECT_DOUBLE_EQ(left.y(), 1.0);
}

TEST(SignConvention, CounterClockwiseNinetyDegreesIsPositiveYaw)
{
  const Eigen::AngleAxisd ccw(M_PI_2, Eigen::Vector3d::UnitZ());
  const auto rpy = bol::rollPitchYaw(ccw.toRotationMatrix());
  EXPECT_NEAR(rpy[2], M_PI_2, 1.0e-12);
}
