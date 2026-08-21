#include "bunker_offline_localization/reference_trajectory.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace bol = bunker_offline_localization;

TEST(ReferenceTrajectory, ParsesTumPose)
{
  std::istringstream input(
    "# time x y z qx qy qz qw\n"
    "1.0 1 2 3 0 0 0 1\n"
    "1.1 2 3 4 0 0 0.7071067811865475 0.7071067811865476\n");
  const auto trajectory = bol::ReferenceTrajectory::load(input, "memory");
  ASSERT_EQ(trajectory.poses().size(), 2U);
  EXPECT_DOUBLE_EQ(trajectory.first().timestamp, 1.0);
  EXPECT_TRUE(trajectory.first().T_map_lidar.translation().isApprox(Eigen::Vector3d(1, 2, 3)));
}

TEST(ReferenceTrajectory, EnforcesAssociationTolerance)
{
  std::istringstream input(
    "1.0 0 0 0 0 0 0 1\n"
    "1.1 1 0 0 0 0 0 1\n");
  const auto trajectory = bol::ReferenceTrajectory::load(input, "memory");
  const auto accepted = trajectory.associateNearest(1.04, 0.05);
  ASSERT_TRUE(accepted.has_value());
  EXPECT_DOUBLE_EQ(accepted->pose->timestamp, 1.0);
  EXPECT_FALSE(trajectory.associateNearest(1.04, 0.03).has_value());
}

TEST(ReferenceTrajectory, RejectsMalformedInput)
{
  std::istringstream input("1.0 0 0 0 0 0 1\n");
  EXPECT_THROW(bol::ReferenceTrajectory::load(input, "memory"), std::runtime_error);
}
