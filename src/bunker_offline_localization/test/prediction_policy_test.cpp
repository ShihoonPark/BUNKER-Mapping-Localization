#include "bunker_offline_localization/accepted_pose_predictor.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>

namespace bol = bunker_offline_localization;

namespace {

Eigen::Isometry3d poseFromRpy(
  const double x, const double y, const double z,
  const double roll, const double pitch, const double yaw)
{
  const Eigen::Quaterniond quaternion =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
  return bol::makeTransform(
    x, y, z, quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
}

}  // namespace

TEST(AcceptedPosePredictor, ExtractsOnlyPlanarEkfRelativeMotion)
{
  const Eigen::Isometry3d anchor = poseFromRpy(3.0, -2.0, 4.0, 0.2, -0.3, 0.4);
  const double current_x = 3.0 + std::cos(0.4) * 1.0 - std::sin(0.4) * -0.5;
  const double current_y = -2.0 + std::sin(0.4) * 1.0 + std::cos(0.4) * -0.5;
  const Eigen::Isometry3d current = poseFromRpy(
    current_x, current_y, 12.0, -0.8, 0.7, 0.65);
  const bol::PlanarRelativeMotion motion = bol::ekfPlanarRelativeMotion(
    anchor, current);
  EXPECT_NEAR(motion.dx, 1.0, 1.0e-12);
  EXPECT_NEAR(motion.dy, -0.5, 1.0e-12);
  EXPECT_NEAR(motion.dyaw, 0.25, 1.0e-12);
}

TEST(AcceptedPosePredictor, EkfZRollPitchCannotAffectPlanarRelativeMotion)
{
  const Eigen::Isometry3d anchor_a = poseFromRpy(1.0, 2.0, 0.0, 0.0, 0.0, -0.3);
  const Eigen::Isometry3d anchor_b = poseFromRpy(1.0, 2.0, 50.0, 1.0, -0.7, -0.3);
  const Eigen::Isometry3d current_a = poseFromRpy(2.0, 4.0, 0.0, 0.0, 0.0, 0.2);
  const Eigen::Isometry3d current_b = poseFromRpy(2.0, 4.0, -50.0, -1.0, 0.7, 0.2);
  const bol::PlanarRelativeMotion motion_a = bol::ekfPlanarRelativeMotion(anchor_a, current_a);
  const bol::PlanarRelativeMotion motion_b = bol::ekfPlanarRelativeMotion(anchor_b, current_b);
  EXPECT_NEAR(motion_a.dx, motion_b.dx, 1.0e-12);
  EXPECT_NEAR(motion_a.dy, motion_b.dy, 1.0e-12);
  EXPECT_NEAR(motion_a.dyaw, motion_b.dyaw, 1.0e-12);
}

TEST(AcceptedPosePredictor, HoldsAcceptedZRollPitchAndAppliesPlanarDelta)
{
  const Eigen::Isometry3d accepted = poseFromRpy(10.0, 20.0, 1.7, 0.2, -0.15, M_PI_2);
  const Eigen::Isometry3d prediction = bol::predictMapLidarFromAcceptedPose(
    accepted, bol::PlanarRelativeMotion{1.0, 0.5, 0.3});
  const auto accepted_rpy = bol::rollPitchYaw(accepted.linear());
  const auto predicted_rpy = bol::rollPitchYaw(prediction.linear());

  EXPECT_NEAR(prediction.translation().x(), 9.5, 1.0e-12);
  EXPECT_NEAR(prediction.translation().y(), 21.0, 1.0e-12);
  EXPECT_NEAR(prediction.translation().z(), 1.7, 1.0e-12);
  EXPECT_NEAR(predicted_rpy[0], accepted_rpy[0], 1.0e-12);
  EXPECT_NEAR(predicted_rpy[1], accepted_rpy[1], 1.0e-12);
  EXPECT_NEAR(predicted_rpy[2], bol::wrapAngle(M_PI_2 + 0.3), 1.0e-12);
}

TEST(AcceptedPosePredictor, OutputIsFiniteNormalizedAndPositiveYawIsCounterClockwise)
{
  const Eigen::Isometry3d prediction = bol::predictMapLidarFromAcceptedPose(
    Eigen::Isometry3d::Identity(), bol::PlanarRelativeMotion{1.0, 0.0, M_PI_2});
  const Eigen::Quaterniond quaternion(prediction.linear());
  EXPECT_TRUE(bol::isFiniteTransform(prediction));
  EXPECT_NEAR(quaternion.norm(), 1.0, 1.0e-12);
  EXPECT_NEAR(bol::rollPitchYaw(prediction.linear())[2], M_PI_2, 1.0e-12);
  EXPECT_NEAR(prediction.translation().x(), 1.0, 1.0e-12);
  EXPECT_NEAR(prediction.translation().y(), 0.0, 1.0e-12);
}

TEST(AcceptedPosePredictor, SmallNegativeYawDoesNotSelectEquivalentPiBranch)
{
  const Eigen::Isometry3d accepted = poseFromRpy(0.0, 0.0, 0.2, -0.003, 0.016, 0.0002);
  const Eigen::Isometry3d prediction = bol::predictMapLidarFromAcceptedPose(
    accepted, bol::PlanarRelativeMotion{0.01, 0.0, -0.0005});
  const auto rpy = bol::rollPitchYaw(prediction.linear());
  EXPECT_NEAR(rpy[0], -0.003, 1.0e-12);
  EXPECT_NEAR(rpy[1], 0.016, 1.0e-12);
  EXPECT_NEAR(rpy[2], -0.0003, 1.0e-12);
  EXPECT_LT(bol::rotationAngle(accepted.linear().transpose() * prediction.linear()), 0.001);
}

TEST(MapBaseConversion, UsesTMapLidarTimesTLidarBaseDirection)
{
  const Eigen::Isometry3d T_map_lidar = poseFromRpy(10.0, 2.0, 1.0, 0.1, -0.2, 0.4);
  const Eigen::Isometry3d T_base_lidar = poseFromRpy(0.8, -0.1, 0.3, 0.0, 0.0, 0.05);
  const Eigen::Isometry3d T_map_base = bol::mapBaseFromMapLidar(
    T_map_lidar, T_base_lidar);
  const Eigen::Isometry3d expected = T_map_lidar * T_base_lidar.inverse();

  EXPECT_TRUE(T_map_base.matrix().isApprox(expected.matrix(), 1.0e-12));
  EXPECT_TRUE((T_map_base * T_base_lidar).matrix().isApprox(
      T_map_lidar.matrix(), 1.0e-12));
}
