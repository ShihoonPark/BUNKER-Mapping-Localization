#include "bunker_offline_localization/metrics.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace bol = bunker_offline_localization;

namespace {

bol::RegistrationOutput validRegistration()
{
  bol::RegistrationOutput output;
  output.converged = true;
  output.num_inliers = 200;
  output.final_error = 100.0;
  output.hessian.setIdentity();
  return output;
}

}  // namespace

TEST(QualityGate, AcceptsValidRegistration)
{
  EXPECT_EQ(
    bol::evaluateRegistration(
      validRegistration(), Eigen::Isometry3d::Identity(), bol::QualityGateSettings{}),
    bol::RejectReason::None);
}

TEST(QualityGate, RejectsNonfiniteTransform)
{
  auto output = validRegistration();
  output.T_map_lidar.translation().x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
    bol::evaluateRegistration(output, Eigen::Isometry3d::Identity(), {}),
    bol::RejectReason::NonfiniteTransform);
}

TEST(QualityGate, RejectsTranslationAndRotationCorrections)
{
  auto translated = validRegistration();
  translated.T_map_lidar.translation().x() = 2.0;
  EXPECT_EQ(
    bol::evaluateRegistration(translated, Eigen::Isometry3d::Identity(), {}),
    bol::RejectReason::TranslationJump);

  auto rotated = validRegistration();
  rotated.T_map_lidar.linear() =
    Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_EQ(
    bol::evaluateRegistration(rotated, Eigen::Isometry3d::Identity(), {}),
    bol::RejectReason::RotationJump);
}
