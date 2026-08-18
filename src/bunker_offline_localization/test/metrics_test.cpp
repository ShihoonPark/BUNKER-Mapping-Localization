#include "bunker_offline_localization/metrics.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <cmath>
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

TEST(PredictionCorrection, ComputesInversePredictionTimesRegistration)
{
  const Eigen::Isometry3d prediction = bol::makeTransform(
    4.0, -2.0, 0.5, 0.0, 0.0, std::sin(0.3), std::cos(0.3));
  Eigen::Isometry3d expected_delta = Eigen::Isometry3d::Identity();
  expected_delta.translation() = Eigen::Vector3d(0.3, -0.4, 0.1);
  expected_delta.linear() =
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Isometry3d registration = prediction * expected_delta;

  const Eigen::Isometry3d actual_delta = bol::predictionToRegistrationDelta(
    prediction, registration);
  EXPECT_TRUE(actual_delta.matrix().isApprox(expected_delta.matrix(), 1.0e-12));
  EXPECT_NEAR(actual_delta.translation().norm(), std::sqrt(0.26), 1.0e-12);
  EXPECT_NEAR(bol::rotationAngle(actual_delta.linear()), 0.2, 1.0e-12);
  EXPECT_NEAR(bol::rollPitchYaw(actual_delta.linear())[2], 0.2, 1.0e-12);
}
