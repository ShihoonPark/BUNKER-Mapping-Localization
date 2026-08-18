#include "bunker_offline_localization/diagnostic_utils.hpp"
#include "bunker_offline_localization/registration.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace bol = bunker_offline_localization;

namespace {

std::vector<Eigen::Vector4d> structuredPoints()
{
  std::vector<Eigen::Vector4d> points;
  for (int x = -5; x <= 5; ++x) {
    for (int y = -5; y <= 5; ++y) {
      for (int layer = 0; layer < 3; ++layer) {
        const double px = 0.17 * x + 0.013 * layer;
        const double py = 0.19 * y - 0.007 * layer;
        const double pz = 0.04 * x - 0.025 * y + 0.11 * layer +
          0.009 * x * y;
        points.emplace_back(px, py, pz, 1.0);
      }
    }
  }
  return points;
}

std::pair<bol::MapRegistrar, bol::RegistrationOutput> registeredSyntheticCloud()
{
  const auto target_points = structuredPoints();
  Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
  expected.linear() = Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  expected.translation() = Eigen::Vector3d(0.08, -0.05, 0.03);
  std::vector<Eigen::Vector4d> source_points;
  source_points.reserve(target_points.size());
  for (const auto& target : target_points) {
    source_points.push_back(expected.inverse() * target);
  }
  bol::RegistrationSettings settings;
  settings.num_threads = 1;
  settings.map_voxel_resolution = 0.04;
  settings.scan_voxel_resolution = 0.04;
  settings.num_neighbors = 10;
  settings.max_correspondence_distance = 0.25;
  settings.max_iterations = 20;
  bol::MapRegistrar registrar(
    std::make_shared<small_gicp::PointCloud>(target_points), settings);
  auto result = registrar.align(small_gicp::PointCloud(source_points), expected);
  return {std::move(registrar), std::move(result)};
}

}  // namespace

TEST(GicpDiagnosticTransform, AppliesTMapLidarWithoutInversion)
{
  Eigen::Isometry3d T_map_lidar = Eigen::Isometry3d::Identity();
  T_map_lidar.linear() =
    Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_map_lidar.translation() = Eigen::Vector3d(3.0, 4.0, 0.5);
  const auto transformed = bol::transformPoints({Eigen::Vector3d(1.0, 0.0, 0.2)}, T_map_lidar);
  ASSERT_EQ(transformed.size(), 1U);
  EXPECT_TRUE(transformed.front().isApprox(Eigen::Vector3d(3.0, 5.0, 0.7), 1.0e-12));
}

TEST(GicpDiagnosticCorrespondence, UsesNearestTargetAndConfiguredMaximumDistance)
{
  auto [registrar, registration] = registeredSyntheticCloud();
  const auto reconstructed = registrar.reconstructFinalCorrespondences(registration);
  ASSERT_FALSE(reconstructed.valid.empty());
  EXPECT_LE(reconstructed.valid.size(), reconstructed.candidate_count);
  for (const auto& item : reconstructed.valid) {
    EXPECT_LE(item.distance_m, registrar.settings().max_correspondence_distance + 1.0e-12);
    double brute_force_minimum = std::numeric_limits<double>::infinity();
    for (const auto& target : registrar.targetCloud().points) {
      brute_force_minimum = std::min(
        brute_force_minimum,
        (target.head<3>() - item.registered_source_map).norm());
    }
    EXPECT_NEAR(item.distance_m, brute_force_minimum, 1.0e-9);
  }
}

TEST(GicpDiagnosticCorrespondence, HonestMethodLabelNeverClaimsExactInternal)
{
  const bol::CorrespondenceReconstruction reconstruction;
  EXPECT_EQ(
    reconstruction.method,
    "posthoc_final_transform_correspondence_reconstruction");
  EXPECT_EQ(reconstruction.method.find("exact_internal"), std::string::npos);
}

TEST(GicpDiagnosticSubsampling, IsDeterministicAndBounded)
{
  const auto first = bol::deterministicSubsampleIndices(1003U, 200U);
  const auto second = bol::deterministicSubsampleIndices(1003U, 200U);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.size(), 200U);
  EXPECT_TRUE(std::is_sorted(first.begin(), first.end()));
  EXPECT_LT(first.back(), 1003U);
}

TEST(GicpDiagnosticSubsampling, DoesNotChangeFullPopulationStatistics)
{
  std::vector<double> full(1003U);
  std::iota(full.begin(), full.end(), 0.0);
  const double mean_before = std::accumulate(full.begin(), full.end(), 0.0) / full.size();
  const auto ignored_visual_indices = bol::deterministicSubsampleIndices(full.size(), 200U);
  EXPECT_EQ(ignored_visual_indices.size(), 200U);
  const double mean_after = std::accumulate(full.begin(), full.end(), 0.0) / full.size();
  EXPECT_DOUBLE_EQ(mean_before, mean_after);
}

TEST(GicpDiagnosticContinuousState, ProcessesFullSequenceWithoutCandidateList)
{
  bol::ContinuousScanState state;
  for (int index = 0; index < 490; ++index) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = 0.01 * index;
    state.record(index % 100 != 0, pose);
  }
  EXPECT_EQ(state.processedCount(), 490U);
  EXPECT_EQ(state.candidateCountUsedForExecution(), 0U);
  EXPECT_EQ(state.acceptedCount() + state.rejectedCount(), 490U);
}

TEST(GicpDiagnosticContinuousState, RejectedPoseNeverEntersAcceptedPath)
{
  bol::ContinuousScanState state;
  state.record(true, Eigen::Isometry3d::Identity());
  Eigen::Isometry3d rejected = Eigen::Isometry3d::Identity();
  rejected.translation().z() = 999.0;
  state.record(false, rejected);
  EXPECT_EQ(state.processedCount(), 2U);
  EXPECT_EQ(state.acceptedPathSize(), 1U);
  EXPECT_DOUBLE_EQ(state.acceptedPoses().front().translation().z(), 0.0);
}

TEST(GicpDiagnosticLatency, SteadyClockDurationIsFiniteAndNonnegative)
{
  const auto start = std::chrono::steady_clock::now();
  double accumulator = 0.0;
  for (int index = 1; index < 1000; ++index) {
    accumulator += std::sqrt(static_cast<double>(index));
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  EXPECT_GT(accumulator, 0.0);
  EXPECT_TRUE(bol::isValidLatencyMilliseconds(elapsed_ms));
  EXPECT_FALSE(bol::isValidLatencyMilliseconds(-0.001));
  EXPECT_FALSE(bol::isValidLatencyMilliseconds(std::numeric_limits<double>::quiet_NaN()));
}
