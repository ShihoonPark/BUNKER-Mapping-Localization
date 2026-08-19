#include "bunker_offline_localization/coarse_global_initializer.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <random>
#include <vector>

namespace bol = bunker_offline_localization;

TEST(CoarseGlobalInitializer, FindsGlobalPoseAndRanksAllRefinedCandidates)
{
  std::mt19937 generator(20260819);
  std::uniform_real_distribution<double> x_distribution(-3.0, 3.0);
  std::uniform_real_distribution<double> y_distribution(-2.0, 3.5);
  std::normal_distribution<double> noise(0.0, 0.005);
  std::vector<Eigen::Vector4d> map_points;
  map_points.reserve(3000);
  for (int index = 0; index < 3000; ++index) {
    const double x = x_distribution(generator);
    const double y = y_distribution(generator);
    const double z = 0.08 * x - 0.04 * y + 0.25 * std::sin(0.7 * x) +
      0.12 * std::cos(1.3 * y) + noise(generator);
    map_points.emplace_back(x, y, z, 1.0);
  }

  Eigen::Isometry3d expected = Eigen::Isometry3d::Identity();
  expected.linear() =
    (Eigen::AngleAxisd(1.41, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.025, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.018, Eigen::Vector3d::UnitX())).toRotationMatrix();
  expected.translation() = Eigen::Vector3d(1.15, -0.45, 0.22);
  std::vector<Eigen::Vector4d> source_points;
  source_points.reserve(map_points.size());
  for (const auto& point : map_points) {
    source_points.push_back(expected.inverse() * point);
  }

  bol::CoarseGlobalInitializationSettings settings;
  settings.num_threads = 2;
  settings.coarse_voxel_resolution = 0.35;
  settings.xy_step = 0.60;
  settings.z_step = 0.25;
  settings.yaw_step = M_PI / 12.0;
  settings.coarse_max_correspondence_distance = 0.75;
  settings.maximum_scored_source_points = 1000;
  settings.broad_candidates_to_keep = 128;
  settings.refinement_candidates = 24;
  settings.reported_candidates = 5;
  settings.refinement.num_threads = 2;
  settings.refinement.map_voxel_resolution = 0.12;
  settings.refinement.scan_voxel_resolution = 0.12;
  settings.refinement.num_neighbors = 20;
  settings.refinement.max_correspondence_distance = 1.0;
  settings.refinement.max_iterations = 40;

  bol::CoarseGlobalInitializer initializer(
    std::make_shared<small_gicp::PointCloud>(map_points), settings);
  const auto result = initializer.initialize(small_gicp::PointCloud(source_points));

  ASSERT_TRUE(result.success);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_GT(result.broad_candidates_evaluated, result.refinement_candidates_evaluated);
  EXPECT_GE(result.refinement_candidates_evaluated, result.candidates.size());
  EXPECT_GE(result.refinement_candidates_evaluated, result.refinement_candidates_succeeded);
  const auto& best = result.candidates.front();
  EXPECT_TRUE(best.converged);
  EXPECT_GT(best.num_inliers, 1000U);
  EXPECT_LT((best.T_map_lidar.translation() - expected.translation()).norm(), 0.08);
  EXPECT_LT(
    bol::rotationAngle(expected.linear().transpose() * best.T_map_lidar.linear()), 0.04);
  EXPECT_TRUE(std::isfinite(best.refined_score));
}

TEST(CoarseGlobalInitializer, RejectsInvalidSearchConfiguration)
{
  std::vector<Eigen::Vector4d> points;
  for (int index = 0; index < 50; ++index) {
    points.emplace_back(0.1 * index, 0.2 * index, 0.01 * index, 1.0);
  }
  bol::CoarseGlobalInitializationSettings settings;
  settings.xy_step = 0.0;
  EXPECT_THROW(
    bol::CoarseGlobalInitializer(
      std::make_shared<small_gicp::PointCloud>(points), settings),
    std::invalid_argument);
}
