#include "bunker_offline_localization/registration.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <random>
#include <vector>

namespace bol = bunker_offline_localization;

TEST(SmallGicpRegistration, RecoversTargetMapFromSourceLidarDirection)
{
  std::mt19937 generator(42);
  std::uniform_real_distribution<double> distribution(-3.0, 3.0);
  std::vector<Eigen::Vector4d> target_points;
  target_points.reserve(2500);
  for (int index = 0; index < 2500; ++index) {
    const double x = distribution(generator);
    const double y = distribution(generator);
    const double z = 0.2 * x + 0.1 * y + 0.15 * std::sin(x * y) +
      0.05 * distribution(generator);
    target_points.emplace_back(x, y, z, 1.0);
  }

  Eigen::Isometry3d expected_T_target_source = Eigen::Isometry3d::Identity();
  expected_T_target_source.linear() =
    (Eigen::AngleAxisd(0.14, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.04, Eigen::Vector3d::UnitY())).toRotationMatrix();
  expected_T_target_source.translation() = Eigen::Vector3d(0.35, -0.22, 0.12);

  std::vector<Eigen::Vector4d> source_points;
  source_points.reserve(target_points.size());
  for (const auto& target_point : target_points) {
    source_points.push_back(expected_T_target_source.inverse() * target_point);
  }

  bol::RegistrationSettings settings;
  settings.num_threads = 2;
  settings.map_voxel_resolution = 0.12;
  settings.scan_voxel_resolution = 0.12;
  settings.num_neighbors = 20;
  settings.max_correspondence_distance = 0.8;
  settings.max_iterations = 40;
  bol::MapRegistrar registrar(
    std::make_shared<small_gicp::PointCloud>(target_points), settings);

  Eigen::Isometry3d initial = expected_T_target_source;
  initial.translation() += Eigen::Vector3d(0.08, -0.05, 0.03);
  initial.linear() = initial.linear() *
    Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const auto result = registrar.align(small_gicp::PointCloud(source_points), initial);

  ASSERT_TRUE(result.converged);
  ASSERT_GT(result.num_inliers, 1000U);
  EXPECT_LT(
    (result.T_map_lidar.translation() - expected_T_target_source.translation()).norm(),
    0.02);
  EXPECT_LT(
    bol::rotationAngle(
      expected_T_target_source.linear().transpose() * result.T_map_lidar.linear()),
    0.01);

  const Eigen::Vector3d p_lidar = source_points.front().head<3>();
  const Eigen::Vector3d p_target = result.T_map_lidar * p_lidar;
  EXPECT_LT((p_target - target_points.front().head<3>()).norm(), 0.04);
}
