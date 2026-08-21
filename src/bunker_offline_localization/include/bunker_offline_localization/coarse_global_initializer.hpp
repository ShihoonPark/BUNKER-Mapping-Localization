#pragma once

#include "bunker_offline_localization/registration.hpp"

#include <small_gicp/points/point_cloud.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <memory>
#include <vector>

namespace bunker_offline_localization {

struct CoarseGlobalInitializationSettings {
  int num_threads{4};
  double coarse_voxel_resolution{0.60};
  double xy_step{0.75};
  double z_step{0.25};
  double yaw_step{0.2617993877991494};
  double coarse_max_correspondence_distance{0.90};
  std::size_t maximum_scored_source_points{2000};
  std::size_t broad_candidates_to_keep{256};
  std::size_t refinement_candidates{48};
  std::size_t reported_candidates{10};
  RegistrationSettings refinement;
};

struct CoarseGlobalCandidate {
  Eigen::Isometry3d coarse_T_map_lidar{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d T_map_lidar{Eigen::Isometry3d::Identity()};
  double coarse_score{};
  std::size_t coarse_inliers{};
  double coarse_inlier_rmse{};
  bool converged{};
  std::size_t iterations{};
  std::size_t num_inliers{};
  double final_error{};
  double final_error_per_inlier{};
  double refined_score{};
  double runtime_ms{};
  std::size_t source_downsampled_points{};
};

struct CoarseGlobalInitializationResult {
  bool success{};
  Eigen::Vector3d map_min{Eigen::Vector3d::Zero()};
  Eigen::Vector3d map_max{Eigen::Vector3d::Zero()};
  std::size_t map_coarse_points{};
  std::size_t source_input_points{};
  std::size_t source_coarse_points{};
  std::size_t broad_candidates_evaluated{};
  std::size_t refinement_candidates_evaluated{};
  std::size_t refinement_candidates_succeeded{};
  double total_runtime_ms{};
  double top_score_gap{};
  double top_score_ratio{};
  std::vector<CoarseGlobalCandidate> candidates;
};

class CoarseGlobalInitializer {
public:
  CoarseGlobalInitializer(
    const small_gicp::PointCloud::Ptr& raw_map,
    const CoarseGlobalInitializationSettings& settings);
  ~CoarseGlobalInitializer();
  CoarseGlobalInitializer(CoarseGlobalInitializer&&) noexcept;
  CoarseGlobalInitializer& operator=(CoarseGlobalInitializer&&) noexcept;
  CoarseGlobalInitializer(const CoarseGlobalInitializer&) = delete;
  CoarseGlobalInitializer& operator=(const CoarseGlobalInitializer&) = delete;

  CoarseGlobalInitializationResult initialize(
    const small_gicp::PointCloud& raw_source) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bunker_offline_localization
