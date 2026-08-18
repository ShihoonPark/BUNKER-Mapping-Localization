#pragma once

#include <small_gicp/points/point_cloud.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bunker_offline_localization {

struct RegistrationSettings {
  int num_threads{4};
  double map_voxel_resolution{0.20};
  double scan_voxel_resolution{0.20};
  int num_neighbors{20};
  double max_correspondence_distance{1.0};
  int max_iterations{30};
  std::string registration_type{"GICP"};
};

struct RegistrationOutput {
  Eigen::Isometry3d T_map_lidar{Eigen::Isometry3d::Identity()};
  bool converged{false};
  std::size_t iterations{0};
  std::size_t num_inliers{0};
  double final_error{0.0};
  Eigen::Matrix<double, 6, 6> hessian{Eigen::Matrix<double, 6, 6>::Zero()};
  double runtime_ms{0.0};
  std::size_t source_downsampled_points{0};
  small_gicp::PointCloud::Ptr preprocessed_source;
};

struct ReconstructedCorrespondence {
  std::size_t source_index{};
  std::size_t target_index{};
  Eigen::Vector3d source_lidar{Eigen::Vector3d::Zero()};
  Eigen::Vector3d registered_source_map{Eigen::Vector3d::Zero()};
  Eigen::Vector3d target_map{Eigen::Vector3d::Zero()};
  Eigen::Vector3d residual_map{Eigen::Vector3d::Zero()};
  double distance_m{};
  double mahalanobis_error_contribution{};
};

struct CorrespondenceReconstruction {
  std::string method{"posthoc_final_transform_correspondence_reconstruction"};
  std::size_t candidate_count{};
  std::vector<ReconstructedCorrespondence> valid;
};

class MapRegistrar {
public:
  MapRegistrar(
    const small_gicp::PointCloud::Ptr& raw_map,
    const RegistrationSettings& settings);
  ~MapRegistrar();
  MapRegistrar(MapRegistrar&&) noexcept;
  MapRegistrar& operator=(MapRegistrar&&) noexcept;
  MapRegistrar(const MapRegistrar&) = delete;
  MapRegistrar& operator=(const MapRegistrar&) = delete;

  RegistrationOutput align(
    const small_gicp::PointCloud& raw_source,
    const Eigen::Isometry3d& init_T_map_lidar) const;

  std::size_t targetPointCount() const;
  const small_gicp::PointCloud& targetCloud() const;
  CorrespondenceReconstruction reconstructFinalCorrespondences(
    const RegistrationOutput& registration) const;
  const RegistrationSettings& settings() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bunker_offline_localization
