#include "bunker_offline_localization/registration.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>

#include <chrono>
#include <stdexcept>

namespace bunker_offline_localization {

struct MapRegistrar::Impl {
  using PointCloud = small_gicp::PointCloud;
  using Tree = small_gicp::KdTree<PointCloud>;

  Impl(const PointCloud::Ptr& raw_map, RegistrationSettings requested_settings)
  : settings(std::move(requested_settings))
  {
    if (!raw_map || raw_map->size() < 10U) {
      throw std::invalid_argument("Global target map must contain at least 10 points");
    }
    if (settings.registration_type != "GICP") {
      throw std::invalid_argument(
              "registration_type must be GICP for this Phase 1 implementation");
    }
    if (settings.num_threads < 1 || settings.num_neighbors < 5 ||
      settings.map_voxel_resolution <= 0.0 || settings.scan_voxel_resolution <= 0.0 ||
      settings.max_correspondence_distance <= 0.0 || settings.max_iterations < 1)
    {
      throw std::invalid_argument("Invalid small_gicp registration setting");
    }

    // Target preprocessing is intentionally performed exactly once and reused for every scan.
    target = small_gicp::voxelgrid_sampling_omp(
      *raw_map, settings.map_voxel_resolution, settings.num_threads);
    target_tree = std::make_shared<Tree>(
      target, small_gicp::KdTreeBuilderOMP(settings.num_threads));
    small_gicp::estimate_covariances_omp(
      *target, *target_tree, settings.num_neighbors, settings.num_threads);
  }

  RegistrationSettings settings;
  PointCloud::Ptr target;
  Tree::Ptr target_tree;
};

MapRegistrar::MapRegistrar(
  const small_gicp::PointCloud::Ptr& raw_map,
  const RegistrationSettings& settings)
: impl_(std::make_unique<Impl>(raw_map, settings))
{
}

MapRegistrar::~MapRegistrar() = default;
MapRegistrar::MapRegistrar(MapRegistrar&&) noexcept = default;
MapRegistrar& MapRegistrar::operator=(MapRegistrar&&) noexcept = default;

RegistrationOutput MapRegistrar::align(
  const small_gicp::PointCloud& raw_source,
  const Eigen::Isometry3d& init_T_map_lidar) const
{
  if (raw_source.size() < 10U) {
    throw std::invalid_argument("Source scan contains fewer than 10 finite points");
  }
  if (!isFiniteTransform(init_T_map_lidar)) {
    throw std::invalid_argument("init_T_map_lidar is not a finite rigid transform");
  }

  const auto start = std::chrono::steady_clock::now();
  auto source = small_gicp::voxelgrid_sampling_omp(
    raw_source, impl_->settings.scan_voxel_resolution, impl_->settings.num_threads);
  if (source->size() < static_cast<std::size_t>(impl_->settings.num_neighbors)) {
    throw std::runtime_error("Downsampled source has too few points for covariance estimation");
  }
  auto source_tree = std::make_shared<Impl::Tree>(
    source, small_gicp::KdTreeBuilderOMP(impl_->settings.num_threads));
  small_gicp::estimate_covariances_omp(
    *source, *source_tree, impl_->settings.num_neighbors, impl_->settings.num_threads);

  small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP>
  registration;
  registration.reduction.num_threads = impl_->settings.num_threads;
  registration.rejector.max_dist_sq =
    impl_->settings.max_correspondence_distance *
    impl_->settings.max_correspondence_distance;
  registration.optimizer.max_iterations = impl_->settings.max_iterations;

  // Official API direction: target=global map, source=current LiDAR scan.
  // Therefore result.T_target_source is T_map_lidar and p_map=T_map_lidar*p_lidar.
  const small_gicp::RegistrationResult result = registration.align(
    *impl_->target, *source, *impl_->target_tree, init_T_map_lidar);
  const auto stop = std::chrono::steady_clock::now();

  RegistrationOutput output;
  output.T_map_lidar = result.T_target_source;
  output.converged = result.converged;
  output.iterations = result.iterations;
  output.num_inliers = result.num_inliers;
  output.final_error = result.error;
  output.hessian = result.H;
  output.runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  output.source_downsampled_points = source->size();
  return output;
}

std::size_t MapRegistrar::targetPointCount() const
{
  return impl_->target->size();
}

const RegistrationSettings& MapRegistrar::settings() const
{
  return impl_->settings;
}

}  // namespace bunker_offline_localization
