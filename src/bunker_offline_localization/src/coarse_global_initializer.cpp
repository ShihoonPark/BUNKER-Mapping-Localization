#include "bunker_offline_localization/coarse_global_initializer.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/ann/traits.hpp>
#include <small_gicp/util/downsampling_omp.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bunker_offline_localization {
namespace {

struct BroadCandidate {
  Eigen::Isometry3d T_map_lidar{Eigen::Isometry3d::Identity()};
  double score{-std::numeric_limits<double>::infinity()};
  std::size_t inliers{};
  double inlier_rmse{std::numeric_limits<double>::infinity()};
};

std::vector<double> inclusiveGrid(
  const double minimum, const double maximum, const double requested_step)
{
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum ||
    !std::isfinite(requested_step) || requested_step <= 0.0)
  {
    throw std::invalid_argument("Invalid coarse search grid bounds");
  }
  const std::size_t intervals = std::max<std::size_t>(
    1U, static_cast<std::size_t>(std::ceil((maximum - minimum) / requested_step)));
  std::vector<double> values;
  values.reserve(intervals + 1U);
  for (std::size_t index = 0; index <= intervals; ++index) {
    values.push_back(minimum + (maximum - minimum) * index / intervals);
  }
  return values;
}

double planarYawDistance(
  const Eigen::Isometry3d& lhs, const Eigen::Isometry3d& rhs)
{
  return std::abs(wrapAngle(
    rollPitchYaw(lhs.linear())[2] - rollPitchYaw(rhs.linear())[2]));
}

bool sameBroadBasin(
  const BroadCandidate& lhs, const BroadCandidate& rhs,
  const CoarseGlobalInitializationSettings& settings)
{
  const double planar_distance =
    (lhs.T_map_lidar.translation().head<2>() -
    rhs.T_map_lidar.translation().head<2>()).norm();
  const double z_distance = std::abs(
    lhs.T_map_lidar.translation().z() - rhs.T_map_lidar.translation().z());
  return planar_distance < 1.5 * settings.xy_step &&
         z_distance < 1.5 * settings.z_step &&
         planarYawDistance(lhs.T_map_lidar, rhs.T_map_lidar) < 1.5 * settings.yaw_step;
}

bool sameRefinedMode(
  const CoarseGlobalCandidate& lhs, const CoarseGlobalCandidate& rhs)
{
  const double translation_distance =
    (lhs.T_map_lidar.translation() - rhs.T_map_lidar.translation()).norm();
  const double rotation_distance = rotationAngle(
    lhs.T_map_lidar.linear().transpose() * rhs.T_map_lidar.linear());
  return translation_distance < 0.35 && rotation_distance < 0.1308996938995747;
}

}  // namespace

struct CoarseGlobalInitializer::Impl {
  using PointCloud = small_gicp::PointCloud;
  using Tree = small_gicp::KdTree<PointCloud>;

  Impl(
    const PointCloud::Ptr& raw_map,
    CoarseGlobalInitializationSettings requested_settings)
  : settings(std::move(requested_settings)), refinement_registrar(raw_map, settings.refinement)
  {
    if (!raw_map || raw_map->size() < 10U) {
      throw std::invalid_argument("Coarse initializer map has fewer than 10 points");
    }
    if (settings.num_threads < 1 || settings.coarse_voxel_resolution <= 0.0 ||
      settings.xy_step <= 0.0 || settings.z_step <= 0.0 || settings.yaw_step <= 0.0 ||
      settings.yaw_step > M_PI || settings.coarse_max_correspondence_distance <= 0.0 ||
      settings.maximum_scored_source_points < 10U ||
      settings.broad_candidates_to_keep < 1U || settings.refinement_candidates < 1U ||
      settings.reported_candidates < 1U)
    {
      throw std::invalid_argument("Invalid coarse global initialization setting");
    }

    map_min.setConstant(std::numeric_limits<double>::infinity());
    map_max.setConstant(-std::numeric_limits<double>::infinity());
    for (std::size_t index = 0; index < raw_map->size(); ++index) {
      const Eigen::Vector3d point = small_gicp::traits::point(*raw_map, index).head<3>();
      map_min = map_min.cwiseMin(point);
      map_max = map_max.cwiseMax(point);
    }
    coarse_target = small_gicp::voxelgrid_sampling_omp(
      *raw_map, settings.coarse_voxel_resolution, settings.num_threads);
    coarse_tree = std::make_shared<Tree>(
      coarse_target, small_gicp::KdTreeBuilderOMP(settings.num_threads));
  }

  BroadCandidate score(
    const std::vector<Eigen::Vector4d>& sampled_source,
    const Eigen::Isometry3d& candidate) const
  {
    const double maximum_distance_squared =
      settings.coarse_max_correspondence_distance *
      settings.coarse_max_correspondence_distance;
    double truncated_squared_error{};
    double inlier_squared_error{};
    std::size_t inliers{};
    for (const auto& source : sampled_source) {
      const Eigen::Vector4d transformed = candidate * source;
      std::size_t target_index{};
      double squared_distance{};
      const bool found = small_gicp::traits::nearest_neighbor_search(
        *coarse_tree, transformed, &target_index, &squared_distance);
      if (!found || !std::isfinite(squared_distance)) {
        truncated_squared_error += maximum_distance_squared;
        continue;
      }
      truncated_squared_error += std::min(squared_distance, maximum_distance_squared);
      if (squared_distance <= maximum_distance_squared) {
        ++inliers;
        inlier_squared_error += squared_distance;
      }
    }
    const double count = static_cast<double>(sampled_source.size());
    const double inlier_fraction = static_cast<double>(inliers) / count;
    const double normalized_truncated_error =
      truncated_squared_error / (count * maximum_distance_squared);
    BroadCandidate output;
    output.T_map_lidar = candidate;
    output.score = inlier_fraction - 0.25 * normalized_truncated_error;
    output.inliers = inliers;
    output.inlier_rmse = inliers > 0U ?
      std::sqrt(inlier_squared_error / static_cast<double>(inliers)) :
      std::numeric_limits<double>::infinity();
    return output;
  }

  CoarseGlobalInitializationSettings settings;
  Eigen::Vector3d map_min{Eigen::Vector3d::Zero()};
  Eigen::Vector3d map_max{Eigen::Vector3d::Zero()};
  PointCloud::Ptr coarse_target;
  Tree::Ptr coarse_tree;
  MapRegistrar refinement_registrar;
};

CoarseGlobalInitializer::CoarseGlobalInitializer(
  const small_gicp::PointCloud::Ptr& raw_map,
  const CoarseGlobalInitializationSettings& settings)
: impl_(std::make_unique<Impl>(raw_map, settings))
{
}

CoarseGlobalInitializer::~CoarseGlobalInitializer() = default;
CoarseGlobalInitializer::CoarseGlobalInitializer(CoarseGlobalInitializer&&) noexcept = default;
CoarseGlobalInitializer& CoarseGlobalInitializer::operator=(
  CoarseGlobalInitializer&&) noexcept = default;

CoarseGlobalInitializationResult CoarseGlobalInitializer::initialize(
  const small_gicp::PointCloud& raw_source) const
{
  if (raw_source.size() < 10U) {
    throw std::invalid_argument("Coarse initializer source has fewer than 10 points");
  }
  const auto start = std::chrono::steady_clock::now();
  auto coarse_source = small_gicp::voxelgrid_sampling_omp(
    raw_source, impl_->settings.coarse_voxel_resolution, impl_->settings.num_threads);
  if (coarse_source->size() < 10U) {
    throw std::runtime_error("Coarse voxel source has fewer than 10 points");
  }

  const std::size_t sample_count = std::min(
    coarse_source->size(), impl_->settings.maximum_scored_source_points);
  std::vector<Eigen::Vector4d> sampled_source;
  sampled_source.reserve(sample_count);
  for (std::size_t index = 0; index < sample_count; ++index) {
    const std::size_t source_index = index * coarse_source->size() / sample_count;
    sampled_source.push_back(small_gicp::traits::point(*coarse_source, source_index));
  }

  const auto x_values = inclusiveGrid(
    impl_->map_min.x(), impl_->map_max.x(), impl_->settings.xy_step);
  const auto y_values = inclusiveGrid(
    impl_->map_min.y(), impl_->map_max.y(), impl_->settings.xy_step);
  const auto z_values = inclusiveGrid(
    impl_->map_min.z(), impl_->map_max.z(), impl_->settings.z_step);
  const std::size_t yaw_count = std::max<std::size_t>(
    1U, static_cast<std::size_t>(std::ceil(2.0 * M_PI / impl_->settings.yaw_step)));
  const std::size_t candidate_count =
    x_values.size() * y_values.size() * z_values.size() * yaw_count;
  std::vector<BroadCandidate> retained;
  retained.reserve(impl_->settings.broad_candidates_to_keep + 1U);

#pragma omp parallel for schedule(dynamic) num_threads(impl_->settings.num_threads)
  for (std::int64_t linear_index = 0;
    linear_index < static_cast<std::int64_t>(candidate_count); ++linear_index)
  {
    std::size_t remainder = static_cast<std::size_t>(linear_index);
    const std::size_t yaw_index = remainder % yaw_count;
    remainder /= yaw_count;
    const std::size_t z_index = remainder % z_values.size();
    remainder /= z_values.size();
    const std::size_t y_index = remainder % y_values.size();
    const std::size_t x_index = remainder / y_values.size();
    const double yaw = -M_PI + 2.0 * M_PI * yaw_index / yaw_count;
    Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
    candidate.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    candidate.translation() = Eigen::Vector3d(
      x_values[x_index], y_values[y_index], z_values[z_index]);
    BroadCandidate scored = impl_->score(sampled_source, candidate);
#pragma omp critical(coarse_candidate_retention)
    {
      if (retained.size() < impl_->settings.broad_candidates_to_keep) {
        retained.push_back(std::move(scored));
      } else {
        auto worst = std::min_element(
          retained.begin(), retained.end(),
          [](const BroadCandidate& lhs, const BroadCandidate& rhs) {
            return lhs.score < rhs.score;
          });
        if (scored.score > worst->score) {
          *worst = std::move(scored);
        }
      }
    }
  }
  std::sort(
    retained.begin(), retained.end(),
    [](const BroadCandidate& lhs, const BroadCandidate& rhs) {
      return lhs.score > rhs.score;
    });

  std::vector<BroadCandidate> refinement_seeds;
  refinement_seeds.reserve(impl_->settings.refinement_candidates);
  for (const auto& candidate : retained) {
    const bool duplicate = std::any_of(
      refinement_seeds.begin(), refinement_seeds.end(),
      [&candidate, this](const BroadCandidate& selected) {
        return sameBroadBasin(candidate, selected, impl_->settings);
      });
    if (!duplicate) {
      refinement_seeds.push_back(candidate);
      if (refinement_seeds.size() >= impl_->settings.refinement_candidates) {
        break;
      }
    }
  }

  std::vector<CoarseGlobalCandidate> refined;
  refined.reserve(refinement_seeds.size());
  for (const auto& seed : refinement_seeds) {
    try {
      const RegistrationOutput registration = impl_->refinement_registrar.align(
        raw_source, seed.T_map_lidar);
      CoarseGlobalCandidate candidate;
      candidate.coarse_T_map_lidar = seed.T_map_lidar;
      candidate.T_map_lidar = registration.T_map_lidar;
      candidate.coarse_score = seed.score;
      candidate.coarse_inliers = seed.inliers;
      candidate.coarse_inlier_rmse = seed.inlier_rmse;
      candidate.converged = registration.converged;
      candidate.iterations = registration.iterations;
      candidate.num_inliers = registration.num_inliers;
      candidate.final_error = registration.final_error;
      candidate.final_error_per_inlier = registration.num_inliers > 0U ?
        registration.final_error / static_cast<double>(registration.num_inliers) :
        std::numeric_limits<double>::infinity();
      candidate.runtime_ms = registration.runtime_ms;
      candidate.source_downsampled_points = registration.source_downsampled_points;
      const double inlier_fraction = registration.source_downsampled_points > 0U ?
        static_cast<double>(registration.num_inliers) /
        static_cast<double>(registration.source_downsampled_points) : 0.0;
      const double nonnegative_error = std::max(0.0, candidate.final_error_per_inlier);
      candidate.refined_score = inlier_fraction / (1.0 + nonnegative_error);
      if (!candidate.converged || !isFiniteTransform(candidate.T_map_lidar) ||
        !std::isfinite(candidate.refined_score))
      {
        candidate.refined_score *= 0.5;
      }
      refined.push_back(std::move(candidate));
    } catch (const std::exception&) {
      // A failed local basin remains absent; all other globally generated seeds are evaluated.
    }
  }
  std::sort(
    refined.begin(), refined.end(),
    [](const CoarseGlobalCandidate& lhs, const CoarseGlobalCandidate& rhs) {
      return lhs.refined_score > rhs.refined_score;
    });

  CoarseGlobalInitializationResult result;
  result.map_min = impl_->map_min;
  result.map_max = impl_->map_max;
  result.map_coarse_points = impl_->coarse_target->size();
  result.source_input_points = raw_source.size();
  result.source_coarse_points = coarse_source->size();
  result.broad_candidates_evaluated = candidate_count;
  result.refinement_candidates_evaluated = refinement_seeds.size();
  result.refinement_candidates_succeeded = refined.size();
  result.candidates.reserve(impl_->settings.reported_candidates);
  for (const auto& candidate : refined) {
    const bool duplicate = std::any_of(
      result.candidates.begin(), result.candidates.end(),
      [&candidate](const CoarseGlobalCandidate& selected) {
        return sameRefinedMode(candidate, selected);
      });
    if (!duplicate) {
      result.candidates.push_back(candidate);
      if (result.candidates.size() >= impl_->settings.reported_candidates) {
        break;
      }
    }
  }
  if (!result.candidates.empty()) {
    const auto& best = result.candidates.front();
    result.success = best.converged && isFiniteTransform(best.T_map_lidar) &&
      best.num_inliers >= 100U && std::isfinite(best.final_error_per_inlier);
  }
  if (result.candidates.size() > 1U) {
    result.top_score_gap =
      result.candidates[0].refined_score - result.candidates[1].refined_score;
    result.top_score_ratio = result.candidates[1].refined_score > 0.0 ?
      result.candidates[0].refined_score / result.candidates[1].refined_score :
      std::numeric_limits<double>::infinity();
  } else {
    result.top_score_gap = std::numeric_limits<double>::infinity();
    result.top_score_ratio = std::numeric_limits<double>::infinity();
  }
  result.total_runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace bunker_offline_localization
