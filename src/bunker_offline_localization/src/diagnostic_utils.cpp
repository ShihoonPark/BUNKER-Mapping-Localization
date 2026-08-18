#include "bunker_offline_localization/diagnostic_utils.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <cmath>
#include <stdexcept>

namespace bunker_offline_localization {

std::vector<std::size_t> deterministicSubsampleIndices(
  const std::size_t input_count, const std::size_t maximum_count)
{
  if (input_count == 0U || maximum_count == 0U) {
    return {};
  }
  if (input_count <= maximum_count) {
    std::vector<std::size_t> indices(input_count);
    for (std::size_t index = 0; index < input_count; ++index) {
      indices[index] = index;
    }
    return indices;
  }
  std::vector<std::size_t> indices;
  indices.reserve(maximum_count);
  for (std::size_t output_index = 0; output_index < maximum_count; ++output_index) {
    indices.push_back(output_index * input_count / maximum_count);
  }
  return indices;
}

std::vector<Eigen::Vector3d> transformPoints(
  const std::vector<Eigen::Vector3d>& points,
  const Eigen::Isometry3d& T_map_lidar)
{
  if (!isFiniteTransform(T_map_lidar)) {
    throw std::invalid_argument("T_map_lidar must be a finite rigid transform");
  }
  std::vector<Eigen::Vector3d> transformed;
  transformed.reserve(points.size());
  for (const auto& point : points) {
    transformed.push_back(T_map_lidar * point);
  }
  return transformed;
}

bool isValidLatencyMilliseconds(const double value)
{
  return std::isfinite(value) && value >= 0.0;
}

void ContinuousScanState::record(
  const bool accepted, const Eigen::Isometry3d& T_map_lidar)
{
  ++processed_count_;
  if (accepted) {
    if (!isFiniteTransform(T_map_lidar)) {
      throw std::invalid_argument("Accepted path pose must be finite");
    }
    accepted_poses_.push_back(T_map_lidar);
  } else {
    ++rejected_count_;
  }
}

}  // namespace bunker_offline_localization
