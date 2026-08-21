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

FullBagRunState::FullBagRunState(
  const double bag_origin_timestamp, const double timing_gap_threshold_sec)
: bag_origin_timestamp_(bag_origin_timestamp),
  timing_gap_threshold_sec_(timing_gap_threshold_sec)
{
  if (!std::isfinite(bag_origin_timestamp_) || bag_origin_timestamp_ <= 0.0 ||
    !std::isfinite(timing_gap_threshold_sec_) || timing_gap_threshold_sec_ <= 0.0)
  {
    throw std::invalid_argument("Invalid full-bag diagnostic timestamps");
  }
}

void FullBagRunState::observeLidarTimestamp(const double timestamp)
{
  if (!std::isfinite(timestamp)) {
    throw std::invalid_argument("LiDAR timestamp must be finite");
  }
  if (last_lidar_timestamp_ && !firstTimingGapDetected() &&
    timestamp - *last_lidar_timestamp_ > timing_gap_threshold_sec_)
  {
    first_gap_previous_timestamp_ = last_lidar_timestamp_;
    first_gap_current_timestamp_ = timestamp;
  }
  last_lidar_timestamp_ = timestamp;
  ++lidar_timestamp_count_;
}

void FullBagRunState::recordProcessed(
  const double timestamp, const bool accepted, const RejectReason reject_reason)
{
  if (!std::isfinite(timestamp)) {
    throw std::invalid_argument("Processed timestamp must be finite");
  }
  if (accepted) {
    if (reject_reason != RejectReason::None) {
      throw std::invalid_argument("Accepted record cannot have a rejection reason");
    }
    final_accepted_timestamp_ = timestamp;
  } else {
    if (reject_reason == RejectReason::None) {
      throw std::invalid_argument("Rejected record must have a rejection reason");
    }
    ++reject_counts_[static_cast<std::size_t>(reject_reason)];
  }

  if (!first_gap_current_timestamp_ || timestamp < *first_gap_current_timestamp_) {
    return;
  }
  ++processed_after_first_gap_;
  if (accepted) {
    ++accepted_after_first_gap_;
    if (!first_accepted_after_gap_timestamp_) {
      first_accepted_after_gap_timestamp_ = timestamp;
    }
  } else {
    ++rejected_after_first_gap_;
    ++reject_counts_after_first_gap_[static_cast<std::size_t>(reject_reason)];
  }
}

std::optional<double> FullBagRunState::firstGapDurationSec() const
{
  if (!first_gap_previous_timestamp_ || !first_gap_current_timestamp_) {
    return std::nullopt;
  }
  return *first_gap_current_timestamp_ - *first_gap_previous_timestamp_;
}

std::size_t FullBagRunState::rejectCount(const RejectReason reason) const
{
  return reject_counts_.at(static_cast<std::size_t>(reason));
}

std::size_t FullBagRunState::rejectCountAfterFirstGap(const RejectReason reason) const
{
  return reject_counts_after_first_gap_.at(static_cast<std::size_t>(reason));
}

}  // namespace bunker_offline_localization
