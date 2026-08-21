#pragma once

#include "bunker_offline_localization/metrics.hpp"

#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace bunker_offline_localization {

std::vector<std::size_t> deterministicSubsampleIndices(
  std::size_t input_count, std::size_t maximum_count);

std::vector<Eigen::Vector3d> transformPoints(
  const std::vector<Eigen::Vector3d>& points,
  const Eigen::Isometry3d& T_map_lidar);

bool isValidLatencyMilliseconds(double value);

class ContinuousScanState {
public:
  void record(bool accepted, const Eigen::Isometry3d& T_map_lidar);

  std::size_t processedCount() const {return processed_count_;}
  std::size_t acceptedCount() const {return accepted_poses_.size();}
  std::size_t rejectedCount() const {return rejected_count_;}
  std::size_t acceptedPathSize() const {return accepted_poses_.size();}
  std::size_t candidateCountUsedForExecution() const {return 0U;}
  const std::vector<Eigen::Isometry3d>& acceptedPoses() const {return accepted_poses_;}

private:
  std::size_t processed_count_{};
  std::size_t rejected_count_{};
  std::vector<Eigen::Isometry3d> accepted_poses_;
};

class FullBagRunState {
public:
  FullBagRunState(double bag_origin_timestamp, double timing_gap_threshold_sec);

  void observeLidarTimestamp(double timestamp);
  void recordProcessed(double timestamp, bool accepted, RejectReason reject_reason);
  void markBagEofReceived() {bag_eof_received_ = true;}

  double bagOriginTimestamp() const {return bag_origin_timestamp_;}
  double timingGapThresholdSec() const {return timing_gap_threshold_sec_;}
  std::size_t lidarTimestampCount() const {return lidar_timestamp_count_;}
  std::optional<double> lastLidarTimestamp() const {return last_lidar_timestamp_;}
  bool firstTimingGapDetected() const {return first_gap_current_timestamp_.has_value();}
  std::optional<double> firstGapPreviousTimestamp() const {
    return first_gap_previous_timestamp_;
  }
  std::optional<double> firstGapCurrentTimestamp() const {return first_gap_current_timestamp_;}
  std::optional<double> firstGapDurationSec() const;
  std::size_t rejectCount(RejectReason reason) const;
  std::size_t processedAfterFirstGap() const {return processed_after_first_gap_;}
  std::size_t acceptedAfterFirstGap() const {return accepted_after_first_gap_;}
  std::size_t rejectedAfterFirstGap() const {return rejected_after_first_gap_;}
  std::size_t rejectCountAfterFirstGap(RejectReason reason) const;
  std::optional<double> firstAcceptedAfterGapTimestamp() const {
    return first_accepted_after_gap_timestamp_;
  }
  std::optional<double> finalAcceptedTimestamp() const {return final_accepted_timestamp_;}
  bool recoveredAfterFirstGap() const {return first_accepted_after_gap_timestamp_.has_value();}
  bool bagEofReceived() const {return bag_eof_received_;}

private:
  static constexpr std::size_t kRejectReasonCount =
    static_cast<std::size_t>(RejectReason::RegistrationException) + 1U;

  double bag_origin_timestamp_{};
  double timing_gap_threshold_sec_{};
  std::size_t lidar_timestamp_count_{};
  std::optional<double> last_lidar_timestamp_;
  std::optional<double> first_gap_previous_timestamp_;
  std::optional<double> first_gap_current_timestamp_;
  std::array<std::size_t, kRejectReasonCount> reject_counts_{};
  std::size_t processed_after_first_gap_{};
  std::size_t accepted_after_first_gap_{};
  std::size_t rejected_after_first_gap_{};
  std::array<std::size_t, kRejectReasonCount> reject_counts_after_first_gap_{};
  std::optional<double> first_accepted_after_gap_timestamp_;
  std::optional<double> final_accepted_timestamp_;
  bool bag_eof_received_{false};
};

}  // namespace bunker_offline_localization
