#pragma once

#include <Eigen/Geometry>

#include <cstddef>
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

}  // namespace bunker_offline_localization
