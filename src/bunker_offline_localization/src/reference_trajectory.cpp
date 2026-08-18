#include "bunker_offline_localization/reference_trajectory.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bunker_offline_localization {

ReferenceTrajectory ReferenceTrajectory::load(const std::string& path)
{
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open reference trajectory: " + path);
  }
  return load(stream, path);
}

ReferenceTrajectory ReferenceTrajectory::load(
  std::istream& stream, const std::string& source_name)
{
  ReferenceTrajectory trajectory;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    const auto first_non_space = line.find_first_not_of(" \t\r");
    if (first_non_space == std::string::npos || line[first_non_space] == '#') {
      continue;
    }

    double timestamp, x, y, z, qx, qy, qz, qw;
    std::istringstream values(line);
    if (!(values >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      throw std::runtime_error(
              source_name + ":" + std::to_string(line_number) +
              ": expected TUM fields: timestamp x y z qx qy qz qw");
    }
    std::string trailing;
    if (values >> trailing) {
      throw std::runtime_error(
              source_name + ":" + std::to_string(line_number) + ": unexpected trailing field");
    }
    if (!std::isfinite(timestamp)) {
      throw std::runtime_error(
              source_name + ":" + std::to_string(line_number) + ": non-finite timestamp");
    }
    trajectory.poses_.push_back(
      TimedPose{timestamp, makeTransform(x, y, z, qx, qy, qz, qw)});
  }

  if (trajectory.poses_.empty()) {
    throw std::runtime_error("Reference trajectory has no valid poses: " + source_name);
  }
  if (!std::is_sorted(
        trajectory.poses_.begin(), trajectory.poses_.end(),
        [](const TimedPose& lhs, const TimedPose& rhs) {
          return lhs.timestamp < rhs.timestamp;
        }))
  {
    throw std::runtime_error("Reference trajectory timestamps are not sorted: " + source_name);
  }
  return trajectory;
}

const TimedPose& ReferenceTrajectory::first() const
{
  if (poses_.empty()) {
    throw std::logic_error("Reference trajectory is empty");
  }
  return poses_.front();
}

std::optional<PoseAssociation> ReferenceTrajectory::associateNearest(
  const double timestamp, const double tolerance) const
{
  if (!std::isfinite(timestamp) || !std::isfinite(tolerance) || tolerance < 0.0 || poses_.empty()) {
    return std::nullopt;
  }
  const auto upper = std::lower_bound(
    poses_.begin(), poses_.end(), timestamp,
    [](const TimedPose& pose, const double time) {return pose.timestamp < time;});

  const TimedPose* nearest = nullptr;
  if (upper != poses_.end()) {
    nearest = &*upper;
  }
  if (upper != poses_.begin()) {
    const TimedPose* lower = &*std::prev(upper);
    if (nearest == nullptr ||
      std::abs(lower->timestamp - timestamp) <= std::abs(nearest->timestamp - timestamp))
    {
      nearest = lower;
    }
  }
  const double difference = std::abs(nearest->timestamp - timestamp);
  if (difference > tolerance) {
    return std::nullopt;
  }
  return PoseAssociation{nearest, difference};
}

}  // namespace bunker_offline_localization
