#pragma once

#include <Eigen/Geometry>

#include <cstddef>
#include <istream>
#include <optional>
#include <string>
#include <vector>

namespace bunker_offline_localization {

struct TimedPose {
  double timestamp{};
  Eigen::Isometry3d T_map_lidar{Eigen::Isometry3d::Identity()};
};

struct PoseAssociation {
  const TimedPose* pose{};
  double absolute_time_difference{};
};

class ReferenceTrajectory {
public:
  static ReferenceTrajectory load(const std::string& path);
  static ReferenceTrajectory load(std::istream& stream, const std::string& source_name);

  const TimedPose& first() const;
  const std::vector<TimedPose>& poses() const { return poses_; }
  std::optional<PoseAssociation> associateNearest(double timestamp, double tolerance) const;

private:
  std::vector<TimedPose> poses_;
};

}  // namespace bunker_offline_localization
