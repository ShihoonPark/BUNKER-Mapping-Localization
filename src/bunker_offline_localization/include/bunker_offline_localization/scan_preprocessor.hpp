#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <small_gicp/points/point_cloud.hpp>

#include <cstddef>

namespace bunker_offline_localization {

struct ExtractedScan {
  small_gicp::PointCloud::Ptr points;
  std::size_t input_point_count{};
  std::size_t finite_point_count{};
};

ExtractedScan extractFiniteXYZ(const sensor_msgs::msg::PointCloud2& message);

}  // namespace bunker_offline_localization
