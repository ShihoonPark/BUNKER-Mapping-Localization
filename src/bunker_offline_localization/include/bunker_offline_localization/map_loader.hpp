#pragma once

#include <small_gicp/points/point_cloud.hpp>

#include <cstddef>
#include <string>

namespace bunker_offline_localization {

struct LoadedMap {
  small_gicp::PointCloud::Ptr points;
  std::size_t raw_point_count{};
};

LoadedMap loadPlyMap(const std::string& path);

}  // namespace bunker_offline_localization
