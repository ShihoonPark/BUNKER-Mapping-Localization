#include "bunker_offline_localization/map_loader.hpp"

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace bunker_offline_localization {

LoadedMap loadPlyMap(const std::string& path)
{
  pcl::PointCloud<pcl::PointXYZI> pcl_points;
  if (pcl::io::loadPLYFile(path, pcl_points) < 0) {
    throw std::runtime_error("PCL failed to load PLY map: " + path);
  }

  std::vector<Eigen::Vector4f> points;
  points.reserve(pcl_points.size());
  for (const auto& point : pcl_points) {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
      points.emplace_back(point.x, point.y, point.z, 1.0F);
    }
  }
  if (points.size() < 10U) {
    throw std::runtime_error("PLY map contains fewer than 10 finite XYZ points: " + path);
  }
  return LoadedMap{std::make_shared<small_gicp::PointCloud>(points), pcl_points.size()};
}

}  // namespace bunker_offline_localization
