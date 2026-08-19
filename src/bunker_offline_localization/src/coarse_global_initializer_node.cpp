#include "bunker_offline_localization/coarse_global_initializer.hpp"
#include "bunker_offline_localization/map_loader.hpp"
#include "bunker_offline_localization/scan_preprocessor.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <small_gicp/ann/traits.hpp>

#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>

namespace bunker_offline_localization {
namespace {

double stampSeconds(const builtin_interfaces::msg::Time& stamp)
{
  return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

void writeTransformCsv(std::ostream& output, const Eigen::Isometry3d& transform)
{
  Eigen::Quaterniond quaternion(transform.linear());
  quaternion.normalize();
  const auto rpy = rollPitchYaw(transform.linear());
  output << transform.translation().x() << ','
         << transform.translation().y() << ','
         << transform.translation().z() << ','
         << quaternion.x() << ',' << quaternion.y() << ',' << quaternion.z() << ','
         << quaternion.w() << ',' << rpy[0] << ',' << rpy[1] << ',' << rpy[2];
}

void writeTransformJson(std::ostream& output, const Eigen::Isometry3d& transform)
{
  Eigen::Quaterniond quaternion(transform.linear());
  quaternion.normalize();
  const auto rpy = rollPitchYaw(transform.linear());
  output << "{\n"
         << "      \"translation\": [" << transform.translation().x() << ", "
         << transform.translation().y() << ", " << transform.translation().z() << "],\n"
         << "      \"rotation_xyzw\": [" << quaternion.x() << ", " << quaternion.y()
         << ", " << quaternion.z() << ", " << quaternion.w() << "],\n"
         << "      \"roll_pitch_yaw_rad\": [" << rpy[0] << ", " << rpy[1] << ", "
         << rpy[2] << "],\n"
         << "      \"roll_pitch_yaw_deg\": [" << rpy[0] * 180.0 / M_PI << ", "
         << rpy[1] * 180.0 / M_PI << ", " << rpy[2] * 180.0 / M_PI << "]\n"
         << "    }";
}

void saveRegisteredSource(
  const std::filesystem::path& path,
  const small_gicp::PointCloud& source,
  const Eigen::Isometry3d& T_map_lidar)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.reserve(source.size());
  for (std::size_t index = 0; index < source.size(); ++index) {
    const Eigen::Vector4d point =
      T_map_lidar * small_gicp::traits::point(source, index);
    cloud.emplace_back(
      static_cast<float>(point.x()),
      static_cast<float>(point.y()),
      static_cast<float>(point.z()));
  }
  if (pcl::io::savePLYFileBinary(path.string(), cloud) < 0) {
    throw std::runtime_error("Failed to save registered coarse-initializer scan");
  }
}

}  // namespace

class CoarseGlobalInitializerNode : public rclcpp::Node {
public:
  CoarseGlobalInitializerNode()
  : Node("coarse_global_initializer")
  {
    map_path_ = declare_parameter<std::string>(
      "map_path",
      "/home/a/Desktop/shihoon/glim_real/20260819_flat/results/flat_bag_C_imu_on.ply");
    output_directory_ = declare_parameter<std::string>(
      "output_directory",
      "/home/a/Desktop/shihoon/bunker_localization_ws/results/"
      "bag_D_flat_20260819_coarse_init");
    expected_lidar_frame_ = declare_parameter<std::string>(
      "expected_lidar_frame", "velodyne");

    CoarseGlobalInitializationSettings settings;
    settings.num_threads = declare_parameter<int>("num_threads", 4);
    settings.coarse_voxel_resolution = declare_parameter<double>(
      "coarse_voxel_resolution", 0.60);
    settings.xy_step = declare_parameter<double>("coarse_xy_step", 0.75);
    settings.z_step = declare_parameter<double>("coarse_z_step", 0.25);
    settings.yaw_step = declare_parameter<double>(
      "coarse_yaw_step_rad", 0.2617993877991494);
    settings.coarse_max_correspondence_distance = declare_parameter<double>(
      "coarse_max_correspondence_distance", 0.90);
    const auto positiveSizeParameter = [this](const std::string& name, const int default_value) {
        const int value = declare_parameter<int>(name, default_value);
        if (value < 1) {
          throw std::invalid_argument(name + " must be positive");
        }
        return static_cast<std::size_t>(value);
      };
    settings.maximum_scored_source_points = positiveSizeParameter(
      "maximum_scored_source_points", 2000);
    settings.broad_candidates_to_keep = positiveSizeParameter(
      "broad_candidates_to_keep", 256);
    settings.refinement_candidates = positiveSizeParameter(
      "refinement_candidates", 48);
    settings.reported_candidates = positiveSizeParameter("reported_candidates", 10);
    settings.refinement.num_threads = settings.num_threads;
    settings.refinement.map_voxel_resolution = declare_parameter<double>(
      "map_voxel_resolution", 0.20);
    settings.refinement.scan_voxel_resolution = declare_parameter<double>(
      "scan_voxel_resolution", 0.20);
    settings.refinement.num_neighbors = declare_parameter<int>("num_neighbors", 20);
    settings.refinement.max_correspondence_distance = declare_parameter<double>(
      "max_correspondence_distance", 1.0);
    settings.refinement.max_iterations = declare_parameter<int>("max_iterations", 30);
    settings.refinement.registration_type = "GICP";

    const LoadedMap loaded_map = loadPlyMap(map_path_);
    raw_map_points_ = loaded_map.raw_point_count;
    settings_ = settings;
    initializer_ = std::make_unique<CoarseGlobalInitializer>(loaded_map.points, settings);
    const std::string cloud_topic = declare_parameter<std::string>(
      "cloud_topic", "/velodyne_points");
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, rclcpp::QoS(5).reliable(),
      std::bind(&CoarseGlobalInitializerNode::cloudCallback, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(),
      "Prepared Bag C map (%zu raw points); waiting for first usable Bag D scan",
      raw_map_points_);
  }

private:
  void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    if (running_) {
      return;
    }
    running_ = true;
    try {
      if (message->header.frame_id != expected_lidar_frame_) {
        throw std::runtime_error(
                "Unexpected LiDAR frame: " + message->header.frame_id);
      }
      const ExtractedScan source = extractFiniteXYZ(*message);
      if (source.finite_point_count < 10U) {
        throw std::runtime_error("First Bag D scan contains too few finite points");
      }
      const double timestamp = stampSeconds(message->header.stamp);
      const CoarseGlobalInitializationResult result = initializer_->initialize(*source.points);
      writeOutputs(timestamp, source, result);
      if (!result.success) {
        RCLCPP_FATAL(
          get_logger(), "Coarse initialization produced no acceptable finite GICP candidate");
      } else {
        const auto& best = result.candidates.front();
        const auto rpy = rollPitchYaw(best.T_map_lidar.linear());
        RCLCPP_INFO(
          get_logger(),
          "Coarse seed: xyz=[%.3f %.3f %.3f] rpy_deg=[%.2f %.2f %.2f] "
          "inliers=%zu error/inlier=%.6f score=%.6f runtime=%.1f ms",
          best.T_map_lidar.translation().x(), best.T_map_lidar.translation().y(),
          best.T_map_lidar.translation().z(), rpy[0] * 180.0 / M_PI,
          rpy[1] * 180.0 / M_PI, rpy[2] * 180.0 / M_PI,
          best.num_inliers, best.final_error_per_inlier, best.refined_score,
          result.total_runtime_ms);
      }
    } catch (const std::exception& error) {
      RCLCPP_FATAL(get_logger(), "Coarse global initialization failed: %s", error.what());
    }
    rclcpp::shutdown();
  }

  void writeOutputs(
    const double timestamp,
    const ExtractedScan& source,
    const CoarseGlobalInitializationResult& result) const
  {
    const std::filesystem::path directory(output_directory_);
    std::filesystem::create_directories(directory);
    std::ofstream csv(directory / "top_candidates.csv", std::ios::trunc);
    csv << std::setprecision(17);
    csv << "rank,coarse_x,coarse_y,coarse_z,coarse_qx,coarse_qy,coarse_qz,coarse_qw,"
           "coarse_roll,coarse_pitch,coarse_yaw,coarse_score,coarse_inliers,"
           "coarse_inlier_rmse,x,y,z,qx,qy,qz,qw,roll,pitch,yaw,converged,iterations,"
           "num_inliers,final_error,final_error_per_inlier,refined_score,runtime_ms,"
           "source_downsampled_points\n";
    for (std::size_t index = 0; index < result.candidates.size(); ++index) {
      const auto& candidate = result.candidates[index];
      csv << index + 1U << ',';
      writeTransformCsv(csv, candidate.coarse_T_map_lidar);
      csv << ',' << candidate.coarse_score << ',' << candidate.coarse_inliers << ','
          << candidate.coarse_inlier_rmse << ',';
      writeTransformCsv(csv, candidate.T_map_lidar);
      csv << ',' << (candidate.converged ? 1 : 0) << ',' << candidate.iterations << ','
          << candidate.num_inliers << ',' << candidate.final_error << ','
          << candidate.final_error_per_inlier << ',' << candidate.refined_score << ','
          << candidate.runtime_ms << ',' << candidate.source_downsampled_points << '\n';
    }

    std::ofstream json(directory / "coarse_initialization.json", std::ios::trunc);
    json << std::setprecision(17);
    json << "{\n"
         << "  \"success\": " << (result.success ? "true" : "false") << ",\n"
         << "  \"map_path\": \"" << map_path_ << "\",\n"
         << "  \"source_bag_role\": \"independent Bag D first usable LiDAR scan\",\n"
         << "  \"source_timestamp\": " << timestamp << ",\n"
         << "  \"scans_used\": 1,\n"
         << "  \"scan_selection_reason\": "
         << "\"first scan is dense and the initial scans are stationary/redundant\",\n"
         << "  \"transform_convention\": \"p_map = T_map_lidar * p_lidar\",\n"
         << "  \"selection_policy\": "
         << "\"rank all refined candidates; never select first convergence\",\n"
         << "  \"raw_map_points\": " << raw_map_points_ << ",\n"
         << "  \"source_input_points\": " << source.input_point_count << ",\n"
         << "  \"source_finite_points\": " << source.finite_point_count << ",\n"
         << "  \"map_coarse_points\": " << result.map_coarse_points << ",\n"
         << "  \"source_coarse_points\": " << result.source_coarse_points << ",\n"
         << "  \"map_bounding_box_min\": [" << result.map_min.x() << ", "
         << result.map_min.y() << ", " << result.map_min.z() << "],\n"
         << "  \"map_bounding_box_max\": [" << result.map_max.x() << ", "
         << result.map_max.y() << ", " << result.map_max.z() << "],\n"
         << "  \"settings\": {\n"
         << "    \"coarse_voxel_resolution\": "
         << settings_.coarse_voxel_resolution << ",\n"
         << "    \"coarse_xy_step\": " << settings_.xy_step << ",\n"
         << "    \"coarse_z_step\": " << settings_.z_step << ",\n"
         << "    \"coarse_yaw_step_rad\": " << settings_.yaw_step << ",\n"
         << "    \"coarse_max_correspondence_distance\": "
         << settings_.coarse_max_correspondence_distance << ",\n"
         << "    \"maximum_scored_source_points\": "
         << settings_.maximum_scored_source_points << ",\n"
         << "    \"broad_candidates_to_keep\": "
         << settings_.broad_candidates_to_keep << ",\n"
         << "    \"refinement_candidates\": "
         << settings_.refinement_candidates << ",\n"
         << "    \"refinement_map_voxel_resolution\": "
         << settings_.refinement.map_voxel_resolution << ",\n"
         << "    \"refinement_scan_voxel_resolution\": "
         << settings_.refinement.scan_voxel_resolution << ",\n"
         << "    \"refinement_num_neighbors\": "
         << settings_.refinement.num_neighbors << ",\n"
         << "    \"refinement_max_correspondence_distance\": "
         << settings_.refinement.max_correspondence_distance << ",\n"
         << "    \"refinement_max_iterations\": "
         << settings_.refinement.max_iterations << "\n"
         << "  },\n"
         << "  \"broad_candidates_evaluated\": "
         << result.broad_candidates_evaluated << ",\n"
         << "  \"refinement_candidates_evaluated\": "
         << result.refinement_candidates_evaluated << ",\n"
         << "  \"refinement_candidates_succeeded\": "
         << result.refinement_candidates_succeeded << ",\n"
         << "  \"reported_distinct_candidates\": " << result.candidates.size() << ",\n"
         << "  \"total_runtime_ms\": " << result.total_runtime_ms << ",\n"
         << "  \"top_score_gap\": " << result.top_score_gap << ",\n"
         << "  \"top_score_ratio\": " << result.top_score_ratio << ",\n"
         << "  \"ambiguity_warning\": "
         << (result.candidates.size() > 1U && result.top_score_ratio < 1.05 ? "true" : "false")
         << ",\n"
         << "  \"best_candidate\": ";
    if (result.candidates.empty()) {
      json << "null\n";
    } else {
      const auto& best = result.candidates.front();
      json << "{\n"
           << "    \"T_map_lidar\": ";
      writeTransformJson(json, best.T_map_lidar);
      json << ",\n    \"coarse_score\": " << best.coarse_score << ",\n"
           << "    \"score\": " << best.refined_score << ",\n"
           << "    \"converged\": " << (best.converged ? "true" : "false") << ",\n"
           << "    \"iterations\": " << best.iterations << ",\n"
           << "    \"inliers\": " << best.num_inliers << ",\n"
           << "    \"final_error\": " << best.final_error << ",\n"
           << "    \"final_error_per_inlier\": " << best.final_error_per_inlier << ",\n"
           << "    \"registration_runtime_ms\": " << best.runtime_ms << "\n"
           << "  }\n";
      saveRegisteredSource(
        directory / "registered_first_scan.ply", *source.points, best.T_map_lidar);

      Eigen::Quaterniond quaternion(best.T_map_lidar.linear());
      quaternion.normalize();
      std::ofstream yaml(directory / "seed_parameter.yaml", std::ios::trunc);
      yaml << std::setprecision(17)
           << "offline_localizer:\n"
           << "  ros__parameters:\n"
           << "    initialization:\n"
           << "      mode: parameter\n"
           << "      translation: [" << best.T_map_lidar.translation().x() << ", "
           << best.T_map_lidar.translation().y() << ", "
           << best.T_map_lidar.translation().z() << "]\n"
           << "      rotation_xyzw: [" << quaternion.x() << ", " << quaternion.y() << ", "
           << quaternion.z() << ", " << quaternion.w() << "]\n";
    }
    json << "}\n";
  }

  std::string map_path_;
  std::string output_directory_;
  std::string expected_lidar_frame_;
  std::size_t raw_map_points_{};
  bool running_{false};
  CoarseGlobalInitializationSettings settings_;
  std::unique_ptr<CoarseGlobalInitializer> initializer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

}  // namespace bunker_offline_localization

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<
      bunker_offline_localization::CoarseGlobalInitializerNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("coarse_global_initializer"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
