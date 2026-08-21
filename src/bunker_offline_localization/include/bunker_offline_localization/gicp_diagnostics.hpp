#pragma once

#include "bunker_offline_localization/metrics.hpp"
#include "bunker_offline_localization/registration.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bunker_offline_localization {

struct SelectedScanSpec {
  std::string role;
  double timestamp{};
  bool audit_state_available{false};
  Eigen::Isometry3d audit_prediction{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d audit_registration{Eigen::Isometry3d::Identity()};
  std::size_t audit_inliers{};
  std::size_t audit_iterations{};
  double audit_final_error{};
  double audit_registration_runtime_ms{};
};

struct GicpDiagnosticSettings {
  bool enabled{false};
  bool publish_visualization{false};
  bool publish_accepted_pose{false};
  int publish_correspondences_every_n_scans{1};
  std::size_t rviz_max_correspondence_lines{200U};
  std::string output_directory;
  std::string map_frame{"map"};
  std::string localized_lidar_frame{"localized_velodyne"};
  double selected_timestamp_tolerance_sec{0.001};
  std::vector<SelectedScanSpec> selected_scans;
  std::string hold_selected_role;
};

class GicpDiagnostics {
public:
  GicpDiagnostics(
    rclcpp::Node* node,
    GicpDiagnosticSettings settings,
    small_gicp::PointCloud::Ptr raw_map,
    const small_gicp::PointCloud& target_map);

  bool needsCorrespondences(std::size_t processed_index, double timestamp) const;
  RegistrationOutput correspondenceRegistration(
    double timestamp, const RegistrationOutput& replay_registration) const;
  void handleProcessedScan(
    const sensor_msgs::msg::PointCloud2& raw_message,
    const small_gicp::PointCloud* finite_source,
    const LocalizationRecord& record,
    const RegistrationOutput* registration,
    const RegistrationOutput* correspondence_registration,
    const CorrespondenceReconstruction* correspondences,
    std::size_t processed_index);

  bool holdSelectedCaptured() const {return hold_selected_captured_;}
  std::size_t acceptedPathSize() const {return accepted_path_.poses.size();}

private:
  struct SelectedVisualizationSnapshot {
    sensor_msgs::msg::PointCloud2 raw_scan;
    sensor_msgs::msg::PointCloud2 registered_scan;
    geometry_msgs::msg::PoseStamped prediction_pose;
    geometry_msgs::msg::PoseStamped gicp_pose;
    nav_msgs::msg::Path gicp_path;
    visualization_msgs::msg::MarkerArray correspondences;
  };

  const SelectedScanSpec* matchSelected(double timestamp) const;
  void publishMap();
  void publishPose(
    const Eigen::Isometry3d& transform,
    const builtin_interfaces::msg::Time& stamp,
    const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& publisher) const;
  void publishCorrespondences(
    const CorrespondenceReconstruction& correspondences,
    const builtin_interfaces::msg::Time& stamp);
  void republishSelectedSnapshot();
  void writeSelectedAudit(
    const SelectedScanSpec& selected,
    const small_gicp::PointCloud& finite_source,
    const LocalizationRecord& record,
    const RegistrationOutput& registration,
    const CorrespondenceReconstruction& correspondences,
    std::size_t processed_index);

  rclcpp::Node* node_{};
  GicpDiagnosticSettings settings_;
  small_gicp::PointCloud::Ptr raw_map_;
  const small_gicp::PointCloud* target_map_{};
  nav_msgs::msg::Path accepted_path_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_scan_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr registered_scan_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr accepted_pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr prediction_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr correspondence_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr selected_snapshot_timer_;
  std::optional<SelectedVisualizationSnapshot> selected_snapshot_;
  bool hold_selected_captured_{false};
};

}  // namespace bunker_offline_localization
