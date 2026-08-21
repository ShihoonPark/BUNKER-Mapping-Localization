#include "bunker_offline_localization/gicp_diagnostics.hpp"

#include "bunker_offline_localization/diagnostic_utils.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <small_gicp/points/traits.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bunker_offline_localization {
namespace {

sensor_msgs::msg::PointCloud2 cloudMessage(
  const small_gicp::PointCloud& points,
  const std::string& frame,
  const builtin_interfaces::msg::Time& stamp,
  const Eigen::Isometry3d& transform = Eigen::Isometry3d::Identity())
{
  sensor_msgs::msg::PointCloud2 message;
  message.header.frame_id = frame;
  message.header.stamp = stamp;
  message.height = 1U;
  message.width = static_cast<std::uint32_t>(points.size());
  sensor_msgs::PointCloud2Modifier modifier(message);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(points.size());
  sensor_msgs::PointCloud2Iterator<float> x(message, "x");
  sensor_msgs::PointCloud2Iterator<float> y(message, "y");
  sensor_msgs::PointCloud2Iterator<float> z(message, "z");
  for (std::size_t index = 0; index < points.size(); ++index, ++x, ++y, ++z) {
    const Eigen::Vector4d point = transform * small_gicp::traits::point(points, index);
    *x = static_cast<float>(point.x());
    *y = static_cast<float>(point.y());
    *z = static_cast<float>(point.z());
  }
  return message;
}

geometry_msgs::msg::Pose poseMessage(const Eigen::Isometry3d& transform)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  Eigen::Quaterniond quaternion(transform.linear());
  quaternion.normalize();
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  return pose;
}

geometry_msgs::msg::PoseStamped poseStampedMessage(
  const Eigen::Isometry3d& transform,
  const std::string& frame,
  const builtin_interfaces::msg::Time& stamp)
{
  geometry_msgs::msg::PoseStamped message;
  message.header.frame_id = frame;
  message.header.stamp = stamp;
  message.pose = poseMessage(transform);
  return message;
}

visualization_msgs::msg::MarkerArray correspondenceMessage(
  const CorrespondenceReconstruction& correspondences,
  const std::string& frame,
  const builtin_interfaces::msg::Time& stamp,
  const std::size_t max_lines)
{
  visualization_msgs::msg::MarkerArray array;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  array.markers.push_back(clear);
  visualization_msgs::msg::Marker lines;
  lines.header.frame_id = frame;
  lines.header.stamp = stamp;
  lines.ns = "posthoc_final_transform_correspondences";
  lines.id = 0;
  lines.type = visualization_msgs::msg::Marker::LINE_LIST;
  lines.action = visualization_msgs::msg::Marker::ADD;
  // A selected snapshot must remain visible until it is replaced or the process exits.
  lines.lifetime.sec = 0;
  lines.lifetime.nanosec = 0U;
  lines.scale.x = 0.012;
  lines.color.r = 1.0F;
  lines.color.g = 0.25F;
  lines.color.b = 0.05F;
  lines.color.a = 0.75F;
  const auto indices = deterministicSubsampleIndices(correspondences.valid.size(), max_lines);
  lines.points.reserve(indices.size() * 2U);
  for (const std::size_t index : indices) {
    const auto& correspondence = correspondences.valid[index];
    geometry_msgs::msg::Point source;
    source.x = correspondence.registered_source_map.x();
    source.y = correspondence.registered_source_map.y();
    source.z = correspondence.registered_source_map.z();
    geometry_msgs::msg::Point target;
    target.x = correspondence.target_map.x();
    target.y = correspondence.target_map.y();
    target.z = correspondence.target_map.z();
    lines.points.push_back(source);
    lines.points.push_back(target);
  }
  array.markers.push_back(std::move(lines));
  return array;
}

void writePly(
  const std::filesystem::path& path,
  const small_gicp::PointCloud& points,
  const Eigen::Isometry3d& transform = Eigen::Isometry3d::Identity())
{
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("Failed to write PLY: " + path.string());
  }
  output << "ply\nformat ascii 1.0\nelement vertex " << points.size()
         << "\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
         << std::setprecision(17);
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Eigen::Vector4d point = transform * small_gicp::traits::point(points, index);
    output << point.x() << ' ' << point.y() << ' ' << point.z() << '\n';
  }
}

void writeTransformJson(std::ostream& output, const Eigen::Isometry3d& transform)
{
  Eigen::Quaterniond quaternion(transform.linear());
  quaternion.normalize();
  output << "{\"x\":" << transform.translation().x()
         << ",\"y\":" << transform.translation().y()
         << ",\"z\":" << transform.translation().z()
         << ",\"qx\":" << quaternion.x()
         << ",\"qy\":" << quaternion.y()
         << ",\"qz\":" << quaternion.z()
         << ",\"qw\":" << quaternion.w() << '}';
}

}  // namespace

GicpDiagnostics::GicpDiagnostics(
  rclcpp::Node* node,
  GicpDiagnosticSettings settings,
  small_gicp::PointCloud::Ptr raw_map,
  const small_gicp::PointCloud& target_map)
: node_(node), settings_(std::move(settings)), raw_map_(std::move(raw_map)),
  target_map_(&target_map)
{
  if (!node_ || !raw_map_ || settings_.selected_timestamp_tolerance_sec <= 0.0 ||
    settings_.publish_correspondences_every_n_scans < 0 ||
    settings_.rviz_max_correspondence_lines == 0U)
  {
    throw std::invalid_argument("Invalid GICP diagnostic settings");
  }
  accepted_path_.header.frame_id = settings_.map_frame;
  if (!settings_.enabled) {
    return;
  }
  std::filesystem::create_directories(settings_.output_directory);
  if (settings_.publish_accepted_pose) {
    auto accepted_pose_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    accepted_pose_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/localization/pose", accepted_pose_qos);
    status_publisher_ = node_->create_publisher<std_msgs::msg::String>(
      "/gicp_status", accepted_pose_qos);
  }
  if (!settings_.publish_visualization) {
    return;
  }
  auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  auto live_qos = rclcpp::QoS(rclcpp::KeepLast(2)).best_effort().durability_volatile();
  map_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/map_cloud", map_qos);
  raw_scan_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/raw_scan", live_qos);
  registered_scan_publisher_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/registered_scan", live_qos);
  pose_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/gicp_pose", live_qos);
  prediction_publisher_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/prediction_pose", live_qos);
  path_publisher_ = node_->create_publisher<nav_msgs::msg::Path>(
    "/gicp_path", live_qos);
  correspondence_publisher_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/gicp_correspondences", live_qos);
  if (!status_publisher_) {
    status_publisher_ = node_->create_publisher<std_msgs::msg::String>(
      "/gicp_status", live_qos);
  }
  transform_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);
  if (!settings_.hold_selected_role.empty()) {
    selected_snapshot_timer_ = node_->create_wall_timer(
      std::chrono::milliseconds(500), [this]() {republishSelectedSnapshot();});
  }
  publishMap();
}

const SelectedScanSpec* GicpDiagnostics::matchSelected(const double timestamp) const
{
  const auto match = std::find_if(
    settings_.selected_scans.begin(), settings_.selected_scans.end(),
    [this, timestamp](const SelectedScanSpec& selected) {
      return std::abs(selected.timestamp - timestamp) <=
             settings_.selected_timestamp_tolerance_sec;
    });
  return match == settings_.selected_scans.end() ? nullptr : &*match;
}

bool GicpDiagnostics::needsCorrespondences(
  const std::size_t processed_index, const double timestamp) const
{
  if (!settings_.enabled) {
    return false;
  }
  if (matchSelected(timestamp)) {
    return true;
  }
  return settings_.publish_visualization &&
         settings_.publish_correspondences_every_n_scans > 0 &&
         processed_index % static_cast<std::size_t>(
    settings_.publish_correspondences_every_n_scans) == 0U;
}

RegistrationOutput GicpDiagnostics::correspondenceRegistration(
  const double timestamp, const RegistrationOutput& replay_registration) const
{
  RegistrationOutput audit = replay_registration;
  const SelectedScanSpec* selected = matchSelected(timestamp);
  if (!selected || !selected->audit_state_available) {
    return audit;
  }
  // The five roles were selected from a protected production result. Reconstruct their
  // correspondences at that recorded full-6DoF solution even if a new asynchronous replay lands
  // in a different local basin. The source cloud is still the actual production-preprocessed scan.
  audit.T_map_lidar = selected->audit_registration;
  audit.converged = true;
  audit.num_inliers = selected->audit_inliers;
  audit.iterations = selected->audit_iterations;
  audit.final_error = selected->audit_final_error;
  audit.runtime_ms = selected->audit_registration_runtime_ms;
  return audit;
}

void GicpDiagnostics::publishMap()
{
  builtin_interfaces::msg::Time stamp;
  map_publisher_->publish(cloudMessage(*raw_map_, settings_.map_frame, stamp));
}

void GicpDiagnostics::publishPose(
  const Eigen::Isometry3d& transform,
  const builtin_interfaces::msg::Time& stamp,
  const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr& publisher) const
{
  if (!publisher || !isFiniteTransform(transform)) {
    return;
  }
  publisher->publish(poseStampedMessage(transform, settings_.map_frame, stamp));
}

void GicpDiagnostics::publishCorrespondences(
  const CorrespondenceReconstruction& correspondences,
  const builtin_interfaces::msg::Time& stamp)
{
  if (!correspondence_publisher_) {
    return;
  }
  correspondence_publisher_->publish(correspondenceMessage(
    correspondences, settings_.map_frame, stamp, settings_.rviz_max_correspondence_lines));
}

void GicpDiagnostics::republishSelectedSnapshot()
{
  if (!selected_snapshot_) {
    return;
  }
  raw_scan_publisher_->publish(selected_snapshot_->raw_scan);
  registered_scan_publisher_->publish(selected_snapshot_->registered_scan);
  prediction_publisher_->publish(selected_snapshot_->prediction_pose);
  pose_publisher_->publish(selected_snapshot_->gicp_pose);
  path_publisher_->publish(selected_snapshot_->gicp_path);
  correspondence_publisher_->publish(selected_snapshot_->correspondences);
}

void GicpDiagnostics::writeSelectedAudit(
  const SelectedScanSpec& selected,
  const small_gicp::PointCloud& finite_source,
  const LocalizationRecord& record,
  const RegistrationOutput& registration,
  const CorrespondenceReconstruction& correspondences,
  const std::size_t processed_index)
{
  const auto directory = std::filesystem::path(settings_.output_directory) /
    "correspondence_audit" / selected.role;
  std::filesystem::create_directories(directory);
  writePly(directory / "raw_finite_scan.ply", finite_source);
  writePly(directory / "preprocessed_source_scan.ply", *registration.preprocessed_source);
  writePly(
    directory / "registered_scan.ply", *registration.preprocessed_source,
    registration.T_map_lidar);

  std::ofstream csv(directory / "correspondences.csv", std::ios::trunc);
  csv << std::setprecision(17)
      << "source_index,target_index,source_x,source_y,source_z,registered_x,registered_y"
      << ",registered_z,target_x,target_y,target_z,residual_x,residual_y,residual_z"
      << ",distance_m,mahalanobis_error_contribution\n";
  for (const auto& item : correspondences.valid) {
    csv << item.source_index << ',' << item.target_index << ','
        << item.source_lidar.x() << ',' << item.source_lidar.y() << ','
        << item.source_lidar.z() << ',' << item.registered_source_map.x() << ','
        << item.registered_source_map.y() << ',' << item.registered_source_map.z() << ','
        << item.target_map.x() << ',' << item.target_map.y() << ',' << item.target_map.z() << ','
        << item.residual_map.x() << ',' << item.residual_map.y() << ','
        << item.residual_map.z() << ',' << item.distance_m << ','
        << item.mahalanobis_error_contribution << '\n';
  }

  std::ofstream metadata(directory / "scan_metadata.json", std::ios::trunc);
  metadata << std::setprecision(17)
           << "{\n  \"selection_role\": \"" << selected.role << "\",\n"
           << "  \"timestamp\": " << record.timestamp << ",\n"
           << "  \"processed_index\": " << processed_index << ",\n"
           << "  \"status\": \"" << (record.accepted ? "ACCEPTED" : "REJECTED") << "\",\n"
           << "  \"correspondence_method\": \"" << correspondences.method << "\",\n"
           << "  \"exact_internal_correspondence_claimed\": false,\n"
           << "  \"audit_pose_source\": \"protected_production_localization_csv\",\n"
           << "  \"raw_finite_points\": " << finite_source.size() << ",\n"
           << "  \"source_points_after_voxel\": "
           << registration.source_downsampled_points << ",\n"
           << "  \"candidate_correspondences\": " << correspondences.candidate_count << ",\n"
           << "  \"valid_correspondences\": " << correspondences.valid.size() << ",\n"
           << "  \"gicp_inliers\": " << record.num_inliers << ",\n"
           << "  \"final_error\": " << record.final_error << ",\n"
           << "  \"iterations\": " << record.iterations << ",\n"
           << "  \"registration_runtime_ms\": " << record.runtime_ms << ",\n"
           << "  \"core_localization_latency_ms\": "
           << record.core_localization_latency_ms << ",\n"
           << "  \"prediction_T_map_lidar\": ";
  writeTransformJson(metadata, record.prediction);
  metadata << ",\n  \"final_T_map_lidar\": ";
  writeTransformJson(metadata, record.registration);
  metadata << "\n}\n";
}

void GicpDiagnostics::handleProcessedScan(
  const sensor_msgs::msg::PointCloud2& raw_message,
  const small_gicp::PointCloud* finite_source,
  const LocalizationRecord& record,
  const RegistrationOutput* registration,
  const RegistrationOutput* correspondence_registration,
  const CorrespondenceReconstruction* correspondences,
  const std::size_t processed_index)
{
  if (!settings_.enabled) {
    return;
  }
  const SelectedScanSpec* selected = matchSelected(record.timestamp);
  LocalizationRecord audit_record = record;
  if (selected && selected->audit_state_available) {
    audit_record.prediction = selected->audit_prediction;
    audit_record.registration = selected->audit_registration;
    audit_record.num_inliers = selected->audit_inliers;
    audit_record.iterations = selected->audit_iterations;
    audit_record.final_error = selected->audit_final_error;
    audit_record.runtime_ms = selected->audit_registration_runtime_ms;
    audit_record.converged = true;
    audit_record.accepted = true;
    audit_record.reject_reason = RejectReason::None;
  }
  if (selected && finite_source && correspondence_registration && correspondences) {
    writeSelectedAudit(
      *selected, *finite_source, audit_record, *correspondence_registration,
      *correspondences, processed_index);
  }
  if (record.accepted && accepted_pose_publisher_) {
    publishPose(record.registration, raw_message.header.stamp, accepted_pose_publisher_);
  }
  const bool selected_hold = selected && finite_source && correspondence_registration &&
    correspondences && !settings_.hold_selected_role.empty() &&
    selected->role == settings_.hold_selected_role;
  if (status_publisher_) {
    const LocalizationRecord& status_record = selected_hold ? audit_record : record;
    std_msgs::msg::String status;
    status.data = status_record.accepted ? std::string("ACCEPTED") :
      std::string("REJECTED_") + toString(status_record.reject_reason);
    status_publisher_->publish(status);
  }
  if (!settings_.publish_visualization) {
    hold_selected_captured_ = hold_selected_captured_ || selected_hold;
    return;
  }

  raw_scan_publisher_->publish(raw_message);
  const LocalizationRecord& visualization_record = selected_hold ? audit_record : record;
  const RegistrationOutput* visualization_registration = selected_hold ?
    correspondence_registration : registration;
  publishPose(
    visualization_record.prediction, raw_message.header.stamp, prediction_publisher_);
  if (visualization_registration && visualization_registration->preprocessed_source &&
    isFiniteTransform(visualization_record.registration))
  {
    registered_scan_publisher_->publish(cloudMessage(
      *visualization_registration->preprocessed_source, settings_.map_frame,
      raw_message.header.stamp, visualization_record.registration));
    publishPose(
      visualization_record.registration, raw_message.header.stamp, pose_publisher_);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = settings_.map_frame;
    transform.header.stamp = raw_message.header.stamp;
    transform.child_frame_id = settings_.localized_lidar_frame;
    transform.transform.translation.x = visualization_record.registration.translation().x();
    transform.transform.translation.y = visualization_record.registration.translation().y();
    transform.transform.translation.z = visualization_record.registration.translation().z();
    Eigen::Quaterniond quaternion(visualization_record.registration.linear());
    quaternion.normalize();
    transform.transform.rotation.x = quaternion.x();
    transform.transform.rotation.y = quaternion.y();
    transform.transform.rotation.z = quaternion.z();
    transform.transform.rotation.w = quaternion.w();
    transform_broadcaster_->sendTransform(transform);
  }
  if (visualization_record.accepted) {
    geometry_msgs::msg::PoseStamped path_pose;
    path_pose.header.frame_id = settings_.map_frame;
    path_pose.header.stamp = raw_message.header.stamp;
    path_pose.pose = poseMessage(visualization_record.registration);
    accepted_path_.poses.push_back(path_pose);
    accepted_path_.header.stamp = raw_message.header.stamp;
  }
  path_publisher_->publish(accepted_path_);
  if (correspondences) {
    publishCorrespondences(*correspondences, raw_message.header.stamp);
  }
  if (selected_hold && visualization_registration &&
    visualization_registration->preprocessed_source && correspondences &&
    isFiniteTransform(visualization_record.prediction) &&
    isFiniteTransform(visualization_record.registration))
  {
    SelectedVisualizationSnapshot snapshot;
    // Keep the unregistered LiDAR-frame XYZ geometry while avoiding repeatedly serializing
    // unused raw packet fields in the persistent best-effort selected snapshot.
    snapshot.raw_scan = cloudMessage(
      *finite_source, raw_message.header.frame_id, raw_message.header.stamp);
    snapshot.registered_scan = cloudMessage(
      *visualization_registration->preprocessed_source, settings_.map_frame,
      raw_message.header.stamp, visualization_record.registration);
    snapshot.prediction_pose = poseStampedMessage(
      visualization_record.prediction, settings_.map_frame, raw_message.header.stamp);
    snapshot.gicp_pose = poseStampedMessage(
      visualization_record.registration, settings_.map_frame, raw_message.header.stamp);
    snapshot.gicp_path = accepted_path_;

    snapshot.correspondences = correspondenceMessage(
      *correspondences, settings_.map_frame, raw_message.header.stamp,
      settings_.rviz_max_correspondence_lines);
    selected_snapshot_ = std::move(snapshot);
    hold_selected_captured_ = true;
  }
}

}  // namespace bunker_offline_localization
