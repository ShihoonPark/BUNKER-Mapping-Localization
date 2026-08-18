#include "bunker_offline_localization/accepted_pose_predictor.hpp"
#include "bunker_offline_localization/map_loader.hpp"
#include "bunker_offline_localization/metrics.hpp"
#include "bunker_offline_localization/reference_trajectory.hpp"
#include "bunker_offline_localization/registration.hpp"
#include "bunker_offline_localization/scan_preprocessor.hpp"
#include "bunker_offline_localization/time_window.hpp"
#include "bunker_offline_localization/transforms.hpp"

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bunker_offline_localization {
namespace {

double stampSeconds(const builtin_interfaces::msg::Time& stamp)
{
  return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

Eigen::Isometry3d odometryPose(const nav_msgs::msg::Odometry& message)
{
  const auto& position = message.pose.pose.position;
  const auto& orientation = message.pose.pose.orientation;
  return makeTransform(
    position.x, position.y, position.z,
    orientation.x, orientation.y, orientation.z, orientation.w);
}

Eigen::Isometry3d nanTransform()
{
  Eigen::Isometry3d transform;
  transform.matrix().setConstant(std::numeric_limits<double>::quiet_NaN());
  return transform;
}

struct TimedPrediction {
  double timestamp{};
  Eigen::Isometry3d T_odom_base{Eigen::Isometry3d::Identity()};
};

struct PredictionAssociation {
  TimedPrediction prediction;
  double time_difference{};
};

struct TimedFilterRuntime {
  double timestamp{};
  double runtime_ms{};
};

class OfflineLocalizerNode : public rclcpp::Node {
public:
  OfflineLocalizerNode()
  : Node("offline_localizer"),
    map_path_(declare_parameter<std::string>(
        "map_path",
        "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/classroom_150626.ply")),
    reference_path_(declare_parameter<std::string>(
        "reference_path",
        "/home/a/Desktop/shihoon/glim_real/20260814_classroom/results/"
        "dump_150626_direct_20260814_222719/traj_lidar.txt")),
    results_directory_(declare_parameter<std::string>(
        "results_directory",
        "/home/a/Desktop/shihoon/bunker_localization_ws/results"))
  {
    reference_tolerance_ = declare_parameter<double>("reference_timestamp_tolerance", 0.06);
    prediction_tolerance_ = declare_parameter<double>("prediction_timestamp_tolerance", 0.10);
    filter_timing_tolerance_ = declare_parameter<double>("filter_timing_tolerance", 0.05);
    require_filter_timing_ = declare_parameter<bool>("require_filter_timing", false);
    filter_type_label_ = declare_parameter<std::string>("filter_type_label", "ekf");
    max_scans_ = declare_parameter<int>("max_scans", 0);
    if (reference_tolerance_ <= 0.0 || prediction_tolerance_ <= 0.0 ||
      filter_timing_tolerance_ <= 0.0 || max_scans_ < 0)
    {
      throw std::invalid_argument("Timestamp tolerances must be positive and max_scans nonnegative");
    }

    initialization_mode_ = declare_parameter<std::string>("initialization.mode", "reference");
    if (initialization_mode_ == "reference") {
      reference_.emplace(ReferenceTrajectory::load(reference_path_));
      initial_T_map_lidar_ = reference_->first().T_map_lidar;
    } else if (initialization_mode_ == "parameter") {
      const auto translation = declare_parameter<std::vector<double>>(
        "initialization.translation", std::vector<double>{});
      const auto rotation = declare_parameter<std::vector<double>>(
        "initialization.rotation_xyzw", std::vector<double>{});
      if (translation.size() != 3U || rotation.size() != 4U) {
        throw std::invalid_argument(
                "Parameter initialization requires translation[3] and rotation_xyzw[4]");
      }
      initial_T_map_lidar_ = makeTransform(
        translation[0], translation[1], translation[2],
        rotation[0], rotation[1], rotation[2], rotation[3]);
    } else {
      throw std::invalid_argument("initialization.mode must be 'reference' or 'parameter'");
    }

    time_window_ = TimeWindow(
      declare_parameter<bool>("time_window.enabled", false),
      declare_parameter<double>("time_window.origin_timestamp", 0.0),
      declare_parameter<double>("time_window.start_offset_sec", 0.0),
      declare_parameter<double>("time_window.end_offset_sec", 0.0));

    RegistrationSettings registration_settings;
    registration_settings.num_threads = declare_parameter<int>("num_threads", 4);
    registration_settings.map_voxel_resolution =
      declare_parameter<double>("map_voxel_resolution", 0.20);
    registration_settings.scan_voxel_resolution =
      declare_parameter<double>("scan_voxel_resolution", 0.20);
    registration_settings.num_neighbors = declare_parameter<int>("num_neighbors", 20);
    registration_settings.max_correspondence_distance =
      declare_parameter<double>("max_correspondence_distance", 1.0);
    registration_settings.max_iterations = declare_parameter<int>("max_iterations", 30);
    registration_settings.registration_type =
      declare_parameter<std::string>("registration_type", "GICP");

    const int min_inliers = declare_parameter<int>("min_inliers", 100);
    quality_settings_.min_inliers = static_cast<std::size_t>(min_inliers);
    quality_settings_.max_final_error_per_inlier =
      declare_parameter<double>("max_final_error_per_inlier", 5.0);
    quality_settings_.max_translation_correction =
      declare_parameter<double>("max_translation_correction", 1.0);
    quality_settings_.max_rotation_correction =
      declare_parameter<double>("max_rotation_correction", 0.5235987755982988);
    if (min_inliers < 1 || quality_settings_.max_final_error_per_inlier <= 0.0 ||
      quality_settings_.max_translation_correction <= 0.0 ||
      quality_settings_.max_rotation_correction <= 0.0)
    {
      throw std::invalid_argument("Quality-gate thresholds must be positive");
    }

    configureBaseToLidar();
    const LoadedMap loaded_map = loadPlyMap(map_path_);
    registrar_ = std::make_unique<MapRegistrar>(loaded_map.points, registration_settings);
    writer_ = std::make_unique<ResultWriter>(results_directory_);
    raw_map_points_ = loaded_map.raw_point_count;
    target_map_points_ = registrar_->targetPointCount();

    const std::string cloud_topic = declare_parameter<std::string>(
      "cloud_topic", "/velodyne_points");
    const std::string prediction_topic = declare_parameter<std::string>(
      "prediction_topic", "/localization/odometry/filtered");
    const std::string filter_timing_topic = declare_parameter<std::string>(
      "filter_timing_topic", "/localization/filter_timing");
    expected_lidar_frame_ = declare_parameter<std::string>("expected_lidar_frame", "velodyne");

    cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, rclcpp::QoS(50).reliable(),
      std::bind(&OfflineLocalizerNode::cloudCallback, this, std::placeholders::_1));
    prediction_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      prediction_topic, rclcpp::QoS(500),
      std::bind(&OfflineLocalizerNode::predictionCallback, this, std::placeholders::_1));
    filter_timing_subscription_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      filter_timing_topic, rclcpp::QoS(500),
      std::bind(&OfflineLocalizerNode::filterTimingCallback, this, std::placeholders::_1));

    if (reference_) {
      RCLCPP_INFO(
        get_logger(),
        "Prepared global target map once: raw=%zu downsampled=%zu; first reference=%.9f",
        raw_map_points_, target_map_points_, reference_->first().timestamp);
    } else {
      const auto& translation = initial_T_map_lidar_.translation();
      RCLCPP_INFO(
        get_logger(),
        "Prepared global target map once: raw=%zu downsampled=%zu; parameter seed="
        "[%.3f, %.3f, %.3f]",
        raw_map_points_, target_map_points_, translation.x(), translation.y(), translation.z());
    }
    if (time_window_.enabled()) {
      RCLCPP_INFO(
        get_logger(), "Using inclusive bag-relative window %.3f..%.3f s from %.9f",
        time_window_.startOffset(), time_window_.endOffset(), time_window_.originTimestamp());
    }
  }

  ~OfflineLocalizerNode() override
  {
    try {
      drainPending(true);
      writeRunSummary();
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "Finalization failed: %s", error.what());
    }
  }

private:
  void configureBaseToLidar()
  {
    const bool available = declare_parameter<bool>("base_to_lidar.available", false);
    const bool allow_identity = declare_parameter<bool>(
      "base_to_lidar.allow_identity_for_phase1_smoke_test", false);
    const auto translation = declare_parameter<std::vector<double>>(
      "base_to_lidar.translation", std::vector<double>{});
    const auto rotation = declare_parameter<std::vector<double>>(
      "base_to_lidar.rotation_xyzw", std::vector<double>{});

    if (available) {
      if (translation.size() != 3U || rotation.size() != 4U) {
        throw std::invalid_argument(
                "Available base_to_lidar requires translation[3] and rotation_xyzw[4]");
      }
      T_base_lidar_ = makeTransform(
        translation[0], translation[1], translation[2],
        rotation[0], rotation[1], rotation[2], rotation[3]);
      prediction_approximation_ = false;
      return;
    }
    if (!allow_identity) {
      throw std::runtime_error(
              "base_link->velodyne extrinsic is unavailable. Set an actual transform, or explicitly "
              "enable the Phase 1 identity approximation");
    }
    T_base_lidar_ = Eigen::Isometry3d::Identity();
    prediction_approximation_ = true;
    RCLCPP_WARN(
      get_logger(),
      "TEST-ONLY APPROXIMATION: T_base_lidar=identity. T_map_lidar can be evaluated, but "
      "base_link localization/map->odom is not validated");
  }

  void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr message)
  {
    ++total_clouds_received_;
    const double timestamp = stampSeconds(message->header.stamp);
    if (!std::isfinite(timestamp)) {
      return;
    }
    const TimeWindowPosition window_position = time_window_.classify(timestamp);
    if (window_position == TimeWindowPosition::Before) {
      ++skipped_before_window_;
      return;
    }
    if (window_position == TimeWindowPosition::After) {
      ++skipped_after_window_;
      // The first cloud beyond the inclusive end is a deterministic stop signal. Drain any
      // earlier scan that was waiting for a prediction, but never process this cloud.
      drainPending(true);
      finished_ = true;
      writeRunSummary();
      RCLCPP_INFO(
        get_logger(), "Reached bag-relative time-window end %.3f s; shutting down",
        time_window_.endOffset());
      rclcpp::shutdown();
      return;
    }
    // GLIM and PointCloud2 stamps differ by sub-millisecond serialization/frame timing. Treat
    // the scan associated with the first reference pose as the first usable scan; only scans
    // earlier than the configured association window are unconditionally skipped.
    if (reference_ && timestamp < reference_->first().timestamp - reference_tolerance_) {
      ++skipped_before_reference_;
      return;
    }
    pending_scans_.push_back(std::move(message));
    drainPending(false);
  }

  void predictionCallback(nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    try {
      const double timestamp = stampSeconds(message->header.stamp);
      if (!std::isfinite(timestamp)) {
        return;
      }
      if (time_window_.classify(timestamp) != TimeWindowPosition::Inside) {
        return;
      }
      predictions_.push_back(TimedPrediction{timestamp, odometryPose(*message)});
      while (predictions_.size() > 2000U) {
        predictions_.pop_front();
      }
      drainPending(false);
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(), "Ignoring invalid EKF prediction: %s", error.what());
    }
  }

  void filterTimingCallback(geometry_msgs::msg::Vector3Stamped::ConstSharedPtr message)
  {
    const double timestamp = stampSeconds(message->header.stamp);
    if (!std::isfinite(timestamp) || !std::isfinite(message->vector.x) ||
      message->vector.x < 0.0)
    {
      return;
    }
    filter_runtimes_.push_back(TimedFilterRuntime{timestamp, message->vector.x});
    while (filter_runtimes_.size() > 2000U) {
      filter_runtimes_.pop_front();
    }
    drainPending(false);
  }

  std::optional<PredictionAssociation> nearestPrediction(const double timestamp) const
  {
    if (predictions_.empty()) {
      return std::nullopt;
    }
    const auto nearest = std::min_element(
      predictions_.begin(), predictions_.end(),
      [timestamp](const TimedPrediction& lhs, const TimedPrediction& rhs) {
        return std::abs(lhs.timestamp - timestamp) < std::abs(rhs.timestamp - timestamp);
      });
    return PredictionAssociation{*nearest, std::abs(nearest->timestamp - timestamp)};
  }

  std::optional<std::pair<double, double>> nearestFilterRuntime(const double timestamp) const
  {
    if (filter_runtimes_.empty()) {
      return std::nullopt;
    }
    const auto nearest = std::min_element(
      filter_runtimes_.begin(), filter_runtimes_.end(),
      [timestamp](const TimedFilterRuntime& lhs, const TimedFilterRuntime& rhs) {
        return std::abs(lhs.timestamp - timestamp) < std::abs(rhs.timestamp - timestamp);
      });
    const double difference = std::abs(nearest->timestamp - timestamp);
    if (difference > filter_timing_tolerance_) {
      return std::nullopt;
    }
    return std::make_pair(nearest->runtime_ms, difference);
  }

  void drainPending(const bool final)
  {
    while (!pending_scans_.empty() && !finished_) {
      const double timestamp = stampSeconds(pending_scans_.front()->header.stamp);
      if (!final && (predictions_.empty() || predictions_.back().timestamp < timestamp)) {
        return;
      }
      if (!final && require_filter_timing_ &&
        (filter_runtimes_.empty() ||
        filter_runtimes_.back().timestamp < timestamp - prediction_tolerance_))
      {
        return;
      }
      auto message = pending_scans_.front();
      pending_scans_.pop_front();
      processScan(*message, nearestPrediction(timestamp));
    }
  }

  void processScan(
    const sensor_msgs::msg::PointCloud2& message,
    const std::optional<PredictionAssociation>& association)
  {
    const double timestamp = stampSeconds(message.header.stamp);
    LocalizationRecord record;
    record.timestamp = timestamp;
    record.prediction = nanTransform();
    record.registration = nanTransform();
    record.reference_time_difference = std::numeric_limits<double>::quiet_NaN();
    record.prediction_time_difference = std::numeric_limits<double>::quiet_NaN();
    record.filter_runtime_ms = std::numeric_limits<double>::quiet_NaN();
    record.filter_runtime_time_difference = std::numeric_limits<double>::quiet_NaN();
    record.correction_translation_m = std::numeric_limits<double>::quiet_NaN();
    record.correction_roll_rad = std::numeric_limits<double>::quiet_NaN();
    record.correction_pitch_rad = std::numeric_limits<double>::quiet_NaN();
    record.correction_yaw_rad = std::numeric_limits<double>::quiet_NaN();
    record.prediction_approximation = prediction_approximation_;

    std::optional<PoseAssociation> reference_association;
    if (reference_) {
      reference_association = reference_->associateNearest(timestamp, reference_tolerance_);
    }
    if (reference_association) {
      record.reference_time_difference = reference_association->absolute_time_difference;
    }

    if (!association) {
      record.reject_reason = RejectReason::NoPrediction;
      finishRecord(record);
      return;
    }
    record.prediction_time_difference = association->time_difference;
    const auto filter_runtime = nearestFilterRuntime(association->prediction.timestamp);
    if (filter_runtime) {
      record.filter_runtime_ms = filter_runtime->first;
      record.filter_runtime_time_difference = filter_runtime->second;
    }
    if (association->time_difference > prediction_tolerance_) {
      record.reject_reason = RejectReason::TimestampMismatch;
      finishRecord(record);
      return;
    }

    if (!anchor_prediction_) {
      if (reference_) {
        const double first_reference_difference =
          std::abs(timestamp - reference_->first().timestamp);
        if (!reference_association || first_reference_difference > reference_tolerance_) {
          record.reject_reason = RejectReason::TimestampMismatch;
          finishRecord(record);
          return;
        }
      }
      anchor_prediction_ = association->prediction.T_odom_base;
      anchor_T_map_lidar_ = initial_T_map_lidar_;
    }

    // The EKF is only a planar motion source. Anchor every prediction to the last accepted
    // full-6DoF GICP LiDAR pose so unobserved EKF z/roll/pitch can never accumulate.
    const PlanarRelativeMotion planar_motion = ekfPlanarRelativeMotion(
      *anchor_prediction_, association->prediction.T_odom_base);
    record.prediction = predictMapLidarFromAcceptedPose(
      anchor_T_map_lidar_, planar_motion);
    record.prediction_available = true;

    if (message.header.frame_id != expected_lidar_frame_) {
      RCLCPP_ERROR(
        get_logger(), "Unexpected cloud frame '%s' (expected '%s')",
        message.header.frame_id.c_str(), expected_lidar_frame_.c_str());
      record.reject_reason = RejectReason::RegistrationException;
      finishRecord(record);
      return;
    }

    try {
      const ExtractedScan scan = extractFiniteXYZ(message);
      record.input_points = scan.input_point_count;
      record.finite_points = scan.finite_point_count;
      if (!scan.points || scan.points->size() < 10U) {
        record.reject_reason = RejectReason::EmptyScan;
        finishRecord(record);
        return;
      }
      const RegistrationOutput registration = registrar_->align(*scan.points, record.prediction);
      record.registration = registration.T_map_lidar;
      record.converged = registration.converged;
      record.iterations = registration.iterations;
      record.num_inliers = registration.num_inliers;
      record.final_error = registration.final_error;
      record.runtime_ms = registration.runtime_ms;
      record.downsampled_points = registration.source_downsampled_points;
      record.hessian = registration.hessian;
      if (isFiniteTransform(registration.T_map_lidar)) {
        const Eigen::Isometry3d correction = predictionToRegistrationDelta(
          record.prediction, registration.T_map_lidar);
        const auto correction_rpy = rollPitchYaw(correction.linear());
        record.correction_translation_m = correction.translation().norm();
        record.correction_roll_rad = correction_rpy[0];
        record.correction_pitch_rad = correction_rpy[1];
        record.correction_yaw_rad = correction_rpy[2];
      }
      record.reject_reason = evaluateRegistration(
        registration, record.prediction, quality_settings_);
      record.accepted = record.reject_reason == RejectReason::None;
      if (record.accepted) {
        anchor_prediction_ = association->prediction.T_odom_base;
        anchor_T_map_lidar_ = registration.T_map_lidar;
      }
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "Registration exception at %.9f: %s", timestamp, error.what());
      record.reject_reason = RejectReason::RegistrationException;
    }
    finishRecord(record);
  }

  void finishRecord(const LocalizationRecord& record)
  {
    writer_->write(record);
    ++processed_scans_;
    if (record.accepted) {
      ++accepted_scans_;
    } else {
      ++rejected_scans_;
    }
    if (std::isfinite(record.filter_runtime_ms)) {
      ++filter_timing_associated_scans_;
    }
    if (max_scans_ > 0 && processed_scans_ >= static_cast<std::size_t>(max_scans_)) {
      finished_ = true;
      writeRunSummary();
      RCLCPP_INFO(get_logger(), "Reached max_scans=%d; shutting down", max_scans_);
      rclcpp::shutdown();
    }
  }

  void writeRunSummary()
  {
    if (summary_written_) {
      return;
    }
    std::filesystem::create_directories(results_directory_);
    std::ofstream output(results_directory_ + "/run_summary.json", std::ios::trunc);
    output << std::setprecision(17);
    output << "{\n"
           << "  \"filter_type\": \"" << filter_type_label_ << "\",\n"
           << "  \"initialization_mode\": \"" << initialization_mode_ << "\",\n"
           << "  \"total_clouds_received\": " << total_clouds_received_ << ",\n"
           << "  \"skipped_before_window\": " << skipped_before_window_ << ",\n"
           << "  \"skipped_after_window\": " << skipped_after_window_ << ",\n"
           << "  \"skipped_before_reference\": " << skipped_before_reference_ << ",\n"
           << "  \"processed_scans\": " << processed_scans_ << ",\n"
           << "  \"accepted_scans\": " << accepted_scans_ << ",\n"
           << "  \"rejected_scans\": " << rejected_scans_ << ",\n"
           << "  \"filter_timing_associated_scans\": "
           << filter_timing_associated_scans_ << ",\n"
           << "  \"raw_map_points\": " << raw_map_points_ << ",\n"
           << "  \"target_map_points\": " << target_map_points_ << ",\n"
           << "  \"time_window_enabled\": " << (time_window_.enabled() ? "true" : "false")
           << ",\n"
           << "  \"time_window_origin_timestamp\": " << time_window_.originTimestamp() << ",\n"
           << "  \"time_window_start_offset_sec\": " << time_window_.startOffset() << ",\n"
           << "  \"time_window_end_offset_sec\": " << time_window_.endOffset() << ",\n"
           << "  \"prediction_uses_identity_base_to_lidar_approximation\": "
           << (prediction_approximation_ ? "true" : "false") << ",\n"
           << "  \"prediction_policy\": "
           << "\"accepted_gicp_6dof_plus_ekf_planar_delta\",\n"
           << "  \"output_frame\": \"T_map_lidar\",\n"
           << "  \"map_base_output_available\": "
           << (prediction_approximation_ ? "false" : "true") << "\n"
           << "}\n";
    summary_written_ = true;
  }

  std::string map_path_;
  std::string reference_path_;
  std::string results_directory_;
  std::optional<ReferenceTrajectory> reference_;
  std::string initialization_mode_;
  Eigen::Isometry3d initial_T_map_lidar_{Eigen::Isometry3d::Identity()};
  TimeWindow time_window_;
  double reference_tolerance_{};
  double prediction_tolerance_{};
  double filter_timing_tolerance_{};
  bool require_filter_timing_{false};
  std::string filter_type_label_;
  int max_scans_{};
  std::string expected_lidar_frame_;
  QualityGateSettings quality_settings_;
  Eigen::Isometry3d T_base_lidar_{Eigen::Isometry3d::Identity()};
  bool prediction_approximation_{false};

  std::unique_ptr<MapRegistrar> registrar_;
  std::unique_ptr<ResultWriter> writer_;
  std::deque<TimedPrediction> predictions_;
  std::deque<TimedFilterRuntime> filter_runtimes_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> pending_scans_;
  std::optional<Eigen::Isometry3d> anchor_prediction_;
  Eigen::Isometry3d anchor_T_map_lidar_{Eigen::Isometry3d::Identity()};

  std::size_t total_clouds_received_{0};
  std::size_t skipped_before_window_{0};
  std::size_t skipped_after_window_{0};
  std::size_t skipped_before_reference_{0};
  std::size_t processed_scans_{0};
  std::size_t accepted_scans_{0};
  std::size_t rejected_scans_{0};
  std::size_t filter_timing_associated_scans_{0};
  std::size_t raw_map_points_{0};
  std::size_t target_map_points_{0};
  bool finished_{false};
  bool summary_written_{false};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr prediction_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr
    filter_timing_subscription_;
};

}  // namespace
}  // namespace bunker_offline_localization

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<bunker_offline_localization::OfflineLocalizerNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("offline_localizer"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
