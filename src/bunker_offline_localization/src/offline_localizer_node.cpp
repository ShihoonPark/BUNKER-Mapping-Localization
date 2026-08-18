#include "bunker_offline_localization/map_loader.hpp"
#include "bunker_offline_localization/metrics.hpp"
#include "bunker_offline_localization/reference_trajectory.hpp"
#include "bunker_offline_localization/registration.hpp"
#include "bunker_offline_localization/scan_preprocessor.hpp"
#include "bunker_offline_localization/transforms.hpp"

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
        "/home/a/Desktop/shihoon/bunker_localization_ws/results")),
    reference_(ReferenceTrajectory::load(reference_path_))
  {
    reference_tolerance_ = declare_parameter<double>("reference_timestamp_tolerance", 0.06);
    prediction_tolerance_ = declare_parameter<double>("prediction_timestamp_tolerance", 0.10);
    max_scans_ = declare_parameter<int>("max_scans", 0);
    if (reference_tolerance_ <= 0.0 || prediction_tolerance_ <= 0.0 || max_scans_ < 0) {
      throw std::invalid_argument("Timestamp tolerances must be positive and max_scans nonnegative");
    }

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

    quality_settings_.min_inliers = static_cast<std::size_t>(
      declare_parameter<int>("min_inliers", 100));
    quality_settings_.max_final_error_per_inlier =
      declare_parameter<double>("max_final_error_per_inlier", 5.0);
    quality_settings_.max_translation_correction =
      declare_parameter<double>("max_translation_correction", 1.0);
    quality_settings_.max_rotation_correction =
      declare_parameter<double>("max_rotation_correction", 0.5235987755982988);

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
    expected_lidar_frame_ = declare_parameter<std::string>("expected_lidar_frame", "velodyne");

    cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, rclcpp::QoS(50).reliable(),
      std::bind(&OfflineLocalizerNode::cloudCallback, this, std::placeholders::_1));
    prediction_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      prediction_topic, rclcpp::QoS(500),
      std::bind(&OfflineLocalizerNode::predictionCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Prepared global target map once: raw=%zu downsampled=%zu; first reference=%.9f",
      raw_map_points_, target_map_points_, reference_.first().timestamp);
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
    // GLIM and PointCloud2 stamps differ by sub-millisecond serialization/frame timing. Treat
    // the scan associated with the first reference pose as the first usable scan; only scans
    // earlier than the configured association window are unconditionally skipped.
    if (timestamp < reference_.first().timestamp - reference_tolerance_) {
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
      predictions_.push_back(TimedPrediction{timestamp, odometryPose(*message)});
      while (predictions_.size() > 2000U) {
        predictions_.pop_front();
      }
      drainPending(false);
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(), "Ignoring invalid EKF prediction: %s", error.what());
    }
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

  void drainPending(const bool final)
  {
    while (!pending_scans_.empty() && !finished_) {
      const double timestamp = stampSeconds(pending_scans_.front()->header.stamp);
      if (!final && (predictions_.empty() || predictions_.back().timestamp < timestamp)) {
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
    record.prediction_approximation = prediction_approximation_;

    const auto reference_association = reference_.associateNearest(
      timestamp, reference_tolerance_);
    if (reference_association) {
      record.reference_time_difference = reference_association->absolute_time_difference;
    }

    if (!association) {
      record.reject_reason = RejectReason::NoPrediction;
      finishRecord(record);
      return;
    }
    record.prediction_time_difference = association->time_difference;
    if (association->time_difference > prediction_tolerance_) {
      record.reject_reason = RejectReason::TimestampMismatch;
      finishRecord(record);
      return;
    }

    if (!anchor_prediction_) {
      const double first_reference_difference =
        std::abs(timestamp - reference_.first().timestamp);
      if (!reference_association || first_reference_difference > reference_tolerance_) {
        record.reject_reason = RejectReason::TimestampMismatch;
        finishRecord(record);
        return;
      }
      anchor_prediction_ = association->prediction.T_odom_base;
      anchor_T_map_lidar_ = reference_.first().T_map_lidar;
    }

    // p_base = T_base_lidar*p_lidar. Propagate relative base motion from the last accepted
    // correction anchor, then convert back to a LiDAR pose.
    const Eigen::Isometry3d T_anchor_base_current =
      anchor_prediction_->inverse() * association->prediction.T_odom_base;
    record.prediction = anchor_T_map_lidar_ * T_base_lidar_.inverse() *
      T_anchor_base_current * T_base_lidar_;
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
    output << "{\n"
           << "  \"total_clouds_received\": " << total_clouds_received_ << ",\n"
           << "  \"skipped_before_reference\": " << skipped_before_reference_ << ",\n"
           << "  \"processed_scans\": " << processed_scans_ << ",\n"
           << "  \"accepted_scans\": " << accepted_scans_ << ",\n"
           << "  \"rejected_scans\": " << rejected_scans_ << ",\n"
           << "  \"raw_map_points\": " << raw_map_points_ << ",\n"
           << "  \"target_map_points\": " << target_map_points_ << ",\n"
           << "  \"prediction_uses_identity_base_to_lidar_approximation\": "
           << (prediction_approximation_ ? "true" : "false") << "\n"
           << "}\n";
    summary_written_ = true;
  }

  std::string map_path_;
  std::string reference_path_;
  std::string results_directory_;
  ReferenceTrajectory reference_;
  double reference_tolerance_{};
  double prediction_tolerance_{};
  int max_scans_{};
  std::string expected_lidar_frame_;
  QualityGateSettings quality_settings_;
  Eigen::Isometry3d T_base_lidar_{Eigen::Isometry3d::Identity()};
  bool prediction_approximation_{false};

  std::unique_ptr<MapRegistrar> registrar_;
  std::unique_ptr<ResultWriter> writer_;
  std::deque<TimedPrediction> predictions_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> pending_scans_;
  std::optional<Eigen::Isometry3d> anchor_prediction_;
  Eigen::Isometry3d anchor_T_map_lidar_{Eigen::Isometry3d::Identity()};

  std::size_t total_clouds_received_{0};
  std::size_t skipped_before_reference_{0};
  std::size_t processed_scans_{0};
  std::size_t accepted_scans_{0};
  std::size_t rejected_scans_{0};
  std::size_t raw_map_points_{0};
  std::size_t target_map_points_{0};
  bool finished_{false};
  bool summary_written_{false};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr prediction_subscription_;
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
