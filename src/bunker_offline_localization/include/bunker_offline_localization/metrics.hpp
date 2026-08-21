#pragma once

#include "bunker_offline_localization/registration.hpp"

#include <Eigen/Geometry>

#include <cstddef>
#include <fstream>
#include <string>

namespace bunker_offline_localization {

enum class RejectReason {
  None,
  NotConverged,
  LowInliers,
  HighError,
  TranslationJump,
  RotationJump,
  NonfiniteTransform,
  NoPrediction,
  TimestampMismatch,
  EmptyScan,
  RegistrationException,
};

std::string toString(RejectReason reason);

struct QualityGateSettings {
  std::size_t min_inliers{100};
  double max_final_error_per_inlier{5.0};
  double max_translation_correction{1.0};
  double max_rotation_correction{0.5235987755982988};
};

Eigen::Isometry3d predictionToRegistrationDelta(
  const Eigen::Isometry3d& prediction,
  const Eigen::Isometry3d& registration);

RejectReason evaluateRegistration(
  const RegistrationOutput& registration,
  const Eigen::Isometry3d& prediction,
  const QualityGateSettings& settings);

struct LocalizationRecord {
  double timestamp{};
  Eigen::Isometry3d prediction{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d registration{Eigen::Isometry3d::Identity()};
  bool prediction_available{false};
  bool prediction_approximation{false};
  bool converged{false};
  bool accepted{false};
  RejectReason reject_reason{RejectReason::None};
  std::size_t iterations{0};
  std::size_t num_inliers{0};
  double final_error{0.0};
  double runtime_ms{0.0};
  double core_localization_latency_ms{0.0};
  double filter_runtime_ms{0.0};
  double filter_runtime_time_difference{0.0};
  double correction_translation_m{0.0};
  double correction_roll_rad{0.0};
  double correction_pitch_rad{0.0};
  double correction_yaw_rad{0.0};
  std::size_t input_points{0};
  std::size_t finite_points{0};
  std::size_t downsampled_points{0};
  double prediction_time_difference{0.0};
  double reference_time_difference{0.0};
  Eigen::Matrix<double, 6, 6> hessian{Eigen::Matrix<double, 6, 6>::Zero()};
};

class LatencyWriter {
public:
  LatencyWriter(const std::string& results_directory, double origin_timestamp);
  void write(const LocalizationRecord& record, std::size_t processed_index);

private:
  double origin_timestamp_{};
  std::ofstream csv_;
};

class ResultWriter {
public:
  explicit ResultWriter(const std::string& results_directory);
  void write(const LocalizationRecord& record);
  const std::string& resultsDirectory() const { return results_directory_; }

private:
  std::string results_directory_;
  std::ofstream csv_;
  std::ofstream trajectory_;
};

}  // namespace bunker_offline_localization
