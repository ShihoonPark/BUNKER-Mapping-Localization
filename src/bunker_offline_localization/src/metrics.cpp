#include "bunker_offline_localization/metrics.hpp"

#include "bunker_offline_localization/transforms.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace bunker_offline_localization {

std::string toString(const RejectReason reason)
{
  switch (reason) {
    case RejectReason::None: return "NONE";
    case RejectReason::NotConverged: return "NOT_CONVERGED";
    case RejectReason::LowInliers: return "LOW_INLIERS";
    case RejectReason::HighError: return "HIGH_ERROR";
    case RejectReason::TranslationJump: return "TRANSLATION_JUMP";
    case RejectReason::RotationJump: return "ROTATION_JUMP";
    case RejectReason::NonfiniteTransform: return "NONFINITE_TRANSFORM";
    case RejectReason::NoPrediction: return "NO_PREDICTION";
    case RejectReason::TimestampMismatch: return "TIMESTAMP_MISMATCH";
    case RejectReason::EmptyScan: return "EMPTY_SCAN";
    case RejectReason::RegistrationException: return "REGISTRATION_EXCEPTION";
  }
  return "UNKNOWN";
}

RejectReason evaluateRegistration(
  const RegistrationOutput& registration,
  const Eigen::Isometry3d& prediction,
  const QualityGateSettings& settings)
{
  if (!isFiniteTransform(registration.T_map_lidar) ||
    !registration.hessian.allFinite() || !std::isfinite(registration.final_error))
  {
    return RejectReason::NonfiniteTransform;
  }
  if (!registration.converged) {
    return RejectReason::NotConverged;
  }
  if (registration.num_inliers < settings.min_inliers) {
    return RejectReason::LowInliers;
  }
  if (registration.final_error / static_cast<double>(registration.num_inliers) >
    settings.max_final_error_per_inlier)
  {
    return RejectReason::HighError;
  }

  const Eigen::Isometry3d correction = prediction.inverse() * registration.T_map_lidar;
  if (correction.translation().norm() > settings.max_translation_correction) {
    return RejectReason::TranslationJump;
  }
  if (rotationAngle(correction.linear()) > settings.max_rotation_correction) {
    return RejectReason::RotationJump;
  }
  return RejectReason::None;
}

namespace {

void writePose(std::ostream& stream, const Eigen::Isometry3d& pose)
{
  if (!pose.matrix().allFinite()) {
    for (int index = 0; index < 10; ++index) {
      stream << ",nan";
    }
    return;
  }
  const Eigen::Quaterniond quaternion(pose.linear());
  const auto rpy = rollPitchYaw(pose.linear());
  stream << ',' << pose.translation().x()
         << ',' << pose.translation().y()
         << ',' << pose.translation().z()
         << ',' << quaternion.x()
         << ',' << quaternion.y()
         << ',' << quaternion.z()
         << ',' << quaternion.w()
         << ',' << rpy[0]
         << ',' << rpy[1]
         << ',' << rpy[2];
}

}  // namespace

ResultWriter::ResultWriter(const std::string& results_directory)
: results_directory_(results_directory)
{
  std::filesystem::create_directories(results_directory_);
  csv_.open(results_directory_ + "/localization.csv", std::ios::trunc);
  trajectory_.open(results_directory_ + "/estimated_traj_lidar.tum", std::ios::trunc);
  if (!csv_ || !trajectory_) {
    throw std::runtime_error("Failed to open result files under: " + results_directory_);
  }
  csv_ << std::setprecision(17);
  trajectory_ << std::setprecision(17);
  csv_ << "timestamp"
       << ",pred_x,pred_y,pred_z,pred_qx,pred_qy,pred_qz,pred_qw,pred_roll,pred_pitch,pred_yaw"
       << ",gicp_x,gicp_y,gicp_z,gicp_qx,gicp_qy,gicp_qz,gicp_qw,gicp_roll,gicp_pitch,gicp_yaw"
       << ",prediction_available,prediction_approximation,converged,accepted,reject_reason"
       << ",iterations,num_inliers,final_error,runtime_ms"
       << ",input_points,finite_points,downsampled_points"
       << ",prediction_time_difference,reference_time_difference";
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      csv_ << ",hessian_" << row << column;
    }
  }
  csv_ << '\n';
}

void ResultWriter::write(const LocalizationRecord& record)
{
  csv_ << record.timestamp;
  writePose(csv_, record.prediction);
  writePose(csv_, record.registration);
  csv_ << ',' << static_cast<int>(record.prediction_available)
       << ',' << static_cast<int>(record.prediction_approximation)
       << ',' << static_cast<int>(record.converged)
       << ',' << static_cast<int>(record.accepted)
       << ',' << toString(record.reject_reason)
       << ',' << record.iterations
       << ',' << record.num_inliers
       << ',' << record.final_error
       << ',' << record.runtime_ms
       << ',' << record.input_points
       << ',' << record.finite_points
       << ',' << record.downsampled_points
       << ',' << record.prediction_time_difference
       << ',' << record.reference_time_difference;
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      csv_ << ',' << record.hessian(row, column);
    }
  }
  csv_ << '\n';
  csv_.flush();

  if (record.accepted) {
    const Eigen::Quaterniond quaternion(record.registration.linear());
    trajectory_ << record.timestamp << ' '
                << record.registration.translation().x() << ' '
                << record.registration.translation().y() << ' '
                << record.registration.translation().z() << ' '
                << quaternion.x() << ' ' << quaternion.y() << ' '
                << quaternion.z() << ' ' << quaternion.w() << '\n';
    trajectory_.flush();
  }
}

}  // namespace bunker_offline_localization
