#include "bunker_offline_localization/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bunker_offline_localization {

Eigen::Quaterniond normalizedQuaternion(
  const double qx, const double qy, const double qz, const double qw)
{
  Eigen::Quaterniond quaternion(qw, qx, qy, qz);
  if (!quaternion.coeffs().allFinite() || quaternion.norm() < 1.0e-12) {
    throw std::invalid_argument("Quaternion is non-finite or has near-zero norm");
  }
  quaternion.normalize();
  return quaternion;
}

Eigen::Isometry3d makeTransform(
  const double x, const double y, const double z,
  const double qx, const double qy, const double qz, const double qw)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    throw std::invalid_argument("Translation is non-finite");
  }
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = normalizedQuaternion(qx, qy, qz, qw).toRotationMatrix();
  transform.translation() = Eigen::Vector3d(x, y, z);
  return transform;
}

bool isFiniteTransform(const Eigen::Isometry3d& transform)
{
  if (!transform.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d should_be_identity = transform.linear().transpose() * transform.linear();
  return should_be_identity.isApprox(Eigen::Matrix3d::Identity(), 1.0e-6) &&
         std::abs(transform.linear().determinant() - 1.0) < 1.0e-6;
}

double rotationAngle(const Eigen::Matrix3d& rotation)
{
  const double cosine = std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine);
}

double wrapAngle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

std::array<double, 3> rollPitchYaw(const Eigen::Matrix3d& rotation)
{
  const Eigen::Vector3d rpy = rotation.eulerAngles(2, 1, 0).reverse();
  return {wrapAngle(rpy.x()), wrapAngle(rpy.y()), wrapAngle(rpy.z())};
}

}  // namespace bunker_offline_localization
