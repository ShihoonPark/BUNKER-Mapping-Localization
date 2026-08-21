#include "bunker_offline_localization/time_window.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace bunker_offline_localization {
namespace {

double stampSeconds(const builtin_interfaces::msg::Time& stamp)
{
  return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

template<std::size_t N>
std::array<double, N * N> diagonalCovariance(const std::vector<double>& diagonal)
{
  if (diagonal.size() != N) {
    throw std::invalid_argument("Covariance diagonal has incorrect length");
  }
  std::array<double, N * N> covariance{};
  for (std::size_t index = 0; index < N; ++index) {
    if (diagonal[index] <= 0.0) {
      throw std::invalid_argument("Configured covariance diagonal values must be positive");
    }
    covariance[index * N + index] = diagonal[index];
  }
  return covariance;
}

class CovarianceAdapterNode : public rclcpp::Node {
public:
  CovarianceAdapterNode()
  : Node("covariance_adapter")
  {
    const auto odom_input = declare_parameter<std::string>("odom_input", "/odom");
    const auto imu_input = declare_parameter<std::string>("imu_input", "/imu/data");
    const auto odom_output = declare_parameter<std::string>(
      "odom_output", "/localization/odom_with_covariance");
    const auto imu_output = declare_parameter<std::string>(
      "imu_output", "/localization/imu_with_covariance");
    time_window_ = TimeWindow(
      declare_parameter<bool>("time_window.enabled", false),
      declare_parameter<double>("time_window.origin_timestamp", 0.0),
      declare_parameter<double>("time_window.start_offset_sec", 0.0),
      declare_parameter<double>("time_window.end_offset_sec", 0.0));

    odom_pose_covariance_ = diagonalCovariance<6>(
      declare_parameter<std::vector<double>>(
        "odom_pose_covariance_diagonal", {0.0225, 0.0225, 1000.0, 1000.0, 1000.0, 0.01}));
    odom_twist_covariance_ = diagonalCovariance<6>(
      declare_parameter<std::vector<double>>(
        "odom_twist_covariance_diagonal", {0.04, 1000.0, 1000.0, 1000.0, 1000.0, 0.04}));
    imu_angular_velocity_covariance_ = diagonalCovariance<3>(
      declare_parameter<std::vector<double>>(
        "imu_angular_velocity_covariance_diagonal", {1.0, 1.0, 0.0009}));

    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>(odom_output, rclcpp::QoS(200));
    imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>(imu_output, rclcpp::QoS(500));
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_input, rclcpp::QoS(500).reliable(),
      [this](nav_msgs::msg::Odometry::ConstSharedPtr input) {
        if (time_window_.classify(stampSeconds(input->header.stamp)) !=
          TimeWindowPosition::Inside)
        {
          return;
        }
        auto output = *input;
        output.pose.covariance = odom_pose_covariance_;
        output.twist.covariance = odom_twist_covariance_;
        odom_publisher_->publish(output);
      });
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_input, rclcpp::QoS(1000).reliable(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr input) {
        if (time_window_.classify(stampSeconds(input->header.stamp)) !=
          TimeWindowPosition::Inside)
        {
          return;
        }
        auto output = *input;
        // Bag orientation is fixed identity for all 12,555 messages. Mark it unavailable and
        // never present it to robot_localization as an orientation measurement.
        output.orientation_covariance.fill(0.0);
        output.orientation_covariance[0] = -1.0;
        output.angular_velocity_covariance = imu_angular_velocity_covariance_;
        // Linear acceleration is intentionally not fused in the Phase 1 EKF configuration.
        output.linear_acceleration_covariance.fill(0.0);
        output.linear_acceleration_covariance[0] = -1.0;
        imu_publisher_->publish(output);
      });

    RCLCPP_WARN(
      get_logger(),
      "Applying configurable initial-tuning covariance assumptions on separate topics; "
      "original /odom and /imu/data are unchanged");
  }

private:
  std::array<double, 36> odom_pose_covariance_{};
  std::array<double, 36> odom_twist_covariance_{};
  std::array<double, 9> imu_angular_velocity_covariance_{};
  TimeWindow time_window_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
};

}  // namespace
}  // namespace bunker_offline_localization

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
      std::make_shared<bunker_offline_localization::CovarianceAdapterNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("covariance_adapter"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
