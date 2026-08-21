#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>

namespace bunker_offline_localization {
namespace {

double stampSeconds(const builtin_interfaces::msg::Time& stamp)
{
  return static_cast<double>(stamp.sec) + 1.0e-9 * static_cast<double>(stamp.nanosec);
}

struct InputArrival {
  double timestamp{};
  std::chrono::steady_clock::time_point arrival;
  double source_id{};
};

class FilterTimingMonitorNode : public rclcpp::Node {
public:
  FilterTimingMonitorNode()
  : Node("filter_timing_monitor")
  {
    const auto odom_input = declare_parameter<std::string>(
      "odom_input", "/localization/odom_with_covariance");
    const auto imu_input = declare_parameter<std::string>(
      "imu_input", "/localization/imu_with_covariance");
    const auto filter_output = declare_parameter<std::string>(
      "filter_output", "/localization/odometry/filtered");
    const auto timing_output = declare_parameter<std::string>(
      "timing_output", "/localization/filter_timing");
    association_tolerance_ = declare_parameter<double>("association_tolerance", 0.05);
    if (association_tolerance_ <= 0.0) {
      throw std::invalid_argument("association_tolerance must be positive");
    }

    timing_publisher_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
      timing_output, rclcpp::QoS(500));
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_input, rclcpp::QoS(500).reliable(),
      [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
        recordInput(stampSeconds(message->header.stamp), 0.0);
      });
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_input, rclcpp::QoS(1000).reliable(),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {
        recordInput(stampSeconds(message->header.stamp), 1.0);
      });
    output_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      filter_output, rclcpp::QoS(500),
      std::bind(&FilterTimingMonitorNode::outputCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Measuring sensor-input to filtered-odometry steady-clock latency; this is an end-to-end "
      "runtime proxy, not internal filter CPU time");
  }

private:
  void recordInput(const double timestamp, const double source_id)
  {
    if (!std::isfinite(timestamp)) {
      return;
    }
    input_arrivals_.push_back(InputArrival{
      timestamp, std::chrono::steady_clock::now(), source_id});
    while (input_arrivals_.size() > 5000U) {
      input_arrivals_.pop_front();
    }
  }

  void outputCallback(const nav_msgs::msg::Odometry::ConstSharedPtr message)
  {
    const double output_timestamp = stampSeconds(message->header.stamp);
    if (!std::isfinite(output_timestamp) || input_arrivals_.empty()) {
      return;
    }
    const auto nearest = std::min_element(
      input_arrivals_.begin(), input_arrivals_.end(),
      [output_timestamp](const InputArrival& lhs, const InputArrival& rhs) {
        return std::abs(lhs.timestamp - output_timestamp) <
               std::abs(rhs.timestamp - output_timestamp);
      });
    const double timestamp_difference = std::abs(nearest->timestamp - output_timestamp);
    if (timestamp_difference > association_tolerance_) {
      return;
    }
    const double latency_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - nearest->arrival).count();
    if (!std::isfinite(latency_ms) || latency_ms < 0.0) {
      return;
    }

    geometry_msgs::msg::Vector3Stamped timing;
    timing.header = message->header;
    timing.vector.x = latency_ms;
    timing.vector.y = timestamp_difference;
    timing.vector.z = nearest->source_id;
    timing_publisher_->publish(timing);

    while (!input_arrivals_.empty() &&
      input_arrivals_.front().timestamp < output_timestamp - 1.0)
    {
      input_arrivals_.pop_front();
    }
  }

  double association_tolerance_{};
  std::deque<InputArrival> input_arrivals_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr timing_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr output_subscription_;
};

}  // namespace
}  // namespace bunker_offline_localization

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
      std::make_shared<bunker_offline_localization::FilterTimingMonitorNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("filter_timing_monitor"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
