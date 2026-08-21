#include "bunker_offline_localization/scan_preprocessor.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bunker_offline_localization {
namespace {

const sensor_msgs::msg::PointField& findField(
  const sensor_msgs::msg::PointCloud2& message, const std::string& name)
{
  const auto field = std::find_if(
    message.fields.begin(), message.fields.end(),
    [&name](const sensor_msgs::msg::PointField& item) {return item.name == name;});
  if (field == message.fields.end()) {
    throw std::runtime_error("PointCloud2 has no '" + name + "' field");
  }
  if (field->count != 1U ||
    (field->datatype != sensor_msgs::msg::PointField::FLOAT32 &&
    field->datatype != sensor_msgs::msg::PointField::FLOAT64))
  {
    throw std::runtime_error("PointCloud2 field '" + name + "' must be scalar FLOAT32/FLOAT64");
  }
  return *field;
}

template<typename T>
T byteSwap(T value)
{
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  std::reverse(bytes.begin(), bytes.end());
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

double readScalar(
  const std::uint8_t* data, const sensor_msgs::msg::PointField& field, const bool big_endian)
{
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    float value;
    std::memcpy(&value, data + field.offset, sizeof(value));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    if (big_endian) {
      value = byteSwap(value);
    }
#else
    if (!big_endian) {
      value = byteSwap(value);
    }
#endif
    return value;
  }
  double value;
  std::memcpy(&value, data + field.offset, sizeof(value));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  if (big_endian) {
    value = byteSwap(value);
  }
#else
  if (!big_endian) {
    value = byteSwap(value);
  }
#endif
  return value;
}

}  // namespace

ExtractedScan extractFiniteXYZ(const sensor_msgs::msg::PointCloud2& message)
{
  const auto& x_field = findField(message, "x");
  const auto& y_field = findField(message, "y");
  const auto& z_field = findField(message, "z");
  const std::uint32_t required_step = std::max({
      x_field.offset + (x_field.datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4U : 8U),
      y_field.offset + (y_field.datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4U : 8U),
      z_field.offset + (z_field.datatype == sensor_msgs::msg::PointField::FLOAT32 ? 4U : 8U)});
  if (message.point_step < required_step ||
    message.row_step < message.point_step * message.width ||
    message.data.size() < static_cast<std::size_t>(message.row_step) * message.height)
  {
    throw std::runtime_error("PointCloud2 layout is inconsistent with its data buffer");
  }

  const std::size_t input_count = static_cast<std::size_t>(message.width) * message.height;
  std::vector<Eigen::Vector4d> points;
  points.reserve(input_count);
  for (std::uint32_t row = 0; row < message.height; ++row) {
    const std::uint8_t* row_data = message.data.data() + row * message.row_step;
    for (std::uint32_t column = 0; column < message.width; ++column) {
      const std::uint8_t* data = row_data + column * message.point_step;
      const double x = readScalar(data, x_field, message.is_bigendian);
      const double y = readScalar(data, y_field, message.is_bigendian);
      const double z = readScalar(data, z_field, message.is_bigendian);
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        points.emplace_back(x, y, z, 1.0);
      }
    }
  }
  return ExtractedScan{
    std::make_shared<small_gicp::PointCloud>(points), input_count, points.size()};
}

}  // namespace bunker_offline_localization
