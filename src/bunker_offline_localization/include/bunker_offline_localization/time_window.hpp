#pragma once

#include <cmath>
#include <stdexcept>

namespace bunker_offline_localization {

enum class TimeWindowPosition {Before, Inside, After};

// Inclusive bag-relative time window. The origin is the rosbag metadata starting timestamp,
// not the first message timestamp of an individual sensor.
class TimeWindow {
public:
  TimeWindow() = default;

  TimeWindow(
    const bool enabled, const double origin_timestamp,
    const double start_offset, const double end_offset)
  : enabled_(enabled),
    origin_timestamp_(origin_timestamp),
    start_offset_(start_offset),
    end_offset_(end_offset)
  {
    if (enabled_ &&
      (!std::isfinite(origin_timestamp_) || !std::isfinite(start_offset_) ||
      !std::isfinite(end_offset_) || origin_timestamp_ <= 0.0 || start_offset_ < 0.0 ||
      end_offset_ < start_offset_))
    {
      throw std::invalid_argument("Invalid bag-relative time window");
    }
  }

  TimeWindowPosition classify(const double timestamp) const
  {
    if (!enabled_) {
      return TimeWindowPosition::Inside;
    }
    if (!std::isfinite(timestamp)) {
      return TimeWindowPosition::After;
    }
    const double offset = timestamp - origin_timestamp_;
    if (offset < start_offset_) {
      return TimeWindowPosition::Before;
    }
    if (offset > end_offset_) {
      return TimeWindowPosition::After;
    }
    return TimeWindowPosition::Inside;
  }

  bool enabled() const {return enabled_;}
  double originTimestamp() const {return origin_timestamp_;}
  double startOffset() const {return start_offset_;}
  double endOffset() const {return end_offset_;}

private:
  bool enabled_{false};
  double origin_timestamp_{0.0};
  double start_offset_{0.0};
  double end_offset_{0.0};
};

}  // namespace bunker_offline_localization
