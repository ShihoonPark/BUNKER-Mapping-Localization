#include "bunker_offline_localization/time_window.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace bol = bunker_offline_localization;

TEST(TimeWindow, UsesInclusiveBagRelativeBounds)
{
  const bol::TimeWindow window(true, 1000.0, 5.0, 50.0);
  EXPECT_EQ(window.classify(1004.999), bol::TimeWindowPosition::Before);
  EXPECT_EQ(window.classify(1005.0), bol::TimeWindowPosition::Inside);
  EXPECT_EQ(window.classify(1050.0), bol::TimeWindowPosition::Inside);
  EXPECT_EQ(window.classify(1050.001), bol::TimeWindowPosition::After);
}

TEST(TimeWindow, DisabledWindowAcceptsAllFiniteTimes)
{
  const bol::TimeWindow window;
  EXPECT_EQ(window.classify(-1000.0), bol::TimeWindowPosition::Inside);
  EXPECT_EQ(window.classify(1000000.0), bol::TimeWindowPosition::Inside);
}

TEST(TimeWindow, RejectsInvalidConfiguration)
{
  EXPECT_THROW(bol::TimeWindow(true, 1000.0, -1.0, 50.0), std::invalid_argument);
  EXPECT_THROW(bol::TimeWindow(true, 1000.0, 51.0, 50.0), std::invalid_argument);
  EXPECT_THROW(
    bol::TimeWindow(
      true, std::numeric_limits<double>::quiet_NaN(), 0.0, 50.0),
    std::invalid_argument);
}
