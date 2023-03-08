/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef QUERYENGINE_EXTRACTSTRINGFROMTIME_H
#define QUERYENGINE_EXTRACTSTRINGFROMTIME_H

#include "ExtractFromTime.h"

#include <cmath>
#include <iomanip>
#include <sstream>

std::string getStrDayFromSeconds(const int64_t seconds_tstamp) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << extract_day(seconds_tstamp);
  return oss.str();
}

std::string getStrMonthFromSeconds(const int64_t seconds_tstamp) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << extract_month_fast(seconds_tstamp);
  return oss.str();
}

std::string getStrYearFromSeconds(const int64_t seconds_tstamp) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << extract_year_fast(seconds_tstamp);
  return oss.str();
}

std::string getStrDateFromSeconds(const int64_t seconds_tstamp) {
  std::ostringstream oss;
  oss << getStrYearFromSeconds(seconds_tstamp) << "-"
      << getStrMonthFromSeconds(seconds_tstamp) << "-"
      << getStrDayFromSeconds(seconds_tstamp);
  return oss.str();
}

std::string getStrTimeFromSeconds(const int64_t seconds_tstamp) {
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << extract_hour(seconds_tstamp) << ":"
      << std::setfill('0') << std::setw(2) << extract_minute(seconds_tstamp) << ":"
      << std::setfill('0') << std::setw(2) << extract_second(seconds_tstamp);
  return oss.str();
}

std::string getStrTimeStampSecondsScaled(const int64_t tstamp,
                                         const int64_t seconds_scale) {
  std::ostringstream oss;
  oss << getStrDateFromSeconds(tstamp / seconds_scale) << " "
      << getStrTimeFromSeconds(tstamp / seconds_scale);
  if (seconds_scale > 1) {
    oss << "." << std::setfill('0') << std::setw(log10(seconds_scale))
        << tstamp % seconds_scale;
  }
  return oss.str();
}

std::string getStrTStamp(const int64_t tstamp, hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kSecond:
      return getStrTimeStampSecondsScaled(tstamp);
    case hdk::ir::TimeUnit::kMilli:
      return getStrTimeStampSecondsScaled(tstamp, kMilliSecsPerSec);
    case hdk::ir::TimeUnit::kMicro:
      return getStrTimeStampSecondsScaled(tstamp, kMicroSecsPerSec);
    case hdk::ir::TimeUnit::kNano:
      return getStrTimeStampSecondsScaled(tstamp, kNanoSecsPerSec);
    case hdk::ir::TimeUnit::kMonth:
      return getStrMonthFromSeconds(tstamp);
    case hdk::ir::TimeUnit::kDay:
      return getStrDayFromSeconds(tstamp);
  }
}

#endif  // QUERYENGINE_EXTRACTSTRINGFROMTIME_H
