/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>

namespace hdk::ir {

enum class TimeUnit {
  kMonth,
  kDay,
  kSecond,
  kMilli,
  kMicro,
  kNano,
};

enum class DateAddField {
  kYear,
  kQuarter,
  kMonth,
  kDay,
  kHour,
  kMinute,
  kSecond,
  kMillennium,
  kCentury,
  kDecade,
  kMilli,
  kMicro,
  kNano,
  kWeek,
  kQuarterDay,
  kWeekDay,
  kDayOfYear,
  kInvalid
};

enum class DateTruncField {
  kYear,
  kQuarter,
  kMonth,
  kDay,
  kHour,
  kMinute,
  kSecond,
  kMilli,
  kMicro,
  kNano,
  kMillennium,
  kCentury,
  kDecade,
  kWeek,
  kWeekSunday,
  kWeekSaturday,
  kQuarterDay,
  kInvalid
};

}  // namespace hdk::ir

inline std::string toString(hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kMonth:
      return "Month";
    case hdk::ir::TimeUnit::kDay:
      return "Day";
    case hdk::ir::TimeUnit::kSecond:
      return "Second";
    case hdk::ir::TimeUnit::kMilli:
      return "Milli";
    case hdk::ir::TimeUnit::kMicro:
      return "Micro";
    case hdk::ir::TimeUnit::kNano:
      return "Nano";
    default:
      return "InvalidTimeUnit";
  }
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::TimeUnit unit) {
  os << toString(unit);
  return os;
}

inline std::string toString(hdk::ir::DateAddField field) {
  switch (field) {
    case hdk::ir::DateAddField::kYear:
      return "Year";
    case hdk::ir::DateAddField::kQuarter:
      return "Quarter";
    case hdk::ir::DateAddField::kMonth:
      return "Month";
    case hdk::ir::DateAddField::kDay:
      return "Day";
    case hdk::ir::DateAddField::kHour:
      return "Hour";
    case hdk::ir::DateAddField::kMinute:
      return "Minute";
    case hdk::ir::DateAddField::kSecond:
      return "Second";
    case hdk::ir::DateAddField::kMillennium:
      return "Millennium";
    case hdk::ir::DateAddField::kCentury:
      return "Century";
    case hdk::ir::DateAddField::kDecade:
      return "Decade";
    case hdk::ir::DateAddField::kMilli:
      return "Milli";
    case hdk::ir::DateAddField::kMicro:
      return "Micro";
    case hdk::ir::DateAddField::kNano:
      return "Nano";
    case hdk::ir::DateAddField::kWeek:
      return "Week";
    case hdk::ir::DateAddField::kQuarterDay:
      return "QuarterDay";
    case hdk::ir::DateAddField::kWeekDay:
      return "WeekDay";
    case hdk::ir::DateAddField::kDayOfYear:
      return "DayOfYear";
    case hdk::ir::DateAddField::kInvalid:
      return "Invalid";
    default:
      return "InvalidDateAddField";
  }
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::DateAddField field) {
  os << toString(field);
  return os;
}

inline std::string toString(hdk::ir::DateTruncField field) {
  switch (field) {
    case hdk::ir::DateTruncField::kYear:
      return "Year";
    case hdk::ir::DateTruncField::kQuarter:
      return "Quarter";
    case hdk::ir::DateTruncField::kMonth:
      return "Month";
    case hdk::ir::DateTruncField::kDay:
      return "Day";
    case hdk::ir::DateTruncField::kHour:
      return "Hour";
    case hdk::ir::DateTruncField::kMinute:
      return "Minute";
    case hdk::ir::DateTruncField::kSecond:
      return "Second";
    case hdk::ir::DateTruncField::kMilli:
      return "Milli";
    case hdk::ir::DateTruncField::kMicro:
      return "Micro";
    case hdk::ir::DateTruncField::kNano:
      return "Nano";
    case hdk::ir::DateTruncField::kMillennium:
      return "Millennium";
    case hdk::ir::DateTruncField::kCentury:
      return "Century";
    case hdk::ir::DateTruncField::kDecade:
      return "Decade";
    case hdk::ir::DateTruncField::kWeek:
      return "Week";
    case hdk::ir::DateTruncField::kWeekSunday:
      return "WeekSunday";
    case hdk::ir::DateTruncField::kWeekSaturday:
      return "WeekSaturday";
    case hdk::ir::DateTruncField::kQuarterDay:
      return "QuarterDay";
    case hdk::ir::DateTruncField::kInvalid:
      return "Invalid";
    default:
      return "InvalidDateTruncField";
  }
}

inline std::ostream& operator<<(std::ostream& os, hdk::ir::DateTruncField field) {
  os << toString(field);
  return os;
}

namespace hdk::ir {

inline int64_t unitsPerSecond(TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kSecond:
      return 1;
    case hdk::ir::TimeUnit::kMilli:
      return 1'000;
    case hdk::ir::TimeUnit::kMicro:
      return 1'000'000;
    case hdk::ir::TimeUnit::kNano:
      return 1'000'000'000;
    default:
      throw std::runtime_error("Enexpected unit in unitsPerSecond: " + ::toString(unit));
  }
}

}  // namespace hdk::ir
