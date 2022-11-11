/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

enum class DateExtractField {
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
  kDayOfWeek,
  kIsoDayOfWeek,
  kDayOfYear,
  kEpoch,
  kQuarterDay,
  kWeek,
  kWeekSunday,
  kWeekSaturday,
  kDateEpoch
};

}  // namespace hdk::ir
