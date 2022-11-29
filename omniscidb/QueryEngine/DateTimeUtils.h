/*
 * Copyright 2018 OmniSci, Inc.
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

#pragma once

#include "DateTruncate.h"

#include "IR/OpType.h"
#include "IR/Type.h"

#include <cstdint>
#include <ctime>
#include <map>
#include <string>

#include "../Shared/sqldefs.h"

namespace {

static const std::map<std::pair<int32_t, hdk::ir::DateExtractField>,
                      std::pair<hdk::ir::OpType, int64_t>>
    orig_extract_precision_lookup = {{{3, hdk::ir::DateExtractField::kMicro},
                                      {hdk::ir::OpType::kMul, kMilliSecsPerSec}},
                                     {{3, hdk::ir::DateExtractField::kNano},
                                      {hdk::ir::OpType::kMul, kMicroSecsPerSec}},
                                     {{6, hdk::ir::DateExtractField::kMilli},
                                      {hdk::ir::OpType::kDiv, kMilliSecsPerSec}},
                                     {{6, hdk::ir::DateExtractField::kNano},
                                      {hdk::ir::OpType::kMul, kMilliSecsPerSec}},
                                     {{9, hdk::ir::DateExtractField::kMilli},
                                      {hdk::ir::OpType::kDiv, kMicroSecsPerSec}},
                                     {{9, hdk::ir::DateExtractField::kMicro},
                                      {hdk::ir::OpType::kDiv, kMilliSecsPerSec}}};

static const std::map<std::pair<hdk::ir::TimeUnit, hdk::ir::DateExtractField>,
                      std::pair<hdk::ir::OpType, int64_t>>
    extract_precision_lookup = {
        {{hdk::ir::TimeUnit::kMilli, hdk::ir::DateExtractField::kMicro},
         {hdk::ir::OpType::kMul, kMilliSecsPerSec}},
        {{hdk::ir::TimeUnit::kMilli, hdk::ir::DateExtractField::kNano},
         {hdk::ir::OpType::kMul, kMicroSecsPerSec}},
        {{hdk::ir::TimeUnit::kMicro, hdk::ir::DateExtractField::kMilli},
         {hdk::ir::OpType::kDiv, kMilliSecsPerSec}},
        {{hdk::ir::TimeUnit::kMicro, hdk::ir::DateExtractField::kNano},
         {hdk::ir::OpType::kMul, kMilliSecsPerSec}},
        {{hdk::ir::TimeUnit::kNano, hdk::ir::DateExtractField::kMilli},
         {hdk::ir::OpType::kDiv, kMicroSecsPerSec}},
        {{hdk::ir::TimeUnit::kNano, hdk::ir::DateExtractField::kMicro},
         {hdk::ir::OpType::kDiv, kMilliSecsPerSec}}};

static const std::map<std::pair<hdk::ir::TimeUnit, hdk::ir::DateTruncField>, int64_t>
    datetrunc_precision_lookup = {
        {{hdk::ir::TimeUnit::kMicro, hdk::ir::DateTruncField::kMilli}, kMilliSecsPerSec},
        {{hdk::ir::TimeUnit::kNano, hdk::ir::DateTruncField::kMicro}, kMilliSecsPerSec},
        {{hdk::ir::TimeUnit::kNano, hdk::ir::DateTruncField::kMilli}, kMicroSecsPerSec}};

}  // namespace

namespace DateTimeUtils {

// Enum helpers for precision scaling up/down.
enum ScalingType { ScaleUp, ScaleDown };

constexpr inline int64_t get_timestamp_precision_scale(const int32_t dimen) {
  switch (dimen) {
    case 0:
      return 1;
    case 3:
      return kMilliSecsPerSec;
    case 6:
      return kMicroSecsPerSec;
    case 9:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
  }
  return -1;
}

constexpr inline int64_t get_dateadd_timestamp_precision_scale(
    const hdk::ir::DateAddField field) {
  switch (field) {
    case hdk::ir::DateAddField::kMilli:
      return kMilliSecsPerSec;
    case hdk::ir::DateAddField::kMicro:
      return kMicroSecsPerSec;
    case hdk::ir::DateAddField::kNano:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown field = " + toString(field));
  }
  return -1;
}

constexpr inline int64_t get_extract_timestamp_precision_scale(
    const hdk::ir::DateExtractField field) {
  switch (field) {
    case hdk::ir::DateExtractField::kMilli:
      return kMilliSecsPerSec;
    case hdk::ir::DateExtractField::kMicro:
      return kMicroSecsPerSec;
    case hdk::ir::DateExtractField::kNano:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown field = " + toString(field));
  }
  return -1;
}

constexpr inline bool is_subsecond_extract_field(const hdk::ir::DateExtractField& field) {
  return field == hdk::ir::DateExtractField::kMilli ||
         field == hdk::ir::DateExtractField::kMicro ||
         field == hdk::ir::DateExtractField::kNano;
}

constexpr inline bool is_subsecond_dateadd_field(const hdk::ir::DateAddField field) {
  return field == hdk::ir::DateAddField::kMilli ||
         field == hdk::ir::DateAddField::kMicro || field == hdk::ir::DateAddField::kNano;
}

constexpr inline bool is_subsecond_datetrunc_field(const hdk::ir::DateTruncField field) {
  return field == hdk::ir::DateTruncField::kMilli ||
         field == hdk::ir::DateTruncField::kMicro ||
         field == hdk::ir::DateTruncField::kNano;
}

const inline std::pair<hdk::ir::OpType, int64_t>
get_dateadd_high_precision_adjusted_scale(const hdk::ir::DateAddField field,
                                          int32_t dimen) {
  switch (field) {
    case hdk::ir::DateAddField::kNano:
      switch (dimen) {
        case 9:
          return {};
        case 6:
          return {hdk::ir::OpType::kDiv, kMilliSecsPerSec};
        case 3:
          return {hdk::ir::OpType::kDiv, kMicroSecsPerSec};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    case hdk::ir::DateAddField::kMicro:
      switch (dimen) {
        case 9:
          return {hdk::ir::OpType::kMul, kMilliSecsPerSec};
        case 6:
          return {};
        case 3:
          return {hdk::ir::OpType::kDiv, kMilliSecsPerSec};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    case hdk::ir::DateAddField::kMilli:
      switch (dimen) {
        case 9:
          return {hdk::ir::OpType::kMul, kMicroSecsPerSec};
        case 6:
          return {hdk::ir::OpType::kMul, kMilliSecsPerSec};
        case 3:
          return {};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    default:
      throw std::runtime_error("Unknown field = " + toString(field));
  }
  return {};
}

const inline std::pair<hdk::ir::OpType, int64_t>
get_extract_high_precision_adjusted_scale(const hdk::ir::DateExtractField& field,
                                          const hdk::ir::TimeUnit unit) {
  const auto result = extract_precision_lookup.find(std::make_pair(unit, field));
  if (result != extract_precision_lookup.end()) {
    return result->second;
  }
  return {};
}

const inline int64_t get_datetrunc_high_precision_scale(
    const hdk::ir::DateTruncField& field,
    const hdk::ir::TimeUnit unit) {
  const auto result = datetrunc_precision_lookup.find(std::make_pair(unit, field));
  if (result != datetrunc_precision_lookup.end()) {
    return result->second;
  }
  return -1;
}

constexpr inline int64_t get_datetime_scaled_epoch(const ScalingType direction,
                                                   const int64_t epoch,
                                                   const int32_t dimen) {
  switch (direction) {
    case ScaleUp: {
      const auto scaled_epoch = epoch * get_timestamp_precision_scale(dimen);
      if (epoch && epoch != scaled_epoch / get_timestamp_precision_scale(dimen)) {
        throw std::runtime_error(
            "Value Overflow/underflow detected while scaling DateTime precision.");
      }
      return scaled_epoch;
    }
    case ScaleDown:
      return epoch / get_timestamp_precision_scale(dimen);
    default:
      abort();
  }
  return std::numeric_limits<int64_t>::min();
}

constexpr inline int64_t get_nanosecs_in_unit(hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kDay:
      return 86'400'000'000'000;
    case hdk::ir::TimeUnit::kSecond:
      return 1'000'000'000;
    case hdk::ir::TimeUnit::kMilli:
      return 1'000'000;
    case hdk::ir::TimeUnit::kMicro:
      return 1'000;
    case hdk::ir::TimeUnit::kNano:
      return 1;
    default:
      throw std::runtime_error("Unexpected time unit: " + toString(unit));
  }
  return -1;
}

constexpr inline int64_t get_datetime_scaled_epoch(int64_t epoch,
                                                   hdk::ir::TimeUnit old_unit,
                                                   hdk::ir::TimeUnit new_unit) {
  auto old_scale = get_nanosecs_in_unit(old_unit);
  auto new_scale = get_nanosecs_in_unit(new_unit);
  if (old_scale > new_scale) {
    auto scaled_epoch = epoch * (old_scale / new_scale);
    if (epoch && epoch != scaled_epoch / (old_scale / new_scale)) {
      throw std::runtime_error(
          "Value Overflow/underflow detected while scaling DateTime precision.");
    }
    return scaled_epoch;
  } else {
    return epoch / (new_scale / old_scale);
  }
}

}  // namespace DateTimeUtils
