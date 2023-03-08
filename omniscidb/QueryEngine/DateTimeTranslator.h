/*
 * Copyright 2019 OmniSci, Inc.
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

#include "IR/Expr.h"
#include "Utils/ExtractFromTime.h"

#include "DateTimeUtils.h"

#include <memory>
#include <string>

using namespace DateTimeUtils;

class DateTimeTranslator {
 public:
  static inline int64_t getExtractFromTimeConstantValue(
      const int64_t& timeval,
      const hdk::ir::DateExtractField& field,
      const hdk::ir::Type* type) {
    auto unit = type->isTimestamp() ? type->as<hdk::ir::TimestampType>()->unit()
                                    : hdk::ir::TimeUnit::kSecond;
    if (type->isTimestamp() && unit > hdk::ir::TimeUnit::kSecond) {
      if (is_subsecond_extract_field(field)) {
        const auto result = get_extract_high_precision_adjusted_scale(field, unit);
        return result.second ? ExtractFromTime(field,
                                               result.first == hdk::ir::OpType::kDiv
                                                   ? floor_div(timeval, result.second)
                                                   : timeval * result.second)
                             : ExtractFromTime(field, timeval);
      } else {
        return ExtractFromTime(field, floor_div(timeval, hdk::ir::unitsPerSecond(unit)));
      }
    } else if (is_subsecond_extract_field(field)) {
      return ExtractFromTime(field,
                             timeval * get_extract_timestamp_precision_scale(field));
    }
    return ExtractFromTime(field, timeval);
  }

  static inline int64_t getDateTruncConstantValue(const int64_t& timeval,
                                                  const hdk::ir::DateTruncField& field,
                                                  const hdk::ir::Type* type) {
    CHECK(type->isDateTime());
    auto unit = type->as<hdk::ir::DateTimeBaseType>()->unit();
    if (type->isTimestamp() && unit > hdk::ir::TimeUnit::kSecond) {
      if (is_subsecond_datetrunc_field(field)) {
        int64_t date_truncate = DateTruncate(field, timeval);
        const auto result = get_datetrunc_high_precision_scale(field, unit);
        if (result != -1) {
          date_truncate -= unsigned_mod(date_truncate, result);
        }
        return date_truncate;
      } else {
        const int64_t scale = hdk::ir::unitsPerSecond(unit);
        return DateTruncate(field, floor_div(timeval, scale)) * scale;
      }
    }
    return DateTruncate(field, timeval);
  }
};

class ExtractExpr : protected DateTimeTranslator {
 public:
  ExtractExpr(const hdk::ir::ExprPtr expr, const hdk::ir::DateExtractField& field)
      : from_expr_(expr), field_(field) {}
  ExtractExpr(const hdk::ir::ExprPtr expr, const std::string& field)
      : from_expr_(expr), field_(to_extract_field(field)) {}

  static hdk::ir::ExprPtr generate(const hdk::ir::ExprPtr, const std::string&);
  static hdk::ir::ExprPtr generate(const hdk::ir::ExprPtr,
                                   const hdk::ir::DateExtractField&);

  const hdk::ir::ExprPtr generate() const { return generate(from_expr_, field_); }

 private:
  static hdk::ir::DateExtractField to_extract_field(const std::string& field);

  hdk::ir::ExprPtr from_expr_;
  hdk::ir::DateExtractField field_;
};

class DateTruncExpr : protected DateTimeTranslator {
 public:
  DateTruncExpr(const hdk::ir::ExprPtr expr, const hdk::ir::DateTruncField& field)
      : from_expr_(expr), field_(field) {}
  DateTruncExpr(const hdk::ir::ExprPtr expr, const std::string& field)
      : from_expr_(expr), field_(to_datetrunc_field(field)) {}

  static hdk::ir::ExprPtr generate(const hdk::ir::ExprPtr, const std::string&);
  static hdk::ir::ExprPtr generate(const hdk::ir::ExprPtr,
                                   const hdk::ir::DateTruncField&);

  const hdk::ir::ExprPtr generate() const { return generate(from_expr_, field_); }

 private:
  static hdk::ir::DateTruncField to_datetrunc_field(const std::string& field);

  hdk::ir::ExprPtr from_expr_;
  hdk::ir::DateTruncField field_;
};
