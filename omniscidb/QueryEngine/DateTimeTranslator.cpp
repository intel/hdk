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

#include "DateTimeTranslator.h"

#include <boost/algorithm/string.hpp>

namespace {

std::string from_extract_field(const hdk::ir::DateExtractField& fieldno) {
  switch (fieldno) {
    case hdk::ir::DateExtractField::kYear:
      return "year";
    case hdk::ir::DateExtractField::kQuarter:
      return "quarter";
    case hdk::ir::DateExtractField::kMonth:
      return "month";
    case hdk::ir::DateExtractField::kDay:
      return "day";
    case hdk::ir::DateExtractField::kHour:
      return "hour";
    case hdk::ir::DateExtractField::kMinute:
      return "minute";
    case hdk::ir::DateExtractField::kSecond:
      return "second";
    case hdk::ir::DateExtractField::kMilli:
      return "millisecond";
    case hdk::ir::DateExtractField::kMicro:
      return "microsecond";
    case hdk::ir::DateExtractField::kNano:
      return "nanosecond";
    case hdk::ir::DateExtractField::kDayOfWeek:
      return "dow";
    case hdk::ir::DateExtractField::kIsoDayOfWeek:
      return "isodow";
    case hdk::ir::DateExtractField::kDayOfYear:
      return "doy";
    case hdk::ir::DateExtractField::kEpoch:
      return "epoch";
    case hdk::ir::DateExtractField::kQuarterDay:
      return "quarterday";
    case hdk::ir::DateExtractField::kWeek:
      return "week";
    case hdk::ir::DateExtractField::kWeekSunday:
      return "week_sunday";
    case hdk::ir::DateExtractField::kWeekSaturday:
      return "week_saturday";
    case hdk::ir::DateExtractField::kDateEpoch:
      return "dateepoch";
    default:
      UNREACHABLE();
  }
  return "";
}

std::string from_datetrunc_field(const hdk::ir::DateTruncField& fieldno) {
  switch (fieldno) {
    case hdk::ir::DateTruncField::kYear:
      return "year";
    case hdk::ir::DateTruncField::kQuarter:
      return "quarter";
    case hdk::ir::DateTruncField::kMonth:
      return "month";
    case hdk::ir::DateTruncField::kQuarterDay:
      return "quarterday";
    case hdk::ir::DateTruncField::kDay:
      return "day";
    case hdk::ir::DateTruncField::kHour:
      return "hour";
    case hdk::ir::DateTruncField::kMinute:
      return "minute";
    case hdk::ir::DateTruncField::kSecond:
      return "second";
    case hdk::ir::DateTruncField::kMillennium:
      return "millennium";
    case hdk::ir::DateTruncField::kCentury:
      return "century";
    case hdk::ir::DateTruncField::kDecade:
      return "decade";
    case hdk::ir::DateTruncField::kMilli:
      return "millisecond";
    case hdk::ir::DateTruncField::kMicro:
      return "microsecond";
    case hdk::ir::DateTruncField::kNano:
      return "nanosecond";
    case hdk::ir::DateTruncField::kWeek:
      return "week";
    case hdk::ir::DateTruncField::kWeekSunday:
      return "week_sunday";
    case hdk::ir::DateTruncField::kWeekSaturday:
      return "week_saturday";
    case hdk::ir::DateTruncField::kInvalid:
    default:
      UNREACHABLE();
  }
  return "";
}

}  // namespace

hdk::ir::DateExtractField ExtractExpr::to_extract_field(const std::string& field) {
  hdk::ir::DateExtractField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = hdk::ir::DateExtractField::kYear;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = hdk::ir::DateExtractField::kQuarter;
  } else if (boost::iequals(field, "month")) {
    fieldno = hdk::ir::DateExtractField::kMonth;
  } else if (boost::iequals(field, "day")) {
    fieldno = hdk::ir::DateExtractField::kDay;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = hdk::ir::DateExtractField::kQuarterDay;
  } else if (boost::iequals(field, "hour")) {
    fieldno = hdk::ir::DateExtractField::kHour;
  } else if (boost::iequals(field, "minute")) {
    fieldno = hdk::ir::DateExtractField::kMinute;
  } else if (boost::iequals(field, "second")) {
    fieldno = hdk::ir::DateExtractField::kSecond;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = hdk::ir::DateExtractField::kMilli;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = hdk::ir::DateExtractField::kMicro;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = hdk::ir::DateExtractField::kNano;
  } else if (boost::iequals(field, "dow")) {
    fieldno = hdk::ir::DateExtractField::kDayOfWeek;
  } else if (boost::iequals(field, "isodow")) {
    fieldno = hdk::ir::DateExtractField::kIsoDayOfWeek;
  } else if (boost::iequals(field, "doy")) {
    fieldno = hdk::ir::DateExtractField::kDayOfYear;
  } else if (boost::iequals(field, "epoch")) {
    fieldno = hdk::ir::DateExtractField::kEpoch;
  } else if (boost::iequals(field, "week")) {
    fieldno = hdk::ir::DateExtractField::kWeek;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = hdk::ir::DateExtractField::kWeekSunday;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = hdk::ir::DateExtractField::kWeekSaturday;
  } else if (boost::iequals(field, "dateepoch")) {
    fieldno = hdk::ir::DateExtractField::kDateEpoch;
  } else {
    throw std::runtime_error("Unsupported field in EXTRACT function " + field);
  }
  return fieldno;
}

hdk::ir::ExprPtr ExtractExpr::generate(const hdk::ir::ExprPtr from_expr,
                                       const std::string& field_name) {
  const auto field = to_extract_field(field_name);
  return ExtractExpr::generate(from_expr, field);
}

hdk::ir::ExprPtr ExtractExpr::generate(const hdk::ir::ExprPtr from_expr,
                                       const hdk::ir::DateExtractField& field) {
  const auto expr_type = from_expr->type();
  if (!expr_type->isDateTime()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
  }
  if (expr_type->isTime() && field != hdk::ir::DateExtractField::kHour &&
      field != hdk::ir::DateExtractField::kMinute &&
      field != hdk::ir::DateExtractField::kSecond) {
    throw std::runtime_error("Cannot EXTRACT " + from_extract_field(field) +
                             " from TIME.");
  }
  auto type = expr_type->ctx().int64(expr_type->nullable());
  auto constant = from_expr->as<hdk::ir::Constant>();
  if (constant != nullptr) {
    Datum d;
    d.bigintval = field == hdk::ir::DateExtractField::kEpoch
                      ? floor_div(constant->value().bigintval,
                                  hdk::ir::unitsPerSecond(
                                      expr_type->as<hdk::ir::DateTimeBaseType>()->unit()))
                      : getExtractFromTimeConstantValue(
                            constant->value().bigintval, field, expr_type);
    return hdk::ir::makeExpr<hdk::ir::Constant>(
        type, constant->isNull(), d, constant->cacheable());
  }
  return hdk::ir::makeExpr<hdk::ir::ExtractExpr>(
      type, from_expr->containsAgg(), field, from_expr->decompress());
}

hdk::ir::DateTruncField DateTruncExpr::to_datetrunc_field(const std::string& field) {
  hdk::ir::DateTruncField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = hdk::ir::DateTruncField::kYear;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = hdk::ir::DateTruncField::kQuarter;
  } else if (boost::iequals(field, "month")) {
    fieldno = hdk::ir::DateTruncField::kMonth;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = hdk::ir::DateTruncField::kQuarterDay;
  } else if (boost::iequals(field, "day")) {
    fieldno = hdk::ir::DateTruncField::kDay;
  } else if (boost::iequals(field, "hour")) {
    fieldno = hdk::ir::DateTruncField::kHour;
  } else if (boost::iequals(field, "minute")) {
    fieldno = hdk::ir::DateTruncField::kMinute;
  } else if (boost::iequals(field, "second")) {
    fieldno = hdk::ir::DateTruncField::kSecond;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = hdk::ir::DateTruncField::kMillennium;
  } else if (boost::iequals(field, "century")) {
    fieldno = hdk::ir::DateTruncField::kCentury;
  } else if (boost::iequals(field, "decade")) {
    fieldno = hdk::ir::DateTruncField::kDecade;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = hdk::ir::DateTruncField::kMilli;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = hdk::ir::DateTruncField::kMicro;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = hdk::ir::DateTruncField::kNano;
  } else if (boost::iequals(field, "week")) {
    fieldno = hdk::ir::DateTruncField::kWeek;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = hdk::ir::DateTruncField::kWeekSunday;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = hdk::ir::DateTruncField::kWeekSaturday;
  } else {
    throw std::runtime_error("Invalid field in DATE_TRUNC function " + field);
  }
  return fieldno;
}

hdk::ir::ExprPtr DateTruncExpr::generate(const hdk::ir::ExprPtr from_expr,
                                         const std::string& field_name) {
  const auto field = to_datetrunc_field(field_name);
  return DateTruncExpr::generate(from_expr, field);
}

hdk::ir::ExprPtr DateTruncExpr::generate(const hdk::ir::ExprPtr from_expr,
                                         const hdk::ir::DateTruncField& field) {
  auto expr_type = from_expr->type();
  if (!expr_type->isDateTime()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in DATE_TRUNC function.");
  }
  if (expr_type->isTime() && field != hdk::ir::DateTruncField::kHour &&
      field != hdk::ir::DateTruncField::kMinute &&
      field != hdk::ir::DateTruncField::kSecond) {
    throw std::runtime_error("Cannot DATE_TRUNC " + from_datetrunc_field(field) +
                             " from TIME.");
  }
  auto unit = expr_type->isTimestamp() ? expr_type->as<hdk::ir::TimestampType>()->unit()
                                       : hdk::ir::TimeUnit::kSecond;
  auto type = expr_type->ctx().timestamp(unit, expr_type->nullable());
  auto constant = from_expr->as<hdk::ir::Constant>();
  if (constant) {
    Datum d{0};
    d.bigintval =
        getDateTruncConstantValue(constant->value().bigintval, field, expr_type);
    return hdk::ir::makeExpr<hdk::ir::Constant>(
        type, constant->isNull(), d, constant->cacheable());
  }
  return hdk::ir::makeExpr<hdk::ir::DateTruncExpr>(
      type, from_expr->containsAgg(), field, from_expr->decompress());
}
