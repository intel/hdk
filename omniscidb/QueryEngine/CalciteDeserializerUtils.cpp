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

#include "CalciteDeserializerUtils.h"

#include "../Analyzer/Analyzer.h"
#include "IR/Context.h"
#include "IR/DateTime.h"
#include "IR/Type.h"
#include "Logger/Logger.h"

#include <boost/algorithm/string.hpp>

const hdk::ir::Type* get_agg_type(hdk::ir::AggType agg_kind,
                                  const hdk::ir::Expr* arg_expr,
                                  bool bigint_count) {
  auto& ctx = arg_expr ? arg_expr->type()->ctx() : hdk::ir::Context::defaultCtx();
  switch (agg_kind) {
    case hdk::ir::AggType::kCount:
      return ctx.integer(bigint_count ? 8 : 4);
    case hdk::ir::AggType::kMin:
    case hdk::ir::AggType::kMax:
      return arg_expr->type();
    case hdk::ir::AggType::kSum:
      return arg_expr->type()->isInteger() ? ctx.int64() : arg_expr->type();
    case hdk::ir::AggType::kAvg:
      return ctx.fp64();
    case hdk::ir::AggType::kApproxCountDistinct:
      return ctx.int64();
    case hdk::ir::AggType::kApproxQuantile:
      return ctx.fp64();
    case hdk::ir::AggType::kSingleValue:
      if (arg_expr->type()->isVarLen()) {
        throw std::runtime_error("SINGLE_VALUE not supported on '" +
                                 arg_expr->type()->toString() + "' input.");
      }
    case hdk::ir::AggType::kSample:
      return arg_expr->type();
    case hdk::ir::AggType::kTopK:
      return ctx.arrayVarLen(arg_expr->type(), 4, false);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

hdk::ir::DateExtractField to_datepart_field(const std::string& field) {
  hdk::ir::DateExtractField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = hdk::ir::DateExtractField::kYear;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = hdk::ir::DateExtractField::kQuarter;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = hdk::ir::DateExtractField::kMonth;
  } else if (boost::iequals(field, "dayofyear") || boost::iequals(field, "dy") ||
             boost::iequals(field, "y")) {
    fieldno = hdk::ir::DateExtractField::kDayOfYear;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = hdk::ir::DateExtractField::kDay;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = hdk::ir::DateExtractField::kWeek;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = hdk::ir::DateExtractField::kWeekSunday;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = hdk::ir::DateExtractField::kWeekSaturday;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = hdk::ir::DateExtractField::kHour;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = hdk::ir::DateExtractField::kMinute;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = hdk::ir::DateExtractField::kSecond;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = hdk::ir::DateExtractField::kMilli;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = hdk::ir::DateExtractField::kMicro;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = hdk::ir::DateExtractField::kNano;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = hdk::ir::DateExtractField::kIsoDayOfWeek;
  } else if (boost::iequals(field, "quarterday") || boost::iequals(field, "dq")) {
    fieldno = hdk::ir::DateExtractField::kQuarterDay;
  } else {
    throw std::runtime_error("Unsupported field in DATEPART function: " + field);
  }
  return fieldno;
}

hdk::ir::DateAddField to_dateadd_field(const std::string& field) {
  hdk::ir::DateAddField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = hdk::ir::DateAddField::kYear;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = hdk::ir::DateAddField::kQuarter;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = hdk::ir::DateAddField::kMonth;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = hdk::ir::DateAddField::kDay;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = hdk::ir::DateAddField::kWeek;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = hdk::ir::DateAddField::kHour;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = hdk::ir::DateAddField::kMinute;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = hdk::ir::DateAddField::kSecond;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = hdk::ir::DateAddField::kMilli;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = hdk::ir::DateAddField::kMicro;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = hdk::ir::DateAddField::kNano;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = hdk::ir::DateAddField::kWeekDay;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = hdk::ir::DateAddField::kDecade;
  } else if (boost::iequals(field, "century")) {
    fieldno = hdk::ir::DateAddField::kCentury;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = hdk::ir::DateAddField::kMillennium;
  } else {
    throw std::runtime_error("Unsupported field in DATEADD function: " + field);
  }
  return fieldno;
}

hdk::ir::DateTruncField to_datediff_field(const std::string& field) {
  hdk::ir::DateTruncField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = hdk::ir::DateTruncField::kYear;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = hdk::ir::DateTruncField::kQuarter;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = hdk::ir::DateTruncField::kMonth;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = hdk::ir::DateTruncField::kWeek;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = hdk::ir::DateTruncField::kWeekSunday;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = hdk::ir::DateTruncField::kWeekSaturday;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = hdk::ir::DateTruncField::kDay;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = hdk::ir::DateTruncField::kQuarterDay;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = hdk::ir::DateTruncField::kHour;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = hdk::ir::DateTruncField::kMinute;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = hdk::ir::DateTruncField::kSecond;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = hdk::ir::DateTruncField::kMilli;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = hdk::ir::DateTruncField::kMicro;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = hdk::ir::DateTruncField::kNano;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = hdk::ir::DateTruncField::kDecade;
  } else if (boost::iequals(field, "century")) {
    fieldno = hdk::ir::DateTruncField::kCentury;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = hdk::ir::DateTruncField::kMillennium;
  } else {
    throw std::runtime_error("Unsupported field in DATEDIFF function: " + field);
  }
  return fieldno;
}

std::shared_ptr<const hdk::ir::Constant> make_fp_constant(const int64_t val,
                                                          const hdk::ir::Type* type) {
  Datum d;
  if (type->isFp32()) {
    d.floatval = val;
  } else {
    CHECK(type->isFp64());
    d.doubleval = val;
  }
  return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
}
