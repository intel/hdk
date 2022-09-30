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
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

ExtractField to_datepart_field(const std::string& field) {
  ExtractField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = kYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = kQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = kMONTH;
  } else if (boost::iequals(field, "dayofyear") || boost::iequals(field, "dy") ||
             boost::iequals(field, "y")) {
    fieldno = kDOY;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = kDAY;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = kWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = kWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = kWEEK_SATURDAY;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = kHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = kMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = kSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = kMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = kMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = kNANOSECOND;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = kISODOW;
  } else if (boost::iequals(field, "quarterday") || boost::iequals(field, "dq")) {
    fieldno = kQUARTERDAY;
  } else {
    throw std::runtime_error("Unsupported field in DATEPART function: " + field);
  }
  return fieldno;
}

DateAddField to_dateadd_field(const std::string& field) {
  DateAddField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = daYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = daQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = daMONTH;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = daDAY;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = daWEEK;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = daHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = daMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = daSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = daMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = daMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = daNANOSECOND;
  } else if (boost::iequals(field, "weekday") || boost::iequals(field, "dw")) {
    fieldno = daWEEKDAY;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = daDECADE;
  } else if (boost::iequals(field, "century")) {
    fieldno = daCENTURY;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = daMILLENNIUM;
  } else {
    throw std::runtime_error("Unsupported field in DATEADD function: " + field);
  }
  return fieldno;
}

DateTruncField to_datediff_field(const std::string& field) {
  DateTruncField fieldno;
  if (boost::iequals(field, "year") || boost::iequals(field, "yy") ||
      boost::iequals(field, "yyyy") || boost::iequals(field, "sql_tsi_year")) {
    fieldno = dtYEAR;
  } else if (boost::iequals(field, "quarter") || boost::iequals(field, "qq") ||
             boost::iequals(field, "q") || boost::iequals(field, "sql_tsi_quarter")) {
    fieldno = dtQUARTER;
  } else if (boost::iequals(field, "month") || boost::iequals(field, "mm") ||
             boost::iequals(field, "m") || boost::iequals(field, "sql_tsi_month")) {
    fieldno = dtMONTH;
  } else if (boost::iequals(field, "week") || boost::iequals(field, "ww") ||
             boost::iequals(field, "w") || boost::iequals(field, "sql_tsi_week")) {
    fieldno = dtWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = dtWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = dtWEEK_SATURDAY;
  } else if (boost::iequals(field, "day") || boost::iequals(field, "dd") ||
             boost::iequals(field, "d") || boost::iequals(field, "sql_tsi_day")) {
    fieldno = dtDAY;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = dtQUARTERDAY;
  } else if (boost::iequals(field, "hour") || boost::iequals(field, "hh") ||
             boost::iequals(field, "sql_tsi_hour")) {
    fieldno = dtHOUR;
  } else if (boost::iequals(field, "minute") || boost::iequals(field, "mi") ||
             boost::iequals(field, "n") || boost::iequals(field, "sql_tsi_minute")) {
    fieldno = dtMINUTE;
  } else if (boost::iequals(field, "second") || boost::iequals(field, "ss") ||
             boost::iequals(field, "s") || boost::iequals(field, "sql_tsi_second")) {
    fieldno = dtSECOND;
  } else if (boost::iequals(field, "millisecond") || boost::iequals(field, "ms")) {
    fieldno = dtMILLISECOND;
  } else if (boost::iequals(field, "microsecond") || boost::iequals(field, "us") ||
             boost::iequals(field, "sql_tsi_microsecond") ||
             boost::iequals(field, "frac_second")) {
    fieldno = dtMICROSECOND;
  } else if (boost::iequals(field, "nanosecond") || boost::iequals(field, "ns") ||
             boost::iequals(field, "sql_tsi_frac_second")) {
    fieldno = dtNANOSECOND;
  } else if (boost::iequals(field, "decade") || boost::iequals(field, "dc")) {
    fieldno = dtDECADE;
  } else if (boost::iequals(field, "century")) {
    fieldno = dtCENTURY;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = dtMILLENNIUM;
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
