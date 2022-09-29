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

std::string from_extract_field(const ExtractField& fieldno) {
  switch (fieldno) {
    case kYEAR:
      return "year";
    case kQUARTER:
      return "quarter";
    case kMONTH:
      return "month";
    case kDAY:
      return "day";
    case kHOUR:
      return "hour";
    case kMINUTE:
      return "minute";
    case kSECOND:
      return "second";
    case kMILLISECOND:
      return "millisecond";
    case kMICROSECOND:
      return "microsecond";
    case kNANOSECOND:
      return "nanosecond";
    case kDOW:
      return "dow";
    case kISODOW:
      return "isodow";
    case kDOY:
      return "doy";
    case kEPOCH:
      return "epoch";
    case kQUARTERDAY:
      return "quarterday";
    case kWEEK:
      return "week";
    case kWEEK_SUNDAY:
      return "week_sunday";
    case kWEEK_SATURDAY:
      return "week_saturday";
    case kDATEEPOCH:
      return "dateepoch";
    default:
      UNREACHABLE();
  }
  return "";
}

std::string from_datetrunc_field(const DateTruncField& fieldno) {
  switch (fieldno) {
    case dtYEAR:
      return "year";
    case dtQUARTER:
      return "quarter";
    case dtMONTH:
      return "month";
    case dtQUARTERDAY:
      return "quarterday";
    case dtDAY:
      return "day";
    case dtHOUR:
      return "hour";
    case dtMINUTE:
      return "minute";
    case dtSECOND:
      return "second";
    case dtMILLENNIUM:
      return "millennium";
    case dtCENTURY:
      return "century";
    case dtDECADE:
      return "decade";
    case dtMILLISECOND:
      return "millisecond";
    case dtMICROSECOND:
      return "microsecond";
    case dtNANOSECOND:
      return "nanosecond";
    case dtWEEK:
      return "week";
    case dtWEEK_SUNDAY:
      return "week_sunday";
    case dtWEEK_SATURDAY:
      return "week_saturday";
    case dtINVALID:
    default:
      UNREACHABLE();
  }
  return "";
}

}  // namespace

ExtractField ExtractExpr::to_extract_field(const std::string& field) {
  ExtractField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = kYEAR;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = kQUARTER;
  } else if (boost::iequals(field, "month")) {
    fieldno = kMONTH;
  } else if (boost::iequals(field, "day")) {
    fieldno = kDAY;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = kQUARTERDAY;
  } else if (boost::iequals(field, "hour")) {
    fieldno = kHOUR;
  } else if (boost::iequals(field, "minute")) {
    fieldno = kMINUTE;
  } else if (boost::iequals(field, "second")) {
    fieldno = kSECOND;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = kMILLISECOND;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = kMICROSECOND;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = kNANOSECOND;
  } else if (boost::iequals(field, "dow")) {
    fieldno = kDOW;
  } else if (boost::iequals(field, "isodow")) {
    fieldno = kISODOW;
  } else if (boost::iequals(field, "doy")) {
    fieldno = kDOY;
  } else if (boost::iequals(field, "epoch")) {
    fieldno = kEPOCH;
  } else if (boost::iequals(field, "week")) {
    fieldno = kWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = kWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = kWEEK_SATURDAY;
  } else if (boost::iequals(field, "dateepoch")) {
    fieldno = kDATEEPOCH;
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
                                       const ExtractField& field) {
  const auto expr_type = from_expr->type();
  if (!expr_type->isDateTime()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in EXTRACT function.");
  }
  if (expr_type->isTime() && field != kHOUR && field != kMINUTE && field != kSECOND) {
    throw std::runtime_error("Cannot EXTRACT " + from_extract_field(field) +
                             " from TIME.");
  }
  auto type = expr_type->ctx().int64(expr_type->nullable());
  auto constant = from_expr->as<hdk::ir::Constant>();
  if (constant != nullptr) {
    Datum d;
    d.bigintval = field == kEPOCH
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

DateTruncField DateTruncExpr::to_datetrunc_field(const std::string& field) {
  DateTruncField fieldno;
  if (boost::iequals(field, "year")) {
    fieldno = dtYEAR;
  } else if (boost::iequals(field, "quarter")) {
    fieldno = dtQUARTER;
  } else if (boost::iequals(field, "month")) {
    fieldno = dtMONTH;
  } else if (boost::iequals(field, "quarterday")) {
    fieldno = dtQUARTERDAY;
  } else if (boost::iequals(field, "day")) {
    fieldno = dtDAY;
  } else if (boost::iequals(field, "hour")) {
    fieldno = dtHOUR;
  } else if (boost::iequals(field, "minute")) {
    fieldno = dtMINUTE;
  } else if (boost::iequals(field, "second")) {
    fieldno = dtSECOND;
  } else if (boost::iequals(field, "millennium")) {
    fieldno = dtMILLENNIUM;
  } else if (boost::iequals(field, "century")) {
    fieldno = dtCENTURY;
  } else if (boost::iequals(field, "decade")) {
    fieldno = dtDECADE;
  } else if (boost::iequals(field, "millisecond")) {
    fieldno = dtMILLISECOND;
  } else if (boost::iequals(field, "microsecond")) {
    fieldno = dtMICROSECOND;
  } else if (boost::iequals(field, "nanosecond")) {
    fieldno = dtNANOSECOND;
  } else if (boost::iequals(field, "week")) {
    fieldno = dtWEEK;
  } else if (boost::iequals(field, "week_sunday")) {
    fieldno = dtWEEK_SUNDAY;
  } else if (boost::iequals(field, "week_saturday")) {
    fieldno = dtWEEK_SATURDAY;
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
                                         const DateTruncField& field) {
  auto expr_type = from_expr->type();
  if (!expr_type->isDateTime()) {
    throw std::runtime_error(
        "Only TIME, TIMESTAMP and DATE types can be in DATE_TRUNC function.");
  }
  if (expr_type->isTime() && field != dtHOUR && field != dtMINUTE && field != dtSECOND) {
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
