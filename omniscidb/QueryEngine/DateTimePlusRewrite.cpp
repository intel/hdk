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

#include "DateTimePlusRewrite.h"
#include "Execute.h"

#include "IR/Expr.h"
#include "Logger/Logger.h"

#include "DateTimeTranslator.h"

namespace {

const hdk::ir::Expr* remove_truncate_int(const hdk::ir::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto func_oper = dynamic_cast<const hdk::ir::FunctionOper*>(expr);
  if (!func_oper || func_oper->name() != "TRUNCATE") {
    return nullptr;
  }
  CHECK_EQ(size_t(2), func_oper->getArity());
  const auto arg = func_oper->getArg(0);
  auto arg_type = arg->type();
  return arg_type->isInteger() ? arg : nullptr;
}

bool match_const_integer(const hdk::ir::Expr* expr, const int64_t v) {
  const auto const_expr = dynamic_cast<const hdk::ir::Constant*>(expr);
  if (!const_expr) {
    return false;
  }
  auto const_type = const_expr->type();
  if (!const_type->isInteger()) {
    return false;
  }
  const auto& datum = const_expr->value();
  switch (const_type->size()) {
    case 1:
      return v == datum.tinyintval;
    case 2:
      return v == datum.smallintval;
    case 4:
      return v == datum.intval;
    case 8:
      return v == datum.bigintval;
    default:
      break;
  }
  return false;
}

DateTruncField get_dt_field(const hdk::ir::Expr* ts,
                            const hdk::ir::Expr* interval_multiplier,
                            const bool dt_hour) {
  if (dt_hour) {
    const auto extract_fn =
        dynamic_cast<const hdk::ir::ExtractExpr*>(interval_multiplier);
    return (extract_fn && extract_fn->field() == kHOUR && *extract_fn->from() == *ts)
               ? dtHOUR
               : dtINVALID;
  }
  const auto interval_multiplier_fn =
      remove_truncate_int(remove_cast_to_int(interval_multiplier));
  if (!interval_multiplier_fn) {
    return dtINVALID;
  }
  const auto interval_multiplier_mul =
      dynamic_cast<const hdk::ir::BinOper*>(interval_multiplier_fn);
  if (!interval_multiplier_mul || !interval_multiplier_mul->isMul() ||
      !match_const_integer(interval_multiplier_mul->leftOperand(), -1)) {
    return dtINVALID;
  }
  const auto extract_minus_one =
      dynamic_cast<const hdk::ir::BinOper*>(interval_multiplier_mul->rightOperand());
  if (!extract_minus_one || !extract_minus_one->isMinus() ||
      !match_const_integer(extract_minus_one->rightOperand(), 1)) {
    return dtINVALID;
  }
  const auto extract_fn =
      dynamic_cast<const hdk::ir::ExtractExpr*>(extract_minus_one->leftOperand());
  if (!extract_fn || !(*extract_fn->from() == *ts)) {
    return dtINVALID;
  }
  switch (extract_fn->field()) {
    case kDAY:
      return dtMONTH;
    case kDOY:
      return dtYEAR;
    default:
      break;
  }
  return dtINVALID;
}

DateTruncField get_dt_field(const hdk::ir::Expr* ts, const hdk::ir::Expr* off_arg) {
  const auto mul_by_interval = dynamic_cast<const hdk::ir::BinOper*>(off_arg);
  if (!mul_by_interval) {
    return dtINVALID;
  }
  auto interval = dynamic_cast<const hdk::ir::Constant*>(mul_by_interval->rightOperand());
  auto interval_multiplier = mul_by_interval->leftOperand();
  if (!interval) {
    interval = dynamic_cast<const hdk::ir::Constant*>(mul_by_interval->leftOperand());
    if (!interval) {
      return dtINVALID;
    }
    interval_multiplier = mul_by_interval->rightOperand();
  }
  auto interval_type = interval->type();
  if (!interval_type->isInterval() ||
      interval_type->as<hdk::ir::IntervalType>()->unit() != hdk::ir::TimeUnit::kMilli) {
    return dtINVALID;
  }
  const auto& datum = interval->value();
  switch (datum.bigintval) {
    case 86400000:
      return get_dt_field(ts, interval_multiplier, false);
    case 3600000:
      return get_dt_field(ts, interval_multiplier, true);
    default:
      break;
  }
  return dtINVALID;
}

hdk::ir::ExprPtr remove_cast_to_date(const hdk::ir::Expr* expr) {
  if (!expr) {
    return nullptr;
  }
  const auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr);
  if (!uoper || !uoper->isCast()) {
    return nullptr;
  }
  auto operand_type = uoper->operand()->type();
  auto target_type = uoper->type();
  if (!operand_type->isTimestamp() || !target_type->isDate()) {
    return nullptr;
  }
  return uoper->operandShared();
}

}  // namespace

hdk::ir::ExprPtr rewrite_to_date_trunc(const hdk::ir::FunctionOper* dt_plus) {
  CHECK_EQ("DATETIME_PLUS", dt_plus->name());
  CHECK_EQ(size_t(2), dt_plus->getArity());
  const auto ts = remove_cast_to_date(dt_plus->getArg(0));
  if (!ts) {
    return nullptr;
  }
  const auto off_arg = dt_plus->getArg(1);
  const auto dt_field = get_dt_field(ts.get(), off_arg);
  if (dt_field == dtINVALID) {
    return nullptr;
  }
  return DateTruncExpr::generate(ts, dt_field);
}
