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

#ifndef QUERYENGINE_CALCITEDESERIALIZERUTILS_H
#define QUERYENGINE_CALCITEDESERIALIZERUTILS_H

#include "DateTruncate.h"

#include "../Shared/sqldefs.h"
#include "../Shared/sqltypes.h"
#include "IR/DateTime.h"
#include "IR/OpType.h"
#include "Logger/Logger.h"

inline hdk::ir::OpType to_sql_op(const std::string& op_str) {
  if (op_str == std::string(">")) {
    return hdk::ir::OpType::kGt;
  }
  if (op_str == std::string("IS NOT DISTINCT FROM")) {
    return hdk::ir::OpType::kBwEq;
  }
  if (op_str == std::string(">=")) {
    return hdk::ir::OpType::kGe;
  }
  if (op_str == std::string("<")) {
    return hdk::ir::OpType::kLt;
  }
  if (op_str == std::string("<=")) {
    return hdk::ir::OpType::kLe;
  }
  if (op_str == std::string("=")) {
    return hdk::ir::OpType::kEq;
  }
  if (op_str == std::string("<>")) {
    return hdk::ir::OpType::kNe;
  }
  if (op_str == std::string("+")) {
    return hdk::ir::OpType::kPlus;
  }
  if (op_str == std::string("-")) {
    return hdk::ir::OpType::kMinus;
  }
  if (op_str == std::string("*")) {
    return hdk::ir::OpType::kMul;
  }
  if (op_str == std::string("/")) {
    return hdk::ir::OpType::kDiv;
  }
  if (op_str == "MOD") {
    return hdk::ir::OpType::kMod;
  }
  if (op_str == std::string("AND")) {
    return hdk::ir::OpType::kAnd;
  }
  if (op_str == std::string("OR")) {
    return hdk::ir::OpType::kOr;
  }
  if (op_str == std::string("CAST")) {
    return hdk::ir::OpType::kCast;
  }
  if (op_str == std::string("NOT")) {
    return hdk::ir::OpType::kNot;
  }
  if (op_str == std::string("IS NULL")) {
    return hdk::ir::OpType::kIsNull;
  }
  if (op_str == std::string("IS NOT NULL")) {
    return hdk::ir::OpType::kIsNotNull;
  }
  if (op_str == std::string("PG_UNNEST")) {
    return hdk::ir::OpType::kUnnest;
  }
  if (op_str == std::string("PG_ANY") || op_str == std::string("PG_ALL")) {
    throw std::runtime_error("Invalid use of " + op_str + " operator");
  }
  if (op_str == std::string("IN")) {
    return hdk::ir::OpType::kIn;
  }
  return hdk::ir::OpType::kFunction;
}

inline hdk::ir::AggType to_agg_kind(const std::string& agg_name) {
  if (agg_name == std::string("COUNT")) {
    return hdk::ir::AggType::kCount;
  }
  if (agg_name == std::string("MIN")) {
    return hdk::ir::AggType::kMin;
  }
  if (agg_name == std::string("MAX")) {
    return hdk::ir::AggType::kMax;
  }
  if (agg_name == std::string("SUM")) {
    return hdk::ir::AggType::kSum;
  }
  if (agg_name == std::string("AVG")) {
    return hdk::ir::AggType::kAvg;
  }
  if (agg_name == std::string("APPROX_COUNT_DISTINCT")) {
    return hdk::ir::AggType::kApproxCountDistinct;
  }
  if (agg_name == "APPROX_MEDIAN" || agg_name == "APPROX_PERCENTILE" ||
      agg_name == "APPROX_QUANTILE") {
    return hdk::ir::AggType::kApproxQuantile;
  }
  if (agg_name == std::string("ANY_VALUE") || agg_name == std::string("SAMPLE") ||
      agg_name == std::string("LAST_SAMPLE")) {
    return hdk::ir::AggType::kSample;
  }
  if (agg_name == std::string("SINGLE_VALUE")) {
    return hdk::ir::AggType::kSingleValue;
  }
  throw std::runtime_error("Aggregate function " + agg_name + " not supported");
}

namespace hdk::ir {

class Constant;
class Expr;
class Type;

}  // namespace hdk::ir

const hdk::ir::Type* get_agg_type(hdk::ir::AggType agg_kind,
                                  const hdk::ir::Expr* arg_expr,
                                  bool bigint_count);

ExtractField to_datepart_field(const std::string&);

hdk::ir::DateAddField to_dateadd_field(const std::string&);

DateTruncField to_datediff_field(const std::string&);

std::shared_ptr<const hdk::ir::Constant> make_fp_constant(const int64_t val,
                                                          const hdk::ir::Type* type);

#endif  // QUERYENGINE_CALCITEDESERIALIZERUTILS_H
