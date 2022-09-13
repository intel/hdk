/*
 * Copyright 2021 OmniSci, Inc.
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

#include "ExpressionRange.h"
#include "DateTimeTranslator.h"
#include "DateTimeUtils.h"
#include "DateTruncate.h"
#include "Descriptors/InputDescriptors.h"
#include "Execute.h"
#include "ExtractFromTime.h"
#include "IR/TypeUtils.h"
#include "QueryPhysicalInputsCollector.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

#define DEF_OPERATOR(fname, op)                                                    \
  ExpressionRange fname(const ExpressionRange& other) const {                      \
    if (type_ == ExpressionRangeType::Invalid ||                                   \
        other.type_ == ExpressionRangeType::Invalid) {                             \
      return ExpressionRange::makeInvalidRange();                                  \
    }                                                                              \
    CHECK(type_ == other.type_);                                                   \
    switch (type_) {                                                               \
      case ExpressionRangeType::Integer:                                           \
        return binOp<int64_t>(other, [](const int64_t x, const int64_t y) {        \
          return int64_t(checked_int64_t(x) op y);                                 \
        });                                                                        \
      case ExpressionRangeType::Float:                                             \
        return binOp<float>(other, [](const float x, const float y) {              \
          std::feclearexcept(FE_OVERFLOW);                                         \
          std::feclearexcept(FE_UNDERFLOW);                                        \
          auto result = x op y;                                                    \
          if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) { \
            throw std::runtime_error("overflow / underflow");                      \
          }                                                                        \
          return result;                                                           \
        });                                                                        \
      case ExpressionRangeType::Double:                                            \
        return binOp<double>(other, [](const double x, const double y) {           \
          std::feclearexcept(FE_OVERFLOW);                                         \
          std::feclearexcept(FE_UNDERFLOW);                                        \
          auto result = x op y;                                                    \
          if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) { \
            throw std::runtime_error("overflow / underflow");                      \
          }                                                                        \
          return result;                                                           \
        });                                                                        \
      default:                                                                     \
        CHECK(false);                                                              \
    }                                                                              \
    CHECK(false);                                                                  \
    return ExpressionRange::makeInvalidRange();                                    \
  }

DEF_OPERATOR(ExpressionRange::operator+, +)
DEF_OPERATOR(ExpressionRange::operator-, -)
DEF_OPERATOR(ExpressionRange::operator*, *)

void apply_fp_qual(const Datum const_datum,
                   const hdk::ir::Type* const_type,
                   const SQLOps sql_op,
                   ExpressionRange& qual_range) {
  double const_val = extract_fp_type_from_datum(const_datum, const_type);
  switch (sql_op) {
    case kGT:
    case kGE:
      qual_range.setFpMin(std::max(qual_range.getFpMin(), const_val));
      break;
    case kLT:
    case kLE:
      qual_range.setFpMax(std::min(qual_range.getFpMax(), const_val));
      break;
    case kEQ:
      qual_range.setFpMin(std::max(qual_range.getFpMin(), const_val));
      qual_range.setFpMax(std::min(qual_range.getFpMax(), const_val));
      break;
    default:  // there may be other operators, but don't do anything with them
      break;
  }
}

void apply_int_qual(const Datum const_datum,
                    const hdk::ir::Type* const_type,
                    const SQLOps sql_op,
                    ExpressionRange& qual_range) {
  int64_t const_val = extract_int_type_from_datum(const_datum, const_type);
  switch (sql_op) {
    case kGT:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val + 1));
      break;
    case kGE:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val));
      break;
    case kLT:
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val - 1));
      break;
    case kLE:
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val));
      break;
    case kEQ:
      qual_range.setIntMin(std::max(qual_range.getIntMin(), const_val));
      qual_range.setIntMax(std::min(qual_range.getIntMax(), const_val));
      break;
    default:  // there may be other operators, but don't do anything with them
      break;
  }
}

void apply_hpt_qual(const Datum const_datum,
                    const hdk::ir::Type* const_type,
                    const hdk::ir::Type* col_type,
                    const SQLOps sql_op,
                    ExpressionRange& qual_range) {
  auto const_unit = const_type->isTimestamp()
                        ? const_type->as<hdk::ir::TimestampType>()->unit()
                        : hdk::ir::TimeUnit::kSecond;
  auto col_unit = col_type->isTimestamp() ? col_type->as<hdk::ir::TimestampType>()->unit()
                                          : hdk::ir::TimeUnit::kSecond;
  Datum datum{0};
  datum.bigintval = get_datetime_scaled_epoch(
      extract_int_type_from_datum(const_datum, const_type), const_unit, col_unit);
  apply_int_qual(datum, const_type, sql_op, qual_range);
}

ExpressionRange apply_simple_quals(
    const hdk::ir::ColumnVar* col_expr,
    const ExpressionRange& col_range,
    const boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  if (!simple_quals) {
    return col_range;
  }
  ExpressionRange qual_range(col_range);
  for (auto const& itr : simple_quals.get()) {
    auto qual_bin_oper = dynamic_cast<hdk::ir::BinOper*>(itr.get());
    if (!qual_bin_oper) {
      continue;
    }
    const hdk::ir::Expr* left_operand = qual_bin_oper->get_left_operand();
    auto qual_col = dynamic_cast<const hdk::ir::ColumnVar*>(left_operand);
    if (!qual_col) {
      // Check for possibility that column is wrapped in a cast
      // Presumes that only simple casts (i.e. timestamp to timestamp or int to int) have
      // been passed through by BinOper::normalize_simple_predicate
      auto u_expr = dynamic_cast<const hdk::ir::UOper*>(left_operand);
      if (!u_expr) {
        continue;
      }
      qual_col = dynamic_cast<const hdk::ir::ColumnVar*>(u_expr->get_operand());
      if (!qual_col) {
        continue;
      }
    }
    if (qual_col->get_table_id() != col_expr->get_table_id() ||
        qual_col->get_column_id() != col_expr->get_column_id()) {
      continue;
    }
    const hdk::ir::Expr* right_operand = qual_bin_oper->get_right_operand();
    auto qual_const = dynamic_cast<const hdk::ir::Constant*>(right_operand);
    if (!qual_const) {
      continue;
    }
    if (qual_range.getType() == ExpressionRangeType::Float ||
        qual_range.getType() == ExpressionRangeType::Double) {
      apply_fp_qual(qual_const->get_constval(),
                    qual_const->type(),
                    qual_bin_oper->get_optype(),
                    qual_range);
    } else if (qual_col->type()->isTimestamp() || qual_const->type()->isTimestamp()) {
      CHECK(qual_const->type()->isDateTime());
      CHECK(qual_col->type()->isDateTime());
      apply_hpt_qual(qual_const->get_constval(),
                     qual_const->type(),
                     qual_col->type(),
                     qual_bin_oper->get_optype(),
                     qual_range);
    } else {
      apply_int_qual(qual_const->get_constval(),
                     qual_const->type(),
                     qual_bin_oper->get_optype(),
                     qual_range);
    }
  }
  return qual_range;
}

ExpressionRange ExpressionRange::div(const ExpressionRange& other,
                                     bool null_div_by_zero) const {
  if (type_ != ExpressionRangeType::Integer ||
      other.type_ != ExpressionRangeType::Integer) {
    return ExpressionRange::makeInvalidRange();
  }
  if (other.int_min_ * other.int_max_ <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return ExpressionRange::makeInvalidRange();
  }
  auto div_range = binOp<int64_t>(other, [](const int64_t x, const int64_t y) {
    return int64_t(checked_int64_t(x) / y);
  });
  if (null_div_by_zero) {
    div_range.setHasNulls();
  }
  return div_range;
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return ExpressionRange::makeInvalidRange();
  }
  ExpressionRange result;
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return ExpressionRange::makeInvalidRange();
    case ExpressionRangeType::Integer: {
      result.type_ = ExpressionRangeType::Integer;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.int_min_ = std::min(int_min_, other.int_min_);
      result.int_max_ = std::max(int_max_, other.int_max_);
      result.bucket_ = std::min(bucket_, other.bucket_);
      break;
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      result.type_ = type_;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.fp_min_ = std::min(fp_min_, other.fp_min_);
      result.fp_max_ = std::max(fp_max_, other.fp_max_);
      break;
    }
    default:
      CHECK(false);
  }
  return result;
}

bool ExpressionRange::operator==(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return false;
  }
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return true;
    case ExpressionRangeType::Integer: {
      return has_nulls_ == other.has_nulls_ && int_min_ == other.int_min_ &&
             int_max_ == other.int_max_;
    }
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      return has_nulls_ == other.has_nulls_ && fp_min_ == other.fp_min_ &&
             fp_max_ == other.fp_max_;
    }
    default:
      CHECK(false);
  }
  return false;
}

bool ExpressionRange::typeSupportsRange(const hdk::ir::Type* type) {
  if (type->isArray()) {
    return typeSupportsRange(type->as<hdk::ir::ArrayBaseType>()->elemType());
  } else {
    return (type->isNumber() || type->isBoolean() || type->isDateTime() ||
            type->isExtDictionary());
  }
}

ExpressionRange getExpressionRange(
    const hdk::ir::BinOper* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(const hdk::ir::Constant* expr);

ExpressionRange getExpressionRange(
    const hdk::ir::ColumnVar* col_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(const hdk::ir::LikeExpr* like_expr);

ExpressionRange getExpressionRange(const hdk::ir::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(
    const hdk::ir::UOper* u_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(
    const hdk::ir::ExtractExpr* extract_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor*,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(
    const hdk::ir::DatetruncExpr* datetrunc_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(
    const hdk::ir::WidthBucketExpr* width_bucket_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals);

ExpressionRange getExpressionRange(
    const hdk::ir::Expr* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  if (!ExpressionRange::typeSupportsRange(expr->type())) {
    return ExpressionRange::makeInvalidRange();
  }
  auto bin_oper_expr = dynamic_cast<const hdk::ir::BinOper*>(expr);
  if (bin_oper_expr) {
    return getExpressionRange(bin_oper_expr, query_infos, executor, simple_quals);
  }
  auto constant_expr = dynamic_cast<const hdk::ir::Constant*>(expr);
  if (constant_expr) {
    return getExpressionRange(constant_expr);
  }
  auto column_var_expr = dynamic_cast<const hdk::ir::ColumnVar*>(expr);
  if (column_var_expr) {
    return getExpressionRange(column_var_expr, query_infos, executor, simple_quals);
  }
  auto like_expr = dynamic_cast<const hdk::ir::LikeExpr*>(expr);
  if (like_expr) {
    return getExpressionRange(like_expr);
  }
  auto case_expr = dynamic_cast<const hdk::ir::CaseExpr*>(expr);
  if (case_expr) {
    return getExpressionRange(case_expr, query_infos, executor);
  }
  auto u_expr = dynamic_cast<const hdk::ir::UOper*>(expr);
  if (u_expr) {
    return getExpressionRange(u_expr, query_infos, executor, simple_quals);
  }
  auto extract_expr = dynamic_cast<const hdk::ir::ExtractExpr*>(expr);
  if (extract_expr) {
    return getExpressionRange(extract_expr, query_infos, executor, simple_quals);
  }
  auto datetrunc_expr = dynamic_cast<const hdk::ir::DatetruncExpr*>(expr);
  if (datetrunc_expr) {
    return getExpressionRange(datetrunc_expr, query_infos, executor, simple_quals);
  }
  auto width_bucket_expr = dynamic_cast<const hdk::ir::WidthBucketExpr*>(expr);
  if (width_bucket_expr) {
    return getExpressionRange(width_bucket_expr, query_infos, executor, simple_quals);
  }
  return ExpressionRange::makeInvalidRange();
}

namespace {

int64_t scale_up_interval_endpoint(const int64_t endpoint,
                                   const hdk::ir::DecimalType* type) {
  return endpoint * static_cast<int64_t>(exp_to_scale(type->scale()));
}

}  // namespace

ExpressionRange getExpressionRange(
    const hdk::ir::BinOper* expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  const auto& lhs =
      getExpressionRange(expr->get_left_operand(), query_infos, executor, simple_quals);
  const auto& rhs =
      getExpressionRange(expr->get_right_operand(), query_infos, executor, simple_quals);
  switch (expr->get_optype()) {
    case kPLUS:
      return lhs + rhs;
    case kMINUS:
      return lhs - rhs;
    case kMULTIPLY:
      return lhs * rhs;
    case kDIVIDE: {
      bool null_div_by_zero = executor->getConfig().exec.codegen.null_div_by_zero;
      auto lhs_type = expr->get_left_operand()->type();
      if (lhs_type->isDecimal() && lhs.getType() != ExpressionRangeType::Invalid) {
        CHECK(lhs.getType() == ExpressionRangeType::Integer);
        auto dec_type = lhs_type->as<hdk::ir::DecimalType>();
        const auto adjusted_lhs = ExpressionRange::makeIntRange(
            scale_up_interval_endpoint(lhs.getIntMin(), dec_type),
            scale_up_interval_endpoint(lhs.getIntMax(), dec_type),
            0,
            lhs.hasNulls());
        return adjusted_lhs.div(rhs, null_div_by_zero);
      }
      return lhs.div(rhs, null_div_by_zero);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(const hdk::ir::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto constant_type = constant_expr->type();
  const auto datum = constant_expr->get_constval();
  switch (constant_type->id()) {
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kDate: {
      int64_t v = extract_int_type_from_datum(datum, constant_type);
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case hdk::ir::Type::kFloatingPoint:
      switch (constant_type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return ExpressionRange::makeFloatRange(datum.floatval, datum.floatval, false);
        case hdk::ir::FloatingPointType::kDouble:
          return ExpressionRange::makeDoubleRange(
              datum.doubleval, datum.doubleval, false);
        default:
          break;
      }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#define FIND_STAT_FRAG(stat_name)                                                    \
  const auto stat_name##_frag_index = std::stat_name##_element(                      \
      nonempty_fragment_indices.begin(),                                             \
      nonempty_fragment_indices.end(),                                               \
      [&fragments, &has_nulls, col_id, col_type](const size_t lhs_idx,               \
                                                 const size_t rhs_idx) {             \
        const auto& lhs = fragments[lhs_idx];                                        \
        const auto& rhs = fragments[rhs_idx];                                        \
        auto lhs_meta_it = lhs.getChunkMetadataMap().find(col_id);                   \
        if (lhs_meta_it == lhs.getChunkMetadataMap().end()) {                        \
          return false;                                                              \
        }                                                                            \
        auto rhs_meta_it = rhs.getChunkMetadataMap().find(col_id);                   \
        CHECK(rhs_meta_it != rhs.getChunkMetadataMap().end());                       \
        if (lhs_meta_it->second->chunkStats.has_nulls ||                             \
            rhs_meta_it->second->chunkStats.has_nulls) {                             \
          has_nulls = true;                                                          \
        }                                                                            \
        if (col_type->isFloatingPoint()) {                                           \
          return extract_##stat_name##_stat_fp_type(lhs_meta_it->second->chunkStats, \
                                                    col_type) <                      \
                 extract_##stat_name##_stat_fp_type(rhs_meta_it->second->chunkStats, \
                                                    col_type);                       \
        }                                                                            \
        return extract_##stat_name##_stat_int_type(lhs_meta_it->second->chunkStats,  \
                                                   col_type) <                       \
               extract_##stat_name##_stat_int_type(rhs_meta_it->second->chunkStats,  \
                                                   col_type);                        \
      });                                                                            \
  if (stat_name##_frag_index == nonempty_fragment_indices.end()) {                   \
    return ExpressionRange::makeInvalidRange();                                      \
  }

namespace {

int64_t get_conservative_datetrunc_bucket(const DatetruncField datetrunc_field) {
  const int64_t day_seconds{24 * 3600};
  const int64_t year_days{365};
  switch (datetrunc_field) {
    case dtYEAR:
      return year_days * day_seconds;
    case dtQUARTER:
      return 90 * day_seconds;  // 90 is least number of days in any quater
    case dtMONTH:
      return 28 * day_seconds;
    case dtDAY:
      return day_seconds;
    case dtHOUR:
      return 3600;
    case dtMINUTE:
      return 60;
    case dtMILLENNIUM:
      return 1000 * year_days * day_seconds;
    case dtCENTURY:
      return 100 * year_days * day_seconds;
    case dtDECADE:
      return 10 * year_days * day_seconds;
    case dtWEEK:
    case dtWEEK_SUNDAY:
    case dtWEEK_SATURDAY:
      return 7 * day_seconds;
    case dtQUARTERDAY:
      return 4 * 60 * 50;
    default:
      return 0;
  }
}

}  // namespace

ExpressionRange getLeafColumnRange(const hdk::ir::ColumnVar* col_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor,
                                   const bool is_outer_join_proj) {
  bool has_nulls = is_outer_join_proj;
  int col_id = col_expr->get_column_id();
  const auto& col_phys_type =
      col_expr->type()->isArray()
          ? col_expr->type()->as<hdk::ir::ArrayBaseType>()->elemType()
          : col_expr->type();
  const auto col_type = logicalType(col_phys_type);
  switch (col_type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kExtDictionary:
    case hdk::ir::Type::kFloatingPoint: {
      std::optional<size_t> ti_idx;
      for (size_t i = 0; i < query_infos.size(); ++i) {
        if (col_expr->get_table_id() == query_infos[i].table_id) {
          ti_idx = i;
          break;
        }
      }
      CHECK(ti_idx);
      const auto& query_info = query_infos[*ti_idx].info;
      const auto& fragments = query_info.fragments;
      if (col_expr->is_virtual()) {
        CHECK(col_type->isInt64());
        const int64_t num_tuples = query_info.getNumTuples();
        return ExpressionRange::makeIntRange(
            0, std::max(num_tuples - 1, int64_t(0)), 0, has_nulls);
      }
      if (query_info.getNumTuples() == 0) {
        // The column doesn't contain any values, synthesize an empty range.
        if (col_type->isFloatingPoint()) {
          return col_type->size() == 4 ? ExpressionRange::makeFloatRange(0, -1, false)
                                       : ExpressionRange::makeDoubleRange(0, -1, false);
        }
        return ExpressionRange::makeIntRange(0, -1, 0, false);
      }
      std::vector<size_t> nonempty_fragment_indices;
      for (size_t i = 0; i < fragments.size(); ++i) {
        const auto& fragment = fragments[i];
        if (!fragment.isEmptyPhysicalFragment()) {
          nonempty_fragment_indices.push_back(i);
        }
      }
      FIND_STAT_FRAG(min);
      FIND_STAT_FRAG(max);
      const auto& min_frag = fragments[*min_frag_index];
      const auto min_it = min_frag.getChunkMetadataMap().find(col_id);
      if (min_it == min_frag.getChunkMetadataMap().end()) {
        return ExpressionRange::makeInvalidRange();
      }
      const auto& max_frag = fragments[*max_frag_index];
      const auto max_it = max_frag.getChunkMetadataMap().find(col_id);
      CHECK(max_it != max_frag.getChunkMetadataMap().end());
      for (const auto& fragment : fragments) {
        const auto it = fragment.getChunkMetadataMap().find(col_id);
        if (it != fragment.getChunkMetadataMap().end()) {
          if (it->second->chunkStats.has_nulls) {
            has_nulls = true;
            break;
          }
        }
      }
      if (col_type->isFloatingPoint()) {
        const auto min_val =
            extract_min_stat_fp_type(min_it->second->chunkStats, col_type);
        const auto max_val =
            extract_max_stat_fp_type(max_it->second->chunkStats, col_type);
        return col_type->size() == 4
                   ? ExpressionRange::makeFloatRange(min_val, max_val, has_nulls)
                   : ExpressionRange::makeDoubleRange(min_val, max_val, has_nulls);
      }
      const auto min_val =
          extract_min_stat_int_type(min_it->second->chunkStats, col_type);
      const auto max_val =
          extract_max_stat_int_type(max_it->second->chunkStats, col_type);
      if (max_val < min_val) {
        // The column doesn't contain any non-null values, synthesize an empty range.
        CHECK_GT(min_val, 0);
        return ExpressionRange::makeIntRange(0, -1, 0, has_nulls);
      }
      const int64_t bucket =
          col_type->isDate() ? get_conservative_datetrunc_bucket(dtDAY) : 0;
      return ExpressionRange::makeIntRange(min_val, max_val, bucket, has_nulls);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(
    const hdk::ir::ColumnVar* col_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  const int rte_idx = col_expr->get_rte_idx();
  CHECK_GE(rte_idx, 0);
  CHECK_LT(static_cast<size_t>(rte_idx), query_infos.size());
  bool is_outer_join_proj = rte_idx > 0 && executor->containsLeftDeepOuterJoin();
  if (col_expr->get_table_id() > 0) {
    auto col_range = executor->getColRange(
        PhysicalInput{col_expr->get_column_id(), col_expr->get_table_id()});
    if (is_outer_join_proj) {
      col_range.setHasNulls();
    }
    return apply_simple_quals(col_expr, col_range, simple_quals);
  }
  return getLeafColumnRange(col_expr, query_infos, executor, is_outer_join_proj);
}

ExpressionRange getExpressionRange(const hdk::ir::LikeExpr* like_expr) {
  const auto& type = like_expr->type();
  CHECK(type->isBoolean());
  const auto& arg_type = like_expr->get_arg()->type();
  return ExpressionRange::makeIntRange(
      arg_type->nullable() ? inline_null_value<bool>() : 0, 1, 0, false);
}

ExpressionRange getExpressionRange(const hdk::ir::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  auto expr_range = ExpressionRange::makeInvalidRange();
  bool has_nulls = false;
  for (const auto& expr_pair : expr_pair_list) {
    CHECK(expr_pair.first->type()->isBoolean());
    const auto crt_range =
        getExpressionRange(expr_pair.second.get(), query_infos, executor);
    if (crt_range.getType() == ExpressionRangeType::Null) {
      has_nulls = true;
      continue;
    }
    if (crt_range.getType() == ExpressionRangeType::Invalid) {
      return ExpressionRange::makeInvalidRange();
    }
    expr_range = (expr_range.getType() != ExpressionRangeType::Invalid)
                     ? expr_range || crt_range
                     : crt_range;
  }
  if (has_nulls && !(expr_range.getType() == ExpressionRangeType::Invalid)) {
    expr_range.setHasNulls();
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  const auto else_null_expr = dynamic_cast<const hdk::ir::Constant*>(else_expr);
  if (else_null_expr && else_null_expr->get_is_null()) {
    expr_range.setHasNulls();
    return expr_range;
  }
  return expr_range || getExpressionRange(else_expr, query_infos, executor);
}

namespace {

ExpressionRange fpRangeFromDecimal(const ExpressionRange& arg_range,
                                   const int64_t scale,
                                   const hdk::ir::Type* target_type) {
  CHECK(target_type->isFloatingPoint());
  if (target_type->size() == 4) {
    return ExpressionRange::makeFloatRange(
        static_cast<float>(arg_range.getIntMin()) / scale,
        static_cast<float>(arg_range.getIntMax()) / scale,
        arg_range.hasNulls());
  }
  return ExpressionRange::makeDoubleRange(
      static_cast<double>(arg_range.getIntMin()) / scale,
      static_cast<double>(arg_range.getIntMax()) / scale,
      arg_range.hasNulls());
}

ExpressionRange getDateTimePrecisionCastRange(const ExpressionRange& arg_range,
                                              const hdk::ir::Type* oper_type,
                                              const hdk::ir::Type* target_type) {
  if (oper_type->isTimestamp() && target_type->isDate()) {
    const auto field = dtDAY;
    auto oper_unit = oper_type->as<hdk::ir::TimestampType>()->unit();
    bool is_hpt = oper_unit > hdk::ir::TimeUnit::kSecond;
    const int64_t scale =
        hdk::ir::unitsPerSecond(oper_type->as<hdk::ir::TimestampType>()->unit());
    const int64_t min_ts = is_hpt ? DateTruncate(field, arg_range.getIntMin() / scale)
                                  : DateTruncate(field, arg_range.getIntMin());
    const int64_t max_ts = is_hpt ? DateTruncate(field, arg_range.getIntMax() / scale)
                                  : DateTruncate(field, arg_range.getIntMax());
    const int64_t bucket = get_conservative_datetrunc_bucket(field);

    return ExpressionRange::makeIntRange(min_ts, max_ts, bucket, arg_range.hasNulls());
  }

  CHECK(oper_type->isDateTime());
  CHECK(target_type->isDateTime());
  auto oper_unit = oper_type->isTimestamp()
                       ? oper_type->as<hdk::ir::TimestampType>()->unit()
                       : hdk::ir::TimeUnit::kSecond;
  auto target_unit = target_type->isTimestamp()
                         ? target_type->as<hdk::ir::TimestampType>()->unit()
                         : hdk::ir::TimeUnit::kSecond;
  CHECK(oper_unit != target_unit);
  int64_t min_ts = DateTimeUtils::get_datetime_scaled_epoch(
      arg_range.getIntMin(), oper_unit, target_unit);
  int64_t max_ts = DateTimeUtils::get_datetime_scaled_epoch(
      arg_range.getIntMax(), oper_unit, target_unit);

  return ExpressionRange::makeIntRange(min_ts, max_ts, 0, arg_range.hasNulls());
}

}  // namespace

ExpressionRange getExpressionRange(
    const hdk::ir::UOper* u_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  if (u_expr->get_optype() == kUNNEST) {
    return getExpressionRange(u_expr->get_operand(), query_infos, executor, simple_quals);
  }
  if (u_expr->get_optype() != kCAST) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto& type = u_expr->type();
  if (type->isExtDictionary()) {
    const auto sdp = executor->getStringDictionaryProxy(
        type->as<hdk::ir::ExtDictionaryType>()->dictId(),
        executor->getRowSetMemoryOwner(),
        true);
    CHECK(sdp);
    const auto const_operand =
        dynamic_cast<const hdk::ir::Constant*>(u_expr->get_operand());
    if (!const_operand) {
      // casted subquery result. return invalid for now, but we could attempt to pull the
      // range from the subquery result in the future
      CHECK(u_expr->get_operand());
      VLOG(1) << "Unable to determine expression range for dictionary encoded expression "
              << u_expr->get_operand()->toString() << ", proceeding with invalid range.";
      return ExpressionRange::makeInvalidRange();
    }

    if (const_operand->get_is_null()) {
      return ExpressionRange::makeNullRange();
    }
    CHECK(const_operand->get_constval().stringval);
    const int64_t v = sdp->getIdOfString(*const_operand->get_constval().stringval);
    return ExpressionRange::makeIntRange(v, v, 0, false);
  }
  const auto arg_range =
      getExpressionRange(u_expr->get_operand(), query_infos, executor, simple_quals);
  const auto& arg_type = u_expr->get_operand()->type();
  // Timestamp to Date OR Date/Timestamp casts with different precision
  auto arg_unit = arg_type->isTimestamp()
                      ? arg_type->as<hdk::ir::DateTimeBaseType>()->unit()
                      : hdk::ir::TimeUnit::kSecond;
  auto type_unit = type->isTimestamp() ? type->as<hdk::ir::DateTimeBaseType>()->unit()
                                       : hdk::ir::TimeUnit::kSecond;
  if ((type->isTimestamp() && (type_unit != arg_unit)) ||
      (arg_type->isTimestamp() && type->isDate())) {
    return getDateTimePrecisionCastRange(arg_range, arg_type, type);
  }
  switch (arg_range.getType()) {
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double: {
      if (type->isFloatingPoint()) {
        return type->size() == 8
                   ? ExpressionRange::makeDoubleRange(
                         arg_range.getFpMin(), arg_range.getFpMax(), arg_range.hasNulls())
                   : ExpressionRange::makeFloatRange(arg_range.getFpMin(),
                                                     arg_range.getFpMax(),
                                                     arg_range.hasNulls());
      }
      if (type->isInteger()) {
        return ExpressionRange::makeIntRange(std::floor(arg_range.getFpMin()),
                                             std::ceil(arg_range.getFpMax()),
                                             0,
                                             arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Integer: {
      if (type->isDecimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        auto type_scale = type->as<hdk::ir::DecimalType>()->scale();
        auto arg_scale =
            arg_type->isDecimal() ? arg_type->as<hdk::ir::DecimalType>()->scale() : 0;
        const int64_t scale = exp_to_scale(type_scale - arg_scale);
        return ExpressionRange::makeIntRange(arg_range.getIntMin() * scale,
                                             arg_range.getIntMax() * scale,
                                             0,
                                             arg_range.hasNulls());
      }
      if (arg_type->isDecimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        const int64_t scale = exp_to_scale(arg_type->as<hdk::ir::DecimalType>()->scale());
        const int64_t scale_half = scale / 2;
        if (type->isFloatingPoint()) {
          return fpRangeFromDecimal(arg_range, scale, type);
        }
        return ExpressionRange::makeIntRange((arg_range.getIntMin() - scale_half) / scale,
                                             (arg_range.getIntMax() + scale_half) / scale,
                                             0,
                                             arg_range.hasNulls());
      }
      if (type->isInteger() || type->isDateTime()) {
        return arg_range;
      }
      if (type->isFp32()) {
        return ExpressionRange::makeFloatRange(
            arg_range.getIntMin(), arg_range.getIntMax(), arg_range.hasNulls());
      }
      if (type->isFp64()) {
        return ExpressionRange::makeDoubleRange(
            arg_range.getIntMin(), arg_range.getIntMax(), arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Invalid:
      break;
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(
    const hdk::ir::ExtractExpr* extract_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  const int32_t extract_field{extract_expr->get_field()};
  const auto arg_range = getExpressionRange(
      extract_expr->get_from_expr(), query_infos, executor, simple_quals);
  const bool has_nulls =
      arg_range.getType() == ExpressionRangeType::Invalid || arg_range.hasNulls();
  const auto& extract_expr_type = extract_expr->get_from_expr()->type();
  auto unit = extract_expr_type->isTimestamp()
                  ? extract_expr_type->as<hdk::ir::TimestampType>()->unit()
                  : hdk::ir::TimeUnit::kSecond;
  bool is_hpt = extract_expr_type->isTimestamp() && (unit > hdk::ir::TimeUnit::kSecond);
  switch (extract_field) {
    case kYEAR: {
      if (arg_range.getType() == ExpressionRangeType::Invalid) {
        return ExpressionRange::makeInvalidRange();
      }
      CHECK(arg_range.getType() == ExpressionRangeType::Integer);
      const int64_t year_range_min =
          is_hpt ? ExtractFromTime(kYEAR,
                                   arg_range.getIntMin() / hdk::ir::unitsPerSecond(unit))
                 : ExtractFromTime(kYEAR, arg_range.getIntMin());
      const int64_t year_range_max =
          is_hpt ? ExtractFromTime(kYEAR,
                                   arg_range.getIntMax() / hdk::ir::unitsPerSecond(unit))
                 : ExtractFromTime(kYEAR, arg_range.getIntMax());
      return ExpressionRange::makeIntRange(
          year_range_min, year_range_max, 0, arg_range.hasNulls());
    }
    case kEPOCH:
    case kDATEEPOCH:
      return arg_range;
    case kQUARTERDAY:
    case kQUARTER:
      return ExpressionRange::makeIntRange(1, 4, 0, has_nulls);
    case kMONTH:
      return ExpressionRange::makeIntRange(1, 12, 0, has_nulls);
    case kDAY:
      return ExpressionRange::makeIntRange(1, 31, 0, has_nulls);
    case kHOUR:
      return ExpressionRange::makeIntRange(0, 23, 0, has_nulls);
    case kMINUTE:
      return ExpressionRange::makeIntRange(0, 59, 0, has_nulls);
    case kSECOND:
      return ExpressionRange::makeIntRange(0, 60, 0, has_nulls);
    case kMILLISECOND:
      return ExpressionRange::makeIntRange(0, 999, 0, has_nulls);
    case kMICROSECOND:
      return ExpressionRange::makeIntRange(0, 999999, 0, has_nulls);
    case kNANOSECOND:
      return ExpressionRange::makeIntRange(0, 999999999, 0, has_nulls);
    case kDOW:
      return ExpressionRange::makeIntRange(0, 6, 0, has_nulls);
    case kISODOW:
      return ExpressionRange::makeIntRange(1, 7, 0, has_nulls);
    case kDOY:
      return ExpressionRange::makeIntRange(1, 366, 0, has_nulls);
    case kWEEK:
    case kWEEK_SUNDAY:
    case kWEEK_SATURDAY:
      return ExpressionRange::makeIntRange(1, 53, 0, has_nulls);
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(
    const hdk::ir::DatetruncExpr* datetrunc_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  const auto arg_range = getExpressionRange(
      datetrunc_expr->get_from_expr(), query_infos, executor, simple_quals);
  if (arg_range.getType() == ExpressionRangeType::Invalid) {
    return ExpressionRange::makeInvalidRange();
  }
  auto datetrunc_expr_type = datetrunc_expr->get_from_expr()->type();
  auto unit = datetrunc_expr_type->isTimestamp()
                  ? datetrunc_expr_type->as<hdk::ir::TimestampType>()->unit()
                  : hdk::ir::TimeUnit::kSecond;
  const int64_t min_ts = DateTimeTranslator::getDateTruncConstantValue(
      arg_range.getIntMin(), datetrunc_expr->get_field(), datetrunc_expr_type);
  const int64_t max_ts = DateTimeTranslator::getDateTruncConstantValue(
      arg_range.getIntMax(), datetrunc_expr->get_field(), datetrunc_expr_type);
  const int64_t bucket = get_conservative_datetrunc_bucket(datetrunc_expr->get_field()) *
                         hdk::ir::unitsPerSecond(unit);

  return ExpressionRange::makeIntRange(min_ts, max_ts, bucket, arg_range.hasNulls());
}

ExpressionRange getExpressionRange(
    const hdk::ir::WidthBucketExpr* width_bucket_expr,
    const std::vector<InputTableInfo>& query_infos,
    const Executor* executor,
    boost::optional<std::list<hdk::ir::ExprPtr>> simple_quals) {
  auto target_value_expr = width_bucket_expr->get_target_value();
  auto target_value_range = getExpressionRange(target_value_expr, query_infos, executor);
  auto target_type = target_value_expr->type();
  if (width_bucket_expr->is_constant_expr() &&
      target_value_range.getType() != ExpressionRangeType::Invalid) {
    auto const_target_value = dynamic_cast<const hdk::ir::Constant*>(target_value_expr);
    if (const_target_value) {
      if (const_target_value->get_is_null()) {
        // null constant, return default width_bucket range
        return ExpressionRange::makeIntRange(
            0, width_bucket_expr->get_partition_count_val(), 0, true);
      } else {
        CHECK(target_value_range.getFpMax() == target_value_range.getFpMin());
        auto target_value_bucket =
            width_bucket_expr->compute_bucket(target_value_range.getFpMax(), target_type);
        return ExpressionRange::makeIntRange(
            target_value_bucket, target_value_bucket, 0, target_value_range.hasNulls());
      }
    }
    // compute possible bucket range based on lower and upper bound constants
    // to elucidate a target bucket range
    const auto target_value_range_with_qual =
        getExpressionRange(target_value_expr, query_infos, executor, simple_quals);
    auto compute_bucket_range = [&width_bucket_expr](const ExpressionRange& target_range,
                                                     const hdk::ir::Type* type) {
      // we casted bucket bound exprs to double
      auto lower_bound_bucket =
          width_bucket_expr->compute_bucket<double>(target_range.getFpMin(), type);
      auto upper_bound_bucket =
          width_bucket_expr->compute_bucket<double>(target_range.getFpMax(), type);
      return ExpressionRange::makeIntRange(
          lower_bound_bucket, upper_bound_bucket, 0, target_range.hasNulls());
    };
    auto res_range = compute_bucket_range(target_value_range_with_qual, target_type);
    // check target_value expression's col range to be not nullable iff it has its filter
    // expression i.e., in simple_quals
    // todo (yoonmin) : need to search simple_quals to cover more cases?
    if (target_value_range.getFpMin() < target_value_range_with_qual.getFpMin() ||
        target_value_range.getFpMax() > target_value_range_with_qual.getFpMax()) {
      res_range.setNulls(false);
    }
    return res_range;
  } else {
    // we cannot determine a possibility of skipping oob check safely
    const bool has_nulls = target_value_range.getType() == ExpressionRangeType::Invalid ||
                           target_value_range.hasNulls();
    auto partition_expr_range = getExpressionRange(
        width_bucket_expr->get_partition_count(), query_infos, executor, simple_quals);
    auto res = ExpressionRange::makeIntRange(0, INT32_MAX, 0, has_nulls);
    switch (partition_expr_range.getType()) {
      case ExpressionRangeType::Integer: {
        res.setIntMax(partition_expr_range.getIntMax() + 1);
        break;
      }
      case ExpressionRangeType::Float:
      case ExpressionRangeType::Double: {
        res.setIntMax(static_cast<int64_t>(partition_expr_range.getFpMax()) + 1);
        break;
      }
      default:
        break;
    }
    return res;
  }
}
