/*
 * Copyright 2022 Intel Corporation.
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

#include "QueryEngine/MemoryLayoutBuilder.h"

#include "QueryEngine/CardinalityEstimator.h"
#include "QueryEngine/ColRangeInfo.h"
#include "QueryEngine/OutputBufferInitialization.h"
#include "QueryEngine/UsedColumnsCollector.h"
#include "ResultSet/HyperLogLog.h"

#include <boost/algorithm/cxx11/any_of.hpp>

MemoryLayoutBuilder::MemoryLayoutBuilder(const RelAlgExecutionUnit& ra_exe_unit)
    : ra_exe_unit_(ra_exe_unit) {
  for (const auto& groupby_expr : ra_exe_unit_.groupby_exprs) {
    if (!groupby_expr) {
      continue;
    }
    auto groupby_type = groupby_expr->type();
    if (groupby_type->isText()) {
      throw std::runtime_error(
          "Cannot group by string columns which are not dictionary encoded.");
    }
    if (groupby_type->isBuffer()) {
      throw std::runtime_error("Group by buffer not supported");
    }
  }
}

namespace {

bool has_count_distinct(const RelAlgExecutionUnit& ra_exe_unit, bool bigint_count) {
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = get_target_info(target_expr, bigint_count);
    if (agg_info.is_agg && is_distinct_target(agg_info)) {
      return true;
    }
  }
  return false;
}

bool is_column_range_too_big_for_perfect_hash(const ColRangeInfo& col_range_info,
                                              const int64_t max_entry_count) {
  try {
    return static_cast<int64_t>(checked_int64_t(col_range_info.max) -
                                checked_int64_t(col_range_info.min)) >= max_entry_count;
  } catch (...) {
    return true;
  }
}

bool cardinality_estimate_less_than_column_range(const int64_t cardinality_estimate,
                                                 const ColRangeInfo& col_range_info) {
  try {
    // the cardinality estimate is the size of the baseline hash table. further penalize
    // the baseline hash table by a factor of 2x due to overhead in computing baseline
    // hash. This has the overall effect of penalizing baseline hash over perfect hash by
    // 4x; i.e. if the cardinality of the filtered data is less than 25% of the entry
    // count of the column, we use baseline hash on the filtered set
    return checked_int64_t(cardinality_estimate) * 2 <
           static_cast<int64_t>(checked_int64_t(col_range_info.max) -
                                checked_int64_t(col_range_info.min));
  } catch (...) {
    return false;
  }
}

bool expr_is_rowid(const hdk::ir::Expr* expr) {
  const auto col = dynamic_cast<const hdk::ir::ColumnVar*>(expr);
  if (!col) {
    return false;
  }
  return col->isVirtual();
}

ColRangeInfo get_col_range_info(const RelAlgExecutionUnit& ra_exe_unit,
                                const std::vector<InputTableInfo>& query_infos,
                                std::optional<int64_t> group_cardinality_estimation,
                                Executor* executor,
                                const ExecutorDeviceType device_type) {
  if (ra_exe_unit.shuffle_fn) {
    CHECK(!ra_exe_unit.target_exprs.empty());
    // For shuffle COUNT(*) query we use keyless perfect hash.
    if (ra_exe_unit.isShuffleCount()) {
      return {QueryDescriptionType::GroupByPerfectHash,
              0,
              static_cast<int64_t>(ra_exe_unit.shuffle_fn->partitions),
              0,
              false};
    }
    // For actual shuffle we use Projection.
    CHECK(ra_exe_unit.isShuffle());
    return {QueryDescriptionType::Projection, 0, 1, 0, false};
  }
  const Config& config = executor->getConfig();
  // Use baseline layout more eagerly on the GPU if the query uses count distinct,
  // because our HyperLogLog implementation is 4x less memory efficient on GPU.
  // Technically, this only applies to APPROX_COUNT_DISTINCT, but in practice we
  // can expect this to be true anyway for grouped queries since the precise version
  // uses significantly more memory.
  int64_t baseline_threshold = config.exec.group_by.baseline_threshold;
  if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count) &&
      device_type == ExecutorDeviceType::GPU) {
    baseline_threshold = baseline_threshold / 4;
  }
  if (ra_exe_unit.groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality{1};
      bool has_nulls{false};
      for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
        auto col_range_info =
            get_expr_range_info(ra_exe_unit, query_infos, groupby_expr.get(), executor);
        if (col_range_info.hash_type_ != QueryDescriptionType::GroupByPerfectHash) {
          // going through baseline hash if a non-integer type is encountered
          return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
        }
        auto crt_col_cardinality = col_range_info.getBucketedCardinality();
        CHECK_GE(crt_col_cardinality, 0);
        cardinality *= crt_col_cardinality;
        if (col_range_info.has_nulls) {
          has_nulls = true;
        }
      }
      // For zero or high cardinalities, use baseline layout.
      if (!cardinality || cardinality > baseline_threshold) {
        return {QueryDescriptionType::GroupByBaselineHash,
                0,
                int64_t(cardinality),
                0,
                has_nulls};
      }
      return {QueryDescriptionType::GroupByPerfectHash,
              0,
              int64_t(cardinality),
              0,
              has_nulls};
    } catch (...) {  // overflow when computing cardinality
      return {QueryDescriptionType::GroupByBaselineHash,
              0,
              std::numeric_limits<int64_t>::max(),
              0,
              false};
    }
  }
  // For single column groupby on high timestamps, force baseline hash due to wide ranges
  // we are likely to encounter when applying quals to the expression range
  // TODO: consider allowing TIMESTAMP(9) (nanoseconds) with quals to use perfect hash if
  // the range is small enough
  if (ra_exe_unit.groupby_exprs.front() &&
      ra_exe_unit.groupby_exprs.front()->type()->isTimestamp() &&
      ra_exe_unit.groupby_exprs.front()->type()->as<hdk::ir::TimestampType>()->unit() >
          hdk::ir::TimeUnit::kSecond &&
      ra_exe_unit.simple_quals.size() > 0) {
    return {QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
  }
  const auto col_range_info = get_expr_range_info(
      ra_exe_unit, query_infos, ra_exe_unit.groupby_exprs.front().get(), executor);
  if (!ra_exe_unit.groupby_exprs.front()) {
    return col_range_info;
  }
  static const int64_t MAX_BUFFER_SIZE = 1 << 30;
  const int64_t col_count =
      ra_exe_unit.groupby_exprs.size() + ra_exe_unit.target_exprs.size();
  int64_t max_entry_count = MAX_BUFFER_SIZE / (col_count * sizeof(int64_t));
  if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count)) {
    max_entry_count = std::min(max_entry_count, baseline_threshold);
  }
  auto groupby_expr_type = ra_exe_unit.groupby_exprs.front()->type();
  if (groupby_expr_type->isExtDictionary() && !col_range_info.bucket) {
    const bool has_filters =
        !ra_exe_unit.quals.empty() || !ra_exe_unit.simple_quals.empty();
    if (has_filters &&
        is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count)) {
      // if filters are present, we can use the filter to narrow the cardinality of the
      // group by in the case of ranges too big for perfect hash. Otherwise, we are better
      // off attempting perfect hash (since we know the range will be made of
      // monotonically increasing numbers from min to max for dictionary encoded strings)
      // and failing later due to excessive memory use.
      // Check the conditions where baseline hash can provide a performance increase and
      // return baseline hash (potentially forcing an estimator query) as the range type.
      // Otherwise, return col_range_info which will likely be perfect hash, though could
      // be baseline from a previous call of this function prior to the estimator query.
      if (!ra_exe_unit.sort_info.order_entries.empty()) {
        // TODO(adb): allow some sorts to pass through this block by centralizing sort
        // algorithm decision making
        if (has_count_distinct(ra_exe_unit, config.exec.group_by.bigint_count) &&
            is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count)) {
          // always use baseline hash for column range too big for perfect hash with count
          // distinct descriptors. We will need 8GB of CPU memory minimum for the perfect
          // hash group by in this case.
          return {QueryDescriptionType::GroupByBaselineHash,
                  col_range_info.min,
                  col_range_info.max,
                  0,
                  col_range_info.has_nulls};
        } else {
          // use original col range for sort
          return col_range_info;
        }
      }
      // if filters are present and the filtered range is less than the cardinality of
      // the column, consider baseline hash
      if (!group_cardinality_estimation ||
          cardinality_estimate_less_than_column_range(*group_cardinality_estimation,
                                                      col_range_info)) {
        return {QueryDescriptionType::GroupByBaselineHash,
                col_range_info.min,
                col_range_info.max,
                0,
                col_range_info.has_nulls};
      }
    }
  } else if ((!expr_is_rowid(ra_exe_unit.groupby_exprs.front().get())) &&
             is_column_range_too_big_for_perfect_hash(col_range_info, max_entry_count) &&
             !col_range_info.bucket) {
    return {QueryDescriptionType::GroupByBaselineHash,
            col_range_info.min,
            col_range_info.max,
            0,
            col_range_info.has_nulls};
  }
  return col_range_info;
}

/**
 * This function goes through all target expressions and answers two questions:
 * 1. Is it possible to have keyless hash?
 * 2. If yes to 1, then what aggregate expression should be considered to represent the
 * key's presence, if needed (e.g., in detecting empty entries in the result set).
 *
 * NOTE: Keyless hash is only valid with single-column group by at the moment.
 *
 */
KeylessInfo get_keyless_info(const RelAlgExecutionUnit& ra_exe_unit,
                             const std::vector<InputTableInfo>& query_infos,
                             const bool is_group_by,
                             Executor* executor) {
  // Shuffle counters always go keyless.
  if (ra_exe_unit.isShuffleCount()) {
    return {true, 0};
  }

  bool keyless{true}, found{false};
  int32_t num_agg_expr{0};
  int32_t index{0};
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info =
        get_target_info(target_expr, executor->getConfig().exec.group_by.bigint_count);
    const auto chosen_type = get_compact_type(agg_info);
    if (agg_info.is_agg) {
      num_agg_expr++;
    }
    if (!found && agg_info.is_agg && !is_distinct_target(agg_info)) {
      auto agg_expr = dynamic_cast<const hdk::ir::AggExpr*>(target_expr);
      CHECK(agg_expr);
      const auto arg_expr = agg_arg(target_expr);
      const bool float_argument_input = takes_float_argument(agg_info);
      switch (agg_info.agg_kind) {
        case hdk::ir::AggType::kAvg:
          ++index;
          if (arg_expr && arg_expr->type()->nullable()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
                expr_range_info.hasNulls()) {
              break;
            }
          }
          found = true;
          break;
        case hdk::ir::AggType::kCount:
          if (arg_expr && arg_expr->type()->nullable()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
                expr_range_info.hasNulls()) {
              break;
            }
          }
          found = true;
          break;
        case hdk::ir::AggType::kSum: {
          auto arg_type = arg_expr->type();
          if (constrained_not_null(arg_expr, ra_exe_unit.quals)) {
            arg_type = arg_type->withNullable(false);
          }
          if (arg_type->nullable()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            if (expr_range_info.getType() != ExpressionRangeType::Invalid &&
                !expr_range_info.hasNulls()) {
              found = true;
            }
          } else {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos, executor);
            switch (expr_range_info.getType()) {
              case ExpressionRangeType::Float:
              case ExpressionRangeType::Double:
                if (expr_range_info.getFpMax() < 0 || expr_range_info.getFpMin() > 0) {
                  found = true;
                }
                break;
              case ExpressionRangeType::Integer:
                if (expr_range_info.getIntMax() < 0 || expr_range_info.getIntMin() > 0) {
                  found = true;
                }
                break;
              default:
                break;
            }
          }
          break;
        }
        case hdk::ir::AggType::kMin: {
          CHECK(agg_expr && agg_expr->arg());
          auto arg_type = agg_expr->arg()->type();
          if (arg_type->isString() || arg_type->isExtDictionary() ||
              arg_type->isBuffer()) {
            break;
          }
          auto expr_range_info =
              getExpressionRange(agg_expr->arg(), query_infos, executor);
          auto init_max = get_agg_initial_val(agg_info.agg_kind,
                                              chosen_type,
                                              is_group_by || float_argument_input,
                                              float_argument_input ? sizeof(float) : 8);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              auto double_max =
                  *reinterpret_cast<const double*>(may_alias_ptr(&init_max));
              if (expr_range_info.getFpMax() < double_max) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              if (expr_range_info.getIntMax() < init_max) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        case hdk::ir::AggType::kMax: {
          CHECK(agg_expr && agg_expr->arg());
          auto arg_type = agg_expr->arg()->type();
          if (arg_type->isString() || arg_type->isExtDictionary() ||
              arg_type->isBuffer()) {
            break;
          }
          auto expr_range_info =
              getExpressionRange(agg_expr->arg(), query_infos, executor);
          // NULL sentinel and init value for kMax are identical, which results in
          // ambiguity in detecting empty keys in presence of nulls.
          if (expr_range_info.getType() == ExpressionRangeType::Invalid ||
              expr_range_info.hasNulls()) {
            break;
          }
          auto init_min = get_agg_initial_val(agg_info.agg_kind,
                                              chosen_type,
                                              is_group_by || float_argument_input,
                                              float_argument_input ? sizeof(float) : 8);
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              auto double_min =
                  *reinterpret_cast<const double*>(may_alias_ptr(&init_min));
              if (expr_range_info.getFpMin() > double_min) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              if (expr_range_info.getIntMin() > init_min) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        default:
          keyless = false;
          break;
      }
    }
    if (!keyless) {
      break;
    }
    if (!found) {
      ++index;
    }
  }

  // shouldn't use keyless for projection only
  return {
      keyless && found,
      index,
  };
}

CountDistinctDescriptors init_count_distinct_descriptors(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const ExecutorDeviceType device_type,
    Executor* executor,
    size_t group_by_slots_count,
    QueryDescriptionType group_by_hash_type) {
  CountDistinctDescriptors count_distinct_descriptors;
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    auto agg_info =
        get_target_info(target_expr, executor->getConfig().exec.group_by.bigint_count);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg);
      CHECK(agg_info.agg_kind == hdk::ir::AggType::kCount ||
            agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct);
      const auto agg_expr = static_cast<const hdk::ir::AggExpr*>(target_expr);
      auto arg_type = agg_expr->arg()->type();
      if (arg_type->isText()) {
        throw std::runtime_error(
            "Strings must be dictionary-encoded for COUNT(DISTINCT).");
      }
      if (agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct &&
          arg_type->isBuffer()) {
        throw std::runtime_error("APPROX_COUNT_DISTINCT on arrays not supported yet");
      }
      ColRangeInfo no_range_info{QueryDescriptionType::Projection, 0, 0, 0, false};
      auto arg_range_info =
          arg_type->isFloatingPoint()
              ? no_range_info
              : get_expr_range_info(ra_exe_unit, query_infos, agg_expr->arg(), executor);
      CountDistinctImplType count_distinct_impl_type{CountDistinctImplType::HashSet};
      int64_t bitmap_sz_bits{0};
      if (agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct) {
        const auto error_rate =
            agg_expr->arg1() ? agg_expr->arg1()->as<hdk::ir::Constant>() : nullptr;
        if (error_rate) {
          CHECK(error_rate->type()->isInt32());
          CHECK_GE(error_rate->value().intval, 1);
          bitmap_sz_bits = hll_size_for_rate(error_rate->value().smallintval);
        } else {
          bitmap_sz_bits = executor->getConfig().exec.group_by.hll_precision_bits;
        }
      }
      if (arg_range_info.isEmpty()) {
        count_distinct_descriptors.emplace_back(CountDistinctDescriptor{
            CountDistinctImplType::Bitmap,
            0,
            64,
            agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct,
            device_type,
            1});
        continue;
      }
      if (arg_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
          !arg_type->isBuffer()) {  // TODO(alex): allow bitmap
                                    // implementation for arrays
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        if (agg_info.agg_kind == hdk::ir::AggType::kCount) {
          bitmap_sz_bits = arg_range_info.max - arg_range_info.min + 1;

          if (group_by_hash_type == QueryDescriptionType::GroupByBaselineHash) {
            const int64_t MAX_TOTAL_BITMAPS_BITS = 8 * 8 * 1000 * 1000 * 1000LL;  // 8GB
            int64_t total_bitmaps_size = bitmap_sz_bits * group_by_slots_count / 2;

            if (total_bitmaps_size <= 0 || total_bitmaps_size > MAX_TOTAL_BITMAPS_BITS) {
              count_distinct_impl_type = CountDistinctImplType::HashSet;
            }
          } else {
            const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000LL};
            if (bitmap_sz_bits <= 0 || bitmap_sz_bits > MAX_BITMAP_BITS) {
              count_distinct_impl_type = CountDistinctImplType::HashSet;
            }
          }
        }
      }
      if (agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct &&
          count_distinct_impl_type == CountDistinctImplType::HashSet &&
          !arg_type->isArray()) {
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
      }

      if (executor->getConfig().exec.watchdog.enable && !(arg_range_info.isEmpty()) &&
          count_distinct_impl_type == CountDistinctImplType::HashSet) {
        throw WatchdogException("Cannot use a fast path for COUNT distinct");
      }
      const auto sub_bitmap_count =
          get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit, device_type);
      count_distinct_descriptors.emplace_back(CountDistinctDescriptor{
          count_distinct_impl_type,
          arg_range_info.min,
          bitmap_sz_bits,
          agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct,
          device_type,
          sub_bitmap_count});
    } else {
      count_distinct_descriptors.emplace_back(CountDistinctDescriptor{
          CountDistinctImplType::Invalid, 0, 0, false, device_type, 0});
    }
  }
  return count_distinct_descriptors;
}

template <class T>
inline std::vector<int8_t> get_col_byte_widths(const T& col_expr_list,
                                               bool bigint_count) {
  std::vector<int8_t> col_widths;
  size_t col_expr_idx = 0;
  for (const auto& col_expr : col_expr_list) {
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = get_target_info(col_expr, bigint_count);
      const auto chosen_type = get_compact_type(agg_info);
      if (chosen_type->isString() || chosen_type->isArray()) {
        col_widths.push_back(sizeof(int64_t));
        col_widths.push_back(sizeof(int64_t));
        ++col_expr_idx;
        continue;
      }
      const auto col_expr_bitwidth = get_bit_width(chosen_type);
      CHECK_EQ(size_t(0), col_expr_bitwidth % 8);
      col_widths.push_back(static_cast<int8_t>(col_expr_bitwidth >> 3));
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == hdk::ir::AggType::kAvg) {
        CHECK(agg_info.is_agg);
        col_widths.push_back(sizeof(int64_t));
      }
    }
    ++col_expr_idx;
  }
  return col_widths;
}

bool is_int_and_no_bigger_than(const hdk::ir::Type* type, const size_t byte_width) {
  if (!type->isInteger()) {
    return false;
  }
  return get_bit_width(type) <= (byte_width * 8);
}

int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<InputTableInfo>& query_infos,
                                 const int8_t crt_min_byte_width,
                                 bool bigint_count) {
  // Currently, we cannot handle 32-bit shuffle counters.
  if (bigint_count || ra_exe_unit.isShuffleCount()) {
    return sizeof(int64_t);
  }
  int8_t compact_width{0};
  auto col_it = ra_exe_unit.input_col_descs.begin();
  auto const end = ra_exe_unit.input_col_descs.end();
  int unnest_array_col_id{std::numeric_limits<int>::min()};
  for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto uoper = dynamic_cast<const hdk::ir::UOper*>(groupby_expr.get());
    if (uoper && uoper->isUnnest()) {
      auto arg_type = uoper->operand()->type();
      CHECK(arg_type->isArray());
      auto elem_type = arg_type->as<hdk::ir::ArrayBaseType>()->elemType();
      if (elem_type->isExtDictionary()) {
        unnest_array_col_id = (*col_it)->getColId();
      } else {
        compact_width = crt_min_byte_width;
        break;
      }
    }
    if (col_it != end) {
      ++col_it;
    }
  }
  if (!compact_width &&
      (ra_exe_unit.groupby_exprs.size() != 1 || !ra_exe_unit.groupby_exprs.front())) {
    compact_width = crt_min_byte_width;
  }
  if (!compact_width) {
    col_it = ra_exe_unit.input_col_descs.begin();
    std::advance(col_it, ra_exe_unit.groupby_exprs.size());
    for (const auto target : ra_exe_unit.target_exprs) {
      auto type = target->type();
      const auto agg = target->as<hdk::ir::AggExpr>();
      if (agg && agg->arg()) {
        compact_width = crt_min_byte_width;
        break;
      }

      if (agg) {
        CHECK_EQ(hdk::ir::AggType::kCount, agg->aggType());
        CHECK(!agg->isDistinct());
        if (col_it != end) {
          ++col_it;
        }
        continue;
      }

      if (is_int_and_no_bigger_than(type, 4) || (type->isExtDictionary())) {
        if (col_it != end) {
          ++col_it;
        }
        continue;
      }

      const auto uoper = target->as<hdk::ir::UOper>();
      if (uoper && uoper->isUnnest() && (*col_it)->getColId() == unnest_array_col_id) {
        auto arg_type = uoper->operand()->type();
        CHECK(arg_type->isArray());
        auto elem_type = arg_type->as<hdk::ir::ArrayBaseType>()->elemType();
        if (elem_type->isExtDictionary()) {
          if (col_it != end) {
            ++col_it;
          }
          continue;
        }
      }

      compact_width = crt_min_byte_width;
      break;
    }
  }
  if (!compact_width) {
    size_t total_tuples{0};
    for (const auto& qi : query_infos) {
      total_tuples += qi.info.getNumTuples();
    }
    return total_tuples <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                   unnest_array_col_id != std::numeric_limits<int>::min()
               ? 4
               : crt_min_byte_width;
  } else {
    // TODO(miyu): relax this condition to allow more cases just w/o padding
    for (auto wid : get_col_byte_widths(ra_exe_unit.target_exprs, bigint_count)) {
      compact_width = std::max(compact_width, wid);
    }
    return compact_width;
  }
}

bool is_valid_int32_range(const ExpressionRange& range) {
  return range.getIntMin() > INT32_MIN && range.getIntMax() < EMPTY_KEY_32 - 1;
}

int8_t pick_baseline_key_component_width(const ExpressionRange& range,
                                         const size_t group_col_width) {
  if (range.getType() == ExpressionRangeType::Invalid) {
    return sizeof(int64_t);
  }
  switch (range.getType()) {
    case ExpressionRangeType::Integer:
      if (group_col_width == sizeof(int64_t) && range.hasNulls()) {
        return sizeof(int64_t);
      }
      return is_valid_int32_range(range) ? sizeof(int32_t) : sizeof(int64_t);
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double:
      return sizeof(int64_t);  // No compaction for floating point yet.
    default:
      UNREACHABLE();
  }
  return sizeof(int64_t);
}

// TODO(miyu): make sure following setting of compact width is correct in all cases.
int8_t pick_baseline_key_width(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<InputTableInfo>& query_infos,
                               const Executor* executor) {
  int8_t compact_width{4};
  for (const auto& groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto expr_range = getExpressionRange(groupby_expr.get(), query_infos, executor);
    compact_width = std::max(
        compact_width,
        pick_baseline_key_component_width(expr_range, groupby_expr->type()->size()));
  }
  return compact_width;
}

bool use_streaming_top_n(const RelAlgExecutionUnit& ra_exe_unit,
                         const bool output_columnar,
                         bool streaming_topn_max) {
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    if (dynamic_cast<const hdk::ir::AggExpr*>(target_expr)) {
      return false;
    }
    if (dynamic_cast<const hdk::ir::WindowFunction*>(target_expr)) {
      return false;
    }
  }

  // TODO: Allow streaming top n for columnar output
  if (!output_columnar && ra_exe_unit.sort_info.order_entries.size() == 1 &&
      ra_exe_unit.sort_info.limit &&
      ra_exe_unit.sort_info.algorithm == SortAlgorithm::StreamingTopN) {
    const auto only_order_entry = ra_exe_unit.sort_info.order_entries.front();
    CHECK_GT(only_order_entry.tle_no, int(0));
    CHECK_LE(static_cast<size_t>(only_order_entry.tle_no),
             ra_exe_unit.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit.target_exprs[only_order_entry.tle_no - 1];
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    if ((order_entry_expr->type()->isNumber() ||
         order_entry_expr->type()->isDateTime()) &&
        n <= streaming_topn_max) {
      return true;
    }
  }

  return false;
}

std::vector<int64_t> target_expr_group_by_indices(
    const std::list<hdk::ir::ExprPtr>& groupby_exprs,
    const std::vector<const hdk::ir::Expr*>& target_exprs) {
  std::vector<int64_t> indices(target_exprs.size(), -1);
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    if (dynamic_cast<const hdk::ir::AggExpr*>(target_expr)) {
      continue;
    }
    const auto var_expr = dynamic_cast<const hdk::ir::Var*>(target_expr);
    if (var_expr && var_expr->whichRow() == hdk::ir::Var::kGROUPBY) {
      indices[target_idx] = var_expr->varNo() - 1;
      continue;
    }
  }
  return indices;
}

std::vector<int64_t> target_expr_proj_indices(const RelAlgExecutionUnit& ra_exe_unit,
                                              SchemaProviderPtr schema_provider) {
  if (ra_exe_unit.input_descs.size() > 1 ||
      !ra_exe_unit.sort_info.order_entries.empty()) {
    return {};
  }
  std::vector<int64_t> target_indices(ra_exe_unit.target_exprs.size(), -1);
  UsedColumnsCollector columns_collector;
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
    columns_collector.visit(simple_qual.get());
  }
  for (const auto& qual : ra_exe_unit.quals) {
    columns_collector.visit(qual.get());
  }
  for (const auto& target : ra_exe_unit.target_exprs) {
    const auto col_var = dynamic_cast<const hdk::ir::ColumnVar*>(target);
    if (col_var && !col_var->isVirtual()) {
      continue;
    }
    columns_collector.visit(target);
  }
  const auto& used_columns = columns_collector.result();
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size();
       ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    CHECK(target_expr);
    auto type = target_expr->type();
    // TODO: add proper lazy fetch for varlen types in result set
    if (type->isString() || type->isArray()) {
      continue;
    }
    const auto col_var = dynamic_cast<const hdk::ir::ColumnVar*>(target_expr);
    if (!col_var) {
      continue;
    }
    if (used_columns.find(col_var->columnId()) == used_columns.end()) {
      // setting target index to be zero so that later it can be decoded properly (in lazy
      // fetch, the zeroth target index indicates the corresponding rowid column for the
      // projected entry)
      target_indices[target_idx] = 0;
    }
  }
  return target_indices;
}

bool anyOf(std::vector<const hdk::ir::Expr*> const& target_exprs,
           hdk::ir::AggType agg_kind) {
  return boost::algorithm::any_of(target_exprs, [agg_kind](hdk::ir::Expr const* expr) {
    auto const* const agg = dynamic_cast<hdk::ir::AggExpr const*>(expr);
    return agg && agg->aggType() == agg_kind;
  });
}

std::unique_ptr<QueryMemoryDescriptor> build_query_memory_descriptor(
    const Executor* executor,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const ColRangeInfo& col_range_info,
    const KeylessInfo& keyless_info,
    const bool allow_multifrag,
    const ExecutorDeviceType device_type,
    const int8_t crt_min_byte_width,
    const bool sort_on_gpu_hint,
    const size_t max_groups_buffer_entry_count,
    const CountDistinctDescriptors count_distinct_descriptors,
    const bool must_use_baseline_sort,
    const bool output_columnar_hint,
    const bool streaming_top_n_hint) {
  auto group_col_widths = get_col_byte_widths(
      ra_exe_unit.groupby_exprs, executor->getConfig().exec.group_by.bigint_count);
  const bool is_group_by{!group_col_widths.empty()};

  auto col_slot_context = ColSlotContext(
      ra_exe_unit.target_exprs, {}, executor->getConfig().exec.group_by.bigint_count);

  const auto min_slot_size =
      pick_target_compact_width(ra_exe_unit,
                                query_infos,
                                crt_min_byte_width,
                                executor->getConfig().exec.group_by.bigint_count);

  col_slot_context.setAllSlotsPaddedSize(min_slot_size);
  col_slot_context.validate();

  if (!is_group_by) {
    CHECK(!must_use_baseline_sort);

    return std::make_unique<QueryMemoryDescriptor>(
        executor->getDataMgr(),
        executor->getConfigPtr(),
        query_infos,
        false,
        allow_multifrag,
        false,
        false,
        -1,
        ColRangeInfo{ra_exe_unit.estimator ? QueryDescriptionType::Estimator
                                           : QueryDescriptionType::NonGroupedAggregate,
                     0,
                     0,
                     0,
                     false},
        col_slot_context,
        std::vector<int8_t>{},
        /*group_col_compact_width=*/0,
        std::vector<int64_t>{},
        /*entry_count=*/1,
        count_distinct_descriptors,
        false,
        output_columnar_hint,
        must_use_baseline_sort,
        /*use_streaming_top_n=*/false);
  }

  size_t entry_count = 1;
  auto actual_col_range_info = col_range_info;
  bool interleaved_bins_on_gpu = false;
  bool keyless_hash = false;
  bool streaming_top_n = false;
  int8_t group_col_compact_width = 0;
  int32_t idx_target_as_key = -1;
  auto output_columnar = output_columnar_hint;
  std::vector<int64_t> target_groupby_indices;

  switch (col_range_info.hash_type_) {
    case QueryDescriptionType::GroupByPerfectHash: {
      // keyless hash: whether or not group columns are stored at the beginning of the
      // output buffer
      keyless_hash =
          (!sort_on_gpu_hint ||
           !QueryMemoryDescriptor::many_entries(
               col_range_info.max, col_range_info.min, col_range_info.bucket)) &&
          !col_range_info.bucket && !must_use_baseline_sort && keyless_info.keyless;

      // if keyless, then this target index indicates wheter an entry is empty or not
      // (acts as a key)
      idx_target_as_key = keyless_info.target_index;

      if (ra_exe_unit.isShuffleCount() || group_col_widths.size() > 1) {
        // col range info max contains the expected cardinality of the output
        entry_count = static_cast<size_t>(actual_col_range_info.max);
        actual_col_range_info.bucket = 0;
      } else {
        // single column perfect hash
        entry_count = std::max(col_range_info.getBucketedCardinality(), int64_t(1));
        const size_t interleaved_max_threshold{512};

        if (must_use_baseline_sort) {
          target_groupby_indices = target_expr_group_by_indices(ra_exe_unit.groupby_exprs,
                                                                ra_exe_unit.target_exprs);
          col_slot_context =
              ColSlotContext(ra_exe_unit.target_exprs,
                             target_groupby_indices,
                             executor->getConfig().exec.group_by.bigint_count);
        }

        bool has_varlen_sample_agg = false;
        for (const auto& target_expr : ra_exe_unit.target_exprs) {
          if (target_expr->containsAgg()) {
            const auto agg_expr = target_expr->as<hdk::ir::AggExpr>();
            CHECK(agg_expr);
            if (agg_expr->aggType() == hdk::ir::AggType::kSample &&
                (agg_expr->type()->isString() || agg_expr->type()->isArray())) {
              has_varlen_sample_agg = true;
              break;
            }
          }
        }

        interleaved_bins_on_gpu = keyless_hash && !has_varlen_sample_agg &&
                                  (entry_count <= interleaved_max_threshold) &&
                                  (device_type == ExecutorDeviceType::GPU) &&
                                  QueryMemoryDescriptor::countDescriptorsLogicallyEmpty(
                                      count_distinct_descriptors) &&
                                  !output_columnar;
      }
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      entry_count = max_groups_buffer_entry_count;
      target_groupby_indices = target_expr_group_by_indices(ra_exe_unit.groupby_exprs,
                                                            ra_exe_unit.target_exprs);
      col_slot_context = ColSlotContext(ra_exe_unit.target_exprs,
                                        target_groupby_indices,
                                        executor->getConfig().exec.group_by.bigint_count);

      group_col_compact_width =
          output_columnar ? 8
                          : pick_baseline_key_width(ra_exe_unit, query_infos, executor);

      actual_col_range_info =
          ColRangeInfo{QueryDescriptionType::GroupByBaselineHash, 0, 0, 0, false};
      break;
    }
    case QueryDescriptionType::Projection: {
      CHECK(!must_use_baseline_sort);

      bool streaming_top_n_supported_by_platform =
          device_type == ExecutorDeviceType::CPU ||
          (executor->getDataMgr() && executor->getDataMgr()->getGpuMgr() &&
           executor->getDataMgr()->getGpuMgr()->getPlatform() != GpuMgrPlatform::L0);
      if (streaming_top_n_hint &&
          use_streaming_top_n(ra_exe_unit,
                              output_columnar,
                              executor->getConfig().exec.streaming_topn_max) &&
          streaming_top_n_supported_by_platform) {
        streaming_top_n = true;
        entry_count = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
      } else {
        entry_count = ra_exe_unit.scan_limit ? static_cast<size_t>(ra_exe_unit.scan_limit)
                                             : max_groups_buffer_entry_count;
      }

      target_groupby_indices =
          executor->isLazyFetchAllowed()
              ? target_expr_proj_indices(ra_exe_unit, executor->getSchemaProvider())
              : std::vector<int64_t>{};

      col_slot_context = ColSlotContext(ra_exe_unit.target_exprs,
                                        target_groupby_indices,
                                        executor->getConfig().exec.group_by.bigint_count);
      break;
    }
    default:
      UNREACHABLE() << "Unknown query type";
  }

  auto approx_quantile =
      anyOf(ra_exe_unit.target_exprs, hdk::ir::AggType::kApproxQuantile);
  return std::make_unique<QueryMemoryDescriptor>(executor->getDataMgr(),
                                                 executor->getConfigPtr(),
                                                 query_infos,
                                                 approx_quantile,
                                                 allow_multifrag,
                                                 keyless_hash,
                                                 interleaved_bins_on_gpu,
                                                 idx_target_as_key,
                                                 actual_col_range_info,
                                                 col_slot_context,
                                                 group_col_widths,
                                                 group_col_compact_width,
                                                 target_groupby_indices,
                                                 entry_count,
                                                 count_distinct_descriptors,
                                                 sort_on_gpu_hint,
                                                 output_columnar,
                                                 must_use_baseline_sort,
                                                 streaming_top_n);
}

std::unique_ptr<QueryMemoryDescriptor> build_query_memory_descriptor(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    const bool sort_on_gpu_hint,
    const bool must_use_baseline_sort,
    const bool output_columnar_hint,
    std::optional<int64_t> group_cardinality_estimation,
    Executor* executor,
    const ExecutorDeviceType device_type) {
  const bool is_group_by{!ra_exe_unit.groupby_exprs.empty()};

  auto col_range_info = get_col_range_info(
      ra_exe_unit, query_infos, group_cardinality_estimation, executor, device_type);

  const auto count_distinct_descriptors =
      init_count_distinct_descriptors(ra_exe_unit,
                                      query_infos,
                                      device_type,
                                      executor,
                                      max_groups_buffer_entry_count,
                                      col_range_info.hash_type_);

  // Non-grouped aggregates do not support accessing aggregated ranges
  // Keyless hash is currently only supported with single-column perfect hash
  const auto keyless_info =
      !(is_group_by &&
        col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash)
          ? KeylessInfo{false, -1}
          : get_keyless_info(ra_exe_unit, query_infos, is_group_by, executor);

  if (executor->getConfig().exec.watchdog.enable &&
      ((col_range_info.hash_type_ == QueryDescriptionType::GroupByBaselineHash &&
        max_groups_buffer_entry_count >
            executor->getConfig().exec.watchdog.baseline_max_groups) ||
       (col_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
        ra_exe_unit.groupby_exprs.size() == 1 &&
        (col_range_info.max - col_range_info.min) /
                std::max(col_range_info.bucket, int64_t(1)) >
            130000000))) {
    throw WatchdogException("Query would use too much memory");
  }
  try {
    return build_query_memory_descriptor(executor,
                                         ra_exe_unit,
                                         query_infos,
                                         col_range_info,
                                         keyless_info,
                                         allow_multifrag,
                                         device_type,
                                         crt_min_byte_width,
                                         sort_on_gpu_hint,
                                         max_groups_buffer_entry_count,
                                         count_distinct_descriptors,
                                         must_use_baseline_sort,
                                         output_columnar_hint,
                                         /*streaming_top_n_hint=*/true);
  } catch (const StreamingTopNOOM& e) {
    LOG(WARNING) << e.what() << " Disabling Streaming Top N.";
    return build_query_memory_descriptor(executor,
                                         ra_exe_unit,
                                         query_infos,
                                         col_range_info,
                                         keyless_info,
                                         allow_multifrag,
                                         device_type,
                                         crt_min_byte_width,
                                         sort_on_gpu_hint,
                                         max_groups_buffer_entry_count,
                                         count_distinct_descriptors,
                                         must_use_baseline_sort,
                                         output_columnar_hint,
                                         /*streaming_top_n_hint=*/false);
  }
}

bool gpu_can_handle_order_entries(const RelAlgExecutionUnit& ra_exe_unit,
                                  const std::vector<InputTableInfo>& query_infos,
                                  const std::list<hdk::ir::OrderEntry>& order_entries,
                                  Executor* executor) {
  if (order_entries.size() > 1) {  // TODO(alex): lift this restriction
    return false;
  }
  for (const auto& order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(static_cast<size_t>(order_entry.tle_no), ra_exe_unit.target_exprs.size());
    const auto target_expr = ra_exe_unit.target_exprs[order_entry.tle_no - 1];
    if (!target_expr->is<hdk::ir::AggExpr>()) {
      return false;
    }
    // TODO(alex): relax the restrictions
    auto agg_expr = target_expr->as<hdk::ir::AggExpr>();
    if (agg_expr->isDistinct() || agg_expr->aggType() == hdk::ir::AggType::kAvg ||
        agg_expr->aggType() == hdk::ir::AggType::kMin ||
        agg_expr->aggType() == hdk::ir::AggType::kMax ||
        agg_expr->aggType() == hdk::ir::AggType::kApproxCountDistinct) {
      return false;
    }
    if (agg_expr->arg()) {
      auto arg_type = agg_expr->arg()->type();
      if (arg_type->isFloatingPoint()) {
        return false;
      }
      auto expr_range_info =
          get_expr_range_info(ra_exe_unit, query_infos, agg_expr->arg(), executor);
      // TOD(adb): QMD not actually initialized here?
      if ((!(expr_range_info.hash_type_ == QueryDescriptionType::GroupByPerfectHash &&
             /* query_mem_desc.getGroupbyColCount() == 1 */ false) ||
           expr_range_info.has_nulls) &&
          order_entry.is_desc == order_entry.nulls_first) {
        return false;
      }
    }
    auto target_type = target_expr->type();
    CHECK(!target_type->isBuffer());
    if (!target_type->isInteger()) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::unique_ptr<QueryMemoryDescriptor> MemoryLayoutBuilder::build(
    const std::vector<InputTableInfo>& query_infos,
    const bool allow_multifrag,
    const size_t max_groups_buffer_entry_count,
    const int8_t crt_min_byte_width,
    const bool output_columnar_hint,
    const bool just_explain,
    std::optional<int64_t> group_cardinality_estimation,
    Executor* executor,
    const ExecutorDeviceType device_type) {
  bool sort_on_gpu_hint =
      (device_type == ExecutorDeviceType::GPU && executor->getDataMgr()->getGpuMgr() &&
       executor->getDataMgr()->getGpuMgr()->getPlatform() != GpuMgrPlatform::L0) &&
      allow_multifrag && !ra_exe_unit_.sort_info.order_entries.empty() &&
      gpu_can_handle_order_entries(
          ra_exe_unit_, query_infos, ra_exe_unit_.sort_info.order_entries, executor);
  // must_use_baseline_sort is true iff we'd sort on GPU with the old algorithm
  // but the total output buffer size would be too big or it's a sharded top query.
  // For the sake of managing risk, use the new result set way very selectively for
  // this case only (alongside the baseline layout we've enabled for a while now).
  bool must_use_baseline_sort = false;
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;
  while (true) {
    query_mem_desc = build_query_memory_descriptor(ra_exe_unit_,
                                                   query_infos,
                                                   allow_multifrag,
                                                   max_groups_buffer_entry_count,
                                                   crt_min_byte_width,
                                                   sort_on_gpu_hint,
                                                   must_use_baseline_sort,
                                                   output_columnar_hint,
                                                   group_cardinality_estimation,
                                                   executor,
                                                   device_type);
    CHECK(query_mem_desc);
    if (query_mem_desc->sortOnGpu() &&
        (query_mem_desc->getBufferSizeBytes(device_type) +
         align_to_int64(query_mem_desc->getEntryCount() * sizeof(int32_t))) >
            2 * 1024 * 1024 * 1024LL) {
      must_use_baseline_sort = true;
      sort_on_gpu_hint = false;
    } else {
      break;
    }
  }

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByBaselineHash &&
      !group_cardinality_estimation && !just_explain) {
    const auto col_range_info = get_col_range_info(
        ra_exe_unit_, query_infos, group_cardinality_estimation, executor, device_type);
    LOG(INFO) << "Request query retry with estimator for group col range ("
              << col_range_info.min << ", " << col_range_info.max << ")";
    throw CardinalityEstimationRequired(col_range_info.max - col_range_info.min);
  }

  return query_mem_desc;
}

size_t MemoryLayoutBuilder::gpuSharedMemorySize(
    QueryMemoryDescriptor* query_mem_desc,
    const GpuMgr* gpu_mgr,
    Executor* executor,
    const ExecutorDeviceType device_type) const {
  if (device_type == ExecutorDeviceType::CPU || !gpu_mgr) {
    return 0;
  }
  /* We only use shared memory strategy if GPU hardware provides native shared
   * memory atomics support. */
  if (!gpu_mgr->hasSharedMemoryAtomicsSupport()) {
    return 0;
  }

  const auto gpu_blocksize = executor->blockSize();
  const auto num_blocks_per_mp = executor->numBlocksPerMP();

  /**
   * To simplify the implementation for practical purposes, we
   * initially provide shared memory support for cases where there are at most as many
   * entries in the output buffer as there are threads within each GPU device. In
   * order to relax this assumption later, we need to add a for loop in generated
   * codes such that each thread loops over multiple entries.
   * TODO: relax this if necessary
   */
  if (gpu_blocksize < query_mem_desc->getEntryCount()) {
    return 0;
  }

  CHECK(query_mem_desc);
  if (query_mem_desc->didOutputColumnar()) {
    return 0;
  }

  const Config& config = executor->getConfig();

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::NonGroupedAggregate &&
      config.exec.group_by.enable_gpu_smem_non_grouped_agg &&
      query_mem_desc->countDistinctDescriptorsLogicallyEmpty()) {
    // skip shared memory usage when dealing with 1) variable length targets, 2)
    // not a COUNT aggregate
    const auto target_infos = target_exprs_to_infos(
        ra_exe_unit_.target_exprs, *query_mem_desc, config.exec.group_by.bigint_count);
    const std::unordered_set<hdk::ir::AggType> supported_aggs{hdk::ir::AggType::kCount};
    auto is_supported = [&supported_aggs](const TargetInfo& ti) {
      return !(ti.type->isString() || ti.type->isArray() ||
               !supported_aggs.count(ti.agg_kind));
    };

    if (std::all_of(target_infos.begin(), target_infos.end(), is_supported)) {
      return query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
    }
  }

  if (query_mem_desc->getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash &&
      config.exec.group_by.enable_gpu_smem_group_by) {
    // Fundamentally, we should use shared memory whenever the output buffer
    // is small enough so that we can fit it in the shared memory and yet expect
    // good occupancy.
    // For now, we allow keyless, row-wise layout, and only for perfect hash
    // group by operations.
    if (query_mem_desc->hasKeylessHash() &&
        query_mem_desc->countDistinctDescriptorsLogicallyEmpty() &&
        !query_mem_desc->useStreamingTopN()) {
      const size_t smem_threshold = config.exec.group_by.gpu_smem_threshold == 0
                                        ? SIZE_MAX
                                        : config.exec.group_by.gpu_smem_threshold;
      const size_t shared_memory_threshold_bytes = std::min(
          smem_threshold,
          gpu_mgr->getMinSharedMemoryPerBlockForAllDevices() / num_blocks_per_mp);
      const auto output_buffer_size =
          query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
      if (output_buffer_size > shared_memory_threshold_bytes) {
        return 0;
      }

      // skip shared memory usage when dealing with 1) variable length targets, 2)
      // non-basic aggregates (COUNT, SUM, MIN, MAX, AVG)
      // TODO: relax this if necessary
      const auto target_infos = target_exprs_to_infos(
          ra_exe_unit_.target_exprs, *query_mem_desc, config.exec.group_by.bigint_count);
      std::unordered_set<hdk::ir::AggType> supported_aggs{hdk::ir::AggType::kCount};
      if (config.exec.group_by.enable_gpu_smem_grouped_non_count_agg) {
        supported_aggs = {hdk::ir::AggType::kCount,
                          hdk::ir::AggType::kMin,
                          hdk::ir::AggType::kMax,
                          hdk::ir::AggType::kSum,
                          hdk::ir::AggType::kAvg};
      }
      auto is_supported = [&supported_aggs](const TargetInfo& ti) {
        return !(ti.type->isString() || ti.type->isArray() ||
                 !supported_aggs.count(ti.agg_kind));
      };

      if (std::all_of(target_infos.begin(), target_infos.end(), is_supported)) {
        return query_mem_desc->getRowSize() * query_mem_desc->getEntryCount();
      }
    }
  }
  return 0;
}
