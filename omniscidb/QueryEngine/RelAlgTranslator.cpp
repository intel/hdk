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

#include "RelAlgTranslator.h"
#include "DeepCopyVisitor.h"
#include "ExpressionRewrite.h"
#include "RelAlgDagBuilder.h"

#include "Analyzer/Analyzer.h"
#include "Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/SessionInfo.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/likely.h"

namespace {

std::pair<Datum, bool> datum_from_scalar_tv(const ScalarTargetValue* scalar_tv,
                                            const hdk::ir::Type* type) noexcept {
  Datum d{0};
  bool is_null_const{false};
  switch (type->id()) {
    case hdk::ir::Type::kBoolean: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_value(type)) {
        is_null_const = true;
      } else {
        d.boolval = *ival;
      }
      break;
    }
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (type->size()) {
        case 1: {
          const auto ival = boost::get<int64_t>(scalar_tv);
          CHECK(ival);
          if (*ival == inline_int_null_value(type)) {
            is_null_const = true;
          } else {
            d.tinyintval = *ival;
          }
          break;
        }
        case 2: {
          const auto ival = boost::get<int64_t>(scalar_tv);
          CHECK(ival);
          if (*ival == inline_int_null_value(type)) {
            is_null_const = true;
          } else {
            d.smallintval = *ival;
          }
          break;
        }
        case 4: {
          const auto ival = boost::get<int64_t>(scalar_tv);
          CHECK(ival);
          if (*ival == inline_int_null_value(type)) {
            is_null_const = true;
          } else {
            d.intval = *ival;
          }
          break;
        }
        case 8: {
          const auto ival = boost::get<int64_t>(scalar_tv);
          CHECK(ival);
          if (*ival == inline_int_null_value(type)) {
            is_null_const = true;
          } else {
            d.bigintval = *ival;
          }
          break;
        }
        default:
          CHECK(false);
      }
      break;
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp: {
      const auto ival = boost::get<int64_t>(scalar_tv);
      CHECK(ival);
      if (*ival == inline_int_null_value(type)) {
        is_null_const = true;
      } else {
        d.bigintval = *ival;
      }
      break;
    }
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kDouble: {
          const auto dval = boost::get<double>(scalar_tv);
          CHECK(dval);
          if (*dval == inline_fp_null_value(type)) {
            is_null_const = true;
          } else {
            d.doubleval = *dval;
          }
          break;
        }
        case hdk::ir::FloatingPointType::kFloat: {
          const auto fval = boost::get<float>(scalar_tv);
          CHECK(fval);
          if (*fval == inline_fp_null_value(type)) {
            is_null_const = true;
          } else {
            d.floatval = *fval;
          }
          break;
        }
        default:
          CHECK(false);
      }
      break;
    case hdk::ir::Type::kExtDictionary:
    case hdk::ir::Type::kText:
    case hdk::ir::Type::kVarChar: {
      auto nullable_sptr = boost::get<NullableString>(scalar_tv);
      CHECK(nullable_sptr);
      if (boost::get<void*>(nullable_sptr)) {
        is_null_const = true;
      } else {
        auto sptr = boost::get<std::string>(nullable_sptr);
        d.stringval = new std::string(*sptr);
      }
      break;
    }
    default:
      CHECK(false) << "Unhandled type: " << type->toString();
  }
  return {d, is_null_const};
}

hdk::ir::ExprPtr translateScalarSubqueryResult(
    std::shared_ptr<const ExecutionResult> result) {
  auto row_set = result->getRows();
  const size_t row_count = row_set->rowCount();
  CHECK_EQ(size_t(1), row_set->colCount());
  auto type = row_set->colType(0);
  if (row_count > size_t(1)) {
    throw std::runtime_error("Scalar sub-query returned multiple rows");
  }
  if (row_count == size_t(0)) {
    if (row_set->isValidationOnlyRes()) {
      Datum d{0};
      return hdk::ir::makeExpr<hdk::ir::Constant>(type, false, d);
    }
    throw std::runtime_error("Scalar sub-query returned no results");
  }
  CHECK_EQ(row_count, size_t(1));
  row_set->moveToBegin();
  auto first_row = row_set->getNextRow(false, false);
  CHECK_EQ(first_row.size(), size_t(1));
  auto scalar_tv = boost::get<ScalarTargetValue>(&first_row[0]);
  if (type->isString() || type->isExtDictionary()) {
    throw std::runtime_error("Scalar sub-queries which return strings not supported");
  }
  Datum d{0};
  bool is_null_const{false};
  std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, type);
  return hdk::ir::makeExpr<hdk::ir::Constant>(type, is_null_const, d);
}

class NormalizerVisitor : public DeepCopyVisitor {
 public:
  NormalizerVisitor(const RelAlgTranslator& translator,
                    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
                    const std::vector<JoinType>& join_types,
                    const Executor* executor)
      : translator_(translator)
      , input_to_nest_level_(input_to_nest_level)
      , join_types_(join_types)
      , executor_(executor) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    auto source = col_ref->getNode();
    auto col_idx = col_ref->getIndex();
    const auto it_rte_idx = input_to_nest_level_.find(source);
    const int rte_idx = it_rte_idx == input_to_nest_level_.end() ? 0 : it_rte_idx->second;
    const auto scan_source = dynamic_cast<const RelScan*>(source);
    const auto& in_metainfo = source->getOutputMetainfo();
    if (scan_source) {
      // We're at leaf (scan) level and not supposed to have input metadata,
      // the name and type information come directly from the catalog.
      CHECK(in_metainfo.empty());
      auto col_info = scan_source->getColumnInfo(col_idx);
      auto col_type = col_info->type;
      if (col_type->isVarChar()) {
        col_type = col_type->ctx().text(col_type->nullable());
      }
      CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
      if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
        col_type = col_type->withNullable(true);
      }
      if (!col_type->equal(col_info->type)) {
        col_info = std::make_shared<ColumnInfo>(col_info->db_id,
                                                col_info->table_id,
                                                col_info->column_id,
                                                col_info->name,
                                                col_type,
                                                col_info->is_rowid);
      }
      return std::make_shared<hdk::ir::ColumnVar>(col_info, rte_idx);
    }
    CHECK_GE(rte_idx, 0);
    CHECK(!in_metainfo.empty()) << "for " << source->toString();
    CHECK_LT(col_idx, in_metainfo.size());
    auto col_type = in_metainfo[col_idx].type();

    if (join_types_.size() > 0) {
      CHECK_LE(static_cast<size_t>(rte_idx), join_types_.size());
      if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT) {
        col_type = col_type->withNullable(true);
      }
    }

    return std::make_shared<hdk::ir::ColumnVar>(
        col_type, -source->getId(), col_idx, rte_idx);
  }

  hdk::ir::ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) const override {
    auto lhs = visit(bin_oper->get_left_operand());
    auto rhs = visit(bin_oper->get_right_operand());
    // Some binary expressions are not results of operation translation. They are
    // not covered in Analyzer normalization.
    if (bin_oper->get_optype() == kARRAY_AT) {
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bin_oper->type(),
          lhs->containsAgg() || rhs->containsAgg(),
          bin_oper->get_optype(),
          bin_oper->get_qualifier(),
          std::move(lhs),
          std::move(rhs));
    }
    return Analyzer::normalizeOperExpr(bin_oper->get_optype(),
                                       bin_oper->get_qualifier(),
                                       std::move(lhs),
                                       std::move(rhs),
                                       executor_);
  }

  hdk::ir::ExprPtr visitUOper(const hdk::ir::UOper* uoper) const override {
    // Casts introduced on DAG build stage might become NOPs.
    if (uoper->get_optype() == kCAST) {
      auto op = visit(uoper->get_operand());
      if (uoper->type()->equal(op->type())) {
        return op;
      }
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          uoper->type(), op->containsAgg(), kCAST, op);
    }
    return DeepCopyVisitor::visitUOper(uoper);
  }

  hdk::ir::ExprPtr visitCaseExpr(const hdk::ir::CaseExpr* case_expr) const override {
    std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> expr_list;
    for (auto& pr : case_expr->get_expr_pair_list()) {
      expr_list.emplace_back(visit(pr.first.get()), visit(pr.second.get()));
    }
    auto else_expr = visit(case_expr->get_else_expr());
    return Analyzer::normalizeCaseExpr(expr_list, else_expr, executor_);
  }

  hdk::ir::ExprPtr visitScalarSubquery(
      const hdk::ir::ScalarSubquery* subquery) const override {
    return translateScalarSubqueryResult(subquery->getNode()->getResult());
  }

  hdk::ir::ExprPtr visitInSubquery(const hdk::ir::InSubquery* subquery) const override {
    return translator_.translateInSubquery(visit(subquery->getArg().get()),
                                           subquery->getNode()->getResult());
  }

 private:
  const RelAlgTranslator& translator_;
  const std::unordered_map<const RelAlgNode*, int> input_to_nest_level_;
  const std::vector<JoinType> join_types_;
  const Executor* executor_;
};

}  // namespace

RelAlgTranslator::RelAlgTranslator(
    const Executor* executor,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const std::vector<JoinType>& join_types,
    const time_t now,
    const bool just_explain)
    : executor_(executor)
    , config_(executor->getConfig())
    , input_to_nest_level_(input_to_nest_level)
    , join_types_(join_types)
    , now_(now)
    , just_explain_(just_explain)
    , for_dag_builder_(false) {}

RelAlgTranslator::RelAlgTranslator(const Config& config,
                                   const time_t now,
                                   const bool just_explain)
    : executor_(nullptr)
    , config_(config)
    , now_(now)
    , just_explain_(just_explain)
    , for_dag_builder_(true) {}

hdk::ir::ExprPtr RelAlgTranslator::normalize(const hdk::ir::Expr* expr) const {
  NormalizerVisitor visitor(*this, input_to_nest_level_, join_types_, executor_);
  return visitor.visit(expr);
}

namespace {

hdk::ir::ExprPtr get_in_values_expr(hdk::ir::ExprPtr arg,
                                    const ResultSet& val_set,
                                    bool enable_watchdog) {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  if (val_set.rowCount() > 5000000 && enable_watchdog) {
    throw std::runtime_error(
        "Unable to handle 'expr IN (subquery)', subquery returned 5M+ rows.");
  }
  std::list<hdk::ir::ExprPtr> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::list<hdk::ir::ExprPtr>> expr_set(fetcher_count,
                                                    std::list<hdk::ir::ExprPtr>());
  std::vector<std::future<void>> fetcher_threads;
  auto type = arg->type();
  const auto entry_count = val_set.entryCount();
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, entry_count);
    fetcher_threads.push_back(std::async(
        std::launch::async,
        [&](std::list<hdk::ir::ExprPtr>& in_vals, const size_t start, const size_t end) {
          for (auto index = start; index < end; ++index) {
            auto row = val_set.getRowAt(index);
            if (row.empty()) {
              continue;
            }
            auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
            Datum d{0};
            bool is_null_const{false};
            std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, type);
            if (type->isExtDictionary()) {
              auto none_encoded_string = hdk::ir::makeExpr<hdk::ir::Constant>(
                  type->ctx().text(is_null_const), is_null_const, d);
              auto dict_encoded_string = std::make_shared<hdk::ir::UOper>(
                  type, false, kCAST, none_encoded_string);
              in_vals.push_back(dict_encoded_string);
            } else {
              in_vals.push_back(
                  hdk::ir::makeExpr<hdk::ir::Constant>(type, is_null_const, d));
            }
          }
        },
        std::ref(expr_set[i]),
        start_entry,
        end_entry));
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  for (auto& exprs : expr_set) {
    value_exprs.splice(value_exprs.end(), exprs);
  }
  return hdk::ir::makeExpr<hdk::ir::InValues>(arg, value_exprs);
}

}  // namespace

hdk::ir::ExprPtr RelAlgTranslator::translateInSubquery(
    hdk::ir::ExprPtr lhs,
    std::shared_ptr<const ExecutionResult> result) const {
  CHECK(result);
  auto type = lhs->type();
  auto& row_set = result->getRows();
  CHECK_EQ(size_t(1), row_set->colCount());
  const auto& rhs_type = row_set->colType(0);
  if (rhs_type->id() != type->id() || rhs_type->size() != type->size()) {
    throw std::runtime_error(
        "The two sides of the IN operator must have the same type; found " +
        type->toString() + " and " + rhs_type->toString());
  }
  row_set->moveToBegin();
  if (row_set->entryCount() > 10000) {
    hdk::ir::ExprPtr expr;
    if ((type->isInteger() || type->isExtDictionary()) &&
        !row_set->getQueryMemDesc().didOutputColumnar()) {
      expr = getInIntegerSetExpr(lhs, *row_set);
      // Handle the highly unlikely case when the InIntegerSet ended up being tiny.
      // Just let it fall through the usual InValues path at the end of this method,
      // its codegen knows to use inline comparisons for few values.
      if (expr && expr->as<hdk::ir::InIntegerSet>()->get_value_list().size() <= 100) {
        expr = nullptr;
      }
    } else {
      expr = get_in_values_expr(lhs, *row_set, config_.exec.watchdog.enable);
    }
    if (expr) {
      return expr;
    }
  }
  std::list<hdk::ir::ExprPtr> value_exprs;
  while (true) {
    auto row = row_set->getNextRow(true, false);
    if (row.empty()) {
      break;
    }
    if (config_.exec.watchdog.enable && value_exprs.size() >= 10000) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 10000+ rows.");
    }
    auto scalar_tv = boost::get<ScalarTargetValue>(&row[0]);
    Datum d{0};
    bool is_null_const{false};
    std::tie(d, is_null_const) = datum_from_scalar_tv(scalar_tv, type);
    if (type->isExtDictionary()) {
      auto none_encoded_string = hdk::ir::makeExpr<hdk::ir::Constant>(
          type->ctx().text(is_null_const), is_null_const, d);
      auto dict_encoded_string =
          std::make_shared<hdk::ir::UOper>(type, false, kCAST, none_encoded_string);
      value_exprs.push_back(dict_encoded_string);
    } else {
      value_exprs.push_back(hdk::ir::makeExpr<hdk::ir::Constant>(type, is_null_const, d));
    }
  }
  return hdk::ir::makeExpr<hdk::ir::InValues>(lhs, value_exprs);
}

namespace {

const size_t g_max_integer_set_size{1 << 25};

void fill_dictionary_encoded_in_vals(
    std::vector<int64_t>& in_vals,
    std::atomic<size_t>& total_in_vals_count,
    const ResultSet* values_rowset,
    const std::pair<int64_t, int64_t> values_rowset_slice,
    const StringDictionaryProxy* source_dict,
    const StringDictionaryProxy* dest_dict,
    const int64_t needle_null_val,
    bool enable_watchdog) {
  CHECK(in_vals.empty());
  bool dicts_are_equal = source_dict == dest_dict;
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (UNLIKELY(!row.valid)) {
      continue;
    }
    if (dicts_are_equal) {
      in_vals.push_back(row.value);
    } else {
      const int string_id =
          row.value == needle_null_val
              ? needle_null_val
              : dest_dict->getIdOfString(source_dict->getString(row.value));
      if (string_id != StringDictionary::INVALID_STR_ID) {
        in_vals.push_back(string_id);
      }
    }
    if (UNLIKELY(enable_watchdog && (in_vals.size() & 1023) == 0 &&
                 total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
      throw std::runtime_error(
          "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
    }
  }
}

void fill_integer_in_vals(std::vector<int64_t>& in_vals,
                          std::atomic<size_t>& total_in_vals_count,
                          const ResultSet* values_rowset,
                          const std::pair<int64_t, int64_t> values_rowset_slice,
                          bool enable_watchdog) {
  CHECK(in_vals.empty());
  for (auto index = values_rowset_slice.first; index < values_rowset_slice.second;
       ++index) {
    const auto row = values_rowset->getOneColRow(index);
    if (row.valid) {
      in_vals.push_back(row.value);
      if (UNLIKELY(enable_watchdog && (in_vals.size() & 1023) == 0 &&
                   total_in_vals_count.fetch_add(1024) >= g_max_integer_set_size)) {
        throw std::runtime_error(
            "Unable to handle 'expr IN (subquery)', subquery returned 30M+ rows.");
      }
    }
  }
}

}  // namespace

// The typical IN subquery involves either dictionary-encoded strings or integers.
// hdk::ir::InValues is a very heavy representation of the right hand side of such
// a query since we already know the right hand would be a list of hdk::ir::Constant
// shared pointers. We can avoid the big overhead of each hdk::ir::Constant and the
// refcounting associated with shared pointers by creating an abbreviated InIntegerSet
// representation of the IN expression which takes advantage of the this information.
hdk::ir::ExprPtr RelAlgTranslator::getInIntegerSetExpr(hdk::ir::ExprPtr arg,
                                                       const ResultSet& val_set) const {
  if (!result_set::can_use_parallel_algorithms(val_set)) {
    return nullptr;
  }
  std::vector<int64_t> value_exprs;
  const size_t fetcher_count = cpu_threads();
  std::vector<std::vector<int64_t>> expr_set(fetcher_count);
  std::vector<std::future<void>> fetcher_threads;
  auto arg_type = arg->type();
  const auto entry_count = val_set.entryCount();
  CHECK_EQ(size_t(1), val_set.colCount());
  auto col_type = val_set.colType(0);
  std::atomic<size_t> total_in_vals_count{0};
  for (size_t i = 0,
              start_entry = 0,
              stride = (entry_count + fetcher_count - 1) / fetcher_count;
       i < fetcher_count && start_entry < entry_count;
       ++i, start_entry += stride) {
    expr_set[i].reserve(entry_count / fetcher_count);
    const auto end_entry = std::min(start_entry + stride, entry_count);
    if (arg_type->isExtDictionary()) {
      auto dest_dict_id = arg_type->as<hdk::ir::ExtDictionaryType>()->dictId();
      CHECK(col_type->isExtDictionary());
      auto source_dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();
      const auto dd = executor_->getStringDictionaryProxy(
          dest_dict_id, val_set.getRowSetMemOwner(), true);
      const auto sd = executor_->getStringDictionaryProxy(
          source_dict_id, val_set.getRowSetMemOwner(), true);
      CHECK(sd);
      const auto needle_null_val = inline_int_null_value(arg_type);
      fetcher_threads.push_back(std::async(
          std::launch::async,
          [this, &val_set, &total_in_vals_count, sd, dd, needle_null_val](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            fill_dictionary_encoded_in_vals(in_vals,
                                            total_in_vals_count,
                                            &val_set,
                                            {start, end},
                                            sd,
                                            dd,
                                            needle_null_val,
                                            config_.exec.watchdog.enable);
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    } else {
      CHECK(arg_type->isInteger());
      fetcher_threads.push_back(std::async(
          std::launch::async,
          [this, &val_set, &total_in_vals_count](
              std::vector<int64_t>& in_vals, const size_t start, const size_t end) {
            fill_integer_in_vals(in_vals,
                                 total_in_vals_count,
                                 &val_set,
                                 {start, end},
                                 config_.exec.watchdog.enable);
          },
          std::ref(expr_set[i]),
          start_entry,
          end_entry));
    }
  }
  for (auto& child : fetcher_threads) {
    child.get();
  }

  val_set.moveToBegin();
  value_exprs.reserve(entry_count);
  for (auto& exprs : expr_set) {
    value_exprs.insert(value_exprs.end(), exprs.begin(), exprs.end());
  }
  return hdk::ir::makeExpr<hdk::ir::InIntegerSet>(
      arg, value_exprs, !arg_type->nullable() && !col_type->nullable());
}

namespace {

bool simple_predicate_has_simple_cast(const hdk::ir::ExprPtr cast_operand,
                                      const hdk::ir::ExprPtr const_operand) {
  if (hdk::ir::expr_is<hdk::ir::UOper>(cast_operand) &&
      hdk::ir::expr_is<hdk::ir::Constant>(const_operand)) {
    auto u_expr = cast_operand->as<hdk::ir::UOper>();
    if (u_expr->get_optype() != kCAST) {
      return false;
    }
    if (!(hdk::ir::expr_is<hdk::ir::ColumnVar>(u_expr->get_own_operand()) &&
          !hdk::ir::expr_is<hdk::ir::Var>(u_expr->get_own_operand()))) {
      return false;
    }
    auto type = u_expr->type();
    if (type->isDateTime() && u_expr->get_operand()->type()->isDateTime()) {
      // Allow casts between time types to pass through
      return true;
    } else if (type->isInteger() && u_expr->get_operand()->type()->isInteger()) {
      // Allow casts between integer types to pass through
      return true;
    }
  }
  return false;
}

/*
 * @brief normalize_simple_predicate only applies to boolean expressions.
 * it checks if it is an expression comparing a column
 * with a constant.  if so, it returns a normalized copy of the predicate with ColumnVar
 * always as the left operand with rte_idx set to the rte_idx of the ColumnVar.
 * it returns nullptr with rte_idx set to -1 otherwise.
 */
hdk::ir::ExprPtr normalize_simple_predicate(const hdk::ir::BinOper* bin_oper,
                                            int& rte_idx) {
  rte_idx = -1;
  if (!IS_COMPARISON(bin_oper->get_optype()) || bin_oper->get_qualifier() != kONE) {
    return nullptr;
  }

  auto left_operand = bin_oper->get_own_left_operand();
  auto right_operand = bin_oper->get_own_right_operand();
  if (hdk::ir::expr_is<hdk::ir::UOper>(left_operand)) {
    if (simple_predicate_has_simple_cast(left_operand, right_operand)) {
      auto uo = left_operand->as<hdk::ir::UOper>();
      auto cv = uo->get_own_operand()->as<hdk::ir::ColumnVar>();
      rte_idx = cv->get_rte_idx();
      return bin_oper->deep_copy();
    }
  } else if (hdk::ir::expr_is<hdk::ir::UOper>(right_operand)) {
    if (simple_predicate_has_simple_cast(right_operand, left_operand)) {
      auto uo = right_operand->as<hdk::ir::UOper>();
      auto cv = uo->get_own_operand()->as<hdk::ir::ColumnVar>();
      rte_idx = cv->get_rte_idx();
      return hdk::ir::makeExpr<hdk::ir::BinOper>(
          bin_oper->type(),
          bin_oper->containsAgg(),
          COMMUTE_COMPARISON(bin_oper->get_optype()),
          bin_oper->get_qualifier(),
          right_operand->deep_copy(),
          left_operand->deep_copy());
    }
  } else if (hdk::ir::expr_is<hdk::ir::ColumnVar>(left_operand) &&
             !hdk::ir::expr_is<hdk::ir::Var>(left_operand) &&
             hdk::ir::expr_is<hdk::ir::Constant>(right_operand)) {
    auto cv = left_operand->as<hdk::ir::ColumnVar>();
    rte_idx = cv->get_rte_idx();
    return bin_oper->deep_copy();
  } else if (hdk::ir::expr_is<hdk::ir::Constant>(left_operand) &&
             hdk::ir::expr_is<hdk::ir::ColumnVar>(right_operand) &&
             !hdk::ir::expr_is<hdk::ir::Var>(right_operand)) {
    auto cv = right_operand->as<hdk::ir::ColumnVar>();
    rte_idx = cv->get_rte_idx();
    return hdk::ir::makeExpr<hdk::ir::BinOper>(bin_oper->type(),
                                               bin_oper->containsAgg(),
                                               COMMUTE_COMPARISON(bin_oper->get_optype()),
                                               bin_oper->get_qualifier(),
                                               right_operand->deep_copy(),
                                               left_operand->deep_copy());
  }
  return nullptr;
}

}  // namespace

QualsConjunctiveForm qual_to_conjunctive_form(const hdk::ir::ExprPtr qual_expr) {
  CHECK(qual_expr);
  auto bin_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {{}, {rewritten_qual_expr ? rewritten_qual_expr : qual_expr}};
  }

  if (bin_oper->get_optype() == kAND) {
    const auto lhs_cf = qual_to_conjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_cf = qual_to_conjunctive_form(bin_oper->get_own_right_operand());
    auto simple_quals = lhs_cf.simple_quals;
    simple_quals.insert(
        simple_quals.end(), rhs_cf.simple_quals.begin(), rhs_cf.simple_quals.end());
    auto quals = lhs_cf.quals;
    quals.insert(quals.end(), rhs_cf.quals.begin(), rhs_cf.quals.end());
    return {simple_quals, quals};
  }
  int rte_idx{0};
  const auto simple_qual = normalize_simple_predicate(bin_oper.get(), rte_idx);
  return simple_qual ? QualsConjunctiveForm{{simple_qual}, {}}
                     : QualsConjunctiveForm{{}, {qual_expr}};
}

std::vector<hdk::ir::ExprPtr> qual_to_disjunctive_form(
    const hdk::ir::ExprPtr& qual_expr) {
  CHECK(qual_expr);
  const auto bin_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(qual_expr);
  if (!bin_oper) {
    const auto rewritten_qual_expr = rewrite_expr(qual_expr.get());
    return {rewritten_qual_expr ? rewritten_qual_expr : qual_expr};
  }
  if (bin_oper->get_optype() == kOR) {
    const auto lhs_df = qual_to_disjunctive_form(bin_oper->get_own_left_operand());
    const auto rhs_df = qual_to_disjunctive_form(bin_oper->get_own_right_operand());
    auto quals = lhs_df;
    quals.insert(quals.end(), rhs_df.begin(), rhs_df.end());
    return quals;
  }
  return {qual_expr};
}
