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

#include "QueryRewrite.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "Logger/Logger.h"
#include "Shared/sqltypes.h"

RelAlgExecutionUnit QueryRewriter::rewrite(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto rewritten_exe_unit = rewriteConstrainedByIn(ra_exe_unit_in);
  auto rewritten_exe_unit_for_agg_on_gby_col =
      rewriteAggregateOnGroupByColumn(rewritten_exe_unit);
  return rewritten_exe_unit_for_agg_on_gby_col;
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByIn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  if (ra_exe_unit_in.groupby_exprs.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.groupby_exprs.size() == 1 && !ra_exe_unit_in.groupby_exprs.front()) {
    return ra_exe_unit_in;
  }
  if (!ra_exe_unit_in.simple_quals.empty()) {
    return ra_exe_unit_in;
  }
  if (ra_exe_unit_in.quals.size() != 1) {
    return ra_exe_unit_in;
  }
  auto in_vals =
      std::dynamic_pointer_cast<const hdk::ir::InValues>(ra_exe_unit_in.quals.front());
  if (!in_vals) {
    in_vals = std::dynamic_pointer_cast<const hdk::ir::InValues>(
        rewrite_expr(ra_exe_unit_in.quals.front().get()));
  }
  if (!in_vals || in_vals->valueList().empty()) {
    return ra_exe_unit_in;
  }
  for (const auto& in_val : in_vals->valueList()) {
    if (!in_val->is<hdk::ir::Constant>()) {
      break;
    }
  }
  if (dynamic_cast<const hdk::ir::CaseExpr*>(in_vals->arg())) {
    return ra_exe_unit_in;
  }
  auto case_expr = generateCaseForDomainValues(in_vals.get());
  return rewriteConstrainedByInImpl(ra_exe_unit_in, case_expr, in_vals.get());
}

RelAlgExecutionUnit QueryRewriter::rewriteConstrainedByInImpl(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::shared_ptr<const hdk::ir::CaseExpr> case_expr,
    const hdk::ir::InValues* in_vals) const {
  std::list<hdk::ir::ExprPtr> new_groupby_list;
  std::vector<const hdk::ir::Expr*> new_target_exprs;
  bool rewrite{false};
  size_t groupby_idx{0};
  auto it = ra_exe_unit_in.groupby_exprs.begin();
  for (const auto& group_expr : ra_exe_unit_in.groupby_exprs) {
    CHECK(group_expr);
    ++groupby_idx;
    if (*group_expr == *in_vals->arg()) {
      const auto expr_range = getExpressionRange(it->get(), query_infos_, executor_);
      if (expr_range.getType() != ExpressionRangeType::Integer) {
        ++it;
        continue;
      }
      const size_t range_sz = expr_range.getIntMax() - expr_range.getIntMin() + 1;
      if (range_sz <= in_vals->valueList().size() *
                          executor_->getConfig().opts.constrained_by_in_threshold) {
        ++it;
        continue;
      }
      new_groupby_list.push_back(case_expr);
      for (size_t i = 0; i < ra_exe_unit_in.target_exprs.size(); ++i) {
        const auto target = ra_exe_unit_in.target_exprs[i];
        if (*target == *in_vals->arg()) {
          auto var_case_expr = hdk::ir::makeExpr<hdk::ir::Var>(
              case_expr->type(), hdk::ir::Var::kGROUPBY, groupby_idx);
          target_exprs_owned_.push_back(var_case_expr);
          new_target_exprs.push_back(var_case_expr.get());
        } else {
          new_target_exprs.push_back(target);
        }
      }
      rewrite = true;
    } else {
      new_groupby_list.push_back(group_expr);
    }
    ++it;
  }
  if (!rewrite) {
    return ra_exe_unit_in;
  }
  return {ra_exe_unit_in.input_descs,
          ra_exe_unit_in.input_col_descs,
          ra_exe_unit_in.simple_quals,
          ra_exe_unit_in.quals,
          ra_exe_unit_in.join_quals,
          new_groupby_list,
          new_target_exprs,
          nullptr,
          ra_exe_unit_in.sort_info,
          ra_exe_unit_in.scan_limit,
          ra_exe_unit_in.query_hint,
          ra_exe_unit_in.query_plan_dag,
          ra_exe_unit_in.hash_table_build_plan_dag,
          ra_exe_unit_in.table_id_to_node_map};
}

std::shared_ptr<const hdk::ir::CaseExpr> QueryRewriter::generateCaseForDomainValues(
    const hdk::ir::InValues* in_vals) {
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> case_expr_list;
  auto in_val_arg = in_vals->argShared();
  for (const auto& in_val : in_vals->valueList()) {
    auto case_cond = hdk::ir::makeExpr<hdk::ir::BinOper>(
        in_vals->ctx().boolean(false), false, kEQ, kONE, in_val_arg, in_val);
    auto in_val_copy = in_val;
    auto type = in_val_copy->type();
    if (type->isExtDictionary()) {
      type = type->ctx().extDict(
          type->as<hdk::ir::ExtDictionaryType>()->elemType(), 0, type->size());
      in_val_copy = in_val_copy->withType(type);
    }
    case_expr_list.emplace_back(case_cond, in_val_copy);
  }
  // TODO(alex): refine the expression range for case with empty else expression;
  //             for now, add a dummy else which should never be taken
  auto else_expr = case_expr_list.front().second;
  return hdk::ir::makeExpr<hdk::ir::CaseExpr>(
      case_expr_list.front().second->type(), false, case_expr_list, else_expr);
}

std::shared_ptr<const hdk::ir::CaseExpr>
QueryRewriter::generateCaseExprForCountDistinctOnGroupByCol(
    hdk::ir::ExprPtr expr,
    const hdk::ir::Type* type) const {
  std::list<std::pair<hdk::ir::ExprPtr, hdk::ir::ExprPtr>> case_expr_list;
  auto& ctx = expr->ctx();
  auto is_null = std::make_shared<hdk::ir::UOper>(ctx.boolean(false), kISNULL, expr);
  auto is_not_null = std::make_shared<hdk::ir::UOper>(ctx.boolean(false), kNOT, is_null);
  const auto then_constant = hdk::ir::Constant::make(type, 1);
  case_expr_list.emplace_back(is_not_null, then_constant);
  const auto else_constant = hdk::ir::Constant::make(type, 0);
  auto case_expr =
      hdk::ir::makeExpr<hdk::ir::CaseExpr>(type, false, case_expr_list, else_constant);
  return case_expr;
}

std::pair<bool, std::set<size_t>> QueryRewriter::is_all_groupby_exprs_are_col_var(
    const std::list<hdk::ir::ExprPtr>& groupby_exprs) const {
  std::set<size_t> gby_col_exprs_hash;
  for (auto& gby_expr : groupby_exprs) {
    if (gby_expr && gby_expr->is<hdk::ir::ColumnVar>()) {
      gby_col_exprs_hash.insert(boost::hash_value(gby_expr->toString()));
    } else {
      return {false, {}};
    }
  }
  return {true, gby_col_exprs_hash};
}

RelAlgExecutionUnit QueryRewriter::rewriteAggregateOnGroupByColumn(
    const RelAlgExecutionUnit& ra_exe_unit_in) const {
  auto check_precond = is_all_groupby_exprs_are_col_var(ra_exe_unit_in.groupby_exprs);
  auto is_expr_on_gby_col = [&check_precond](const hdk::ir::AggExpr* agg_expr) {
    CHECK(agg_expr);
    if (agg_expr->arg()) {
      // some expr does not have its own arg, i.e., count(*)
      auto agg_expr_hash = boost::hash_value(agg_expr->arg()->toString());
      // a valid expr should have hashed value > 0
      CHECK_GT(agg_expr_hash, 0u);
      if (check_precond.second.count(agg_expr_hash)) {
        return true;
      }
    }
    return false;
  };
  if (!check_precond.first) {
    // return the input ra_exe_unit if we have gby expr which is not col_var
    // i.e., group by x+1, y instead of group by x, y
    // todo (yoonmin) : can we relax this with a simple analysis of groupby / agg exprs?
    return ra_exe_unit_in;
  }

  std::vector<const hdk::ir::Expr*> new_target_exprs;
  for (auto expr : ra_exe_unit_in.target_exprs) {
    bool rewritten = false;
    if (auto agg_expr = expr->as<hdk::ir::AggExpr>()) {
      if (is_expr_on_gby_col(agg_expr)) {
        auto target_expr = agg_expr->arg();
        // we have some issues when this rewriting is applied to float_type groupby column
        // in subquery, i.e., SELECT MIN(v1) FROM (SELECT v1, AGG(v1) FROM T GROUP BY v1);
        if (target_expr && !target_expr->type()->isFp32()) {
          switch (agg_expr->aggType()) {
            case SQLAgg::kCOUNT:
            case SQLAgg::kAPPROX_COUNT_DISTINCT: {
              if (agg_expr->aggType() == SQLAgg::kCOUNT && !agg_expr->isDistinct()) {
                break;
              }
              auto case_expr = generateCaseExprForCountDistinctOnGroupByCol(
                  agg_expr->argShared(), agg_expr->type());
              new_target_exprs.push_back(case_expr.get());
              target_exprs_owned_.emplace_back(case_expr);
              rewritten = true;
              break;
            }
            case SQLAgg::kAPPROX_QUANTILE:
            case SQLAgg::kAVG:
            case SQLAgg::kSAMPLE:
            case SQLAgg::kMAX:
            case SQLAgg::kMIN: {
              // we just replace the agg_expr into a plain expr
              // i.e, avg(x1) --> x1
              auto agg_expr_type = agg_expr->type();
              auto target_expr = agg_expr->argShared();
              if (!agg_expr_type->equal(target_expr->type())) {
                target_expr = target_expr->cast(agg_expr_type);
              }
              new_target_exprs.push_back(target_expr.get());
              target_exprs_owned_.emplace_back(target_expr);
              rewritten = true;
              break;
            }
            default:
              break;
          }
        }
      }
    }
    if (!rewritten) {
      new_target_exprs.push_back(expr);
    }
  }

  RelAlgExecutionUnit rewritten_exe_unit{ra_exe_unit_in.input_descs,
                                         ra_exe_unit_in.input_col_descs,
                                         ra_exe_unit_in.simple_quals,
                                         ra_exe_unit_in.quals,
                                         ra_exe_unit_in.join_quals,
                                         ra_exe_unit_in.groupby_exprs,
                                         new_target_exprs,
                                         ra_exe_unit_in.estimator,
                                         ra_exe_unit_in.sort_info,
                                         ra_exe_unit_in.scan_limit,
                                         ra_exe_unit_in.query_hint,
                                         ra_exe_unit_in.query_plan_dag,
                                         ra_exe_unit_in.hash_table_build_plan_dag,
                                         ra_exe_unit_in.table_id_to_node_map,
                                         ra_exe_unit_in.use_bump_allocator,
                                         ra_exe_unit_in.union_all};
  return rewritten_exe_unit;
}
