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

#pragma once

#include "ExprVisitor.h"

namespace hdk::ir {

class ExprRewriter : public ExprVisitor<ExprPtr> {
 protected:
  ExprPtr visitColumnVarTuple(const hdk::ir::ExpressionTuple* col_var_tuple) override {
    bool rewrite = false;
    ExprPtrVector new_tuple;
    new_tuple.reserve(col_var_tuple->tuple().size());
    for (const auto& component : col_var_tuple->tuple()) {
      new_tuple.emplace_back(visit(component.get()));
      rewrite = rewrite || new_tuple.back().get() != component.get();
    }
    if (rewrite) {
      return makeExpr<ExpressionTuple>(std::move(new_tuple));
    }
    return defaultResult(col_var_tuple);
  }

  ExprPtr visitUOper(const hdk::ir::UOper* uoper) override {
    auto new_op = visit(uoper->operand());
    if (new_op.get() != uoper->operand()) {
      return hdk::ir::makeExpr<hdk::ir::UOper>(
          uoper->type(), uoper->containsAgg(), uoper->opType(), new_op);
    }
    return defaultResult(uoper);
  }

  ExprPtr visitBinOper(const hdk::ir::BinOper* bin_oper) override {
    auto new_lhs = visit(bin_oper->leftOperand());
    auto new_rhs = visit(bin_oper->rightOperand());
    if (new_lhs.get() != bin_oper->leftOperand() ||
        new_rhs.get() != bin_oper->rightOperand()) {
      return hdk::ir::makeExpr<hdk::ir::BinOper>(bin_oper->type(),
                                                 bin_oper->containsAgg(),
                                                 bin_oper->opType(),
                                                 bin_oper->qualifier(),
                                                 new_lhs,
                                                 new_rhs);
    }
    return defaultResult(bin_oper);
  }

  ExprPtr visitInValues(const hdk::ir::InValues* in_values) override {
    bool rewrite = false;
    std::list<ExprPtr> new_list;
    for (const auto& in_value : in_values->valueList()) {
      new_list.push_back(visit(in_value.get()));
      rewrite = rewrite || new_list.back().get() != in_value.get();
    }
    auto new_arg = visit(in_values->arg());
    rewrite = rewrite || new_arg.get() != in_values->arg();
    if (rewrite) {
      return hdk::ir::makeExpr<hdk::ir::InValues>(new_arg, new_list);
    }
    return defaultResult(in_values);
  }

  ExprPtr visitInIntegerSet(const hdk::ir::InIntegerSet* in_integer_set) override {
    auto new_arg = visit(in_integer_set->arg());
    if (new_arg.get() != in_integer_set->arg()) {
      return hdk::ir::makeExpr<hdk::ir::InIntegerSet>(
          new_arg, in_integer_set->valueList(), !in_integer_set->type()->nullable());
    }
    return defaultResult(in_integer_set);
  }

  ExprPtr visitInSubquery(const hdk::ir::InSubquery* in_subquery) override {
    auto new_arg = visit(in_subquery->arg());
    if (new_arg.get() != in_subquery->arg()) {
      return hdk::ir::makeExpr<hdk::ir::InSubquery>(
          in_subquery->type(), new_arg, in_subquery->nodeShared());
    }
    return defaultResult(in_subquery);
  }

  ExprPtr visitCharLength(const hdk::ir::CharLengthExpr* char_length) override {
    auto new_arg = visit(char_length->arg());
    if (new_arg.get() != char_length->arg()) {
      return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(new_arg,
                                                        char_length->calcEncodedLength());
    }
    return defaultResult(char_length);
  }

  ExprPtr visitKeyForString(const hdk::ir::KeyForStringExpr* expr) override {
    auto new_arg = visit(expr->arg());
    if (new_arg.get() != expr->arg()) {
      return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(new_arg);
    }
    return defaultResult(expr);
  }

  ExprPtr visitSampleRatio(const hdk::ir::SampleRatioExpr* expr) override {
    auto new_arg = visit(expr->arg());
    if (new_arg.get() != expr->arg()) {
      return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(new_arg);
    }
    return defaultResult(expr);
  }

  ExprPtr visitLower(const hdk::ir::LowerExpr* expr) override {
    auto new_arg = visit(expr->arg());
    if (new_arg.get() != expr->arg()) {
      return hdk::ir::makeExpr<hdk::ir::LowerExpr>(new_arg);
    }
    return defaultResult(expr);
  }

  ExprPtr visitCardinality(const hdk::ir::CardinalityExpr* cardinality) override {
    auto new_arg = visit(cardinality->arg());
    if (new_arg.get() != cardinality->arg()) {
      return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(new_arg);
    }
    return defaultResult(cardinality);
  }

  ExprPtr visitLikeExpr(const hdk::ir::LikeExpr* like) override {
    auto new_arg = visit(like->arg());
    auto new_like = visit(like->likeExpr());
    ExprPtr new_escape = like->escapeExpr() ? visit(like->escapeExpr()) : nullptr;
    if (new_arg.get() != like->arg() || new_like.get() != like->likeExpr() ||
        new_escape.get() != like->escapeExpr()) {
      return hdk::ir::makeExpr<hdk::ir::LikeExpr>(
          new_arg, new_like, new_escape, like->isIlike(), like->isSimple());
    }
    return defaultResult(like);
  }

  ExprPtr visitRegexpExpr(const hdk::ir::RegexpExpr* regexp) override {
    auto new_arg = visit(regexp->arg());
    auto new_pattern = visit(regexp->patternExpr());
    ExprPtr new_escape = regexp->escapeExpr() ? visit(regexp->escapeExpr()) : nullptr;
    if (new_arg.get() != regexp->arg() || new_pattern.get() != regexp->patternExpr() ||
        new_escape.get() != regexp->escapeExpr()) {
      return hdk::ir::makeExpr<hdk::ir::RegexpExpr>(new_arg, new_pattern, new_escape);
    }
    return defaultResult(regexp);
  }

  ExprPtr visitWidthBucket(const hdk::ir::WidthBucketExpr* width_bucket_expr) override {
    auto new_target = visit(width_bucket_expr->targetValue());
    auto new_lower = visit(width_bucket_expr->lowerBound());
    auto new_upper = visit(width_bucket_expr->upperBound());
    auto new_partitions = visit(width_bucket_expr->partitionCount());
    if (new_target.get() != width_bucket_expr->targetValue() ||
        new_lower.get() != width_bucket_expr->lowerBound() ||
        new_upper.get() != width_bucket_expr->upperBound() ||
        new_partitions.get() != width_bucket_expr->partitionCount()) {
      return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
          new_target, new_lower, new_upper, new_partitions);
    }
    return defaultResult(width_bucket_expr);
  }

  ExprPtr visitCaseExpr(const hdk::ir::CaseExpr* case_expr) override {
    bool rewrite = false;
    std::list<std::pair<ExprPtr, ExprPtr>> new_list;
    for (auto p : case_expr->exprPairs()) {
      new_list.emplace_back(visit(p.first.get()), visit(p.second.get()));
      rewrite = rewrite || new_list.back().first.get() != p.first.get() ||
                new_list.back().second.get() != p.second.get();
    }
    ExprPtr new_else = case_expr->elseExpr() ? visit(case_expr->elseExpr()) : nullptr;
    rewrite = rewrite || new_else.get() != case_expr->elseExpr();
    if (rewrite) {
      return hdk::ir::makeExpr<hdk::ir::CaseExpr>(
          case_expr->type(), case_expr->containsAgg(), new_list, new_else);
    }
    return defaultResult(case_expr);
  }

  ExprPtr visitDateTruncExpr(const hdk::ir::DateTruncExpr* datetrunc) override {
    auto new_from = visit(datetrunc->from());
    if (new_from.get() != datetrunc->from()) {
      return hdk::ir::makeExpr<hdk::ir::DateTruncExpr>(
          datetrunc->type(), datetrunc->containsAgg(), datetrunc->field(), new_from);
    }
    return defaultResult(datetrunc);
  }

  ExprPtr visitExtractExpr(const hdk::ir::ExtractExpr* extract) override {
    auto new_from = visit(extract->from());
    if (new_from.get() != extract->from()) {
      return hdk::ir::makeExpr<hdk::ir::ExtractExpr>(
          extract->type(), extract->containsAgg(), extract->field(), new_from);
    }
    return defaultResult(extract);
  }

  ExprPtr visitArrayOper(const hdk::ir::ArrayExpr* array_expr) override {
    bool rewrite = false;
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < array_expr->elementCount(); ++i) {
      args_copy.push_back(visit(array_expr->element(i)));
      rewrite = rewrite || args_copy.back().get() != array_expr->element(i);
    }
    auto type = array_expr->type();
    if (rewrite) {
      return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(
          type, args_copy, array_expr->isNull(), array_expr->isLocalAlloc());
    }
    return defaultResult(array_expr);
  }

  ExprPtr visitWindowFunction(const hdk::ir::WindowFunction* window_func) override {
    bool rewrite = false;
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (const auto& arg : window_func->args()) {
      args_copy.push_back(visit(arg.get()));
      rewrite = rewrite || args_copy.back().get() != arg.get();
    }
    std::vector<hdk::ir::ExprPtr> partition_keys_copy;
    for (const auto& partition_key : window_func->partitionKeys()) {
      partition_keys_copy.push_back(visit(partition_key.get()));
      rewrite = rewrite || partition_keys_copy.back().get() != partition_key.get();
    }
    std::vector<hdk::ir::ExprPtr> order_keys_copy;
    for (const auto& order_key : window_func->orderKeys()) {
      order_keys_copy.push_back(visit(order_key.get()));
      rewrite = rewrite || order_keys_copy.back().get() != order_key.get();
    }
    if (rewrite) {
      const auto& type = window_func->type();
      return hdk::ir::makeExpr<hdk::ir::WindowFunction>(type,
                                                        window_func->kind(),
                                                        args_copy,
                                                        partition_keys_copy,
                                                        order_keys_copy,
                                                        window_func->collation());
    }
    return defaultResult(window_func);
  }

  ExprPtr visitFunctionOper(const hdk::ir::FunctionOper* func_oper) override {
    bool rewrite = false;
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->arity(); ++i) {
      args_copy.push_back(visit(func_oper->arg(i)));
      rewrite = rewrite || args_copy.back().get() != func_oper->arg(i);
    }
    if (rewrite) {
      const auto& type = func_oper->type();
      return hdk::ir::makeExpr<hdk::ir::FunctionOper>(type, func_oper->name(), args_copy);
    }
    return defaultResult(func_oper);
  }

  ExprPtr visitDateDiffExpr(const hdk::ir::DateDiffExpr* datediff) override {
    auto new_start = visit(datediff->start());
    auto new_end = visit(datediff->end());
    if (new_start.get() != datediff->start() || new_end.get() != datediff->end()) {
      return hdk::ir::makeExpr<hdk::ir::DateDiffExpr>(
          datediff->type(), datediff->field(), new_start, new_end);
    }
    return defaultResult(datediff);
  }

  ExprPtr visitDateAddExpr(const hdk::ir::DateAddExpr* dateadd) override {
    auto new_number = visit(dateadd->number());
    auto new_datetime = visit(dateadd->datetime());
    if (new_number.get() != dateadd->number() ||
        new_datetime.get() != dateadd->datetime()) {
      return hdk::ir::makeExpr<hdk::ir::DateAddExpr>(
          dateadd->type(), dateadd->field(), new_number, new_datetime);
    }
    return defaultResult(dateadd);
  }

  ExprPtr visitFunctionOperWithCustomTypeHandling(
      const hdk::ir::FunctionOperWithCustomTypeHandling* func_oper) override {
    bool rewrite = false;
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->arity(); ++i) {
      args_copy.push_back(visit(func_oper->arg(i)));
      rewrite = rewrite || args_copy.back().get() != func_oper->arg(i);
    }
    if (rewrite) {
      const auto& type = func_oper->type();
      return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
          type, func_oper->name(), args_copy);
    }
    return defaultResult(func_oper);
  }

  ExprPtr visitLikelihood(const hdk::ir::LikelihoodExpr* likelihood) override {
    auto new_arg = visit(likelihood->arg());
    if (new_arg.get() != likelihood->arg()) {
      return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(new_arg,
                                                        likelihood->likelihood());
    }
    return defaultResult(likelihood);
  }

  ExprPtr visitAggExpr(const hdk::ir::AggExpr* agg) override {
    ExprPtr new_arg = agg->arg() ? visit(agg->arg()) : nullptr;
    if (new_arg.get() != agg->arg()) {
      return hdk::ir::makeExpr<hdk::ir::AggExpr>(
          agg->type(), agg->aggType(), new_arg, agg->isDistinct(), agg->arg1());
    }
    return defaultResult(agg);
  }

  ExprPtr defaultResult(const hdk::ir::Expr* expr) const override {
    return expr->shared_from_this();
  }
};

}  // namespace hdk::ir
