/*
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Expr.h"

namespace hdk::ir {

template <class T>
class ExprVisitor {
 public:
  virtual ~ExprVisitor() {}

  virtual T visit(const hdk::ir::Expr* expr) {
    CHECK(expr);
    if (auto var = expr->as<Var>()) {
      return visitVar(var);
    }
    if (auto column_var = expr->as<ColumnVar>()) {
      return visitColumnVar(column_var);
    }
    if (auto column_ref = expr->as<ColumnRef>()) {
      return visitColumnRef(column_ref);
    }
    if (auto column_var_tuple = expr->as<ExpressionTuple>()) {
      return visitColumnVarTuple(column_var_tuple);
    }
    if (auto constant = expr->as<Constant>()) {
      return visitConstant(constant);
    }
    if (auto uoper = expr->as<UOper>()) {
      return visitUOper(uoper);
    }
    if (auto bin_oper = expr->as<BinOper>()) {
      return visitBinOper(bin_oper);
    }
    if (auto scalar_subquery = expr->as<ScalarSubquery>()) {
      return visitScalarSubquery(scalar_subquery);
    }
    if (auto in_values = expr->as<InValues>()) {
      return visitInValues(in_values);
    }
    if (auto in_integer_set = expr->as<InIntegerSet>()) {
      return visitInIntegerSet(in_integer_set);
    }
    if (auto in_subquery = expr->as<InSubquery>()) {
      return visitInSubquery(in_subquery);
    }
    if (auto char_length = expr->as<CharLengthExpr>()) {
      return visitCharLength(char_length);
    }
    if (auto key_for_string = expr->as<KeyForStringExpr>()) {
      return visitKeyForString(key_for_string);
    }
    if (auto sample_ratio = expr->as<SampleRatioExpr>()) {
      return visitSampleRatio(sample_ratio);
    }
    if (auto width_bucket = expr->as<WidthBucketExpr>()) {
      return visitWidthBucket(width_bucket);
    }
    if (auto lower = expr->as<LowerExpr>()) {
      return visitLower(lower);
    }
    if (auto cardinality = expr->as<CardinalityExpr>()) {
      return visitCardinality(cardinality);
    }
    if (auto width_bucket_expr = expr->as<WidthBucketExpr>()) {
      return visitWidthBucket(width_bucket_expr);
    }
    if (auto like_expr = expr->as<LikeExpr>()) {
      return visitLikeExpr(like_expr);
    }
    if (auto regexp_expr = expr->as<RegexpExpr>()) {
      return visitRegexpExpr(regexp_expr);
    }
    if (auto case_ = expr->as<CaseExpr>()) {
      return visitCaseExpr(case_);
    }
    if (auto datetrunc = expr->as<DateTruncExpr>()) {
      return visitDateTruncExpr(datetrunc);
    }
    if (auto extract = expr->as<ExtractExpr>()) {
      return visitExtractExpr(extract);
    }
    if (auto window_func = expr->as<WindowFunction>()) {
      return visitWindowFunction(window_func);
    }
    if (auto func_with_custom_type_handling =
            expr->as<FunctionOperWithCustomTypeHandling>()) {
      return visitFunctionOperWithCustomTypeHandling(func_with_custom_type_handling);
    }
    if (auto func = expr->as<FunctionOper>()) {
      return visitFunctionOper(func);
    }
    if (auto array = expr->as<ArrayExpr>()) {
      return visitArrayOper(array);
    }
    if (auto datediff = expr->as<DateDiffExpr>()) {
      return visitDateDiffExpr(datediff);
    }
    if (auto dateadd = expr->as<DateAddExpr>()) {
      return visitDateAddExpr(dateadd);
    }
    if (auto likelihood = expr->as<LikelihoodExpr>()) {
      return visitLikelihood(likelihood);
    }
    if (auto offset_in_fragment = expr->as<OffsetInFragment>()) {
      return visitOffsetInFragment(offset_in_fragment);
    }
    if (auto agg = expr->as<AggExpr>()) {
      return visitAggExpr(agg);
    }
    if (auto shuffle = expr->as<ShuffleStore>()) {
      return visitShuffleStore(shuffle);
    }
    CHECK(false) << "Unhandled expr: " << expr->toString();
    return defaultResult(expr);
  }

 protected:
  virtual T visitVar(const hdk::ir::Var* var) { return defaultResult(var); }

  virtual T visitColumnVar(const hdk::ir::ColumnVar* col_var) {
    return defaultResult(col_var);
  }

  virtual T visitColumnRef(const hdk::ir::ColumnRef* col_ref) {
    return defaultResult(col_ref);
  }

  virtual T visitColumnVarTuple(const hdk::ir::ExpressionTuple* tuple) {
    for (const auto& component : tuple->tuple()) {
      visit(component.get());
    }
    return defaultResult(tuple);
  }

  virtual T visitConstant(const hdk::ir::Constant* cst) { return defaultResult(cst); }

  virtual T visitUOper(const hdk::ir::UOper* uoper) {
    visit(uoper->operand());
    return defaultResult(uoper);
  }

  virtual T visitBinOper(const hdk::ir::BinOper* bin_oper) {
    visit(bin_oper->leftOperand());
    visit(bin_oper->rightOperand());
    return defaultResult(bin_oper);
  }

  virtual T visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) {
    return defaultResult(subquery);
  }

  virtual T visitInValues(const hdk::ir::InValues* in_values) {
    visit(in_values->arg());
    const auto& value_list = in_values->valueList();
    for (const auto& in_value : value_list) {
      visit(in_value.get());
    }
    return defaultResult(in_values);
  }

  virtual T visitInIntegerSet(const hdk::ir::InIntegerSet* in_integer_set) {
    return visit(in_integer_set->arg());
  }

  virtual T visitInSubquery(const hdk::ir::InSubquery* in_subquery) {
    return visit(in_subquery->arg());
  }

  virtual T visitCharLength(const hdk::ir::CharLengthExpr* char_length) {
    return visit(char_length->arg());
  }

  virtual T visitKeyForString(const hdk::ir::KeyForStringExpr* key_for_string) {
    return visit(key_for_string->arg());
  }

  virtual T visitSampleRatio(const hdk::ir::SampleRatioExpr* sample_ratio) {
    return visit(sample_ratio->arg());
  }

  virtual T visitLower(const hdk::ir::LowerExpr* lower_expr) {
    return visit(lower_expr->arg());
  }

  virtual T visitCardinality(const hdk::ir::CardinalityExpr* cardinality) {
    return visit(cardinality->arg());
  }

  virtual T visitLikeExpr(const hdk::ir::LikeExpr* like) {
    visit(like->arg());
    visit(like->likeExpr());
    if (like->escapeExpr()) {
      visit(like->escapeExpr());
    }
    return defaultResult(like);
  }

  virtual T visitRegexpExpr(const hdk::ir::RegexpExpr* regexp) {
    visit(regexp->arg());
    visit(regexp->patternExpr());
    if (regexp->escapeExpr()) {
      visit(regexp->escapeExpr());
    }
    return defaultResult(regexp);
  }

  virtual T visitWidthBucket(const hdk::ir::WidthBucketExpr* width_bucket_expr) {
    visit(width_bucket_expr->targetValue());
    visit(width_bucket_expr->lowerBound());
    visit(width_bucket_expr->upperBound());
    visit(width_bucket_expr->partitionCount());
    return defaultResult(width_bucket_expr);
  }

  virtual T visitCaseExpr(const hdk::ir::CaseExpr* case_expr) {
    const auto& expr_pair_list = case_expr->exprPairs();
    for (const auto& expr_pair : expr_pair_list) {
      visit(expr_pair.first.get());
      visit(expr_pair.second.get());
    }
    visit(case_expr->elseExpr());
    return defaultResult(case_expr);
  }

  virtual T visitDateTruncExpr(const hdk::ir::DateTruncExpr* datetrunc) {
    return visit(datetrunc->from());
  }

  virtual T visitExtractExpr(const hdk::ir::ExtractExpr* extract) {
    return visit(extract->from());
  }

  virtual T visitFunctionOperWithCustomTypeHandling(
      const hdk::ir::FunctionOperWithCustomTypeHandling* func_oper) {
    return visitFunctionOper(func_oper);
  }

  virtual T visitArrayOper(hdk::ir::ArrayExpr const* array_expr) {
    for (size_t i = 0; i < array_expr->elementCount(); ++i) {
      visit(array_expr->element(i));
    }
    return defaultResult(array_expr);
  }

  virtual T visitFunctionOper(const hdk::ir::FunctionOper* func_oper) {
    for (size_t i = 0; i < func_oper->arity(); ++i) {
      visit(func_oper->arg(i));
    }
    return defaultResult(func_oper);
  }

  virtual T visitWindowFunction(const hdk::ir::WindowFunction* window_func) {
    for (const auto& arg : window_func->args()) {
      visit(arg.get());
    }
    for (const auto& partition_key : window_func->partitionKeys()) {
      visit(partition_key.get());
    }
    for (const auto& order_key : window_func->orderKeys()) {
      visit(order_key.get());
    }
    return defaultResult(window_func);
  }

  virtual T visitDateDiffExpr(const hdk::ir::DateDiffExpr* datediff) {
    visit(datediff->start());
    visit(datediff->end());
    return defaultResult(datediff);
  }

  virtual T visitDateAddExpr(const hdk::ir::DateAddExpr* dateadd) {
    visit(dateadd->number());
    visit(dateadd->datetime());
    return defaultResult(dateadd);
  }

  virtual T visitLikelihood(const hdk::ir::LikelihoodExpr* likelihood) {
    return visit(likelihood->arg());
  }

  virtual T visitOffsetInFragment(const hdk::ir::OffsetInFragment* offs_in_fragment) {
    return defaultResult(offs_in_fragment);
  }

  virtual T visitAggExpr(const hdk::ir::AggExpr* agg) {
    if (agg->arg()) {
      visit(agg->arg());
    }
    if (agg->arg1()) {
      visit(agg->arg1());
    }
    return defaultResult(agg);
  }

  virtual T visitShuffleStore(const hdk::ir::ShuffleStore* shuffle) {
    visit(shuffle->val());
    visit(shuffle->outBuffers());
    visit(shuffle->offsets());
    return defaultResult(shuffle);
  }

  virtual T defaultResult(const hdk::ir::Expr*) const {
    if constexpr (!std::is_same<T, void>::value) {
      return T{};
    }
  }
};

}  // namespace hdk::ir
