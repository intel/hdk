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

#include "ScalarExprVisitor.h"

class DeepCopyVisitor : public ScalarExprVisitor<hdk::ir::ExprPtr> {
 protected:
  using RetType = hdk::ir::ExprPtr;
  RetType visitColumnVar(const hdk::ir::ColumnVar* col_var) const override {
    return col_var->deep_copy();
  }

  RetType visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    return col_ref->deep_copy();
  }

  RetType visitGroupColumnRef(const hdk::ir::GroupColumnRef* col_ref) const override {
    return col_ref->deep_copy();
  }

  RetType visitColumnVarTuple(
      const hdk::ir::ExpressionTuple* col_var_tuple) const override {
    return col_var_tuple->deep_copy();
  }

  RetType visitVar(const hdk::ir::Var* var) const override { return var->deep_copy(); }

  RetType visitConstant(const hdk::ir::Constant* constant) const override {
    return constant->deep_copy();
  }

  RetType visitUOper(const hdk::ir::UOper* uoper) const override {
    return hdk::ir::makeExpr<hdk::ir::UOper>(
        uoper->type(), uoper->containsAgg(), uoper->opType(), visit(uoper->operand()));
  }

  RetType visitBinOper(const hdk::ir::BinOper* bin_oper) const override {
    return hdk::ir::makeExpr<hdk::ir::BinOper>(bin_oper->type(),
                                               bin_oper->containsAgg(),
                                               bin_oper->opType(),
                                               bin_oper->qualifier(),
                                               visit(bin_oper->leftOperand()),
                                               visit(bin_oper->rightOperand()));
  }

  RetType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    return subquery->deep_copy();
  }

  RetType visitInValues(const hdk::ir::InValues* in_values) const override {
    const auto& value_list = in_values->valueList();
    std::list<RetType> new_list;
    for (const auto& in_value : value_list) {
      new_list.push_back(visit(in_value.get()));
    }
    return hdk::ir::makeExpr<hdk::ir::InValues>(visit(in_values->arg()), new_list);
  }

  RetType visitInIntegerSet(const hdk::ir::InIntegerSet* in_integer_set) const override {
    return hdk::ir::makeExpr<hdk::ir::InIntegerSet>(visit(in_integer_set->arg()),
                                                    in_integer_set->valueList(),
                                                    !in_integer_set->type()->nullable());
  }

  RetType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    return hdk::ir::makeExpr<hdk::ir::InSubquery>(
        in_subquery->type(), visit(in_subquery->arg()), in_subquery->nodeShared());
  }

  RetType visitCharLength(const hdk::ir::CharLengthExpr* char_length) const override {
    return hdk::ir::makeExpr<hdk::ir::CharLengthExpr>(visit(char_length->arg()),
                                                      char_length->calcEncodedLength());
  }

  RetType visitKeyForString(const hdk::ir::KeyForStringExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::KeyForStringExpr>(visit(expr->arg()));
  }

  RetType visitSampleRatio(const hdk::ir::SampleRatioExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::SampleRatioExpr>(visit(expr->arg()));
  }

  RetType visitLower(const hdk::ir::LowerExpr* expr) const override {
    return hdk::ir::makeExpr<hdk::ir::LowerExpr>(visit(expr->arg()));
  }

  RetType visitCardinality(const hdk::ir::CardinalityExpr* cardinality) const override {
    return hdk::ir::makeExpr<hdk::ir::CardinalityExpr>(visit(cardinality->arg()));
  }

  RetType visitLikeExpr(const hdk::ir::LikeExpr* like) const override {
    auto escape_expr = like->escapeExpr();
    return hdk::ir::makeExpr<hdk::ir::LikeExpr>(
        visit(like->arg()),
        visit(like->likeExpr()),
        escape_expr ? visit(escape_expr) : nullptr,
        like->isIlike(),
        like->isSimple());
  }

  RetType visitRegexpExpr(const hdk::ir::RegexpExpr* regexp) const override {
    auto escape_expr = regexp->escapeExpr();
    return hdk::ir::makeExpr<hdk::ir::RegexpExpr>(
        visit(regexp->arg()),
        visit(regexp->patternExpr()),
        escape_expr ? visit(escape_expr) : nullptr);
  }

  RetType visitWidthBucket(
      const hdk::ir::WidthBucketExpr* width_bucket_expr) const override {
    return hdk::ir::makeExpr<hdk::ir::WidthBucketExpr>(
        visit(width_bucket_expr->targetValue()),
        visit(width_bucket_expr->lowerBound()),
        visit(width_bucket_expr->upperBound()),
        visit(width_bucket_expr->partitionCount()));
  }

  RetType visitCaseExpr(const hdk::ir::CaseExpr* case_expr) const override {
    std::list<std::pair<RetType, RetType>> new_list;
    for (auto p : case_expr->exprPairs()) {
      new_list.emplace_back(visit(p.first.get()), visit(p.second.get()));
    }
    auto else_expr = case_expr->elseExpr();
    return hdk::ir::makeExpr<hdk::ir::CaseExpr>(
        case_expr->type(),
        case_expr->containsAgg(),
        new_list,
        else_expr == nullptr ? nullptr : visit(else_expr));
  }

  RetType visitDatetruncExpr(const hdk::ir::DatetruncExpr* datetrunc) const override {
    return hdk::ir::makeExpr<hdk::ir::DatetruncExpr>(datetrunc->type(),
                                                     datetrunc->containsAgg(),
                                                     datetrunc->get_field(),
                                                     visit(datetrunc->get_from_expr()));
  }

  RetType visitExtractExpr(const hdk::ir::ExtractExpr* extract) const override {
    return hdk::ir::makeExpr<hdk::ir::ExtractExpr>(extract->type(),
                                                   extract->containsAgg(),
                                                   extract->field(),
                                                   visit(extract->from()));
  }

  RetType visitArrayOper(const hdk::ir::ArrayExpr* array_expr) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < array_expr->getElementCount(); ++i) {
      args_copy.push_back(visit(array_expr->getElement(i)));
    }
    auto type = array_expr->type();
    return hdk::ir::makeExpr<hdk::ir::ArrayExpr>(
        type, args_copy, array_expr->isNull(), array_expr->isLocalAlloc());
  }

  RetType visitWindowFunction(const hdk::ir::WindowFunction* window_func) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (const auto& arg : window_func->getArgs()) {
      args_copy.push_back(visit(arg.get()));
    }
    std::vector<hdk::ir::ExprPtr> partition_keys_copy;
    for (const auto& partition_key : window_func->getPartitionKeys()) {
      partition_keys_copy.push_back(visit(partition_key.get()));
    }
    std::vector<hdk::ir::ExprPtr> order_keys_copy;
    for (const auto& order_key : window_func->getOrderKeys()) {
      order_keys_copy.push_back(visit(order_key.get()));
    }
    const auto& type = window_func->type();
    return hdk::ir::makeExpr<hdk::ir::WindowFunction>(type,
                                                      window_func->getKind(),
                                                      args_copy,
                                                      partition_keys_copy,
                                                      order_keys_copy,
                                                      window_func->getCollation());
  }

  RetType visitFunctionOper(const hdk::ir::FunctionOper* func_oper) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type = func_oper->type();
    return hdk::ir::makeExpr<hdk::ir::FunctionOper>(
        type, func_oper->getName(), args_copy);
  }

  RetType visitDatediffExpr(const hdk::ir::DatediffExpr* datediff) const override {
    return hdk::ir::makeExpr<hdk::ir::DatediffExpr>(datediff->type(),
                                                    datediff->get_field(),
                                                    visit(datediff->get_start_expr()),
                                                    visit(datediff->get_end_expr()));
  }

  RetType visitDateaddExpr(const hdk::ir::DateaddExpr* dateadd) const override {
    return hdk::ir::makeExpr<hdk::ir::DateaddExpr>(dateadd->type(),
                                                   dateadd->get_field(),
                                                   visit(dateadd->get_number_expr()),
                                                   visit(dateadd->get_datetime_expr()));
  }

  RetType visitFunctionOperWithCustomTypeHandling(
      const hdk::ir::FunctionOperWithCustomTypeHandling* func_oper) const override {
    std::vector<hdk::ir::ExprPtr> args_copy;
    for (size_t i = 0; i < func_oper->getArity(); ++i) {
      args_copy.push_back(visit(func_oper->getArg(i)));
    }
    const auto& type = func_oper->type();
    return hdk::ir::makeExpr<hdk::ir::FunctionOperWithCustomTypeHandling>(
        type, func_oper->getName(), args_copy);
  }

  RetType visitLikelihood(const hdk::ir::LikelihoodExpr* likelihood) const override {
    return hdk::ir::makeExpr<hdk::ir::LikelihoodExpr>(visit(likelihood->arg()),
                                                      likelihood->likelihood());
  }

  RetType visitAggExpr(const hdk::ir::AggExpr* agg) const override {
    RetType arg = agg->arg() ? visit(agg->arg()) : nullptr;
    return hdk::ir::makeExpr<hdk::ir::AggExpr>(
        agg->type(), agg->aggType(), arg, agg->isDistinct(), agg->arg1());
  }

  RetType visitOffsetInFragment(const hdk::ir::OffsetInFragment*) const override {
    return hdk::ir::makeExpr<hdk::ir::OffsetInFragment>();
  }
};
