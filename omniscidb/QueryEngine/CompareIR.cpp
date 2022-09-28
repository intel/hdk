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

#include "CodeGenerator.h"
#include "Execute.h"

#include <typeinfo>

namespace {

llvm::CmpInst::Predicate llvm_icmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::ICmpInst::ICMP_EQ;
    case kNE:
      return llvm::ICmpInst::ICMP_NE;
    case kLT:
      return llvm::ICmpInst::ICMP_SLT;
    case kGT:
      return llvm::ICmpInst::ICMP_SGT;
    case kLE:
      return llvm::ICmpInst::ICMP_SLE;
    case kGE:
      return llvm::ICmpInst::ICMP_SGE;
    default:
      abort();
  }
}

std::string icmp_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "lt";
    case kGT:
      return "gt";
    case kLE:
      return "le";
    case kGE:
      return "ge";
    default:
      abort();
  }
}

std::string icmp_arr_name(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "eq";
    case kNE:
      return "ne";
    case kLT:
      return "gt";
    case kGT:
      return "lt";
    case kLE:
      return "ge";
    case kGE:
      return "le";
    default:
      abort();
  }
}

llvm::CmpInst::Predicate llvm_fcmp_pred(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return llvm::CmpInst::FCMP_OEQ;
    case kNE:
      return llvm::CmpInst::FCMP_ONE;
    case kLT:
      return llvm::CmpInst::FCMP_OLT;
    case kGT:
      return llvm::CmpInst::FCMP_OGT;
    case kLE:
      return llvm::CmpInst::FCMP_OLE;
    case kGE:
      return llvm::CmpInst::FCMP_OGE;
    default:
      abort();
  }
}

}  // namespace

namespace {

std::string string_cmp_func(const SQLOps optype) {
  switch (optype) {
    case kLT:
      return "string_lt";
    case kLE:
      return "string_le";
    case kGT:
      return "string_gt";
    case kGE:
      return "string_ge";
    case kEQ:
      return "string_eq";
    case kNE:
      return "string_ne";
    default:
      abort();
  }
}

std::shared_ptr<const hdk::ir::BinOper> lower_bw_eq(const hdk::ir::BinOper* bw_eq) {
  auto& ctx = bw_eq->ctx();
  const auto eq_oper = std::make_shared<hdk::ir::BinOper>(bw_eq->type(),
                                                          bw_eq->get_contains_agg(),
                                                          kEQ,
                                                          bw_eq->get_qualifier(),
                                                          bw_eq->get_own_left_operand(),
                                                          bw_eq->get_own_right_operand());
  const auto lhs_is_null = std::make_shared<hdk::ir::UOper>(
      ctx.boolean(false), kISNULL, bw_eq->get_own_left_operand());
  const auto rhs_is_null = std::make_shared<hdk::ir::UOper>(
      ctx.boolean(false), kISNULL, bw_eq->get_own_right_operand());
  const auto both_are_null =
      Analyzer::normalizeOperExpr(kAND, kONE, lhs_is_null, rhs_is_null);
  const auto bw_eq_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(
      Analyzer::normalizeOperExpr(kOR, kONE, eq_oper, both_are_null));
  CHECK(bw_eq_oper);
  return bw_eq_oper;
}

std::shared_ptr<const hdk::ir::BinOper> make_eq(const hdk::ir::ExprPtr& lhs,
                                                const hdk::ir::ExprPtr& rhs,
                                                const SQLOps optype) {
  CHECK(IS_EQUIVALENCE(optype));
  // Sides of a tuple equality are stripped of cast operators to simplify the logic
  // in the hash table construction algorithm. Add them back here.
  auto eq_oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(
      Analyzer::normalizeOperExpr(optype, kONE, lhs, rhs));
  CHECK(eq_oper);
  return optype == kBW_EQ ? lower_bw_eq(eq_oper.get()) : eq_oper;
}

// Convert a column tuple equality expression back to a conjunction of comparisons
// so that it can be handled by the regular code generation methods.
std::shared_ptr<const hdk::ir::BinOper> lower_multicol_compare(
    const hdk::ir::BinOper* multicol_compare) {
  const auto left_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(multicol_compare->get_left_operand());
  const auto right_tuple_expr = dynamic_cast<const hdk::ir::ExpressionTuple*>(
      multicol_compare->get_right_operand());
  CHECK(left_tuple_expr && right_tuple_expr);
  const auto& left_tuple = left_tuple_expr->getTuple();
  const auto& right_tuple = right_tuple_expr->getTuple();
  CHECK_EQ(left_tuple.size(), right_tuple.size());
  CHECK_GT(left_tuple.size(), size_t(1));
  auto acc =
      make_eq(left_tuple.front(), right_tuple.front(), multicol_compare->get_optype());
  for (size_t i = 1; i < left_tuple.size(); ++i) {
    auto crt = make_eq(left_tuple[i], right_tuple[i], multicol_compare->get_optype());
    const bool nullable = acc->type()->nullable() || crt->type()->nullable();
    acc = hdk::ir::makeExpr<hdk::ir::BinOper>(
        acc->type()->ctx().boolean(nullable), false, kAND, kONE, acc, crt);
  }
  return acc;
}

void check_array_comp_cond(const hdk::ir::BinOper* bin_oper) {
  auto lhs_cv = dynamic_cast<const hdk::ir::ColumnVar*>(bin_oper->get_left_operand());
  auto rhs_cv = dynamic_cast<const hdk::ir::ColumnVar*>(bin_oper->get_right_operand());
  auto comp_op = IS_COMPARISON(bin_oper->get_optype());
  if (lhs_cv && rhs_cv && comp_op) {
    auto lhs_type = lhs_cv->type();
    auto rhs_type = rhs_cv->type();
    if (lhs_type->isArray() && rhs_type->isArray()) {
      throw std::runtime_error(
          "Comparing two full array columns is not supported yet. Please consider "
          "rewriting the full array comparison to a comparison between indexed array "
          "columns "
          "(i.e., arr1[1] {<, <=, >, >=} arr2[1]).");
    }
  }
  auto lhs_bin_oper = dynamic_cast<const hdk::ir::BinOper*>(bin_oper->get_left_operand());
  auto rhs_bin_oper =
      dynamic_cast<const hdk::ir::BinOper*>(bin_oper->get_right_operand());
  // we can do (non-)equivalence check of two encoded string
  // even if they are (indexed) array cols
  auto theta_comp = IS_COMPARISON(bin_oper->get_optype()) &&
                    !IS_EQUIVALENCE(bin_oper->get_optype()) &&
                    bin_oper->get_optype() != SQLOps::kNE;
  if (lhs_bin_oper && rhs_bin_oper && theta_comp &&
      lhs_bin_oper->get_optype() == SQLOps::kARRAY_AT &&
      rhs_bin_oper->get_optype() == SQLOps::kARRAY_AT) {
    auto lhs_arr_cv =
        dynamic_cast<const hdk::ir::ColumnVar*>(lhs_bin_oper->get_left_operand());
    auto lhs_arr_idx =
        dynamic_cast<const hdk::ir::Constant*>(lhs_bin_oper->get_right_operand());
    auto rhs_arr_cv =
        dynamic_cast<const hdk::ir::ColumnVar*>(rhs_bin_oper->get_left_operand());
    auto rhs_arr_idx =
        dynamic_cast<const hdk::ir::Constant*>(rhs_bin_oper->get_right_operand());
    if (lhs_arr_cv && rhs_arr_cv && lhs_arr_idx && rhs_arr_idx &&
        ((lhs_arr_cv->type()->isArray() &&
          lhs_arr_cv->type()->as<hdk::ir::ArrayBaseType>()->elemType()->isText()) ||
         (rhs_arr_cv->type()->isArray() &&
          rhs_arr_cv->type()->as<hdk::ir::ArrayBaseType>()->elemType()->isText()))) {
      throw std::runtime_error(
          "Comparison between string array columns is not supported yet.");
    }
  }
}

}  // namespace

llvm::Value* CodeGenerator::codegenCmp(const hdk::ir::BinOper* bin_oper,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto qualifier = bin_oper->get_qualifier();
  const auto lhs = bin_oper->get_left_operand();
  const auto rhs = bin_oper->get_right_operand();
  if (dynamic_cast<const hdk::ir::ExpressionTuple*>(lhs)) {
    CHECK(dynamic_cast<const hdk::ir::ExpressionTuple*>(rhs));
    const auto lowered = lower_multicol_compare(bin_oper);
    const auto lowered_lvs = codegen(lowered.get(), true, co);
    CHECK_EQ(size_t(1), lowered_lvs.size());
    return lowered_lvs.front();
  }
  const auto optype = bin_oper->get_optype();
  if (optype == kBW_EQ) {
    const auto bw_eq_oper = lower_bw_eq(bin_oper);
    return codegenLogical(bw_eq_oper.get(), co);
  }
  if (is_unnest(lhs) || is_unnest(rhs)) {
    throw std::runtime_error("Unnest not supported in comparisons");
  }
  check_array_comp_cond(bin_oper);
  const auto& lhs_type = lhs->type();
  const auto& rhs_type = rhs->type();

  if ((lhs_type->isString() || lhs_type->isExtDictionary()) &&
      (rhs_type->isString() || rhs_type->isExtDictionary()) &&
      !(IS_EQUIVALENCE(optype) || optype == kNE)) {
    auto cmp_str = codegenStrCmp(optype,
                                 qualifier,
                                 bin_oper->get_own_left_operand(),
                                 bin_oper->get_own_right_operand(),
                                 co);
    if (cmp_str) {
      return cmp_str;
    }
  }

  if (lhs_type->isDecimal()) {
    auto cmp_decimal_const =
        codegenCmpDecimalConst(optype, qualifier, lhs, lhs_type, rhs, co);
    if (cmp_decimal_const) {
      return cmp_decimal_const;
    }
  }
  auto lhs_lvs = codegen(lhs, true, co);
  return codegenCmp(optype, qualifier, lhs_lvs, lhs_type, rhs, co);
}

llvm::Value* CodeGenerator::codegenStrCmp(const SQLOps optype,
                                          const SQLQualifier qualifier,
                                          const hdk::ir::ExprPtr lhs,
                                          const hdk::ir::ExprPtr rhs,
                                          const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto lhs_type = lhs->type();
  const auto rhs_type = rhs->type();

  CHECK(lhs_type->isString() || lhs_type->isExtDictionary());
  CHECK(rhs_type->isString() || rhs_type->isExtDictionary());

  if (lhs_type->isExtDictionary() && rhs_type->isExtDictionary()) {
    auto lhs_dict_id = lhs_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto rhs_dict_id = rhs_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    if (lhs_dict_id == rhs_dict_id) {
      // Both operands share a dictionary

      // check if query is trying to compare a columnt against literal

      auto ir = codegenDictStrCmp(lhs, rhs, optype, co);
      if (ir) {
        return ir;
      }
    } else {
      // Both operands don't share a dictionary
      return nullptr;
    }
  }
  return nullptr;
}

llvm::Value* CodeGenerator::codegenCmpDecimalConst(const SQLOps optype,
                                                   const SQLQualifier qualifier,
                                                   const hdk::ir::Expr* lhs,
                                                   const hdk::ir::Type* lhs_type,
                                                   const hdk::ir::Expr* rhs,
                                                   const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto u_oper = dynamic_cast<const hdk::ir::UOper*>(lhs);
  if (!u_oper || u_oper->get_optype() != kCAST) {
    return nullptr;
  }
  auto rhs_constant = dynamic_cast<const hdk::ir::Constant*>(rhs);
  if (!rhs_constant) {
    return nullptr;
  }
  const auto operand = u_oper->get_operand();
  const auto& operand_type = operand->type();
  CHECK(lhs_type->isDecimal());
  auto lhs_scale = lhs_type->as<hdk::ir::DecimalType>()->scale();
  auto operand_scale =
      operand_type->isDecimal() ? operand_type->as<hdk::ir::DecimalType>()->scale() : 0;
  if (operand_type->isDecimal() && operand_scale < lhs_scale) {
    // lhs decimal type has smaller scale
  } else if (operand_type->isInteger() && 0 < lhs_scale) {
    // lhs is integer, no need to scale it all the way up to the cmp expr scale
  } else {
    return nullptr;
  }

  auto scale_diff = lhs_scale - operand_scale - 1;
  int64_t bigintval = rhs_constant->get_constval().bigintval;
  bool negative = false;
  if (bigintval < 0) {
    negative = true;
    bigintval = -bigintval;
  }
  int64_t truncated_decimal = bigintval / exp_to_scale(scale_diff);
  int64_t decimal_tail = bigintval % exp_to_scale(scale_diff);
  if (truncated_decimal % 10 == 0 && decimal_tail > 0) {
    truncated_decimal += 1;
  }
  auto new_type =
      lhs_type->ctx().decimal64(19, lhs_scale - scale_diff, operand_type->nullable());
  if (negative) {
    truncated_decimal = -truncated_decimal;
  }
  Datum d;
  d.bigintval = truncated_decimal;
  const auto new_rhs_lit =
      hdk::ir::makeExpr<hdk::ir::Constant>(new_type, rhs_constant->get_is_null(), d);
  const auto operand_lv = codegen(operand, true, co).front();
  const auto lhs_lv = codegenCast(operand_lv, operand_type, new_type, false, false, co);
  return codegenCmp(optype, qualifier, {lhs_lv}, new_type, new_rhs_lit.get(), co);
}

llvm::Value* CodeGenerator::codegenCmp(const SQLOps optype,
                                       const SQLQualifier qualifier,
                                       std::vector<llvm::Value*> lhs_lvs,
                                       const hdk::ir::Type* lhs_type,
                                       const hdk::ir::Expr* rhs,
                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(IS_COMPARISON(optype));
  const auto& rhs_type = rhs->type();
  if (rhs_type->isArray()) {
    return codegenQualifierCmp(optype, qualifier, lhs_lvs, rhs, co);
  }
  auto rhs_lvs = codegen(rhs, true, co);
  CHECK_EQ(kONE, qualifier);
  CHECK((lhs_type->isString() && rhs_type->isString()) ||
        (lhs_type->id() == rhs_type->id()))
      << lhs_type->toString() << " " << rhs_type->toString();
  const auto null_check_suffix = get_null_check_suffix(lhs_type, rhs_type);
  if (lhs_type->isInteger() || lhs_type->isDecimal() || lhs_type->isDateTime() ||
      lhs_type->isBoolean() || lhs_type->isString() || lhs_type->isExtDictionary() ||
      lhs_type->isInterval()) {
    if (lhs_type->isString() || lhs_type->isExtDictionary()) {
      CHECK_EQ(lhs_type->isString(), rhs_type->isString())
          << lhs_type->toString() << " " << rhs_type->toString();
      if (!lhs_type->isExtDictionary()) {
        // unpack pointer + length if necessary
        if (lhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), lhs_lvs.size());
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {lhs_lvs.front()}));
          lhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {lhs_lvs.front()}));
        }
        if (rhs_lvs.size() != 3) {
          CHECK_EQ(size_t(1), rhs_lvs.size());
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_ptr", {rhs_lvs.front()}));
          rhs_lvs.push_back(cgen_state_->emitCall("extract_str_len", {rhs_lvs.front()}));
        }
        std::vector<llvm::Value*> str_cmp_args{
            lhs_lvs[1], lhs_lvs[2], rhs_lvs[1], rhs_lvs[2]};
        if (!null_check_suffix.empty()) {
          str_cmp_args.push_back(cgen_state_->inlineIntNull(lhs_type->ctx().boolean()));
        }
        return cgen_state_->emitCall(
            string_cmp_func(optype) + (null_check_suffix.empty() ? "" : "_nullable"),
            str_cmp_args);
      } else {
        CHECK(optype == kEQ || optype == kNE);
      }
    }

    if (lhs_type->isBoolean() && rhs_type->isBoolean()) {
      auto& lhs_lv = lhs_lvs.front();
      auto& rhs_lv = rhs_lvs.front();
      CHECK(lhs_lv->getType()->isIntegerTy());
      CHECK(rhs_lv->getType()->isIntegerTy());
      if (lhs_lv->getType()->getIntegerBitWidth() <
          rhs_lv->getType()->getIntegerBitWidth()) {
        lhs_lv =
            cgen_state_->castToTypeIn(lhs_lv, rhs_lv->getType()->getIntegerBitWidth());
      } else {
        rhs_lv =
            cgen_state_->castToTypeIn(rhs_lv, lhs_lv->getType()->getIntegerBitWidth());
      }
    }

    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateICmp(
                     llvm_icmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(
                     icmp_name(optype) + "_" + numeric_type_name(lhs_type) +
                         null_check_suffix,
                     {lhs_lvs.front(),
                      rhs_lvs.front(),
                      cgen_state_->llInt(inline_int_null_value(lhs_type)),
                      cgen_state_->inlineIntNull(lhs_type->ctx().boolean())});
  }

  if (lhs_type->isFloatingPoint()) {
    return null_check_suffix.empty()
               ? cgen_state_->ir_builder_.CreateFCmp(
                     llvm_fcmp_pred(optype), lhs_lvs.front(), rhs_lvs.front())
               : cgen_state_->emitCall(
                     icmp_name(optype) + "_" + numeric_type_name(lhs_type) +
                         null_check_suffix,
                     {lhs_lvs.front(),
                      rhs_lvs.front(),
                      lhs_type->isFp32() ? cgen_state_->llFp(NULL_FLOAT)
                                         : cgen_state_->llFp(NULL_DOUBLE),
                      cgen_state_->inlineIntNull(lhs_type->ctx().boolean())});
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* CodeGenerator::codegenQualifierCmp(const SQLOps optype,
                                                const SQLQualifier qualifier,
                                                std::vector<llvm::Value*> lhs_lvs,
                                                const hdk::ir::Expr* rhs,
                                                const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto rhs_type = rhs->type();
  CHECK(rhs_type->isArray());
  auto target_type = rhs_type->as<hdk::ir::ArrayBaseType>()->elemType();
  const hdk::ir::Expr* arr_expr{rhs};
  if (dynamic_cast<const hdk::ir::UOper*>(rhs)) {
    const auto cast_arr = static_cast<const hdk::ir::UOper*>(rhs);
    CHECK_EQ(kCAST, cast_arr->get_optype());
    arr_expr = cast_arr->get_operand();
  }
  auto arr_type = arr_expr->type();
  CHECK(arr_type->isArray());
  auto elem_type = arr_type->as<hdk::ir::ArrayBaseType>()->elemType();
  auto rhs_lvs = codegen(arr_expr, true, co);
  CHECK_NE(kONE, qualifier);
  std::string fname{std::string("array_") + (qualifier == kANY ? "any" : "all") + "_" +
                    icmp_arr_name(optype)};
  if (target_type->isString()) {
    if (config_.exec.watchdog.enable) {
      throw WatchdogException(
          "Comparison between a dictionary-encoded and a none-encoded string would be "
          "slow");
    }
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    fname += "_str";
  }
  if (elem_type->isInteger() || elem_type->isBoolean() || elem_type->isString() ||
      elem_type->isExtDictionary() || elem_type->isDecimal()) {
    fname += ("_" + numeric_type_name(elem_type));
  } else {
    CHECK(elem_type->isFloatingPoint());
    fname += elem_type->isFp64() ? "_double" : "_float";
  }
  if (target_type->isString()) {
    CHECK_EQ(size_t(3), lhs_lvs.size());
    CHECK(elem_type->isExtDictionary());
    return cgen_state_->emitExternalCall(
        fname,
        get_int_type(1, cgen_state_->context_),
        {rhs_lvs.front(),
         posArg(arr_expr),
         lhs_lvs[1],
         lhs_lvs[2],
         cgen_state_->llInt(int64_t(executor()->getStringDictionaryProxy(
             elem_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
             executor()->getRowSetMemoryOwner(),
             true))),
         cgen_state_->inlineIntNull(elem_type)});
  }
  if (target_type->isInteger() || target_type->isBoolean() || target_type->isString() ||
      target_type->isExtDictionary() || target_type->isDecimal()) {
    fname += ("_" + numeric_type_name(target_type));
  } else {
    CHECK(target_type->isFloatingPoint());
    fname += target_type->isFp64() ? "_double" : "_float";
  }
  return cgen_state_->emitExternalCall(
      fname,
      get_int_type(1, cgen_state_->context_),
      {rhs_lvs.front(),
       posArg(arr_expr),
       lhs_lvs.front(),
       elem_type->isFloatingPoint()
           ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(elem_type))
           : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(elem_type))});
}
