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

#include "IR/TypeUtils.h"

#include "CodeGenerator.h"
#include "Codec.h"
#include "Execute.h"
#include "WindowContext.h"

// Code generation routines and helpers for working with column expressions.

namespace {

// Return the right decoder for a given column expression. Doesn't handle
// variable length data. The decoder encapsulates the code generation logic.
std::shared_ptr<Decoder> get_col_decoder(const hdk::ir::ColumnVar* col_var) {
  const auto& type = col_var->type();
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
      return std::make_shared<FixedWidthInt>(type->size());
    case hdk::ir::Type::kFloatingPoint:
      return std::make_shared<FixedWidthReal>(type->isFp64());
    case hdk::ir::Type::kExtDictionary:
      if (type->size() < type->canonicalSize()) {
        return std::make_shared<FixedWidthUnsigned>(type->size());
      }
      return std::make_shared<FixedWidthInt>(type->size());
    case hdk::ir::Type::kDate:
      if (type->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay) {
        return std::make_shared<FixedWidthSmallDate>(type->size());
      }
      return std::make_shared<FixedWidthInt>(type->size());
    default:
      abort();
  }
}

size_t get_col_bit_width(const hdk::ir::ColumnVar* col_var) {
  return get_bit_width(col_var->type());
}

int adjusted_range_table_index(const hdk::ir::ColumnVar* col_var) {
  return col_var->rteIdx() == -1 ? 0 : col_var->rteIdx();
}

}  // namespace

std::vector<llvm::Value*> CodeGenerator::codegenColumn(const hdk::ir::ColumnVar* col_var,
                                                       const bool fetch_column,
                                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (col_var->rteIdx() <= 0 || cgen_state_->outer_join_match_found_per_level_.empty() ||
      !foundOuterJoinMatch(col_var->rteIdx())) {
    return codegenColVar(col_var, fetch_column, true, co);
  }
  return codegenOuterJoinNullPlaceholder(col_var, fetch_column, co);
}

std::vector<llvm::Value*> CodeGenerator::codegenColVar(const hdk::ir::ColumnVar* col_var,
                                                       const bool fetch_column,
                                                       const bool update_query_plan,
                                                       const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const bool hoist_literals = co.hoist_literals;
  const int rte_idx = adjusted_range_table_index(col_var);
  CHECK_LT(static_cast<size_t>(rte_idx), cgen_state_->frag_offsets_.size());
  if (col_var->tableId() > 0) {
    if (col_var->isVirtual()) {
      return {codegenRowId(col_var, co)};
    }
  }
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
  }
  const int local_col_id = plan_state_->getLocalColumnId(col_var, fetch_column);
  const auto window_func_context =
      WindowProjectNodeContext::getActiveWindowFunctionContext(executor());
  // only generate the decoding code once; if a column has been previously
  // fetched in the generated IR, we'll reuse it
  if (!window_func_context) {
    auto it = cgen_state_->fetch_cache_.find(local_col_id);
    if (it != cgen_state_->fetch_cache_.end()) {
      return {it->second};
    }
  }
  const auto hash_join_lhs = hashJoinLhs(col_var);
  // Note(jclay): This has been prone to cause failures in some overlaps joins.
  // I believe most of the issues are worked out now, but a good place to check if
  // failures are happening.

  // Use the already fetched left-hand side of an equi-join if the types are identical.
  // Currently, types can only be different because of different underlying dictionaries.
  if (hash_join_lhs && hash_join_lhs->type()->equal(col_var->type())) {
    if (plan_state_->isLazyFetchColumn(col_var)) {
      plan_state_->columns_to_fetch_.insert(column_var_to_descriptor(col_var));
    }
    return codegen(hash_join_lhs.get(), fetch_column, co);
  }
  auto pos_arg = posArg(col_var);
  if (window_func_context) {
    pos_arg = codegenWindowPosition(window_func_context, pos_arg);
  }
  auto col_byte_stream = colByteStream(col_var, fetch_column, hoist_literals);
  if (plan_state_->isLazyFetchColumn(col_var)) {
    if (update_query_plan) {
      plan_state_->columns_to_not_fetch_.insert(column_var_to_descriptor(col_var));
    }
    if (rte_idx > 0) {
      const auto offset = cgen_state_->frag_offsets_[rte_idx];
      if (offset) {
        return {cgen_state_->ir_builder_.CreateAdd(pos_arg, offset)};
      } else {
        return {pos_arg};
      }
    }
    return {pos_arg};
  }
  const auto& col_type = col_var->type();
  if (col_type->isString()) {
    const auto varlen_str_column_lvs =
        codegenVariableLengthStringColVar(col_byte_stream, pos_arg);
    if (!window_func_context) {
      auto it_ok = cgen_state_->fetch_cache_.insert(
          std::make_pair(local_col_id, varlen_str_column_lvs));
      CHECK(it_ok.second);
    }
    return varlen_str_column_lvs;
  }
  if (col_type->isArray()) {
    return {col_byte_stream};
  }
  if (window_func_context) {
    return {codegenFixedLengthColVarInWindow(col_var, col_byte_stream, pos_arg)};
  }
  const auto fixed_length_column_lv =
      codegenFixedLengthColVar(col_var, col_byte_stream, pos_arg);
  auto it_ok = cgen_state_->fetch_cache_.insert(
      std::make_pair(local_col_id, std::vector<llvm::Value*>{fixed_length_column_lv}));
  return {it_ok.first->second};
}

llvm::Value* CodeGenerator::codegenWindowPosition(
    WindowFunctionContext* window_func_context,
    llvm::Value* pos_arg) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto window_position = cgen_state_->emitCall(
      "row_number_window_func",
      {cgen_state_->llInt(reinterpret_cast<const int64_t>(window_func_context->output())),
       pos_arg});
  return window_position;
}

// Generate code for fixed length column types (number, timestamp or date,
// dictionary-encoded string)
llvm::Value* CodeGenerator::codegenFixedLengthColVar(const hdk::ir::ColumnVar* col_var,
                                                     llvm::Value* col_byte_stream,
                                                     llvm::Value* pos_arg) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto decoder = get_col_decoder(col_var);
  auto dec_val = decoder->codegenDecode(col_byte_stream, pos_arg, cgen_state_->module_);
  cgen_state_->ir_builder_.Insert(dec_val);
  auto dec_type = dec_val->getType();
  llvm::Value* dec_val_cast{nullptr};
  auto col_type = col_var->type();
  if (dec_type->isIntegerTy()) {
    auto dec_width = static_cast<llvm::IntegerType*>(dec_type)->getBitWidth();
    auto col_width = get_col_bit_width(col_var);
    dec_val_cast = cgen_state_->ir_builder_.CreateCast(
        static_cast<size_t>(col_width) > dec_width ? llvm::Instruction::CastOps::SExt
                                                   : llvm::Instruction::CastOps::Trunc,
        dec_val,
        get_int_type(col_width, cgen_state_->context_));
    if (col_type->size() < col_type->canonicalSize() && !col_type->isDate() &&
        col_type->nullable()) {
      dec_val_cast = codgenAdjustFixedEncNull(dec_val_cast, col_type);
    }
  } else {
    CHECK(col_type->size() == col_type->canonicalSize());
    CHECK(dec_type->isFloatTy() || dec_type->isDoubleTy());
    if (dec_type->isDoubleTy()) {
      CHECK(col_type->isFp64());
    } else if (dec_type->isFloatTy()) {
      CHECK(col_type->isFp32());
    }
    dec_val_cast = dec_val;
  }
  CHECK(dec_val_cast);
  return dec_val_cast;
}

llvm::Value* CodeGenerator::codegenFixedLengthColVarInWindow(
    const hdk::ir::ColumnVar* col_var,
    llvm::Value* col_byte_stream,
    llvm::Value* pos_arg) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  const auto pos_is_valid =
      cgen_state_->ir_builder_.CreateICmpSGE(pos_arg, cgen_state_->llInt(int64_t(0)));
  const auto pos_valid_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "window.pos_valid", cgen_state_->current_func_);
  const auto pos_notvalid_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "window.pos_notvalid", cgen_state_->current_func_);
  cgen_state_->ir_builder_.CreateCondBr(pos_is_valid, pos_valid_bb, pos_notvalid_bb);
  cgen_state_->ir_builder_.SetInsertPoint(pos_valid_bb);
  const auto fixed_length_column_lv =
      codegenFixedLengthColVar(col_var, col_byte_stream, pos_arg);
  cgen_state_->ir_builder_.CreateBr(pos_notvalid_bb);
  cgen_state_->ir_builder_.SetInsertPoint(pos_notvalid_bb);
  const auto window_func_call_phi =
      cgen_state_->ir_builder_.CreatePHI(fixed_length_column_lv->getType(), 2);
  window_func_call_phi->addIncoming(fixed_length_column_lv, pos_valid_bb);
  auto col_type = col_var->type();
  const auto null_lv =
      col_type->isFloatingPoint()
          ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(col_type))
          : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(col_type));
  window_func_call_phi->addIncoming(null_lv, orig_bb);
  return window_func_call_phi;
}

std::vector<llvm::Value*> CodeGenerator::codegenVariableLengthStringColVar(
    llvm::Value* col_byte_stream,
    llvm::Value* pos_arg) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  // real (not dictionary-encoded) strings; store the pointer to the payload
  auto ptr_and_len =
      cgen_state_->emitExternalCall("string_decode",
                                    get_int_type(64, cgen_state_->context_),
                                    {col_byte_stream, pos_arg});
  // Unpack the pointer + length, see string_decode function.
  auto str_lv = cgen_state_->emitCall("extract_str_ptr", {ptr_and_len});
  auto len_lv = cgen_state_->emitCall("extract_str_len", {ptr_and_len});
  return {ptr_and_len, str_lv, len_lv};
}

llvm::Value* CodeGenerator::codegenRowId(const hdk::ir::ColumnVar* col_var,
                                         const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto offset_lv = cgen_state_->frag_offsets_[adjusted_range_table_index(col_var)];
  llvm::Value* start_rowid_lv{nullptr};
  const auto& table_generation = executor()->getTableGeneration(col_var->tableId());
  if (table_generation.start_rowid > 0) {
    // Handle the multi-node case: each leaf receives a start rowid used
    // to offset the local rowid and generate a cluster-wide unique rowid.
    Datum d;
    d.bigintval = table_generation.start_rowid;
    const auto start_rowid =
        hdk::ir::makeExpr<hdk::ir::Constant>(col_var->type(), false, d);
    const auto start_rowid_lvs = codegen(start_rowid.get(), col_var->type(), co);
    CHECK_EQ(size_t(1), start_rowid_lvs.size());
    start_rowid_lv = start_rowid_lvs.front();
  }
  auto rowid_lv = posArg(col_var);
  if (offset_lv) {
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, offset_lv);
  } else if (col_var->rteIdx() > 0) {
    auto frag_off_ptr = get_arg_by_name(cgen_state_->row_func_, "frag_row_off");
    auto input_off_ptr = cgen_state_->ir_builder_.CreateGEP(
        frag_off_ptr->getType()->getScalarType()->getPointerElementType(),
        frag_off_ptr,
        cgen_state_->llInt(int32_t(col_var->rteIdx())));
    auto rowid_offset_lv = cgen_state_->ir_builder_.CreateLoad(
        input_off_ptr->getType()->getPointerElementType(), input_off_ptr);
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, rowid_offset_lv);
  }
  if (table_generation.start_rowid > 0) {
    CHECK(start_rowid_lv);
    rowid_lv = cgen_state_->ir_builder_.CreateAdd(rowid_lv, start_rowid_lv);
  }
  return rowid_lv;
}

llvm::Value* CodeGenerator::codgenAdjustFixedEncNull(llvm::Value* val,
                                                     const hdk::ir::Type* col_type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK_LT(col_type->size(), col_type->canonicalSize());
  const auto col_phys_width = col_type->size() * 8;
  auto from_typename = "int" + std::to_string(col_phys_width) + "_t";
  auto adjusted = cgen_state_->ir_builder_.CreateCast(
      llvm::Instruction::CastOps::Trunc,
      val,
      get_int_type(col_phys_width, cgen_state_->context_));
  if (col_type->isExtDictionary()) {
    from_typename = "u" + from_typename;
    llvm::Value* from_null{nullptr};
    switch (col_type->size()) {
      case 1:
        from_null = cgen_state_->llInt(std::numeric_limits<uint8_t>::max());
        break;
      case 2:
        from_null = cgen_state_->llInt(std::numeric_limits<uint16_t>::max());
        break;
      default:
        CHECK(false);
    }
    return cgen_state_->emitCall(
        "cast_" + from_typename + "_to_" + numeric_type_name(col_type) + "_nullable",
        {adjusted, from_null, cgen_state_->inlineIntNull(col_type)});
  }
  auto col_phys_type = col_type->ctx().integer(col_type->size());
  return cgen_state_->emitCall(
      "cast_" + from_typename + "_to_" + numeric_type_name(col_type) + "_nullable",
      {adjusted,
       cgen_state_->inlineIntNull(col_phys_type),
       cgen_state_->inlineIntNull(col_type)});
}

llvm::Value* CodeGenerator::foundOuterJoinMatch(const size_t nesting_level) const {
  CHECK_GE(nesting_level, size_t(1));
  CHECK_LE(nesting_level,
           static_cast<size_t>(cgen_state_->outer_join_match_found_per_level_.size()));
  return cgen_state_->outer_join_match_found_per_level_[nesting_level - 1];
}

std::vector<llvm::Value*> CodeGenerator::codegenOuterJoinNullPlaceholder(
    const hdk::ir::ColumnVar* col_var,
    const bool fetch_column,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto grouped_col_lv = resolveGroupedColumnReference(col_var);
  if (grouped_col_lv) {
    return {grouped_col_lv};
  }
  const auto outer_join_args_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_args", cgen_state_->current_func_);
  const auto outer_join_nulls_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_nulls", cgen_state_->current_func_);
  const auto phi_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "outer_join_phi", cgen_state_->current_func_);
  const auto outer_join_match_lv = foundOuterJoinMatch(col_var->rteIdx());
  CHECK(outer_join_match_lv);
  cgen_state_->ir_builder_.CreateCondBr(
      outer_join_match_lv, outer_join_args_bb, outer_join_nulls_bb);
  const auto back_from_outer_join_bb = llvm::BasicBlock::Create(
      cgen_state_->context_, "back_from_outer_join", cgen_state_->current_func_);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_args_bb);
  Executor::FetchCacheAnchor anchor(cgen_state_);
  const auto orig_lvs = codegenColVar(col_var, fetch_column, true, co);
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  cgen_state_->ir_builder_.SetInsertPoint(outer_join_nulls_bb);
  auto null_type = col_var->type();
  if ((null_type->isString()) || null_type->isArray()) {
    throw std::runtime_error("Projection type " + null_type->toString() +
                             " not supported for outer joins yet");
  }
  const auto null_constant =
      hdk::ir::makeExpr<hdk::ir::Constant>(null_type, true, Datum{0});
  const auto null_target_lvs =
      codegen(null_constant.get(),
              false,
              CompilationOptions{
                  ExecutorDeviceType::CPU, false, ExecutorOptLevel::Default, false});
  cgen_state_->ir_builder_.CreateBr(phi_bb);
  CHECK_EQ(orig_lvs.size(), null_target_lvs.size());
  cgen_state_->ir_builder_.SetInsertPoint(phi_bb);
  std::vector<llvm::Value*> target_lvs;
  for (size_t i = 0; i < orig_lvs.size(); ++i) {
    const auto target_type = orig_lvs[i]->getType();
    CHECK_EQ(target_type, null_target_lvs[i]->getType());
    auto target_phi = cgen_state_->ir_builder_.CreatePHI(target_type, 2);
    target_phi->addIncoming(orig_lvs[i], outer_join_args_bb);
    target_phi->addIncoming(null_target_lvs[i], outer_join_nulls_bb);
    target_lvs.push_back(target_phi);
  }
  cgen_state_->ir_builder_.CreateBr(back_from_outer_join_bb);
  cgen_state_->ir_builder_.SetInsertPoint(back_from_outer_join_bb);
  return target_lvs;
}

llvm::Value* CodeGenerator::resolveGroupedColumnReference(
    const hdk::ir::ColumnVar* col_var) {
  auto col_id = col_var->columnId();
  if (col_var->rteIdx() >= 0) {
    return nullptr;
  }
  CHECK((col_id == 0) || (col_var->rteIdx() >= 0 && col_var->tableId() > 0));
  const auto var = dynamic_cast<const hdk::ir::Var*>(col_var);
  CHECK(var);
  col_id = var->varNo();
  CHECK_GE(col_id, 1);
  if (var->whichRow() == hdk::ir::Var::kGROUPBY) {
    CHECK_LE(static_cast<size_t>(col_id), cgen_state_->group_by_expr_cache_.size());
    return cgen_state_->group_by_expr_cache_[col_id - 1];
  }
  return nullptr;
}

// returns the byte stream argument and the position for the given column
llvm::Value* CodeGenerator::colByteStream(const hdk::ir::ColumnVar* col_var,
                                          const bool fetch_column,
                                          const bool hoist_literals) {
  CHECK_GE(cgen_state_->row_func_->arg_size(), size_t(3));
  const auto stream_arg_name =
      "col_buf" + std::to_string(plan_state_->getLocalColumnId(col_var, fetch_column));
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == stream_arg_name) {
      CHECK(arg.getType()->isPointerTy());
      CHECK_EQ(arg.getType(),
               llvm::Type::getInt8PtrTy(cgen_state_->context_,
                                        arg.getType()->getPointerAddressSpace()))
          << ", actual arg type is " << toString(arg.getType());
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* CodeGenerator::posArg(const hdk::ir::Expr* expr) const {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto col_var = dynamic_cast<const hdk::ir::ColumnVar*>(expr);
  if (col_var && col_var->rteIdx() > 0) {
    const auto hash_pos_it = cgen_state_->scan_idx_to_hash_pos_.find(col_var->rteIdx());
    CHECK(hash_pos_it != cgen_state_->scan_idx_to_hash_pos_.end());
    if (hash_pos_it->second->getType()->isPointerTy()) {
      CHECK(hash_pos_it->second->getType()->getPointerElementType()->isIntegerTy(32));
      llvm::Value* result = cgen_state_->ir_builder_.CreateLoad(
          hash_pos_it->second->getType()->getPointerElementType(), hash_pos_it->second);
      result = cgen_state_->ir_builder_.CreateSExt(
          result, get_int_type(64, cgen_state_->context_));
      return result;
    }
    return hash_pos_it->second;
  }
  for (auto& arg : cgen_state_->row_func_->args()) {
    if (arg.getName() == "pos") {
      CHECK(arg.getType()->isIntegerTy(64));
      return &arg;
    }
  }
  abort();
}

const hdk::ir::Expr* remove_cast_to_int(const hdk::ir::Expr* expr) {
  const auto uoper = dynamic_cast<const hdk::ir::UOper*>(expr);
  if (!uoper || !uoper->isCast()) {
    return nullptr;
  }
  const auto& target_type = uoper->type();
  if (!target_type->isInteger()) {
    return nullptr;
  }
  return uoper->operand();
}

hdk::ir::ExprPtr CodeGenerator::hashJoinLhs(const hdk::ir::ColumnVar* rhs) const {
  for (const auto& tautological_eq : plan_state_->join_info_.equi_join_tautologies_) {
    CHECK(tautological_eq->isEquivalence());
    if (dynamic_cast<const hdk::ir::ExpressionTuple*>(tautological_eq->leftOperand())) {
      auto lhs_col = hashJoinLhsTuple(rhs, tautological_eq.get());
      if (lhs_col) {
        return lhs_col;
      }
    } else {
      auto eq_right_op = tautological_eq->rightOperand();
      if (!(rhs->type()->isString() || rhs->type()->isExtDictionary())) {
        eq_right_op = remove_cast_to_int(eq_right_op);
      }
      if (!eq_right_op) {
        eq_right_op = tautological_eq->rightOperand();
      }
      if (*eq_right_op == *rhs) {
        auto eq_left_op = tautological_eq->leftOperand();
        if (!(eq_left_op->type()->isString() || eq_left_op->type()->isExtDictionary())) {
          eq_left_op = remove_cast_to_int(eq_left_op);
        }
        if (!eq_left_op) {
          eq_left_op = tautological_eq->leftOperand();
        }
        const auto eq_left_op_col = dynamic_cast<const hdk::ir::ColumnVar*>(eq_left_op);
        CHECK(eq_left_op_col);
        if (eq_left_op_col->rteIdx() != 0) {
          return nullptr;
        }
        if (rhs->type()->isString() || rhs->type()->isExtDictionary()) {
          return eq_left_op->shared();
        }
        if (rhs->type()->isArray()) {
          // Note(jclay): Can this be restored from copy as above?
          // If we fall through to the below return statement,
          // a superfulous cast from DOUBLE[] to DOUBLE[] is made and
          // this fails at a later stage in codegen.
          return nullptr;
        }
        return hdk::ir::makeExpr<hdk::ir::UOper>(
            rhs->type(), false, kCAST, eq_left_op->shared());
      }
    }
  }
  return nullptr;
}

std::shared_ptr<const hdk::ir::ColumnVar> CodeGenerator::hashJoinLhsTuple(
    const hdk::ir::ColumnVar* rhs,
    const hdk::ir::BinOper* tautological_eq) const {
  const auto lhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(tautological_eq->leftOperand());
  const auto rhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(tautological_eq->rightOperand());
  CHECK(lhs_tuple_expr && rhs_tuple_expr);
  const auto& lhs_tuple = lhs_tuple_expr->tuple();
  const auto& rhs_tuple = rhs_tuple_expr->tuple();
  CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
  for (size_t i = 0; i < lhs_tuple.size(); ++i) {
    if (*rhs_tuple[i] == *rhs) {
      const auto lhs_col =
          std::static_pointer_cast<const hdk::ir::ColumnVar>(lhs_tuple[i]);
      return lhs_col->rteIdx() == 0 ? lhs_col : nullptr;
    }
  }
  return nullptr;
}
