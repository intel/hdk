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
#include "Compiler/Backend.h"
#include "Execute.h"

std::vector<llvm::Value*> CodeGenerator::codegen(const hdk::ir::Constant* constant,
                                                 bool use_dict_encoding,
                                                 int dict_id,
                                                 const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  compiler::CodegenTraits cgen_traits =
      compiler::CodegenTraits::get(co.codegen_traits_desc);
  if (co.hoist_literals) {
    std::vector<const hdk::ir::Constant*> constants(
        executor()->deviceCount(co.device_type), constant);
    return codegenHoistedConstants(constants, use_dict_encoding, dict_id);
  }
  const auto& cst_type = constant->type();
  switch (cst_type->id()) {
    case hdk::ir::Type::kBoolean:
      return {llvm::ConstantInt::get(get_int_type(8, cgen_state_->context_),
                                     constant->value().boolval)};
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kInterval:
      return {CodeGenerator::codegenIntConst(constant, cgen_state_)};
    case hdk::ir::Type::kFloatingPoint:
      if (cst_type->isFp32()) {
        return {llvm::ConstantFP::get(llvm::Type::getFloatTy(cgen_state_->context_),
                                      constant->value().floatval)};
      } else {
        CHECK(cst_type->isFp64());
        return {llvm::ConstantFP::get(llvm::Type::getDoubleTy(cgen_state_->context_),
                                      constant->value().doubleval)};
      }
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText: {
      CHECK(constant->value().stringval || constant->isNull());
      if (constant->isNull()) {
        if (use_dict_encoding) {
          return {
              cgen_state_->llInt(static_cast<int32_t>(inline_int_null_value(cst_type)))};
        }
        return {cgen_state_->llInt(int64_t(0)),
                llvm::Constant::getNullValue(
                    cgen_traits.localPointerType(get_int_type(8, cgen_state_->context_))),
                cgen_state_->llInt(int32_t(0))};
      }
      const auto& str_const = *constant->value().stringval;
      if (use_dict_encoding) {
        return {
            cgen_state_->llInt(executor()
                                   ->getStringDictionaryProxy(
                                       dict_id, executor()->getRowSetMemoryOwner(), true)
                                   ->getIdOfString(str_const))};
      }
      return {cgen_state_->llInt(int64_t(0)),
              cgen_state_->addStringConstant(str_const, co),
              cgen_state_->llInt(static_cast<int32_t>(str_const.size()))};
    }
    default:
      CHECK(false);
  }
  abort();
}

llvm::ConstantInt* CodeGenerator::codegenIntConst(const hdk::ir::Constant* constant,
                                                  CgenState* cgen_state) {
  auto type = constant->type();
  if (constant->isNull()) {
    return cgen_state->inlineIntNull(type);
  }
  switch (type->id()) {
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (type->size()) {
        case 1:
          return cgen_state->llInt(constant->value().tinyintval);
        case 2:
          return cgen_state->llInt(constant->value().smallintval);
        case 4:
          return cgen_state->llInt(constant->value().intval);
        case 8:
          return cgen_state->llInt(constant->value().bigintval);
        default:
          UNREACHABLE();
      }
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kInterval:
      return cgen_state->llInt(constant->value().bigintval);
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return nullptr;
}

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstantsLoads(
    const hdk::ir::Type* type,
    const bool use_dict_encoding,
    const int dict_id,
    const int16_t lit_off) {
  AUTOMATIC_IR_METADATA(cgen_state_);

  std::string literal_name = "literal_" + std::to_string(lit_off);
  auto lit_buff_query_func_lv = get_arg_by_name(cgen_state_->query_func_, "literals");
  const auto lit_buf_start = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
      get_int_type(8, cgen_state_->context_),
      lit_buff_query_func_lv,
      cgen_state_->llInt(lit_off));
  if (type->isString() && !use_dict_encoding) {
    CHECK_EQ(size_t(4),
             CgenState::literalBytes(CgenState::LiteralValue(std::string(""))));
    auto off_and_len_ptr = cgen_state_->query_func_entry_ir_builder_.CreateBitCast(
        lit_buf_start,
        llvm::PointerType::get(get_int_type(32, cgen_state_->context_),
                               lit_buf_start->getType()->getPointerAddressSpace()));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->query_func_entry_ir_builder_.CreateLoad(
        get_int_type(32, cgen_state_->context_), off_and_len_ptr);
    auto off_lv = cgen_state_->query_func_entry_ir_builder_.CreateLShr(
        cgen_state_->query_func_entry_ir_builder_.CreateAnd(
            off_and_len, cgen_state_->llInt(int32_t(0xffff0000))),
        cgen_state_->llInt(int32_t(16)));
    auto len_lv = cgen_state_->query_func_entry_ir_builder_.CreateAnd(
        off_and_len, cgen_state_->llInt(int32_t(0x0000ffff)));

    auto var_start = cgen_state_->llInt(int64_t(0));
    auto var_start_address = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
        llvm::PointerType::get(
            cgen_state_->context_,
            lit_buff_query_func_lv->getType()->getPointerAddressSpace()),
        lit_buff_query_func_lv,
        off_lv);
    auto var_length = len_lv;

    var_start->setName(literal_name + "_start");
    var_start_address->setName(literal_name + "_start_address");
    var_length->setName(literal_name + "_length");

    return {var_start, var_start_address, var_length};
  } else if (type->isArray() && !use_dict_encoding) {
    auto off_and_len_ptr = cgen_state_->query_func_entry_ir_builder_.CreateBitCast(
        lit_buf_start,
        llvm::PointerType::get(get_int_type(32, cgen_state_->context_),
                               lit_buf_start->getType()->getPointerAddressSpace()));
    // packed offset + length, 16 bits each
    auto off_and_len = cgen_state_->query_func_entry_ir_builder_.CreateLoad(
        off_and_len_ptr->getType()->getPointerElementType(), off_and_len_ptr);
    auto off_lv = cgen_state_->query_func_entry_ir_builder_.CreateLShr(
        cgen_state_->query_func_entry_ir_builder_.CreateAnd(
            off_and_len, cgen_state_->llInt(int32_t(0xffff0000))),
        cgen_state_->llInt(int32_t(16)));
    auto len_lv = cgen_state_->query_func_entry_ir_builder_.CreateAnd(
        off_and_len, cgen_state_->llInt(int32_t(0x0000ffff)));

    auto var_start_address = cgen_state_->query_func_entry_ir_builder_.CreateGEP(
        lit_buff_query_func_lv->getType()->getScalarType()->getPointerElementType(),
        lit_buff_query_func_lv,
        off_lv);
    auto var_length = len_lv;

    var_start_address->setName(literal_name + "_start_address");
    var_length->setName(literal_name + "_length");

    return {var_start_address, var_length};
  }

  llvm::Type* val_type{nullptr};
  const auto val_bits = get_bit_width(type);
  CHECK_EQ(size_t(0), val_bits % 8);
  if (type->isInteger() || type->isDecimal() || type->isDateTime() ||
      type->isInterval() || type->isString() || type->isBoolean()) {
    val_type = get_int_type(val_bits, cgen_state_->context_);
  } else {
    CHECK(type->isFloatingPoint());
    val_type = (type->isFp32()) ? get_fp_type(32, cgen_state_->context_)
                                : get_fp_type(64, cgen_state_->context_);
  }
  auto lit_lv =
      cgen_state_->query_func_entry_ir_builder_.CreateLoad(val_type, lit_buf_start);
  lit_lv->setName(literal_name);
  return {lit_lv};
}

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstantsPlaceholders(
    const hdk::ir::Type* type,
    const bool use_dict_encoding,
    const int16_t lit_off,
    const std::vector<llvm::Value*>& literal_loads) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(codegen_traits_desc);

  std::string literal_name = "literal_" + std::to_string(lit_off);

  if (type->isString() && !use_dict_encoding) {
    CHECK_EQ(literal_loads.size(), 3u);

    llvm::Value* var_start = literal_loads[0];
    llvm::Value* var_start_address = literal_loads[1];
    llvm::Value* var_length = literal_loads[2];
    llvm::PointerType* placeholder0_type =
        cgen_traits.localPointerType(var_start->getType());
    auto* int_to_ptr0 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder0_type);
    auto placeholder0 = cgen_state_->ir_builder_.CreateLoad(
        var_start->getType(), int_to_ptr0, "__placeholder__" + literal_name + "_start");
    llvm::PointerType* placeholder1_type =
        cgen_traits.localPointerType(var_start_address->getType());
    auto* int_to_ptr1 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder1_type);
    auto placeholder1 = cgen_state_->ir_builder_.CreateLoad(
        placeholder1_type,
        int_to_ptr1,
        "__placeholder__" + literal_name + "_start_address");
    llvm::PointerType* placeholder2_type =
        cgen_traits.localPointerType(var_length->getType());
    auto* int_to_ptr2 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder2_type);
    auto placeholder2 = cgen_state_->ir_builder_.CreateLoad(
        placeholder2_type, int_to_ptr2, "__placeholder__" + literal_name + "_length");

    cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};
    cgen_state_->row_func_hoisted_literals_[placeholder1] = {lit_off, 1};
    cgen_state_->row_func_hoisted_literals_[placeholder2] = {lit_off, 2};

    return {placeholder0, placeholder1, placeholder2};
  }

  if (type->isArray() && !use_dict_encoding) {
    CHECK_EQ(literal_loads.size(), 2u);

    llvm::Value* var_start_address = literal_loads[0];
    llvm::Value* var_length = literal_loads[1];

    llvm::PointerType* placeholder0_type =
        cgen_traits.localPointerType(var_start_address->getType());
    auto* int_to_ptr0 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder0_type);
    auto placeholder0 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr0->getType()->getPointerElementType(),
        int_to_ptr0,
        "__placeholder__" + literal_name + "_start_address");
    llvm::PointerType* placeholder1_type =
        cgen_traits.localPointerType(var_length->getType());
    auto* int_to_ptr1 =
        cgen_state_->ir_builder_.CreateIntToPtr(cgen_state_->llInt(0), placeholder1_type);
    auto placeholder1 = cgen_state_->ir_builder_.CreateLoad(
        int_to_ptr1->getType()->getPointerElementType(),
        int_to_ptr1,
        "__placeholder__" + literal_name + "_length");

    cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};
    cgen_state_->row_func_hoisted_literals_[placeholder1] = {lit_off, 1};

    return {placeholder0, placeholder1};
  }

  CHECK_EQ(literal_loads.size(), 1u);
  llvm::Value* to_return_lv = literal_loads[0];

  auto* int_to_ptr = cgen_state_->ir_builder_.CreateIntToPtr(
      cgen_state_->llInt(0), cgen_traits.localPointerType(to_return_lv->getType()));
  auto placeholder0 = cgen_state_->ir_builder_.CreateLoad(
      to_return_lv->getType(), int_to_ptr, "__placeholder__" + literal_name);

  cgen_state_->row_func_hoisted_literals_[placeholder0] = {lit_off, 0};

  return {placeholder0};
}

std::vector<llvm::Value*> CodeGenerator::codegenHoistedConstants(
    const std::vector<const hdk::ir::Constant*>& constants,
    bool use_dict_encoding,
    int dict_id) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(!constants.empty());
  auto type = constants.front()->type();
  checked_int16_t checked_lit_off{0};
  int16_t lit_off{-1};
  try {
    for (size_t device_id = 0; device_id < constants.size(); ++device_id) {
      const auto constant = constants[device_id];
      auto crt_type = constant->type();
      CHECK(type->equal(crt_type));
      checked_lit_off =
          cgen_state_->getOrAddLiteral(constant, use_dict_encoding, dict_id, device_id);
      if (device_id) {
        CHECK_EQ(lit_off, checked_lit_off);
      } else {
        lit_off = (int16_t)checked_lit_off;
      }
    }
  } catch (const std::range_error& e) {
    // detect literal buffer overflow when trying to
    // assign literal buf offset which is not in a valid range
    // to checked_type variable
    throw TooManyLiterals();
  }
  std::vector<llvm::Value*> hoisted_literal_loads;
  auto entry = cgen_state_->query_func_literal_loads_.find(lit_off);

  if (entry == cgen_state_->query_func_literal_loads_.end()) {
    hoisted_literal_loads =
        codegenHoistedConstantsLoads(type, use_dict_encoding, dict_id, lit_off);
    cgen_state_->query_func_literal_loads_[lit_off] = hoisted_literal_loads;
  } else {
    hoisted_literal_loads = entry->second;
  }

  std::vector<llvm::Value*> literal_placeholders = codegenHoistedConstantsPlaceholders(
      type, use_dict_encoding, lit_off, hoisted_literal_loads);
  return literal_placeholders;
}
