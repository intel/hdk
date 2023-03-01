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

#include "CodeGenerator.h"
#include "Execute.h"
#include "StringDictionaryTranslationMgr.h"

llvm::Value* CodeGenerator::codegenCast(const hdk::ir::UOper* uoper,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(uoper->isCast());
  const auto& type = uoper->type();
  const auto operand = uoper->operand();
  const auto operand_as_const = dynamic_cast<const hdk::ir::Constant*>(operand);
  // For dictionary encoded constants, the cast holds the dictionary id
  // information as the compression parameter; handle this case separately.
  llvm::Value* operand_lv{nullptr};
  if (operand_as_const) {
    const auto operand_lvs = codegen(
        operand_as_const,
        type->isExtDictionary(),
        type->isExtDictionary() ? type->as<hdk::ir::ExtDictionaryType>()->dictId() : 0,
        co);
    if (operand_lvs.size() == 3) {
      operand_lv = cgen_state_->emitCall("string_pack", {operand_lvs[1], operand_lvs[2]});
    } else {
      operand_lv = operand_lvs.front();
    }
  } else {
    operand_lv = codegen(operand, true, co).front();
  }
  const auto& operand_type = operand->type();
  return codegenCast(
      operand_lv, operand_type, type, operand_as_const, uoper->isDictIntersection(), co);
}

namespace {

bool byte_array_cast(const hdk::ir::Type* operand_type, const hdk::ir::Type* type) {
  if (!operand_type->isArray() || !type->isArray()) {
    return false;
  }

  auto elem_type = type->as<hdk::ir::ArrayBaseType>()->elemType();
  return (elem_type->isInt8() && operand_type->size() > 0 &&
          operand_type->size() == type->size());
}

}  // namespace

llvm::Value* CodeGenerator::codegenCast(llvm::Value* operand_lv,
                                        const hdk::ir::Type* operand_type,
                                        const hdk::ir::Type* type,
                                        const bool operand_is_const,
                                        bool is_dict_intersection,
                                        const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (byte_array_cast(operand_type, type)) {
    auto* byte_array_type = get_int_array_type(8, type->size(), cgen_state_->context_);
    return cgen_state_->ir_builder_.CreatePointerCast(operand_lv,
                                                      byte_array_type->getPointerTo());
  }
  if (operand_lv->getType()->isIntegerTy()) {
    if (operand_type->isString() || operand_type->isExtDictionary()) {
      return codegenCastFromString(
          operand_lv, operand_type, type, operand_is_const, is_dict_intersection, co);
    }
    CHECK(operand_type->isInteger() || operand_type->isDecimal() ||
          operand_type->isDateTime() || operand_type->isBoolean());
    if (operand_type->isBoolean()) {
      // cast boolean to int8
      CHECK(operand_lv->getType()->isIntegerTy(1) ||
            operand_lv->getType()->isIntegerTy(8));
      if (operand_lv->getType()->isIntegerTy(1)) {
        operand_lv = cgen_state_->castToTypeIn(operand_lv, 8);
      }
      if (type->isBoolean()) {
        return operand_lv;
      }
    }
    if (operand_type->isInteger() && operand_lv->getType()->isIntegerTy(8) &&
        type->isBoolean()) {
      // cast int8 to boolean
      return codegenCastBetweenIntTypes(operand_lv, operand_type, type);
    }
    if (operand_type->isTimestamp() && type->isDate()) {
      // Maybe we should instead generate DateTruncExpr directly from RelAlgTranslator
      // for this pattern. However, DateTruncExpr is supposed to return a timestamp,
      // whereas this cast returns a date. The underlying type for both is still the same,
      // but it still doesn't look like a good idea to misuse DateTruncExpr.
      // Date will have default precision of day, but TIMESTAMP dimension would
      // matter but while converting date through seconds
      return codegenCastTimestampToDate(
          operand_lv,
          operand_type->as<hdk::ir::TimestampType>()->unit(),
          type->nullable());
    }
    if ((operand_type->isTimestamp() || operand_type->isDate()) && type->isTimestamp()) {
      const auto operand_unit = (operand_type->isTimestamp())
                                    ? operand_type->as<hdk::ir::TimestampType>()->unit()
                                    : hdk::ir::TimeUnit::kSecond;
      if (operand_unit != type->as<hdk::ir::TimestampType>()->unit()) {
        return codegenCastBetweenTimestamps(
            operand_lv, operand_type, type, type->nullable());
      }
    }
    if (type->isInteger() || type->isDecimal() || type->isDateTime()) {
      return codegenCastBetweenIntTypes(operand_lv, operand_type, type);
    } else {
      return codegenCastToFp(operand_lv, operand_type, type);
    }
  } else {
    return codegenCastFromFp(operand_lv, operand_type, type);
  }
  CHECK(false);
  return nullptr;
}

llvm::Value* CodeGenerator::codegenCastTimestampToDate(llvm::Value* ts_lv,
                                                       const hdk::ir::TimeUnit unit,
                                                       const bool nullable) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto& ctx = hdk::ir::Context::defaultCtx();
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (unit > hdk::ir::TimeUnit::kSecond) {
    if (nullable) {
      return cgen_state_->emitCall("DateTruncateHighPrecisionToDateNullable",
                                   {{ts_lv,
                                     cgen_state_->llInt(hdk::ir::unitsPerSecond(unit)),
                                     cgen_state_->inlineIntNull(ctx.int64())}});
    }
    return cgen_state_->emitCall(
        "DateTruncateHighPrecisionToDate",
        {{ts_lv, cgen_state_->llInt(hdk::ir::unitsPerSecond(unit))}});
  }
  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  if (nullable) {
    auto type = ctx.timestamp(unit, nullable);
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(
        cgen_state_, executor(), ts_lv, type, "cast_timestamp_nullcheck");
  }
  auto ret = cgen_state_->emitCall("datetrunc_day", {ts_lv});
  if (nullcheck_codegen) {
    ret = nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ret);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenCastBetweenTimestamps(
    llvm::Value* ts_lv,
    const hdk::ir::Type* operand_type,
    const hdk::ir::Type* target_type,
    const bool nullable) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto operand_unit = operand_type->isTimestamp()
                                ? operand_type->as<hdk::ir::TimestampType>()->unit()
                                : hdk::ir::TimeUnit::kSecond;
  const auto target_unit = target_type->isTimestamp()
                               ? target_type->as<hdk::ir::TimestampType>()->unit()
                               : hdk::ir::TimeUnit::kSecond;
  if (operand_unit == target_unit) {
    return ts_lv;
  }
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (operand_unit < target_unit) {
    const auto scale =
        hdk::ir::unitsPerSecond(target_unit) / hdk::ir::unitsPerSecond(operand_unit);
    codegenCastBetweenIntTypesOverflowChecks(ts_lv, operand_type, target_type, scale);
    return nullable
               ? cgen_state_->emitCall("mul_int64_t_nullable_lhs",
                                       {ts_lv,
                                        cgen_state_->llInt(static_cast<int64_t>(scale)),
                                        cgen_state_->inlineIntNull(operand_type)})
               : cgen_state_->ir_builder_.CreateMul(
                     ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)));
  }
  const auto scale =
      hdk::ir::unitsPerSecond(operand_unit) / hdk::ir::unitsPerSecond(target_unit);
  return nullable
             ? cgen_state_->emitCall("floor_div_nullable_lhs",
                                     {ts_lv,
                                      cgen_state_->llInt(static_cast<int64_t>(scale)),
                                      cgen_state_->inlineIntNull(operand_type)})
             : cgen_state_->ir_builder_.CreateSDiv(
                   ts_lv, cgen_state_->llInt(static_cast<int64_t>(scale)));
}

llvm::Value* CodeGenerator::codegenCastFromString(llvm::Value* operand_lv,
                                                  const hdk::ir::Type* operand_type,
                                                  const hdk::ir::Type* type,
                                                  const bool operand_is_const,
                                                  bool is_dict_intersection,
                                                  const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!type->isString() && !type->isExtDictionary()) {
    throw std::runtime_error("Cast from " + operand_type->toString() + " to " +
                             type->toString() + " not supported");
  }
  if (operand_type->isString() && type->isString()) {
    return operand_lv;
  }
  if (type->isExtDictionary() && operand_type->isExtDictionary()) {
    auto dict_id = type->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto operand_dict_id = operand_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    if (dict_id == operand_dict_id) {
      return operand_lv;
    }

    auto string_dictionary_translation_mgr =
        std::make_unique<StringDictionaryTranslationMgr>(
            operand_dict_id,
            dict_id,
            is_dict_intersection,
            co.device_type == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                      : Data_Namespace::CPU_LEVEL,
            executor()->deviceCount(co.device_type),
            executor(),
            executor()->getDataMgr());
    string_dictionary_translation_mgr->buildTranslationMap();
    string_dictionary_translation_mgr->createKernelBuffers();

    return cgen_state_
        ->moveStringDictionaryTranslationMgr(std::move(string_dictionary_translation_mgr))
        ->codegenCast(
            operand_lv, operand_type, true, co.codegen_traits_desc /* add_nullcheck */);
  }
  // dictionary encode non-constant
  if (operand_type->isString() && !operand_is_const) {
    if (config_.exec.watchdog.enable) {
      throw WatchdogException(
          "Cast from none-encoded string to dictionary-encoded would be slow");
    }
    CHECK(type->isExtDictionary());
    CHECK(operand_lv->getType()->isIntegerTy(64));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    return cgen_state_->emitExternalCall(
        "string_compress",
        get_int_type(32, cgen_state_->context_),
        {operand_lv,
         cgen_state_->llInt(int64_t(executor()->getStringDictionaryProxy(
             type->as<hdk::ir::ExtDictionaryType>()->dictId(),
             executor()->getRowSetMemoryOwner(),
             true)))});
  }
  CHECK(operand_lv->getType()->isIntegerTy(32));
  if (type->isString()) {
    if (config_.exec.watchdog.enable) {
      throw WatchdogException(
          "Cast from dictionary-encoded string to none-encoded would be slow");
    }
    CHECK(operand_type->isExtDictionary());
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
    auto operand_dict_id = operand_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    const int64_t string_dictionary_ptr =
        operand_dict_id == 0
            ? reinterpret_cast<int64_t>(
                  executor()->getRowSetMemoryOwner()->getLiteralStringDictProxy())
            : reinterpret_cast<int64_t>(executor()->getStringDictionaryProxy(
                  operand_dict_id, executor()->getRowSetMemoryOwner(), true));
    CHECK(string_dictionary_ptr);
    return cgen_state_->emitExternalCall(
        "string_decompress",
        get_int_type(64, cgen_state_->context_),
        {operand_lv, cgen_state_->llInt(string_dictionary_ptr)});
  }
  CHECK(operand_is_const);
  CHECK(type->isExtDictionary());
  return operand_lv;
}

llvm::Value* CodeGenerator::codegenCastBetweenIntTypes(llvm::Value* operand_lv,
                                                       const hdk::ir::Type* operand_type,
                                                       const hdk::ir::Type* type,
                                                       bool upscale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto target_scale = type->isDecimal() ? type->as<hdk::ir::DecimalType>()->scale() : 0;
  auto op_scale =
      operand_type->isDecimal() ? operand_type->as<hdk::ir::DecimalType>()->scale() : 0;
  if (type->isDecimal() && (!operand_type->isDecimal() || op_scale <= target_scale)) {
    if (upscale) {
      if (op_scale < target_scale) {  // scale only if needed
        auto scale = exp_to_scale(target_scale - op_scale);
        const auto scale_lv =
            llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);
        operand_lv = cgen_state_->ir_builder_.CreateSExt(
            operand_lv, get_int_type(64, cgen_state_->context_));

        codegenCastBetweenIntTypesOverflowChecks(operand_lv, operand_type, type, scale);

        if (operand_type->nullable()) {
          operand_lv = cgen_state_->emitCall(
              "scale_decimal_up",
              {operand_lv,
               scale_lv,
               cgen_state_->llInt(inline_int_null_value(operand_type)),
               cgen_state_->inlineIntNull(operand_type->ctx().int64())});
        } else {
          operand_lv = cgen_state_->ir_builder_.CreateMul(operand_lv, scale_lv);
        }
      }
    }
  } else if (operand_type->isDecimal()) {
    // rounded scale down
    auto scale = (int64_t)exp_to_scale(op_scale - target_scale);
    const auto scale_lv =
        llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), scale);

    const auto operand_width =
        static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();

    std::string method_name = "scale_decimal_down_nullable";
    if (!operand_type->nullable()) {
      method_name = "scale_decimal_down_not_nullable";
    }

    CHECK(operand_width == 64);
    operand_lv = cgen_state_->emitCall(
        method_name,
        {operand_lv, scale_lv, cgen_state_->llInt(inline_int_null_value(operand_type))});
  }
  if (type->isInteger() && operand_type->isInteger() &&
      operand_type->size() > type->size()) {
    codegenCastBetweenIntTypesOverflowChecks(operand_lv, operand_type, type, 1);
  }

  const auto operand_width =
      static_cast<llvm::IntegerType*>(operand_lv->getType())->getBitWidth();
  const auto target_width = get_bit_width(type);
  if (target_width == operand_width) {
    return operand_lv;
  }
  if (!operand_type->nullable()) {
    return cgen_state_->ir_builder_.CreateCast(
        target_width > operand_width ? llvm::Instruction::CastOps::SExt
                                     : llvm::Instruction::CastOps::Trunc,
        operand_lv,
        get_int_type(target_width, cgen_state_->context_));
  }
  return cgen_state_->emitCall("cast_" + numeric_type_name(operand_type) + "_to_" +
                                   numeric_type_name(type) + "_nullable",
                               {operand_lv,
                                cgen_state_->inlineIntNull(operand_type),
                                cgen_state_->inlineIntNull(type)});
}

void CodeGenerator::codegenCastBetweenIntTypesOverflowChecks(
    llvm::Value* operand_lv,
    const hdk::ir::Type* operand_type,
    const hdk::ir::Type* type,
    const int64_t scale) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) =
      cgen_state_->inlineIntMaxMin(type->canonicalSize(), true);

  cgen_state_->needs_error_check_ = true;
  auto cast_ok = llvm::BasicBlock::Create(
      cgen_state_->context_, "cast_ok", cgen_state_->current_func_);
  auto cast_fail = llvm::BasicBlock::Create(
      cgen_state_->context_, "cast_fail", cgen_state_->current_func_);
  auto operand_max = static_cast<llvm::ConstantInt*>(chosen_max)->getSExtValue() / scale;
  auto operand_min = static_cast<llvm::ConstantInt*>(chosen_min)->getSExtValue() / scale;
  const auto ti_llvm_type =
      get_int_type(8 * type->canonicalSize(), cgen_state_->context_);
  llvm::Value* operand_max_lv = llvm::ConstantInt::get(ti_llvm_type, operand_max);
  llvm::Value* operand_min_lv = llvm::ConstantInt::get(ti_llvm_type, operand_min);
  const bool is_narrowing = operand_type->canonicalSize() > type->canonicalSize();
  if (is_narrowing) {
    const auto operand_ti_llvm_type =
        get_int_type(8 * operand_type->canonicalSize(), cgen_state_->context_);
    operand_max_lv =
        cgen_state_->ir_builder_.CreateSExt(operand_max_lv, operand_ti_llvm_type);
    operand_min_lv =
        cgen_state_->ir_builder_.CreateSExt(operand_min_lv, operand_ti_llvm_type);
  }
  llvm::Value* over{nullptr};
  llvm::Value* under{nullptr};
  if (!operand_type->nullable()) {
    over = cgen_state_->ir_builder_.CreateICmpSGT(operand_lv, operand_max_lv);
    under = cgen_state_->ir_builder_.CreateICmpSLE(operand_lv, operand_min_lv);
  } else {
    const auto type_name =
        is_narrowing ? numeric_type_name(operand_type) : numeric_type_name(type);
    const auto null_operand_val = cgen_state_->llInt(inline_int_null_value(operand_type));
    const auto null_bool_val = cgen_state_->inlineIntNull(type->ctx().boolean());
    over = toBool(cgen_state_->emitCall(
        "gt_" + type_name + "_nullable_lhs",
        {operand_lv, operand_max_lv, null_operand_val, null_bool_val}));
    under = toBool(cgen_state_->emitCall(
        "le_" + type_name + "_nullable_lhs",
        {operand_lv, operand_min_lv, null_operand_val, null_bool_val}));
  }
  const auto detected = cgen_state_->ir_builder_.CreateOr(over, under, "overflow");
  cgen_state_->ir_builder_.CreateCondBr(detected, cast_fail, cast_ok);

  cgen_state_->ir_builder_.SetInsertPoint(cast_fail);
  cgen_state_->ir_builder_.CreateRet(
      cgen_state_->llInt(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

  cgen_state_->ir_builder_.SetInsertPoint(cast_ok);
}

llvm::Value* CodeGenerator::codegenCastToFp(llvm::Value* operand_lv,
                                            const hdk::ir::Type* operand_type,
                                            const hdk::ir::Type* type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!type->isFloatingPoint()) {
    throw std::runtime_error("Cast from " + operand_type->toString() + " to " +
                             type->toString() + " not supported");
  }
  auto scale =
      operand_type->isDecimal() ? operand_type->as<hdk::ir::DecimalType>()->scale() : 0;
  llvm::Value* result_lv;
  if (!operand_type->nullable()) {
    auto const fp_type = type->isFp32() ? llvm::Type::getFloatTy(cgen_state_->context_)
                                        : llvm::Type::getDoubleTy(cgen_state_->context_);
    result_lv = cgen_state_->ir_builder_.CreateSIToFP(operand_lv, fp_type);
    if (scale) {
      double const multiplier = shared::power10inv(scale);
      result_lv = cgen_state_->ir_builder_.CreateFMul(
          result_lv, llvm::ConstantFP::get(result_lv->getType(), multiplier));
    }
  } else {
    if (scale) {
      double const multiplier = shared::power10inv(scale);
      auto const fp_type = type->isFp32()
                               ? llvm::Type::getFloatTy(cgen_state_->context_)
                               : llvm::Type::getDoubleTy(cgen_state_->context_);
      result_lv =
          cgen_state_->emitCall("cast_" + numeric_type_name(operand_type) + "_to_" +
                                    numeric_type_name(type) + "_scaled_nullable",
                                {operand_lv,
                                 cgen_state_->inlineIntNull(operand_type),
                                 cgen_state_->inlineFpNull(type),
                                 llvm::ConstantFP::get(fp_type, multiplier)});
    } else {
      result_lv =
          cgen_state_->emitCall("cast_" + numeric_type_name(operand_type) + "_to_" +
                                    numeric_type_name(type) + "_nullable",
                                {operand_lv,
                                 cgen_state_->inlineIntNull(operand_type),
                                 cgen_state_->inlineFpNull(type)});
    }
  }
  CHECK(result_lv);
  return result_lv;
}

llvm::Value* CodeGenerator::codegenCastFromFp(llvm::Value* operand_lv,
                                              const hdk::ir::Type* operand_type,
                                              const hdk::ir::Type* type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (!operand_type->isFloatingPoint() || !type->isNumber() || type->isDecimal()) {
    throw std::runtime_error("Cast from " + operand_type->toString() + " to " +
                             type->toString() + " not supported");
  }
  if (operand_type->id() == type->id() && operand_type->size() == type->size()) {
    // Should not have been called when both dimensions are same.
    return operand_lv;
  }
  CHECK(operand_lv->getType()->isFloatTy() || operand_lv->getType()->isDoubleTy());
  if (!operand_type->nullable()) {
    if (type->isFp64()) {
      return cgen_state_->ir_builder_.CreateFPExt(
          operand_lv, llvm::Type::getDoubleTy(cgen_state_->context_));
    } else if (type->isFp32()) {
      return cgen_state_->ir_builder_.CreateFPTrunc(
          operand_lv, llvm::Type::getFloatTy(cgen_state_->context_));
    } else if (type->isInteger()) {
      // Round by adding/subtracting 0.5 before fptosi.
      auto* fp_type = operand_lv->getType()->isFloatTy()
                          ? llvm::Type::getFloatTy(cgen_state_->context_)
                          : llvm::Type::getDoubleTy(cgen_state_->context_);
      auto* zero = llvm::ConstantFP::get(fp_type, 0.0);
      auto* mhalf = llvm::ConstantFP::get(fp_type, -0.5);
      auto* phalf = llvm::ConstantFP::get(fp_type, 0.5);
      auto* is_negative = cgen_state_->ir_builder_.CreateFCmpOLT(operand_lv, zero);
      auto* offset = cgen_state_->ir_builder_.CreateSelect(is_negative, mhalf, phalf);
      operand_lv = cgen_state_->ir_builder_.CreateFAdd(operand_lv, offset);
      return cgen_state_->ir_builder_.CreateFPToSI(
          operand_lv, get_int_type(get_bit_width(type), cgen_state_->context_));
    } else {
      CHECK(false);
    }
  } else {
    const auto from_tname = numeric_type_name(operand_type);
    const auto to_tname = numeric_type_name(type);
    if (type->isFloatingPoint()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv,
                                    cgen_state_->inlineFpNull(operand_type),
                                    cgen_state_->inlineFpNull(type)});
    } else if (type->isInteger()) {
      return cgen_state_->emitCall("cast_" + from_tname + "_to_" + to_tname + "_nullable",
                                   {operand_lv,
                                    cgen_state_->inlineFpNull(operand_type),
                                    cgen_state_->inlineIntNull(type)});
    } else {
      CHECK(false);
    }
  }
  CHECK(false);
  return nullptr;
}
