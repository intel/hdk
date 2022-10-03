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

#include "DateTimeUtils.h"
#include "Execute.h"

#include "DateTruncateLookupTable.h"

using namespace DateTimeUtils;

namespace {

const char* get_extract_function_name(ExtractField field) {
  switch (field) {
    case kEPOCH:
      return "extract_epoch";
    case kDATEEPOCH:
      return "extract_dateepoch";
    case kQUARTERDAY:
      return "extract_quarterday";
    case kHOUR:
      return "extract_hour";
    case kMINUTE:
      return "extract_minute";
    case kSECOND:
      return "extract_second";
    case kMILLISECOND:
      return "extract_millisecond";
    case kMICROSECOND:
      return "extract_microsecond";
    case kNANOSECOND:
      return "extract_nanosecond";
    case kDOW:
      return "extract_dow";
    case kISODOW:
      return "extract_isodow";
    case kDAY:
      return "extract_day";
    case kWEEK:
      return "extract_week_monday";
    case kWEEK_SUNDAY:
      return "extract_week_sunday";
    case kWEEK_SATURDAY:
      return "extract_week_saturday";
    case kDOY:
      return "extract_day_of_year";
    case kMONTH:
      return "extract_month";
    case kQUARTER:
      return "extract_quarter";
    case kYEAR:
      return "extract_year";
  }
  UNREACHABLE();
  return "";
}

}  // namespace

llvm::Value* CodeGenerator::codegen(const hdk::ir::ExtractExpr* extract_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto from_expr = codegen(extract_expr->from(), true, co).front();
  const int32_t extract_field{extract_expr->field()};
  auto extract_expr_type = extract_expr->from()->type();
  if (extract_field == kEPOCH) {
    CHECK(extract_expr_type->isTimestamp() || extract_expr_type->isDate());
    if (from_expr->getType()->isIntegerTy(32)) {
      from_expr =
          cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                              from_expr,
                                              get_int_type(64, cgen_state_->context_));
      return from_expr;
    }
  }
  CHECK(from_expr->getType()->isIntegerTy(64));
  bool is_hpt = extract_expr_type->isTimestamp() &&
                extract_expr_type->as<hdk::ir::TimestampType>()->unit() >
                    hdk::ir::TimeUnit::kSecond;
  if (is_hpt) {
    from_expr = codegenExtractHighPrecisionTimestamps(
        from_expr, extract_expr_type, extract_expr->field());
  }
  if (!is_hpt && is_subsecond_extract_field(extract_expr->field())) {
    from_expr =
        !extract_expr_type->nullable()
            ? cgen_state_->ir_builder_.CreateMul(
                  from_expr,
                  cgen_state_->llInt(
                      get_extract_timestamp_precision_scale(extract_expr->field())))
            : cgen_state_->emitCall(
                  "mul_int64_t_nullable_lhs",
                  {from_expr,
                   cgen_state_->llInt(
                       get_extract_timestamp_precision_scale(extract_expr->field())),
                   cgen_state_->inlineIntNull(extract_expr_type)});
  }
  const auto extract_fname = get_extract_function_name(extract_expr->field());
  if (extract_expr_type->nullable()) {
    llvm::BasicBlock* extract_nullcheck_bb{nullptr};
    llvm::PHINode* extract_nullcheck_value{nullptr};
    {
      DiamondCodegen null_check(cgen_state_->ir_builder_.CreateICmp(
                                    llvm::ICmpInst::ICMP_EQ,
                                    from_expr,
                                    cgen_state_->inlineIntNull(extract_expr_type)),
                                executor(),
                                false,
                                "extract_nullcheck",
                                nullptr,
                                false);
      // generate a phi node depending on whether we got a null or not
      extract_nullcheck_bb = llvm::BasicBlock::Create(
          cgen_state_->context_, "extract_nullcheck_bb", cgen_state_->current_func_);

      // update the blocks created by diamond codegen to point to the newly created phi
      // block
      cgen_state_->ir_builder_.SetInsertPoint(null_check.cond_true_);
      cgen_state_->ir_builder_.CreateBr(extract_nullcheck_bb);
      cgen_state_->ir_builder_.SetInsertPoint(null_check.cond_false_);
      auto extract_call =
          cgen_state_->emitExternalCall(extract_fname,
                                        get_int_type(64, cgen_state_->context_),
                                        std::vector<llvm::Value*>{from_expr});
      cgen_state_->ir_builder_.CreateBr(extract_nullcheck_bb);

      cgen_state_->ir_builder_.SetInsertPoint(extract_nullcheck_bb);
      extract_nullcheck_value = cgen_state_->ir_builder_.CreatePHI(
          get_int_type(64, cgen_state_->context_), 2, "extract_value");
      extract_nullcheck_value->addIncoming(extract_call, null_check.cond_false_);
      extract_nullcheck_value->addIncoming(cgen_state_->inlineIntNull(extract_expr_type),
                                           null_check.cond_true_);
    }

    // diamond codegen will set the insert point in its destructor. override it to
    // continue using the extract nullcheck bb
    CHECK(extract_nullcheck_bb);
    cgen_state_->ir_builder_.SetInsertPoint(extract_nullcheck_bb);
    CHECK(extract_nullcheck_value);
    return extract_nullcheck_value;
  } else {
    return cgen_state_->emitExternalCall(extract_fname,
                                         get_int_type(64, cgen_state_->context_),
                                         std::vector<llvm::Value*>{from_expr});
  }
}

namespace {

int32_t unitToDimension(hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kSecond:
      return 0;
    case hdk::ir::TimeUnit::kMilli:
      return 3;
    case hdk::ir::TimeUnit::kMicro:
      return 6;
    case hdk::ir::TimeUnit::kNano:
      return 9;
    default:
      CHECK(false);
  }
  return 0;
}

}  // namespace

llvm::Value* CodeGenerator::codegen(const hdk::ir::DateAddExpr* dateadd_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto dateadd_expr_type = dateadd_expr->type();
  CHECK(dateadd_expr_type->isTimestamp() || dateadd_expr_type->isDate());
  auto dateadd_unit = dateadd_expr_type->isTimestamp()
                          ? dateadd_expr_type->as<hdk::ir::TimestampType>()->unit()
                          : hdk::ir::TimeUnit::kSecond;
  auto datetime = codegen(dateadd_expr->datetime(), true, co).front();
  CHECK(datetime->getType()->isIntegerTy(64));
  auto number = codegen(dateadd_expr->number(), true, co).front();

  auto datetime_type = dateadd_expr->datetime()->type();
  auto datetime_unit = datetime_type->isTimestamp()
                           ? datetime_type->as<hdk::ir::TimestampType>()->unit()
                           : hdk::ir::TimeUnit::kSecond;
  std::vector<llvm::Value*> dateadd_args{
      cgen_state_->llInt(static_cast<int32_t>(dateadd_expr->field())), number, datetime};
  std::string dateadd_fname{"DateAdd"};
  if (is_subsecond_dateadd_field(dateadd_expr->field()) ||
      dateadd_unit > hdk::ir::TimeUnit::kSecond) {
    dateadd_fname += "HighPrecision";
    dateadd_args.push_back(cgen_state_->llInt(unitToDimension(datetime_unit)));
  }
  if (datetime_type->nullable()) {
    dateadd_args.push_back(cgen_state_->inlineIntNull(datetime_type));
    dateadd_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(dateadd_fname,
                                       get_int_type(64, cgen_state_->context_),
                                       dateadd_args,
                                       {llvm::Attribute::NoUnwind,
                                        llvm::Attribute::ReadNone,
                                        llvm::Attribute::Speculatable});
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::DateDiffExpr* datediff_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto start = codegen(datediff_expr->start(), true, co).front();
  CHECK(start->getType()->isIntegerTy(64));
  auto end = codegen(datediff_expr->end(), true, co).front();
  CHECK(end->getType()->isIntegerTy(32) || end->getType()->isIntegerTy(64));
  auto start_type = datediff_expr->start()->type();
  auto end_type = datediff_expr->end()->type();
  std::vector<llvm::Value*> datediff_args{
      cgen_state_->llInt(static_cast<int32_t>(datediff_expr->field())), start, end};
  std::string datediff_fname{"DateDiff"};
  auto start_unit = start_type->isTimestamp()
                        ? start_type->as<hdk::ir::TimestampType>()->unit()
                        : hdk::ir::TimeUnit::kSecond;
  auto end_unit = end_type->isTimestamp() ? end_type->as<hdk::ir::TimestampType>()->unit()
                                          : hdk::ir::TimeUnit::kSecond;
  if (start_unit > hdk::ir::TimeUnit::kSecond || end_unit > hdk::ir::TimeUnit::kSecond) {
    datediff_fname += "HighPrecision";
    datediff_args.push_back(cgen_state_->llInt(unitToDimension(start_unit)));
    datediff_args.push_back(cgen_state_->llInt(unitToDimension(end_unit)));
  }
  auto ret_type = datediff_expr->type();
  if (start_type->nullable() || end_type->nullable()) {
    datediff_args.push_back(cgen_state_->inlineIntNull(ret_type));
    datediff_fname += "Nullable";
  }
  return cgen_state_->emitExternalCall(
      datediff_fname, get_int_type(64, cgen_state_->context_), datediff_args);
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::DateTruncExpr* datetrunc_expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto from_expr = codegen(datetrunc_expr->from(), true, co).front();
  auto datetrunc_expr_type = datetrunc_expr->from()->type();
  CHECK(from_expr->getType()->isIntegerTy(64));
  auto field = datetrunc_expr->field();
  if (datetrunc_expr_type->isTimestamp() &&
      datetrunc_expr_type->as<hdk::ir::TimestampType>()->unit() >
          hdk::ir::TimeUnit::kSecond) {
    return codegenDateTruncHighPrecisionTimestamps(from_expr, datetrunc_expr_type, field);
  }
  static_assert(
      (int)hdk::ir::DateTruncField::kSecond + 1 == (int)hdk::ir::DateTruncField::kMilli,
      "Please keep these consecutive.");
  static_assert(
      (int)hdk::ir::DateTruncField::kMilli + 1 == (int)hdk::ir::DateTruncField::kMicro,
      "Please keep these consecutive.");
  static_assert(
      (int)hdk::ir::DateTruncField::kMicro + 1 == (int)hdk::ir::DateTruncField::kNano,
      "Please keep these consecutive.");
  if (hdk::ir::DateTruncField::kSecond <= field &&
      field <= hdk::ir::DateTruncField::kNano) {
    return cgen_state_->ir_builder_.CreateCast(llvm::Instruction::CastOps::SExt,
                                               from_expr,
                                               get_int_type(64, cgen_state_->context_));
  }
  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  const bool is_nullable = datetrunc_expr_type->nullable();
  if (is_nullable) {
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(
        cgen_state_, executor(), from_expr, datetrunc_expr_type, "date_trunc_nullcheck");
  }
  char const* const fname = datetrunc_fname_lookup.at((size_t)field);
  auto ret = cgen_state_->emitExternalCall(
      fname, get_int_type(64, cgen_state_->context_), {from_expr});
  if (is_nullable) {
    ret = nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ret);
  }
  return ret;
}

llvm::Value* CodeGenerator::codegenExtractHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const hdk::ir::Type* type,
    const ExtractField& field) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(type->isTimestamp());
  auto unit = type->as<hdk::ir::TimestampType>()->unit();
  CHECK(unit > hdk::ir::TimeUnit::kSecond);
  CHECK(ts_lv->getType()->isIntegerTy(64));
  if (is_subsecond_extract_field(field)) {
    const auto result = get_extract_high_precision_adjusted_scale(field, unit);
    if (result.first == hdk::ir::OpType::kMul) {
      return !type->nullable()
                 ? cgen_state_->ir_builder_.CreateMul(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "mul_int64_t_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(type)});
    } else if (result.first == hdk::ir::OpType::kDiv) {
      return !type->nullable()
                 ? cgen_state_->ir_builder_.CreateSDiv(
                       ts_lv, cgen_state_->llInt(static_cast<int64_t>(result.second)))
                 : cgen_state_->emitCall(
                       "floor_div_nullable_lhs",
                       {ts_lv,
                        cgen_state_->llInt(static_cast<int64_t>(result.second)),
                        cgen_state_->inlineIntNull(type)});
    } else {
      return ts_lv;
    }
  }
  return !type->nullable()
             ? cgen_state_->ir_builder_.CreateSDiv(
                   ts_lv, cgen_state_->llInt(hdk::ir::unitsPerSecond(unit)))
             : cgen_state_->emitCall("floor_div_nullable_lhs",
                                     {ts_lv,
                                      cgen_state_->llInt(hdk::ir::unitsPerSecond(unit)),
                                      cgen_state_->inlineIntNull(type)});
}

llvm::Value* CodeGenerator::codegenDateTruncHighPrecisionTimestamps(
    llvm::Value* ts_lv,
    const hdk::ir::Type* type,
    const hdk::ir::DateTruncField& field) {
  // Only needed for i in { 0, 3, 6, 9 }.
  constexpr int64_t pow10[10]{1, 0, 0, 1000, 0, 0, 1000000, 0, 0, 1000000000};
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(type->isTimestamp());
  auto unit = type->as<hdk::ir::TimestampType>()->unit();
  CHECK(unit > hdk::ir::TimeUnit::kSecond);
  CHECK(ts_lv->getType()->isIntegerTy(64));
  bool const is_nullable = type->nullable();
  static_assert(
      (int)hdk::ir::DateTruncField::kSecond + 1 == (int)hdk::ir::DateTruncField::kMilli,
      "Please keep these consecutive.");
  static_assert(
      (int)hdk::ir::DateTruncField::kMilli + 1 == (int)hdk::ir::DateTruncField::kMicro,
      "Please keep these consecutive.");
  static_assert(
      (int)hdk::ir::DateTruncField::kMicro + 1 == (int)hdk::ir::DateTruncField::kNano,
      "Please keep these consecutive.");
  if (hdk::ir::DateTruncField::kSecond <= field &&
      field <= hdk::ir::DateTruncField::kNano) {
    unsigned const start_dim = unitToDimension(unit);  // 0, 3, 6, 9
    unsigned const trunc_dim =
        ((int)field - (int)hdk::ir::DateTruncField::kSecond) * 3;  // 0, 3, 6, 9
    if (start_dim <= trunc_dim) {
      return ts_lv;  // Truncating to an equal or higher precision has no effect.
    }
    int64_t const dscale = pow10[start_dim - trunc_dim];  // 1e3, 1e6, 1e9
    if (is_nullable) {
      ts_lv = cgen_state_->emitCall(
          "floor_div_nullable_lhs",
          {ts_lv, cgen_state_->llInt(dscale), cgen_state_->inlineIntNull(type)});
      return cgen_state_->emitCall(
          "mul_int64_t_nullable_lhs",
          {ts_lv, cgen_state_->llInt(dscale), cgen_state_->inlineIntNull(type)});
    } else {
      ts_lv = cgen_state_->ir_builder_.CreateSDiv(ts_lv, cgen_state_->llInt(dscale));
      return cgen_state_->ir_builder_.CreateMul(ts_lv, cgen_state_->llInt(dscale));
    }
  }
  int64_t const scale = hdk::ir::unitsPerSecond(unit);
  ts_lv = is_nullable
              ? cgen_state_->emitCall(
                    "floor_div_nullable_lhs",
                    {ts_lv, cgen_state_->llInt(scale), cgen_state_->inlineIntNull(type)})
              : cgen_state_->ir_builder_.CreateSDiv(ts_lv, cgen_state_->llInt(scale));

  std::unique_ptr<CodeGenerator::NullCheckCodegen> nullcheck_codegen;
  if (is_nullable) {
    nullcheck_codegen = std::make_unique<NullCheckCodegen>(
        cgen_state_, executor(), ts_lv, type, "date_trunc_hp_nullcheck");
  }
  char const* const fname = datetrunc_fname_lookup.at((size_t)field);
  ts_lv = cgen_state_->emitExternalCall(
      fname, get_int_type(64, cgen_state_->context_), {ts_lv});
  if (is_nullable) {
    ts_lv =
        nullcheck_codegen->finalize(ll_int(NULL_BIGINT, cgen_state_->context_), ts_lv);
  }

  return is_nullable
             ? cgen_state_->emitCall(
                   "mul_int64_t_nullable_lhs",
                   {ts_lv, cgen_state_->llInt(scale), cgen_state_->inlineIntNull(type)})
             : cgen_state_->ir_builder_.CreateMul(ts_lv, cgen_state_->llInt(scale));
}
