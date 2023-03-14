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
#include "ExtensionFunctions.hpp"
#include "ExtensionFunctionsBinding.h"
#include "ExtensionFunctionsWhitelist.h"

#include <tuple>

extern std::unique_ptr<llvm::Module> udf_gpu_module;
extern std::unique_ptr<llvm::Module> udf_cpu_module;

namespace {

llvm::StructType* get_buffer_struct_type(CgenState* cgen_state,
                                         const std::string& ext_func_name,
                                         size_t param_num,
                                         llvm::Type* elem_type,
                                         bool has_is_null) {
  CHECK(elem_type);
  CHECK(elem_type->isPointerTy());
  llvm::StructType* generated_struct_type =
      (has_is_null ? llvm::StructType::get(cgen_state->context_,
                                           {elem_type,
                                            llvm::Type::getInt64Ty(cgen_state->context_),
                                            llvm::Type::getInt8Ty(cgen_state->context_)},
                                           false)
                   : llvm::StructType::get(
                         cgen_state->context_,
                         {elem_type, llvm::Type::getInt64Ty(cgen_state->context_)},
                         false));
  llvm::Function* udf_func = cgen_state->module_->getFunction(ext_func_name);
  if (udf_func) {
    // Compare expected array struct type with type from the function
    // definition from the UDF module, but use the type from the
    // module
    llvm::FunctionType* udf_func_type = udf_func->getFunctionType();
    CHECK_LE(param_num, udf_func_type->getNumParams());
    llvm::Type* param_pointer_type = udf_func_type->getParamType(param_num);
    CHECK(param_pointer_type->isPointerTy());
    llvm::Type* param_type = param_pointer_type->getPointerElementType();
    CHECK(param_type->isStructTy());
    llvm::StructType* struct_type = llvm::cast<llvm::StructType>(param_type);
    CHECK_GE(struct_type->getStructNumElements(),
             generated_struct_type->getStructNumElements())
        << serialize_llvm_object(struct_type);

    const auto expected_elems = generated_struct_type->elements();
    const auto current_elems = struct_type->elements();
    for (size_t i = 0; i < expected_elems.size(); i++) {
      CHECK_EQ(expected_elems[i], current_elems[i])
          << "[" << ::toString(expected_elems[i]) << ", " << ::toString(current_elems[i])
          << "]";
    }

    if (struct_type->isLiteral()) {
      return struct_type;
    }

    llvm::StringRef struct_name = struct_type->getStructName();
    return struct_type->getTypeByName(cgen_state->context_, struct_name);
  }
  return generated_struct_type;
}

llvm::Type* ext_arg_type_to_llvm_type(const ExtArgumentType ext_arg_type,
                                      llvm::LLVMContext& ctx) {
  switch (ext_arg_type) {
    case ExtArgumentType::Bool:  // pass thru to Int8
    case ExtArgumentType::Int8:
      return get_int_type(8, ctx);
    case ExtArgumentType::Int16:
      return get_int_type(16, ctx);
    case ExtArgumentType::Int32:
      return get_int_type(32, ctx);
    case ExtArgumentType::Int64:
      return get_int_type(64, ctx);
    case ExtArgumentType::Float:
      return llvm::Type::getFloatTy(ctx);
    case ExtArgumentType::Double:
      return llvm::Type::getDoubleTy(ctx);
    case ExtArgumentType::ArrayInt64:
    case ExtArgumentType::ArrayInt32:
    case ExtArgumentType::ArrayInt16:
    case ExtArgumentType::ArrayBool:
    case ExtArgumentType::ArrayInt8:
    case ExtArgumentType::ArrayDouble:
    case ExtArgumentType::ArrayFloat:
    case ExtArgumentType::ColumnInt64:
    case ExtArgumentType::ColumnInt32:
    case ExtArgumentType::ColumnInt16:
    case ExtArgumentType::ColumnBool:
    case ExtArgumentType::ColumnInt8:
    case ExtArgumentType::ColumnDouble:
    case ExtArgumentType::ColumnFloat:
    case ExtArgumentType::TextEncodingNone:
    case ExtArgumentType::ColumnListInt64:
    case ExtArgumentType::ColumnListInt32:
    case ExtArgumentType::ColumnListInt16:
    case ExtArgumentType::ColumnListBool:
    case ExtArgumentType::ColumnListInt8:
    case ExtArgumentType::ColumnListDouble:
    case ExtArgumentType::ColumnListFloat:
      return llvm::Type::getVoidTy(ctx);
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

inline const hdk::ir::Type* get_type_from_llvm_type(const llvm::Type* ll_type) {
  CHECK(ll_type);
  auto& ctx = hdk::ir::Context::defaultCtx();
  const auto bits = ll_type->getPrimitiveSizeInBits();

  if (ll_type->isFloatingPointTy()) {
    switch (bits) {
      case 32:
        return ctx.fp32();
      case 64:
        return ctx.fp64();
      default:
        LOG(FATAL) << "Unsupported llvm floating point type: " << bits
                   << ", only 32 and 64 bit floating point is supported.";
    }
  } else {
    switch (bits) {
      case 1:
        return ctx.boolean();
      case 8:
        return ctx.int8();
      case 16:
        return ctx.int16();
      case 32:
        return ctx.int32();
      case 64:
        return ctx.int64();
      default:
        LOG(FATAL) << "Unrecognized llvm type for SQL type: "
                   << bits;  // TODO let's get the real name here
    }
  }
  UNREACHABLE();
  return nullptr;
}

inline llvm::Type* get_llvm_type_from_array_type(
    const hdk::ir::Type* type,
    llvm::LLVMContext& ctx,
    const compiler::CodegenTraitsDescriptor& codegen_traits_desc) {
  CHECK(type->isBuffer());
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(codegen_traits_desc);
  if (type->isText()) {
    return cgen_traits.localPointerType(get_int_type(8, ctx));
  }

  const auto& elem_type = type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType()
                          : type->isColumn()
                              ? type->as<hdk::ir::ColumnType>()->columnType()
                              : type->as<hdk::ir::ColumnListType>()->columnType();
  if (elem_type->isFloatingPoint()) {
    switch (elem_type->size()) {
      case 4:
        return cgen_traits.localPointerType(get_fp_type(32, ctx));
      case 8:
        return cgen_traits.localPointerType(get_fp_type(64, ctx));
    }
  }

  if (elem_type->isBoolean()) {
    return cgen_traits.localPointerType(get_int_type(8, ctx));
  }

  CHECK(elem_type->isInteger());
  switch (elem_type->size()) {
    case 1:
      return cgen_traits.localPointerType(get_int_type(8, ctx));
    case 2:
      return cgen_traits.localPointerType(get_int_type(16, ctx));
    case 4:
      return cgen_traits.localPointerType(get_int_type(32, ctx));
    case 8:
      return cgen_traits.localPointerType(get_int_type(64, ctx));
  }

  UNREACHABLE();
  return nullptr;
}

bool ext_func_call_requires_nullcheck(const hdk::ir::FunctionOper* function_oper) {
  const auto& func_type = function_oper->type();
  for (size_t i = 0; i < function_oper->arity(); ++i) {
    const auto arg = function_oper->arg(i);
    const auto& arg_type = arg->type();
    if ((func_type->isArray() && arg_type->isArray()) ||
        (func_type->isText() && arg_type->isText())) {
      // If the function returns an array and any of the arguments are arrays, allow NULL
      // scalars.
      // TODO: Make this a property of the FunctionOper following `RETURN NULL ON NULL`
      // semantics.
      return false;
    } else if (arg_type->nullable() && !arg_type->isBuffer()) {
      return true;
    } else {
      continue;
    }
  }
  return false;
}

}  // namespace

extern "C" RUNTIME_EXPORT void register_buffer_with_executor_rsm(int64_t exec,
                                                                 int8_t* buffer) {
  Executor* exec_ptr = reinterpret_cast<Executor*>(exec);
  if (buffer != nullptr) {
    exec_ptr->getRowSetMemoryOwner()->addVarlenBuffer(buffer);
  }
}

llvm::Value* CodeGenerator::codegenFunctionOper(
    const hdk::ir::FunctionOper* function_oper,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  ExtensionFunction ext_func_sig = [=]() {
    if (co.device_type == ExecutorDeviceType::GPU) {
      try {
        return bind_function(function_oper, /* is_gpu= */ true);
      } catch (ExtensionFunctionBindingError& e) {
        LOG(WARNING) << "codegenFunctionOper[GPU]: " << e.what() << " Redirecting "
                     << function_oper->name() << " to run on CPU.";
        throw QueryMustRunOnCpu();
      }
    } else {
      try {
        return bind_function(function_oper, /* is_gpu= */ false);
      } catch (ExtensionFunctionBindingError& e) {
        LOG(WARNING) << "codegenFunctionOper[CPU]: " << e.what();
        throw;
      }
    }
  }();

  const auto& ret_type = function_oper->type();
  CHECK(ret_type->isInteger() || ret_type->isFloatingPoint() || ret_type->isBoolean() ||
        ret_type->isBuffer());
  if (ret_type->isBuffer() && co.device_type == ExecutorDeviceType::GPU) {
    // TODO: This is not necessary for runtime UDFs because RBC does
    // not generated GPU LLVM IR when the UDF is using Buffer objects.
    // However, we cannot remove it until C++ UDFs can be defined for
    // different devices independently.
    throw QueryMustRunOnCpu();
  }

  auto ret_ty = ext_arg_type_to_llvm_type(ext_func_sig.getRet(), cgen_state_->context_);
  const auto current_bb = cgen_state_->ir_builder_.GetInsertBlock();
  for (auto it : cgen_state_->ext_call_cache_) {
    if (*it.foper == *function_oper) {
      auto inst = llvm::dyn_cast<llvm::Instruction>(it.lv);
      if (inst && inst->getParent() == current_bb) {
        return it.lv;
      }
    }
  }
  std::vector<llvm::Value*> orig_arg_lvs;
  std::vector<size_t> orig_arg_lvs_index;
  std::unordered_map<llvm::Value*, llvm::Value*> const_arr_size;

  for (size_t i = 0; i < function_oper->arity(); ++i) {
    orig_arg_lvs_index.push_back(orig_arg_lvs.size());
    const auto arg = function_oper->arg(i);
    const auto arg_cast = dynamic_cast<const hdk::ir::UOper*>(arg);
    const auto arg0 = (arg_cast && arg_cast->isCast()) ? arg_cast->operand() : arg;
    const auto array_expr_arg = dynamic_cast<const hdk::ir::ArrayExpr*>(arg0);
    auto is_local_alloc =
        ret_type->isBuffer() || (array_expr_arg && array_expr_arg->isLocalAlloc());
    const auto& arg_type = arg->type();
    const auto arg_lvs = codegen(arg, true, co);
    if (arg_type->isText()) {
      CHECK_EQ(size_t(3), arg_lvs.size());
      /* arg_lvs contains:
         c = string_decode(&col_buf0, pos)
         ptr = extract_str_ptr(c)
         sz = extract_str_len(c)
      */
      for (size_t j = 0; j < arg_lvs.size(); j++) {
        orig_arg_lvs.push_back(arg_lvs[j]);
      }
    } else {
      if (arg_lvs.size() > 1) {
        CHECK(arg_type->isArray());
        CHECK_EQ(size_t(2), arg_lvs.size());
        const_arr_size[arg_lvs.front()] = arg_lvs.back();
      } else {
        CHECK_EQ(size_t(1), arg_lvs.size());
        /* arg_lvs contains:
             &col_buf1
         */
        if (is_local_alloc && arg_type->size() > 0) {
          const_arr_size[arg_lvs.front()] = cgen_state_->llInt(arg_type->size());
        }
      }
      orig_arg_lvs.push_back(arg_lvs.front());
    }
  }
  // The extension function implementations don't handle NULL, they work under
  // the assumption that the inputs are validated before calling them. Generate
  // code to do the check at the call site: if any argument is NULL, return NULL
  // without calling the function at all.
  const auto [bbs, null_buffer_ptr] = beginArgsNullcheck(function_oper, orig_arg_lvs);
  CHECK_GE(orig_arg_lvs.size(), function_oper->arity());
  // Arguments must be converted to the types the extension function can handle.
  auto args = codegenFunctionOperCastArgs(
      function_oper, &ext_func_sig, orig_arg_lvs, orig_arg_lvs_index, const_arr_size, co);

  llvm::Value* buffer_ret{nullptr};
  if (ret_type->isBuffer()) {
    // codegen buffer return as first arg
    CHECK(ret_type->isArray() || ret_type->isText());
    ret_ty = llvm::Type::getVoidTy(cgen_state_->context_);
    const auto struct_ty = get_buffer_struct_type(
        cgen_state_,
        function_oper->name(),
        0,
        get_llvm_type_from_array_type(
            ret_type, cgen_state_->context_, codegen_traits_desc),
        /* has_is_null = */ ret_type->isArray() || ret_type->isText());
    buffer_ret = cgen_state_->ir_builder_.CreateAlloca(struct_ty);
    if (buffer_ret->getType()->getPointerAddressSpace() !=
        codegen_traits_desc.local_addr_space_) {
      buffer_ret = cgen_state_->ir_builder_.CreateAddrSpaceCast(
          buffer_ret,
          llvm::PointerType::get(buffer_ret->getType()->getPointerElementType(),
                                 codegen_traits_desc.local_addr_space_),
          "buffer.ret.cast");
    }
    args.insert(args.begin(), buffer_ret);
  }

  const auto ext_call = cgen_state_->emitExternalCall(
      ext_func_sig.getName(), ret_ty, args, {}, ret_type->isBuffer());
  auto ext_call_nullcheck = endArgsNullcheck(bbs,
                                             ret_type->isBuffer() ? buffer_ret : ext_call,
                                             null_buffer_ptr,
                                             function_oper,
                                             co);

  // Cast the return of the extension function to match the FunctionOper
  if (!(ret_type->isBuffer())) {
    const auto extension_ret_type = get_type_from_llvm_type(ret_ty);
    if (bbs.args_null_bb &&
        ((extension_ret_type->id() != function_oper->type()->id()) ||
         (extension_ret_type->size() != function_oper->type()->size())) &&
        // Skip i1-->i8 casts for ST_ functions.
        // function_oper ret type is i1, extension ret type is 'upgraded' to i8
        // during type deserialization to 'handle' NULL returns, hence i1-->i8.
        // ST_ functions can't return NULLs, we just need to check arg nullness
        // and if any args are NULL then ST_ function is not called
        function_oper->name().substr(0, 3) != std::string("ST_")) {
      ext_call_nullcheck = codegenCast(ext_call_nullcheck,
                                       extension_ret_type,
                                       function_oper->type(),
                                       false,
                                       false,
                                       co);
    }
  }

  cgen_state_->ext_call_cache_.push_back({function_oper, ext_call_nullcheck});
  return ext_call_nullcheck;
}

// Start the control flow needed for a call site check of NULL arguments.
std::tuple<CodeGenerator::ArgNullcheckBBs, llvm::Value*>
CodeGenerator::beginArgsNullcheck(const hdk::ir::FunctionOper* function_oper,
                                  const std::vector<llvm::Value*>& orig_arg_lvs) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  llvm::BasicBlock* args_null_bb{nullptr};
  llvm::BasicBlock* args_notnull_bb{nullptr};
  llvm::BasicBlock* orig_bb = cgen_state_->ir_builder_.GetInsertBlock();
  llvm::Value* null_array_alloca{nullptr};
  // Only generate the check if required (at least one argument must be nullable).
  if (ext_func_call_requires_nullcheck(function_oper)) {
    const auto func_type = function_oper->type();
    if (func_type->isBuffer()) {
      const auto arr_struct_ty = get_buffer_struct_type(
          cgen_state_,
          function_oper->name(),
          0,
          get_llvm_type_from_array_type(
              func_type, cgen_state_->context_, codegen_traits_desc),
          func_type->isArray() || func_type->isText());
      null_array_alloca = cgen_state_->ir_builder_.CreateAlloca(arr_struct_ty);
      if (null_array_alloca->getType()->getPointerAddressSpace() !=
          codegen_traits_desc.local_addr_space_) {
        null_array_alloca = cgen_state_->ir_builder_.CreateAddrSpaceCast(
            null_array_alloca,
            llvm::PointerType::get(null_array_alloca->getType()->getPointerElementType(),
                                   codegen_traits_desc.local_addr_space_),
            "null.array.alloca.cast");
      }
    }
    const auto args_notnull_lv = cgen_state_->ir_builder_.CreateNot(
        codegenFunctionOperNullArg(function_oper, orig_arg_lvs));
    args_notnull_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "args_notnull", cgen_state_->current_func_);
    args_null_bb = llvm::BasicBlock::Create(
        cgen_state_->context_, "args_null", cgen_state_->current_func_);
    cgen_state_->ir_builder_.CreateCondBr(args_notnull_lv, args_notnull_bb, args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(args_notnull_bb);
  }
  return std::make_tuple(
      CodeGenerator::ArgNullcheckBBs{args_null_bb, args_notnull_bb, orig_bb},
      null_array_alloca);
}

// Wrap up the control flow needed for NULL argument handling.
llvm::Value* CodeGenerator::endArgsNullcheck(const ArgNullcheckBBs& bbs,
                                             llvm::Value* fn_ret_lv,
                                             llvm::Value* null_array_ptr,
                                             const hdk::ir::FunctionOper* function_oper,
                                             const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  compiler::CodegenTraits cgen_traits = compiler::CodegenTraits::get(codegen_traits_desc);
  if (bbs.args_null_bb) {
    CHECK(bbs.args_notnull_bb);
    cgen_state_->ir_builder_.CreateBr(bbs.args_null_bb);
    cgen_state_->ir_builder_.SetInsertPoint(bbs.args_null_bb);

    llvm::PHINode* ext_call_phi{nullptr};
    llvm::Value* null_lv{nullptr};
    const auto func_type = function_oper->type();
    if (!func_type->isBuffer()) {
      // The pre-cast SQL equivalent of the type returned by the extension function.
      const auto extension_ret_type = get_type_from_llvm_type(fn_ret_lv->getType());

      ext_call_phi = cgen_state_->ir_builder_.CreatePHI(
          extension_ret_type->isFloatingPoint()
              ? get_fp_type(extension_ret_type->size() * 8, cgen_state_->context_)
              : get_int_type(extension_ret_type->size() * 8, cgen_state_->context_),
          2);

      null_lv =
          extension_ret_type->isFloatingPoint()
              ? static_cast<llvm::Value*>(cgen_state_->inlineFpNull(extension_ret_type))
              : static_cast<llvm::Value*>(cgen_state_->inlineIntNull(extension_ret_type));
    } else {
      const auto arr_struct_ty = get_buffer_struct_type(
          cgen_state_,
          function_oper->name(),
          0,
          get_llvm_type_from_array_type(
              func_type, cgen_state_->context_, codegen_traits_desc),
          true);
      ext_call_phi = cgen_state_->ir_builder_.CreatePHI(
          cgen_traits.localPointerType(arr_struct_ty), 2);

      CHECK(null_array_ptr);
      const auto arr_null_bool =
          cgen_state_->ir_builder_.CreateStructGEP(arr_struct_ty, null_array_ptr, 2);
      cgen_state_->ir_builder_.CreateStore(
          llvm::ConstantInt::get(get_int_type(8, cgen_state_->context_), 1),
          arr_null_bool);

      const auto arr_null_size =
          cgen_state_->ir_builder_.CreateStructGEP(arr_struct_ty, null_array_ptr, 1);
      cgen_state_->ir_builder_.CreateStore(
          llvm::ConstantInt::get(get_int_type(64, cgen_state_->context_), 0),
          arr_null_size);
    }
    ext_call_phi->addIncoming(fn_ret_lv, bbs.args_notnull_bb);
    ext_call_phi->addIncoming(func_type->isBuffer() ? null_array_ptr : null_lv,
                              bbs.orig_bb);

    return ext_call_phi;
  }
  return fn_ret_lv;
}

namespace {

bool call_requires_custom_type_handling(const hdk::ir::FunctionOper* function_oper) {
  const auto& ret_type = function_oper->type();
  if (!ret_type->isInteger() && !ret_type->isFloatingPoint()) {
    return true;
  }
  for (size_t i = 0; i < function_oper->arity(); ++i) {
    const auto arg = function_oper->arg(i);
    const auto& arg_type = arg->type();
    if (!arg_type->isInteger() && !arg_type->isFloatingPoint()) {
      return true;
    }
  }
  return false;
}

}  // namespace

llvm::Value* CodeGenerator::codegenFunctionOperWithCustomTypeHandling(
    const hdk::ir::FunctionOperWithCustomTypeHandling* function_oper,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (call_requires_custom_type_handling(function_oper)) {
    // Some functions need the return type to be the same as the input type.
    if (function_oper->name() == "FLOOR" || function_oper->name() == "CEIL") {
      CHECK_EQ(size_t(1), function_oper->arity());
      const auto arg = function_oper->arg(0);
      auto arg_type = arg->type();
      CHECK(arg_type->isDecimal());
      auto arg_scale = arg_type->as<hdk::ir::DecimalType>()->scale();
      const auto arg_lvs = codegen(arg, true, co);
      CHECK_EQ(size_t(1), arg_lvs.size());
      const auto arg_lv = arg_lvs.front();
      CHECK(arg_lv->getType()->isIntegerTy(64));
      CodeGenerator::ArgNullcheckBBs bbs;
      std::tie(bbs, std::ignore) = beginArgsNullcheck(function_oper, {arg_lvs});
      const std::string func_name =
          (function_oper->name() == "FLOOR") ? "decimal_floor" : "decimal_ceil";
      const auto covar_result_lv = cgen_state_->emitCall(
          func_name, {arg_lv, cgen_state_->llInt(exp_to_scale(arg_scale))});
      auto ret_type = function_oper->type();
      CHECK(ret_type->isDecimal());
      CHECK_EQ(0, ret_type->as<hdk::ir::DecimalType>()->scale());
      const auto result_lv = cgen_state_->ir_builder_.CreateSDiv(
          covar_result_lv, cgen_state_->llInt(exp_to_scale(arg_scale)));
      return endArgsNullcheck(bbs, result_lv, nullptr, function_oper, co);
    } else if (function_oper->name() == "ROUND" &&
               function_oper->arg(0)->type()->isDecimal()) {
      CHECK_EQ(size_t(2), function_oper->arity());

      const auto arg0 = function_oper->arg(0);
      auto arg0_type = arg0->type();
      auto arg0_scale =
          arg0_type->isDecimal() ? arg0_type->as<hdk::ir::DecimalType>()->scale() : 0;
      const auto arg0_lvs = codegen(arg0, true, co);
      CHECK_EQ(size_t(1), arg0_lvs.size());
      const auto arg0_lv = arg0_lvs.front();
      CHECK(arg0_lv->getType()->isIntegerTy(64));

      const auto arg1 = function_oper->arg(1);
      auto arg1_type = arg1->type();
      CHECK(arg1_type->isInteger());
      const auto arg1_lvs = codegen(arg1, true, co);
      auto arg1_lv = arg1_lvs.front();
      if (!arg1_type->isInt32()) {
        arg1_lv = codegenCast(
            arg1_lv, arg1_type, arg1_type->ctx().int32(false), false, false, co);
      }

      CodeGenerator::ArgNullcheckBBs bbs0;
      std::tie(bbs0, std::ignore) =
          beginArgsNullcheck(function_oper, {arg0_lv, arg1_lvs.front()});

      const std::string func_name = "Round__4";
      auto ret_type = function_oper->type();
      CHECK(ret_type->isDecimal());
      const auto result_lv = cgen_state_->emitExternalCall(
          func_name,
          get_int_type(64, cgen_state_->context_),
          {arg0_lv, arg1_lv, cgen_state_->llInt(arg0_scale)});

      return endArgsNullcheck(bbs0, result_lv, nullptr, function_oper, co);
    }
    throw std::runtime_error("Type combination not supported for function " +
                             function_oper->name());
  }
  return codegenFunctionOper(function_oper, co);
}

// Generates code which returns true iff at least one of the arguments is NULL.
llvm::Value* CodeGenerator::codegenFunctionOperNullArg(
    const hdk::ir::FunctionOper* function_oper,
    const std::vector<llvm::Value*>& orig_arg_lvs) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  llvm::Value* one_arg_null =
      llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(cgen_state_->context_), false);
  for (size_t i = 0; i < function_oper->arity(); ++i) {
    const auto arg = function_oper->arg(i);
    auto arg_type = arg->type();
    if (!arg_type->nullable()) {
      continue;
    }
    if (arg_type->isBuffer()) {
      auto fname = "array_is_null";
      auto is_null_lv = cgen_state_->emitExternalCall(
          fname, get_int_type(1, cgen_state_->context_), {orig_arg_lvs[i], posArg(arg)});
      one_arg_null = cgen_state_->ir_builder_.CreateOr(one_arg_null, is_null_lv);
      continue;
    }
    CHECK(arg_type->isNumber() || arg_type->isBoolean());
    one_arg_null = cgen_state_->ir_builder_.CreateOr(
        one_arg_null, codegenIsNullNumber(orig_arg_lvs[i], arg_type));
  }
  return one_arg_null;
}

void CodeGenerator::codegenBufferArgs(const std::string& ext_func_name,
                                      size_t param_num,
                                      llvm::Value* buffer_buf,
                                      llvm::Value* buffer_size,
                                      llvm::Value* buffer_null,
                                      std::vector<llvm::Value*>& output_args) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(buffer_buf);
  CHECK(buffer_size);

  auto buffer_abstraction = get_buffer_struct_type(
      cgen_state_, ext_func_name, param_num, buffer_buf->getType(), !!(buffer_null));
  llvm::Value* alloc_mem = cgen_state_->ir_builder_.CreateAlloca(buffer_abstraction);
  if (alloc_mem->getType()->getPointerAddressSpace() !=
      codegen_traits_desc.local_addr_space_) {
    alloc_mem = cgen_state_->ir_builder_.CreateAddrSpaceCast(
        alloc_mem,
        llvm::PointerType::get(alloc_mem->getType()->getPointerElementType(),
                               codegen_traits_desc.local_addr_space_),
        "alloc.mem.cast");
  }
  auto buffer_buf_ptr =
      cgen_state_->ir_builder_.CreateStructGEP(buffer_abstraction, alloc_mem, 0);
  cgen_state_->ir_builder_.CreateStore(buffer_buf, buffer_buf_ptr);

  auto buffer_size_ptr =
      cgen_state_->ir_builder_.CreateStructGEP(buffer_abstraction, alloc_mem, 1);
  cgen_state_->ir_builder_.CreateStore(buffer_size, buffer_size_ptr);

  if (buffer_null) {
    auto bool_extended_type = llvm::Type::getInt8Ty(cgen_state_->context_);
    auto buffer_null_extended =
        cgen_state_->ir_builder_.CreateZExt(buffer_null, bool_extended_type);
    auto buffer_is_null_ptr =
        cgen_state_->ir_builder_.CreateStructGEP(buffer_abstraction, alloc_mem, 2);
    cgen_state_->ir_builder_.CreateStore(buffer_null_extended, buffer_is_null_ptr);
  }
  output_args.push_back(alloc_mem);
}

namespace {

inline bool is_ext_arg_type_pointer(const ExtArgumentType ext_arg_type) {
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
    case ExtArgumentType::PInt16:
    case ExtArgumentType::PInt32:
    case ExtArgumentType::PInt64:
    case ExtArgumentType::PFloat:
    case ExtArgumentType::PDouble:
    case ExtArgumentType::PBool:
      return true;

    default:
      return false;
  }
}

}  // namespace

// Generate CAST operations for arguments in `orig_arg_lvs` to the types required by
// `ext_func_sig`.
std::vector<llvm::Value*> CodeGenerator::codegenFunctionOperCastArgs(
    const hdk::ir::FunctionOper* function_oper,
    const ExtensionFunction* ext_func_sig,
    const std::vector<llvm::Value*>& orig_arg_lvs,
    const std::vector<size_t>& orig_arg_lvs_index,
    const std::unordered_map<llvm::Value*, llvm::Value*>& const_arr_size,
    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  CHECK(ext_func_sig);
  const auto& ext_func_args = ext_func_sig->getArgs();
  CHECK_LE(function_oper->arity(), ext_func_args.size());
  auto func_type = function_oper->type();
  std::vector<llvm::Value*> args;
  /*
    i: argument in RA for the function operand
    j: extra offset in ext_func_args
    k: origin_arg_lvs counter, equal to orig_arg_lvs_index[i]
    ij: ext_func_args counter, equal to i + j
    dj: offset when UDF implementation first argument corresponds to return value
   */
  for (size_t i = 0, j = 0, dj = (func_type->isBuffer() ? 1 : 0);
       i < function_oper->arity();
       ++i) {
    size_t k = orig_arg_lvs_index[i];
    size_t ij = i + j;
    const auto arg = function_oper->arg(i);
    const auto ext_func_arg = ext_func_args[ij];
    auto arg_type = arg->type();
    llvm::Value* arg_lv{nullptr};
    if (arg_type->isText()) {
      CHECK(ext_func_arg == ExtArgumentType::TextEncodingNone)
          << ::toString(ext_func_arg);
      const auto ptr_lv = orig_arg_lvs[k + 1];
      const auto len_lv = orig_arg_lvs[k + 2];
      auto& builder = cgen_state_->ir_builder_;
      auto string_buf_arg = builder.CreatePointerCast(
          ptr_lv,
          llvm::Type::getInt8PtrTy(cgen_state_->context_,
                                   ptr_lv->getType()->getPointerAddressSpace()));
      auto string_size_arg =
          builder.CreateZExt(len_lv, get_int_type(64, cgen_state_->context_));
      codegenBufferArgs(ext_func_sig->getName(),
                        ij + dj,
                        string_buf_arg,
                        string_size_arg,
                        nullptr,
                        args);
    } else if (arg_type->isArray()) {
      bool const_arr = (const_arr_size.count(orig_arg_lvs[k]) > 0);
      auto elem_type = arg_type->as<hdk::ir::ArrayBaseType>()->elemType();
      // TODO: switch to fast fixlen variants
      const auto ptr_lv =
          (const_arr) ? orig_arg_lvs[k]
                      : cgen_state_->emitExternalCall(
                            "array_buff",
                            llvm::Type::getInt8PtrTy(
                                cgen_state_->context_,
                                orig_arg_lvs[k]->getType()->getPointerAddressSpace()),
                            {orig_arg_lvs[k], posArg(arg)});
      const auto len_lv =
          (const_arr) ? const_arr_size.at(orig_arg_lvs[k])
                      : cgen_state_->emitExternalCall(
                            "array_size",
                            get_int_type(32, cgen_state_->context_),
                            {orig_arg_lvs[k],
                             posArg(arg),
                             cgen_state_->llInt(log2_bytes(elem_type->canonicalSize()))});

      if (is_ext_arg_type_pointer(ext_func_arg)) {
        args.push_back(castArrayPointer(ptr_lv, elem_type));
        args.push_back(cgen_state_->ir_builder_.CreateZExt(
            len_lv, get_int_type(64, cgen_state_->context_)));
        j++;
      } else if (is_ext_arg_type_array(ext_func_arg)) {
        auto array_buf_arg = castArrayPointer(ptr_lv, elem_type);
        auto& builder = cgen_state_->ir_builder_;
        auto array_size_arg =
            builder.CreateZExt(len_lv, get_int_type(64, cgen_state_->context_));
        auto array_null_arg =
            cgen_state_->emitExternalCall("array_is_null",
                                          get_int_type(1, cgen_state_->context_),
                                          {orig_arg_lvs[k], posArg(arg)});
        codegenBufferArgs(ext_func_sig->getName(),
                          ij + dj,
                          array_buf_arg,
                          array_size_arg,
                          array_null_arg,
                          args);
      } else {
        UNREACHABLE();
      }
    } else {
      const auto arg_target_type = ext_arg_type_to_type(arg_type->ctx(), ext_func_arg);
      if ((arg_type->id() != arg_target_type->id()) ||
          (arg_type->size() != arg_target_type->size())) {
        arg_lv =
            codegenCast(orig_arg_lvs[k], arg_type, arg_target_type, false, false, co);
      } else {
        arg_lv = orig_arg_lvs[k];
      }
      CHECK_EQ(arg_lv->getType(),
               ext_arg_type_to_llvm_type(ext_func_arg, cgen_state_->context_));
      args.push_back(arg_lv);
    }
  }
  return args;
}

llvm::Value* CodeGenerator::castArrayPointer(llvm::Value* ptr,
                                             const hdk::ir::Type* elem_type) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const unsigned ptr_address_space = ptr->getType()->getPointerAddressSpace();
  if (elem_type->isFp32()) {
    return cgen_state_->ir_builder_.CreatePointerCast(
        ptr, llvm::Type::getFloatPtrTy(cgen_state_->context_, ptr_address_space));
  }
  if (elem_type->isFp64()) {
    return cgen_state_->ir_builder_.CreatePointerCast(
        ptr, llvm::Type::getDoublePtrTy(cgen_state_->context_, ptr_address_space));
  }
  CHECK(elem_type->isInteger() || elem_type->isBoolean() || elem_type->isExtDictionary());
  switch (elem_type->size()) {
    case 1:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt8PtrTy(cgen_state_->context_, ptr_address_space));
    case 2:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt16PtrTy(cgen_state_->context_, ptr_address_space));
    case 4:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt32PtrTy(cgen_state_->context_, ptr_address_space));
    case 8:
      return cgen_state_->ir_builder_.CreatePointerCast(
          ptr, llvm::Type::getInt64PtrTy(cgen_state_->context_, ptr_address_space));
    default:
      CHECK(false);
  }
  return nullptr;
}
