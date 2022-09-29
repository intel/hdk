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

#include "../Shared/funcannotations.h"
#include "../Shared/sqldefs.h"

#include <boost/locale/conversion.hpp>

extern "C" RUNTIME_EXPORT uint64_t string_decode(int8_t* chunk_iter_, int64_t pos) {
  auto chunk_iter = reinterpret_cast<ChunkIter*>(chunk_iter_);
  VarlenDatum vd;
  bool is_end;
  ChunkIter_get_nth(chunk_iter, pos, false, &vd, &is_end);
  CHECK(!is_end);
  return vd.is_null ? 0
                    : (reinterpret_cast<uint64_t>(vd.pointer) & 0xffffffffffff) |
                          (static_cast<uint64_t>(vd.length) << 48);
}

extern "C" RUNTIME_EXPORT uint64_t string_decompress(const int32_t string_id,
                                                     const int64_t string_dict_handle) {
  if (string_id == NULL_INT) {
    return 0;
  }
  auto string_dict_proxy =
      reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  auto string_bytes = string_dict_proxy->getStringBytes(string_id);
  CHECK(string_bytes.first);
  return (reinterpret_cast<uint64_t>(string_bytes.first) & 0xffffffffffff) |
         (static_cast<uint64_t>(string_bytes.second) << 48);
}

extern "C" RUNTIME_EXPORT int32_t string_compress(const int64_t ptr_and_len,
                                                  const int64_t string_dict_handle) {
  std::string raw_str(reinterpret_cast<char*>(extract_str_ptr_noinline(ptr_and_len)),
                      extract_str_len_noinline(ptr_and_len));
  auto string_dict_proxy =
      reinterpret_cast<const StringDictionaryProxy*>(string_dict_handle);
  return string_dict_proxy->getIdOfString(raw_str);
}

extern "C" RUNTIME_EXPORT int32_t lower_encoded(int32_t string_id,
                                                int64_t string_dict_proxy_address) {
  StringDictionaryProxy* string_dict_proxy =
      reinterpret_cast<StringDictionaryProxy*>(string_dict_proxy_address);
  auto str = string_dict_proxy->getString(string_id);
  return string_dict_proxy->getOrAddTransient(boost::locale::to_lower(str));
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::CharLengthExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto str_lv = codegen(expr->arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    if (config_.exec.watchdog.enable) {
      throw WatchdogException(
          "LENGTH / CHAR_LENGTH on dictionary-encoded strings would be slow");
    }
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  std::vector<llvm::Value*> charlength_args{str_lv[1], str_lv[2]};
  std::string fn_name("char_length");
  if (expr->calcEncodedLength()) {
    fn_name += "_encoded";
  }
  const bool is_nullable{expr->arg()->type()->nullable()};
  if (is_nullable) {
    fn_name += "_nullable";
    charlength_args.push_back(cgen_state_->inlineIntNull(expr->type()));
  }
  return expr->calcEncodedLength()
             ? cgen_state_->emitExternalCall(
                   fn_name, get_int_type(32, cgen_state_->context_), charlength_args)
             : cgen_state_->emitCall(fn_name, charlength_args);
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::KeyForStringExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto str_lv = codegen(expr->arg(), true, co);
  CHECK_EQ(size_t(1), str_lv.size());
  return cgen_state_->emitCall("key_for_string_encoded", str_lv);
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::LowerExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (co.device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }

  auto str_id_lv = codegen(expr->arg(), true, co);
  CHECK_EQ(size_t(1), str_id_lv.size());

  CHECK(expr->type()->isExtDictionary());
  const auto string_dictionary_proxy = executor()->getStringDictionaryProxy(
      expr->type()->as<hdk::ir::ExtDictionaryType>()->dictId(),
      executor()->getRowSetMemoryOwner(),
      true);
  CHECK(string_dictionary_proxy);

  std::vector<llvm::Value*> args{
      str_id_lv[0],
      cgen_state_->llInt(reinterpret_cast<int64_t>(string_dictionary_proxy))};

  return cgen_state_->emitExternalCall(
      "lower_encoded", get_int_type(32, cgen_state_->context_), args);
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::LikeExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (is_unnest(extract_cast_arg(expr->arg()))) {
    throw std::runtime_error("LIKE not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr =
        dynamic_cast<const hdk::ir::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->type()->isString());
    CHECK_EQ(size_t(1), escape_char_expr->value().stringval->size());
    escape_char = (*escape_char_expr->value().stringval)[0];
  }
  auto pattern = dynamic_cast<const hdk::ir::Constant*>(expr->likeExpr());
  CHECK(pattern);
  auto fast_dict_like_lv = codegenDictLike(
      expr->argShared(), pattern, expr->isIlike(), expr->isSimple(), escape_char, co);
  if (fast_dict_like_lv) {
    return fast_dict_like_lv;
  }
  const auto& type = expr->arg()->type();
  CHECK(type->isString() || type->isExtDictionary());
  if (config_.exec.watchdog.enable && type->isExtDictionary()) {
    throw WatchdogException(
        "Cannot do LIKE / ILIKE on this dictionary encoded column, its cardinality is "
        "too high");
  }
  auto str_lv = codegen(expr->arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
    if (co.device_type == ExecutorDeviceType::GPU) {
      throw QueryMustRunOnCpu();
    }
  }
  auto like_expr_arg_lvs = codegen(expr->likeExpr(), true, co);
  CHECK_EQ(size_t(3), like_expr_arg_lvs.size());
  const bool is_nullable{expr->arg()->type()->nullable()};
  std::vector<llvm::Value*> str_like_args{
      str_lv[1], str_lv[2], like_expr_arg_lvs[1], like_expr_arg_lvs[2]};
  std::string fn_name{expr->isIlike() ? "string_ilike" : "string_like"};
  if (expr->isSimple()) {
    fn_name += "_simple";
  } else {
    str_like_args.push_back(cgen_state_->llInt(int8_t(escape_char)));
  }
  if (is_nullable) {
    fn_name += "_nullable";
    str_like_args.push_back(cgen_state_->inlineIntNull(expr->type()));
  }
  return cgen_state_->emitCall(fn_name, str_like_args);
}

llvm::Value* CodeGenerator::codegenDictLike(const hdk::ir::ExprPtr like_arg,
                                            const hdk::ir::Constant* pattern,
                                            const bool ilike,
                                            const bool is_simple,
                                            const char escape_char,
                                            const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto cast_oper = like_arg->as<hdk::ir::UOper>();
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK(cast_oper->isCast());
  const auto dict_like_arg = cast_oper->operandShared();
  const auto& dict_like_arg_type = dict_like_arg->type();
  if (!dict_like_arg_type->isExtDictionary()) {
    throw(std::runtime_error("Cast from " + dict_like_arg_type->toString() + " to " +
                             cast_oper->type()->toString() + " not supported"));
  }
  const auto sdp = executor()->getStringDictionaryProxy(
      dict_like_arg_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
      executor()->getRowSetMemoryOwner(),
      true);
  if (sdp->storageEntryCount() > 200000000) {
    return nullptr;
  }
  const auto& pattern_type = pattern->type();
  CHECK(pattern_type->isString());
  const auto& pattern_datum = pattern->value();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_ids = sdp->getLike(pattern_str, ilike, is_simple, escape_char);
  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());
  const auto in_values = std::make_shared<hdk::ir::InIntegerSet>(
      dict_like_arg, matching_ids_64, !dict_like_arg_type->nullable());
  return codegen(in_values.get(), co);
}

namespace {

std::vector<int32_t> get_compared_ids(const StringDictionaryProxy* dict,
                                      const SQLOps compare_operator,
                                      const std::string& pattern) {
  std::vector<int> ret;
  switch (compare_operator) {
    case kLT:
      ret = dict->getCompare(pattern, "<");
      break;
    case kLE:
      ret = dict->getCompare(pattern, "<=");
      break;
    case kEQ:
    case kBW_EQ:
      ret = dict->getCompare(pattern, "=");
      break;
    case kGT:
      ret = dict->getCompare(pattern, ">");
      break;
    case kGE:
      ret = dict->getCompare(pattern, ">=");
      break;
    case kNE:
      ret = dict->getCompare(pattern, "<>");
      break;
    default:
      std::runtime_error("unsuported operator for string comparision");
  }
  return ret;
}
}  // namespace

llvm::Value* CodeGenerator::codegenDictStrCmp(const hdk::ir::ExprPtr lhs,
                                              const hdk::ir::ExprPtr rhs,
                                              const SQLOps compare_operator,
                                              const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  auto rhs_cast_oper = std::dynamic_pointer_cast<const hdk::ir::UOper>(rhs);
  auto lhs_cast_oper = std::dynamic_pointer_cast<const hdk::ir::UOper>(lhs);
  auto rhs_col_var = std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(rhs);
  auto lhs_col_var = std::dynamic_pointer_cast<const hdk::ir::ColumnVar>(lhs);
  std::shared_ptr<const hdk::ir::UOper> cast_oper;
  std::shared_ptr<const hdk::ir::ColumnVar> col_var;
  auto compare_opr = compare_operator;
  if (lhs_col_var && rhs_col_var) {
    CHECK(lhs_col_var->type()->isExtDictionary());
    CHECK(rhs_col_var->type()->isExtDictionary());
    if (lhs_col_var->type()->as<hdk::ir::ExtDictionaryType>()->dictId() ==
        rhs_col_var->type()->as<hdk::ir::ExtDictionaryType>()->dictId()) {
      if (compare_operator == kEQ || compare_operator == kNE) {
        // TODO (vraj): implement compare between two dictionary encoded columns which
        // share a dictionary
        return nullptr;
      }
    }
    // TODO (vraj): implement compare between two dictionary encoded columns which don't
    // shared dictionary
    throw std::runtime_error("Decoding two Dictionary encoded columns will be slow");
  } else if (lhs_col_var && rhs_cast_oper) {
    cast_oper.swap(rhs_cast_oper);
    col_var.swap(lhs_col_var);
  } else if (lhs_cast_oper && rhs_col_var) {
    cast_oper.swap(lhs_cast_oper);
    col_var.swap(rhs_col_var);
    switch (compare_operator) {
      case kLT:
        compare_opr = kGT;
        break;
      case kLE:
        compare_opr = kGE;
        break;
      case kGT:
        compare_opr = kLT;
        break;
      case kGE:
        compare_opr = kLE;
      default:
        break;
    }
  }
  if (!cast_oper || !col_var) {
    return nullptr;
  }
  CHECK(cast_oper->isCast());

  const auto const_expr = cast_oper->operand()->as<hdk::ir::Constant>();
  if (!const_expr) {
    // Analyzer casts dictionary encoded columns to none encoded if there is a comparison
    // between two encoded columns. Which we currently do not handle.
    return nullptr;
  }
  const auto& const_val = const_expr->value();

  const auto col_type = col_var->type();
  CHECK(col_type->isExtDictionary());
  const auto sdp = executor()->getStringDictionaryProxy(
      col_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
      executor()->getRowSetMemoryOwner(),
      true);

  if (sdp->storageEntryCount() > 200000000) {
    std::runtime_error("Cardinality for string dictionary is too high");
    return nullptr;
  }

  const auto& pattern_str = *const_val.stringval;
  const auto matching_ids = get_compared_ids(sdp, compare_opr, pattern_str);

  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());

  const auto in_values = std::make_shared<hdk::ir::InIntegerSet>(
      col_var, matching_ids_64, !col_type->nullable());
  return codegen(in_values.get(), co);
}

llvm::Value* CodeGenerator::codegen(const hdk::ir::RegexpExpr* expr,
                                    const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  if (is_unnest(extract_cast_arg(expr->get_arg()))) {
    throw std::runtime_error("REGEXP not supported for unnested expressions");
  }
  char escape_char{'\\'};
  if (expr->get_escape_expr()) {
    auto escape_char_expr =
        dynamic_cast<const hdk::ir::Constant*>(expr->get_escape_expr());
    CHECK(escape_char_expr);
    CHECK(escape_char_expr->type()->isString());
    CHECK_EQ(size_t(1), escape_char_expr->value().stringval->size());
    escape_char = (*escape_char_expr->value().stringval)[0];
  }
  auto pattern = dynamic_cast<const hdk::ir::Constant*>(expr->get_pattern_expr());
  CHECK(pattern);
  auto fast_dict_pattern_lv =
      codegenDictRegexp(expr->get_own_arg(), pattern, escape_char, co);
  if (fast_dict_pattern_lv) {
    return fast_dict_pattern_lv;
  }
  const auto& type = expr->get_arg()->type();
  CHECK(type->isString() || type->isExtDictionary());
  if (config_.exec.watchdog.enable && type->isExtDictionary()) {
    throw WatchdogException(
        "Cannot do REGEXP_LIKE on this dictionary encoded column, its cardinality is too "
        "high");
  }
  // Now we know we are working on NONE ENCODED column. So switch back to CPU
  if (co.device_type == ExecutorDeviceType::GPU) {
    throw QueryMustRunOnCpu();
  }
  auto str_lv = codegen(expr->get_arg(), true, co);
  if (str_lv.size() != 3) {
    CHECK_EQ(size_t(1), str_lv.size());
    str_lv.push_back(cgen_state_->emitCall("extract_str_ptr", {str_lv.front()}));
    str_lv.push_back(cgen_state_->emitCall("extract_str_len", {str_lv.front()}));
  }
  auto regexp_expr_arg_lvs = codegen(expr->get_pattern_expr(), true, co);
  CHECK_EQ(size_t(3), regexp_expr_arg_lvs.size());
  const bool is_nullable{expr->get_arg()->type()->nullable()};
  std::vector<llvm::Value*> regexp_args{
      str_lv[1], str_lv[2], regexp_expr_arg_lvs[1], regexp_expr_arg_lvs[2]};
  std::string fn_name("regexp_like");
  regexp_args.push_back(cgen_state_->llInt(int8_t(escape_char)));
  if (is_nullable) {
    fn_name += "_nullable";
    regexp_args.push_back(cgen_state_->inlineIntNull(expr->type()));
    return cgen_state_->emitExternalCall(
        fn_name, get_int_type(8, cgen_state_->context_), regexp_args);
  }
  return cgen_state_->emitExternalCall(
      fn_name, get_int_type(1, cgen_state_->context_), regexp_args);
}

llvm::Value* CodeGenerator::codegenDictRegexp(const hdk::ir::ExprPtr pattern_arg,
                                              const hdk::ir::Constant* pattern,
                                              const char escape_char,
                                              const CompilationOptions& co) {
  AUTOMATIC_IR_METADATA(cgen_state_);
  const auto cast_oper = pattern_arg->as<hdk::ir::UOper>();
  if (!cast_oper) {
    return nullptr;
  }
  CHECK(cast_oper);
  CHECK(cast_oper->isCast());
  const auto dict_regexp_arg = cast_oper->operandShared();
  const auto& dict_regexp_arg_type = dict_regexp_arg->type();
  CHECK(dict_regexp_arg_type->isExtDictionary());
  const auto dict_id = dict_regexp_arg_type->as<hdk::ir::ExtDictionaryType>()->dictId();
  const auto sdp = executor()->getStringDictionaryProxy(
      dict_id, executor()->getRowSetMemoryOwner(), true);
  if (sdp->storageEntryCount() > 15000000) {
    return nullptr;
  }
  const auto& pattern_type = pattern->type();
  CHECK(pattern_type->isString());
  const auto& pattern_datum = pattern->value();
  const auto& pattern_str = *pattern_datum.stringval;
  const auto matching_ids = sdp->getRegexpLike(pattern_str, escape_char);
  // InIntegerSet requires 64-bit values
  std::vector<int64_t> matching_ids_64(matching_ids.size());
  std::copy(matching_ids.begin(), matching_ids.end(), matching_ids_64.begin());
  const auto in_values = std::make_shared<hdk::ir::InIntegerSet>(
      dict_regexp_arg, matching_ids_64, !dict_regexp_arg_type->nullable());
  return codegen(in_values.get(), co);
}
