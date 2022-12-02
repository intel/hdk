/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

#include <mutex>

extern bool g_enable_table_functions;

namespace table_functions {

namespace {

const hdk::ir::Type* ext_arg_pointer_type_to_type(const ExtArgumentType ext_arg_type) {
  auto& ctx = hdk::ir::Context::defaultCtx();
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
      return ctx.int8();
    case ExtArgumentType::PInt16:
      return ctx.int16();
    case ExtArgumentType::PInt32:
      return ctx.int32();
    case ExtArgumentType::PInt64:
      return ctx.int64();
    case ExtArgumentType::PFloat:
      return ctx.fp32();
    case ExtArgumentType::PDouble:
      return ctx.fp64();
    case ExtArgumentType::PBool:
      return ctx.boolean();
    case ExtArgumentType::ColumnInt8:
      return ctx.column(ctx.int8());
    case ExtArgumentType::ColumnInt16:
      return ctx.column(ctx.int16());
    case ExtArgumentType::ColumnInt32:
      return ctx.column(ctx.int32());
    case ExtArgumentType::ColumnInt64:
      return ctx.column(ctx.int64());
    case ExtArgumentType::ColumnFloat:
      return ctx.column(ctx.fp32());
    case ExtArgumentType::ColumnDouble:
      return ctx.column(ctx.fp64());
    case ExtArgumentType::ColumnBool:
      return ctx.column(ctx.boolean());
    case ExtArgumentType::ColumnListInt8:
      return ctx.columnList(ctx.int8(), 0);
    case ExtArgumentType::ColumnListInt16:
      return ctx.columnList(ctx.int16(), 0);
    case ExtArgumentType::ColumnListInt32:
      return ctx.columnList(ctx.int32(), 0);
    case ExtArgumentType::ColumnListInt64:
      return ctx.columnList(ctx.int64(), 0);
    case ExtArgumentType::ColumnListFloat:
      return ctx.columnList(ctx.fp32(), 0);
    case ExtArgumentType::ColumnListDouble:
      return ctx.columnList(ctx.fp64(), 0);
    case ExtArgumentType::ColumnListBool:
      return ctx.columnList(ctx.boolean(), 0);
    default:
      LOG(WARNING) << "ext_arg_pointer_type_to_type: ExtArgumentType `"
                   << ExtensionFunctionsWhitelist::toString(ext_arg_type)
                   << "` conversion to Type not implemented.";
      UNREACHABLE();
  }
  UNREACHABLE();
  return nullptr;
}

const hdk::ir::Type* ext_arg_type_to_type_output(const ExtArgumentType ext_arg_type) {
  auto& ctx = hdk::ir::Context::defaultCtx();
  switch (ext_arg_type) {
    case ExtArgumentType::PInt8:
    case ExtArgumentType::ColumnInt8:
    case ExtArgumentType::ColumnListInt8:
    case ExtArgumentType::Int8:
      return ctx.int8();
    case ExtArgumentType::PInt16:
    case ExtArgumentType::ColumnInt16:
    case ExtArgumentType::ColumnListInt16:
    case ExtArgumentType::Int16:
      return ctx.int16();
    case ExtArgumentType::PInt32:
    case ExtArgumentType::ColumnInt32:
    case ExtArgumentType::ColumnListInt32:
    case ExtArgumentType::Int32:
      return ctx.int32();
    case ExtArgumentType::PInt64:
    case ExtArgumentType::ColumnInt64:
    case ExtArgumentType::ColumnListInt64:
    case ExtArgumentType::Int64:
      return ctx.int64();
    case ExtArgumentType::PFloat:
    case ExtArgumentType::ColumnFloat:
    case ExtArgumentType::ColumnListFloat:
    case ExtArgumentType::Float:
      return ctx.fp32();
    case ExtArgumentType::PDouble:
    case ExtArgumentType::ColumnDouble:
    case ExtArgumentType::ColumnListDouble:
    case ExtArgumentType::Double:
      return ctx.fp64();
    case ExtArgumentType::PBool:
    case ExtArgumentType::ColumnBool:
    case ExtArgumentType::ColumnListBool:
    case ExtArgumentType::Bool:
      return ctx.boolean();
    case ExtArgumentType::ColumnTextEncodingDict:
    case ExtArgumentType::ColumnListTextEncodingDict:
    case ExtArgumentType::TextEncodingDict:
      return ctx.extDict(ctx.text(), 0);
    default:
      LOG(WARNING) << "ext_arg_type_to_type_output: ExtArgumentType `"
                   << ExtensionFunctionsWhitelist::toString(ext_arg_type)
                   << "` conversion to Type not implemented.";
      UNREACHABLE();
  }
  UNREACHABLE();
  return nullptr;
}

}  // namespace

const hdk::ir::Type* TableFunction::getInputType(const size_t idx) const {
  CHECK_LT(idx, input_args_.size());
  return ext_arg_pointer_type_to_type(input_args_[idx]);
}

const hdk::ir::Type* TableFunction::getOutputType(const size_t idx) const {
  CHECK_LT(idx, output_args_.size());
  // TODO(adb): conditionally handle nulls
  return ext_arg_type_to_type_output(output_args_[idx]);
}

int32_t TableFunction::countScalarArgs() const {
  int32_t scalar_args = 0;
  for (const auto& ext_arg : input_args_) {
    if (is_ext_arg_type_scalar(ext_arg)) {
      scalar_args += 1;
    }
  }
  return scalar_args;
}

const std::map<std::string, std::string>& TableFunction::getAnnotation(
    const size_t idx) const {
  if (annotations_.size() == 0) {
    static const std::map<std::string, std::string> empty = {};
    return empty;
  }
  CHECK_LT(idx, annotations_.size());
  return annotations_[idx];
}

const std::map<std::string, std::string>& TableFunction::getInputAnnotation(
    const size_t input_arg_idx) const {
  CHECK_LT(input_arg_idx, input_args_.size());
  return getAnnotation(input_arg_idx);
}

const std::map<std::string, std::string>& TableFunction::getOutputAnnotation(
    const size_t output_arg_idx) const {
  CHECK_LT(output_arg_idx, output_args_.size());
  return getAnnotation(output_arg_idx + sql_args_.size());
}

std::pair<int32_t, int32_t> TableFunction::getInputID(const size_t idx) const {
  // if the annotation is of the form args<INT,INT>, it is refering to a column list
#define PREFIX_LENGTH 5
  const auto& annotation = getOutputAnnotation(idx);
  auto annot = annotation.find("input_id");
  if (annot == annotation.end()) {
    size_t lo = 0;
    for (const auto& ext_arg : input_args_) {
      switch (ext_arg) {
        case ExtArgumentType::TextEncodingDict:
        case ExtArgumentType::ColumnTextEncodingDict:
        case ExtArgumentType::ColumnListTextEncodingDict:
          return std::make_pair(lo, 0);
        default:
          lo++;
      }
    }
    UNREACHABLE();
  }

  const std::string& input_id = annot->second;

  size_t comma = input_id.find(",");
  int32_t gt = input_id.size() - 1;
  int32_t lo = std::stoi(input_id.substr(PREFIX_LENGTH, comma - 1));

  if (comma == std::string::npos) {
    return std::make_pair(lo, 0);
  }
  int32_t hi = std::stoi(input_id.substr(comma + 1, gt - comma - 1));
  return std::make_pair(lo, hi);
}

size_t TableFunction::getSqlOutputRowSizeParameter() const {
  /*
    This function differs from getOutputRowSizeParameter() since it returns the correct
    index for the sizer in the sql_args list. For instance, consider the example below:

      RowMultiplier=4
      input_args=[{i32*, i64}, {i32*, i64}, {i32*, i64}, i32, {i32*, i64}, {i32*, i64},
    i32] sql_args=[cursor, i32, cursor, i32]

    Non-scalar args are aggregated in a cursor inside the sql_args list and the new
    sizer index is 2 rather than 4 originally specified.
  */

  if (hasUserSpecifiedOutputSizeMultiplier()) {
    size_t sizer = getOutputRowSizeParameter();  // lookup until reach the sizer arg
    int32_t ext_arg_index = 0, sql_arg_index = 0;

    auto same_kind = [&](const ExtArgumentType& ext_arg, const ExtArgumentType& sql_arg) {
      return ((is_ext_arg_type_scalar(ext_arg) && is_ext_arg_type_scalar(sql_arg)) ||
              (is_ext_arg_type_nonscalar(ext_arg) && is_ext_arg_type_nonscalar(sql_arg)));
    };

    while ((size_t)ext_arg_index < sizer) {
      if ((size_t)ext_arg_index == sizer - 1)
        return sql_arg_index;

      const auto& ext_arg = input_args_[ext_arg_index];
      const auto& sql_arg = sql_args_[sql_arg_index];

      if (same_kind(ext_arg, sql_arg)) {
        ++ext_arg_index;
        ++sql_arg_index;
      } else {
        CHECK(same_kind(ext_arg, sql_args_[sql_arg_index - 1]));
        ext_arg_index += 1;
      }
    }

    CHECK(false);
  }

  return getOutputRowSizeParameter();
}

void TableFunctionsFactory::add(
    const std::string& name,
    const TableFunctionOutputRowSizer sizer,
    const std::vector<ExtArgumentType>& input_args,
    const std::vector<ExtArgumentType>& output_args,
    const std::vector<ExtArgumentType>& sql_args,
    const std::vector<std::map<std::string, std::string>>& annotations,
    bool is_runtime,
    bool uses_manager) {
  auto tf = TableFunction(name,
                          sizer,
                          input_args,
                          output_args,
                          sql_args,
                          annotations,
                          is_runtime,
                          uses_manager);
  auto sig = tf.getSignature();
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.getName() == name) {
      if (it->second.isRuntime()) {
        LOG(WARNING)
            << "Overriding existing run-time table function (reset not called?): "
            << name;
        it = functions_.erase(it);
      } else {
        throw std::runtime_error("Will not override existing load-time table function: " +
                                 name);
      }
    } else {
      if (sig == it->second.getSignature() &&
          ((tf.isCPU() && it->second.isCPU()) || (tf.isGPU() && it->second.isGPU()))) {
        LOG(WARNING)
            << "The existing (1) and added (2) table functions have the same signature `"
            << sig << "`:\n"
            << "  1: " << it->second.toString() << "\n  2: " << tf.toString() << "\n";
      }
      ++it;
    }
  }

  functions_.emplace(name, tf);
  if (sizer.type == OutputBufferSizeType::kUserSpecifiedRowMultiplier) {
    auto input_args2 = input_args;
    input_args2.erase(input_args2.begin() + sizer.val - 1);

    auto sql_args2 = sql_args;
    auto sql_sizer_pos = tf.getSqlOutputRowSizeParameter();
    sql_args2.erase(sql_args2.begin() + sql_sizer_pos);

    auto tf2 = TableFunction(name + DEFAULT_ROW_MULTIPLIER_SUFFIX,
                             sizer,
                             input_args2,
                             output_args,
                             sql_args2,
                             annotations,
                             is_runtime,
                             uses_manager);
    auto sig = tf2.getSignature();
    for (auto it = functions_.begin(); it != functions_.end();) {
      if (sig == it->second.getSignature() &&
          ((tf2.isCPU() && it->second.isCPU()) || (tf2.isGPU() && it->second.isGPU()))) {
        LOG(WARNING)
            << "The existing (1) and added (2) table functions have the same signature `"
            << sig << "`:\n"
            << "  1: " << it->second.toString() << "\n  2: " << tf2.toString() << "\n";
      }
      ++it;
    }
    functions_.emplace(name + DEFAULT_ROW_MULTIPLIER_SUFFIX, tf2);
  }
}

/*
  The implementation for `void TableFunctionsFactory::init()` is
  generated by QueryEngine/scripts/generate_TableFunctionsFactory_init.py
*/

// removes existing runtime table functions
void TableFunctionsFactory::reset() {
  if (!g_enable_table_functions) {
    return;
  }
  for (auto it = functions_.begin(); it != functions_.end();) {
    if (it->second.isRuntime()) {
      it = functions_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(const std::string& name,
                                                                  const bool is_gpu) {
  std::vector<TableFunction> table_funcs;
  auto table_func_name = name;
  boost::algorithm::to_lower(table_func_name);
  for (const auto& pair : functions_) {
    auto fname = ExtensionFunction::drop_suffix(pair.first);
    if (fname == table_func_name &&
        (is_gpu ? pair.second.isGPU() : pair.second.isCPU())) {
      table_funcs.push_back(pair.second);
    }
  }
  return table_funcs;
}

std::vector<TableFunction> TableFunctionsFactory::get_table_funcs(const bool is_runtime) {
  std::vector<TableFunction> table_funcs;
  for (const auto& pair : functions_) {
    if (pair.second.isRuntime() == is_runtime) {
      table_funcs.push_back(pair.second);
    }
  }
  return table_funcs;
}

std::unordered_map<std::string, TableFunction> TableFunctionsFactory::functions_;

}  // namespace table_functions