/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CalciteJNI.h"

#include "OSDependent/omnisci_path.h"

#include <boost/algorithm/string.hpp>

#include <fstream>
#include <regex>
#include <unordered_map>

namespace {

std::string canonizeType(const std::string& type) {
  static const std::unordered_map<std::string, std::string> type_mapping = {
      {"bool", "i1"},
      {"_Bool", "i1"},
      {"int8_t", "i8"},
      {"int8", "i8"},
      {"char", "i8"},
      {"int16_t", "i16"},
      {"int16", "i16"},
      {"short", "i16"},
      {"int32_t", "i32"},
      {"int32", "i32"},
      {"int", "i32"},
      {"int64_t", "i64"},
      {"size_t", "i64"},
      {"long", "i64"},
      {"int64", "i64"},
      {"float", "float"},
      {"float32", "float"},
      {"double", "double"},
      {"float64", "double"},
      {"", "void"},
      {"void", "void"},
      {"Array<bool>", "{i1*, i64, i8}*"},
      {"Array<int8_t>", "{i8*, i64, i8}*"},
      {"Array<char>", "{81*, i64, i8}*"},
      {"Array<int16_t>", "{i16*, i64, i8}*"},
      {"Array<short>", "{i16*, i64, i8}*"},
      {"Array<int16_t>", "{i16*, i64, i8}*"},
      {"Array<short>", "{i16*, i64, i8}*"},
      {"Array<int32_t>", "{i32*, i64, i8}*"},
      {"Array<int>", "{i32*, i64, i8}*"},
      {"Array<int64_t>", "{i64*, i64, i8}*"},
      {"Array<size_t>", "{i64*, i64, i8}*"},
      {"Array<long>", "{i64*, i64, i8}*"},
      {"Array<float>", "{float*, i64, i8}*"},
      {"Array<double>", "{double*, i64, i8}*"}};

  static const std::string const_prefix = "const ";
  static const std::string std_prefix = "std::";

  if (type.substr(0, const_prefix.size()) == const_prefix) {
    return canonizeType(type.substr(const_prefix.size()));
  }

  if (type.substr(0, std_prefix.size()) == std_prefix) {
    return canonizeType(type.substr(std_prefix.size()));
  }

  auto it = type_mapping.find(type);
  if (it == type_mapping.end()) {
    throw std::runtime_error("Unknown type string in extension function: " + type);
  }

  return it->second;
}

std::string parseExtensionFunctionSignatures() {
  auto root_abs_path = omnisci::get_root_abs_path();
  std::string ext_ast_path = root_abs_path + "/QueryEngine/ExtensionFunctions.ast";
  std::ifstream fs(ext_ast_path);
  if (!fs.is_open()) {
    throw std::runtime_error("Cannot open extension functions file: " + ext_ast_path);
  }

  std::regex sig_parser("\\| (?:[\\` ]|used)+ ([\\w]+) '([\\w<>]+) \\((.*)\\)'");
  std::smatch match_res;
  std::stringstream ss;
  ss << "[\n";
  bool first = true;
  for (std::string line; std::getline(fs, line);) {
    if (!std::regex_match(line, match_res, sig_parser)) {
      continue;
    }

    std::vector<std::string> arg_types;
    boost::split(arg_types, match_res[3].str(), boost::is_any_of(","));

    if (first) {
      first = false;
    } else {
      ss << ",\n";
    }

    ss << "{\n  \"name\": \"" << match_res[1].str() << "\",\n"
       << "  \"ret\": \"" << canonizeType(match_res[2].str()) << "\",\n"
       << "  \"args\": [\n";
    for (size_t i = 0; i < arg_types.size(); ++i) {
      ss << (i ? ",\n    \"" : "    \"") << canonizeType(boost::trim_copy(arg_types[i]))
         << "\"";
    }
    ss << "\n  ]\n}";
  }
  ss << "\n]\n";

  return ss.str();
}

}  // namespace

CalciteMgr::~CalciteMgr() {}

CalciteMgr* CalciteMgr::get(const std::string& udf_filename,
                            const std::string& log_dir,
                            size_t calcite_max_mem_mb) {
  std::call_once(instance_init_flag_, [=] {
    instance_ = std::unique_ptr<CalciteMgr>(
        new CalciteMgr(udf_filename, calcite_max_mem_mb, log_dir));
  });
  return instance_.get();
}

std::string CalciteMgr::process(const std::string&,
                                const std::string& sql_string,
                                SchemaProvider*,
                                Config*,
                                const std::vector<FilterPushDownInfo>&,
                                const bool,
                                const bool,
                                const bool) {
  static const std::string ra_prefix = "execute calcite";
  // For RA JSON queries simply remove the prefix and return.
  if (sql_string.substr(0, ra_prefix.size()) == ra_prefix) {
    return sql_string.substr(ra_prefix.size());
  }
  throw std::runtime_error("This HDK build doesn't include SQL support.");
}

std::string CalciteMgr::getExtensionFunctionWhitelist() {
  static std::string cached_res = "";
  if (cached_res.empty()) {
    cached_res = parseExtensionFunctionSignatures();
  }
  return cached_res;
}

std::string CalciteMgr::getUserDefinedFunctionWhitelist() {
  return "[]";
}

std::string CalciteMgr::getRuntimeExtensionFunctionWhitelist() {
  return "[]";
}

void CalciteMgr::setRuntimeExtensionFunctions(const std::vector<ExtensionFunction>&,
                                              bool) {}

CalciteMgr::CalciteMgr(const std::string&, size_t, const std::string&) {}

std::once_flag CalciteMgr::instance_init_flag_;
std::unique_ptr<CalciteMgr> CalciteMgr::instance_;
