/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CalciteJNI.h"

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
  return "[]";
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
