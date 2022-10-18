/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HDK.h"

#include <memory>

#include "ArrowStorage/ArrowStorage.h"
#include "Calcite/CalciteJNI.h"
#include "DataMgr/DataMgr.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "Shared/Config.h"
#include "Shared/SystemParameters.h"

extern bool g_enable_debug_timer;

// Stores objects needed for various endpoints. Allows us to avoid including all headers
// in the externally available API header.
struct Internal {
  std::shared_ptr<Config> config;
  std::shared_ptr<ArrowStorage> storage;
  std::shared_ptr<CalciteJNI> calcite;
  std::shared_ptr<Executor> executor;
};

void HDK::read(std::shared_ptr<arrow::Table>& table, const std::string& table_name) {
  CHECK(internal_);
  CHECK(internal_->storage);
  internal_->storage->importArrowTable(table, table_name);
}

ExecutionResult HDK::query(const std::string& sql, const bool is_explain) {}

namespace {

std::shared_ptr<Config> buildConfig(const bool enable_debug_timer = false) {
  if (enable_debug_timer) {
    g_enable_debug_timer = true;
  }

  auto config = std::make_shared<Config>();
  return config;
}

}  // namespace

HDK::HDK() : internal_(new Internal()) {
  CHECK(internal_);

  std::string hdk_name{"HDK"};
  logger::LogOptions log_options(hdk_name.c_str());
  logger::init(log_options);

  internal_->config = buildConfig();

  const int schema_id = 1;
  internal_->storage =
      std::make_shared<ArrowStorage>(schema_id, /*schema_name=*/hdk_name, /*db_id=*/0);

  SystemParameters sys_params;
  std::map<GpuMgrPlatform, std::unique_ptr<GpuMgr>> gpu_mgrs;

  auto data_mgr = std::make_shared<Data_Namespace::DataMgr>(
      *internal_->config.get(), sys_params, std::move(gpu_mgrs), 1 << 27, 0);
  CHECK(data_mgr);
  data_mgr->getPersistentStorageMgr()->registerDataProvider(schema_id,
                                                            internal_->storage);

  // Calcite
  internal_->calcite = std::make_shared<CalciteJNI>(internal_->storage,
                                                    internal_->config,
                                                    /*udf_filename=*/"",
                                                    /*calcite_max_mem_mb=*/1024);

  // Executor
  internal_->executor = Executor::getExecutor(data_mgr.get(),
                                              data_mgr->getBufferProvider(),
                                              internal_->config,
                                              "",
                                              "",
                                              sys_params);
}

HDK::~HDK() {
  if (internal_) {
    delete internal_;
  }
}

#if 0
HDK HDK::init() {
  std::string hdk_name{"HDK"};
  logger::LogOptions log_options(hdk_name.c_str());
  logger::init(log_options);

  auto config = buildConfig();

  const int schema_id = 1;
  auto storage =
      std::make_shared<ArrowStorage>(schema_id, /*schema_name=*/hdk_name, /*db_id=*/0);

  SystemParameters sys_params;
  std::map<GpuMgrPlatform, std::unique_ptr<GpuMgr>> gpu_mgrs;

  auto data_mgr = std::make_shared<Data_Namespace::DataMgr>(
      *config.get(), sys_params, std::move(gpu_mgrs), 1 << 27, 0);
  CHECK(data_mgr);
  data_mgr->getPersistentStorageMgr()->registerDataProvider(schema_id, storage);

  // Calcite
  auto calcite = std::make_shared<CalciteJNI>(
      storage, config, /*udf_filename=*/"", /*calcite_max_mem_mb=*/1024);

  // Executor
  auto executor = Executor::getExecutor(
      data_mgr.get(), data_mgr->getBufferProvider(), config, "", "", sys_params);

  return HDK();
}
#endif