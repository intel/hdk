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

void HDK::read() {}

ExecutionResult HDK::query(const std::string& sql, const bool is_explain) {}

/**
pyhdk.initLogger()
config = pyhdk.buildConfig()
storage = pyhdk.storage.ArrowStorage(1)
data_mgr = pyhdk.storage.DataMgr(config)
data_mgr.registerDataProvider(storage)

calcite = pyhdk.sql.Calcite(storage, config)
executor = pyhdk.Executor(data_mgr, config)
*/

namespace {

std::shared_ptr<Config> buildConfig(const bool enable_debug_timer = false) {
  if (enable_debug_timer) {
    g_enable_debug_timer = true;
  }

  auto config = std::make_shared<Config>();
  return config;
}

}  // namespace

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