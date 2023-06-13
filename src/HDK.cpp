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
#include "QueryEngine/RelAlgExecutor.h"
#include "Shared/Config.h"

// Stores objects needed for various endpoints. Allows us to avoid including all headers
// in the externally available API header.
struct Internal {
  const std::string db_name{"HDK"};
  const int schema_id{1};
  const int db_id{(schema_id << 24) + 1};
  std::shared_ptr<Config> config;
  std::shared_ptr<ArrowStorage> storage;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr;
  CalciteMgr* calcite;
  std::shared_ptr<Executor> executor;
};

void HDK::read(std::shared_ptr<arrow::Table>& table, const std::string& table_name) {
  CHECK(internal_);
  CHECK(internal_->storage);
  internal_->storage->importArrowTable(table, table_name);
}

ExecutionResult HDK::query(const std::string& sql, const bool is_explain) {
  CHECK(internal_);
  CHECK(internal_->calcite);
  auto ra = internal_->calcite->process(internal_->db_name,
                                        sql,
                                        internal_->storage.get(),
                                        internal_->config.get(),
                                        {},
                                        /*legacy_syntax=*/true);

  CHECK(internal_->storage);
  CHECK(internal_->config);
  auto dag = std::make_unique<RelAlgDagBuilder>(
      ra, internal_->db_id, internal_->storage, internal_->config);

  CHECK(internal_->executor);
  CHECK(internal_->data_mgr);
  RelAlgExecutor ra_executor(
      internal_->executor.get(), internal_->storage, std::move(dag));

  auto co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
  auto eo = ExecutionOptions::fromConfig(*internal_->config.get());
  return ra_executor.executeRelAlgQuery(co, eo, /*just_explain_plan=*/false);
}

namespace {

std::shared_ptr<Config> buildConfig(const bool enable_debug_timer = false) {
  if (enable_debug_timer) {
    g_enable_debug_timer = true;
  }

  auto config = std::make_shared<Config>();
  return config;
}

}  // namespace

HDK::HDK() : internal_(std::make_unique<Internal>()) {
  CHECK(internal_);

  logger::LogOptions log_options(internal_->db_name.c_str());
  logger::init(log_options);

  internal_->config = buildConfig();

  internal_->storage = std::make_shared<ArrowStorage>(
      internal_->schema_id, internal_->db_name, internal_->db_id, internal_->config);

  internal_->data_mgr =
      std::make_shared<Data_Namespace::DataMgr>(*internal_->config.get());
  internal_->data_mgr->getPersistentStorageMgr()->registerDataProvider(
      internal_->schema_id, internal_->storage);

  // Calcite
  internal_->calcite = CalciteMgr::get(/*udf_filename=*/"",
                                       /*hdk_log_dir=*/"hdk_log",
                                       /*calcite_max_mem_mb=*/1024);

  // Executor
  internal_->executor =
      Executor::getExecutor(internal_->data_mgr.get(), internal_->config, "", "");
  internal_->executor->setSchemaProvider(internal_->storage);
}

HDK::~HDK() {}
