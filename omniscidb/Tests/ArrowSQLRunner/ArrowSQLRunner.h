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

#pragma once

#include "ArrowStorage/ArrowStorage.h"
#include "IR/Context.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "ResultSet/ArrowResultSet.h"
#include "ResultSetRegistry/ResultSetRegistry.h"
#include "Shared/Config.h"

#include "BufferPoolStats.h"

class CalciteMgr;
class Executor;
class RelAlgExecutor;

namespace TestHelpers::ArrowSQLRunner {

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;

void init(ConfigPtr config = nullptr, const std::string& udf_filename = "");

void reset();

Config& config();

ConfigPtr configPtr();

bool gpusPresent();

void printStats();

std::vector<ExecutorDeviceType> testedDevices();

void createTable(
    const std::string& table_name,
    const std::vector<ArrowStorage::ColumnDescription>& columns,
    const ArrowStorage::TableOptions& options = ArrowStorage::TableOptions());

void dropTable(const std::string& table_name);

void insertCsvValues(const std::string& table_name, const std::string& values);

void insertJsonValues(const std::string& table_name, const std::string& values);

std::string getSqlQueryRelAlg(const std::string& query_str);

ExecutionResult runSqlQuery(const std::string& sql,
                            const CompilationOptions& co,
                            const ExecutionOptions& eo);

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            const ExecutionOptions& eo);

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            bool allow_loop_joins);

ExecutionResult runQuery(std::unique_ptr<hdk::ir::QueryDag> dag,
                         ExecutorDeviceType device_type = ExecutorDeviceType::CPU,
                         bool allow_loop_joins = false);

ExecutionOptions getExecutionOptions(bool allow_loop_joins, bool just_explain = false);

CompilationOptions getCompilationOptions(ExecutorDeviceType device_type);

hdk::ResultSetTableTokenPtr run_multiple_agg(const std::string& query_str,
                                             const ExecutorDeviceType device_type,
                                             const bool allow_loop_joins = true);

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool allow_loop_joins = true);

void run_sqlite_query(const std::string& query_string);

void sqlite_batch_insert(const std::string& table_name,
                         std::vector<std::vector<std::string>>& insert_vals);

void c(const std::string& query_string, const ExecutorDeviceType device_type);

void c(const std::string& query_string,
       const std::string& sqlite_query_string,
       const ExecutorDeviceType device_type);

/* timestamp approximate checking for NOW() */
void cta(const std::string& query_string,
         const std::string& sqlite_query_string,
         const ExecutorDeviceType device_type);

void c_arrow(
    const std::string& query_string,
    const ExecutorDeviceType device_type,
    size_t min_result_size_for_bulk_dictionary_fetch =
        ArrowResultSetConverter::default_min_result_size_for_bulk_dictionary_fetch,
    double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch =
        ArrowResultSetConverter::
            default_max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);

void clearCpuMemory();

BufferPoolStats getBufferPoolStats(const Data_Namespace::MemoryLevel memory_level =
                                       Data_Namespace::MemoryLevel::CPU_LEVEL);

std::shared_ptr<ArrowStorage> getStorage();

SchemaProviderPtr getSchemaProvider();

std::shared_ptr<hdk::ResultSetRegistry> getResultSetRegistry();

DataMgr* getDataMgr();

Executor* getExecutor();

CalciteMgr* getCalcite();

std::unique_ptr<RelAlgExecutor> makeRelAlgExecutor(const std::string& query_str);

inline hdk::ir::Context& ctx() {
  return hdk::ir::Context::defaultCtx();
}

}  // namespace TestHelpers::ArrowSQLRunner
