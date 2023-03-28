/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ColumnarResults.h"
#include "ResultSetTableToken.h"

#include "DataMgr/AbstractDataProvider.h"
#include "DataProvider/DictDescriptor.h"
#include "ResultSet/ResultSet.h"
#include "SchemaMgr/SimpleSchemaProvider.h"
#include "Shared/Config.h"
#include "Shared/mapd_shared_mutex.h"

namespace Data_Namespace {
class DataMgr;
}

namespace hdk {

class ResultSetRegistry : public SimpleSchemaProvider,
                          public AbstractDataProvider,
                          public std::enable_shared_from_this<ResultSetRegistry> {
 public:
  constexpr static int SCHEMA_ID = 100;
  constexpr static int DB_ID = (SCHEMA_ID << 24) + 1;

  ResultSetRegistry(ConfigPtr config,
                    const std::string& schema_name = "rs_registry",
                    int db_id = DB_ID);

  static std::shared_ptr<ResultSetRegistry> getOrCreate(Data_Namespace::DataMgr* data_mgr,
                                                        ConfigPtr config);

  ResultSetTableTokenPtr put(ResultSetTable table);
  ResultSetPtr get(const ResultSetTableToken& token, size_t frag_id) const;
  void drop(const ResultSetTableToken& token);

  void fetchBuffer(const ChunkKey& key,
                   Data_Namespace::AbstractBuffer* dest,
                   const size_t num_bytes = 0) override;

  std::unique_ptr<Data_Namespace::AbstractDataToken> getZeroCopyBufferMemory(
      const ChunkKey& key,
      size_t num_bytes) override;

  TableFragmentsInfo getTableMetadata(int db_id, int table_id) const override;

  const DictDescriptor* getDictMetadata(int dict_id, bool load_dict = true) override;

 private:
  bool useColumnarResults(const ResultSet& rs) const;

  ChunkStats getChunkStats(int table_id, size_t frag_idx, size_t col_idx) const;

  struct DataFragment {
    size_t offset = 0;
    size_t row_count = 0;
    ResultSetPtr rs;
    std::unique_ptr<ColumnarResults> columnar_res;
    std::unique_ptr<mapd_shared_mutex> mutex;
    ChunkMetadataMap meta;
  };

  struct TableData {
    mapd_shared_mutex mutex;
    std::vector<DataFragment> fragments;
    size_t row_count;
  };

  const int db_id_;
  const int schema_id_;
  int next_table_id_ = 1;
  std::unordered_map<int, std::unique_ptr<TableData>> tables_;
  const ConfigPtr config_;
  mutable mapd_shared_mutex data_mutex_;
};

}  // namespace hdk
