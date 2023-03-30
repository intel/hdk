/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ResultSetRegistry.h"

#include "DataMgr/DataMgr.h"

#include <iomanip>

namespace hdk {

namespace {

class ResultSetDataToken : public Data_Namespace::AbstractDataToken {
 public:
  ResultSetDataToken(ResultSetPtr rs, const int8_t* buf, size_t size)
      : rs_(std::move(rs)), buf_(buf), size_(size) {}
  ~ResultSetDataToken() override {}

  const int8_t* getMemoryPtr() const override { return buf_; }
  size_t getSize() const override { return size_; }

 private:
  ResultSetPtr rs_;
  const int8_t* buf_;
  size_t size_;
};

TableFragmentsInfo getEmptyTableMetadata(int table_id) {
  TableFragmentsInfo res;
  res.setPhysicalNumTuples(0);

  // Executor requires dummy empty fragment for empty tables
  FragmentInfo& empty_frag = res.fragments.emplace_back();
  empty_frag.fragmentId = 0;
  empty_frag.shadowNumTuples = 0;
  empty_frag.setPhysicalNumTuples(0);
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
  empty_frag.physicalTableId = table_id;
  res.fragments.push_back(empty_frag);

  return res;
}

}  // namespace

ResultSetRegistry::ResultSetRegistry(ConfigPtr config,
                                     const std::string& schema_name,
                                     int db_id)
    : SimpleSchemaProvider(hdk::ir::Context::defaultCtx(),
                           getSchemaId(db_id),
                           schema_name)
    , db_id_(db_id)
    , schema_id_(getSchemaId(db_id))
    , config_(config) {}

std::shared_ptr<ResultSetRegistry> ResultSetRegistry::getOrCreate(
    Data_Namespace::DataMgr* data_mgr,
    ConfigPtr config) {
  static mapd_shared_mutex res_registry_init_mutex;
  auto* ps_mgr = data_mgr->getPersistentStorageMgr();

  {
    mapd_shared_lock<mapd_shared_mutex> check_init_lock(res_registry_init_mutex);
    if (!ps_mgr->hasDataProvider(hdk::ResultSetRegistry::SCHEMA_ID)) {
      check_init_lock.unlock();
      mapd_unique_lock<mapd_shared_mutex> init_lock(res_registry_init_mutex);
      if (!ps_mgr->hasDataProvider(hdk::ResultSetRegistry::SCHEMA_ID)) {
        ps_mgr->registerDataProvider(hdk::ResultSetRegistry::SCHEMA_ID,
                                     std::make_shared<hdk::ResultSetRegistry>(config));
      }
    }
  }

  auto provider = ps_mgr->getDataProvider(hdk::ResultSetRegistry::SCHEMA_ID);
  auto res = std::dynamic_pointer_cast<hdk::ResultSetRegistry>(provider);
  CHECK(res);
  return res;
}

ResultSetTableTokenPtr ResultSetRegistry::put(ResultSetTable table) {
  CHECK(!table.empty());

  mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);

  auto table_id = next_table_id_++;
  // Add schema information for the ResultSet.
  auto tinfo = addTableInfo(db_id_,
                            table_id,
                            ResultSetTableToken::tableName(table_id),
                            false,
                            Data_Namespace::MemoryLevel::CPU_LEVEL,
                            table.size());
  auto& first_rs = table.result(0);
  for (size_t col_idx = 0; col_idx < first_rs->colCount(); ++col_idx) {
    addColumnInfo(db_id_,
                  table_id,
                  static_cast<int>(col_idx + 1),
                  first_rs->colName(col_idx),
                  first_rs->colType(col_idx),
                  false);
  }

  auto table_data = std::make_unique<TableData>();
  size_t row_count = 0;
  for (auto& rs : table.results()) {
    DataFragment frag;
    frag.offset = row_count;
    frag.row_count = rs->rowCount();
    frag.rs = rs;
    if (useColumnarResults(*rs)) {
      frag.mutex = std::make_unique<std::mutex>();
    }

    row_count += frag.row_count;
    table_data->fragments.emplace_back(std::move(frag));
  }
  table_data->row_count = row_count;

  tables_[table_id] = std::move(table_data);

  return std::make_shared<ResultSetTableToken>(tinfo, row_count, shared_from_this());
}

ResultSetPtr ResultSetRegistry::get(const ResultSetTableToken& token, size_t idx) const {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK(tables_.count(token.tableId()));
  auto* table = tables_.at(token.tableId()).get();
  mapd_shared_lock<mapd_shared_mutex> table_lock(table->mutex);
  CHECK_LT(idx, table->fragments.size());
  return table->fragments[idx].rs;
}

void ResultSetRegistry::drop(const ResultSetTableToken& token) {
  mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);

  CHECK(tables_.count(token.tableId()));
  std::unique_ptr<TableData> table = std::move(tables_.at(token.tableId()));
  mapd_unique_lock<mapd_shared_mutex> table_lock(table->mutex);
  tables_.erase(token.tableId());

  SimpleSchemaProvider::dropTable(token.dbId(), token.tableId());
}

void ResultSetRegistry::fetchBuffer(const ChunkKey& key,
                                    Data_Namespace::AbstractBuffer* dest,
                                    const size_t num_bytes) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  size_t col_idx = static_cast<size_t>(key[CHUNK_KEY_COLUMN_IDX] - 1);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  auto& rs = table.fragments[frag_idx].rs;
  dest->reserve(num_bytes);

  CHECK(!useColumnarResults(*rs));
  rs->copyColumnIntoBuffer(col_idx, dest->getMemoryPtr(), num_bytes);
}

std::unique_ptr<Data_Namespace::AbstractDataToken>
ResultSetRegistry::getZeroCopyBufferMemory(const ChunkKey& key, size_t num_bytes) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  size_t col_idx = static_cast<size_t>(key[CHUNK_KEY_COLUMN_IDX] - 1);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  auto& frag = table.fragments[frag_idx];
  const int8_t* buf = nullptr;

  // When ColumnarResults is used, we pretend it is a zero-copy fetch
  // because we will have all required buffers already allocated.
  // TODO: To avoid having data cached here we don't need anymore, we should
  // clean-up tokens we are not going to use in TemporaryTables of
  // RelAlgExecutor.
  if (useColumnarResults(*frag.rs)) {
    if (frag.columnar_res) {
      buf = frag.columnar_res->getColumnBuffers()[col_idx];
    } else {
      CHECK(frag.mutex);
      std::lock_guard<std::mutex> columnaraize_lock(*frag.mutex);
      if (!frag.columnar_res) {
        std::vector<const hdk::ir::Type*> col_types;
        for (size_t i = 0; i < frag.rs->colCount(); ++i) {
          col_types.push_back(frag.rs->colType(i)->canonicalize());
        }
        frag.columnar_res =
            std::make_unique<ColumnarResults>(frag.rs->getRowSetMemOwner(),
                                              *frag.rs,
                                              frag.rs->colCount(),
                                              col_types,
                                              0,
                                              *config_);
      }
      buf = frag.columnar_res->getColumnBuffers()[col_idx];
    }
  } else if (frag.rs->isZeroCopyColumnarConversionPossible(col_idx)) {
    buf = frag.rs->getColumnarBuffer(col_idx);
  }

  return buf ? std::make_unique<ResultSetDataToken>(frag.rs, buf, num_bytes) : nullptr;
}

bool ResultSetRegistry::useColumnarResults(const ResultSet& rs) const {
  // We use ColumnarResults for all cases except those when can use
  // zero-copy fetch or copyColumnIntoBuffer for all columns.
  // It would be better to convert comlumns more lazily especially when
  // columnar format is used and multiple columns might be fetched in
  // parallel.
  return !rs.isDirectColumnarConversionPossible() ||
         rs.getQueryDescriptionType() != QueryDescriptionType::Projection ||
         rs.areAnyColumnsLazyFetched();
}

TableFragmentsInfo ResultSetRegistry::getTableMetadata(int db_id, int table_id) const {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(db_id, db_id_);
  CHECK_EQ(tables_.count(table_id), (size_t)1);
  auto& table = *tables_.at(table_id);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  if (table.fragments.empty()) {
    return getEmptyTableMetadata(table_id);
  }

  TableFragmentsInfo res;
  res.setPhysicalNumTuples(table.row_count);
  for (size_t frag_idx = 0; frag_idx < table.fragments.size(); ++frag_idx) {
    auto& frag = table.fragments[frag_idx];
    auto& frag_info = res.fragments.emplace_back();
    frag_info.fragmentId = static_cast<int>(frag_idx + 1);
    frag_info.physicalTableId = table_id;
    frag_info.setPhysicalNumTuples(frag.row_count);
    frag_info.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
    frag_info.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
    frag_info.resultSet = frag.rs.get();
    frag_info.resultSetMutex = std::make_shared<std::mutex>();
  }

  return res;
}

const DictDescriptor* ResultSetRegistry::getDictMetadata(int dict_id, bool load_dict) {
  // Currently, we don't hold any dictionaries in the registry.
  UNREACHABLE();
  return nullptr;
}

}  // namespace hdk
