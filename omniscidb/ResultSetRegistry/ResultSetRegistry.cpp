/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ResultSetRegistry.h"
#include "ResultSetMetadata.h"

#include "DataMgr/DataMgr.h"

#include <iomanip>

namespace hdk {

namespace {

class ResultSetDataToken : public Data_Namespace::AbstractDataToken {
 public:
  ResultSetDataToken(ResultSetPtr rs,
                     const hdk::ir::Type* type,
                     const int8_t* buf,
                     size_t size)
      : rs_(std::move(rs)), type_(type), buf_(buf), size_(size) {}
  ~ResultSetDataToken() override {}

  const int8_t* getMemoryPtr() const override { return buf_; }
  size_t getSize() const override { return size_; }
  const hdk::ir::Type* getType() const override { return type_; }

 private:
  ResultSetPtr rs_;
  const hdk::ir::Type* type_;
  const int8_t* buf_;
  size_t size_;
};

TableFragmentsInfo getEmptyTableMetadata(int table_id) {
  TableFragmentsInfo res;
  res.setPhysicalNumTuples(0);

  // Executor requires dummy empty fragment for empty tables
  FragmentInfo& empty_frag = res.fragments.emplace_back();
  empty_frag.fragmentId = 0;
  empty_frag.setPhysicalNumTuples(0);
  // Add ids for DISK_LEVEL, CPU_LEVEL, and GPU_LEVEL
  empty_frag.deviceIds.resize(3, 0);
  empty_frag.physicalTableId = table_id;
  res.fragments.push_back(empty_frag);

  return res;
}

int columnId(size_t col_idx) {
  return ResultSetTableToken::columnId(col_idx);
}

size_t columnIndex(int col_id) {
  return ResultSetTableToken::columnIndex(col_id);
}

}  // namespace

ResultSetRegistry::ResultSetRegistry(ConfigPtr config)
    : ResultSetRegistry(config, "rs_registry") {}

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
  auto table_name = std::string("__result_set_") + std::to_string(table_id);
  // Add schema information for the ResultSet.
  auto tinfo = addTableInfo(db_id_, table_id, table_name, false, table.size());
  auto& first_rs = table.result(0);
  bool has_varlen = false;
  bool has_array = false;
  for (size_t col_idx = 0; col_idx < first_rs->colCount(); ++col_idx) {
    addColumnInfo(db_id_,
                  table_id,
                  columnId(col_idx),
                  first_rs->colName(col_idx),
                  first_rs->colType(col_idx),
                  false);
    has_varlen = has_varlen || first_rs->colType(col_idx)->isVarLen();
    has_array = has_array || first_rs->colType(col_idx)->isArray();
  }
  addRowidColumn(db_id_, table_id, columnId(first_rs->colCount()));

  // TODO: lazily compute row count and try to avoid global write
  // locks for Date
  auto table_data = std::make_unique<TableData>();
  size_t row_count = 0;
  for (auto& rs : table.results()) {
    DataFragment frag;
    frag.offset = row_count;
    frag.row_count = rs->rowCount();
    frag.rs = rs;
    frag.mutex = std::make_unique<mapd_shared_mutex>();

    row_count += frag.row_count;
    table_data->fragments.emplace_back(std::move(frag));
  }
  table_data->row_count = row_count;
  table_data->has_varlen_col = has_varlen;
  // We use ColumnarResults for all cases except those when can use
  // zero-copy fetch or copyColumnIntoBuffer for all columns.
  // It would be better to convert comlumns more lazily especially when
  // columnar format is used and multiple columns might be fetched in
  // parallel.
  table_data->use_columnar_res =
      !first_rs->isDirectColumnarConversionPossible() ||
      first_rs->getQueryDescriptionType() != QueryDescriptionType::Projection ||
      first_rs->areAnyColumnsLazyFetched() || has_varlen || has_array;

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

ChunkStats ResultSetRegistry::getChunkStats(int table_id,
                                            size_t frag_idx,
                                            size_t col_idx) const {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK(tables_.count(table_id));
  auto& table = *tables_.at(table_id);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  CHECK_LT(frag_idx, table.fragments.size());
  auto& frag = table.fragments[frag_idx];
  mapd_shared_lock<mapd_shared_mutex> frag_read_lock(*frag.mutex);

  if (frag.meta.empty()) {
    frag_read_lock.unlock();
    mapd_unique_lock<mapd_shared_mutex> frag_write_lock(*frag.mutex);
    if (frag.meta.empty()) {
      frag.meta = synthesizeMetadata(frag.rs.get());
    }
  }
  CHECK(frag.meta.count(columnId(col_idx)));
  return frag.meta.at(columnId(col_idx))->chunkStats();
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

  size_t col_idx = columnIndex(key[CHUNK_KEY_COLUMN_IDX]);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  auto& rs = table.fragments[frag_idx].rs;
  dest->reserve(num_bytes);

  CHECK(!table.use_columnar_res);
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

  size_t col_idx = columnIndex(key[CHUNK_KEY_COLUMN_IDX]);
  size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
  CHECK_LT(frag_idx, table.fragments.size());
  auto& frag = table.fragments[frag_idx];
  const int8_t* buf = nullptr;

  // When ColumnarResults is used, we pretend it is a zero-copy fetch
  // because we will have all required buffers already allocated.
  // TODO: To avoid having data cached here we don't need anymore, we should
  // clean-up tokens we are not going to use in TemporaryTables of
  // RelAlgExecutor.
  if (table.use_columnar_res) {
    mapd_shared_lock<mapd_shared_mutex> frag_read_lock(*frag.mutex);
    if (frag.columnar_res) {
      if (key.size() < 5 || key[CHUNK_KEY_VARLEN_IDX] == 1) {
        buf = frag.columnar_res->getColumnBuffers()[col_idx];
      } else {
        buf = frag.columnar_res->getOffsetBuffers()[col_idx];
      }
    } else {
      frag_read_lock.unlock();
      mapd_unique_lock<mapd_shared_mutex> frag_write_lock(*frag.mutex);
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
      if (key.size() < 5 || key[CHUNK_KEY_VARLEN_IDX] == 1) {
        buf = frag.columnar_res->getColumnBuffers()[col_idx];
      } else {
        buf = frag.columnar_res->getOffsetBuffers()[col_idx];
      }
    }
  } else if (frag.rs->isZeroCopyColumnarConversionPossible(col_idx)) {
    CHECK_EQ(key.size(), (size_t)4);
    buf = frag.rs->getColumnarBuffer(col_idx);
  }

  return buf ? std::make_unique<ResultSetDataToken>(
                   frag.rs, frag.rs->colType(col_idx), buf, num_bytes)
             : nullptr;
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
    // Add ids for DISK_LEVEL, CPU_LEVEL, and GPU_LEVEL
    frag_info.deviceIds.resize(3, 0);
    mapd_shared_lock<mapd_shared_mutex> frag_lock(*frag.mutex);
    if (frag.meta.empty()) {
      // For now, we don't have lazy fragment size computation. For varlen columns
      // it means we should compute real fragment size right now. Do it through
      // stats materialization.
      if (table.has_varlen_col) {
        frag_lock.unlock();
        getChunkStats(table_id, frag_idx, 0);
        frag_lock.lock();
        CHECK(!frag.meta.empty());
        frag_info.setChunkMetadataMap(frag.meta);
      } else {
        for (size_t col_idx = 0; col_idx < (size_t)frag.rs->colCount(); ++col_idx) {
          auto col_type = frag.rs->colType(col_idx);
          auto meta = std::make_shared<ChunkMetadata>(
              col_type,
              frag.rs->rowCount() * col_type->size(),
              frag.rs->rowCount(),
              [this, table_id, frag_idx, col_idx](ChunkStats& stats) {
                stats = this->getChunkStats(table_id, frag_idx, col_idx);
              });
          frag_info.setChunkMetadata(columnId(col_idx), meta);
        }
      }
    } else {
      frag_info.setChunkMetadataMap(frag.meta);
    }
  }

  return res;
}

const DictDescriptor* ResultSetRegistry::getDictMetadata(int dict_id, bool load_dict) {
  // Currently, we don't hold any dictionaries in the registry.
  UNREACHABLE();
  return nullptr;
}

}  // namespace hdk
