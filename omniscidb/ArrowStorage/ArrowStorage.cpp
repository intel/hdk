/*
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

#include "ArrowStorage.h"
#include "ArrowStorageUtils.h"

#include "IR/Type.h"
#include "Shared/ArrowUtil.h"
#include "Shared/measure.h"
#include "Shared/threading.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <arrow/csv/reader.h>
#include <arrow/io/api.h>
#include <arrow/json/reader.h>
#include <arrow/util/decimal.h>
#include <arrow/util/value_parsing.h>
#include <parquet/api/reader.h>
#include <parquet/arrow/reader.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace std::string_literals;

namespace {

size_t computeTotalStringsLength(std::shared_ptr<arrow::ChunkedArray> arr,
                                 size_t offset,
                                 size_t rows) {
  size_t start_offset = offset;
  size_t chunk_no = 0;
  while (static_cast<size_t>(arr->chunk(chunk_no)->length()) <= start_offset) {
    start_offset -= arr->chunk(chunk_no)->length();
    ++chunk_no;
  }

  size_t rows_remain = rows;
  size_t total_bytes = 0;
  while (rows_remain) {
    auto chunk = arr->chunk(chunk_no);
    size_t rows_in_chunk = std::min(rows_remain, chunk->length() - start_offset);
    const int32_t* offsets = chunk->data()->GetValues<int32_t>(1);
    total_bytes +=
        std::abs(offsets[start_offset + rows_in_chunk]) - std::abs(offsets[start_offset]);
    rows_remain -= rows_in_chunk;
    start_offset = 0;
    ++chunk_no;
  }

  return total_bytes;
}

/**
 * Get column ID by its 0-based index (position) in the table.
 */
int columnId(size_t col_idx) {
  return static_cast<int>(col_idx + 1000);
}

/**
 * Translate column ID to 0-based index (position) in the table.
 */
size_t columnIndex(int col_id) {
  return static_cast<size_t>(col_id - 1000);
}

}  // anonymous namespace

void ArrowStorage::fetchBuffer(const ChunkKey& key,
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
  CHECK_LT(col_idx, table.col_data.size());

  auto col_type =
      getColumnInfo(
          key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX], key[CHUNK_KEY_COLUMN_IDX])
          ->type;
  dest->reserve(num_bytes);
  if (!col_type->isVarLen()) {
    CHECK_EQ(key.size(), (size_t)4);
    size_t elem_size = col_type->size();
    fetchFixedLenData(table, frag_idx, col_idx, dest, num_bytes, elem_size);
  } else {
    CHECK_EQ(key.size(), (size_t)5);
    if (key[CHUNK_KEY_VARLEN_IDX] == 1) {
      if (!dest->hasEncoder()) {
        dest->initEncoder(col_type);
      }
      if (col_type->isString()) {
        fetchVarLenData(table, frag_idx, col_idx, dest, num_bytes);
      } else {
        CHECK(col_type->isVarLenArray());
        fetchVarLenArrayData(table,
                             frag_idx,
                             col_idx,
                             dest,
                             col_type->as<hdk::ir::ArrayBaseType>()->elemType()->size(),
                             num_bytes);
      }
    } else {
      CHECK_EQ(key[CHUNK_KEY_VARLEN_IDX], 2);
      fetchVarLenOffsets(table, frag_idx, col_idx, dest, num_bytes);
    }
  }
  dest->setSize(num_bytes);
}

std::unique_ptr<AbstractDataToken> ArrowStorage::getZeroCopyBufferMemory(
    const ChunkKey& key,
    size_t num_bytes) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  CHECK_EQ(key[CHUNK_KEY_DB_IDX], db_id_);
  CHECK_EQ(tables_.count(key[CHUNK_KEY_TABLE_IDX]), (size_t)1);
  auto& table = *tables_.at(key[CHUNK_KEY_TABLE_IDX]);
  mapd_shared_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  auto col_type =
      getColumnInfo(
          key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX], key[CHUNK_KEY_COLUMN_IDX])
          ->type;

  if (col_type->isExtDictionary()) {
    auto dict_id = col_type->as<hdk::ir::ExtDictionaryType>()->dictId();
    auto dict_descriptor = getDictMetadata(
        dict_id);  // this will force materialize the dictionary. it is thread safe
    CHECK(dict_descriptor);
  }

  if (!col_type->isVarLen()) {
    size_t col_idx = columnIndex(key[CHUNK_KEY_COLUMN_IDX]);
    size_t frag_idx = static_cast<size_t>(key[CHUNK_KEY_FRAGMENT_IDX] - 1);
    CHECK_EQ(key.size(), (size_t)4);
    size_t elem_size = col_type->size();
    auto& frag = table.fragments[frag_idx];
    size_t rows_to_fetch = num_bytes ? num_bytes / elem_size : frag.row_count;
    const auto* fixed_type =
        dynamic_cast<const arrow::FixedWidthType*>(table.col_data[col_idx]->type().get());
    CHECK(fixed_type) << table.col_data[col_idx]->type()->ToString() << " (table "
                      << key[CHUNK_KEY_TABLE_IDX] << ", column " << col_idx << ")";
    size_t arrow_elem_size = fixed_type->bit_width() / 8;
    // For fixed size arrays we simply use elem type in arrow and therefore have to scale
    // to get a proper slice.
    size_t elems = elem_size / arrow_elem_size;
    CHECK_GT(elems, (size_t)0);
    auto data_to_fetch =
        table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset * elems),
                                       static_cast<int64_t>(rows_to_fetch * elems));
    if (data_to_fetch->num_chunks() == 1) {
      auto chunk = data_to_fetch->chunk(0);
      const int8_t* ptr =
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * arrow_elem_size);
      size_t chunk_size = chunk->length() * arrow_elem_size;
      return std::make_unique<ArrowChunkDataToken>(
          std::move(chunk), col_type, ptr, chunk_size);
    }
  }

  return nullptr;
}

void ArrowStorage::fetchFixedLenData(const TableData& table,
                                     size_t frag_idx,
                                     size_t col_idx,
                                     Data_Namespace::AbstractBuffer* dest,
                                     size_t num_bytes,
                                     size_t elem_size) const {
  auto& frag = table.fragments[frag_idx];
  size_t rows_to_fetch = num_bytes ? num_bytes / elem_size : frag.row_count;
  const auto* fixed_type =
      dynamic_cast<const arrow::FixedWidthType*>(table.col_data[col_idx]->type().get());
  CHECK(fixed_type);
  size_t arrow_elem_size = fixed_type->bit_width() / 8;
  // For fixed size arrays we simply use elem type in arrow and therefore have to scale
  // to get a proper slice.
  size_t elems = elem_size / arrow_elem_size;
  CHECK_GT(elems, (size_t)0);
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset * elems),
                                     static_cast<int64_t>(rows_to_fetch * elems));
  int8_t* dst_ptr = dest->getMemoryPtr();
  for (auto& chunk : data_to_fetch->chunks()) {
    size_t chunk_size = chunk->length() * arrow_elem_size;
    const int8_t* src_ptr =
        chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * arrow_elem_size);
    memcpy(dst_ptr, src_ptr, chunk_size);
    dst_ptr += chunk_size;
  }
}

void ArrowStorage::fetchVarLenOffsets(const TableData& table,
                                      size_t frag_idx,
                                      size_t col_idx,
                                      Data_Namespace::AbstractBuffer* dest,
                                      size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  CHECK_EQ(num_bytes, (frag.row_count + 1) * sizeof(uint32_t));
  // Number of fetched offsets is 1 greater than number of fetched rows.
  size_t rows_to_fetch = num_bytes ? num_bytes / sizeof(uint32_t) - 1 : frag.row_count;
  auto data_to_fetch = table.col_data[col_idx]->Slice(
      static_cast<int64_t>(frag.offset), static_cast<int64_t>(rows_to_fetch));
  uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dest->getMemoryPtr());
  uint32_t delta = 0;
  for (auto& chunk : data_to_fetch->chunks()) {
    const uint32_t* src_ptr = chunk->data()->GetValues<uint32_t>(1);
    delta -= *src_ptr;
    dst_ptr = std::transform(
        src_ptr, src_ptr + chunk->length(), dst_ptr, [delta](uint32_t val) {
          return val + delta;
        });
    delta += src_ptr[chunk->length()];
  }
  *dst_ptr = delta;
}

void ArrowStorage::fetchVarLenData(const TableData& table,
                                   size_t frag_idx,
                                   size_t col_idx,
                                   Data_Namespace::AbstractBuffer* dest,
                                   size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset), frag.row_count);
  int8_t* dst_ptr = dest->getMemoryPtr();
  size_t remained = num_bytes;
  for (auto& chunk : data_to_fetch->chunks()) {
    if (remained == 0) {
      break;
    }

    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    size_t chunk_size = offsets[chunk->length()] - offsets[0];
    chunk_size = std::min(chunk_size, num_bytes);
    memcpy(dst_ptr, chunk->data()->GetValues<int8_t>(2, offsets[0]), chunk_size);
    remained -= chunk_size;
    dst_ptr += chunk_size;
  }
}

void ArrowStorage::fetchVarLenArrayData(const TableData& table,
                                        size_t frag_idx,
                                        size_t col_idx,
                                        Data_Namespace::AbstractBuffer* dest,
                                        size_t elem_size,
                                        size_t num_bytes) const {
  auto& frag = table.fragments[frag_idx];
  auto data_to_fetch =
      table.col_data[col_idx]->Slice(static_cast<int64_t>(frag.offset), frag.row_count);
  int8_t* dst_ptr = dest->getMemoryPtr();
  size_t remained = num_bytes;
  for (auto& chunk : data_to_fetch->chunks()) {
    if (remained == 0) {
      break;
    }

    const uint32_t* offsets = chunk->data()->GetValues<uint32_t>(1);
    size_t chunk_size = offsets[chunk->length()] - offsets[0];
    chunk_size = std::min(chunk_size, num_bytes);
    auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
    auto elem_array = chunk_list->values();
    memcpy(dst_ptr, elem_array->data()->GetValues<int8_t>(1, offsets[0]), chunk_size);
    remained -= chunk_size;
    dst_ptr += chunk_size;
  }
}

TableFragmentsInfo ArrowStorage::getTableMetadata(int db_id, int table_id) const {
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
    for (size_t col_idx = 0; col_idx < frag.metadata.size(); ++col_idx) {
      frag_info.setChunkMetadata(columnId(col_idx), frag.metadata[col_idx]);
    }
  }
  return res;
}

TableFragmentsInfo ArrowStorage::getEmptyTableMetadata(int table_id) const {
  TableFragmentsInfo res;
  res.setPhysicalNumTuples(0);

  // Executor requires dummy empty fragment for empty tables
  FragmentInfo& empty_frag = res.fragments.emplace_back();
  empty_frag.fragmentId = 0;
  empty_frag.setPhysicalNumTuples(0);
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::DISK_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::CPU_LEVEL
  empty_frag.deviceIds.push_back(0);  // Data_Namespace::GPU_LEVEL
  empty_frag.physicalTableId = table_id;
  res.fragments.push_back(empty_frag);

  return res;
}

const DictDescriptor* ArrowStorage::getDictMetadata(int dict_id, bool load_dict) {
  mapd_shared_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  CHECK_EQ(getSchemaId(dict_id), schema_id_);
  if (dicts_.count(dict_id)) {
    auto dict = dicts_.at(dict_id).get();
    CHECK(dict);
    if (!dict->is_materialized && load_dict) {
      std::lock_guard<std::mutex> materialization_lock(dict->mutex);
      // after we get the lock, check the atomic again
      if (!dict->is_materialized) {
        materializeDictionary(dict);
      }
    }
    return dict->dict_descriptor.get();
  }
  return nullptr;
}

void ArrowStorage::materializeDictionary(DictionaryData* dict) {
  CHECK(dict);
  CHECK(!dict->table_ids.empty());

  const auto dict_desc = dict->dict_descriptor.get();
  CHECK(dict_desc);
  auto* string_dict = dict_desc->stringDict.get();
  CHECK(string_dict);
  const int elem_size = dict_desc->dictNBits / 8;

  for (const auto table_id : dict->table_ids) {
    mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
    auto& table = *tables_.at(table_id);
    data_lock.unlock();

    // TODO: should we add the data lock here?
    mapd_unique_lock<mapd_shared_mutex> table_lock(table.mutex);

    if (table.row_count == 0) {
      // skip empty tables
      continue;
    }
    auto col_ids = dict->table_ids_to_column_ids.at(table_id);
    CHECK(!col_ids.empty());

    for (const auto col_id : col_ids) {
      CHECK_LT(col_id, int(table.col_data.size()))
          << "(" << table_id << ", " << col_id << ")";
      auto col_data = table.col_data[col_id];
      CHECK(col_data);

      if (col_data->type() != arrow::utf8()) {
        VLOG(1) << "Skipping dictionary materialization for already materialized column "
                << col_id << " in table " << table_id;
        continue;
      }

      arrow::ArrayVector col_indices_data;
      const auto& chunks = col_data->chunks();
      for (const auto& crt_chunk : chunks) {
        auto col_strings =
            std::static_pointer_cast<arrow::StringArray>(crt_chunk->Slice(0));
        CHECK(col_strings);

        const size_t bulk_size = static_cast<size_t>(col_strings->length());

        // dictionary conversion
        std::vector<std::string_view> bulk(bulk_size);
        for (size_t j = 0; j < bulk_size; j++) {
          if (!col_strings->IsNull(j)) {
            auto view = col_strings->GetView(j);
            bulk[j] = std::string_view(view.data(), view.length());
          }
        }

        std::shared_ptr<arrow::Array> indices_chunk;
        if (elem_size == 4) {
          // not an encoded dictionary
          std::shared_ptr<arrow::Buffer> indices_buf;
          auto res = arrow::AllocateBuffer(bulk_size * 4);
          CHECK(res.ok());
          indices_buf = std::move(res).ValueOrDie();
          auto raw_data = reinterpret_cast<int32_t*>(indices_buf->mutable_data());
          string_dict->getOrAddBulk(bulk, raw_data);
          indices_chunk = std::make_shared<arrow::Int32Array>(bulk_size, indices_buf);
        } else {
          // encoded
          std::vector<int32_t> indices_buffer(bulk_size);
          string_dict->getOrAddBulk(bulk, indices_buffer.data());

          // create arrow buffer of encoded size and copy into it
          std::shared_ptr<arrow::Buffer> encoded_indices_buf;
          auto res = arrow::AllocateBuffer(bulk_size * elem_size);
          CHECK(res.ok());
          encoded_indices_buf = std::move(res).ValueOrDie();
          switch (elem_size) {
            case 1: {
              auto encoded_indices_buf_ptr =
                  reinterpret_cast<int8_t*>(encoded_indices_buf->mutable_data());
              CHECK(encoded_indices_buf_ptr);
              for (size_t i = 0; i < bulk_size; i++) {
                encoded_indices_buf_ptr[i] =
                    indices_buffer[i] == std::numeric_limits<int32_t>::min() ||
                            indices_buffer[i] > std::numeric_limits<uint8_t>::max()
                        ? std::numeric_limits<uint8_t>::max()
                        : indices_buffer[i];
              }
              indices_chunk =
                  std::make_shared<arrow::Int8Array>(bulk_size, encoded_indices_buf);
              break;
            }
            case 2: {
              auto encoded_indices_buf_ptr =
                  reinterpret_cast<int16_t*>(encoded_indices_buf->mutable_data());
              CHECK(encoded_indices_buf_ptr);
              for (size_t i = 0; i < bulk_size; i++) {
                encoded_indices_buf_ptr[i] =
                    indices_buffer[i] == std::numeric_limits<int32_t>::min() ||
                            indices_buffer[i] > std::numeric_limits<uint16_t>::max()
                        ? std::numeric_limits<uint16_t>::max()
                        : indices_buffer[i];
              }
              indices_chunk =
                  std::make_shared<arrow::Int16Array>(bulk_size, encoded_indices_buf);
              break;
              break;
            }
            default:
              LOG(FATAL) << "Unrecognized element size " << elem_size;
          }
        }
        CHECK(indices_chunk);
        col_indices_data.push_back(indices_chunk);
      }

      CHECK_EQ(col_indices_data.size(), chunks.size());
      auto new_col_data = arrow::ChunkedArray::Make(col_indices_data).ValueOrDie();

      VLOG(1) << "Materialized string dictionary for column " << col_id << " in table "
              << table_id;
      for (auto& frag : table.fragments) {
        CHECK_LT(col_id, frag.metadata.size());
        auto& meta = frag.metadata[col_id];
        // compute chunk stats is multi threaded, so we single thread this
        meta->fillChunkStats(
            computeStats(new_col_data->Slice(frag.offset, frag.row_count), dict->type));
      }

      table.col_data[col_id] = new_col_data;
    }  // per column
  }    // per table
  dict->is_materialized = true;
}

TableInfoPtr ArrowStorage::createTable(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) {
  TableInfoPtr res;
  int table_id;
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);
  size_t next_col_idx = 0;
  {
    mapd_unique_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
    mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);
    table_id = next_table_id_++;
    checkNewTableParams(table_name, columns, options);
    res = addTableInfo(db_id_, table_id, table_name, false, 0, 0);
    std::unordered_map<int, int> dict_ids;
    for (auto& col : columns) {
      auto type = col.type;
      auto elem_type =
          type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
      // Positive dictionary id means we use existing dictionary. Other values
      // mean we have to create new dictionaries. Columns with equal negative
      // dict ids will share dictionaries.
      if (elem_type->isExtDictionary()) {
        auto dict_type = elem_type->as<hdk::ir::ExtDictionaryType>();
        auto sharing_id = dict_type->dictId();
        if (sharing_id < 0 && dict_ids.count(sharing_id)) {
          const auto dict_id = dict_ids.at(sharing_id);
          elem_type = ctx_.extDict(dict_type->elemType(), dict_id, dict_type->size());
          auto* dict_data = dicts_.at(dict_id).get();
          dict_data->addTableColumnPair(table_id, next_col_idx);
        } else if (sharing_id <= 0) {
          if (next_dict_id_ > MAX_DB_ID) {
            throw std::runtime_error("Dictionary count limit exceeded.");
          }

          int dict_id = addSchemaIdChecked(next_dict_id_++, schema_id_);
          auto dict_desc =
              std::make_unique<DictDescriptor>(db_id_,
                                               dict_id,
                                               col.name,
                                               /*nbits=*/dict_type->size() * 8,
                                               /*is_shared=*/true,
                                               /*refcount=*/1,
                                               table_name,
                                               /*temp=*/true);
          dict_desc->stringDict =
              std::make_shared<StringDictionary>(DictRef{db_id_, dict_id});
          if (sharing_id < 0) {
            dict_ids.emplace(sharing_id, dict_id);
          }
          if (dicts_.find(dict_id) == dicts_.end()) {
            auto dict_data_owned = std::make_unique<DictionaryData>(
                std::move(dict_desc),
                dict_type,
                config_->storage.enable_lazy_dict_materialization);
            CHECK(dicts_.emplace(dict_id, std::move(dict_data_owned)).second);
          }
          auto* dict_data = dicts_.at(dict_id).get();
          dict_data->addTableColumnPair(table_id, next_col_idx);
          elem_type = ctx_.extDict(dict_type->elemType(), dict_id, dict_type->size());
        } else {
          CHECK_GT(sharing_id, 0);
          auto* dict_data = dicts_.at(sharing_id).get();
          dict_data->addTableColumnPair(table_id, next_col_idx);
        }

        if (type->isFixedLenArray()) {
          type = ctx_.arrayFixed(type->as<hdk::ir::FixedLenArrayType>()->numElems(),
                                 elem_type,
                                 type->nullable());
        } else if (type->isVarLenArray()) {
          type = ctx_.arrayVarLen(elem_type,
                                  type->as<hdk::ir::VarLenArrayType>()->offsetSize(),
                                  type->nullable());
        } else {
          type = elem_type;
        }
      }
      auto col_info = addColumnInfo(
          db_id_, table_id, columnId(next_col_idx++), col.name, type, false);
    }
    addRowidColumn(db_id_, table_id, columnId(next_col_idx++));
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  fields.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    auto& name = columns[i].name;
    auto& type = columns[i].type;
    auto field = arrow::field(name, getArrowImportType(ctx_, type), type->nullable());
    fields.push_back(field);
  }
  auto schema = arrow::schema(fields);

  {
    auto [iter, inserted] = tables_.emplace(table_id, std::make_unique<TableData>());
    CHECK(inserted);
    auto& table = *iter->second;
    table.fragment_size = options.fragment_size;
    table.schema = schema;
  }

  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const std::vector<ColumnDescription>& columns,
                                            const TableOptions& options) {
  auto res = createTable(table_name, columns, options);
  appendArrowTable(at, table_name);
  return res;
}

TableInfoPtr ArrowStorage::importArrowTable(std::shared_ptr<arrow::Table> at,
                                            const std::string& table_name,
                                            const TableOptions& options) {
  std::vector<ColumnDescription> columns;
  for (auto& field : at->schema()->fields()) {
    ColumnDescription desc{field->name(), getTargetImportType(ctx_, *field->type())};
    columns.emplace_back(std::move(desc));
  }
  return importArrowTable(at, table_name, columns, options);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at,
                                    const std::string& table_name) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendArrowTable(at, tinfo->table_id);
}

void ArrowStorage::appendArrowTable(std::shared_ptr<arrow::Table> at, int table_id) {
  mapd_shared_lock<mapd_shared_mutex> data_lock(data_mutex_);
  if (!tables_.count(table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto& table = *tables_.at(table_id);
  compareSchemas(table.schema, at->schema());

  mapd_unique_lock<mapd_shared_mutex> table_lock(table.mutex);
  data_lock.unlock();

  std::vector<std::shared_ptr<arrow::ChunkedArray>> col_data;
  col_data.resize(at->columns().size());

  std::vector<DataFragment> fragments;
  // Compute size of the fragment. If the last existing fragment is not full, then it will
  // be merged with the first new fragment.
  size_t first_frag_size =
      std::min(table.fragment_size, static_cast<size_t>(at->num_rows()));
  if (!table.fragments.empty()) {
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      first_frag_size =
          std::min(first_frag_size, table.fragment_size - last_frag.row_count);
    }
  }
  // Now we can compute number of fragments to create.
  size_t frag_count =
      (static_cast<size_t>(at->num_rows()) + table.fragment_size - 1 - first_frag_size) /
          table.fragment_size +
      1;
  fragments.resize(frag_count);
  for (auto& frag : fragments) {
    frag.metadata.resize(at->columns().size());
  }

  mapd_shared_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  std::vector<bool> lazy_fetch_cols(at->columns().size(), false);
  if (config_->storage.enable_lazy_dict_materialization) {
    VLOG(1) << "Appending arrow table with lazy dictionary materialization enabled";
    for (size_t col_idx = 0; col_idx < at->columns().size(); col_idx++) {
      auto col_info = getColumnInfo(db_id_, table_id, columnId(col_idx));
      CHECK(col_info);
      auto col_type = col_info->type;
      auto col_arr = at->column(col_idx);
      if (col_type->isExtDictionary() && col_arr->type()->id() == arrow::Type::STRING) {
        auto dict_data =
            dicts_.at(dynamic_cast<const hdk::ir::ExtDictionaryType*>(col_type)->dictId())
                .get();
        CHECK(dict_data);
        // appends to materialized dictionaries are automatically materialiezd
        if (!dict_data->is_materialized) {
          lazy_fetch_cols[col_idx] = true;
        }
      }
    }
  }

  threading::parallel_for(
      threading::blocked_range(0, (int)at->columns().size()), [&](auto range) {
        for (auto col_idx = range.begin(); col_idx != range.end(); col_idx++) {
          auto col_info = getColumnInfo(db_id_, table_id, columnId(col_idx));
          auto col_type = col_info->type;
          auto col_arr = at->column(col_idx);

          // Conversion of empty string to Nulls and further processing handled
          // separately.
          if (!col_type->nullable() && col_arr->null_count() != 0 &&
              col_arr->type()->id() != arrow::Type::STRING) {
            throw std::runtime_error("Null values used in non-nullable type: "s +
                                     col_type->toString());
          }

          DictionaryData* dict_data = nullptr;
          auto elem_type =
              col_type->isArray()
                  ? dynamic_cast<const hdk::ir::ArrayBaseType*>(col_type)->elemType()
                  : col_type;
          if (elem_type->isExtDictionary()) {
            dict_data = dicts_
                            .at(dynamic_cast<const hdk::ir::ExtDictionaryType*>(elem_type)
                                    ->dictId())
                            .get();
          }

          if (col_type->isDecimal()) {
            col_arr = convertDecimalToInteger(col_arr, col_type);
          } else if (col_type->isExtDictionary()) {
            switch (col_arr->type()->id()) {
              case arrow::Type::STRING:
                // if the dictionary has already been materialized, append indices
                if (!config_->storage.enable_lazy_dict_materialization ||
                    dict_data->is_materialized) {
                  col_arr = createDictionaryEncodedColumn(
                      dict_data->dict()->stringDict.get(), col_arr, col_type);
                }
                break;
              case arrow::Type::DICTIONARY:
                col_arr = convertArrowDictionary(
                    dict_data->dict()->stringDict.get(), col_arr, col_type);
                break;
              default:
                CHECK(false);
            }
          } else if (col_type->isString()) {
          } else {
            col_arr = replaceNullValues(
                col_arr,
                col_type,
                dict_data ? dict_data->dict()->stringDict.get() : nullptr);
          }

          col_data[col_idx] = col_arr;

          bool compute_stats = !col_type->isString();
          if (compute_stats) {
            size_t elems_count = 1;
            if (col_type->isFixedLenArray()) {
              elems_count = col_type->size() / elem_type->size();
            }
            // Compute stats for each fragment.
            threading::parallel_for(
                threading::blocked_range(size_t(0), frag_count), [&](auto frag_range) {
                  for (size_t frag_idx = frag_range.begin(); frag_idx != frag_range.end();
                       ++frag_idx) {
                    auto& frag = fragments[frag_idx];

                    frag.offset =
                        frag_idx
                            ? ((frag_idx - 1) * table.fragment_size + first_frag_size)
                            : 0;
                    frag.row_count =
                        frag_idx
                            ? std::min(table.fragment_size,
                                       static_cast<size_t>(at->num_rows()) - frag.offset)
                            : first_frag_size;

                    size_t num_bytes;
                    if (col_type->isFixedLenArray()) {
                      num_bytes = frag.row_count * col_type->size();
                    } else if (col_type->isVarLenArray()) {
                      num_bytes =
                          computeTotalStringsLength(col_arr, frag.offset, frag.row_count);
                    } else {
                      num_bytes = frag.row_count * col_type->size();
                    }
                    auto meta = std::make_shared<ChunkMetadata>(
                        col_info->type, num_bytes, frag.row_count);

                    if (!lazy_fetch_cols[col_idx]) {
                      meta->fillChunkStats(computeStats(
                          col_arr->Slice(frag.offset, frag.row_count * elems_count),
                          col_type));
                    } else {
                      int32_t min = 0;
                      int32_t max = -1;
                      meta->fillChunkStats(min, max, /*has_nulls=*/true);
                    }
                    frag.metadata[col_idx] = meta;
                  }
                });  // each fragment
          } else {
            for (size_t frag_idx = 0; frag_idx < frag_count; ++frag_idx) {
              auto& frag = fragments[frag_idx];
              frag.offset =
                  frag_idx ? ((frag_idx - 1) * table.fragment_size + first_frag_size) : 0;
              frag.row_count =
                  frag_idx ? std::min(table.fragment_size,
                                      static_cast<size_t>(at->num_rows()) - frag.offset)
                           : first_frag_size;
              CHECK(col_type->isText());
              auto meta = std::make_shared<ChunkMetadata>(
                  col_info->type,
                  computeTotalStringsLength(col_arr, frag.offset, frag.row_count),
                  frag.row_count);
              meta->fillStringChunkStats(
                  col_arr->Slice(frag.offset, frag.row_count)->null_count());

              frag.metadata[col_idx] = meta;
            }
          }
        }
      });  // each column
  dict_lock.unlock();

  if (table.row_count) {
    // If table is not empty then we have to merge chunked arrays.
    CHECK_EQ(table.col_data.size(), col_data.size());
    for (size_t i = 0; i < table.col_data.size(); ++i) {
      arrow::ArrayVector lhs = table.col_data[i]->chunks();
      arrow::ArrayVector rhs = col_data[i]->chunks();
      lhs.insert(lhs.end(), rhs.begin(), rhs.end());
      table.col_data[i] = arrow::ChunkedArray::Make(std::move(lhs)).ValueOrDie();
    }

    // Probably need to merge the last existing fragment with the first new one.
    size_t start_frag = 0;
    auto& last_frag = table.fragments.back();
    if (last_frag.row_count < table.fragment_size) {
      auto& first_frag = fragments.front();
      last_frag.row_count += first_frag.row_count;
      for (size_t col_idx = 0; col_idx < last_frag.metadata.size(); ++col_idx) {
        auto col_type = getColumnInfo(db_id_, table_id, columnId(col_idx))->type;
        size_t num_elems = last_frag.metadata[col_idx]->numElements() +
                           first_frag.metadata[col_idx]->numElements();
        size_t num_bytes = last_frag.metadata[col_idx]->numBytes() +
                           first_frag.metadata[col_idx]->numBytes();
        auto stats = last_frag.metadata[col_idx]->chunkStats();
        mergeStats(stats, first_frag.metadata[col_idx]->chunkStats(), col_type);
        last_frag.metadata[col_idx] =
            std::make_shared<ChunkMetadata>(col_type, num_bytes, num_elems, stats);
      }
      start_frag = 1;
    }

    // Copy the rest of fragments adjusting offset.
    table.fragments.reserve(table.fragments.size() + fragments.size() - start_frag);
    for (size_t frag_idx = start_frag; frag_idx < fragments.size(); ++frag_idx) {
      table.fragments.emplace_back(std::move(fragments[frag_idx]));
      table.fragments.back().offset += table.row_count;
    }

    table.row_count += at->num_rows();
  } else {
    CHECK_EQ(table.row_count, (size_t)0);
    table.col_data = std::move(col_data);
    table.fragments = std::move(fragments);
    table.row_count = at->num_rows();
  }

  auto table_info = getTableInfo(db_id_, table_id);
  table_info->fragments = table.fragments.size();
  table_info->row_count = table.row_count;
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const std::vector<ColumnDescription>& columns,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  std::unordered_map<std::string, const hdk::ir::Type*> col_types;
  ColumnInfoList col_infos;
  col_infos.reserve(columns.size());
  for (auto& col : columns) {
    col_infos.emplace_back(
        std::make_shared<ColumnInfo>(-1, -1, -1, col.name, col.type, false));
    if (col.type) {
      col_types.emplace(col.name, col.type);
    }
  }
  auto at = parseCsvFile(file_name, parse_options, col_infos);
  // We allow partial schema specification in columns arg which
  // means missing columns and/or column types. Fill missing
  // info using parsed table schema.
  std::vector<ColumnDescription> updated_columns;
  updated_columns.reserve(at->columns().size());
  for (size_t i = 0; i < at->columns().size(); ++i) {
    ColumnDescription col_desc;
    col_desc.name = at->schema()->field(i)->name();
    if (col_types.count(col_desc.name)) {
      col_desc.type = col_types.at(col_desc.name);
    } else {
      col_desc.type = getTargetImportType(ctx_, *at->schema()->field(i)->type());
    }
    updated_columns.emplace_back(std::move(col_desc));
  }

  auto res = createTable(table_name, updated_columns, options);
  appendArrowTable(at, res->table_id);
  return res;
}

TableInfoPtr ArrowStorage::importCsvFile(const std::string& file_name,
                                         const std::string& table_name,
                                         const TableOptions& options,
                                         const CsvParseOptions parse_options) {
  auto at = parseCsvFile(file_name, parse_options);
  return importArrowTable(at, table_name, options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 const std::string& table_name,
                                 const CsvParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendCsvFile(file_name, tinfo->table_id, parse_options);
}

void ArrowStorage::appendCsvFile(const std::string& file_name,
                                 int table_id,
                                 const CsvParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseCsvFile(file_name, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

void ArrowStorage::appendCsvData(const std::string& csv_data,
                                 const std::string& table_name,
                                 const CsvParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendCsvData(csv_data, tinfo->table_id, parse_options);
}

void ArrowStorage::appendCsvData(const std::string& csv_data,
                                 int table_id,
                                 const CsvParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseCsvData(csv_data, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

void ArrowStorage::appendJsonData(const std::string& json_data,
                                  const std::string& table_name,
                                  const JsonParseOptions parse_options) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendJsonData(json_data, tinfo->table_id, parse_options);
}

void ArrowStorage::appendJsonData(const std::string& json_data,
                                  int table_id,
                                  const JsonParseOptions parse_options) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto col_infos = listColumns(db_id_, table_id);
  auto at = parseJsonData(json_data, parse_options, col_infos);
  appendArrowTable(at, table_id);
}

TableInfoPtr ArrowStorage::importParquetFile(const std::string& file_name,
                                             const std::string& table_name,
                                             const TableOptions& options) {
  auto at = parseParquetFile(file_name);
  return importArrowTable(at, table_name, options);
}

void ArrowStorage::appendParquetFile(const std::string& file_name,
                                     const std::string& table_name) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    throw std::runtime_error("Unknown table: "s + table_name);
  }
  appendParquetFile(file_name, tinfo->table_id);
}

void ArrowStorage::appendParquetFile(const std::string& file_name, int table_id) {
  if (!getTableInfo(db_id_, table_id)) {
    throw std::runtime_error("Invalid table id: "s + std::to_string(table_id));
  }

  auto at = parseParquetFile(file_name);
  appendArrowTable(at, table_id);
}

void ArrowStorage::dropTable(const std::string& table_name, bool throw_if_not_exist) {
  auto tinfo = getTableInfo(db_id_, table_name);
  if (!tinfo) {
    if (throw_if_not_exist) {
      throw std::runtime_error("Cannot srop unknown table: "s + table_name);
    }
    return;
  }
  dropTable(tinfo->table_id);
}

void ArrowStorage::dropTable(int table_id, bool throw_if_not_exist) {
  mapd_unique_lock<mapd_shared_mutex> data_lock(data_mutex_);
  mapd_unique_lock<mapd_shared_mutex> dict_lock(dict_mutex_);
  mapd_unique_lock<mapd_shared_mutex> schema_lock(schema_mutex_);

  if (!tables_.count(table_id)) {
    if (throw_if_not_exist) {
      throw std::runtime_error("Cannot drop table with invalid id: "s +
                               std::to_string(table_id));
    }
  }

  std::unique_ptr<TableData> table = std::move(tables_.at(table_id));
  mapd_unique_lock<mapd_shared_mutex> table_lock(table->mutex);
  tables_.erase(table_id);

  std::unordered_set<int> dicts_to_remove;
  auto col_infos = listColumnsNoLock(db_id_, table_id);
  for (auto& col_info : col_infos) {
    if (col_info->type->isExtDictionary()) {
      dicts_to_remove.insert(col_info->type->as<hdk::ir::ExtDictionaryType>()->dictId());
    }
  }

  SimpleSchemaProvider::dropTable(db_id_, table_id);
  // TODO: clean-up shared dictionaries without a full scan of
  // existing columns.
  for (auto& pr : column_infos_) {
    if (pr.second->type->isExtDictionary()) {
      dicts_to_remove.erase(pr.second->type->as<hdk::ir::ExtDictionaryType>()->dictId());
    }
  }

  for (auto dict_id : dicts_to_remove) {
    dicts_.erase(dict_id);
  }
}

void ArrowStorage::checkNewTableParams(const std::string& table_name,
                                       const std::vector<ColumnDescription>& columns,
                                       const TableOptions& options) const {
  if (columns.empty()) {
    throw std::runtime_error("Cannot create table with no columns");
  }

  if (table_name.empty()) {
    throw std::runtime_error("Cannot create table with empty name");
  }

  auto tinfo = getTableInfoNoLock(db_id_, table_name);
  if (tinfo) {
    throw std::runtime_error("Table with name '"s + table_name + "' already exists"s);
  }

  std::unordered_set<std::string> col_names;
  for (auto& col : columns) {
    if (col.name.empty()) {
      throw std::runtime_error("Empty column name is not allowed");
    }

    if (col.name == "rowid") {
      throw std::runtime_error("Reserved column name is not allowed: "s + col.name);
    }

    if (col_names.count(col.name)) {
      throw std::runtime_error("Duplicated column name: "s + col.name);
    }

    switch (col.type->id()) {
      case hdk::ir::Type::kBoolean:
      case hdk::ir::Type::kInteger:
      case hdk::ir::Type::kFloatingPoint:
      case hdk::ir::Type::kVarChar:
      case hdk::ir::Type::kText:
      case hdk::ir::Type::kDecimal:
      case hdk::ir::Type::kTime:
      case hdk::ir::Type::kDate:
      case hdk::ir::Type::kTimestamp:
      case hdk::ir::Type::kFixedLenArray:
      case hdk::ir::Type::kVarLenArray:
      case hdk::ir::Type::kInterval:
        break;
      case hdk::ir::Type::kExtDictionary: {
        auto dict_id = col.type->as<hdk::ir::ExtDictionaryType>()->dictId();
        if (dict_id > 0 && dicts_.count(dict_id) == 0) {
          throw std::runtime_error("Unknown dictionary ID is referenced in column '"s +
                                   col.name + "': "s + std::to_string(dict_id));
        }
      } break;
      default:
        throw std::runtime_error("Unsupported type for Arrow import: "s +
                                 col.type->toString());
    }

    col_names.insert(col.name);
  }
}

void ArrowStorage::compareSchemas(std::shared_ptr<arrow::Schema> lhs,
                                  std::shared_ptr<arrow::Schema> rhs) {
  auto& lhs_fields = lhs->fields();
  auto& rhs_fields = rhs->fields();
  if (lhs_fields.size() != rhs_fields.size()) {
    throw std::runtime_error("Mismatched clumns count: "s +
                             std::to_string(lhs_fields.size()) + " != "s +
                             std::to_string(rhs_fields.size()));
  }

  for (size_t i = 0; i < lhs_fields.size(); ++i) {
    auto lhs_type = lhs_fields[i]->type();
    auto rhs_type = rhs_fields[i]->type();

    if (!lhs_type->Equals(rhs_type) && (lhs_type->id() != arrow::Type::NA) &&
        (rhs_type->id() != arrow::Type::NA)) {
      throw std::runtime_error(
          "Mismatched type for column "s + lhs_fields[i]->name() + ": "s +
          lhs_type->ToString() + " [id: " + std::to_string(lhs_type->id()) + "] vs. "s +
          rhs_type->ToString() + " [id: " + std::to_string(lhs_type->id()) + "]");
    }
  }
}

ChunkStats ArrowStorage::computeStats(std::shared_ptr<arrow::ChunkedArray> arr,
                                      const hdk::ir::Type* type) {
  auto elem_type =
      type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
  std::unique_ptr<Encoder> encoder(Encoder::Create(nullptr, elem_type));
  for (auto& chunk : arr->chunks()) {
    if (type->isVarLenArray()) {
      auto elem_size = elem_type->size();
      auto chunk_list = std::dynamic_pointer_cast<arrow::ListArray>(chunk);
      CHECK(chunk_list);
      auto offs = std::abs(chunk_list->value_offset(0)) / elem_size;
      auto len = std::abs(chunk_list->value_offset(chunk->length())) / elem_size - offs;
      auto elems = chunk_list->values();
      encoder->updateStatsEncoded(elems->data()->GetValues<int8_t>(
                                      1, (elems->data()->offset + offs) * type->size()),
                                  len);
    } else if (type->isFixedLenArray()) {
      encoder->updateStatsEncoded(
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * elem_type->size()),
          chunk->length(),
          true);
    } else if (chunk->length() != 0) {
      encoder->updateStatsEncoded(
          chunk->data()->GetValues<int8_t>(1, chunk->data()->offset * elem_type->size()),
          chunk->length());
    }
  }

  ChunkStats stats;
  encoder->fillChunkStats(stats, elem_type);
  return stats;
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsvFile(
    const std::string& file_name,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) const {
  std::shared_ptr<arrow::io::ReadableFile> inp;
  auto file_result = arrow::io::ReadableFile::Open(file_name.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  return parseCsv(file_result.ValueOrDie(), parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsvData(
    const std::string& csv_data,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) const {
  auto input = std::make_shared<arrow::io::BufferReader>(csv_data);
  return parseCsv(input, parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseCsv(
    std::shared_ptr<arrow::io::InputStream> input,
    const CsvParseOptions parse_options,
    const ColumnInfoList& col_infos) const {
  auto io_context = arrow::io::default_io_context();

  auto arrow_parse_options = arrow::csv::ParseOptions::Defaults();
  arrow_parse_options.quoting = false;
  arrow_parse_options.escaping = false;
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.delimiter = parse_options.delimiter;

  auto arrow_read_options = arrow::csv::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;
  arrow_read_options.block_size = parse_options.block_size;
  arrow_read_options.autogenerate_column_names =
      !parse_options.header && col_infos.empty();
  arrow_read_options.skip_rows = parse_options.skip_rows;

  auto arrow_convert_options = arrow::csv::ConvertOptions::Defaults();
  arrow_convert_options.check_utf8 = false;
  arrow_convert_options.include_columns = arrow_read_options.column_names;
  arrow_convert_options.strings_can_be_null = true;

  for (auto& col_info : col_infos) {
    if (!col_info->is_rowid) {
      if (!parse_options.header) {
        arrow_read_options.column_names.push_back(col_info->name);
      }
      if (col_info->type) {
        arrow_convert_options.column_types.emplace(
            col_info->name, getArrowImportType(ctx_, col_info->type));
      }
    }
  }

  auto table_reader_result = arrow::csv::TableReader::Make(
      io_context, input, arrow_read_options, arrow_parse_options, arrow_convert_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> at;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    at = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow CSV in " << time << "ms";

  return at;
}

std::shared_ptr<arrow::Table> ArrowStorage::parseJsonData(
    const std::string& json_data,
    const JsonParseOptions parse_options,
    const ColumnInfoList& col_infos) const {
  auto input = std::make_shared<arrow::io::BufferReader>(json_data);
  return parseJson(input, parse_options, col_infos);
}

std::shared_ptr<arrow::Table> ArrowStorage::parseJson(
    std::shared_ptr<arrow::io::InputStream> input,
    const JsonParseOptions parse_options,
    const ColumnInfoList& col_infos) const {
  arrow::FieldVector fields;
  fields.reserve(col_infos.size());
  for (auto& col_info : col_infos) {
    if (!col_info->is_rowid) {
      fields.emplace_back(
          std::make_shared<arrow::Field>(col_info->name,
                                         getArrowImportType(ctx_, col_info->type),
                                         col_info->type->nullable()));
    }
  }
  auto schema = std::make_shared<arrow::Schema>(std::move(fields));

  auto arrow_parse_options = arrow::json::ParseOptions::Defaults();
  arrow_parse_options.newlines_in_values = false;
  arrow_parse_options.explicit_schema = schema;

  auto arrow_read_options = arrow::json::ReadOptions::Defaults();
  arrow_read_options.use_threads = true;
  arrow_read_options.block_size = parse_options.block_size;

  auto table_reader_result = arrow::json::TableReader::Make(
      arrow::default_memory_pool(), input, arrow_read_options, arrow_parse_options);
  ARROW_THROW_NOT_OK(table_reader_result.status());
  auto table_reader = table_reader_result.ValueOrDie();

  std::shared_ptr<arrow::Table> at;
  auto time = measure<>::execution([&]() {
    auto arrow_table_result = table_reader->Read();
    ARROW_THROW_NOT_OK(arrow_table_result.status());
    at = arrow_table_result.ValueOrDie();
  });

  VLOG(1) << "Read Arrow JSON in " << time << "ms";

  return at;
}

std::shared_ptr<arrow::Table> ArrowStorage::parseParquetFile(
    const std::string& file_name) const {
  auto file_result = arrow::io::ReadableFile::Open(file_name.c_str());
  ARROW_THROW_NOT_OK(file_result.status());
  auto inp = file_result.ValueOrDie();

  auto parquet_reader = parquet::ParquetFileReader::Open(inp);

  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  // Allow multithreding.
  parquet::ArrowReaderProperties prop(true);
  auto st = parquet::arrow::FileReader::Make(
      arrow::default_memory_pool(), std::move(parquet_reader), prop, &arrow_reader);
  if (!st.ok()) {
    throw std::runtime_error(st.ToString());
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  st = arrow_reader->ReadTable(&table);
  if (!st.ok()) {
    throw std::runtime_error(st.ToString());
  }

  return table;
}

ArrowStorage::DictionaryData::DictionaryData(
    std::unique_ptr<DictDescriptor>&& dict_descriptor,
    const hdk::ir::ExtDictionaryType* type,
    const bool enable_lazy_materialization)
    : dict_descriptor(std::move(dict_descriptor)), type(type) {
  if (!enable_lazy_materialization) {
    is_materialized = true;
  }
}
