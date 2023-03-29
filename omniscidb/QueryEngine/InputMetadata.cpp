/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "InputMetadata.h"
#include "Execute.h"

InputTableInfoCache::InputTableInfoCache(Executor* executor) : executor_(executor) {}

namespace {

TableFragmentsInfo copy_table_info(const TableFragmentsInfo& table_info) {
  TableFragmentsInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = table_info.chunkKeyPrefix;
  table_info_copy.fragments = table_info.fragments;
  table_info_copy.setPhysicalNumTuples(table_info.getPhysicalNumTuples());
  return table_info_copy;
}

}  // namespace

TableFragmentsInfo InputTableInfoCache::getTableInfo(int db_id, int table_id) {
  const auto it = cache_.find({db_id, table_id});
  if (it != cache_.end()) {
    const auto& table_info = it->second;
    return copy_table_info(table_info);
  }
  const auto data_mgr = executor_->getDataMgr();
  CHECK(data_mgr);
  auto table_info = data_mgr->getTableMetadata(db_id, table_id);
  auto it_ok =
      cache_.emplace(std::make_pair(db_id, table_id), copy_table_info(table_info));
  CHECK(it_ok.second);
  return copy_table_info(table_info);
}

void InputTableInfoCache::clear() {
  decltype(cache_)().swap(cache_);
}

namespace {

TableFragmentsInfo synthesize_table_info(hdk::ResultSetTableTokenPtr token) {
  std::vector<FragmentInfo> result;
  bool non_empty = false;
  for (int frag_id = 0; frag_id < static_cast<int>(token->resultSetCount()); ++frag_id) {
    result.emplace_back();
    auto& fragment = result.back();
    fragment.fragmentId = frag_id;
    fragment.deviceIds.resize(3);
    fragment.resultSet = token->resultSet(frag_id).get();
    fragment.resultSetMutex.reset(new std::mutex());
    fragment.setPhysicalNumTuples(fragment.resultSet->entryCount());

    for (size_t col_idx = 0;
         col_idx < static_cast<size_t>(fragment.resultSet->colCount());
         ++col_idx) {
      auto meta = std::make_shared<ChunkMetadata>(
          fragment.resultSet->colType(col_idx),
          0,
          0,
          [frag_id, col_idx, token](ChunkStats& stats) {
            stats = token->getChunkStats(frag_id, col_idx);
          });
      fragment.setChunkMetadata(static_cast<int>(col_idx), meta);
    }

    non_empty |= (fragment.resultSet != nullptr);
  }
  TableFragmentsInfo table_info;
  if (non_empty)
    table_info.fragments = std::move(result);
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  std::unordered_map<TableRef, size_t> info_cache;
  for (const auto& input_desc : input_descs) {
    int db_id = input_desc.getDatabaseId();
    const auto table_id = input_desc.getTableId();
    const auto cached_index_it = info_cache.find({db_id, table_id});
    if (cached_index_it != info_cache.end()) {
      CHECK_LT(cached_index_it->second, table_infos.size());
      table_infos.push_back(
          {db_id, table_id, copy_table_info(table_infos[cached_index_it->second].info)});
      continue;
    }
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      LOG_IF(FATAL, it == temporary_tables->end())
          << "Failed to find previous query result for node " << -table_id;
      table_infos.push_back({db_id, table_id, synthesize_table_info(it->second)});
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({db_id, table_id, executor->getTableInfo(db_id, table_id)});
    }
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(TableRef{db_id, table_id}, table_infos.size() - 1));
  }
}

}  // namespace

size_t get_frag_count_of_table(const int db_id, const int table_id, Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  CHECK(temporary_tables);
  auto it = temporary_tables->find(table_id);
  if (it != temporary_tables->end()) {
    CHECK_GE(int(0), table_id);
    return size_t(1);
  } else {
    const auto table_info = executor->getTableInfo(db_id, table_id);
    return table_info.fragments.size();
  }
}

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, input_descs, executor);
  return table_infos;
}

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor) {
  INJECT_TIMER(get_table_infos);
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, ra_exe_unit.input_descs, executor);
  return table_infos;
}

const ChunkMetadataMap& FragmentInfo::getChunkMetadataMap() const {
  return chunkMetadataMap;
}

ChunkMetadataMap FragmentInfo::getChunkMetadataMapPhysicalCopy() const {
  ChunkMetadataMap metadata_map;
  for (const auto& [column_id, chunk_metadata] : chunkMetadataMap) {
    metadata_map[column_id] = std::make_shared<ChunkMetadata>(*chunk_metadata);
  }
  return metadata_map;
}

size_t FragmentInfo::getNumTuples() const {
  std::unique_ptr<std::lock_guard<std::mutex>> lock;
  if (resultSetMutex) {
    lock.reset(new std::lock_guard<std::mutex>(*resultSetMutex));
  }
  CHECK_EQ(!!resultSet, !!resultSetMutex);
  if (resultSet && !synthesizedNumTuplesIsValid) {
    numTuples = resultSet->rowCount();
    synthesizedNumTuplesIsValid = true;
  }
  return numTuples;
}

size_t TableFragmentsInfo::getNumTuples() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().getNumTuples();
  }
  return numTuples;
}

size_t TableFragmentsInfo::getNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  return numTuples;
}

size_t TableFragmentsInfo::getFragmentNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  size_t fragment_num_tupples_upper_bound = 0;
  for (const auto& fragment : fragments) {
    fragment_num_tupples_upper_bound =
        std::max(fragment.getNumTuples(), fragment_num_tupples_upper_bound);
  }
  return fragment_num_tupples_upper_bound;
}
