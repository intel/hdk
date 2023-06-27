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

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
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
    table_infos.push_back({db_id, table_id, executor->getTableInfo(db_id, table_id)});
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(TableRef{db_id, table_id}, table_infos.size() - 1));
  }
}

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
