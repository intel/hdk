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

#ifndef QUERYENGINE_INPUTMETADATA_H
#define QUERYENGINE_INPUTMETADATA_H

#include "DataProvider/DataProvider.h"
#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "ResultSetRegistry/ResultSetTableToken.h"
#include "Shared/hash.h"

#include <unordered_map>

class Executor;

using TemporaryTables = std::unordered_map<int, hdk::ResultSetTableTokenPtr>;

struct InputTableInfo {
  int db_id;
  int table_id;
  TableFragmentsInfo info;
};

class InputTableInfoCache {
 public:
  InputTableInfoCache(Executor* executor);

  TableFragmentsInfo getTableInfo(int db_id, int table_id);

  void clear();

 private:
  std::unordered_map<std::pair<int, int>, TableFragmentsInfo> cache_;
  Executor* executor_;
};

ChunkMetadataMap synthesize_metadata(const ResultSet* rows);

size_t get_frag_count_of_table(const int db_id, const int table_id, Executor* executor);

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor);

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor);

#endif  // QUERYENGINE_INPUTMETADATA_H
