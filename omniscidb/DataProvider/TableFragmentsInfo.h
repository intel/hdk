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

#pragma once

#include "DataMgr/ChunkMetadata.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/types.h"

#include <deque>
#include <list>
#include <map>
#include <mutex>

/**
 * @class FragmentInfo
 * @brief Used by Fragmenter classes to store info about each
 * fragment - the fragment id and number of tuples(rows)
 * currently stored by that fragment
 */

class FragmentInfo {
 public:
  FragmentInfo() : fragmentId(-1), physicalTableId(-1), numTuples(0) {}

  void setChunkMetadataMap(const ChunkMetadataMap& chunk_metadata_map) {
    this->chunkMetadataMap = chunk_metadata_map;
  }

  void setChunkMetadata(const int col, std::shared_ptr<ChunkMetadata> chunkMetadata) {
    chunkMetadataMap[col] = chunkMetadata;
  }

  const ChunkMetadataMap& getChunkMetadataMap() const { return chunkMetadataMap; }

  const ChunkMetadataMap& getChunkMetadataMapPhysical() const { return chunkMetadataMap; }

  ChunkMetadataMap getChunkMetadataMapPhysicalCopy() const {
    ChunkMetadataMap metadata_map;
    for (const auto& [column_id, chunk_metadata] : chunkMetadataMap) {
      metadata_map[column_id] = std::make_shared<ChunkMetadata>(*chunk_metadata);
    }
    return metadata_map;
  }

  size_t getNumTuples() const { return numTuples; }

  size_t getPhysicalNumTuples() const { return numTuples; }

  bool isEmptyPhysicalFragment() const { return physicalTableId >= 0 && !numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  int fragmentId;
  std::vector<int> deviceIds;
  int physicalTableId;

 private:
  mutable size_t numTuples;
  mutable ChunkMetadataMap chunkMetadataMap;
};

class TableFragmentsInfo {
 public:
  TableFragmentsInfo() : numTuples(0) {}

  size_t getNumTuples() const { return numTuples; }

  size_t getNumTuplesUpperBound() const { return numTuples; }

  size_t getPhysicalNumTuples() const { return numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  size_t getFragmentNumTuplesUpperBound() const {
    size_t fragment_num_tupples_upper_bound = 0;
    for (const auto& fragment : fragments) {
      fragment_num_tupples_upper_bound =
          std::max(fragment.getNumTuples(), fragment_num_tupples_upper_bound);
    }
    return fragment_num_tupples_upper_bound;
  }

  std::vector<int> chunkKeyPrefix;
  std::vector<FragmentInfo> fragments;

 private:
  mutable size_t numTuples;
};
