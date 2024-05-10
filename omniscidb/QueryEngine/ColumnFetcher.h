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

#include "DataMgr/Allocators/DeviceAllocator.h"
#include "DataProvider/DataProvider.h"
#include "IR/Expr.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"
#include "ResultSetRegistry/ColumnarResults.h"
#include "Shared/hash.h"

struct FetchResult {
  std::vector<std::vector<const int8_t*>> col_buffers;
  std::vector<std::vector<int64_t>> num_rows;
  std::vector<std::vector<uint64_t>> frag_offsets;
};

using MergedChunk = std::pair<AbstractBuffer*, AbstractBuffer*>;

class ColumnFetcher {
 public:
  ColumnFetcher(Executor* executor,
                DataProvider* data_provider,
                const ColumnCacheMap& column_cache);

  //! Gets one chunk's pointer and element count on either CPU or GPU.
  static std::pair<const int8_t*, size_t> getOneColumnFragment(
      Executor* executor,
      const hdk::ir::ColumnVar& hash_col,
      const FragmentInfo& fragment,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      DeviceAllocator* device_allocator,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache);

  //! Creates a JoinColumn struct containing an array of JoinChunk structs.
  static JoinColumn makeJoinColumn(
      Executor* executor,
      const hdk::ir::ColumnVar& hash_col,
      const std::vector<FragmentInfo>& fragments,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      DeviceAllocator* device_allocator,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      std::vector<std::shared_ptr<void>>& malloc_owner,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache);

  const int8_t* getOneTableColumnFragment(
      ColumnInfoPtr col_info,
      const int frag_id,
      const std::map<TableRef, const TableFragments*>& all_tables_fragments,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  const int8_t* getAllTableColumnFragments(
      ColumnInfoPtr col_info,
      const std::map<TableRef, const TableFragments*>& all_tables_fragments,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator,
      const size_t thread_idx) const;

  const int8_t* linearizeColumnFragments(
      ColumnInfoPtr col_info,
      const std::map<TableRef, const TableFragments*>& all_tables_fragments,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  void freeTemporaryCpuLinearizedIdxBuf();
  void freeLinearizedBuf();

  DataProvider* getDataProvider() const { return data_provider_; }

 private:
  static const int8_t* transferColumnIfNeeded(
      const ColumnarResults* columnar_results,
      const int col_id,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator);

  MergedChunk linearizeVarLenArrayColFrags(
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& local_chunk_holder,
      std::list<ChunkIter>& local_chunk_iter_holder,
      std::list<size_t>& local_chunk_num_tuples,
      MemoryLevel memory_level,
      ColumnInfoPtr col_info,
      const int device_id,
      const size_t total_data_buf_size,
      const size_t total_idx_buf_size,
      const size_t total_num_tuples,
      DeviceAllocator* device_allocator) const;

  MergedChunk linearizeFixedLenArrayColFrags(
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& local_chunk_holder,
      std::list<ChunkIter>& local_chunk_iter_holder,
      std::list<size_t>& local_chunk_num_tuples,
      MemoryLevel memory_level,
      ColumnInfoPtr col_info,
      const int device_id,
      const size_t total_data_buf_size,
      const size_t total_idx_buf_size,
      const size_t total_num_tuples,
      DeviceAllocator* device_allocator) const;

  void addMergedChunkIter(const int table_id,
                          const int col_id,
                          const int device_id,
                          int8_t* chunk_iter_ptr) const;

  const int8_t* getChunkiter(const int table_id,
                             const int col_id,
                             const int device_id = 0) const;

  ChunkIter prepareChunkIter(AbstractBuffer* merged_data_buf,
                             AbstractBuffer* merged_index_buf,
                             ChunkIter& chunk_iter,
                             bool is_true_varlen_type,
                             const size_t total_num_tuples) const;

  Executor* executor_;
  DataProvider* data_provider_;
  mutable std::mutex columnar_fetch_mutex_;
  mutable std::mutex varlen_chunk_fetch_mutex_;
  mutable std::mutex linearization_mutex_;
  mutable std::mutex chunk_list_mutex_;
  mutable std::mutex linearized_col_cache_mutex_;
  mutable ColumnCacheMap columnarized_table_cache_;
  // All caches map [table_id, col_id] to cached data
  mutable std::unordered_map<std::pair<int, int>, std::unique_ptr<const ColumnarResults>>
      columnarized_scan_table_cache_;
  using DeviceMergedChunkIterMap = std::unordered_map<int, int8_t*>;
  using DeviceMergedChunkMap = std::unordered_map<int, AbstractBuffer*>;
  mutable std::unordered_map<std::pair<int, int>, DeviceMergedChunkIterMap>
      linearized_multi_frag_chunk_iter_cache_;
  mutable std::unordered_map<int, AbstractBuffer*>
      linearlized_temporary_cpu_index_buf_cache_;
  mutable std::unordered_map<std::pair<int, int>, DeviceMergedChunkMap>
      linearized_data_buf_cache_;
  mutable std::unordered_map<std::pair<int, int>, DeviceMergedChunkMap>
      linearized_idx_buf_cache_;

  friend class QueryCompilationDescriptor;
};
