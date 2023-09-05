/*
 * Copyright 2020 OmniSci, Inc.
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

#include "QueryEngine/JoinHashTable/PerfectHashTable.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntimeCpu.h"

#include "Shared/scope.h"

class PerfectJoinHashTableBuilder {
 public:
  PerfectJoinHashTableBuilder() {}

  void allocateDeviceMemory(const JoinColumn& join_column,
                            const HashType layout,
                            HashEntryInfo& hash_entry_info,
                            const int device_id,
                            const int device_count,
                            const Executor* executor) {
#ifdef HAVE_CUDA
    const size_t total_count =
        layout == HashType::OneToOne
            ? hash_entry_info.getNormalizedHashEntryCount()
            : 2 * hash_entry_info.getNormalizedHashEntryCount() + join_column.num_elems;
    CHECK(!hash_table_);
    hash_table_ =
        std::make_unique<PerfectHashTable>(executor->getBufferProvider(),
                                           layout,
                                           ExecutorDeviceType::GPU,
                                           hash_entry_info.getNormalizedHashEntryCount(),
                                           join_column.num_elems);
    hash_table_->allocateGpuMemory(total_count, device_id);
#else
    UNREACHABLE();
#endif  // HAVE_CUDA
  }

#ifdef HAVE_CUDA
  void initHashTableOnGpu(const ChunkKey& chunk_key,
                          const JoinColumn& join_column,
                          const ExpressionRange& col_range,
                          const bool is_bitwise_eq,
                          const InnerOuter& cols,
                          const JoinType join_type,
                          const HashType layout,
                          const HashEntryInfo hash_entry_info,
                          const int32_t hash_join_invalid_val,
                          const int device_id,
                          const int device_count,
                          const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    auto buffer_provider = executor->getBufferProvider();
    Data_Namespace::AbstractBuffer* gpu_hash_table_err_buff =
        GpuAllocator::allocGpuAbstractBuffer(buffer_provider, sizeof(int), device_id);
    ScopeGuard cleanup_error_buff = [buffer_provider, gpu_hash_table_err_buff]() {
      buffer_provider->free(gpu_hash_table_err_buff);
    };
    CHECK(gpu_hash_table_err_buff);
    auto dev_err_buff =
        reinterpret_cast<CUdeviceptr>(gpu_hash_table_err_buff->getMemoryPtr());
    int err{0};
    buffer_provider->copyToDevice(reinterpret_cast<int8_t*>(dev_err_buff),
                                  reinterpret_cast<const int8_t*>(&err),
                                  sizeof(err),
                                  device_id);

    CHECK(hash_table_);
    auto gpu_hash_table_buff = hash_table_->getGpuBuffer();

    init_hash_join_buff_on_device(reinterpret_cast<int32_t*>(gpu_hash_table_buff),
                                  hash_entry_info.getNormalizedHashEntryCount(),
                                  hash_join_invalid_val);
    if (chunk_key.empty()) {
      return;
    }

    // TODO: pass this in? duplicated in JoinHashTable currently
    const auto inner_col = cols.first;
    CHECK(inner_col);
    auto type = inner_col->type();

    JoinColumnTypeInfo type_info{static_cast<size_t>(type->size()),
                                 col_range.getIntMin(),
                                 col_range.getIntMax(),
                                 inline_fixed_encoding_null_value(type),
                                 is_bitwise_eq,
                                 col_range.getIntMax() + 1,
                                 get_join_column_type_kind(type)};
    auto use_bucketization = inner_col->type()->isDate();
    if (layout == HashType::OneToOne) {
      fill_hash_join_buff_on_device_bucketized(
          reinterpret_cast<int32_t*>(gpu_hash_table_buff),
          hash_join_invalid_val,
          for_semi_anti_join(join_type),
          reinterpret_cast<int*>(dev_err_buff),
          join_column,
          type_info,
          hash_entry_info.bucket_normalization);
    } else {
      if (use_bucketization) {
        fill_one_to_many_hash_table_on_device_bucketized(
            reinterpret_cast<int32_t*>(gpu_hash_table_buff),
            hash_entry_info,
            hash_join_invalid_val,
            join_column,
            type_info);
      } else {
        fill_one_to_many_hash_table_on_device(
            reinterpret_cast<int32_t*>(gpu_hash_table_buff),
            hash_entry_info,
            hash_join_invalid_val,
            join_column,
            type_info);
      }
    }
    buffer_provider->copyFromDevice(reinterpret_cast<int8_t*>(&err),
                                    reinterpret_cast<int8_t*>(dev_err_buff),
                                    sizeof(err),
                                    device_id);
    if (err) {
      if (layout == HashType::OneToOne) {
        throw NeedsOneToManyHash();
      } else {
        throw std::runtime_error("Unexpected error when building perfect hash table: " +
                                 std::to_string(err));
      }
    }
  }
#endif

  void initOneToOneHashTableOnCpu(
      const JoinColumn& join_column,
      const ExpressionRange& col_range,
      const bool is_bitwise_eq,
      const InnerOuter& cols,
      const StringDictionaryProxy::IdMap* str_proxy_translation_map,
      const JoinType join_type,
      const HashType hash_type,
      const HashEntryInfo hash_entry_info,
      const int32_t hash_join_invalid_val,
      const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    const auto inner_col = cols.first;
    CHECK(inner_col);
    auto type = inner_col->type();

    CHECK(!hash_table_);
    hash_table_ =
        std::make_unique<PerfectHashTable>(executor->getBufferProvider(),
                                           hash_type,
                                           ExecutorDeviceType::CPU,
                                           hash_entry_info.getNormalizedHashEntryCount(),
                                           0);

    auto cpu_hash_table_buff = reinterpret_cast<int32_t*>(hash_table_->getCpuBuffer());

    {
      auto timer_init = DEBUG_TIMER("CPU One-To-One Perfect-Hash: init_hash_join_buff");
      init_hash_join_buff(cpu_hash_table_buff,
                          hash_entry_info.getNormalizedHashEntryCount(),
                          hash_join_invalid_val);
    }
    const bool for_semi_join = for_semi_anti_join(join_type);
    {
      auto timer_fill =
          DEBUG_TIMER("CPU One-To-One Perfect-Hash: fill_hash_join_buff_bucketized_cpu");

      {
        JoinColumnTypeInfo type_info{static_cast<size_t>(type->size()),
                                     col_range.getIntMin(),
                                     col_range.getIntMax(),
                                     inline_fixed_encoding_null_value(type),
                                     is_bitwise_eq,
                                     col_range.getIntMax() + 1,
                                     get_join_column_type_kind(type)};

        int error = fill_hash_join_buff_bucketized_cpu(
            cpu_hash_table_buff,
            hash_join_invalid_val,
            for_semi_join,
            join_column,
            type_info,
            str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
            hash_entry_info.bucket_normalization);
        if (error) {
          // Too many hash entries, need to retry with a 1:many table
          hash_table_ = nullptr;  // clear the hash table buffer
          throw NeedsOneToManyHash();
        }
      }
    }
  }

  void initOneToManyHashTableOnCpu(
      const JoinColumn& join_column,
      const ExpressionRange& col_range,
      const bool is_bitwise_eq,
      const std::pair<const hdk::ir::ColumnVar*, const hdk::ir::Expr*>& cols,
      const StringDictionaryProxy::IdMap* str_proxy_translation_map,
      const HashEntryInfo hash_entry_info,
      const int32_t hash_join_invalid_val,
      const Executor* executor) {
    auto timer = DEBUG_TIMER(__func__);
    const auto inner_col = cols.first;
    CHECK(inner_col);
    auto type = inner_col->type();
    CHECK(!hash_table_);
    hash_table_ =
        std::make_unique<PerfectHashTable>(executor->getBufferProvider(),
                                           HashType::OneToMany,
                                           ExecutorDeviceType::CPU,
                                           hash_entry_info.getNormalizedHashEntryCount(),
                                           join_column.num_elems);

    auto cpu_hash_table_buff = reinterpret_cast<int32_t*>(hash_table_->getCpuBuffer());

    int thread_count = cpu_threads();
    {
      auto timer_init =
          DEBUG_TIMER("CPU One-To-Many Perfect Hash Table Builder: init_hash_join_buff");
      init_hash_join_buff(cpu_hash_table_buff,
                          hash_entry_info.getNormalizedHashEntryCount(),
                          hash_join_invalid_val);
    }
    {
      auto timer_fill = DEBUG_TIMER(
          "CPU One-To-Many Perfect Hash Table Builder: fill_hash_join_buff_bucketized");
      if (type->isDate()) {
        fill_one_to_many_hash_table_bucketized(
            cpu_hash_table_buff,
            hash_entry_info,
            hash_join_invalid_val,
            join_column,
            {static_cast<size_t>(type->size()),
             col_range.getIntMin(),
             col_range.getIntMax(),
             inline_fixed_encoding_null_value(type),
             is_bitwise_eq,
             col_range.getIntMax() + 1,
             get_join_column_type_kind(type)},
            str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
            thread_count);
      } else {
        fill_one_to_many_hash_table(
            cpu_hash_table_buff,
            hash_entry_info,
            hash_join_invalid_val,
            join_column,
            {static_cast<size_t>(type->size()),
             col_range.getIntMin(),
             col_range.getIntMax(),
             inline_fixed_encoding_null_value(type),
             is_bitwise_eq,
             col_range.getIntMax() + 1,
             get_join_column_type_kind(type)},
            str_proxy_translation_map ? str_proxy_translation_map->data() : nullptr,
            thread_count);
      }
    }
  }

  std::unique_ptr<PerfectHashTable> getHashTable() { return std::move(hash_table_); }

  const bool for_semi_anti_join(const JoinType join_type) {
    return join_type == JoinType::SEMI || join_type == JoinType::ANTI;
  }

 private:
  std::unique_ptr<PerfectHashTable> hash_table_;
};
