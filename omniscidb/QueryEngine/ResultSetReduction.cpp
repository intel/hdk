/*
 * Copyright 2021 OmniSci, Inc.
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

/**
 * @file    ResultSetReduction.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Reduction part of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSetReduction.h"
#include "DynamicWatchdog.h"
#include "Execute.h"
#include "ResultSetReductionInterpreter.h"
#include "ResultSetReductionJIT.h"
#include "RuntimeFunctions.h"

#include "ResultSet/CountDistinct.h"
#include "ResultSet/ResultSet.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include <llvm/ExecutionEngine/GenericValue.h>

#include <tbb/parallel_for.h>
#include <algorithm>
#include <future>
#include <numeric>

namespace {

bool use_multithreaded_reduction(const size_t entry_count) {
  return entry_count > 100000;
}

size_t get_row_qw_count(const QueryMemoryDescriptor& query_mem_desc) {
  const auto row_bytes = get_row_bytes(query_mem_desc);
  CHECK_EQ(size_t(0), row_bytes % 8);
  return row_bytes / 8;
}

std::vector<int64_t> make_key(const int64_t* buff,
                              const size_t entry_count,
                              const size_t key_count) {
  std::vector<int64_t> key;
  size_t off = 0;
  for (size_t i = 0; i < key_count; ++i) {
    key.push_back(buff[off]);
    off += entry_count;
  }
  return key;
}

void fill_slots(int64_t* dst_entry,
                const size_t dst_entry_count,
                const int64_t* src_buff,
                const size_t src_entry_idx,
                const size_t src_entry_count,
                const QueryMemoryDescriptor& query_mem_desc) {
  const auto slot_count = query_mem_desc.getBufferColSlotCount();
  const auto key_count = query_mem_desc.getGroupbyColCount();
  if (query_mem_desc.didOutputColumnar()) {
    for (size_t i = 0, dst_slot_off = 0; i < slot_count;
         ++i, dst_slot_off += dst_entry_count) {
      dst_entry[dst_slot_off] =
          src_buff[slot_offset_colwise(src_entry_idx, i, key_count, src_entry_count)];
    }
  } else {
    const auto row_ptr = src_buff + get_row_qw_count(query_mem_desc) * src_entry_idx;
    const auto slot_off_quad = get_slot_off_quad(query_mem_desc);
    for (size_t i = 0; i < slot_count; ++i) {
      dst_entry[i] = row_ptr[slot_off_quad + i];
    }
  }
}

inline int64_t get_component(const int8_t* group_by_buffer,
                             const size_t comp_sz,
                             const size_t index = 0) {
  int64_t ret = std::numeric_limits<int64_t>::min();
  switch (comp_sz) {
    case 1: {
      ret = group_by_buffer[index];
      break;
    }
    case 2: {
      const int16_t* buffer_ptr = reinterpret_cast<const int16_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 4: {
      const int32_t* buffer_ptr = reinterpret_cast<const int32_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 8: {
      const int64_t* buffer_ptr = reinterpret_cast<const int64_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    default:
      CHECK(false);
  }
  return ret;
}

void run_reduction_code(const ReductionCode& reduction_code,
                        int8_t* this_buff,
                        const int8_t* that_buff,
                        const int32_t start_entry_index,
                        const int32_t end_entry_index,
                        const int32_t that_entry_count,
                        const void* this_qmd,
                        const void* that_qmd,
                        const void* serialized_varlen_buffer,
                        const Executor* executor) {
  int err = 0;
  if (reduction_code.func_ptr) {
    err = reduction_code.func_ptr(this_buff,
                                  that_buff,
                                  start_entry_index,
                                  end_entry_index,
                                  that_entry_count,
                                  this_qmd,
                                  that_qmd,
                                  serialized_varlen_buffer);
  } else {
    auto ret = ReductionInterpreter::run(
        reduction_code.ir_reduce_loop.get(),
        {ReductionInterpreter::MakeEvalValue(this_buff),
         ReductionInterpreter::MakeEvalValue(that_buff),
         ReductionInterpreter::MakeEvalValue(start_entry_index),
         ReductionInterpreter::MakeEvalValue(end_entry_index),
         ReductionInterpreter::MakeEvalValue(that_entry_count),
         ReductionInterpreter::MakeEvalValue(this_qmd),
         ReductionInterpreter::MakeEvalValue(that_qmd),
         ReductionInterpreter::MakeEvalValue(serialized_varlen_buffer)},
        executor);
    err = ret.int_val;
  }
  if (err) {
    if (err == Executor::ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES) {
      throw std::runtime_error("Multiple distinct values encountered");
    }
    if (err == Executor::ERR_INTERRUPTED) {
      throw std::runtime_error(
          "Query execution has interrupted during result set reduction");
    }
    throw std::runtime_error(
        "Query execution has exceeded the time limit or was interrupted during result "
        "set reduction");
  }
}

}  // namespace

// Driver method for various buffer layouts, actual work is done by reduceOne* methods.
// Reduces the entries of `this_` into the buffer of `that` ResultSetStorage object.
void ResultSetReduction::reduce(const ResultSetStorage& this_,
                                const ResultSetStorage& that,
                                const std::vector<std::string>& serialized_varlen_buffer,
                                const ReductionCode& reduction_code,
                                const Config& config,
                                const Executor* executor) {
  auto this_query_mem_desc = this_.getQueryMemDesc();
  auto that_query_mem_desc = that.getQueryMemDesc();
  auto entry_count = this_query_mem_desc.getEntryCount();
  CHECK_GT(entry_count, size_t(0));
  if (this_query_mem_desc.didOutputColumnar()) {
    CHECK(this_query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByPerfectHash ||
          this_query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash ||
          this_query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::NonGroupedAggregate);
  }
  const auto that_entry_count = that_query_mem_desc.getEntryCount();
  switch (this_query_mem_desc.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByBaselineHash:
      CHECK_GE(entry_count, that_entry_count);
      break;
    default:
      CHECK_EQ(entry_count, that_entry_count);
  }
  auto this_buff = this_.getUnderlyingBuffer();
  CHECK(this_buff);
  auto that_buff = that.getUnderlyingBuffer();
  CHECK(that_buff);
  CHECK(executor);
  if (this_query_mem_desc.getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    if (!serialized_varlen_buffer.empty()) {
      throw std::runtime_error(
          "Projection of variable length targets with baseline hash group by is not yet "
          "supported in Distributed mode");
    }
    if (use_multithreaded_reduction(that_entry_count)) {
      if (reduction_code.ir_reduce_loop) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, that_entry_count),
                          [&this_query_mem_desc,
                           this_buff,
                           that_buff,
                           that_entry_count,
                           &reduction_code,
                           &that_query_mem_desc,
                           executor](auto r) {
                            run_reduction_code(reduction_code,
                                               this_buff,
                                               that_buff,
                                               r.begin(),
                                               r.end(),
                                               that_entry_count,
                                               &this_query_mem_desc,
                                               &that_query_mem_desc,
                                               nullptr,
                                               executor);
                          });
      } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, that_entry_count),
                          [&this_, &that, this_buff, that_buff, &config](auto r) {
                            for (size_t entry_idx = r.begin(); entry_idx < r.end();
                                 ++entry_idx) {
                              reduceOneEntryBaseline(this_,
                                                     that,
                                                     this_buff,
                                                     that_buff,
                                                     entry_idx,
                                                     config.exec.watchdog.enable_dynamic);
                            }
                          });
      }
    } else {
      if (reduction_code.ir_reduce_loop) {
        run_reduction_code(reduction_code,
                           this_buff,
                           that_buff,
                           0,
                           that_entry_count,
                           that_entry_count,
                           &this_query_mem_desc,
                           &that_query_mem_desc,
                           nullptr,
                           executor);
      } else {
        for (size_t i = 0; i < that_entry_count; ++i) {
          reduceOneEntryBaseline(
              this_, that, this_buff, that_buff, i, config.exec.watchdog.enable_dynamic);
        }
      }
    }
    return;
  }
  if (use_multithreaded_reduction(entry_count)) {
    if (this_query_mem_desc.didOutputColumnar()) {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, entry_count),
          [&this_, &that, this_buff, that_buff, &serialized_varlen_buffer, executor](
              auto r) {
            reduceEntriesNoCollisionsColWise(this_,
                                             that,
                                             this_buff,
                                             that_buff,
                                             r.begin(),
                                             r.end(),
                                             serialized_varlen_buffer,
                                             executor);
          });
    } else {
      CHECK(reduction_code.ir_reduce_loop);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, entry_count),
                        [&this_query_mem_desc,
                         this_buff,
                         that_buff,
                         that_entry_count,
                         &reduction_code,
                         &that_query_mem_desc,
                         &serialized_varlen_buffer,
                         executor](auto r) {
                          run_reduction_code(reduction_code,
                                             this_buff,
                                             that_buff,
                                             r.begin(),
                                             r.end(),
                                             that_entry_count,
                                             &this_query_mem_desc,
                                             &that_query_mem_desc,
                                             &serialized_varlen_buffer,
                                             executor);
                        });
    }
  } else {
    if (this_query_mem_desc.didOutputColumnar()) {
      reduceEntriesNoCollisionsColWise(this_,
                                       that,
                                       this_buff,
                                       that_buff,
                                       0,
                                       this_query_mem_desc.getEntryCount(),
                                       serialized_varlen_buffer,
                                       executor);
    } else {
      CHECK(reduction_code.ir_reduce_loop);
      run_reduction_code(reduction_code,
                         this_buff,
                         that_buff,
                         0,
                         entry_count,
                         that_entry_count,
                         &this_query_mem_desc,
                         &that_query_mem_desc,
                         &serialized_varlen_buffer,
                         executor);
    }
  }
}

namespace {

ALWAYS_INLINE void check_watchdog() {
  if (UNLIKELY(dynamic_watchdog())) {
    // TODO(alex): distinguish between the deadline and interrupt
    throw std::runtime_error(
        "Query execution has exceeded the time limit or was interrupted during result "
        "set reduction");
  }
}

ALWAYS_INLINE void check_watchdog_with_seed(const size_t sample_seed) {
  if (UNLIKELY((sample_seed & 0x3F) == 0 && dynamic_watchdog())) {
    // TODO(alex): distinguish between the deadline and interrupt
    throw std::runtime_error(
        "Query execution has exceeded the time limit or was interrupted during result "
        "set reduction");
  }
}

}  // namespace

void ResultSetReduction::reduceEntriesNoCollisionsColWise(
    const ResultSetStorage& this_,
    const ResultSetStorage& that,
    int8_t* this_buff,
    const int8_t* that_buff,
    const size_t start_index,
    const size_t end_index,
    const std::vector<std::string>& serialized_varlen_buffer,
    const Executor* executor) {
  // TODO(adb / saman): Support column wise output when serializing distributed agg
  // functions
  CHECK(serialized_varlen_buffer.empty());

  auto& query_mem_desc = this_.getQueryMemDesc();
  const auto& col_slot_context = query_mem_desc.getColSlotContext();

  auto this_crt_col_ptr = get_cols_ptr(this_buff, query_mem_desc);
  auto that_crt_col_ptr = get_cols_ptr(that_buff, query_mem_desc);
  for (size_t target_idx = 0; target_idx < this_.getTargetsCount(); ++target_idx) {
    const auto& agg_info = this_.getTarget(target_idx);
    const auto& slots_for_col = col_slot_context.getSlotsForCol(target_idx);

    bool two_slot_target{false};
    if (agg_info.is_agg && (agg_info.agg_kind == hdk::ir::AggType::kAvg ||
                            (agg_info.agg_kind == hdk::ir::AggType::kSample &&
                             (agg_info.type->isString() || agg_info.type->isArray())))) {
      // Note that this assumes if one of the slot pairs in a given target is an array,
      // all slot pairs are arrays.
      two_slot_target = true;
    }
    if (UNLIKELY(
            executor &&
            executor->getConfig().exec.interrupt.enable_non_kernel_time_query_interrupt &&
            executor->checkNonKernelTimeInterrupted())) {
      throw std::runtime_error(
          "Query execution was interrupted during result set reduction");
    }
    if (executor->getConfig().exec.watchdog.enable_dynamic) {
      check_watchdog();
    }
    for (size_t target_slot_idx = slots_for_col.front();
         target_slot_idx < slots_for_col.back() + 1;
         target_slot_idx += 2) {
      const auto this_next_col_ptr = advance_to_next_columnar_target_buff(
          this_crt_col_ptr, query_mem_desc, target_slot_idx);
      const auto that_next_col_ptr = advance_to_next_columnar_target_buff(
          that_crt_col_ptr, query_mem_desc, target_slot_idx);
      for (size_t entry_idx = start_index; entry_idx < end_index; ++entry_idx) {
        if (this_.isEmptyEntryColumnar(entry_idx, that_buff)) {
          continue;
        }
        if (LIKELY(!query_mem_desc.hasKeylessHash())) {
          // copy the key from right hand side
          this_.copyKeyColWise(entry_idx, this_buff, that_buff);
        }
        auto this_ptr1 =
            this_crt_col_ptr +
            entry_idx * query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx);
        auto that_ptr1 =
            that_crt_col_ptr +
            entry_idx * query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx);
        int8_t* this_ptr2{nullptr};
        const int8_t* that_ptr2{nullptr};
        if (UNLIKELY(two_slot_target)) {
          this_ptr2 =
              this_next_col_ptr +
              entry_idx * query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx + 1);
          that_ptr2 =
              that_next_col_ptr +
              entry_idx * query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx + 1);
        }
        reduceOneSlot(this_,
                      that,
                      this_ptr1,
                      this_ptr2,
                      that_ptr1,
                      that_ptr2,
                      agg_info,
                      target_idx,
                      target_slot_idx,
                      target_slot_idx,
                      slots_for_col.front(),
                      serialized_varlen_buffer);
      }

      this_crt_col_ptr = this_next_col_ptr;
      that_crt_col_ptr = that_next_col_ptr;
      if (UNLIKELY(two_slot_target)) {
        this_crt_col_ptr = advance_to_next_columnar_target_buff(
            this_crt_col_ptr, query_mem_desc, target_slot_idx + 1);
        that_crt_col_ptr = advance_to_next_columnar_target_buff(
            that_crt_col_ptr, query_mem_desc, target_slot_idx + 1);
      }
    }
  }
}

namespace {

#ifdef _MSC_VER
#define mapd_cas(address, compare, val)                                      \
  InterlockedCompareExchange64(reinterpret_cast<volatile int64_t*>(address), \
                               static_cast<int64_t>(val),                    \
                               static_cast<int64_t>(compare))
#else
#define mapd_cas(ptr, expected, desired) \
  __atomic_compare_exchange_n(           \
      ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#endif

GroupValueInfo get_matching_group_value_columnar_reduction(int64_t* groups_buffer,
                                                           const uint32_t h,
                                                           const int64_t* key,
                                                           const uint32_t key_qw_count,
                                                           const size_t entry_count) {
  auto off = h;
  auto empty_key = EMPTY_KEY_64;
  if (mapd_cas(&groups_buffer[off], empty_key, *key)) {
    for (size_t i = 0; i < key_qw_count; ++i) {
      groups_buffer[off] = key[i];
      off += entry_count;
    }
    return {&groups_buffer[off], true};
  }
  off = h;
  for (size_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off] != key[i]) {
      return {nullptr, true};
    }
    off += entry_count;
  }
  return {&groups_buffer[off], false};
}

#undef mapd_cas

// TODO(alex): fix synchronization when we enable it
GroupValueInfo get_group_value_columnar_reduction(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count, sizeof(int64_t)) % groups_buffer_entry_count;
  auto matching_gvi = get_matching_group_value_columnar_reduction(
      groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_gvi.first) {
    return matching_gvi;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_gvi = get_matching_group_value_columnar_reduction(
        groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
    if (matching_gvi.first) {
      return matching_gvi;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return {nullptr, true};
}

#ifdef _MSC_VER
template <typename T>
bool hdk_cas(T* ptr, T expected, T desired) {
  if constexpr (sizeof(T) == 4) {
    return InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr),
                                      static_cast<long>(desired),
                                      static_cast<long>(expected)) ==
           static_cast<long>(expected);
  } else if constexpr (sizeof(T) == 8) {
    return InterlockedCompareExchange64(
               reinterpret_cast<volatile int64_t*>(ptr), desired, expected) == expected;
  } else {
    LOG(FATAL) << "Unsupported atomic operation";
  }
}

template <typename T>
void store_cst(T* ptr, T val) {
  if constexpr (sizeof(T) == 4) {
    InterlockedExchange(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val));
  } else if constexpr (sizeof(T) == 8) {
    InterlockedExchange64(reinterpret_cast<volatile int64_t*>(ptr), val);
  } else {
    LOG(FATAL) << "Unsupported atomic operation";
  }
}

template <typename T>
T load_cst(T* ptr) {
  if constexpr (sizeof(T) == 4) {
    return InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr), 0, 0);
  } else if constexpr (sizeof(T) == 8) {
    return InterlockedCompareExchange64(reinterpret_cast<volatile int64_t*>(ptr), 0, 0);
  } else {
    LOG(FATAL) << "Unsupported atomic operation";
  }
}
#else
#define hdk_cas(ptr, expected, desired) \
  __atomic_compare_exchange_n(          \
      ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define store_cst(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define load_cst(ptr) __atomic_load_n(ptr, __ATOMIC_SEQ_CST)
#endif

template <typename T = int64_t>
GroupValueInfo get_matching_group_value_reduction(
    int64_t* groups_buffer,
    const uint32_t h,
    const T* key,
    const uint32_t key_count,
    const QueryMemoryDescriptor& query_mem_desc,
    const int64_t* that_buff_i64,
    const size_t that_entry_idx,
    const size_t that_entry_count,
    const uint32_t row_size_quad) {
  auto off = h * row_size_quad;
  T empty_key = get_empty_key<typename remove_addr_space<T>::type>();
  T write_pending = get_empty_key<typename remove_addr_space<T>::type>() - 1;
  auto row_ptr = reinterpret_cast<T*>(groups_buffer + off);
  const auto slot_off_quad = get_slot_off_quad(query_mem_desc);
  const bool success = hdk_cas(row_ptr, empty_key, write_pending);
  if (success) {
    fill_slots(groups_buffer + off + slot_off_quad,
               query_mem_desc.getEntryCount(),
               that_buff_i64,
               that_entry_idx,
               that_entry_count,
               query_mem_desc);
    if (key_count > 1) {
      memcpy(row_ptr + 1, key + 1, (key_count - 1) * sizeof(T));
    }
    store_cst(row_ptr, *key);
    return {groups_buffer + off + slot_off_quad, true};
  }
  while (load_cst(row_ptr) == write_pending) {
    // spin until the winning thread has finished writing the entire key and the init
    // value
  }
  for (size_t i = 0; i < key_count; ++i) {
    if (load_cst(row_ptr + i) != key[i]) {
      return {nullptr, true};
    }
  }
  return {groups_buffer + off + slot_off_quad, false};
}

#ifndef _MSC_VER
#undef load_cst
#undef store_cst
#undef hdk_cas
#endif

inline GroupValueInfo get_matching_group_value_reduction(
    int64_t* groups_buffer,
    const uint32_t h,
    const int64_t* key,
    const uint32_t key_count,
    const size_t key_width,
    const QueryMemoryDescriptor& query_mem_desc,
    const int64_t* that_buff_i64,
    const size_t that_entry_idx,
    const size_t that_entry_count,
    const uint32_t row_size_quad) {
  switch (key_width) {
    case 4:
      return get_matching_group_value_reduction(groups_buffer,
                                                h,
                                                reinterpret_cast<const int32_t*>(key),
                                                key_count,
                                                query_mem_desc,
                                                that_buff_i64,
                                                that_entry_idx,
                                                that_entry_count,
                                                row_size_quad);
    case 8:
      return get_matching_group_value_reduction(groups_buffer,
                                                h,
                                                key,
                                                key_count,
                                                query_mem_desc,
                                                that_buff_i64,
                                                that_entry_idx,
                                                that_entry_count,
                                                row_size_quad);
    default:
      CHECK(false);
      return {nullptr, true};
  }
}

}  // namespace

GroupValueInfo result_set::get_group_value_reduction(
    int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    const int64_t* key,
    const uint32_t key_count,
    const size_t key_width,
    const QueryMemoryDescriptor& query_mem_desc,
    const int64_t* that_buff_i64,
    const size_t that_entry_idx,
    const size_t that_entry_count,
    const uint32_t row_size_quad) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  auto matching_gvi = get_matching_group_value_reduction(groups_buffer,
                                                         h,
                                                         key,
                                                         key_count,
                                                         key_width,
                                                         query_mem_desc,
                                                         that_buff_i64,
                                                         that_entry_idx,
                                                         that_entry_count,
                                                         row_size_quad);
  if (matching_gvi.first) {
    return matching_gvi;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_gvi = get_matching_group_value_reduction(groups_buffer,
                                                      h_probe,
                                                      key,
                                                      key_count,
                                                      key_width,
                                                      query_mem_desc,
                                                      that_buff_i64,
                                                      that_entry_idx,
                                                      that_entry_count,
                                                      row_size_quad);
    if (matching_gvi.first) {
      return matching_gvi;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return {nullptr, true};
}

// Reduces entry at position that_entry_idx in that_buff into this_buff. This is
// the baseline layout, so the position in this_buff isn't known to be that_entry_idx.
void ResultSetReduction::reduceOneEntryBaseline(const ResultSetStorage& this_,
                                                const ResultSetStorage& that,
                                                int8_t* this_buff,
                                                const int8_t* that_buff,
                                                const size_t that_entry_idx,
                                                bool enable_dynamic_watchdog) {
  if (enable_dynamic_watchdog) {
    check_watchdog_with_seed(that_entry_idx);
  }
  auto& this_query_mem_desc = this_.getQueryMemDesc();
  auto& that_query_mem_desc = that.getQueryMemDesc();
  auto that_entry_count = that_query_mem_desc.getEntryCount();
  const auto key_count = this_query_mem_desc.getGroupbyColCount();
  CHECK(this_query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::GroupByBaselineHash);
  CHECK(!this_query_mem_desc.hasKeylessHash());
  CHECK(this_query_mem_desc.didOutputColumnar());
  const auto key_off =
      key_offset_colwise(that_entry_idx, 0, this_query_mem_desc.didOutputColumnar());
  if (that.isEmptyEntry(that_entry_idx, that_buff)) {
    return;
  }
  auto this_buff_i64 = reinterpret_cast<int64_t*>(this_buff);
  auto that_buff_i64 = reinterpret_cast<const int64_t*>(that_buff);
  const auto key = make_key(&that_buff_i64[key_off], that_entry_count, key_count);
  auto [this_entry_slots, empty_entry] = get_group_value_columnar_reduction(
      this_buff_i64, this_query_mem_desc.getEntryCount(), &key[0], key_count);
  CHECK(this_entry_slots);
  if (empty_entry) {
    fill_slots(this_entry_slots,
               this_query_mem_desc.getEntryCount(),
               that_buff_i64,
               that_entry_idx,
               that_entry_count,
               this_query_mem_desc);
    return;
  }
  reduceOneEntrySlotsBaseline(
      this_, that, this_entry_slots, that_buff_i64, that_entry_idx);
}

void ResultSetReduction::reduceOneEntrySlotsBaseline(const ResultSetStorage& this_,
                                                     const ResultSetStorage& that,
                                                     int64_t* this_entry_slots,
                                                     const int64_t* that_buff,
                                                     const size_t that_entry_idx) {
  auto& this_query_mem_desc = this_.getQueryMemDesc();
  auto& that_query_mem_desc = that.getQueryMemDesc();
  auto that_entry_count = that_query_mem_desc.getEntryCount();
  CHECK(this_query_mem_desc.didOutputColumnar());
  const auto key_count = this_query_mem_desc.getGroupbyColCount();
  size_t j = 0;
  size_t init_agg_val_idx = 0;
  for (size_t target_logical_idx = 0; target_logical_idx < this_.getTargetsCount();
       ++target_logical_idx) {
    const auto& target_info = this_.getTarget(target_logical_idx);
    const auto that_slot_off = slot_offset_colwise(
        that_entry_idx, init_agg_val_idx, key_count, that_entry_count);
    const auto this_slot_off = init_agg_val_idx * this_query_mem_desc.getEntryCount();
    reduceOneSlotBaseline(this_,
                          that,
                          this_entry_slots,
                          this_slot_off,
                          that_buff,
                          that_slot_off,
                          target_info,
                          target_logical_idx,
                          j,
                          init_agg_val_idx);
    if (this_query_mem_desc.targetGroupbyIndicesSize() == 0) {
      init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
    } else {
      if (this_query_mem_desc.getTargetGroupbyIndex(target_logical_idx) < 0) {
        init_agg_val_idx = advance_slot(init_agg_val_idx, target_info, false);
      }
    }
    j = advance_slot(j, target_info, false);
  }
}

void ResultSetReduction::reduceOneSlotBaseline(const ResultSetStorage& this_,
                                               const ResultSetStorage& that,
                                               int64_t* this_buff,
                                               const size_t this_slot,
                                               const int64_t* that_buff,
                                               const size_t that_slot,
                                               const TargetInfo& target_info,
                                               const size_t target_logical_idx,
                                               const size_t target_slot_idx,
                                               const size_t init_agg_val_idx) {
  auto& this_query_mem_desc = this_.getQueryMemDesc();
  auto& that_query_mem_desc = that.getQueryMemDesc();
  CHECK(this_query_mem_desc.didOutputColumnar());
  int8_t* this_ptr2{nullptr};
  const int8_t* that_ptr2{nullptr};
  if (target_info.is_agg &&
      (target_info.agg_kind == hdk::ir::AggType::kAvg ||
       (target_info.agg_kind == hdk::ir::AggType::kSample &&
        (target_info.type->isString() || target_info.type->isArray())))) {
    const auto this_count_off = this_query_mem_desc.getEntryCount();
    const auto that_count_off = that_query_mem_desc.getEntryCount();
    this_ptr2 = reinterpret_cast<int8_t*>(&this_buff[this_slot + this_count_off]);
    that_ptr2 = reinterpret_cast<const int8_t*>(&that_buff[that_slot + that_count_off]);
  }
  reduceOneSlot(this_,
                that,
                reinterpret_cast<int8_t*>(&this_buff[this_slot]),
                this_ptr2,
                reinterpret_cast<const int8_t*>(&that_buff[that_slot]),
                that_ptr2,
                target_info,
                target_logical_idx,
                target_slot_idx,
                init_agg_val_idx,
                target_slot_idx,  // dummy, for now
                {});
}

// During the reduction of two result sets using the baseline strategy, we first create a
// big enough buffer to hold the entries for both and we move the entries from the first
// into it before doing the reduction as usual (into the first buffer).
template <class KeyType>
void ResultSetReduction::moveEntriesToBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                             const int8_t* old_buff,
                                             int8_t* new_buff,
                                             const size_t new_entry_count) {
  CHECK(!query_mem_desc.hasKeylessHash());
  CHECK_GT(new_entry_count, query_mem_desc.getEntryCount());
  auto new_buff_i64 = reinterpret_cast<int64_t*>(new_buff);
  const auto key_count = query_mem_desc.getGroupbyColCount();
  CHECK(QueryDescriptionType::GroupByBaselineHash ==
        query_mem_desc.getQueryDescriptionType());
  const auto src_buff = reinterpret_cast<const int64_t*>(old_buff);
  const auto row_qw_count = get_row_qw_count(query_mem_desc);
  const auto key_byte_width = query_mem_desc.getEffectiveKeyWidth();

  if (use_multithreaded_reduction(query_mem_desc.getEntryCount())) {
    const size_t thread_count = cpu_threads();
    std::vector<std::future<void>> move_threads;

    for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
      const auto thread_entry_count =
          (query_mem_desc.getEntryCount() + thread_count - 1) / thread_count;
      const auto start_index = thread_idx * thread_entry_count;
      const auto end_index =
          std::min(start_index + thread_entry_count, query_mem_desc.getEntryCount());
      move_threads.emplace_back(std::async(
          std::launch::async,
          [&query_mem_desc,
           src_buff,
           new_buff_i64,
           new_entry_count,
           start_index,
           end_index,
           key_count,
           row_qw_count,
           key_byte_width] {
            for (size_t entry_idx = start_index; entry_idx < end_index; ++entry_idx) {
              moveOneEntryToBuffer<KeyType>(query_mem_desc,
                                            entry_idx,
                                            new_buff_i64,
                                            new_entry_count,
                                            key_count,
                                            row_qw_count,
                                            src_buff,
                                            key_byte_width);
            }
          }));
    }
    for (auto& move_thread : move_threads) {
      move_thread.wait();
    }
    for (auto& move_thread : move_threads) {
      move_thread.get();
    }
  } else {
    for (size_t entry_idx = 0; entry_idx < query_mem_desc.getEntryCount(); ++entry_idx) {
      moveOneEntryToBuffer<KeyType>(query_mem_desc,
                                    entry_idx,
                                    new_buff_i64,
                                    new_entry_count,
                                    key_count,
                                    row_qw_count,
                                    src_buff,
                                    key_byte_width);
    }
  }
}

template <class KeyType>
void ResultSetReduction::moveOneEntryToBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                              const size_t entry_index,
                                              int64_t* new_buff_i64,
                                              const size_t new_entry_count,
                                              const size_t key_count,
                                              const size_t row_qw_count,
                                              const int64_t* src_buff,
                                              const size_t key_byte_width) {
  const auto key_off =
      query_mem_desc.didOutputColumnar()
          ? key_offset_colwise(entry_index, 0, query_mem_desc.getEntryCount())
          : row_qw_count * entry_index;
  const auto key_ptr = reinterpret_cast<const KeyType*>(&src_buff[key_off]);
  if (*key_ptr == get_empty_key<typename remove_addr_space<KeyType>::type>()) {
    return;
  }
  int64_t* new_entries_ptr{nullptr};
  if (query_mem_desc.didOutputColumnar()) {
    const auto key =
        make_key(&src_buff[key_off], query_mem_desc.getEntryCount(), key_count);
    new_entries_ptr =
        get_group_value_columnar(new_buff_i64, new_entry_count, &key[0], key_count);
  } else {
    new_entries_ptr = get_group_value(new_buff_i64,
                                      new_entry_count,
                                      &src_buff[key_off],
                                      key_count,
                                      key_byte_width,
                                      row_qw_count);
  }
  CHECK(new_entries_ptr);
  fill_slots(new_entries_ptr,
             new_entry_count,
             src_buff,
             entry_index,
             query_mem_desc.getEntryCount(),
             query_mem_desc);
}

// Driver for reductions. Needed because the result of a reduction on the baseline
// layout, which can have collisions, cannot be done in place and something needs
// to take the ownership of the new result set with the bigger underlying buffer.
ResultSet* ResultSetManager::reduce(std::vector<ResultSet*>& result_sets,
                                    const Config& config,
                                    Executor* executor) {
  CHECK(!result_sets.empty());
  auto result_rs = result_sets.front();
  CHECK(result_rs->getStorage());
  auto& first_result = *result_rs->getStorage();
  auto result = &first_result;
  const auto row_set_mem_owner = result_rs->getRowSetMemOwner();
  for (const auto result_set : result_sets) {
    CHECK_EQ(row_set_mem_owner, result_set->getRowSetMemOwner());
  }
  if (first_result.getQueryMemDesc().getQueryDescriptionType() ==
      QueryDescriptionType::GroupByBaselineHash) {
    const auto total_entry_count =
        std::accumulate(result_sets.begin(),
                        result_sets.end(),
                        size_t(0),
                        [](const size_t init, const ResultSet* rs) {
                          return init + rs->getQueryMemDesc().getEntryCount();
                        });
    CHECK(total_entry_count);
    auto query_mem_desc = first_result.getQueryMemDesc();
    query_mem_desc.setEntryCount(total_entry_count);
    rs_.reset(new ResultSet(first_result.getTargets(),
                            ExecutorDeviceType::CPU,
                            query_mem_desc,
                            row_set_mem_owner,
                            result_rs->getDataManager(),
                            0,
                            0));
    auto result_storage = rs_->allocateStorage(first_result.getInitVals());
    rs_->initializeStorage();
    switch (query_mem_desc.getEffectiveKeyWidth()) {
      case 4:
        ResultSetReduction::moveEntriesToBuffer<int32_t>(
            first_result.getQueryMemDesc(),
            first_result.getUnderlyingBuffer(),
            result_storage->getUnderlyingBuffer(),
            query_mem_desc.getEntryCount());
        break;
      case 8:
        ResultSetReduction::moveEntriesToBuffer<int64_t>(
            first_result.getQueryMemDesc(),
            first_result.getUnderlyingBuffer(),
            result_storage->getUnderlyingBuffer(),
            query_mem_desc.getEntryCount());
        break;
      default:
        CHECK(false);
    }
    result = rs_->getStorage();
    result_rs = rs_.get();
  }

  auto serialized_varlen_buffer = result_sets.front()->getSerializedVarlenBuffer();
  if (!serialized_varlen_buffer.empty()) {
    result->rewriteAggregateBufferOffsets(serialized_varlen_buffer.front());
    for (auto result_it = result_sets.begin() + 1; result_it != result_sets.end();
         ++result_it) {
      auto& result_serialized_varlen_buffer = (*result_it)->getSerializedVarlenBuffer();
      CHECK_EQ(result_serialized_varlen_buffer.size(), size_t(1));
      serialized_varlen_buffer.emplace_back(result_serialized_varlen_buffer.front());
    }
  }

  ResultSetReductionJIT reduction_jit(result_rs->getQueryMemDesc(),
                                      result_rs->getTargetInfos(),
                                      result_rs->getTargetInitVals(),
                                      config,
                                      executor);
  auto reduction_code = reduction_jit.codegen();
  size_t ctr = 1;
  for (auto result_it = result_sets.begin() + 1; result_it != result_sets.end();
       ++result_it) {
    if (!serialized_varlen_buffer.empty()) {
      ResultSetReduction::reduce(*result,
                                 *((*result_it)->getStorage()),
                                 serialized_varlen_buffer[ctr++],
                                 reduction_code,
                                 config,
                                 executor);
    } else {
      ResultSetReduction::reduce(
          *result, *((*result_it)->getStorage()), {}, reduction_code, config, executor);
    }
  }
  return result_rs;
}

std::shared_ptr<ResultSet> ResultSetManager::getOwnResultSet() {
  return rs_;
}

void ResultSetManager::rewriteVarlenAggregates(ResultSet* result_rs) {
  auto result_storage = result_rs->getStorage();
  result_storage->rewriteAggregateBufferOffsets(
      result_rs->getSerializedVarlenBuffer().front());
}

#define AGGREGATE_ONE_VALUE(                                                      \
    agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__)               \
  do {                                                                            \
    const auto type = get_compact_type(agg_info__);                               \
    if (type->isFloatingPoint()) {                                                \
      if (chosen_bytes__ == sizeof(float)) {                                      \
        agg_##agg_kind__##_float(reinterpret_cast<int32_t*>(val_ptr__),           \
                                 *reinterpret_cast<const float*>(other_ptr__));   \
      } else {                                                                    \
        agg_##agg_kind__##_double(reinterpret_cast<int64_t*>(val_ptr__),          \
                                  *reinterpret_cast<const double*>(other_ptr__)); \
      }                                                                           \
    } else {                                                                      \
      if (chosen_bytes__ == sizeof(int32_t)) {                                    \
        auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                     \
        auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);           \
        agg_##agg_kind__##_int32(val_ptr, *other_ptr);                            \
      } else {                                                                    \
        auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                     \
        auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);           \
        agg_##agg_kind__(val_ptr, *other_ptr);                                    \
      }                                                                           \
    }                                                                             \
  } while (0)

#define AGGREGATE_ONE_NULLABLE_VALUE(                                               \
    agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)     \
  do {                                                                              \
    if (agg_info__.skip_null_val) {                                                 \
      const auto type = get_compact_type(agg_info__);                               \
      if (type->isFloatingPoint()) {                                                \
        if (chosen_bytes__ == sizeof(float)) {                                      \
          agg_##agg_kind__##_float_skip_val(                                        \
              reinterpret_cast<int32_t*>(val_ptr__),                                \
              *reinterpret_cast<const float*>(other_ptr__),                         \
              *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));         \
        } else {                                                                    \
          agg_##agg_kind__##_double_skip_val(                                       \
              reinterpret_cast<int64_t*>(val_ptr__),                                \
              *reinterpret_cast<const double*>(other_ptr__),                        \
              *reinterpret_cast<const double*>(may_alias_ptr(&init_val__)));        \
        }                                                                           \
      } else {                                                                      \
        if (chosen_bytes__ == sizeof(int32_t)) {                                    \
          int32_t* val_ptr = reinterpret_cast<int32_t*>(val_ptr__);                 \
          const int32_t* other_ptr = reinterpret_cast<const int32_t*>(other_ptr__); \
          const auto null_val = static_cast<int32_t>(init_val__);                   \
          agg_##agg_kind__##_int32_skip_val(val_ptr, *other_ptr, null_val);         \
        } else {                                                                    \
          int64_t* val_ptr = reinterpret_cast<int64_t*>(val_ptr__);                 \
          const int64_t* other_ptr = reinterpret_cast<const int64_t*>(other_ptr__); \
          const auto null_val = static_cast<int64_t>(init_val__);                   \
          agg_##agg_kind__##_skip_val(val_ptr, *other_ptr, null_val);               \
        }                                                                           \
      }                                                                             \
    } else {                                                                        \
      AGGREGATE_ONE_VALUE(                                                          \
          agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__);          \
    }                                                                               \
  } while (0)

#define AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__)   \
  do {                                                                \
    if (chosen_bytes__ == sizeof(int32_t)) {                          \
      auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);           \
      auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__); \
      agg_sum_int32(val_ptr, *other_ptr);                             \
    } else {                                                          \
      auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);           \
      auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__); \
      agg_sum(val_ptr, *other_ptr);                                   \
    }                                                                 \
  } while (0)

#define AGGREGATE_ONE_NULLABLE_COUNT(                                        \
    val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)          \
  {                                                                          \
    if (agg_info__.skip_null_val) {                                          \
      const auto type = get_compact_type(agg_info__);                        \
      if (type->isFloatingPoint()) {                                         \
        if (chosen_bytes__ == sizeof(float)) {                               \
          agg_sum_float_skip_val(                                            \
              reinterpret_cast<int32_t*>(val_ptr__),                         \
              *reinterpret_cast<const float*>(other_ptr__),                  \
              *reinterpret_cast<const float*>(may_alias_ptr(&init_val__)));  \
        } else {                                                             \
          agg_sum_double_skip_val(                                           \
              reinterpret_cast<int64_t*>(val_ptr__),                         \
              *reinterpret_cast<const double*>(other_ptr__),                 \
              *reinterpret_cast<const double*>(may_alias_ptr(&init_val__))); \
        }                                                                    \
      } else {                                                               \
        if (chosen_bytes__ == sizeof(int32_t)) {                             \
          auto val_ptr = reinterpret_cast<int32_t*>(val_ptr__);              \
          auto other_ptr = reinterpret_cast<const int32_t*>(other_ptr__);    \
          const auto null_val = static_cast<int32_t>(init_val__);            \
          agg_sum_int32_skip_val(val_ptr, *other_ptr, null_val);             \
        } else {                                                             \
          auto val_ptr = reinterpret_cast<int64_t*>(val_ptr__);              \
          auto other_ptr = reinterpret_cast<const int64_t*>(other_ptr__);    \
          const auto null_val = static_cast<int64_t>(init_val__);            \
          agg_sum_skip_val(val_ptr, *other_ptr, null_val);                   \
        }                                                                    \
      }                                                                      \
    } else {                                                                 \
      AGGREGATE_ONE_COUNT(val_ptr__, other_ptr__, chosen_bytes__);           \
    }                                                                        \
  }

// to be used for 8/16-bit kMin and kMax only
#define AGGREGATE_ONE_VALUE_SMALL(                                    \
    agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__)   \
  do {                                                                \
    if (chosen_bytes__ == sizeof(int16_t)) {                          \
      auto val_ptr = reinterpret_cast<int16_t*>(val_ptr__);           \
      auto other_ptr = reinterpret_cast<const int16_t*>(other_ptr__); \
      agg_##agg_kind__##_int16(val_ptr, *other_ptr);                  \
    } else if (chosen_bytes__ == sizeof(int8_t)) {                    \
      auto val_ptr = reinterpret_cast<int8_t*>(val_ptr__);            \
      auto other_ptr = reinterpret_cast<const int8_t*>(other_ptr__);  \
      agg_##agg_kind__##_int8(val_ptr, *other_ptr);                   \
    } else {                                                          \
      UNREACHABLE();                                                  \
    }                                                                 \
  } while (0)

// to be used for 8/16-bit kMin and kMax only
#define AGGREGATE_ONE_NULLABLE_VALUE_SMALL(                                       \
    agg_kind__, val_ptr__, other_ptr__, init_val__, chosen_bytes__, agg_info__)   \
  do {                                                                            \
    if (agg_info__.skip_null_val) {                                               \
      if (chosen_bytes__ == sizeof(int16_t)) {                                    \
        int16_t* val_ptr = reinterpret_cast<int16_t*>(val_ptr__);                 \
        const int16_t* other_ptr = reinterpret_cast<const int16_t*>(other_ptr__); \
        const auto null_val = static_cast<int16_t>(init_val__);                   \
        agg_##agg_kind__##_int16_skip_val(val_ptr, *other_ptr, null_val);         \
      } else if (chosen_bytes == sizeof(int8_t)) {                                \
        int8_t* val_ptr = reinterpret_cast<int8_t*>(val_ptr__);                   \
        const int8_t* other_ptr = reinterpret_cast<const int8_t*>(other_ptr__);   \
        const auto null_val = static_cast<int8_t>(init_val__);                    \
        agg_##agg_kind__##_int8_skip_val(val_ptr, *other_ptr, null_val);          \
      }                                                                           \
    } else {                                                                      \
      AGGREGATE_ONE_VALUE_SMALL(                                                  \
          agg_kind__, val_ptr__, other_ptr__, chosen_bytes__, agg_info__);        \
    }                                                                             \
  } while (0)

int8_t result_set::get_width_for_slot(const size_t target_slot_idx,
                                      const bool float_argument_input,
                                      const QueryMemoryDescriptor& query_mem_desc) {
  if (float_argument_input) {
    return sizeof(float);
  }
  return query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx);
}

void ResultSetReduction::reduceOneSlotSingleValue(
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* this_ptr1,
    const TargetInfo& target_info,
    const size_t target_slot_idx,
    const int64_t init_val,
    const int8_t* that_ptr1) {
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes = result_set::get_width_for_slot(
      target_slot_idx, float_argument_input, query_mem_desc);

  auto reduce = [&](auto const& size_tag) {
    using CastTarget = std::decay_t<decltype(size_tag)>;
    const auto lhs_proj_col = *reinterpret_cast<const CastTarget*>(this_ptr1);
    const auto rhs_proj_col = *reinterpret_cast<const CastTarget*>(that_ptr1);
    if (rhs_proj_col == init_val) {
      // ignore
    } else if (lhs_proj_col == init_val) {
      *reinterpret_cast<CastTarget*>(this_ptr1) = rhs_proj_col;
    } else if (lhs_proj_col != rhs_proj_col) {
      throw std::runtime_error("Multiple distinct values encountered");
    }
  };

  switch (chosen_bytes) {
    case 1: {
      CHECK(query_mem_desc.isLogicalSizedColumnsAllowed());
      reduce(int8_t());
      break;
    }
    case 2: {
      CHECK(query_mem_desc.isLogicalSizedColumnsAllowed());
      reduce(int16_t());
      break;
    }
    case 4: {
      reduce(int32_t());
      break;
    }
    case 8: {
      CHECK(!target_info.type->isString() && !target_info.type->isArray());
      reduce(int64_t());
      break;
    }
    default:
      LOG(FATAL) << "Invalid slot width: " << chosen_bytes;
  }
}

void ResultSetReduction::reduceOneSlot(
    const ResultSetStorage& this_,
    const ResultSetStorage& that,
    int8_t* this_ptr1,
    int8_t* this_ptr2,
    const int8_t* that_ptr1,
    const int8_t* that_ptr2,
    const TargetInfo& target_info,
    const size_t target_logical_idx,
    const size_t target_slot_idx,
    const size_t init_agg_val_idx,
    const size_t first_slot_idx_for_target,
    const std::vector<std::string>& serialized_varlen_buffer) {
  auto& query_mem_desc = this_.getQueryMemDesc();
  if (query_mem_desc.targetGroupbyIndicesSize() > 0) {
    if (query_mem_desc.getTargetGroupbyIndex(target_logical_idx) >= 0) {
      return;
    }
  }
  CHECK_LT(init_agg_val_idx, this_.getInitValsCount());
  const bool float_argument_input = takes_float_argument(target_info);
  const auto chosen_bytes = result_set::get_width_for_slot(
      target_slot_idx, float_argument_input, query_mem_desc);
  int64_t init_val = this_.getInitVal(init_agg_val_idx);  // skip_val for nullable types

  if (target_info.is_agg && target_info.agg_kind == hdk::ir::AggType::kSingleValue) {
    auto init_val = this_.getInitVal(init_agg_val_idx);
    reduceOneSlotSingleValue(
        query_mem_desc, this_ptr1, target_info, target_logical_idx, init_val, that_ptr1);
  } else if (target_info.is_agg && target_info.agg_kind != hdk::ir::AggType::kSample) {
    switch (target_info.agg_kind) {
      case hdk::ir::AggType::kCount:
      case hdk::ir::AggType::kApproxCountDistinct: {
        if (is_distinct_target(target_info)) {
          CHECK_EQ(static_cast<size_t>(chosen_bytes), sizeof(int64_t));
          reduceOneCountDistinctSlot(
              this_, that, this_ptr1, that_ptr1, target_logical_idx);
          break;
        }
        CHECK_EQ(int64_t(0), init_val);
        AGGREGATE_ONE_COUNT(this_ptr1, that_ptr1, chosen_bytes);
        break;
      }
      case hdk::ir::AggType::kAvg: {
        // Ignore float argument compaction for count component for fear of its overflow
        AGGREGATE_ONE_COUNT(this_ptr2,
                            that_ptr2,
                            query_mem_desc.getPaddedSlotWidthBytes(target_slot_idx));
      }
      // fall thru
      case hdk::ir::AggType::kSum: {
        AGGREGATE_ONE_NULLABLE_VALUE(
            sum, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        break;
      }
      case hdk::ir::AggType::kMin: {
        if (static_cast<size_t>(chosen_bytes) <= sizeof(int16_t)) {
          AGGREGATE_ONE_NULLABLE_VALUE_SMALL(
              min, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        } else {
          AGGREGATE_ONE_NULLABLE_VALUE(
              min, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        }
        break;
      }
      case hdk::ir::AggType::kMax: {
        if (static_cast<size_t>(chosen_bytes) <= sizeof(int16_t)) {
          AGGREGATE_ONE_NULLABLE_VALUE_SMALL(
              max, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        } else {
          AGGREGATE_ONE_NULLABLE_VALUE(
              max, this_ptr1, that_ptr1, init_val, chosen_bytes, target_info);
        }
        break;
      }
      case hdk::ir::AggType::kApproxQuantile:
        CHECK_EQ(static_cast<int8_t>(sizeof(int64_t)), chosen_bytes);
        reduceOneApproxQuantileSlot(
            query_mem_desc, this_ptr1, that_ptr1, target_logical_idx);
        break;
      case hdk::ir::AggType::kTopK:
        CHECK_EQ(static_cast<int8_t>(sizeof(int64_t)), chosen_bytes);
        reduceOneTopKSlot(this_ptr1,
                          that_ptr1,
                          target_info.agg_arg_type->canonicalSize(),
                          target_info.agg_arg_type->isFloatingPoint(),
                          target_info.topk_param,
                          target_info.topk_inline_buffer);
        break;
      default:
        UNREACHABLE() << toString(target_info.agg_kind);
    }
  } else {
    switch (chosen_bytes) {
      case 1: {
        CHECK(query_mem_desc.isLogicalSizedColumnsAllowed());
        const auto rhs_proj_col = *reinterpret_cast<const int8_t*>(that_ptr1);
        if (rhs_proj_col != init_val) {
          *reinterpret_cast<int8_t*>(this_ptr1) = rhs_proj_col;
        }
        break;
      }
      case 2: {
        CHECK(query_mem_desc.isLogicalSizedColumnsAllowed());
        const auto rhs_proj_col = *reinterpret_cast<const int16_t*>(that_ptr1);
        if (rhs_proj_col != init_val) {
          *reinterpret_cast<int16_t*>(this_ptr1) = rhs_proj_col;
        }
        break;
      }
      case 4: {
        CHECK(target_info.agg_kind != hdk::ir::AggType::kSample ||
              query_mem_desc.isLogicalSizedColumnsAllowed());
        const auto rhs_proj_col = *reinterpret_cast<const int32_t*>(that_ptr1);
        if (rhs_proj_col != init_val) {
          *reinterpret_cast<int32_t*>(this_ptr1) = rhs_proj_col;
        }
        break;
      }
      case 8: {
        auto rhs_proj_col = *reinterpret_cast<const int64_t*>(that_ptr1);
        if ((target_info.agg_kind == hdk::ir::AggType::kSample &&
             (target_info.type->isString() || target_info.type->isArray())) &&
            !serialized_varlen_buffer.empty()) {
          size_t length_to_elems{0};
          length_to_elems =
              target_info.type->isString()
                  ? 1
                  : target_info.type->as<hdk::ir::ArrayBaseType>()->elemType()->size();

          CHECK_LT(static_cast<size_t>(rhs_proj_col), serialized_varlen_buffer.size());
          const auto& varlen_bytes_str = serialized_varlen_buffer[rhs_proj_col];
          const auto str_ptr = reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
          *reinterpret_cast<int64_t*>(this_ptr1) =
              reinterpret_cast<const int64_t>(str_ptr);
          *reinterpret_cast<int64_t*>(this_ptr2) =
              static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
        } else {
          if (rhs_proj_col != init_val) {
            *reinterpret_cast<int64_t*>(this_ptr1) = rhs_proj_col;
            if ((target_info.agg_kind == hdk::ir::AggType::kSample &&
                 (target_info.type->isString() || target_info.type->isArray()))) {
              CHECK(this_ptr2 && that_ptr2);
              *reinterpret_cast<int64_t*>(this_ptr2) =
                  *reinterpret_cast<const int64_t*>(that_ptr2);
            }
          }
        }

        break;
      }
      default:
        LOG(FATAL) << "Invalid slot width: " << chosen_bytes;
    }
  }
}

void ResultSetReduction::reduceOneApproxQuantileSlot(
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* this_ptr1,
    const int8_t* that_ptr1,
    const size_t target_logical_idx) {
  CHECK_LT(target_logical_idx, query_mem_desc.getCountDistinctDescriptorsSize());
  static_assert(sizeof(int64_t) == sizeof(quantile::TDigest*));
  auto* incoming = *reinterpret_cast<quantile::TDigest* const*>(that_ptr1);
  CHECK(incoming) << "this_ptr1=" << (void*)this_ptr1
                  << ", that_ptr1=" << (void const*)that_ptr1
                  << ", target_logical_idx=" << target_logical_idx;
  if (incoming->centroids().capacity()) {
    auto* accumulator = *reinterpret_cast<quantile::TDigest**>(this_ptr1);
    CHECK(accumulator) << "this_ptr1=" << (void*)this_ptr1
                       << ", that_ptr1=" << (void const*)that_ptr1
                       << ", target_logical_idx=" << target_logical_idx;
    accumulator->allocate();
    accumulator->mergeTDigest(*incoming);
  }
}

void ResultSetReduction::reduceOneCountDistinctSlot(const ResultSetStorage& this_,
                                                    const ResultSetStorage& that,
                                                    int8_t* this_ptr1,
                                                    const int8_t* that_ptr1,
                                                    const size_t target_logical_idx) {
  auto this_query_mem_desc = this_.getQueryMemDesc();
  auto that_query_mem_desc = that.getQueryMemDesc();
  CHECK_LT(target_logical_idx, this_query_mem_desc.getCountDistinctDescriptorsSize());
  const auto& old_count_distinct_desc =
      this_query_mem_desc.getCountDistinctDescriptor(target_logical_idx);
  CHECK(old_count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
  const auto& new_count_distinct_desc =
      that_query_mem_desc.getCountDistinctDescriptor(target_logical_idx);
  CHECK(old_count_distinct_desc.impl_type_ == new_count_distinct_desc.impl_type_);
  CHECK(this_ptr1 && that_ptr1);
  auto old_set_ptr = reinterpret_cast<const int64_t*>(this_ptr1);
  auto new_set_ptr = reinterpret_cast<const int64_t*>(that_ptr1);
  count_distinct_set_union(
      *new_set_ptr, *old_set_ptr, new_count_distinct_desc, old_count_distinct_desc);
}

void ResultSetReduction::reduceOneTopKSlot(int8_t* this_ptr,
                                           const int8_t* that_ptr,
                                           int elem_size,
                                           bool is_fp,
                                           int topk_param,
                                           bool inline_buffer) {
  auto this_handle = reinterpret_cast<int64_t>(this_ptr);
  auto that_handle = reinterpret_cast<int64_t>(that_ptr);
  topk_reduce_jit_rt(
      this_handle, that_handle, elem_size, is_fp, topk_param, inline_buffer);
}

bool ResultSetReduction::reduceSingleRow(const int8_t* row_ptr,
                                         const int8_t warp_count,
                                         const bool is_columnar,
                                         const bool replace_bitmap_ptr_with_bitmap_sz,
                                         std::vector<int64_t>& agg_vals,
                                         const QueryMemoryDescriptor& query_mem_desc,
                                         const std::vector<TargetInfo>& targets,
                                         const std::vector<int64_t>& agg_init_vals) {
  const size_t agg_col_count{agg_vals.size()};
  const auto row_size = query_mem_desc.getRowSize();
  CHECK_EQ(agg_col_count, query_mem_desc.getSlotCount());
  CHECK_GE(agg_col_count, targets.size());
  CHECK_EQ(is_columnar, query_mem_desc.didOutputColumnar());
  CHECK(query_mem_desc.hasKeylessHash());
  std::vector<int64_t> partial_agg_vals(agg_col_count, 0);
  bool discard_row = true;
  for (int8_t warp_idx = 0; warp_idx < warp_count; ++warp_idx) {
    bool discard_partial_result = true;
    for (size_t target_idx = 0, agg_col_idx = 0;
         target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes = float_argument_input
                                    ? sizeof(float)
                                    : query_mem_desc.getPaddedSlotWidthBytes(agg_col_idx);
      auto partial_bin_val = get_component(
          row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx), chosen_bytes);
      partial_agg_vals[agg_col_idx] = partial_bin_val;
      if (is_distinct_target(agg_info)) {
        CHECK_EQ(int8_t(1), warp_count);
        CHECK(agg_info.is_agg &&
              (agg_info.agg_kind == hdk::ir::AggType::kCount ||
               agg_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct));
        partial_bin_val = count_distinct_set_size(
            partial_bin_val, query_mem_desc.getCountDistinctDescriptor(target_idx));
        if (replace_bitmap_ptr_with_bitmap_sz) {
          partial_agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
      if (hdk::ir::AggType::kAvg == agg_info.agg_kind) {
        CHECK(agg_info.is_agg && !agg_info.is_distinct);
        ++agg_col_idx;
        partial_bin_val = partial_agg_vals[agg_col_idx] =
            get_component(row_ptr + query_mem_desc.getColOnlyOffInBytes(agg_col_idx),
                          query_mem_desc.getPaddedSlotWidthBytes(agg_col_idx));
      }
      if (agg_col_idx == static_cast<size_t>(query_mem_desc.getTargetIdxForKey()) &&
          partial_bin_val != agg_init_vals[query_mem_desc.getTargetIdxForKey()]) {
        CHECK(agg_info.is_agg);
        discard_partial_result = false;
      }
    }
    row_ptr += row_size;
    if (discard_partial_result) {
      continue;
    }
    discard_row = false;
    for (size_t target_idx = 0, agg_col_idx = 0;
         target_idx < targets.size() && agg_col_idx < agg_col_count;
         ++target_idx, ++agg_col_idx) {
      auto partial_bin_val = partial_agg_vals[agg_col_idx];
      const auto& agg_info = targets[target_idx];
      const bool float_argument_input = takes_float_argument(agg_info);
      const auto chosen_bytes = float_argument_input
                                    ? sizeof(float)
                                    : query_mem_desc.getPaddedSlotWidthBytes(agg_col_idx);
      auto chosen_type = get_compact_type(agg_info);
      if (agg_info.is_agg && agg_info.agg_kind != hdk::ir::AggType::kSample) {
        try {
          switch (agg_info.agg_kind) {
            case hdk::ir::AggType::kCount:
            case hdk::ir::AggType::kApproxCountDistinct:
              AGGREGATE_ONE_NULLABLE_COUNT(
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            case hdk::ir::AggType::kAvg:
              // Ignore float argument compaction for count component for fear of its
              // overflow
              AGGREGATE_ONE_COUNT(
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx + 1]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx + 1]),
                  query_mem_desc.getPaddedSlotWidthBytes(agg_col_idx));
            // fall thru
            case hdk::ir::AggType::kSum:
              AGGREGATE_ONE_NULLABLE_VALUE(
                  sum,
                  reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                  reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                  agg_init_vals[agg_col_idx],
                  chosen_bytes,
                  agg_info);
              break;
            case hdk::ir::AggType::kMin:
              if (static_cast<size_t>(chosen_bytes) <= sizeof(int16_t)) {
                AGGREGATE_ONE_NULLABLE_VALUE_SMALL(
                    min,
                    reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                    reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                    agg_init_vals[agg_col_idx],
                    chosen_bytes,
                    agg_info);
              } else {
                AGGREGATE_ONE_NULLABLE_VALUE(
                    min,
                    reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                    reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                    agg_init_vals[agg_col_idx],
                    chosen_bytes,
                    agg_info);
              }
              break;
            case hdk::ir::AggType::kMax:
              if (static_cast<size_t>(chosen_bytes) <= sizeof(int16_t)) {
                AGGREGATE_ONE_NULLABLE_VALUE_SMALL(
                    max,
                    reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                    reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                    agg_init_vals[agg_col_idx],
                    chosen_bytes,
                    agg_info);
              } else {
                AGGREGATE_ONE_NULLABLE_VALUE(
                    max,
                    reinterpret_cast<int8_t*>(&agg_vals[agg_col_idx]),
                    reinterpret_cast<int8_t*>(&partial_agg_vals[agg_col_idx]),
                    agg_init_vals[agg_col_idx],
                    chosen_bytes,
                    agg_info);
              }
              break;
            default:
              CHECK(false);
              break;
          }
        } catch (std::runtime_error& e) {
          // TODO(miyu): handle the case where chosen_bytes < 8
          LOG(ERROR) << e.what();
        }
        if (chosen_type->isInteger() || chosen_type->isDecimal()) {
          switch (chosen_bytes) {
            case 8:
              break;
            case 4: {
              int32_t ret = *reinterpret_cast<const int32_t*>(&agg_vals[agg_col_idx]);
              if (!(agg_info.agg_kind == hdk::ir::AggType::kCount &&
                    ret != agg_init_vals[agg_col_idx])) {
                agg_vals[agg_col_idx] = static_cast<int64_t>(ret);
              }
              break;
            }
            default:
              CHECK(false);
          }
        }
        if (hdk::ir::AggType::kAvg == agg_info.agg_kind) {
          ++agg_col_idx;
        }
      } else {
        if (agg_info.agg_kind == hdk::ir::AggType::kSample) {
          CHECK(!agg_info.type->isString() && !agg_info.type->isArray())
              << "Interleaved bins reduction not supported for variable length "
                 "arguments "
                 "to SAMPLE";
        }
        if (agg_vals[agg_col_idx]) {
          if (agg_info.agg_kind == hdk::ir::AggType::kSample) {
            continue;
          }
          CHECK_EQ(agg_vals[agg_col_idx], partial_bin_val);
        } else {
          agg_vals[agg_col_idx] = partial_bin_val;
        }
      }
    }
  }
  return discard_row;
}
