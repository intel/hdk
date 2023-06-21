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

/**
 * @file    ResultSetSort.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Efficient baseline sort implementation.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSetSort.h"

#include "CudaMgr/CudaMgr.h"
#include "Execute.h"
#include "InPlaceSort.h"
#include "ResultSetSortImpl.h"

#include <tbb/parallel_sort.h>
#include "ResultSet/CountDistinct.h"
#include "ResultSet/ResultSet.h"
#include "Shared/Intervals.h"
#include "Shared/likely.h"
#include "Shared/parallel_sort.h"  // TODO: is this used?
#include "Shared/thread_count.h"

#include <future>

#ifdef HAVE_CUDA
std::unique_ptr<CudaMgr_Namespace::CudaMgr> g_cuda_mgr;  // for unit tests only

namespace {

void set_cuda_context(Data_Namespace::DataMgr* data_mgr, const int device_id) {
  if (data_mgr) {
    data_mgr->getCudaMgr()->setContext(device_id);
    return;
  }
  // for unit tests only
  CHECK(g_cuda_mgr);
  g_cuda_mgr->setContext(device_id);
}

int getGpuCount(const Data_Namespace::DataMgr* data_mgr) {
  if (!data_mgr) {
    return g_cuda_mgr ? g_cuda_mgr->getDeviceCount() : 0;
  }
  return data_mgr->gpusPresent() ? data_mgr->getCudaMgr()->getDeviceCount() : 0;
}

void doBaselineSort(ResultSet* rs,
                    const ExecutorDeviceType device_type,
                    const std::list<hdk::ir::OrderEntry>& order_entries,
                    const size_t top_n,
                    const Executor* executor) {
  CHECK_EQ(size_t(1), order_entries.size());
  auto& query_mem_desc = rs->getQueryMemDesc();
  CHECK(!query_mem_desc.didOutputColumnar());
  const auto& oe = order_entries.front();
  CHECK_GT(oe.tle_no, 0);
  auto& targets = rs->getTargetInfos();
  CHECK_LE(static_cast<size_t>(oe.tle_no), targets.size());
  size_t logical_slot_idx = 0;
  size_t physical_slot_off = 0;
  for (size_t i = 0; i < static_cast<size_t>(oe.tle_no - 1); ++i) {
    physical_slot_off += query_mem_desc.getPaddedSlotWidthBytes(logical_slot_idx);
    logical_slot_idx =
        advance_slot(logical_slot_idx, targets[i], rs->getSeparateVarlenStorageValid());
  }
  const auto col_off =
      get_slot_off_quad(query_mem_desc) * sizeof(int64_t) + physical_slot_off;
  const size_t col_bytes = query_mem_desc.getPaddedSlotWidthBytes(logical_slot_idx);
  const auto row_bytes = get_row_bytes(query_mem_desc);
  const auto target_groupby_indices_sz = query_mem_desc.targetGroupbyIndicesSize();
  CHECK(target_groupby_indices_sz == 0 ||
        static_cast<size_t>(oe.tle_no) <= target_groupby_indices_sz);
  const int64_t target_groupby_index{
      target_groupby_indices_sz == 0
          ? -1
          : query_mem_desc.getTargetGroupbyIndex(oe.tle_no - 1)};
  GroupByBufferLayoutInfo layout{query_mem_desc.getEntryCount(),
                                 col_off,
                                 col_bytes,
                                 row_bytes,
                                 targets[oe.tle_no - 1],
                                 target_groupby_index};
  PodOrderEntry pod_oe{oe.tle_no, oe.is_desc, oe.nulls_first};
  auto groupby_buffer = rs->getStorage()->getUnderlyingBuffer();
  auto data_mgr = rs->getDataManager();
  const auto step = static_cast<size_t>(
      device_type == ExecutorDeviceType::GPU ? getGpuCount(data_mgr) : cpu_threads());
  CHECK_GE(step, size_t(1));
  const auto key_bytewidth = query_mem_desc.getEffectiveKeyWidth();
  Permutation permutation;
  if (step > 1) {
    std::vector<std::future<void>> top_futures;
    std::vector<Permutation> strided_permutations(step);
    for (size_t start = 0; start < step; ++start) {
      top_futures.emplace_back(std::async(
          std::launch::async,
          [&strided_permutations,
           data_mgr,
           device_type,
           groupby_buffer,
           pod_oe,
           key_bytewidth,
           layout,
           top_n,
           start,
           step,
           rs] {
            if (device_type == ExecutorDeviceType::GPU) {
              set_cuda_context(data_mgr, start);
            }
            strided_permutations[start] =
                (key_bytewidth == 4) ? baseline_sort<int32_t>(device_type,
                                                              start,
                                                              rs->getBufferProvider(),
                                                              groupby_buffer,
                                                              pod_oe,
                                                              layout,
                                                              top_n,
                                                              start,
                                                              step)
                                     : baseline_sort<int64_t>(device_type,
                                                              start,
                                                              rs->getBufferProvider(),
                                                              groupby_buffer,
                                                              pod_oe,
                                                              layout,
                                                              top_n,
                                                              start,
                                                              step);
          }));
    }
    for (auto& top_future : top_futures) {
      top_future.wait();
    }
    for (auto& top_future : top_futures) {
      top_future.get();
    }
    permutation.reserve(strided_permutations.size() * top_n);
    for (const auto& strided_permutation : strided_permutations) {
      permutation.insert(
          permutation.end(), strided_permutation.begin(), strided_permutation.end());
    }
    auto pv = PermutationView(permutation.data(), permutation.size());
    topPermutation(
        pv, top_n, createComparator(rs, order_entries, pv, executor, false), false);
    if (top_n < permutation.size()) {
      permutation.resize(top_n);
      permutation.shrink_to_fit();
    }
  } else {
    permutation = (key_bytewidth == 4) ? baseline_sort<int32_t>(device_type,
                                                                0,
                                                                rs->getBufferProvider(),
                                                                groupby_buffer,
                                                                pod_oe,
                                                                layout,
                                                                top_n,
                                                                0,
                                                                1)
                                       : baseline_sort<int64_t>(device_type,
                                                                0,
                                                                rs->getBufferProvider(),
                                                                groupby_buffer,
                                                                pod_oe,
                                                                layout,
                                                                top_n,
                                                                0,
                                                                1);
  }
  rs->setPermutationBuffer(std::move(permutation));
}

bool canUseFastBaselineSort(ResultSet* rs,
                            const std::list<hdk::ir::OrderEntry>& order_entries,
                            const size_t top_n) {
  auto& query_mem_desc = rs->getQueryMemDesc();
  if (order_entries.size() != 1 || query_mem_desc.hasKeylessHash() ||
      query_mem_desc.sortOnGpu() || query_mem_desc.didOutputColumnar()) {
    return false;
  }
  const auto& order_entry = order_entries.front();
  CHECK_GE(order_entry.tle_no, 1);
  CHECK_LE(static_cast<size_t>(order_entry.tle_no), rs->getTargetInfos().size());
  const auto& target_info = rs->getTargetInfos()[order_entry.tle_no - 1];
  if (!target_info.type->isNumber() || is_distinct_target(target_info)) {
    return false;
  }
  return (query_mem_desc.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash ||
          query_mem_desc.isSingleColumnGroupByWithPerfectHash()) &&
         top_n;
}

void baselineSort(ResultSet* rs,
                  const std::list<hdk::ir::OrderEntry>& order_entries,
                  const size_t top_n,
                  const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  // If we only have on GPU, it's usually faster to do multi-threaded radix sort on CPU
  if (getGpuCount(rs->getDataManager()) > 1) {
    try {
      doBaselineSort(rs, ExecutorDeviceType::GPU, order_entries, top_n, executor);
    } catch (...) {
      doBaselineSort(rs, ExecutorDeviceType::CPU, order_entries, top_n, executor);
    }
  } else {
    doBaselineSort(rs, ExecutorDeviceType::CPU, order_entries, top_n, executor);
  }
}

}  // namespace
#endif  // HAVE_CUDA

template <typename BUFFER_ITERATOR_TYPE>
void ResultSetComparator<BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumns() {
  for (const auto& order_entry : order_entries_) {
    if (is_distinct_target(result_set_->getTargetInfos()[order_entry.tle_no - 1])) {
      count_distinct_materialized_buffers_.emplace_back(
          materializeCountDistinctColumn(order_entry));
    }
  }
}

template <typename BUFFER_ITERATOR_TYPE>
ApproxQuantileBuffers
ResultSetComparator<BUFFER_ITERATOR_TYPE>::materializeApproxQuantileColumns() const {
  ApproxQuantileBuffers approx_quantile_materialized_buffers;
  for (const auto& order_entry : order_entries_) {
    if (result_set_->getTargetInfos()[order_entry.tle_no - 1].agg_kind ==
        hdk::ir::AggType::kApproxQuantile) {
      approx_quantile_materialized_buffers.emplace_back(
          materializeApproxQuantileColumn(order_entry));
    }
  }
  return approx_quantile_materialized_buffers;
}

template <typename BUFFER_ITERATOR_TYPE>
std::vector<int64_t>
ResultSetComparator<BUFFER_ITERATOR_TYPE>::materializeCountDistinctColumn(
    const hdk::ir::OrderEntry& order_entry) const {
  const size_t num_storage_entries = result_set_->getQueryMemDesc().getEntryCount();
  std::vector<int64_t> count_distinct_materialized_buffer(num_storage_entries);
  const CountDistinctDescriptor count_distinct_descriptor =
      result_set_->getQueryMemDesc().getCountDistinctDescriptor(order_entry.tle_no - 1);
  const size_t num_non_empty_entries = permutation_.size();

  const auto work = [&, query_id = logger::query_id()](const size_t start,
                                                       const size_t end) {
    auto qid_scope_guard = logger::set_thread_local_query_id(query_id);
    for (size_t i = start; i < end; ++i) {
      const PermutationIdx permuted_idx = permutation_[i];
      const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
      const auto storage = storage_lookup_result.storage_ptr;
      const auto off = storage_lookup_result.fixedup_entry_idx;
      const auto value = buffer_itr_.getColumnInternal(storage->getUnderlyingBuffer(),
                                                       off,
                                                       order_entry.tle_no - 1,
                                                       storage_lookup_result);
      count_distinct_materialized_buffer[permuted_idx] =
          count_distinct_set_size(value.i1, count_distinct_descriptor);
    }
  };
  // TODO(tlm): Allow use of tbb after we determine how to easily encapsulate the choice
  // between thread pool types
  if (single_threaded_) {
    work(0, num_non_empty_entries);
  } else {
    tbb::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(0, num_non_empty_entries, cpu_threads())) {
      thread_pool.run([=] { work(interval.begin, interval.end); });
    }
    thread_pool.wait();
  }
  return count_distinct_materialized_buffer;
}

template <typename BUFFER_ITERATOR_TYPE>
ApproxQuantileBuffers::value_type
ResultSetComparator<BUFFER_ITERATOR_TYPE>::materializeApproxQuantileColumn(
    const hdk::ir::OrderEntry& order_entry) const {
  ApproxQuantileBuffers::value_type materialized_buffer(
      result_set_->getQueryMemDesc().getEntryCount());
  const size_t size = permutation_.size();
  const auto work = [&, query_id = logger::query_id()](const size_t start,
                                                       const size_t end) {
    auto qid_scope_guard = logger::set_thread_local_query_id(query_id);
    for (size_t i = start; i < end; ++i) {
      const PermutationIdx permuted_idx = permutation_[i];
      const auto storage_lookup_result = result_set_->findStorage(permuted_idx);
      const auto storage = storage_lookup_result.storage_ptr;
      const auto off = storage_lookup_result.fixedup_entry_idx;
      const auto value = buffer_itr_.getColumnInternal(storage->getUnderlyingBuffer(),
                                                       off,
                                                       order_entry.tle_no - 1,
                                                       storage_lookup_result);
      materialized_buffer[permuted_idx] =
          value.i1 ? ResultSet::calculateQuantile(
                         reinterpret_cast<quantile::TDigest*>(value.i1))
                   : NULL_DOUBLE;
    }
  };
  if (single_threaded_) {
    work(0, size);
  } else {
    tbb::task_group thread_pool;
    for (auto interval : makeIntervals<size_t>(0, size, cpu_threads())) {
      thread_pool.run([=] { work(interval.begin, interval.end); });
    }
    thread_pool.wait();
  }
  return materialized_buffer;
}

template <typename BUFFER_ITERATOR_TYPE>
bool ResultSetComparator<BUFFER_ITERATOR_TYPE>::operator()(
    const PermutationIdx lhs,
    const PermutationIdx rhs) const {
  // NB: The compare function must define a strict weak ordering, otherwise
  // std::sort will trigger a segmentation fault (or corrupt memory).
  const auto lhs_storage_lookup_result = result_set_->findStorage(lhs);
  const auto rhs_storage_lookup_result = result_set_->findStorage(rhs);
  const auto lhs_storage = lhs_storage_lookup_result.storage_ptr;
  const auto rhs_storage = rhs_storage_lookup_result.storage_ptr;
  const auto fixedup_lhs = lhs_storage_lookup_result.fixedup_entry_idx;
  const auto fixedup_rhs = rhs_storage_lookup_result.fixedup_entry_idx;
  size_t materialized_count_distinct_buffer_idx{0};
  size_t materialized_approx_quantile_buffer_idx{0};

  for (const auto& order_entry : order_entries_) {
    CHECK_GE(order_entry.tle_no, 1);
    const auto& agg_info = result_set_->getTargetInfos()[order_entry.tle_no - 1];
    const auto entry_type = get_compact_type(agg_info);
    bool float_argument_input = takes_float_argument(agg_info);
    // Need to determine if the float value has been stored as float
    // or if it has been compacted to a different (often larger 8 bytes)
    // in distributed case the floats are actually 4 bytes
    // TODO the above takes_float_argument() is widely used wonder if this problem
    // exists elsewhere
    if (entry_type->isFp32()) {
      const auto is_col_lazy =
          !result_set_->getLazyFetchInfo().empty() &&
          result_set_->getLazyFetchInfo()[order_entry.tle_no - 1].is_lazily_fetched;
      if (result_set_->getQueryMemDesc().getPaddedSlotWidthBytes(order_entry.tle_no -
                                                                 1) == sizeof(float)) {
        float_argument_input =
            result_set_->getQueryMemDesc().didOutputColumnar() ? !is_col_lazy : true;
      }
    }

    if (UNLIKELY(is_distinct_target(agg_info))) {
      CHECK_LT(materialized_count_distinct_buffer_idx,
               count_distinct_materialized_buffers_.size());

      const auto& count_distinct_materialized_buffer =
          count_distinct_materialized_buffers_[materialized_count_distinct_buffer_idx];
      const auto lhs_sz = count_distinct_materialized_buffer[lhs];
      const auto rhs_sz = count_distinct_materialized_buffer[rhs];
      ++materialized_count_distinct_buffer_idx;
      if (lhs_sz == rhs_sz) {
        continue;
      }
      return (lhs_sz < rhs_sz) != order_entry.is_desc;
    } else if (UNLIKELY(agg_info.agg_kind == hdk::ir::AggType::kApproxQuantile)) {
      CHECK_LT(materialized_approx_quantile_buffer_idx,
               approx_quantile_materialized_buffers_.size());
      const auto& approx_quantile_materialized_buffer =
          approx_quantile_materialized_buffers_[materialized_approx_quantile_buffer_idx];
      const auto lhs_value = approx_quantile_materialized_buffer[lhs];
      const auto rhs_value = approx_quantile_materialized_buffer[rhs];
      ++materialized_approx_quantile_buffer_idx;
      if (lhs_value == rhs_value) {
        continue;
      } else if (entry_type->nullable()) {
        if (lhs_value == NULL_DOUBLE) {
          return order_entry.nulls_first;
        } else if (rhs_value == NULL_DOUBLE) {
          return !order_entry.nulls_first;
        }
      }
      return (lhs_value < rhs_value) != order_entry.is_desc;
    }

    const auto lhs_v = buffer_itr_.getColumnInternal(lhs_storage->getUnderlyingBuffer(),
                                                     fixedup_lhs,
                                                     order_entry.tle_no - 1,
                                                     lhs_storage_lookup_result);
    const auto rhs_v = buffer_itr_.getColumnInternal(rhs_storage->getUnderlyingBuffer(),
                                                     fixedup_rhs,
                                                     order_entry.tle_no - 1,
                                                     rhs_storage_lookup_result);

    if (UNLIKELY(ResultSet::isNull(entry_type, lhs_v, float_argument_input) &&
                 ResultSet::isNull(entry_type, rhs_v, float_argument_input))) {
      continue;
    }
    if (UNLIKELY(ResultSet::isNull(entry_type, lhs_v, float_argument_input) &&
                 !ResultSet::isNull(entry_type, rhs_v, float_argument_input))) {
      return order_entry.nulls_first;
    }
    if (UNLIKELY(ResultSet::isNull(entry_type, rhs_v, float_argument_input) &&
                 !ResultSet::isNull(entry_type, lhs_v, float_argument_input))) {
      return !order_entry.nulls_first;
    }

    if (LIKELY(lhs_v.isInt())) {
      CHECK(rhs_v.isInt());
      if (UNLIKELY(entry_type->isExtDictionary())) {
        CHECK_EQ(4, entry_type->canonicalSize());
        CHECK(executor_);
        const auto string_dict_proxy = executor_->getStringDictionaryProxy(
            entry_type->as<hdk::ir::ExtDictionaryType>()->dictId(),
            result_set_->getRowSetMemOwner(),
            false);
        auto lhs_str = string_dict_proxy->getString(lhs_v.i1);
        auto rhs_str = string_dict_proxy->getString(rhs_v.i1);
        if (lhs_str == rhs_str) {
          continue;
        }
        return (lhs_str < rhs_str) != order_entry.is_desc;
      }

      if (lhs_v.i1 == rhs_v.i1) {
        continue;
      }
      if (entry_type->isFloatingPoint()) {
        if (float_argument_input) {
          const auto lhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&rhs_v.i1));
          return (lhs_dval < rhs_dval) != order_entry.is_desc;
        } else {
          const auto lhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&rhs_v.i1));
          return (lhs_dval < rhs_dval) != order_entry.is_desc;
        }
      }
      return (lhs_v.i1 < rhs_v.i1) != order_entry.is_desc;
    } else {
      if (lhs_v.isPair()) {
        CHECK(rhs_v.isPair());
        const auto lhs =
            pair_to_double({lhs_v.i1, lhs_v.i2}, entry_type, float_argument_input);
        const auto rhs =
            pair_to_double({rhs_v.i1, rhs_v.i2}, entry_type, float_argument_input);
        if (lhs == rhs) {
          continue;
        }
        return (lhs < rhs) != order_entry.is_desc;
      } else {
        CHECK(lhs_v.isStr() && rhs_v.isStr());
        const auto lhs = lhs_v.strVal();
        const auto rhs = rhs_v.strVal();
        if (lhs == rhs) {
          continue;
        }
        return (lhs < rhs) != order_entry.is_desc;
      }
    }
  }
  return false;
}

Comparator createComparator(ResultSet* rs,
                            const std::list<hdk::ir::OrderEntry>& order_entries,
                            const PermutationView permutation,
                            const Executor* executor,
                            const bool single_threaded) {
  auto timer = DEBUG_TIMER(__func__);
  if (rs->getQueryMemDesc().didOutputColumnar()) {
    return
        [rsc = ResultSetComparator<ResultSet::ColumnWiseTargetAccessor>(
             order_entries, rs, permutation, executor, single_threaded)](
            const PermutationIdx lhs, const PermutationIdx rhs) { return rsc(lhs, rhs); };
  } else {
    return
        [rsc = ResultSetComparator<ResultSet::RowWiseTargetAccessor>(
             order_entries, rs, permutation, executor, single_threaded)](
            const PermutationIdx lhs, const PermutationIdx rhs) { return rsc(lhs, rhs); };
  }
}

// Partial sort permutation into top(least by compare) n elements.
// If permutation.size() <= n then sort entire permutation by compare.
// Return PermutationView with new size() = min(n, permutation.size()).
PermutationView topPermutation(PermutationView permutation,
                               const size_t n,
                               const Comparator& compare,
                               const bool single_threaded) {
  auto timer = DEBUG_TIMER(__func__);
  if (n < permutation.size()) {
    std::partial_sort(
        permutation.begin(), permutation.begin() + n, permutation.end(), compare);
    permutation.resize(n);
  } else if (!single_threaded) {
    tbb::parallel_sort(permutation.begin(), permutation.end(), compare);
  } else {
    std::sort(permutation.begin(), permutation.end(), compare);
  }
  return permutation;
}

namespace {

void radixSortOnGpu(ResultSet* rs,
                    const Config& config,
                    const std::list<hdk::ir::OrderEntry>& order_entries) {
  auto timer = DEBUG_TIMER(__func__);
  const int device_id{0};
  GpuAllocator cuda_allocator(rs->getBufferProvider(), device_id);
  CHECK_GT(rs->getBlockSize(), 0);
  CHECK_GT(rs->getGridSize(), 0);
  std::vector<int64_t*> group_by_buffers(rs->getBlockSize());
  group_by_buffers[0] =
      reinterpret_cast<int64_t*>(rs->getStorage()->getUnderlyingBuffer());
  auto dev_group_by_buffers =
      create_dev_group_by_buffers(&cuda_allocator,
                                  config,
                                  group_by_buffers,
                                  rs->getQueryMemDesc(),
                                  rs->getBlockSize(),
                                  rs->getGridSize(),
                                  device_id,
                                  ExecutorDispatchMode::KernelPerFragment,
                                  /*num_input_rows=*/-1,
                                  /*prepend_index_buffer=*/true,
                                  /*always_init_group_by_on_host=*/true,
                                  /*has_varlen_output=*/false,
                                  /*insitu_allocator*=*/nullptr);
  inplace_sort_gpu(order_entries,
                   rs->getQueryMemDesc(),
                   dev_group_by_buffers,
                   rs->getBufferProvider(),
                   device_id);
  copy_group_by_buffers_from_gpu(
      rs->getBufferProvider(),
      group_by_buffers,
      rs->getQueryMemDesc().getBufferSizeBytes(ExecutorDeviceType::GPU),
      dev_group_by_buffers.data,
      rs->getQueryMemDesc(),
      rs->getBlockSize(),
      rs->getGridSize(),
      device_id,
      /*prepend_index_buffer=*/false,
      /*has_varlen_output=*/false);
}

void radixSortOnCpu(ResultSet* rs, const std::list<hdk::ir::OrderEntry>& order_entries) {
  auto timer = DEBUG_TIMER(__func__);
  auto& query_mem_desc = rs->getQueryMemDesc();
  CHECK(!query_mem_desc.hasKeylessHash());
  std::vector<int64_t> tmp_buff(query_mem_desc.getEntryCount());
  std::vector<int32_t> idx_buff(query_mem_desc.getEntryCount());
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = rs->getStorage()->getUnderlyingBuffer();
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff = reinterpret_cast<int64_t*>(
        buffer_ptr + query_mem_desc.getColOffInBytes(target_idx));
    const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(target_idx);
    sort_groups_cpu(sortkey_val_buff,
                    &idx_buff[0],
                    query_mem_desc.getEntryCount(),
                    order_entry.is_desc,
                    slot_width);
    apply_permutation_cpu(reinterpret_cast<int64_t*>(buffer_ptr),
                          &idx_buff[0],
                          query_mem_desc.getEntryCount(),
                          &tmp_buff[0],
                          sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc.getSlotCount();
         ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(target_idx);
      const auto satellite_val_buff = reinterpret_cast<int64_t*>(
          buffer_ptr + query_mem_desc.getColOffInBytes(target_idx));
      apply_permutation_cpu(satellite_val_buff,
                            &idx_buff[0],
                            query_mem_desc.getEntryCount(),
                            &tmp_buff[0],
                            slot_width);
    }
  }
}

void parallelTop(ResultSet* rs,
                 const std::list<hdk::ir::OrderEntry>& order_entries,
                 const size_t top_n,
                 const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t nthreads = cpu_threads();

  CHECK(rs->isPermutationBufferEmpty());

  // Split permutation_ into nthreads subranges and top-sort in-place.
  Permutation permutation;
  permutation.resize(rs->entryCount());
  std::vector<PermutationView> permutation_views(nthreads);
  tbb::task_group top_sort_threads;
  for (auto interval : makeIntervals<PermutationIdx>(0, permutation.size(), nthreads)) {
    top_sort_threads.run([rs,
                          &permutation,
                          &order_entries,
                          &permutation_views,
                          top_n,
                          executor,
                          query_id = logger::query_id(),
                          interval] {
      auto qid_scope_guard = logger::set_thread_local_query_id(query_id);
      PermutationView pv(permutation.data() + interval.begin, 0, interval.size());
      pv = rs->initPermutationBuffer(pv, interval.begin, interval.end);
      const auto compare = createComparator(rs, order_entries, pv, executor, true);
      permutation_views[interval.index] = topPermutation(pv, top_n, compare, true);
    });
  }
  top_sort_threads.wait();

  // In case you are considering implementing a parallel reduction, note that the
  // ResultSetComparator constructor is O(N) in order to materialize some of the aggregate
  // columns as necessary to perform a comparison. This cost is why reduction is chosen to
  // be serial instead; only one more Comparator is needed below.

  // Left-copy disjoint top-sorted subranges into one contiguous range.
  // ++++....+++.....+++++...  ->  ++++++++++++............
  auto end = permutation.begin() + permutation_views.front().size();
  for (size_t i = 1; i < nthreads; ++i) {
    std::copy(permutation_views[i].begin(), permutation_views[i].end(), end);
    end += permutation_views[i].size();
  }

  // Top sort final range.
  PermutationView pv(permutation.data(), end - permutation.begin());
  const auto compare = createComparator(rs, order_entries, pv, executor, false);
  pv = topPermutation(pv, top_n, compare, false);
  permutation.resize(pv.size());
  permutation.shrink_to_fit();

  rs->setPermutationBuffer(std::move(permutation));
}

template <typename T>
void sort_on_cpu(T* val_buff,
                 PermutationView pv,
                 const hdk::ir::OrderEntry& order_entry) {
  int64_t begin = 0;
  int64_t end = pv.size() - 1;

  if (order_entry.nulls_first) {
    while (end >= begin) {
      auto val = val_buff[end];
      if (val == inline_null_value<T>()) {
        if (val_buff[begin] != inline_null_value<T>()) {
          std::swap(val_buff[begin], val_buff[end]);
          std::swap(pv[begin], pv[end]);
          --end;
        }
        ++begin;
      } else {
        --end;
      }
    }
    end = pv.size() - 1;
  } else {
    while (end >= begin) {
      auto val = val_buff[begin];
      if (val == inline_null_value<T>()) {
        if (val_buff[end] != inline_null_value<T>()) {
          std::swap(val_buff[end], val_buff[begin]);
          std::swap(pv[end], pv[begin]);
          ++begin;
        }
        --end;
      } else {
        ++begin;
      }
    }
    begin = 0;
  }

  if (order_entry.is_desc) {
    parallel_sort_by_key(val_buff + begin,
                         pv.begin() + begin,
                         (size_t)(end - begin + 1),
                         std::greater<T>());
  } else {
    parallel_sort_by_key(
        val_buff + begin, pv.begin() + begin, (size_t)(end - begin + 1), std::less<T>());
  }
}

void sort_onecol_cpu(int8_t* val_buff,
                     PermutationView pv,
                     const hdk::ir::Type* type,
                     const size_t slot_width,
                     const hdk::ir::OrderEntry& order_entry) {
  if (type->isInteger() || type->isDecimal()) {
    switch (slot_width) {
      case 1:
        sort_on_cpu(reinterpret_cast<int8_t*>(val_buff), pv, order_entry);
        break;
      case 2:
        sort_on_cpu(reinterpret_cast<int16_t*>(val_buff), pv, order_entry);
        break;
      case 4:
        sort_on_cpu(reinterpret_cast<int32_t*>(val_buff), pv, order_entry);
        break;
      case 8:
        sort_on_cpu(reinterpret_cast<int64_t*>(val_buff), pv, order_entry);
        break;
      default:
        CHECK(false);
    }
  } else if (type->isFloatingPoint()) {
    switch (slot_width) {
      case 4:
        sort_on_cpu(reinterpret_cast<float*>(val_buff), pv, order_entry);
        break;
      case 8:
        sort_on_cpu(reinterpret_cast<double*>(val_buff), pv, order_entry);
        break;
      default:
        CHECK(false);
    }
  } else {
    UNREACHABLE() << "Unsupported element type";
  }
}

}  // namespace

void sortResultSet(ResultSet* rs,
                   const std::list<hdk::ir::OrderEntry>& order_entries,
                   size_t top_n,
                   const Executor* executor) {
  auto timer = DEBUG_TIMER(__func__);

  if (!rs->getStorage()) {
    return;
  }

  rs->invalidateCachedRowCount();
  auto& query_mem_desc = rs->getQueryMemDesc();
  CHECK(!rs->getTargetInfos().empty());
#ifdef HAVE_CUDA
  if (canUseFastBaselineSort(rs, order_entries, top_n)) {
    baselineSort(rs, order_entries, top_n, executor);
    return;
  }
#endif  // HAVE_CUDA
  if (query_mem_desc.sortOnGpu()) {
    try {
      radixSortOnGpu(rs, executor ? executor->getConfig() : Config(), order_entries);
    } catch (const OutOfMemory&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(rs, order_entries);
    } catch (const std::bad_alloc&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(rs, order_entries);
    }
    return;
  }
  // This check isn't strictly required, but allows the index buffer to be 32-bit.
  if (query_mem_desc.getEntryCount() > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }

  CHECK(rs->isPermutationBufferEmpty());

  if (top_n && executor &&
      executor->getConfig().exec.parallel_top_min < rs->entryCount()) {
    if (executor->getConfig().exec.watchdog.enable &&
        executor->getConfig().exec.watchdog.parallel_top_max < rs->entryCount()) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    parallelTop(rs, order_entries, top_n, executor);
  } else {
    if (executor && executor->getConfig().exec.watchdog.enable &&
        executor->getConfig().exec.group_by.baseline_threshold < rs->entryCount()) {
      throw WatchdogException("Sorting the result would be too slow");
    }

    Permutation permutation;
    if (top_n == 0 && size_t(1) == order_entries.size() &&
        (!executor || executor->getConfig().rs.enable_direct_columnarization) &&
        rs->isDirectColumnarConversionPossible() && query_mem_desc.didOutputColumnar() &&
        query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      const auto& order_entry = order_entries.front();
      const auto target_idx = order_entry.tle_no - 1;
      const auto& lazy_fetch_info = rs->getLazyFetchInfo();
      bool is_not_lazy =
          lazy_fetch_info.empty() || !lazy_fetch_info[target_idx].is_lazily_fetched;
      const auto entry_type = get_compact_type(rs->getTargetInfos()[target_idx]);
      const auto slot_width = query_mem_desc.getPaddedSlotWidthBytes(target_idx);
      if (is_not_lazy && slot_width > 0 && entry_type->isNumber()) {
        const size_t buf_size = query_mem_desc.getEntryCount() * slot_width;
        // std::vector<int8_t> sortkey_val_buff(buf_size);
        std::unique_ptr<int8_t[]> sortkey_val_buff(new int8_t[buf_size]);
        rs->copyColumnIntoBuffer(
            target_idx, reinterpret_cast<int8_t*>(&sortkey_val_buff[0]), buf_size);
        permutation.resize(query_mem_desc.getEntryCount());
        PermutationView pv(permutation.data(), 0, permutation.size());
        pv = rs->initPermutationBuffer(pv, 0, permutation.size());
        sort_onecol_cpu(reinterpret_cast<int8_t*>(&sortkey_val_buff[0]),
                        pv,
                        entry_type,
                        slot_width,
                        order_entry);
        if (pv.size() < permutation.size()) {
          permutation.resize(pv.size());
          permutation.shrink_to_fit();
        }
        rs->setPermutationBuffer(std::move(permutation));
        return;
      }
    }
    permutation.resize(query_mem_desc.getEntryCount());
    // PermutationView is used to share common API with parallelTop().
    PermutationView pv(permutation.data(), 0, permutation.size());
    pv = rs->initPermutationBuffer(pv, 0, permutation.size());
    if (top_n == 0) {
      top_n = pv.size();  // top_n == 0 implies a full sort
    }
    pv = topPermutation(
        pv, top_n, createComparator(rs, order_entries, pv, executor, false), false);
    if (pv.size() < permutation.size()) {
      permutation.resize(pv.size());
      permutation.shrink_to_fit();
    }
    rs->setPermutationBuffer(std::move(permutation));
  }
}
