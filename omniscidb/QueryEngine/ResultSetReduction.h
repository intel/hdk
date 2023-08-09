/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2020 OmniSci, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSet/ResultSetStorage.h"

struct ReductionCode;

class ResultSetReduction {
 public:
  static void reduce(const ResultSetStorage& this_,
                     const ResultSetStorage& that,
                     const std::vector<std::string>& serialized_varlen_buffer,
                     const ReductionCode& reduction_code,
                     const Config& config,
                     const Executor* executor);

  // Reduces results for a single row when using interleaved bin layouts
  static bool reduceSingleRow(const int8_t* row_ptr,
                              const int8_t warp_count,
                              const bool is_columnar,
                              const bool replace_bitmap_ptr_with_bitmap_sz,
                              std::vector<int64_t>& agg_vals,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const std::vector<TargetInfo>& targets,
                              const std::vector<int64_t>& agg_init_vals);

  template <class KeyType>
  static void moveEntriesToBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                  const int8_t* old_buff,
                                  int8_t* new_buff,
                                  const size_t new_entry_count);

  template <class KeyType>
  static void moveOneEntryToBuffer(const QueryMemoryDescriptor& query_mem_desc,
                                   const size_t entry_index,
                                   int64_t* new_buff_i64,
                                   const size_t new_entry_count,
                                   const size_t key_count,
                                   const size_t row_qw_count,
                                   const int64_t* src_buff,
                                   const size_t key_byte_width);

 private:
  static void reduceOneEntryBaseline(const ResultSetStorage& this_,
                                     const ResultSetStorage& that,
                                     int8_t* this_buff,
                                     const int8_t* that_buff,
                                     const size_t that_entry_idx,
                                     bool enable_dynamic_watchdog);
  static void reduceOneEntrySlotsBaseline(const ResultSetStorage& this_,
                                          const ResultSetStorage& that,
                                          int64_t* this_entry_slots,
                                          const int64_t* that_buff,
                                          const size_t that_entry_idx);
  static void reduceOneSlotBaseline(const ResultSetStorage& this_,
                                    const ResultSetStorage& that,
                                    int64_t* this_buff,
                                    const size_t this_slot,
                                    const int64_t* that_buff,
                                    const size_t that_slot,
                                    const TargetInfo& target_info,
                                    const size_t target_logical_idx,
                                    const size_t target_slot_idx,
                                    const size_t init_agg_val_idx);
  static void reduceEntriesNoCollisionsColWise(
      const ResultSetStorage& this_,
      const ResultSetStorage& that,
      int8_t* this_buff,
      const int8_t* that_buff,
      const size_t start_index,
      const size_t end_index,
      const std::vector<std::string>& serialized_varlen_buffer,
      const Executor* executor);
  static void reduceOneSlot(const ResultSetStorage& this_,
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
                            const std::vector<std::string>& serialized_varlen_buffer);
  ALWAYS_INLINE
  static void reduceOneSlotSingleValue(const QueryMemoryDescriptor& query_mem_desc,
                                       int8_t* this_ptr1,
                                       const TargetInfo& target_info,
                                       const size_t target_slot_idx,
                                       const int64_t init_val,
                                       const int8_t* that_ptr1);
  static void reduceOneApproxQuantileSlot(const QueryMemoryDescriptor& query_mem_desc,
                                          int8_t* this_ptr1,
                                          const int8_t* that_ptr1,
                                          const size_t target_logical_idx);
  static void reduceOneQuantileSlot(const QueryMemoryDescriptor& query_mem_desc,
                                    int8_t* this_ptr1,
                                    const int8_t* that_ptr1,
                                    const size_t target_logical_idx);
  static void reduceOneCountDistinctSlot(const ResultSetStorage& this_,
                                         const ResultSetStorage& that,
                                         int8_t* this_ptr1,
                                         const int8_t* that_ptr1,
                                         const size_t target_logical_idx);
  static void reduceOneTopKSlot(int8_t* this_ptr,
                                const int8_t* that_ptr,
                                int elem_size,
                                bool is_fp,
                                int topk_param,
                                bool inline_buffer);
};

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&, const Config& config, Executor* executor);

  std::shared_ptr<ResultSet> getOwnResultSet();

  void rewriteVarlenAggregates(ResultSet*);

 private:
  std::shared_ptr<ResultSet> rs_;
};
