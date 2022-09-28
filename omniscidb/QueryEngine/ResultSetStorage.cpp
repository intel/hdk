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

/**
 * @file    ResultSetStorage.cpp
 * @author
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2020 OmniSci, Inc.,  All rights reserved.
 */

#include "ResultSetStorage.h"

#include "DataMgr/Allocators/GpuAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "Shared/SqlTypesLayout.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"

#include <algorithm>
#include <bitset>
#include <future>
#include <numeric>

int8_t* VarlenOutputInfo::computeCpuOffset(const int64_t gpu_offset_address) const {
  const auto gpu_start_address_ptr = reinterpret_cast<int8_t*>(gpu_start_address);
  const auto gpu_offset_address_ptr = reinterpret_cast<int8_t*>(gpu_offset_address);
  if (gpu_offset_address_ptr == 0) {
    return 0;
  }
  const auto offset_bytes =
      static_cast<int64_t>(gpu_offset_address_ptr - gpu_start_address_ptr);
  CHECK_GE(offset_bytes, int64_t(0));
  return cpu_buffer_ptr + offset_bytes;
}

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff,
                                   const bool buff_is_provided)
    : targets_(targets)
    , query_mem_desc_(query_mem_desc)
    , buff_(buff)
    , buff_is_provided_(buff_is_provided)
    , target_init_vals_(result_set::initialize_target_values_for_storage(targets)) {}

int8_t* ResultSetStorage::getUnderlyingBuffer() const {
  return buff_;
}

void ResultSetStorage::addCountDistinctSetPointerMapping(const int64_t remote_ptr,
                                                         const int64_t ptr) {
  const auto it_ok = count_distinct_sets_mapping_.emplace(remote_ptr, ptr);
  CHECK(it_ok.second);
}

int64_t ResultSetStorage::mappedPtr(const int64_t remote_ptr) const {
  const auto it = count_distinct_sets_mapping_.find(remote_ptr);
  // Due to the removal of completely zero bitmaps in a distributed transfer there will be
  // remote ptr that do not not exists. Return 0 if no pointer found
  if (it == count_distinct_sets_mapping_.end()) {
    return int64_t(0);
  }
  return it->second;
}

std::vector<int64_t> result_set::initialize_target_values_for_storage(
    const std::vector<TargetInfo>& targets) {
  std::vector<int64_t> target_init_vals;
  for (const auto& target_info : targets) {
    if (target_info.agg_kind == kCOUNT ||
        target_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      target_init_vals.push_back(0);
      continue;
    }
    if (target_info.type->nullable()) {
      int64_t init_val =
          null_val_bit_pattern(target_info.type, takes_float_argument(target_info));
      target_init_vals.push_back(target_info.is_agg ? init_val : 0);
    } else {
      target_init_vals.push_back(target_info.is_agg ? 0xdeadbeef : 0);
    }
    if (target_info.agg_kind == kAVG) {
      target_init_vals.push_back(0);
    } else if (target_info.agg_kind == kSAMPLE &&
               (target_info.type->isString() || target_info.type->isArray())) {
      target_init_vals.push_back(0);
    }
  }
  return target_init_vals;
}

int64_t result_set::lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch,
                                const int8_t* byte_stream,
                                const int64_t pos) {
  CHECK(col_lazy_fetch.is_lazily_fetched);
  auto type = col_lazy_fetch.type;
  if (type->isFloatingPoint()) {
    if (type->isFp32()) {
      double fval = fixed_width_float_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    } else {
      double fval = fixed_width_double_decode_noinline(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    }
  }
  CHECK(type->isInteger() || type->isDecimal() || type->isDateTime() ||
        type->isInterval() || type->isBoolean() || type->isString() ||
        type->isExtDictionary() || type->isArray());
  size_t type_bitwidth = get_bit_width(type);
  if (type->isTime() || type->isInterval() || type->isExtDictionary()) {
    type_bitwidth = type->size() * 8;
    ;
  }
  CHECK_EQ(size_t(0), type_bitwidth % 8);
  int64_t val;
  bool date_in_days =
      type->isDate() && type->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay;
  if (date_in_days) {
    val = type->size() == 2 ? fixed_width_small_date_decode_noinline(
                                  byte_stream, 2, NULL_SMALLINT, NULL_BIGINT, pos)
                            : fixed_width_small_date_decode_noinline(
                                  byte_stream, 4, NULL_INT, NULL_BIGINT, pos);
  } else {
    val = (type->isExtDictionary() && type->size() < type->canonicalSize() &&
           type->as<hdk::ir::ExtDictionaryType>()->dictId())
              ? fixed_width_unsigned_decode_noinline(byte_stream, type_bitwidth / 8, pos)
              : fixed_width_int_decode_noinline(byte_stream, type_bitwidth / 8, pos);
  }
  if (!date_in_days &&
      ((type->size() < type->canonicalSize()) || type->isExtDictionary())) {
    auto col_logical_type = type->canonicalize();

    if (val == inline_fixed_encoding_null_value(type)) {
      return inline_int_null_value(col_logical_type);
    }
  }
  return val;
}

size_t ResultSetStorage::getColOffInBytes(size_t column_idx) const {
  return query_mem_desc_.getColOffInBytes(column_idx);
}
