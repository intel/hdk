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
#include "DataMgr/StreamDecode.h"
#include "Shared/EmptyKeyValues.h"
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

namespace {

ALWAYS_INLINE
void fill_empty_key_32(int32_t* key_ptr_i32, const size_t key_count) {
  for (size_t i = 0; i < key_count; ++i) {
    key_ptr_i32[i] = EMPTY_KEY_32;
  }
}

ALWAYS_INLINE
void fill_empty_key_64(int64_t* key_ptr_i64, const size_t key_count) {
  for (size_t i = 0; i < key_count; ++i) {
    key_ptr_i64[i] = EMPTY_KEY_64;
  }
}

template <typename T>
inline size_t make_bin_search(size_t l, size_t r, T&& is_empty_fn) {
  // Avoid search if there are no empty keys.
  if (!is_empty_fn(r - 1)) {
    return r;
  }

  --r;
  while (l != r) {
    size_t c = (l + r) / 2;
    if (is_empty_fn(c)) {
      r = c;
    } else {
      l = c + 1;
    }
  }

  return r;
}

// Given the entire buffer for the result set, buff, finds the beginning of the
// column for slot_idx. Only makes sense for column-wise representation.
const int8_t* advance_col_buff_to_slot(const int8_t* buff,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const std::vector<TargetInfo>& targets,
                                       const size_t slot_idx,
                                       const bool separate_varlen_storage) {
  auto crt_col_ptr = get_cols_ptr(buff, query_mem_desc);
  const auto buffer_col_count = query_mem_desc.getBufferColSlotCount();
  size_t agg_col_idx{0};
  for (size_t target_idx = 0; target_idx < targets.size(); ++target_idx) {
    if (agg_col_idx == slot_idx) {
      return crt_col_ptr;
    }
    CHECK_LT(agg_col_idx, buffer_col_count);
    const auto& agg_info = targets[target_idx];
    crt_col_ptr =
        advance_to_next_columnar_target_buff(crt_col_ptr, query_mem_desc, agg_col_idx);
    if (agg_info.is_agg && agg_info.agg_kind == hdk::ir::AggType::kAvg) {
      if (agg_col_idx + 1 == slot_idx) {
        return crt_col_ptr;
      }
      crt_col_ptr = advance_to_next_columnar_target_buff(
          crt_col_ptr, query_mem_desc, agg_col_idx + 1);
    }
    agg_col_idx = advance_slot(agg_col_idx, agg_info, separate_varlen_storage);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

void result_set::fill_empty_key(void* key_ptr,
                                const size_t key_count,
                                const size_t key_width) {
  switch (key_width) {
    case 4: {
      auto key_ptr_i32 = reinterpret_cast<int32_t*>(key_ptr);
      fill_empty_key_32(key_ptr_i32, key_count);
      break;
    }
    case 8: {
      auto key_ptr_i64 = reinterpret_cast<int64_t*>(key_ptr);
      fill_empty_key_64(key_ptr_i64, key_count);
      break;
    }
    default:
      CHECK(false);
  }
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
    if (target_info.agg_kind == hdk::ir::AggType::kCount ||
        target_info.agg_kind == hdk::ir::AggType::kApproxCountDistinct) {
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
    if (target_info.agg_kind == hdk::ir::AggType::kAvg) {
      target_init_vals.push_back(0);
    } else if (target_info.agg_kind == hdk::ir::AggType::kSample &&
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
      double fval = decodeFp<float>(byte_stream, pos);
      return *reinterpret_cast<const int64_t*>(may_alias_ptr(&fval));
    } else {
      double fval = decodeFp<double>(byte_stream, pos);
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
    val = type->size() == 2
              ? decodeSmallDate(byte_stream, 2, NULL_SMALLINT, NULL_BIGINT, pos)
              : decodeSmallDate(byte_stream, 4, NULL_INT, NULL_BIGINT, pos);
  } else {
    val = (type->isExtDictionary() && type->size() < type->canonicalSize() &&
           type->as<hdk::ir::ExtDictionaryType>()->dictId())
              ? decodeUnsignedInt(byte_stream, type_bitwidth / 8, pos)
              : decodeInt(byte_stream, type_bitwidth / 8, pos);
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

/*
 * copy all keys from the columnar prepended group buffer of "that_buff" into
 * "this_buff"
 */
void ResultSetStorage::copyKeyColWise(const size_t entry_idx,
                                      int8_t* this_buff,
                                      const int8_t* that_buff) const {
  CHECK(query_mem_desc_.didOutputColumnar());
  for (size_t group_idx = 0; group_idx < query_mem_desc_.getGroupbyColCount();
       group_idx++) {
    // if the column corresponds to a group key
    const auto column_offset_bytes =
        query_mem_desc_.getPrependedGroupColOffInBytes(group_idx);
    auto lhs_key_ptr = this_buff + column_offset_bytes;
    auto rhs_key_ptr = that_buff + column_offset_bytes;
    switch (query_mem_desc_.groupColWidth(group_idx)) {
      case 8:
        *(reinterpret_cast<int64_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int64_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 4:
        *(reinterpret_cast<int32_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int32_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 2:
        *(reinterpret_cast<int16_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int16_t*>(rhs_key_ptr) + entry_idx);
        break;
      case 1:
        *(reinterpret_cast<int8_t*>(lhs_key_ptr) + entry_idx) =
            *(reinterpret_cast<const int8_t*>(rhs_key_ptr) + entry_idx);
        break;
      default:
        CHECK(false);
        break;
    }
  }
}

// Rewrites the entries of this ResultSetStorage object to point directly into the
// serialized_varlen_buffer rather than using offsets.
void ResultSetStorage::rewriteAggregateBufferOffsets(
    const std::vector<std::string>& serialized_varlen_buffer) const {
  if (serialized_varlen_buffer.empty()) {
    return;
  }

  CHECK(!query_mem_desc_.didOutputColumnar());
  auto entry_count = query_mem_desc_.getEntryCount();
  CHECK_GT(entry_count, size_t(0));
  CHECK(buff_);

  // Row-wise iteration, consider moving to separate function
  for (size_t i = 0; i < entry_count; ++i) {
    if (isEmptyEntry(i, buff_)) {
      continue;
    }
    const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
    const auto key_bytes_with_padding = align_to_int64(key_bytes);
    auto rowwise_targets_ptr =
        row_ptr_rowwise(buff_, query_mem_desc_, i) + key_bytes_with_padding;
    size_t target_slot_idx = 0;
    for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
         ++target_logical_idx) {
      const auto& target_info = targets_[target_logical_idx];
      if ((target_info.type->isString() || target_info.type->isArray()) &&
          target_info.is_agg) {
        CHECK(target_info.agg_kind == hdk::ir::AggType::kSample);
        auto ptr1 = rowwise_targets_ptr;
        auto slot_idx = target_slot_idx;
        auto ptr2 = ptr1 + query_mem_desc_.getPaddedSlotWidthBytes(slot_idx);
        auto offset = *reinterpret_cast<const int64_t*>(ptr1);

        size_t length_to_elems =
            target_info.type->isString()
                ? 1
                : target_info.type->as<hdk::ir::ArrayBaseType>()->elemType()->size();
        CHECK_LT(static_cast<size_t>(offset), serialized_varlen_buffer.size());
        const auto& varlen_bytes_str = serialized_varlen_buffer[offset];
        const auto str_ptr = reinterpret_cast<const int8_t*>(varlen_bytes_str.c_str());
        CHECK(ptr1);
        *reinterpret_cast<int64_t*>(ptr1) = reinterpret_cast<const int64_t>(str_ptr);
        CHECK(ptr2);
        *reinterpret_cast<int64_t*>(ptr2) =
            static_cast<int64_t>(varlen_bytes_str.size() / length_to_elems);
      }

      rowwise_targets_ptr = advance_target_ptr_row_wise(
          rowwise_targets_ptr, target_info, target_slot_idx, query_mem_desc_, false);
      target_slot_idx = advance_slot(target_slot_idx, target_info, false);
    }
  }

  return;
}

void ResultSetStorage::fillOneEntryRowWise(const std::vector<int64_t>& entry) {
  const auto slot_count = query_mem_desc_.getBufferColSlotCount();
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK_EQ(slot_count + key_count, entry.size());
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.didOutputColumnar());
  CHECK_EQ(size_t(1), query_mem_desc_.getEntryCount());
  const auto key_off = key_offset_rowwise(0, key_count, slot_count);
  CHECK_EQ(query_mem_desc_.getEffectiveKeyWidth(), sizeof(int64_t));
  for (size_t i = 0; i < key_count; ++i) {
    this_buff[key_off + i] = entry[i];
  }
  const auto first_slot_off = slot_offset_rowwise(0, 0, key_count, slot_count);
  for (size_t i = 0; i < target_init_vals_.size(); ++i) {
    this_buff[first_slot_off + i] = entry[key_count + i];
  }
}

void ResultSetStorage::initializeRowWise() const {
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  const auto row_size = get_row_bytes(query_mem_desc_);
  CHECK_EQ(row_size % 8, 0u);
  const auto key_bytes_with_padding =
      align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
  CHECK(!query_mem_desc_.hasKeylessHash());
  switch (query_mem_desc_.getEffectiveKeyWidth()) {
    case 4: {
      for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
        auto row_ptr = buff_ + i * row_size;
        fill_empty_key_32(reinterpret_cast<int32_t*>(row_ptr), key_count);
        auto slot_ptr = reinterpret_cast<int64_t*>(row_ptr + key_bytes_with_padding);
        for (size_t j = 0; j < target_init_vals_.size(); ++j) {
          slot_ptr[j] = target_init_vals_[j];
        }
      }
      break;
    }
    case 8: {
      for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
        auto row_ptr = buff_ + i * row_size;
        fill_empty_key_64(reinterpret_cast<int64_t*>(row_ptr), key_count);
        auto slot_ptr = reinterpret_cast<int64_t*>(row_ptr + key_bytes_with_padding);
        for (size_t j = 0; j < target_init_vals_.size(); ++j) {
          slot_ptr[j] = target_init_vals_[j];
        }
      }
      break;
    }
    default:
      CHECK(false);
  }
}

void ResultSetStorage::fillOneEntryColWise(const std::vector<int64_t>& entry) {
  CHECK(query_mem_desc_.didOutputColumnar());
  CHECK_EQ(size_t(1), query_mem_desc_.getEntryCount());
  const auto slot_count = query_mem_desc_.getBufferColSlotCount();
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  CHECK_EQ(slot_count + key_count, entry.size());
  auto this_buff = reinterpret_cast<int64_t*>(buff_);

  for (size_t i = 0; i < key_count; i++) {
    const auto key_offset = key_offset_colwise(0, i, 1);
    this_buff[key_offset] = entry[i];
  }

  for (size_t i = 0; i < target_init_vals_.size(); i++) {
    const auto slot_offset = slot_offset_colwise(0, i, key_count, 1);
    this_buff[slot_offset] = entry[key_count + i];
  }
}

void ResultSetStorage::initializeColWise() const {
  const auto key_count = query_mem_desc_.getGroupbyColCount();
  auto this_buff = reinterpret_cast<int64_t*>(buff_);
  CHECK(!query_mem_desc_.hasKeylessHash());
  for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
    const auto first_key_off =
        key_offset_colwise(0, key_idx, query_mem_desc_.getEntryCount());
    for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
      this_buff[first_key_off + i] = EMPTY_KEY_64;
    }
  }
  for (size_t target_idx = 0; target_idx < target_init_vals_.size(); ++target_idx) {
    const auto first_val_off =
        slot_offset_colwise(0, target_idx, key_count, query_mem_desc_.getEntryCount());
    for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
      this_buff[first_val_off + i] = target_init_vals_[target_idx];
    }
  }
}

// Returns true iff the entry at position entry_idx in buff contains a valid row.
bool ResultSetStorage::isEmptyEntry(const size_t entry_idx, const int8_t* buff) const {
  if (QueryDescriptionType::NonGroupedAggregate ==
      query_mem_desc_.getQueryDescriptionType()) {
    return false;
  }
  if (query_mem_desc_.didOutputColumnar()) {
    return isEmptyEntryColumnar(entry_idx, buff);
  }
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const auto rowwise_target_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx);
    const auto target_slot_off = result_set::get_byteoff_of_slot(
        query_mem_desc_.getTargetIdxForKey(), query_mem_desc_);
    return read_int_from_buff(rowwise_target_ptr + target_slot_off,
                              query_mem_desc_.getPaddedSlotWidthBytes(
                                  query_mem_desc_.getTargetIdxForKey())) ==
           target_init_vals_[query_mem_desc_.getTargetIdxForKey()];
  } else {
    const auto keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, entry_idx);
    switch (query_mem_desc_.getEffectiveKeyWidth()) {
      case 4:
        CHECK(QueryDescriptionType::GroupByPerfectHash !=
              query_mem_desc_.getQueryDescriptionType());
        return *reinterpret_cast<const int32_t*>(keys_ptr) == EMPTY_KEY_32;
      case 8:
        return *reinterpret_cast<const int64_t*>(keys_ptr) == EMPTY_KEY_64;
      default:
        CHECK(false);
        return true;
    }
  }
}

/*
 * Returns true if the entry contain empty keys
 * This function should only be used with columanr format.
 */
bool ResultSetStorage::isEmptyEntryColumnar(const size_t entry_idx,
                                            const int8_t* buff) const {
  CHECK(query_mem_desc_.didOutputColumnar());
  if (query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::NonGroupedAggregate) {
    return false;
  }
  if (query_mem_desc_.hasKeylessHash()) {
    CHECK(query_mem_desc_.getQueryDescriptionType() ==
          QueryDescriptionType::GroupByPerfectHash);
    CHECK_GE(query_mem_desc_.getTargetIdxForKey(), 0);
    CHECK_LT(static_cast<size_t>(query_mem_desc_.getTargetIdxForKey()),
             target_init_vals_.size());
    const auto col_buff = advance_col_buff_to_slot(
        buff, query_mem_desc_, targets_, query_mem_desc_.getTargetIdxForKey(), false);
    const auto entry_buff =
        col_buff + entry_idx * query_mem_desc_.getPaddedSlotWidthBytes(
                                   query_mem_desc_.getTargetIdxForKey());
    return read_int_from_buff(entry_buff,
                              query_mem_desc_.getPaddedSlotWidthBytes(
                                  query_mem_desc_.getTargetIdxForKey())) ==
           target_init_vals_[query_mem_desc_.getTargetIdxForKey()];
  } else {
    // it's enough to find the first group key which is empty
    if (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection) {
      return reinterpret_cast<const int64_t*>(buff)[entry_idx] == EMPTY_KEY_64;
    } else {
      CHECK(query_mem_desc_.getGroupbyColCount() > 0);
      const auto target_buff = buff + query_mem_desc_.getPrependedGroupColOffInBytes(0);
      switch (query_mem_desc_.groupColWidth(0)) {
        case 8:
          return reinterpret_cast<const int64_t*>(target_buff)[entry_idx] == EMPTY_KEY_64;
        case 4:
          return reinterpret_cast<const int32_t*>(target_buff)[entry_idx] == EMPTY_KEY_32;
        case 2:
          return reinterpret_cast<const int16_t*>(target_buff)[entry_idx] == EMPTY_KEY_16;
        case 1:
          return reinterpret_cast<const int8_t*>(target_buff)[entry_idx] == EMPTY_KEY_8;
        default:
          CHECK(false);
      }
    }
    return false;
  }
  return false;
}

size_t ResultSetStorage::binSearchRowCount() const {
  CHECK(query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection);
  CHECK_EQ(query_mem_desc_.getEffectiveKeyWidth(), size_t(8));

  if (!query_mem_desc_.getEntryCount()) {
    return 0;
  }

  if (query_mem_desc_.didOutputColumnar()) {
    return make_bin_search(0, query_mem_desc_.getEntryCount(), [this](size_t idx) {
      return reinterpret_cast<const int64_t*>(buff_)[idx] == EMPTY_KEY_64;
    });
  } else {
    return make_bin_search(0, query_mem_desc_.getEntryCount(), [this](size_t idx) {
      const auto keys_ptr = row_ptr_rowwise(buff_, query_mem_desc_, idx);
      return *reinterpret_cast<const int64_t*>(keys_ptr) == EMPTY_KEY_64;
    });
  }
}

bool ResultSetStorage::isEmptyEntry(const size_t entry_idx) const {
  return isEmptyEntry(entry_idx, buff_);
}
