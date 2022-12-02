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

/*
 * @file    JoinHashImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_GROUPBYFASTIMPL_H
#define QUERYENGINE_GROUPBYFASTIMPL_H

#include <cstdint>
#include <functional>
#include "../../../Shared/funcannotations.h"

#ifdef __CUDACC__
#define insert_key_cas(address, compare, val) atomicCAS(address, compare, val)
#elif defined(_WIN32)
#include "Shared/clean_windows.h"
#define insert_key_cas(address, compare, val)                           \
  InterlockedCompareExchange(reinterpret_cast<volatile long*>(address), \
                             static_cast<long>(val),                    \
                             static_cast<long>(compare))
#else
#define insert_key_cas(address, compare, val) \
  __sync_val_compare_and_swap(address, compare, val)
#endif

extern "C" ALWAYS_INLINE DEVICE int SUFFIX(fill_one_to_one_hashtable)(
    size_t idx,
    GENERIC_ADDR_SPACE int32_t* entry_ptr,
    const int32_t invalid_slot_val) {
  if (insert_key_cas(entry_ptr, invalid_slot_val, idx) != invalid_slot_val) {
    return -1;
  }
  return 0;
}

extern "C" ALWAYS_INLINE DEVICE int SUFFIX(fill_hashtable_for_semi_join)(
    size_t idx,
    GENERIC_ADDR_SPACE int32_t* entry_ptr,
    const int32_t invalid_slot_val) {
  // just mark the existence of value to the corresponding hash slot
  // regardless of hashtable collision
  insert_key_cas(entry_ptr, invalid_slot_val, idx);
  return 0;
}

#undef insert_key_cas

extern "C" ALWAYS_INLINE DEVICE GENERIC_ADDR_SPACE int32_t* SUFFIX(
    get_bucketized_hash_slot)(GENERIC_ADDR_SPACE int32_t* buff,
                              const int64_t key,
                              const int64_t min_key,
                              const int64_t bucket_normalization) {
  return buff + (key - min_key) / bucket_normalization;
}

extern "C" ALWAYS_INLINE DEVICE GENERIC_ADDR_SPACE int32_t* SUFFIX(get_hash_slot)(
    GENERIC_ADDR_SPACE int32_t* buff,
    const int64_t key,
    const int64_t min_key) {
  return buff + (key - min_key);
}

#endif  // QUERYENGINE_GROUPBYFASTIMPL_H