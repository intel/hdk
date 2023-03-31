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
#define hdk_cas(address, compare, val) atomicCAS(address, compare, val)
#elif defined(_MSC_VER)
#define hdk_cas(ptr, expected, desired) template <typename T>
template <typename T>
bool hdk_cas(T* ptr, T* expected, T desired) {
  if constexpr (sizeof(T) == 4) {
    return InterlockedCompareExchange(reinterpret_cast<volatile long*>(ptr),
                                      static_cast<long>(desired),
                                      static_cast<long>(*expected)) ==
           static_cast<long>(*expected);
  } else if constexpr (sizeof(T) == 8) {
    return InterlockedCompareExchange64(
               reinterpret_cast<volatile int64_t*>(ptr), desired, *expected) == *expected;
  } else {
    LOG(FATAL) << "Unsupported atomic operation";
  }
}
#else
#define hdk_cas(ptr, expected, desired) \
  __atomic_compare_exchange_n(          \
      ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#endif

extern "C" ALWAYS_INLINE DEVICE int SUFFIX(fill_one_to_one_hashtable)(
    size_t idx,
    GENERIC_ADDR_SPACE int32_t* entry_ptr,
    const int32_t invalid_slot_val) {
  // the atomic takes the address of invalid_slot_val to write the value of entry_ptr if
  // not equal to invalid_slot_val. make a copy to avoid dereferencing a const value.
  int32_t invalid_slot_val_copy = invalid_slot_val;
  if (!hdk_cas(entry_ptr, invalid_slot_val_copy, static_cast<int32_t>(idx))) {
    // slot is full
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
  auto invalid_slot_val_copy = invalid_slot_val;
  hdk_cas(entry_ptr, invalid_slot_val_copy, idx);
  return 0;
}

#ifndef _MSC_VER
#undef hdk_cas
#endif

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
