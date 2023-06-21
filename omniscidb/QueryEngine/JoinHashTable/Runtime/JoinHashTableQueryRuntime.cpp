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

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "QueryEngine/CompareKeysInl.h"
#include "QueryEngine/MurmurHash.h"

DEVICE bool compare_to_key(GENERIC_ADDR_SPACE const int8_t* entry,
                           GENERIC_ADDR_SPACE const int8_t* key,
                           const size_t key_bytes) {
  for (size_t i = 0; i < key_bytes; ++i) {
    if (entry[i] != key[i]) {
      return false;
    }
  }
  return true;
}

namespace {

const int kNoMatch = -1;
const int kNotPresent = -2;

}  // namespace

template <class T>
DEVICE int64_t get_matching_slot(GENERIC_ADDR_SPACE const int8_t* hash_buff,
                                 const uint32_t h,
                                 GENERIC_ADDR_SPACE const int8_t* key,
                                 const size_t key_bytes) {
  const auto lookup_result_ptr = hash_buff + h * (key_bytes + sizeof(T));
  if (compare_to_key(lookup_result_ptr, key, key_bytes)) {
    return *reinterpret_cast<GENERIC_ADDR_SPACE const T*>(lookup_result_ptr + key_bytes);
  }
  if (*reinterpret_cast<GENERIC_ADDR_SPACE const T*>(lookup_result_ptr) ==
      SUFFIX(get_invalid_key) < typename remove_addr_space<T>::type > ()) {
    return kNotPresent;
  }
  return kNoMatch;
}

template <class T>
FORCE_INLINE DEVICE int64_t
baseline_hash_join_idx_impl(GENERIC_ADDR_SPACE const int8_t* hash_buff,
                            GENERIC_ADDR_SPACE const int8_t* key,
                            const size_t key_bytes,
                            const size_t entry_count) {
  if (!entry_count) {
    return kNoMatch;
  }
  const uint32_t h = MurmurHash1(key, key_bytes, 0) % entry_count;
  int64_t matching_slot = get_matching_slot<T>(hash_buff, h, key, key_bytes);
  if (matching_slot != kNoMatch) {
    return matching_slot;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    matching_slot = get_matching_slot<T>(hash_buff, h_probe, key, key_bytes);
    if (matching_slot != kNoMatch) {
      return matching_slot;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return kNoMatch;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
baseline_hash_join_idx_32(GENERIC_ADDR_SPACE const int8_t* hash_buff,
                          GENERIC_ADDR_SPACE const int8_t* key,
                          const size_t key_bytes,
                          const size_t entry_count) {
  return baseline_hash_join_idx_impl<int32_t>(hash_buff, key, key_bytes, entry_count);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
baseline_hash_join_idx_64(GENERIC_ADDR_SPACE const int8_t* hash_buff,
                          GENERIC_ADDR_SPACE const int8_t* key,
                          const size_t key_bytes,
                          const size_t entry_count) {
  return baseline_hash_join_idx_impl<int64_t>(hash_buff, key, key_bytes, entry_count);
}

template <typename T>
FORCE_INLINE DEVICE int64_t get_bucket_key_for_value_impl(const T value,
                                                          const double bucket_size) {
  return static_cast<int64_t>(floor(static_cast<double>(value) * bucket_size));
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
get_bucket_key_for_range_double(GENERIC_ADDR_SPACE const int8_t* range_bytes,
                                const size_t range_component_index,
                                const double bucket_size) {
  const auto range = reinterpret_cast<GENERIC_ADDR_SPACE const double*>(range_bytes);
  return get_bucket_key_for_value_impl(range[range_component_index], bucket_size);
}

FORCE_INLINE DEVICE int64_t
get_bucket_key_for_range_compressed_impl(GENERIC_ADDR_SPACE const int8_t* range,
                                         const size_t range_component_index,
                                         const double bucket_size) {
  assert(false);
  return -1;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
get_bucket_key_for_range_compressed(GENERIC_ADDR_SPACE const int8_t* range,
                                    const size_t range_component_index,
                                    const double bucket_size) {
  return get_bucket_key_for_range_compressed_impl(
      range, range_component_index, bucket_size);
}

template <typename T>
FORCE_INLINE DEVICE int64_t get_composite_key_index_impl(const T* key,
                                                         const size_t key_component_count,
                                                         const T* composite_key_dict,
                                                         const size_t entry_count) {
  const uint32_t h = MurmurHash1(key, key_component_count * sizeof(T), 0) % entry_count;
  uint32_t off = h * key_component_count;
  if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
    return h;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    off = h_probe * key_component_count;
    if (keys_are_equal(&composite_key_dict[off], key, key_component_count)) {
      return h_probe;
    }
    if (composite_key_dict[off] ==
        SUFFIX(get_invalid_key) < typename remove_addr_space<T>::type > ()) {
      return -1;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return -1;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
get_composite_key_index_32(GENERIC_ADDR_SPACE const int32_t* key,
                           const size_t key_component_count,
                           GENERIC_ADDR_SPACE const int32_t* composite_key_dict,
                           const size_t entry_count) {
  return get_composite_key_index_impl(
      key, key_component_count, composite_key_dict, entry_count);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t
get_composite_key_index_64(GENERIC_ADDR_SPACE const int64_t* key,
                           const size_t key_component_count,
                           GENERIC_ADDR_SPACE const int64_t* composite_key_dict,
                           const size_t entry_count) {
  return get_composite_key_index_impl(
      key, key_component_count, composite_key_dict, entry_count);
}
