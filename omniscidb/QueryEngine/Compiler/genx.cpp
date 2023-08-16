/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>

#include "../GpuRtConstants.h"
#include "CommonRuntimeDefs.h"
#include "QueryEngine/MurmurHash1Inl.h"
#include "Shared/funcannotations.h"

template <typename T = int64_t>
inline DEVICE T SUFFIX(get_invalid_key)() {
  return EMPTY_KEY_64;
}

template <>
inline DEVICE int32_t SUFFIX(get_invalid_key)() {
  return EMPTY_KEY_32;
}

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

template <typename T>
inline bool keys_are_equal(GENERIC_ADDR_SPACE const T* key1,
                           GENERIC_ADDR_SPACE const T* key2,
                           const size_t key_component_count) {
  for (size_t i = 0; i < key_component_count; ++i) {
    if (key1[i] != key2[i]) {
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
DEVICE int64_t baseline_hash_join_idx_impl(GENERIC_ADDR_SPACE const int8_t* hash_buff,
                                           GENERIC_ADDR_SPACE const int8_t* key,
                                           const size_t key_bytes,
                                           const size_t entry_count) {
  if (!entry_count) {
    return kNoMatch;
  }
  const uint32_t h = MurmurHash1Impl(key, key_bytes, 0) % entry_count;
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

template <typename T>
FORCE_INLINE DEVICE int64_t get_composite_key_index_impl(const T* key,
                                                         const size_t key_component_count,
                                                         const T* composite_key_dict,
                                                         const size_t entry_count) {
  const uint32_t h =
      MurmurHash1Impl(key, key_component_count * sizeof(T), 0) % entry_count;
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

extern "C" {
int64_t atomic_cas_int_64(GENERIC_ADDR_SPACE int64_t*, int64_t, int64_t);
int32_t atomic_cas_int_32(GENERIC_ADDR_SPACE int32_t*, int32_t, int32_t);
int64_t atomic_xchg_int_64(GENERIC_ADDR_SPACE int64_t*, int64_t);
int32_t atomic_xchg_int_32(GENERIC_ADDR_SPACE int32_t*, int32_t);
double atomic_min_double(GENERIC_ADDR_SPACE double* addr, const double val);
double atomic_min_float(GENERIC_ADDR_SPACE float* addr, const float val);
double atomic_max_double(GENERIC_ADDR_SPACE double* addr, const double val);
double atomic_max_float(GENERIC_ADDR_SPACE float* addr, const float val);

void agg_max_shared(GENERIC_ADDR_SPACE int64_t* agg, const int64_t val);
int64_t agg_count_shared(GENERIC_ADDR_SPACE int64_t* agg, const int64_t val);
uint32_t agg_count_int32_shared(GENERIC_ADDR_SPACE uint32_t* agg, const int32_t val);

#include "CommonGpuRuntime.cpp"

void agg_id_float_shared(GENERIC_ADDR_SPACE int32_t* agg, const float val) {
  *reinterpret_cast<GENERIC_ADDR_SPACE float*>(agg) = val;
}

void agg_id_double_shared(GENERIC_ADDR_SPACE int64_t* agg, const double val) {
  *reinterpret_cast<GENERIC_ADDR_SPACE double*>(agg) = val;
}

uint32_t agg_count_float_shared(GENERIC_ADDR_SPACE uint32_t* agg, const float val) {
  return agg_count_int32_shared(agg, val);
}

int64_t agg_count_double_shared(GENERIC_ADDR_SPACE int64_t* agg, const double val) {
  return agg_count_shared(agg, static_cast<int64_t>(val));
}

void agg_min_float_shared(GENERIC_ADDR_SPACE int32_t* agg, const float val) {
  atomic_min_float(reinterpret_cast<GENERIC_ADDR_SPACE float*>(agg), val);
}

void agg_min_double_shared(GENERIC_ADDR_SPACE int64_t* agg, const double val) {
  atomic_min_double(reinterpret_cast<GENERIC_ADDR_SPACE double*>(agg), val);
}

void agg_min_float_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                   const float val,
                                   const float skip_val) {
  if (val != skip_val) {
    agg_min_float_shared(agg, val);
  }
}

void agg_min_double_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                    const double val,
                                    const double skip_val) {
  if (val != skip_val) {
    agg_min_double_shared(agg, val);
  }
}

void agg_max_float_shared(GENERIC_ADDR_SPACE int32_t* agg, const float val) {
  atomic_max_float(reinterpret_cast<GENERIC_ADDR_SPACE float*>(agg), val);
}
void agg_max_double_shared(GENERIC_ADDR_SPACE int64_t* agg, const double val) {
  atomic_max_double(reinterpret_cast<GENERIC_ADDR_SPACE double*>(agg), val);
}

void agg_max_float_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                   const float val,
                                   const float skip_val) {
  if (val != skip_val) {
    agg_max_float_shared(agg, val);
  }
}

void agg_max_double_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                    const double val,
                                    const double skip_val) {
  if (val != skip_val) {
    agg_max_double_shared(agg, val);
  }
}
}
