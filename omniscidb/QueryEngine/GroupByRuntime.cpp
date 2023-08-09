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

#include "JoinHashTable/Runtime/JoinHashImpl.h"
#include "MurmurHash.h"

#ifndef __CUDACC__
#include "QueryEngine/TopKAggRuntime.h"
#include "Shared/quantile.h"
#endif

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE uint32_t
key_hash(GENERIC_ADDR_SPACE const int64_t* key,
         const uint32_t key_count,
         const uint32_t key_byte_width) {
  return MurmurHash3(key, key_byte_width * key_count, 0);
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE GENERIC_ADDR_SPACE int64_t* get_group_value(
    GENERIC_ADDR_SPACE int64_t* groups_buffer,
    const uint32_t groups_buffer_entry_count,
    GENERIC_ADDR_SPACE const int64_t* key,
    const uint32_t key_count,
    const uint32_t key_width,
    const uint32_t row_size_quad) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  GENERIC_ADDR_SPACE int64_t* matching_group = get_matching_group_value(
      groups_buffer, h, key, key_count, key_width, row_size_quad);
  if (matching_group) {
    return matching_group;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value(
        groups_buffer, h_probe, key, key_count, key_width, row_size_quad);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return NULL;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE bool dynamic_watchdog();

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_group_value_with_watchdog(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                              const uint32_t groups_buffer_entry_count,
                              GENERIC_ADDR_SPACE const int64_t* key,
                              const uint32_t key_count,
                              const uint32_t key_width,
                              const uint32_t row_size_quad) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  GENERIC_ADDR_SPACE int64_t* matching_group = get_matching_group_value(
      groups_buffer, h, key, key_count, key_width, row_size_quad);
  if (matching_group) {
    return matching_group;
  }
  uint32_t watchdog_countdown = 100;
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value(
        groups_buffer, h_probe, key, key_count, key_width, row_size_quad);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
    if (--watchdog_countdown == 0) {
      if (dynamic_watchdog()) {
        return NULL;
      }
      watchdog_countdown = 100;
    }
  }
  return NULL;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int32_t
get_group_value_columnar_slot(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                              const uint32_t groups_buffer_entry_count,
                              GENERIC_ADDR_SPACE const int64_t* key,
                              const uint32_t key_count,
                              const uint32_t key_width) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  int32_t matching_slot = get_matching_group_value_columnar_slot(
      groups_buffer, groups_buffer_entry_count, h, key, key_count, key_width);
  if (matching_slot != -1) {
    return h;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_slot = get_matching_group_value_columnar_slot(
        groups_buffer, groups_buffer_entry_count, h_probe, key, key_count, key_width);
    if (matching_slot != -1) {
      return h_probe;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return -1;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int32_t
get_group_value_columnar_slot_with_watchdog(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                                            const uint32_t groups_buffer_entry_count,
                                            GENERIC_ADDR_SPACE const int64_t* key,
                                            const uint32_t key_count,
                                            const uint32_t key_width) {
  uint32_t h = key_hash(key, key_count, key_width) % groups_buffer_entry_count;
  int32_t matching_slot = get_matching_group_value_columnar_slot(
      groups_buffer, groups_buffer_entry_count, h, key, key_count, key_width);
  if (matching_slot != -1) {
    return h;
  }
  uint32_t watchdog_countdown = 100;
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_slot = get_matching_group_value_columnar_slot(
        groups_buffer, groups_buffer_entry_count, h_probe, key, key_count, key_width);
    if (matching_slot != -1) {
      return h_probe;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
    if (--watchdog_countdown == 0) {
      if (dynamic_watchdog()) {
        return -1;
      }
      watchdog_countdown = 100;
    }
  }
  return -1;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_group_value_columnar(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                         const uint32_t groups_buffer_entry_count,
                         GENERIC_ADDR_SPACE const int64_t* key,
                         const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count, sizeof(int64_t)) % groups_buffer_entry_count;
  GENERIC_ADDR_SPACE int64_t* matching_group = get_matching_group_value_columnar(
      groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_group) {
    return matching_group;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value_columnar(
        groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return NULL;
}

extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_group_value_columnar_with_watchdog(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                                       const uint32_t groups_buffer_entry_count,
                                       GENERIC_ADDR_SPACE const int64_t* key,
                                       const uint32_t key_qw_count) {
  uint32_t h = key_hash(key, key_qw_count, sizeof(int64_t)) % groups_buffer_entry_count;
  GENERIC_ADDR_SPACE int64_t* matching_group = get_matching_group_value_columnar(
      groups_buffer, h, key, key_qw_count, groups_buffer_entry_count);
  if (matching_group) {
    return matching_group;
  }
  uint32_t watchdog_countdown = 100;
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group = get_matching_group_value_columnar(
        groups_buffer, h_probe, key, key_qw_count, groups_buffer_entry_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
    if (--watchdog_countdown == 0) {
      if (dynamic_watchdog()) {
        return NULL;
      }
      watchdog_countdown = 100;
    }
  }
  return NULL;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_group_value_fast(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                     const int64_t key,
                     const int64_t min_key,
                     const int64_t bucket,
                     const uint32_t row_size_quad) {
  int64_t key_diff = key - min_key;
  if (bucket) {
    key_diff /= bucket;
  }
  int64_t off = key_diff * row_size_quad;
  if (groups_buffer[off] == EMPTY_KEY_64) {
    groups_buffer[off] = key;
  }
  return groups_buffer + off + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_group_value_fast_with_original_key(GENERIC_ADDR_SPACE int64_t* groups_buffer,
                                       const int64_t key,
                                       const int64_t orig_key,
                                       const int64_t min_key,
                                       const int64_t bucket,
                                       const uint32_t row_size_quad) {
  int64_t key_diff = key - min_key;
  if (bucket) {
    key_diff /= bucket;
  }
  int64_t off = key_diff * row_size_quad;
  if (groups_buffer[off] == EMPTY_KEY_64) {
    groups_buffer[off] = orig_key;
  }
  return groups_buffer + off + 1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE uint32_t
get_columnar_group_bin_offset(GENERIC_ADDR_SPACE int64_t* key_base_ptr,
                              const int64_t key,
                              const int64_t min_key,
                              const int64_t bucket) {
  int64_t off = key - min_key;
  if (bucket) {
    off /= bucket;
  }
  if (key_base_ptr[off] == EMPTY_KEY_64) {
    key_base_ptr[off] = key;
  }
  return off;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE GENERIC_ADDR_SPACE int64_t*
get_scan_output_slot(GENERIC_ADDR_SPACE int64_t* output_buffer,
                     const uint32_t output_buffer_entry_count,
                     const uint32_t pos,
                     const int64_t offset_in_fragment,
                     const uint32_t row_size_quad) {
  uint64_t off = static_cast<uint64_t>(pos) * static_cast<uint64_t>(row_size_quad);
  if (pos < output_buffer_entry_count) {
    output_buffer[off] = offset_in_fragment;
    return output_buffer + off + 1;
  }
  return NULL;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int32_t
get_columnar_scan_output_offset(GENERIC_ADDR_SPACE int64_t* output_buffer,
                                const uint32_t output_buffer_entry_count,
                                const uint32_t pos,
                                const int64_t offset_in_fragment) {
  if (pos < output_buffer_entry_count) {
    output_buffer[pos] = offset_in_fragment;
    return pos;
  }
  return -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
bucketized_hash_join_idx(int64_t hash_buff,
                         int64_t const key,
                         int64_t const min_key,
                         int64_t const max_key,
                         int64_t bucket_normalization) {
  if (key >= min_key && key <= max_key) {
    return *SUFFIX(get_bucketized_hash_slot)(
        reinterpret_cast<GENERIC_ADDR_SPACE int32_t*>(hash_buff),
        key,
        min_key,
        bucket_normalization);
  }
  return -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
rowid_hash_join_idx(const int64_t key, const int64_t min_key, const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    return key;
  }
  return -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
hash_join_idx(int64_t hash_buff,
              const int64_t key,
              const int64_t min_key,
              const int64_t max_key) {
  if (key >= min_key && key <= max_key) {
    return *SUFFIX(get_hash_slot)(
        reinterpret_cast<GENERIC_ADDR_SPACE int32_t*>(hash_buff), key, min_key);
  }
  return -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
bucketized_hash_join_idx_nullable(int64_t hash_buff,
                                  const int64_t key,
                                  const int64_t min_key,
                                  const int64_t max_key,
                                  const int64_t null_val,
                                  const int64_t bucket_normalization) {
  return key != null_val ? bucketized_hash_join_idx(
                               hash_buff, key, min_key, max_key, bucket_normalization)
                         : -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
hash_join_idx_nullable(int64_t hash_buff,
                       const int64_t key,
                       const int64_t min_key,
                       const int64_t max_key,
                       const int64_t null_val) {
  return key != null_val ? hash_join_idx(hash_buff, key, min_key, max_key) : -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
bucketized_hash_join_idx_bitwise(int64_t hash_buff,
                                 const int64_t key,
                                 const int64_t min_key,
                                 const int64_t max_key,
                                 const int64_t null_val,
                                 const int64_t translated_val,
                                 const int64_t bucket_normalization) {
  return key != null_val ? bucketized_hash_join_idx(
                               hash_buff, key, min_key, max_key, bucket_normalization)
                         : bucketized_hash_join_idx(hash_buff,
                                                    translated_val,
                                                    min_key,
                                                    translated_val,
                                                    bucket_normalization);
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
rowid_hash_join_idx_nullable(const int64_t key,
                             const int64_t min_key,
                             const int64_t max_key,
                             const int64_t null_val) {
  return key != null_val ? rowid_hash_join_idx(key, min_key, max_key) : -1;
}

extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE int64_t
hash_join_idx_bitwise(int64_t hash_buff,
                      const int64_t key,
                      const int64_t min_key,
                      const int64_t max_key,
                      const int64_t null_val,
                      const int64_t translated_val) {
  return key != null_val
             ? hash_join_idx(hash_buff, key, min_key, max_key)
             : hash_join_idx(hash_buff, translated_val, min_key, translated_val);
}

#define DEF_TRANSLATE_NULL_KEY(key_type)                                               \
  extern "C" RUNTIME_EXPORT NEVER_INLINE DEVICE int64_t translate_null_key_##key_type( \
      const key_type key, const key_type null_val, const int64_t translated_val) {     \
    if (key == null_val) {                                                             \
      return translated_val;                                                           \
    }                                                                                  \
    return key;                                                                        \
  }

DEF_TRANSLATE_NULL_KEY(int8_t)
DEF_TRANSLATE_NULL_KEY(int16_t)
DEF_TRANSLATE_NULL_KEY(int32_t)
DEF_TRANSLATE_NULL_KEY(int64_t)

#undef DEF_TRANSLATE_NULL_KEY

#ifndef __CUDACC__

#define DEF_AGG_TOPK(val_type, suffix)                                             \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE void agg_topk_##suffix(           \
      int64_t* agg, val_type val, val_type empty_val, int k, bool inline_buffer) { \
    agg_topk_impl<val_type>(agg, val, empty_val, k, inline_buffer);                \
  }

#define DEF_AGG_TOPK_SKIP_VAL(val_type, suffix)                                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE void agg_topk_##suffix##_skip_val( \
      int64_t* agg, val_type val, val_type skip_val, int k, bool inline_buffer) {   \
    if (val != skip_val) {                                                          \
      agg_topk_##suffix(agg, val, skip_val, k, inline_buffer);                      \
    }                                                                               \
  }

#define DEF_AGG_TOPK_ALL(val_type, suffix) \
  DEF_AGG_TOPK(val_type, suffix)           \
  DEF_AGG_TOPK_SKIP_VAL(val_type, suffix)

DEF_AGG_TOPK_ALL(int8_t, int8)
DEF_AGG_TOPK_ALL(int16_t, int16)
DEF_AGG_TOPK_ALL(int32_t, int32)
DEF_AGG_TOPK_ALL(int64_t, int64)
DEF_AGG_TOPK_ALL(float, float)
DEF_AGG_TOPK_ALL(double, double)

#undef DEF_AGG_TOPK_ALL
#undef DEF_AGG_TOPK_SKIP_VAL
#undef DEF_AGG_TOPK

template <typename ValueType>
void agg_quantile_impl(int64_t* agg, ValueType val) {
  auto* quantile = reinterpret_cast<hdk::quantile::Quantile*>(*agg);
  quantile->add<ValueType>(val);
}

#define DEF_AGG_QUANTILE(val_type, suffix)                                   \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE void agg_quantile_##suffix( \
      int64_t* agg, val_type val) {                                          \
    agg_quantile_impl<val_type>(agg, val);                                   \
  }

#define DEF_AGG_QUANTILE_SKIP_VAL(val_type, suffix)                                     \
  extern "C" RUNTIME_EXPORT ALWAYS_INLINE DEVICE void agg_quantile_##suffix##_skip_val( \
      int64_t* agg, val_type val, val_type skip_val) {                                  \
    if (val != skip_val) {                                                              \
      agg_quantile_##suffix(agg, val);                                                  \
    }                                                                                   \
  }

#define DEF_AGG_QUANTILE_ALL(val_type, suffix) \
  DEF_AGG_QUANTILE(val_type, suffix)           \
  DEF_AGG_QUANTILE_SKIP_VAL(val_type, suffix)

DEF_AGG_QUANTILE_ALL(int8_t, int8)
DEF_AGG_QUANTILE_ALL(int16_t, int16)
DEF_AGG_QUANTILE_ALL(int32_t, int32)
DEF_AGG_QUANTILE_ALL(int64_t, int64)
DEF_AGG_QUANTILE_ALL(float, float)
DEF_AGG_QUANTILE_ALL(double, double)

#undef DEF_AGG_QUANTILE_ALL
#undef DEF_AGG_QUANTILE_SKIP_VAL
#undef DEF_AGG_QUANTILE

#endif
