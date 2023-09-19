/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "HashJoinRuntimeCpu.h"

#include <tbb/parallel_for.h>

namespace {

template <ColumnType T>
inline int64_t getElem(const int8_t* chunk_mem_ptr,
                       size_t elem_size,
                       size_t elem_ind) = delete;

template <>
inline int64_t getElem<ColumnType::SmallDate>(const int8_t* chunk_mem_ptr,
                                              size_t elem_size,
                                              size_t elem_ind) {
  return fixed_width_small_date_decode_noinline(chunk_mem_ptr,
                                                elem_size,
                                                elem_size == 4 ? NULL_INT : NULL_SMALLINT,
                                                elem_size == 4 ? NULL_INT : NULL_SMALLINT,
                                                elem_ind);
}

template <>
inline int64_t getElem<ColumnType::Signed>(const int8_t* chunk_mem_ptr,
                                           size_t elem_size,
                                           size_t elem_ind) {
  return fixed_width_int_decode_noinline(chunk_mem_ptr, elem_size, elem_ind);
}

template <>
inline int64_t getElem<ColumnType::Unsigned>(const int8_t* chunk_mem_ptr,
                                             size_t elem_size,
                                             size_t elem_ind) {
  return fixed_width_unsigned_decode_noinline(chunk_mem_ptr, elem_size, elem_ind);
}

template <>
inline int64_t getElem<ColumnType::Double>(const int8_t* chunk_mem_ptr,
                                           size_t elem_size,
                                           size_t elem_ind) {
  return fixed_width_double_decode_noinline(chunk_mem_ptr, elem_ind);
}

template <ColumnType T, size_t Elem>
inline int64_t getElem(const int8_t* chunk_mem_ptr, size_t elem_ind) {
  return getElem<T>(chunk_mem_ptr, Elem, elem_ind);
}

template <typename HASHTABLE_FILLING_FUNC, ColumnType T, size_t Elem>
inline int apply_hash_table_elementwise_impl(
    const tbb::blocked_range<size_t>& elems_range,
    const int8_t* chunk_mem_ptr,
    size_t curr_chunk_row_offset,
    const JoinColumnTypeInfo& type_info,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    HASHTABLE_FILLING_FUNC hashtable_filling_func) {
  for (size_t elem_i = elems_range.begin(); elem_i != elems_range.end(); elem_i++) {
    int64_t elem = getElem<T, Elem>(chunk_mem_ptr, elem_i);

    if (elem == type_info.null_val) {
      if (!type_info.uses_bw_eq) {
        continue;
      }
      elem = type_info.translated_null_val;
    }

    if (sd_inner_to_outer_translation_map &&
        (!type_info.uses_bw_eq || elem != type_info.translated_null_val)) {
      const auto outer_id = map_str_id_to_outer_dict(elem,
                                                     min_inner_elem,
                                                     type_info.min_val,
                                                     type_info.max_val,
                                                     sd_inner_to_outer_translation_map);
      if (outer_id == StringDictionary::INVALID_STR_ID) {
        continue;
      }
      elem = outer_id;
    }

    if (hashtable_filling_func(elem, curr_chunk_row_offset + elem_i)) {
      return -1;
    }
  }
  return 0;
}

template <typename HASHTABLE_FILLING_FUNC, ColumnType T>
inline int apply_hash_table_elementwise(const tbb::blocked_range<size_t>& elems_range,
                                        const int8_t* chunk_mem_ptr,
                                        size_t curr_chunk_row_offset,
                                        const JoinColumnTypeInfo& type_info,
                                        const int32_t* sd_inner_to_outer_translation_map,
                                        const int32_t min_inner_elem,
                                        HASHTABLE_FILLING_FUNC hashtable_filling_func) {
  switch (type_info.elem_sz) {
    case 1:
      return apply_hash_table_elementwise_impl<HASHTABLE_FILLING_FUNC, T, 1>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case 2:
      return apply_hash_table_elementwise_impl<HASHTABLE_FILLING_FUNC, T, 2>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case 4:
      return apply_hash_table_elementwise_impl<HASHTABLE_FILLING_FUNC, T, 4>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case 8:
      return apply_hash_table_elementwise_impl<HASHTABLE_FILLING_FUNC, T, 8>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    default:
      break;
  }
  UNREACHABLE();
  return 0;
}

// This templated approach allows to move switch-case outside the loop.
template <typename HASHTABLE_FILLING_FUNC>
inline int apply_hash_table_elementwise(const tbb::blocked_range<size_t>& elems_range,
                                        const int8_t* chunk_mem_ptr,
                                        size_t curr_chunk_row_offset,
                                        const JoinColumnTypeInfo& type_info,
                                        const int32_t* sd_inner_to_outer_translation_map,
                                        const int32_t min_inner_elem,
                                        HASHTABLE_FILLING_FUNC hashtable_filling_func) {
  switch (type_info.column_type) {
    case SmallDate:
      return apply_hash_table_elementwise<HASHTABLE_FILLING_FUNC, SmallDate>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case Signed:
      return apply_hash_table_elementwise<HASHTABLE_FILLING_FUNC, Signed>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case Unsigned:
      return apply_hash_table_elementwise<HASHTABLE_FILLING_FUNC, Unsigned>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    case Double:
      return apply_hash_table_elementwise<HASHTABLE_FILLING_FUNC, Double>(
          elems_range,
          chunk_mem_ptr,
          curr_chunk_row_offset,
          type_info,
          sd_inner_to_outer_translation_map,
          min_inner_elem,
          hashtable_filling_func);
    default:
      break;
  }
  UNREACHABLE();
  return 0;
}

}  // namespace

DEVICE int SUFFIX(fill_hash_join_buff_bucketized_cpu)(
    int32_t* cpu_hash_table_buff,
    const int32_t hash_join_invalid_val,
    const bool for_semi_join,
    const JoinColumn& join_column,
    const JoinColumnTypeInfo& type_info,
    const int32_t* sd_inner_to_outer_translation_map,
    const int32_t min_inner_elem,
    const int64_t bucket_normalization) {
  auto filling_func = for_semi_join ? SUFFIX(fill_hashtable_for_semi_join)
                                    : SUFFIX(fill_one_to_one_hashtable);
  auto hashtable_filling_func = [&](int64_t elem, size_t index) {
    auto entry_ptr = SUFFIX(get_bucketized_hash_slot)(
        cpu_hash_table_buff, elem, type_info.min_val, bucket_normalization);
    return filling_func(index, entry_ptr, hash_join_invalid_val);
  };

  // for some reason int8* ptr is actually JoinChunk* Why?
  auto join_chunk_array =
      reinterpret_cast<const struct JoinChunk*>(join_column.col_chunks_buff);
  // BTW it's vector with sz:
  // join_column.num_chunks

  // It's possible that 1 chunk, but 0 elements.
  if (join_column.num_elems == 0) {
    return 0;
  }

  // This value is tuned to make range of elemnts
  // handled in each thread spend about 10ms according to timers.
  size_t data_to_handle_sz = 512 * 1024;
  size_t granularity = data_to_handle_sz / type_info.elem_sz;

  std::atomic<int> err{0};
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, join_column.num_chunks),
      [&](const tbb::blocked_range<size_t>& join_chunks_range) {
        DEBUG_TIMER("fill_hash_join_buff_bucketized_cpu chunk");
        for (size_t chunk_i = join_chunks_range.begin();
             chunk_i != join_chunks_range.end();
             chunk_i++) {
          auto curr_chunk = join_chunk_array[chunk_i];

          tbb::parallel_for(
              tbb::blocked_range<size_t>(0, curr_chunk.num_elems, granularity),
              [&](const tbb::blocked_range<size_t>& curr_chnunk_elems_range) {
                auto ret = apply_hash_table_elementwise(curr_chnunk_elems_range,
                                                        curr_chunk.col_buff,
                                                        curr_chunk.row_id,
                                                        type_info,
                                                        sd_inner_to_outer_translation_map,
                                                        min_inner_elem,
                                                        hashtable_filling_func);
                if (ret != 0) {
                  int zero{0};
                  err.compare_exchange_strong(zero, ret);
                }
              });
        }
      });
  if (err) {
    return -1;
  }
  return 0;
}
