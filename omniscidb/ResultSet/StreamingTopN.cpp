/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StreamingTopN.h"

#include <cstring>

namespace streaming_top_n {

size_t get_heap_size(const size_t row_size, const size_t n, const size_t thread_count) {
  const auto row_size_quad = row_size / sizeof(int64_t);
  return (1 + n + row_size_quad * n) * thread_count * sizeof(int64_t);
}

size_t get_rows_offset_of_heaps(const size_t n, const size_t thread_count) {
  return (1 + n) * thread_count * sizeof(int64_t);
}

std::vector<int8_t> get_rows_copy_from_heaps(const int64_t* heaps,
                                             const size_t heaps_size,
                                             const size_t n,
                                             const size_t thread_count) {
  const auto rows_offset = streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
  const auto row_buff_size = heaps_size - rows_offset;
  std::vector<int8_t> rows_copy(row_buff_size);
  const auto rows_ptr = reinterpret_cast<const int8_t*>(heaps) + rows_offset;
  std::memcpy(&rows_copy[0], rows_ptr, row_buff_size);
  return rows_copy;
}

}  // namespace streaming_top_n
