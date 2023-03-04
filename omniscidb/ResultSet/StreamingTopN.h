/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace streaming_top_n {

size_t get_heap_size(const size_t row_size, const size_t n, const size_t thread_count);

size_t get_rows_offset_of_heaps(const size_t n, const size_t thread_count);

std::vector<int8_t> get_rows_copy_from_heaps(const int64_t* heaps,
                                             const size_t heaps_size,
                                             const size_t n,
                                             const size_t thread_count);

}  // namespace streaming_top_n
