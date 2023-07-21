/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TopKAggAccessors.h"

#include "IR/Type.h"

int getTopKHeapSize(const int8_t* heap_handle,
                    const hdk::ir::Type* elem_type,
                    int max_heap_size) {
  switch (elem_type->canonicalSize()) {
    case 1:
      return getTopKHeapSize(reinterpret_cast<const int8_t*>(heap_handle),
                             inline_null_value<int8_t>(),
                             max_heap_size);
    case 2:
      return getTopKHeapSize(reinterpret_cast<const int16_t*>(heap_handle),
                             inline_null_value<int16_t>(),
                             max_heap_size);
    case 4:
      if (elem_type->isFloatingPoint()) {
        return getTopKHeapSize(reinterpret_cast<const float*>(heap_handle),
                               inline_null_value<float>(),
                               max_heap_size);
      } else {
        return getTopKHeapSize(reinterpret_cast<const int32_t*>(heap_handle),
                               inline_null_value<int32_t>(),
                               max_heap_size);
      }
    case 8:
      if (elem_type->isFloatingPoint()) {
        return getTopKHeapSize(reinterpret_cast<const double*>(heap_handle),
                               inline_null_value<double>(),
                               max_heap_size);
      } else {
        return getTopKHeapSize(reinterpret_cast<const int64_t*>(heap_handle),
                               inline_null_value<int64_t>(),
                               max_heap_size);
      }
    default:
      CHECK(false);
  }
  return 0;
}
