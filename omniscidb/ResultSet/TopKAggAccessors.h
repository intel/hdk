/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace hdk::ir {
class Type;
}

#include <stdint.h>

template <typename T>
int getTopKHeapSize(const T* vals, T empty_value, int max_size) {
  if (vals[max_size - 1] != empty_value) {
    return max_size;
  }
  int l = 0;
  int r = max_size - 1;
  while (l != r) {
    int cur = (l + r) / 2;
    if (vals[cur] == empty_value) {
      r = cur;
    } else {
      l = cur + 1;
    }
  }

  return l;
}

int getTopKHeapSize(const int8_t* heap_handle,
                    const hdk::ir::Type* elem_type,
                    int max_heap_size);
