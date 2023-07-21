/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSet/TopKAggAccessors.h"

#include <stdint.h>

template <typename T, typename Comp>
void agg_topk_insert_heapify(T* vals, T value, int pos, int max_size) {
  Comp comp;
  if (pos == max_size) {
    // The heap is full, so maybe replace root and re-heapify top-down.
    if (comp(vals[0], value)) {
      int cur_pos = 0;
      int prev_pos;
      do {
        int l = cur_pos * 2 + 1;
        int r = cur_pos * 2 + 2;
        prev_pos = cur_pos;

        if (l < max_size) {
          T l_val = vals[l];
          if (r < max_size) {
            T r_val = vals[r];
            if (comp(l_val, r_val)) {
              if (comp(l_val, value)) {
                vals[cur_pos] = l_val;
                cur_pos = l;
              }
            } else {
              if (comp(r_val, value)) {
                vals[cur_pos] = r_val;
                cur_pos = r;
              }
            }
          } else {
            if (comp(l_val, value)) {
              vals[cur_pos] = l_val;
              cur_pos = l;
            }
          }
        }
      } while (cur_pos != prev_pos);
      vals[cur_pos] = value;
    }
  } else {
    // Bottom-up insert to the end of the heap.
    while (pos) {
      int parent_pos = (pos - 1) / 2;
      T parent_value = vals[parent_pos];
      if (comp(value, parent_value)) {
        vals[pos] = parent_value;
        pos = parent_pos;
      } else {
        break;
      }
    }
    vals[pos] = value;
  }
}

template <typename T>
void agg_topk_impl(int64_t* agg, T val, T empty_val, int k, bool inline_buffer) {
  T* agg_vals = inline_buffer ? reinterpret_cast<T*>(agg) : reinterpret_cast<T*>(*agg);
  auto pos = getTopKHeapSize<T>(agg_vals, empty_val, std::abs(k));
  if (k > 0) {
    agg_topk_insert_heapify<T, std::less<T>>(agg_vals, val, pos, k);
  } else {
    agg_topk_insert_heapify<T, std::greater<T>>(agg_vals, val, pos, -k);
  }
}
