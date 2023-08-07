/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>

#include "Shared/funcannotations.h"

extern "C" {
int64_t atomic_cas_int_64(GENERIC_ADDR_SPACE int64_t*, int64_t, int64_t);
int32_t atomic_cas_int_32(GENERIC_ADDR_SPACE int32_t*, int32_t, int32_t);
int64_t atomic_xchg_int_64(GENERIC_ADDR_SPACE int64_t*, int64_t);
int32_t atomic_xchg_int_32(GENERIC_ADDR_SPACE int32_t*, int32_t);
double atomic_min_double(GENERIC_ADDR_SPACE double* addr, const double val);
double atomic_min_float(GENERIC_ADDR_SPACE float* addr, const float val);
double atomic_max_double(GENERIC_ADDR_SPACE double* addr, const double val);
double atomic_max_float(GENERIC_ADDR_SPACE float* addr, const float val);
void atomic_or(GENERIC_ADDR_SPACE int32_t* addr, const int32_t val);
GENERIC_ADDR_SPACE int64_t* declare_dynamic_shared_memory();

void sync_threadblock();
int64_t get_thread_index();
int64_t get_block_dim();

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

const GENERIC_ADDR_SPACE int64_t* init_shared_mem(
    const GENERIC_ADDR_SPACE int64_t* global_groups_buffer,
    const int32_t groups_buffer_size) {
  auto shared_groups_buffer = declare_dynamic_shared_memory();
  const int32_t buffer_units = groups_buffer_size >> 3;

  for (int32_t pos = get_thread_index(); pos < buffer_units; pos += get_block_dim()) {
    shared_groups_buffer[pos] = global_groups_buffer[pos];
  }
  sync_threadblock();
  return shared_groups_buffer;
}

void agg_count_distinct_bitmap_gpu(GENERIC_ADDR_SPACE int64_t* agg,
                                   const int64_t val,
                                   const int64_t min_val,
                                   const int64_t base_dev_addr,
                                   const int64_t base_host_addr,
                                   const uint64_t sub_bitmap_count,
                                   const uint64_t bitmap_bytes) {
  const uint64_t bitmap_idx = val - min_val;
  const uint32_t byte_idx = bitmap_idx >> 3;
  const uint32_t word_idx = byte_idx >> 2;
  const uint32_t byte_word_idx = byte_idx & 3;
  const int64_t host_addr = *agg;
  GENERIC_ADDR_SPACE int32_t* bitmap =
      (GENERIC_ADDR_SPACE int32_t*)(base_dev_addr + host_addr - base_host_addr +
                                    (get_thread_index() & (sub_bitmap_count - 1)) *
                                        bitmap_bytes);
  switch (byte_word_idx) {
    case 0:
      atomic_or(&bitmap[word_idx], 1 << (bitmap_idx & 7));
      break;
    case 1:
      atomic_or(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 8));
      break;
    case 2:
      atomic_or(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 16));
      break;
    case 3:
      atomic_or(&bitmap[word_idx], 1 << ((bitmap_idx & 7) + 24));
      break;
    default:
      break;
  }
}
}
