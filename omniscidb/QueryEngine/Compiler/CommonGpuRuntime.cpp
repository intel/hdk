/**
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include "Shared/funcannotations.h"

extern "C" {
#define DEF_AGG_ID_INT_SHARED(n)                                         \
  DEVICE void agg_id_int##n##_shared(GENERIC_ADDR_SPACE int##n##_t* agg, \
                                     const int##n##_t val) {             \
    *agg = val;                                                          \
  }

DEF_AGG_ID_INT_SHARED(32)
DEF_AGG_ID_INT_SHARED(16)
DEF_AGG_ID_INT_SHARED(8)

#undef DEF_AGG_ID_INT_SHARED

DEVICE void agg_id_shared(GENERIC_ADDR_SPACE int64_t* agg, const int64_t val) {
  *agg = val;
}

DEVICE GENERIC_ADDR_SPACE int8_t* agg_id_varlen_shared(
    GENERIC_ADDR_SPACE int8_t* varlen_buffer,
    const int64_t offset,
    const GENERIC_ADDR_SPACE int8_t* value,
    const int64_t size_bytes) {
  for (auto i = 0; i < size_bytes; i++) {
    varlen_buffer[offset + i] = value[i];
  }
  return &varlen_buffer[offset];
}

DEVICE void agg_count_distinct_bitmap_gpu(GENERIC_ADDR_SPACE int64_t* agg,
                                          const int64_t val,
                                          const int64_t min_val,
                                          const int64_t base_dev_addr,
                                          const int64_t base_host_addr,
                                          const uint64_t sub_bitmap_count,
                                          const uint64_t bitmap_bytes);

DEVICE void agg_count_distinct_bitmap_skip_val_gpu(GENERIC_ADDR_SPACE int64_t* agg,
                                                   const int64_t val,
                                                   const int64_t min_val,
                                                   const int64_t skip_val,
                                                   const int64_t base_dev_addr,
                                                   const int64_t base_host_addr,
                                                   const uint64_t sub_bitmap_count,
                                                   const uint64_t bitmap_bytes) {
  if (val != skip_val) {
    agg_count_distinct_bitmap_gpu(
        agg, val, min_val, base_dev_addr, base_host_addr, sub_bitmap_count, bitmap_bytes);
  }
}

DEVICE const GENERIC_ADDR_SPACE int64_t* init_shared_mem_nop(
    const GENERIC_ADDR_SPACE int64_t* groups_buffer,
    const int32_t groups_buffer_size) {
  return groups_buffer;
}
}
