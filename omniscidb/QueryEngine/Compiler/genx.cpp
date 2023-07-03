/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>

#include "Shared/funcannotations.h"

extern "C" {
int64_t get_thread_index();

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

const GENERIC_ADDR_SPACE int64_t* init_shared_mem_nop(
    const GENERIC_ADDR_SPACE int64_t* groups_buffer,
    const int32_t groups_buffer_size) {
  return groups_buffer;
}

// TODO: these are almost the same in cuda, move to a single source
#define DEF_AGG_ID_INT_SHARED(n)                                             \
  extern "C" void agg_id_int##n##_shared(GENERIC_ADDR_SPACE int##n##_t* agg, \
                                         const int##n##_t val) {             \
    *agg = val;                                                              \
  }

DEF_AGG_ID_INT_SHARED(32)
DEF_AGG_ID_INT_SHARED(16)
DEF_AGG_ID_INT_SHARED(8)

#undef DEF_AGG_ID_INT_SHARED

void agg_id_shared(GENERIC_ADDR_SPACE int64_t* agg, const int64_t val) {
  *agg = val;
}

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
