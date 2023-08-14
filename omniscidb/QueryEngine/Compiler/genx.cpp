/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "Shared/funcannotations.h"

namespace {
constexpr float HDK_FLT_MIN = std::numeric_limits<float>::min();
constexpr float HDK_FLT_MAX = std::numeric_limits<float>::max();
constexpr double HDK_DBL_MIN = std::numeric_limits<double>::min();
constexpr double HDK_DBL_MAX = std::numeric_limits<double>::max();
inline int32_t hdk_float_as_int32_t(const float x) {
  return *reinterpret_cast<const int32_t*>(&x);
}
inline float hdk_int32_t_as_float(const int32_t x) {
  return *reinterpret_cast<const float*>(&x);
}
inline int64_t hdk_double_as_int64_t(const double x) {
  return *reinterpret_cast<const int64_t*>(&x);
}
inline double hdk_int64_t_as_double(const int64_t x) {
  return *reinterpret_cast<const double*>(&x);
}
}  // namespace

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

int32_t agg_sum_int32_shared(GENERIC_ADDR_SPACE int32_t* agg, const int32_t val);
void agg_sum_float_shared(GENERIC_ADDR_SPACE int32_t* agg, const float val);
void agg_sum_double_shared(GENERIC_ADDR_SPACE int64_t* agg, const double val);
int64_t agg_sum_shared(GENERIC_ADDR_SPACE int64_t* agg, const int64_t val);
void agg_max_int32_shared(GENERIC_ADDR_SPACE int32_t* agg, const int32_t val);
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

double atomic_min_float(GENERIC_ADDR_SPACE float* addr, const float val) {
  GENERIC_ADDR_SPACE int32_t* address_as_ull =
      reinterpret_cast<GENERIC_ADDR_SPACE int32_t*>(addr);
  int32_t old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomic_cas_int_32(
        address_as_ull,
        assumed,
        hdk_float_as_int32_t(std::min(val, hdk_int32_t_as_float(assumed))));
  } while (assumed != old);

  return hdk_int32_t_as_float(old);
}

double atomic_min_double(GENERIC_ADDR_SPACE double* addr, const double val) {
  GENERIC_ADDR_SPACE int64_t* address_as_ull =
      reinterpret_cast<GENERIC_ADDR_SPACE int64_t*>(addr);
  int64_t old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomic_cas_int_64(
        address_as_ull,
        assumed,
        hdk_double_as_int64_t(std::min(val, hdk_int64_t_as_double(assumed))));
  } while (assumed != old);

  return hdk_int64_t_as_double(old);
}

void atomicMinFltSkipVal(GENERIC_ADDR_SPACE int32_t* addr,
                         const float val,
                         const float skip_val) {
  const int32_t flt_max = hdk_float_as_int32_t(HDK_FLT_MAX);
  int32_t old = atomic_xchg_int_32(addr, flt_max);
  agg_min_float_shared(addr,
                       old == hdk_float_as_int32_t(skip_val)
                           ? val
                           : std::min(hdk_int32_t_as_float(old), val));
}

void atomicMinDblSkipVal(GENERIC_ADDR_SPACE int64_t* addr,
                         const double val,
                         const double skip_val) {
  const int64_t dbl_max = hdk_double_as_int64_t(HDK_DBL_MAX);
  int64_t old = atomic_xchg_int_64(addr, dbl_max);
  agg_min_double_shared(addr,
                        old == hdk_double_as_int64_t(skip_val)
                            ? val
                            : std::min(hdk_int64_t_as_double(old), val));
}

void agg_min_float_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                   const float val,
                                   const float skip_val) {
  if (hdk_float_as_int32_t(val) != hdk_float_as_int32_t(skip_val)) {
    atomicMinFltSkipVal(agg, val, skip_val);
  }
}

void agg_min_double_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                    const double val,
                                    const double skip_val) {
  if (hdk_double_as_int64_t(val) != hdk_double_as_int64_t(skip_val)) {
    atomicMinDblSkipVal(agg, val, skip_val);
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
  if (hdk_float_as_int32_t(val) != hdk_float_as_int32_t(skip_val)) {
    const int32_t flt_max = hdk_float_as_int32_t(-HDK_FLT_MAX);
    int32_t old = atomic_xchg_int_32(agg, flt_max);
    agg_max_float_shared(agg,
                         old == hdk_float_as_int32_t(skip_val)
                             ? val
                             : std::max(hdk_int32_t_as_float(old), val));
  }
}

void agg_max_double_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                    const double val,
                                    const double skip_val) {
  if (hdk_double_as_int64_t(val) != hdk_double_as_int64_t(skip_val)) {
    const int64_t dbl_max = hdk_double_as_int64_t(-HDK_DBL_MAX);
    int64_t old = atomic_xchg_int_64(agg, dbl_max);
    agg_max_double_shared(agg,
                          old == hdk_double_as_int64_t(skip_val)
                              ? val
                              : std::max(hdk_int64_t_as_double(old), val));
  }
}

int32_t atomicSum32SkipVal(GENERIC_ADDR_SPACE int32_t* addr,
                           const int32_t val,
                           const int32_t skip_val) {
  int32_t old = atomic_xchg_int_32(addr, 0);
  int32_t old2 = agg_sum_int32_shared(addr, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

int64_t atomicSum64SkipVal(GENERIC_ADDR_SPACE int64_t* addr,
                           const int64_t val,
                           const int64_t skip_val) {
  int64_t old = atomic_xchg_int_64(addr, 0);
  int64_t old2 = agg_sum_shared(addr, old == skip_val ? val : (val + old));
  return old == skip_val ? old2 : (old2 + old);
}

int32_t agg_sum_int32_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                      const int32_t val,
                                      const int32_t skip_val) {
  if (val != skip_val) {
    const int32_t old = atomicSum32SkipVal(agg, val, skip_val);
    return old;
  }
  return 0;
}

void agg_sum_float_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                   const float val,
                                   const float skip_val) {
  if (hdk_float_as_int32_t(val) != hdk_float_as_int32_t(skip_val)) {
    int32_t old = atomic_xchg_int_32(agg, hdk_float_as_int32_t(0.f));
    agg_sum_float_shared(agg, old == hdk_float_as_int32_t(skip_val) ? val : (val + old));
  }
}

void agg_sum_double_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                    const double val,
                                    const double skip_val) {
  if (hdk_double_as_int64_t(val) != hdk_double_as_int64_t(skip_val)) {
    int64_t old = atomic_xchg_int_64(agg, hdk_double_as_int64_t(0.));
    agg_sum_double_shared(agg,
                          old == hdk_double_as_int64_t(skip_val) ? val : (val + old));
  }
}

int64_t agg_sum_int64_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                      const int64_t val,
                                      const int64_t skip_val) {
  if (val != skip_val) {
    const int64_t old = atomicSum64SkipVal(agg, val, skip_val);
    return old;
  }
  return 0;
}

int64_t agg_sum_skip_val_shared(GENERIC_ADDR_SPACE int64_t* agg,
                                const int64_t val,
                                const int64_t skip_val) {
  if (val != skip_val) {
    const int64_t old = atomicSum64SkipVal(agg, val, skip_val);
    return old;
  }
  return 0;
}

void agg_max_int32_skip_val_shared(GENERIC_ADDR_SPACE int32_t* agg,
                                   const int32_t val,
                                   const int32_t skip_val) {
  if (val != skip_val) {
    agg_max_int32_shared(agg, val);
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
