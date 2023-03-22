/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GpuInitGroups.h"

// CUDA versions are implemented in GpuInitGroups.cu
void init_group_by_buffer_on_device_cuda(int64_t* groups_buffer,
                                         const int64_t* init_vals,
                                         const uint32_t groups_buffer_entry_count,
                                         const uint32_t key_count,
                                         const uint32_t key_width,
                                         const uint32_t agg_col_count,
                                         const bool keyless,
                                         const int8_t warp_size,
                                         const size_t block_size_x,
                                         const size_t grid_size_x);

void init_columnar_group_by_buffer_on_device_cuda(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_count,
    const uint32_t agg_col_count,
    const int8_t* col_sizes,
    const bool need_padding,
    const bool keyless,
    const int8_t key_size,
    const size_t block_size_x,
    const size_t grid_size_x);

void init_group_by_buffer_on_device_l0(int64_t* groups_buffer,
                                       const int64_t* init_vals,
                                       const uint32_t groups_buffer_entry_count,
                                       const uint32_t key_count,
                                       const uint32_t key_width,
                                       const uint32_t agg_col_count,
                                       const bool keyless,
                                       const int8_t warp_size,
                                       const size_t block_size_x,
                                       const size_t grid_size_x) {}
void init_columnar_group_by_buffer_on_device_l0(int64_t* groups_buffer,
                                                const int64_t* init_vals,
                                                const uint32_t groups_buffer_entry_count,
                                                const uint32_t key_count,
                                                const uint32_t agg_col_count,
                                                const int8_t* col_sizes,
                                                const bool need_padding,
                                                const bool keyless,
                                                const int8_t key_size,
                                                const size_t block_size_x,
                                                const size_t grid_size_x) {}

void init_group_by_buffer_on_device_impl(int64_t* groups_buffer,
                                         const int64_t* init_vals,
                                         const uint32_t groups_buffer_entry_count,
                                         const uint32_t key_count,
                                         const uint32_t key_width,
                                         const uint32_t agg_col_count,
                                         const bool keyless,
                                         const int8_t warp_size,
                                         const size_t block_size_x,
                                         const size_t grid_size_x,
                                         const GpuMgrPlatform gpu_mgr_platform) {
  if (gpu_mgr_platform == GpuMgrPlatform::CUDA) {
    init_group_by_buffer_on_device_cuda(groups_buffer,
                                        init_vals,
                                        groups_buffer_entry_count,
                                        key_count,
                                        key_width,
                                        agg_col_count,
                                        keyless,
                                        warp_size,
                                        block_size_x,
                                        grid_size_x);
  } else if (gpu_mgr_platform == GpuMgrPlatform::L0) {
    init_group_by_buffer_on_device_l0(groups_buffer,
                                      init_vals,
                                      groups_buffer_entry_count,
                                      key_count,
                                      key_width,
                                      agg_col_count,
                                      keyless,
                                      warp_size,
                                      block_size_x,
                                      grid_size_x);
  }
}

void init_columnar_group_by_buffer_on_device_impl(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_count,
    const uint32_t agg_col_count,
    const int8_t* col_sizes,
    const bool need_padding,
    const bool keyless,
    const int8_t key_size,
    const size_t block_size_x,
    const size_t grid_size_x,
    const GpuMgrPlatform gpu_mgr_platform) {
  if (gpu_mgr_platform == GpuMgrPlatform::CUDA) {
    init_columnar_group_by_buffer_on_device_cuda(groups_buffer,
                                                 init_vals,
                                                 groups_buffer_entry_count,
                                                 key_count,
                                                 agg_col_count,
                                                 col_sizes,
                                                 need_padding,
                                                 keyless,
                                                 key_size,
                                                 block_size_x,
                                                 grid_size_x);
  } else if (gpu_mgr_platform == GpuMgrPlatform::L0) {
    init_columnar_group_by_buffer_on_device_l0(groups_buffer,
                                               init_vals,
                                               groups_buffer_entry_count,
                                               key_count,
                                               agg_col_count,
                                               col_sizes,
                                               need_padding,
                                               keyless,
                                               key_size,
                                               block_size_x,
                                               grid_size_x);
  }
}

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const uint32_t groups_buffer_entry_count,
                                    const uint32_t key_count,
                                    const uint32_t key_width,
                                    const uint32_t agg_col_count,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x,
                                    const GpuMgrPlatform gpu_mgr_platform) {
  if (gpu_mgr_platform == GpuMgrPlatform::CUDA) {
    init_group_by_buffer_on_device_cuda(groups_buffer,
                                        init_vals,
                                        groups_buffer_entry_count,
                                        key_count,
                                        key_width,
                                        agg_col_count,
                                        keyless,
                                        warp_size,
                                        block_size_x,
                                        grid_size_x);
  } else if (gpu_mgr_platform == GpuMgrPlatform::L0) {
    init_group_by_buffer_on_device_l0(groups_buffer,
                                      init_vals,
                                      groups_buffer_entry_count,
                                      key_count,
                                      key_width,
                                      agg_col_count,
                                      keyless,
                                      warp_size,
                                      block_size_x,
                                      grid_size_x);
  }
}

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_count,
                                             const uint32_t agg_col_count,
                                             const int8_t* col_sizes,
                                             const bool need_padding,
                                             const bool keyless,
                                             const int8_t key_size,
                                             const size_t block_size_x,
                                             const size_t grid_size_x,
                                             const GpuMgrPlatform gpu_mgr_platform) {
  if (gpu_mgr_platform == GpuMgrPlatform::CUDA) {
    init_columnar_group_by_buffer_on_device_cuda(groups_buffer,
                                                 init_vals,
                                                 groups_buffer_entry_count,
                                                 key_count,
                                                 agg_col_count,
                                                 col_sizes,
                                                 need_padding,
                                                 keyless,
                                                 key_size,
                                                 block_size_x,
                                                 grid_size_x);
  } else if (gpu_mgr_platform == GpuMgrPlatform::L0) {
    init_columnar_group_by_buffer_on_device_l0(groups_buffer,
                                               init_vals,
                                               groups_buffer_entry_count,
                                               key_count,
                                               agg_col_count,
                                               col_sizes,
                                               need_padding,
                                               keyless,
                                               key_size,
                                               block_size_x,
                                               grid_size_x);
  }
}
