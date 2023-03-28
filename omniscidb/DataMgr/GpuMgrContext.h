/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GpuMgr.h"
#include "Shared/GpuPlatform.h"

struct GpuMgrContext {
  GpuMgr* gpu_mgr;
  size_t gpu_count;
  std::vector<Data_Namespace::AbstractBufferMgr*> buffer_mgrs;
};
