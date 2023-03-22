/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/GpuPlatform.h"
#include "GpuMgr.h"

struct GpuMgrContext {
  GpuMgrPlatform platform;
  GpuMgr* gpu_mgr;
};
