/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace compiler {
enum class CpuAddrSpace {
  kGlobal = 0,
  kLocal = 0,
  kShared = 0,
};

enum class L0AddrSpace {
  kGlobal = 1,
  kLocal = 4,
  kShared = 3,
};

enum class CudaAddrSpace {
  kGlobal = 0,
  kLocal = 0,
  kShared = 3,
};
}  // namespace compiler
