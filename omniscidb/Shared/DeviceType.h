/**
 * Copyright 2022 OmniSci, Inc.
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#ifndef __CUDACC__
#include <ostream>
#endif

enum class ExecutorDeviceType { CPU = 0, GPU };

#ifndef __CUDACC__
inline std::ostream& operator<<(std::ostream& os, ExecutorDeviceType dt) {
  constexpr char const* strings[]{"CPU", "GPU"};
  return os << strings[static_cast<int>(dt)];
}
#endif

inline std::string deviceToString(ExecutorDeviceType dt) {
  return (dt == ExecutorDeviceType::CPU ? "CPU" : "GPU");
}
