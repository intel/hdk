/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#define EMPTY_KEY_64 std::numeric_limits<int64_t>::max()
#define EMPTY_KEY_32 std::numeric_limits<int32_t>::max()
#define EMPTY_KEY_16 std::numeric_limits<int16_t>::max()
#define EMPTY_KEY_8 std::numeric_limits<int8_t>::max()

template <typename T = int64_t>
inline T get_empty_key() {
  static_assert(std::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  return EMPTY_KEY_64;
}

template <>
inline int32_t get_empty_key() {
  return EMPTY_KEY_32;
}
