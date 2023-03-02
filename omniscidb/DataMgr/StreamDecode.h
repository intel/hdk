/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

template <typename T,
          std::enable_if_t<!std::is_same<T, bool>::value && std::is_integral<T>::value,
                           int> = 0>
int64_t decodeInt(const int8_t* byte_stream, const int64_t pos) {
  static_assert(sizeof(T) <= 8);
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
  return *(reinterpret_cast<const T*>(&byte_stream[pos * sizeof(T)]));
}

inline int64_t decodeInt(const int8_t* byte_stream, int32_t byte_width, int64_t pos) {
  switch (byte_width) {
    case 1:
      return decodeInt<int8_t>(byte_stream, pos);
    case 2:
      return decodeInt<int16_t>(byte_stream, pos);
    case 4:
      return decodeInt<int32_t>(byte_stream, pos);
    case 8:
      return decodeInt<int64_t>(byte_stream, pos);
    default:
      return std::numeric_limits<int64_t>::min() + 1;
  }
}

inline int64_t decodeUnsignedInt(const int8_t* byte_stream,
                                 int32_t byte_width,
                                 int64_t pos) {
  switch (byte_width) {
    case 1:
      return decodeInt<uint8_t>(byte_stream, pos);
    case 2:
      return decodeInt<uint16_t>(byte_stream, pos);
    case 4:
      return decodeInt<uint32_t>(byte_stream, pos);
    case 8:
      return decodeInt<uint64_t>(byte_stream, pos);
    default:
      return std::numeric_limits<int64_t>::min() + 1;
  }
}

int64_t decodeSmallDate(const int8_t* byte_stream,
                        int32_t byte_width,
                        int32_t null_val,
                        int64_t ret_null_val,
                        int64_t pos) {
  auto val = decodeInt(byte_stream, byte_width, pos);
  return val == null_val ? ret_null_val : val * 86400;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
T decodeFp(const int8_t* byte_stream, int64_t pos) {
#ifdef WITH_DECODERS_BOUNDS_CHECKING
  assert(pos >= 0);
#endif  // WITH_DECODERS_BOUNDS_CHECKING
  return *(reinterpret_cast<const T*>(&byte_stream[pos * sizeof(T)]));
}
