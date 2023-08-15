/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef INLINENULLVALUES_H
#define INLINENULLVALUES_H

#include "funcannotations.h"

#ifndef _MSC_VER
#include <cassert>
#else
#include <assert.h>
#endif
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <limits>

#define NULL_BOOLEAN INT8_MIN
#define NULL_TINYINT INT8_MIN
#define NULL_SMALLINT INT16_MIN
#define NULL_INT INT32_MIN
#define NULL_BIGINT INT64_MIN
#define NULL_FLOAT FLT_MIN
#define NULL_DOUBLE DBL_MIN

#define NULL_ARRAY_BOOLEAN (INT8_MIN + 1)
#define NULL_ARRAY_TINYINT (INT8_MIN + 1)
#define NULL_ARRAY_SMALLINT (INT16_MIN + 1)
#define NULL_ARRAY_INT (INT32_MIN + 1)
#define NULL_ARRAY_BIGINT (INT64_MIN + 1)
#define NULL_ARRAY_FLOAT (FLT_MIN * 2.0)
#define NULL_ARRAY_DOUBLE (DBL_MIN * 2.0)

#define NULL_ARRAY_COMPRESSED_32 0x80000000U

#if !(defined(__CUDACC__) || defined(NO_BOOST))
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

template <class T>
constexpr inline int64_t inline_int_null_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min()
                                  : std::numeric_limits<T>::max();
}

template <class T>
constexpr inline int64_t inline_int_null_array_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::min() + 1
                                  : std::numeric_limits<T>::max() - 1;
  // TODO: null_array values in signed types would step on max valid value
  // in fixlen unsigned arrays, the max valid value may need to be lowered.
}

template <class T>
constexpr inline int64_t max_valid_int_value() {
  return std::is_signed<T>::value ? std::numeric_limits<T>::max()
                                  : std::numeric_limits<T>::max() - 1;
}

template <typename T>
constexpr inline T inline_fp_null_value() = delete;

template <>
constexpr inline float inline_fp_null_value<float>() {
  return NULL_FLOAT;
}

template <>
constexpr inline double inline_fp_null_value<double>() {
  return NULL_DOUBLE;
}

template <typename T>
DEVICE T inline_fp_null_array_value() = delete;

template <>
DEVICE inline float inline_fp_null_array_value<float>() {
  return NULL_ARRAY_FLOAT;
}

template <>
DEVICE inline double inline_fp_null_array_value<double>() {
  return NULL_ARRAY_DOUBLE;
}

#ifndef NO_BOOST
template <typename TYPE>
inline int64_t inline_int_null_value(const TYPE* type) {
  switch (type->id()) {
    case TYPE::kBoolean:
    case TYPE::kInteger:
    case TYPE::kDecimal:
      switch (type->size()) {
        case 1:
          return inline_int_null_value<int8_t>();
        case 2:
          return inline_int_null_value<int16_t>();
        case 4:
          return inline_int_null_value<int32_t>();
        case 8:
          return inline_int_null_value<int64_t>();
        default:
          abort();
      }
    case TYPE::kExtDictionary:
      return inline_int_null_value<int32_t>();
    case TYPE::kTimestamp:
    case TYPE::kTime:
    case TYPE::kDate:
    case TYPE::kInterval:
      return inline_int_null_value<int64_t>();
    default:
      abort();
  }
}

template <typename TYPE>
inline int64_t inline_fixed_encoding_null_value(const TYPE* type) {
  assert(type->isBoolean() || type->isInteger() || type->isDecimal() ||
         type->isDateTime() || type->isExtDictionary());

  if (type->isExtDictionary()) {
    switch (type->size()) {
      case 1:
        return inline_int_null_value<uint8_t>();
      case 2:
        return inline_int_null_value<uint16_t>();
      case 4:
        return inline_int_null_value<int32_t>();
      default:
        abort();
    }
  }

  switch (type->size()) {
    case 1:
      return inline_int_null_value<int8_t>();
    case 2:
      return inline_int_null_value<int16_t>();
    case 4:
      return inline_int_null_value<int32_t>();
    case 8:
      return inline_int_null_value<int64_t>();
    default:
      abort();
  }
  return 0;
}

template <typename TYPE>
inline double inline_fp_null_value(const TYPE* type) {
  if (type->isFp32()) {
    return inline_fp_null_value<float>();
  } else if (type->isFp64()) {
    return inline_fp_null_value<double>();
  }
  abort();
}

#endif  // NO_BOOST

template <typename V,
          std::enable_if_t<!std::is_same<V, bool>::value && std::is_integral<V>::value,
                           int> = 0>
CONSTEXPR DEVICE inline V inline_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_same<V, bool>::value, int> = 0>
CONSTEXPR DEVICE inline int8_t inline_null_value() {
  return inline_int_null_value<int8_t>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
CONSTEXPR DEVICE inline V inline_null_value() {
  return inline_fp_null_value<V>();
}

template <typename V,
          std::enable_if_t<!std::is_same<V, bool>::value && std::is_integral<V>::value,
                           int> = 0>
CONSTEXPR DEVICE inline V inline_null_array_value() {
  return inline_int_null_array_value<V>();
}

template <typename V, std::enable_if_t<std::is_same<V, bool>::value, int> = 0>
CONSTEXPR DEVICE inline int8_t inline_null_array_value() {
  return inline_int_null_array_value<int8_t>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
CONSTEXPR DEVICE inline V inline_null_array_value() {
  return inline_fp_null_array_value<V>();
}

#include <type_traits>

namespace serialize_detail {
template <int overload>
struct IntType;
template <>
struct IntType<1> {
  using type = uint8_t;
};
template <>
struct IntType<2> {
  using type = uint16_t;
};
template <>
struct IntType<4> {
  using type = uint32_t;
};
template <>
struct IntType<8> {
  using type = uint64_t;
};
}  // namespace serialize_detail

template <typename T, bool array = false>
CONSTEXPR DEVICE inline typename serialize_detail::IntType<sizeof(T)>::type
serialized_null_value() {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  T nv = 0;
  if CONSTEXPR (array) {
    nv = inline_null_array_value<T>();
  } else {
    nv = inline_null_value<T>();
  }
  return *(TT*)(&nv);
}

template <typename T, bool array = false>
CONSTEXPR DEVICE inline bool is_null(const T& value) {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  return serialized_null_value<T, array>() == *(TT*)(&value);
}

template <typename T, bool array = false>
CONSTEXPR DEVICE inline void set_null(T& value) {
  using TT = typename serialize_detail::IntType<sizeof(T)>::type;
  *(TT*)(&value) = serialized_null_value<T, array>();
}

#endif
