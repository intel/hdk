/*
 * Copyright 2020 OmniSci, Inc.
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

/**
 * @file		sqltypes.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Constants for Builtin SQL Types supported by OmniSci
 **/

#pragma once

#include "../Logger/Logger.h"
#include "StringTransform.h"
#include "funcannotations.h"

#include <boost/functional/hash.hpp>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace hdk::ir {
class Type;
}

// must not change because these values persist in catalogs.

struct VarlenDatum {
  size_t length;
  int8_t* pointer;
  bool is_null;

  DEVICE VarlenDatum() : length(0), pointer(nullptr), is_null(true) {}
  DEVICE virtual ~VarlenDatum() {}

  VarlenDatum(const size_t l, int8_t* p, const bool n)
      : length(l), pointer(p), is_null(n) {}
};

struct DoNothingDeleter {
  void operator()(int8_t*) {}
};
struct FreeDeleter {
  void operator()(int8_t* p) { free(p); }
};

struct HostArrayDatum : public VarlenDatum {
  using ManagedPtr = std::shared_ptr<int8_t>;

  HostArrayDatum() = default;

  HostArrayDatum(size_t const l, ManagedPtr p, bool const n)
      : VarlenDatum(l, p.get(), n), data_ptr(p) {}

  HostArrayDatum(size_t const l, int8_t* p, bool const n)
      : VarlenDatum(l, p, n), data_ptr(p, FreeDeleter()){};

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::result_of_t<CUSTOM_DELETER(int8_t*)> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, 0 == l), data_ptr(p, custom_deleter) {}

  template <typename CUSTOM_DELETER,
            typename = std::enable_if_t<
                std::is_void<std::result_of_t<CUSTOM_DELETER(int8_t*)> >::value> >
  HostArrayDatum(size_t const l, int8_t* p, bool const n, CUSTOM_DELETER custom_deleter)
      : VarlenDatum(l, p, n), data_ptr(p, custom_deleter) {}

  ManagedPtr data_ptr;
};

struct DeviceArrayDatum : public VarlenDatum {
  DEVICE DeviceArrayDatum() : VarlenDatum() {}
};

inline DEVICE constexpr bool is_cuda_compiler() {
#ifdef __CUDACC__
  return true;
#else
  return false;
#endif
}

using ArrayDatum =
    std::conditional_t<is_cuda_compiler(), DeviceArrayDatum, HostArrayDatum>;

union Datum {
  int8_t boolval;
  int8_t tinyintval;
  int16_t smallintval;
  int32_t intval;
  int64_t bigintval;
  float floatval;
  double doubleval;
  VarlenDatum* arrayval;
#ifndef __CUDACC__
  std::string* stringval;  // string value
#endif
};

#ifndef __CUDACC__
union DataBlockPtr {
  int8_t* numbersPtr;
  std::vector<std::string>* stringsPtr;
  std::vector<ArrayDatum>* arraysPtr;
};
#endif

#include "InlineNullValues.h"

#define INF_FLOAT HUGE_VALF
#define INF_DOUBLE HUGE_VAL
#define TRANSIENT_DICT_ID 0
#define TRANSIENT_DICT(ID) (-(ID))
#define REGULAR_DICT(TRANSIENTID) (-(TRANSIENTID))

#ifndef __CUDACC__

#include <string_view>

Datum StringToDatum(std::string_view s, const hdk::ir::Type* type);
std::string DatumToString(Datum d, const hdk::ir::Type* type);
#endif

int64_t extract_int_type_from_datum(const Datum datum, const hdk::ir::Type* type);
double extract_fp_type_from_datum(const Datum datum, const hdk::ir::Type* type);

bool DatumEqual(const Datum, const Datum, const hdk::ir::Type* type);
int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const hdk::ir::Type* type,
                                       const hdk::ir::Type* new_type);
size_t hash(Datum datum, const hdk::ir::Type* type);

#include "../QueryEngine/DateTruncate.h"
#include "../QueryEngine/ExtractFromTime.h"

using StringOffsetT = int32_t;
using ArrayOffsetT = int32_t;
