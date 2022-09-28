/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file    SqlTypesLayout.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_SQLTYPESLAYOUT_H
#define QUERYENGINE_SQLTYPESLAYOUT_H

#include "Shared/TargetInfo.h"

#include "Logger/Logger.h"

#include <limits>

class OverflowOrUnderflow : public std::runtime_error {
 public:
  OverflowOrUnderflow() : std::runtime_error("Overflow or underflow") {}
};

inline const hdk::ir::Type* get_compact_type(const TargetInfo& target) {
  if (!target.is_agg) {
    return target.type;
  }
  const auto agg_type = target.agg_kind;
  auto agg_arg = target.agg_arg_type;
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!target.is_distinct);
    return target.type;
  }

  if (is_agg_domain_range_equivalent(agg_type)) {
    return agg_arg;
  } else {
    // Nullability of the target needs to match that of the agg for proper initialization
    // of target (aggregate) values
    return target.type->withNullable(agg_arg->nullable());
  }
}

inline void set_compact_type(TargetInfo& target, const hdk::ir::Type* new_type) {
  if (target.is_agg) {
    const auto agg_type = target.agg_kind;
    if (agg_type != kCOUNT || !target.agg_arg_type) {
      target.agg_arg_type = new_type;
      return;
    }
  }
  target.type = new_type;
}

inline uint64_t exp_to_scale(const unsigned exp) {
  uint64_t res = 1;
  for (unsigned i = 0; i < exp; ++i) {
    res *= 10;
  }
  return res;
}

inline size_t get_bit_width(const hdk::ir::Type* type) {
  size_t res = type->isString() ? 32 : type->canonicalSize() * 8;
  if (res < 0) {
    throw std::runtime_error("Unexpected type: " + type->toString());
  }
  return static_cast<size_t>(res);
}

#endif  // QUERYENGINE_SQLTYPESLAYOUT_H
