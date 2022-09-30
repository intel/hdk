/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>

namespace hdk::ir {

enum class TimeUnit {
  kMonth,
  kDay,
  kSecond,
  kMilli,
  kMicro,
  kNano,
};

}  // namespace hdk::ir

std::string toString(hdk::ir::TimeUnit unit);
std::ostream& operator<<(std::ostream& os, hdk::ir::TimeUnit unit);

namespace hdk::ir {

inline int64_t unitsPerSecond(TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kSecond:
      return 1;
    case hdk::ir::TimeUnit::kMilli:
      return 1'000;
    case hdk::ir::TimeUnit::kMicro:
      return 1'000'000;
    case hdk::ir::TimeUnit::kNano:
      return 1'000'000'000;
    default:
      throw std::runtime_error("Enexpected unit in unitsInSecond: " + ::toString(unit));
  }
}

}  // namespace hdk::ir
