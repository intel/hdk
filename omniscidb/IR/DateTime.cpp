/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DateTime.h"

namespace hdk::ir {}  // namespace hdk::ir

std::string toString(hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kMonth:
      return "Month";
    case hdk::ir::TimeUnit::kDay:
      return "Day";
    case hdk::ir::TimeUnit::kSecond:
      return "Second";
    case hdk::ir::TimeUnit::kMilli:
      return "Milli";
    case hdk::ir::TimeUnit::kMicro:
      return "Micro";
    case hdk::ir::TimeUnit::kNano:
      return "Nano";
    default:
      return "InvalidTimeUnit";
  }
}

std::ostream& operator<<(std::ostream& os, hdk::ir::TimeUnit unit) {
  os << toString(unit);
  return os;
}
