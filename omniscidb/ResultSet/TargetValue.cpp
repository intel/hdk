/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TargetValue.h"

#include "IR/Type.h"
#include "Shared/InlineNullValues.h"

bool isNull(const ScalarTargetValue& val, const hdk::ir::Type* type) {
  auto int_val_p = boost::get<int64_t>(&val);
  if (int_val_p) {
    auto logical_type = type->canonicalize();
    return *int_val_p == inline_int_null_value(logical_type);
  }
  auto float_val_p = boost::get<float>(&val);
  if (float_val_p) {
    return *float_val_p == inline_fp_null_value<float>();
  }
  auto double_val_p = boost::get<double>(&val);
  if (double_val_p) {
    return *double_val_p == inline_fp_null_value<double>();
  }
  auto& str_val = boost::get<NullableString>(val);
  return str_val.type() == typeid(void*);
}
