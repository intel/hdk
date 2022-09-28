/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TypeUtils.h"

#include "Context.h"

namespace hdk::ir {

std::string sqlTypeName(const hdk::ir::Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
      return "BOOLEAN";
    case hdk::ir::Type::kDecimal: {
      auto precision = type->as<hdk::ir::DecimalType>()->precision();
      auto scale = type->as<hdk::ir::DecimalType>()->scale();
      return "DECIMAL(" + std::to_string(precision) + "," + std::to_string(scale) + ")";
    }
    case hdk::ir::Type::kInteger:
      switch (type->size()) {
        case 1:
          return "TINYINT";
        case 2:
          return "SMALLINT";
        case 4:
          return "INT";
        case 8:
          return "BIGINT";
        default:
          break;
      }
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          return "FLOAT";
        case hdk::ir::FloatingPointType::kDouble:
          return "DOUBLE";
        default:
          break;
      }
      break;
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kInterval:
      break;
    case hdk::ir::Type::kExtDictionary:
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      return "TEXT";
    default:
      break;
  }
  throw std::runtime_error("Unsupported type: " + type->toString());
}

}  // namespace hdk::ir
