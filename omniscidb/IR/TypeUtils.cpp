/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TypeUtils.h"

#include "Context.h"

namespace hdk::ir {

const Type* logicalType(const Type* type) {
  if (type->isExtDictionary() && type->size() != 4) {
    auto dict_type = type->as<ExtDictionaryType>();
    return type->ctx().extDict(dict_type->elemType(), dict_type->dictId(), 4);
  } else if (type->isDate()) {
    auto date_type = type->as<DateType>();
    if (date_type->unit() != TimeUnit::kSecond || date_type->size() != 8) {
      return type->ctx().date64(TimeUnit::kSecond, date_type->nullable());
    }
  } else if (type->isTime() && type->size() != 8) {
    auto time_type = type->as<TimeType>();
    return type->ctx().time64(time_type->unit(), time_type->nullable());
  } else if (type->isInterval() && type->size() != 8) {
    auto interval_type = type->as<TimestampType>();
    return type->ctx().interval64(interval_type->unit(), interval_type->nullable());
  } else if (type->isFixedLenArray()) {
    auto array_type = type->as<FixedLenArrayType>();
    return type->ctx().arrayVarLen(array_type->elemType(), 4, array_type->nullable());
  }

  return type;
}

const int logicalSize(const Type* type) {
  switch (type->id()) {
    case hdk::ir::Type::kNull:
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
    case hdk::ir::Type::kFloatingPoint:
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kColumn:
    case hdk::ir::Type::kColumnList:
    case hdk::ir::Type::kVarLenArray:
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      return type->size();
    case hdk::ir::Type::kExtDictionary:
      return 4;
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kInterval:
      return 8;
    default:
      abort();
  }
}

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
