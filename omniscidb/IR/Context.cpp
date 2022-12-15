/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Context.h"
#include "Exception.h"
#include "Type.h"

#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>

#include <regex>
#include <unordered_map>

namespace hdk::ir {

class ContextImpl {
 public:
  ContextImpl(Context& ctx) : ctx_(ctx) { null_type_.reset(new NullType(ctx_)); }

  const NullType* null() { return null_type_.get(); }

  const BooleanType* boolean(bool nullable) {
    auto& res = boolean_types_[nullable];
    if (!res) {
      res.reset(new BooleanType(ctx_, nullable));
    }
    return res.get();
  }

  const IntegerType* integer(int size, bool nullable) {
    auto& res = integer_types_[std::make_pair(size, nullable)];
    if (!res) {
      if (size != 1 && size != 2 && size != 4 && size != 8) {
        throw UnsupportedTypeError()
            << "Unsupported integer size (must be 1, 2, 4, 8): " << size;
      }
      res.reset(new IntegerType(ctx_, size, nullable));
    }
    return res.get();
  }

  const FloatingPointType* fp(FloatingPointType::Precision precision, bool nullable) {
    auto& res = floating_point_types_[std::make_pair(precision, nullable)];
    if (!res) {
      res.reset(new FloatingPointType(ctx_, precision, nullable));
    }
    return res.get();
  }

  const DecimalType* decimal(int size, int precision, int scale, bool nullable) {
    auto& res = decimal_types_[std::make_tuple(size, precision, scale, nullable)];
    if (!res) {
      if (size != 8) {
        throw UnsupportedTypeError() << "Unsupported decimal size (must be 8): " << size;
      }
      if (precision < 0) {
        throw InvalidTypeError() << "Negative precision for decimal: " << precision;
      }
      if (scale < 0) {
        throw InvalidTypeError() << "Negative scale for decimal: " << scale;
      }
      res.reset(new DecimalType(ctx_, size, precision, scale, nullable));
    }
    return res.get();
  }

  const VarCharType* varChar(int max_length, bool nullable) {
    auto& res = varchar_types_[std::make_pair(max_length, nullable)];
    if (!res) {
      if (max_length < 0) {
        throw UnsupportedTypeError()
            << "Negative max length for varchar type: " << max_length;
      }
      res.reset(new VarCharType(ctx_, max_length, nullable));
    }
    return res.get();
  }

  const TextType* text(bool nullable) {
    auto& res = text_types_[nullable];
    if (!res) {
      res.reset(new TextType(ctx_, nullable));
    }
    return res.get();
  }

  const DateType* date(int size, TimeUnit unit, bool nullable) {
    auto& res = date_types_[std::make_tuple(size, unit, nullable)];
    if (!res) {
      if (size != 2 && size != 4 && size != 8) {
        throw UnsupportedTypeError()
            << "Unsupported date size (must be 2, 4, 8): " << size;
      }
      if (unit == TimeUnit::kMonth) {
        throw UnsupportedTypeError() << "Month unit is not supported for dates.";
      }
      if (size == 2 && unit != TimeUnit::kDay) {
        throw InvalidTypeError() << "Only Day unit is allowed for 16-bit date type.";
      }
      if (size == 4 && unit != TimeUnit::kDay && unit != TimeUnit::kSecond) {
        throw InvalidTypeError()
            << "Only Day and Second units are allowed for 32-bit date type.";
      }
      res.reset(new DateType(ctx_, size, unit, nullable));
    }
    return res.get();
  }

  const TimeType* time(int size, TimeUnit unit, bool nullable) {
    auto& res = time_types_[std::make_tuple(size, unit, nullable)];
    if (!res) {
      if (size != 2 && size != 4 && size != 8) {
        throw UnsupportedTypeError()
            << "Unsupported time size (must be 2, 4, 8): " << size;
      }
      if (unit == TimeUnit::kMonth || unit == TimeUnit::kDay) {
        throw InvalidTypeError() << "Time type doesn't support Month or Day unit.";
      }
      if (size == 2 && unit != TimeUnit::kSecond) {
        throw InvalidTypeError() << "Only Second unit is allowed for 16-bit time type.";
      }
      if (size == 4 && (unit == TimeUnit::kMicro || unit == TimeUnit::kNano)) {
        throw InvalidTypeError()
            << "Only Second and Milli units are allowed for 32-bit time type.";
      }
      res.reset(new TimeType(ctx_, size, unit, nullable));
    }
    return res.get();
  }

  const TimestampType* timestamp(TimeUnit unit, bool nullable) {
    auto& res = timestamp_types_[std::make_pair(unit, nullable)];
    if (!res) {
      if (unit == TimeUnit::kMonth || unit == TimeUnit::kDay) {
        throw InvalidTypeError() << "Timestamp type doesn't support Month or Day unit.";
      }
      res.reset(new TimestampType(ctx_, unit, nullable));
    }
    return res.get();
  }

  const IntervalType* interval(int size, TimeUnit unit, bool nullable) {
    auto& res = interval_types_[std::make_tuple(size, unit, nullable)];
    if (!res) {
      res.reset(new IntervalType(ctx_, size, unit, nullable));
    }
    return res.get();
  }

  const FixedLenArrayType* arrayFixed(int num_elems,
                                      const Type* elem_type,
                                      bool nullable) {
    auto& res = fixed_array_types_[std::make_tuple(num_elems, elem_type, nullable)];
    if (!res) {
      if (&ctx_ != &elem_type->ctx()) {
        throw InvalidTypeError()
            << "Element type should belong to the same context as array type.";
      }
      if (num_elems < 0) {
        throw InvalidTypeError()
            << "Fixed sized arrays should have at least one element.";
      }
      if (!elem_type->isBoolean() && !elem_type->isNumber() && !elem_type->isDateTime() &&
          !elem_type->isInterval() && !elem_type->isExtDictionary() &&
          !elem_type->isNull()) {
        throw UnsupportedTypeError()
            << "Unsupported element type for array: " << elem_type;
      }
      res.reset(new FixedLenArrayType(ctx_, num_elems, elem_type, nullable));
    }
    return res.get();
  }

  const VarLenArrayType* arrayVarLen(const Type* elem_type,
                                     int offs_size,
                                     bool nullable) {
    auto& res = varlen_array_types_[std::make_tuple(offs_size, elem_type, nullable)];
    if (!res) {
      if (&ctx_ != &elem_type->ctx()) {
        throw InvalidTypeError()
            << "Element type should belong to the same context as array type.";
      }
      if (offs_size != 4) {
        throw UnsupportedTypeError()
            << "Only 32-bit offsets are supported for varlen arrays.";
      }
      if (!elem_type->isBoolean() && !elem_type->isNumber() && !elem_type->isDateTime() &&
          !elem_type->isInterval() && !elem_type->isExtDictionary() &&
          !elem_type->isNull()) {
        throw UnsupportedTypeError()
            << "Unsupported element type for array: " << elem_type;
      }
      res.reset(new VarLenArrayType(ctx_, elem_type, offs_size, nullable));
    }
    return res.get();
  }

  const ExtDictionaryType* extDict(const Type* elem_type, int dict_id, int index_size) {
    auto& res = ext_dict_types_[std::make_tuple(elem_type, dict_id, index_size)];
    if (!res) {
      if (&ctx_ != &elem_type->ctx()) {
        throw InvalidTypeError()
            << "Element type should belong to the same context as dictionary type.";
      }
      if (index_size != 1 && index_size != 2 && index_size != 4) {
        throw UnsupportedTypeError()
            << "Unsupported index size (must be 1, 2, 4): " << index_size;
      }
      if (!elem_type->isString()) {
        throw UnsupportedTypeError()
            << "Only string types are supported for external dictionary element.";
      }
      res.reset(new ExtDictionaryType(ctx_, elem_type, dict_id, index_size));
    }
    return res.get();
  }

  const ColumnType* column(const Type* column_type, bool nullable) {
    auto& res = column_types_[std::make_pair(column_type, nullable)];
    if (!res) {
      if (&ctx_ != &column_type->ctx()) {
        throw InvalidTypeError()
            << "Column subtype should belong to the same context as column type.";
      }
      res.reset(new ColumnType(ctx_, column_type, nullable));
    }
    return res.get();
  }

  const ColumnListType* columnList(const Type* column_type, int length, bool nullable) {
    auto& res = column_list_types_[std::make_tuple(column_type, length, nullable)];
    if (!res) {
      if (&ctx_ != &column_type->ctx()) {
        throw InvalidTypeError()
            << "Column subtype should belong to the same context as column type.";
      }
      if (length < 1) {
        throw InvalidTypeError() << "Column list length should be positive integer.";
      }
      res.reset(new ColumnListType(ctx_, column_type, length, nullable));
    }
    return res.get();
  }

  const Type* copyType(const Type* type) {
    if (&type->ctx() == &ctx_) {
      return type;
    }

    switch (type->id()) {
      case Type::kNull:
        return null();
      case Type::kBoolean:
        return boolean(type->nullable());
      case Type::kInteger:
        return integer(type->size(), type->nullable());
      case Type::kFloatingPoint: {
        auto fp_type = static_cast<const FloatingPointType*>(type);
        return fp(fp_type->precision(), fp_type->nullable());
      }
      case Type::kDecimal: {
        auto decimal_type = static_cast<const DecimalType*>(type);
        return decimal(decimal_type->size(),
                       decimal_type->precision(),
                       decimal_type->scale(),
                       decimal_type->nullable());
      }
      case Type::kVarChar: {
        auto varchar_type = static_cast<const VarCharType*>(type);
        return varChar(varchar_type->maxLength(), type->nullable());
      }
      case Type::kText:
        return text(type->nullable());
      case Type::kDate: {
        auto date_type = static_cast<const DateType*>(type);
        return date(date_type->size(), date_type->unit(), date_type->nullable());
      }
      case Type::kTime: {
        auto time_type = static_cast<const TimeType*>(type);
        return time(time_type->size(), time_type->unit(), time_type->nullable());
      }
      case Type::kTimestamp: {
        auto timestamp_type = static_cast<const TimestampType*>(type);
        return timestamp(timestamp_type->unit(), timestamp_type->nullable());
      }
      case Type::kInterval: {
        auto interval_type = static_cast<const IntervalType*>(type);
        return interval(
            interval_type->size(), interval_type->unit(), interval_type->nullable());
      }
      case Type::kFixedLenArray: {
        auto fixed_array_type = static_cast<const FixedLenArrayType*>(type);
        return arrayFixed(fixed_array_type->numElems(),
                          fixed_array_type->elemType(),
                          fixed_array_type->nullable());
      }
      case Type::kVarLenArray: {
        auto varlen_array_type = static_cast<const VarLenArrayType*>(type);
        return arrayVarLen(varlen_array_type->elemType(),
                           varlen_array_type->offsetSize(),
                           varlen_array_type->nullable());
      }
      case Type::kExtDictionary: {
        auto ext_dict_type = static_cast<const ExtDictionaryType*>(type);
        return extDict(
            ext_dict_type->elemType(), ext_dict_type->dictId(), ext_dict_type->size());
      }
      default:
        throw InvalidTypeError() << "Unexpected type: " << type;
    }
  }

  const Type* typeFromString(const std::string& val) {
    auto val_lower = boost::trim_copy(boost::to_lower_copy(val));
    std::smatch match_res;

    // Null type.
    if (val_lower == "nullt") {
      return null();
    }
    // Boolean.
    if (std::regex_match(val_lower, match_res, std::regex("bool(\\[nn\\])?"))) {
      return boolean(!match_res[1].matched);
    }
    // Integer types.
    if (std::regex_match(
            val_lower, match_res, std::regex("int(8|16|32|64)?(\\[nn\\])?"))) {
      int size = match_res[1].matched ? std::stoi(match_res[1].str()) / 8 : 8;
      return integer(size, !match_res[2].matched);
    }
    // Floating point types.
    if (std::regex_match(val_lower, match_res, std::regex("fp(32|64)?(\\[nn\\])?"))) {
      FloatingPointType::Precision precision =
          match_res[1].matched ? (match_res[1].str() == "32" ? FloatingPointType::kFloat
                                                             : FloatingPointType::kDouble)
                               : FloatingPointType::kDouble;
      return fp(precision, !match_res[2].matched);
    }
    // Decimal types.
    if (std::regex_match(val_lower,
                         match_res,
                         std::regex("(?:dec|decimal)(?:64)?\\(\\s*(\\d+)\\s*,\\s*(\\d+)"
                                    "\\s*\\)(\\[nn\\])?"))) {
      int precision = std::stoi(match_res[1].str());
      int scale = std::stoi(match_res[2].str());
      return decimal(8, precision, scale, !match_res[3].matched);
    }
    // Varchar type.
    if (std::regex_match(
            val_lower, match_res, std::regex("varchar\\((\\d+)\\)(\\[nn\\])?"))) {
      int max_length = std::stoi(match_res[1].str());
      return varChar(max_length, !match_res[2].matched);
    }
    // Text type.
    if (std::regex_match(val_lower, match_res, std::regex("text(\\[nn\\])?"))) {
      return text(!match_res[1].matched);
    }
    auto parse_unit = [](const std::string& val) -> TimeUnit {
      if (val == "m") {
        return TimeUnit::kMonth;
      } else if (val == "d") {
        return TimeUnit::kDay;
      } else if (val == "s") {
        return TimeUnit::kSecond;
      } else if (val == "ms") {
        return TimeUnit::kMilli;
      } else if (val == "us") {
        return TimeUnit::kMicro;
      } else if (val == "ns") {
        return TimeUnit::kNano;
      }

      throw InvalidTypeError() << "Unexpected time unit string: " << val;
    };
    // Date types.
    if (std::regex_match(
            val_lower,
            match_res,
            std::regex("date(16|32|64)?(?:\\[(m|d|s|ms|us|ns)\\])?(\\[nn\\])?"))) {
      int size = match_res[1].matched ? std::stoi(match_res[1].str()) / 8 : 8;
      auto unit = match_res[2].matched ? parse_unit(match_res[2].str())
                  : size == 2          ? TimeUnit::kDay
                                       : TimeUnit::kSecond;
      return date(size, unit, !match_res[3].matched);
    }
    // Time types.
    if (std::regex_match(
            val_lower,
            match_res,
            std::regex("time(16|32|64)?(?:\\[(m|d|s|ms|us|ns)\\])?(\\[nn\\])?"))) {
      int size = match_res[1].matched ? std::stoi(match_res[1].str()) / 8 : 8;
      auto unit = match_res[2].matched ? parse_unit(match_res[2].str())
                  : size == 2          ? TimeUnit::kSecond
                  : size == 4          ? TimeUnit::kMilli
                                       : TimeUnit::kMicro;
      return time(size, unit, !match_res[3].matched);
    }
    // Timestamp types.
    if (std::regex_match(
            val_lower,
            match_res,
            std::regex("timestamp(?:64)?(?:\\[(m|d|s|ms|us|ns)\\])?(\\[nn\\])?"))) {
      auto unit =
          match_res[1].matched ? parse_unit(match_res[1].str()) : TimeUnit::kMicro;
      return timestamp(unit, !match_res[2].matched);
    }
    // Interval types.
    if (std::regex_match(
            val_lower,
            match_res,
            std::regex("interval(16|32|64)?(?:\\[(m|d|s|ms|us|ns)\\])?(\\[nn\\])?"))) {
      int size = match_res[1].matched ? std::stoi(match_res[1].str()) / 8 : 8;
      auto unit =
          match_res[2].matched ? parse_unit(match_res[2].str()) : TimeUnit::kMicro;
      return interval(size, unit, !match_res[3].matched);
    }
    // Fixed length array types.
    if (std::regex_match(
            val_lower, match_res, std::regex("array\\((.+)\\)\\((\\d+)\\)(\\[nn\\])?"))) {
      const Type* elem_type = typeFromString(match_res[1].str());
      int num_elems = std::stoi(match_res[2].str());
      return arrayFixed(num_elems, elem_type, !match_res[3].matched);
    }
    // Varlen array types.
    if (std::regex_match(
            val_lower, match_res, std::regex("array\\((.+)\\)(\\[nn\\])?"))) {
      const Type* elem_type = typeFromString(match_res[1].str());
      return arrayVarLen(elem_type, 4, !match_res[2].matched);
    }
    // Dictionary types.
    if (std::regex_match(
            val_lower,
            match_res,
            std::regex("dict(8|16|32)?(?:\\((.+)\\))?(?:\\[(-?\\d+)\\])?"))) {
      int size = match_res[1].matched ? std::stoi(match_res[1].str()) / 8 : 4;
      const Type* elem_type =
          match_res[2].matched ? typeFromString(match_res[2].str()) : text(true);
      int dict_id = match_res[3].matched ? std::stoi(match_res[3].str()) : 0;
      return extDict(elem_type, dict_id, size);
    }

    // Datetime types
    // date - date64[m|d|s|ms|us|ns]
    // time
    // timestamp - no size
    // interval

    // Types with subtypes
    // array fix len - array(SUBTYPEx123
    // array varlen - array32(SUBTYPE))
    // dict - dict64(subtype)[dict_id]

    throw InvalidTypeError() << "Invalid type string: " << val;
  }

 private:
  Context& ctx_;
  std::unique_ptr<const NullType> null_type_;
  std::unordered_map<bool, std::unique_ptr<const BooleanType>> boolean_types_;
  std::unordered_map<std::pair<int, bool>,
                     std::unique_ptr<const IntegerType>,
                     boost::hash<std::pair<int, bool>>>
      integer_types_;
  std::unordered_map<std::pair<FloatingPointType::Precision, bool>,
                     std::unique_ptr<const FloatingPointType>,
                     boost::hash<std::pair<FloatingPointType::Precision, bool>>>
      floating_point_types_;
  std::unordered_map<std::tuple<int, int, int, bool>,
                     std::unique_ptr<const DecimalType>,
                     boost::hash<std::tuple<int, int, int, bool>>>
      decimal_types_;
  std::unordered_map<std::pair<int, bool>,
                     std::unique_ptr<const VarCharType>,
                     boost::hash<std::pair<int, bool>>>
      varchar_types_;
  std::unordered_map<bool, std::unique_ptr<const TextType>> text_types_;
  std::unordered_map<std::tuple<int, TimeUnit, bool>,
                     std::unique_ptr<const DateType>,
                     boost::hash<std::tuple<int, TimeUnit, bool>>>
      date_types_;
  std::unordered_map<std::tuple<int, TimeUnit, bool>,
                     std::unique_ptr<const TimeType>,
                     boost::hash<std::tuple<int, TimeUnit, bool>>>
      time_types_;
  std::unordered_map<std::pair<TimeUnit, bool>,
                     std::unique_ptr<const TimestampType>,
                     boost::hash<std::pair<TimeUnit, bool>>>
      timestamp_types_;
  std::unordered_map<std::tuple<int, TimeUnit, bool>,
                     std::unique_ptr<const IntervalType>,
                     boost::hash<std::tuple<int, TimeUnit, bool>>>
      interval_types_;
  std::unordered_map<std::tuple<int, const Type*, bool>,
                     std::unique_ptr<const FixedLenArrayType>,
                     boost::hash<std::tuple<int, const Type*, bool>>>
      fixed_array_types_;
  std::unordered_map<std::tuple<int, const Type*, bool>,
                     std::unique_ptr<const VarLenArrayType>,
                     boost::hash<std::tuple<int, const Type*, bool>>>
      varlen_array_types_;
  std::unordered_map<std::tuple<const Type*, int, int>,
                     std::unique_ptr<const ExtDictionaryType>,
                     boost::hash<std::tuple<const Type*, int, int>>>
      ext_dict_types_;
  std::unordered_map<std::pair<const Type*, bool>,
                     std::unique_ptr<const ColumnType>,
                     boost::hash<std::pair<const Type*, bool>>>
      column_types_;
  std::unordered_map<std::tuple<const Type*, int, bool>,
                     std::unique_ptr<const ColumnListType>,
                     boost::hash<std::tuple<const Type*, int, bool>>>
      column_list_types_;
};

Context::Context() : impl_(new ContextImpl(*this)) {}

Context::~Context() {}

const NullType* Context::null() {
  return impl_->null();
}

const BooleanType* Context::boolean(bool nullable) {
  return impl_->boolean(nullable);
}

const IntegerType* Context::integer(int size, bool nullable) {
  return impl_->integer(size, nullable);
}

const IntegerType* Context::int8(bool nullable) {
  return impl_->integer(1, nullable);
}

const IntegerType* Context::int16(bool nullable) {
  return impl_->integer(2, nullable);
}

const IntegerType* Context::int32(bool nullable) {
  return impl_->integer(4, nullable);
}

const IntegerType* Context::int64(bool nullable) {
  return impl_->integer(8, nullable);
}

const FloatingPointType* Context::fp(FloatingPointType::Precision precision,
                                     bool nullable) {
  return impl_->fp(precision, nullable);
}

const FloatingPointType* Context::fp32(bool nullable) {
  return impl_->fp(FloatingPointType::kFloat, nullable);
}

const FloatingPointType* Context::fp64(bool nullable) {
  return impl_->fp(FloatingPointType::kDouble, nullable);
}

const DecimalType* Context::decimal(int size, int precision, int scale, bool nullable) {
  return impl_->decimal(size, precision, scale, nullable);
}

const DecimalType* Context::decimal64(int precision, int scale, bool nullable) {
  return impl_->decimal(8, precision, scale, nullable);
}

const VarCharType* Context::varChar(int max_length, bool nullable) {
  return impl_->varChar(max_length, nullable);
}

const TextType* Context::text(bool nullable) {
  return impl_->text(nullable);
}

const DateType* Context::date(int size, TimeUnit unit, bool nullable) {
  return impl_->date(size, unit, nullable);
}

const DateType* Context::date16(TimeUnit unit, bool nullable) {
  return impl_->date(2, unit, nullable);
}

const DateType* Context::date32(TimeUnit unit, bool nullable) {
  return impl_->date(4, unit, nullable);
}

const DateType* Context::date64(TimeUnit unit, bool nullable) {
  return impl_->date(8, unit, nullable);
}

const TimeType* Context::time(int size, TimeUnit unit, bool nullable) {
  return impl_->time(size, unit, nullable);
}

const TimeType* Context::time16(TimeUnit unit, bool nullable) {
  return impl_->time(2, unit, nullable);
}

const TimeType* Context::time32(TimeUnit unit, bool nullable) {
  return impl_->time(4, unit, nullable);
}

const TimeType* Context::time64(TimeUnit unit, bool nullable) {
  return impl_->time(8, unit, nullable);
}

const TimestampType* Context::timestamp(TimeUnit unit, bool nullable) {
  return impl_->timestamp(unit, nullable);
}

const IntervalType* Context::interval(int size, TimeUnit unit, bool nullable) {
  return impl_->interval(size, unit, nullable);
}
const IntervalType* Context::interval16(TimeUnit unit, bool nullable) {
  return impl_->interval(2, unit, nullable);
}
const IntervalType* Context::interval32(TimeUnit unit, bool nullable) {
  return impl_->interval(4, unit, nullable);
}
const IntervalType* Context::interval64(TimeUnit unit, bool nullable) {
  return impl_->interval(8, unit, nullable);
}

const FixedLenArrayType* Context::arrayFixed(int num_elems,
                                             const Type* elem_type,
                                             bool nullable) {
  return impl_->arrayFixed(num_elems, elem_type, nullable);
}

const VarLenArrayType* Context::arrayVarLen(const Type* elem_type,
                                            int offs_size,
                                            bool nullable) {
  return impl_->arrayVarLen(elem_type, offs_size, nullable);
}

const ExtDictionaryType* Context::extDict(const Type* elem_type,
                                          int dict_id,
                                          int index_size) {
  return impl_->extDict(elem_type, dict_id, index_size);
}

const ColumnType* Context::column(const Type* column_type, bool nullable) {
  return impl_->column(column_type, nullable);
}

const ColumnListType* Context::columnList(const Type* column_type,
                                          int length,
                                          bool nullable) {
  return impl_->columnList(column_type, length, nullable);
}

const Type* Context::copyType(const Type* type) {
  return impl_->copyType(type);
}

const Type* Context::typeFromString(const std::string& val) {
  return impl_->typeFromString(val);
}

Context& Context::defaultCtx() {
  static Context default_context;
  return default_context;
}

}  // namespace hdk::ir