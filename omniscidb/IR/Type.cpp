/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Type.h"
#include "Context.h"
#include "Exception.h"

#include <iostream>

using namespace std::string_literals;

namespace hdk::ir {

Type::Type(Context& ctx, Id id, int size, bool nullable)
    : ctx_(ctx), id_(id), size_(size), nullable_(nullable) {}

const Type* Type::copyTo(Context& ctx) const {
  return ctx.copyType(this);
}

bool Type::equal(const Type& other) const {
  if (&ctx_ == &other.ctx_) {
    return this == &other;
  }
  return id_ == other.id_ && size_ == other.size_ && nullable_ == other.nullable_;
}

bool Type::operator==(const Type& other) const {
  return equal(other);
}

void Type::print() const {
  std::cout << toString() << std::endl;
}

std::string_view Type::nullableStr() const {
  return nullable_ ? "" : "[NN]";
}

NullType::NullType(Context& ctx) : Type(ctx, kNull, 0, true) {}

const NullType* NullType::make(Context& ctx) {
  return ctx.null();
}

const Type* NullType::withNullable(bool nullable) const {
  return this;
}

std::string NullType::toString() const {
  return "NULLT";
}

BooleanType::BooleanType(Context& ctx, bool nullable)
    : Type(ctx, kBoolean, 1, nullable) {}

const BooleanType* BooleanType::make(Context& ctx, bool nullable) {
  return ctx.boolean(nullable);
}

const Type* BooleanType::withNullable(bool nullable) const {
  return make(ctx_, nullable);
}

std::string BooleanType::toString() const {
  std::stringstream ss;
  ss << "BOOL"s << nullableStr();
  return ss.str();
}

IntegerType::IntegerType(Context& ctx, int size, bool nullable)
    : Type(ctx, kInteger, size, nullable) {}

const IntegerType* IntegerType::make(Context& ctx, int size, bool nullable) {
  return ctx.integer(size, nullable);
}

const Type* IntegerType::withNullable(bool nullable) const {
  return make(ctx_, size_, nullable);
}

std::string IntegerType::toString() const {
  std::stringstream ss;
  ss << "INT" << (size() * 8) << nullableStr();
  return ss.str();
}

FloatingPointType::FloatingPointType(Context& ctx, Precision precision, bool nullable)
    : Type(ctx, kFloatingPoint, precisiontToSize(precision), nullable)
    , precision_(precision) {}

int FloatingPointType::precisiontToSize(Precision precision) {
  switch (precision) {
    case kFloat:
      return 4;
    case kDouble:
      return 8;
    default:
      throw InvalidTypeError() << "Unexpected precision value: " << precision << " ("
                               << (int)precision << ")";
  }
}

const FloatingPointType* FloatingPointType::make(Context& ctx,
                                                 Precision precision,
                                                 bool nullable) {
  return ctx.fp(precision, nullable);
}

const Type* FloatingPointType::withNullable(bool nullable) const {
  return make(ctx_, precision_, nullable);
}

bool FloatingPointType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_fp = static_cast<const FloatingPointType*>(&other);
  return precision_ == other_fp->precision_;
}

std::string FloatingPointType::toString() const {
  std::stringstream ss;
  ss << "FP" << (size() * 8) << nullableStr();
  return ss.str();
}

DecimalType::DecimalType(Context& ctx, int size, int precision, int scale, bool nullable)
    : Type(ctx, kDecimal, size, nullable), precision_(precision), scale_(scale) {}

const DecimalType* DecimalType::make(Context& ctx,
                                     int size,
                                     int precision,
                                     int scale,
                                     bool nullable) {
  return ctx.decimal(size, precision, scale, nullable);
}

const Type* DecimalType::withNullable(bool nullable) const {
  return make(ctx_, size_, precision_, scale_, nullable);
}

bool DecimalType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_decimal = static_cast<const DecimalType*>(&other);
  return precision_ == other_decimal->precision_ && scale_ == other_decimal->scale_;
}

std::string DecimalType::toString() const {
  std::stringstream ss;
  ss << "DEC" << size() * 8 << "(" << precision_ << "," << scale_ << ")" << nullableStr();
  return ss.str();
}

VarCharType::VarCharType(Context& ctx, int max_length, bool nullable)
    : Type(ctx, kVarChar, -1, nullable), max_length_(max_length) {}

const VarCharType* VarCharType::make(Context& ctx, int max_length, bool nullable) {
  return ctx.varChar(max_length, nullable);
}

const Type* VarCharType::withNullable(bool nullable) const {
  return make(ctx_, max_length_, nullable);
}

bool VarCharType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_varchar = static_cast<const VarCharType*>(&other);
  return max_length_ == other_varchar->max_length_;
}

std::string VarCharType::toString() const {
  std::stringstream ss;
  ss << "VARCHAR(" << max_length_ << ")" << nullableStr();
  return ss.str();
}

TextType::TextType(Context& ctx, bool nullable) : Type(ctx, kText, -1, nullable) {}

const TextType* TextType::make(Context& ctx, bool nullable) {
  return ctx.text(nullable);
}

const Type* TextType::withNullable(bool nullable) const {
  return make(ctx_, nullable);
}

std::string TextType::toString() const {
  std::stringstream ss;
  ss << "TEXT" << nullableStr();
  return ss.str();
}

DateTimeBaseType::DateTimeBaseType(Context& ctx,
                                   Id id,
                                   int size,
                                   TimeUnit unit,
                                   bool nullable)
    : Type(ctx, id, size, nullable), unit_(unit) {}

bool DateTimeBaseType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_datetime = static_cast<const DateTimeBaseType*>(&other);
  return unit_ == other_datetime->unit_;
}

std::string_view DateTimeBaseType::unitStr() const {
  switch (unit_) {
    case TimeUnit::kMonth:
      return "[m]";
    case TimeUnit::kDay:
      return "[d]";
    case TimeUnit::kSecond:
      return "[s]";
    case TimeUnit::kMilli:
      return "[ms]";
    case TimeUnit::kMicro:
      return "[us]";
    case TimeUnit::kNano:
      return "[ns]";
    default:
      throw InvalidTypeError() << "Unexpected TimeUnit: " << unit_ << "(" << (int)unit_
                               << ")";
  }
}

DateType::DateType(Context& ctx, int size, TimeUnit unit, bool nullable)
    : DateTimeBaseType(ctx, kDate, size, unit, nullable) {}

const DateType* DateType::make(Context& ctx, int size, TimeUnit unit, bool nullable) {
  return ctx.date(size, unit, nullable);
}

const Type* DateType::withNullable(bool nullable) const {
  return make(ctx_, size_, unit_, nullable);
}

const Type* DateType::canonicalize() const {
  if (unit_ != TimeUnit::kSecond || size_ != 8) {
    return ctx_.date64(TimeUnit::kSecond, nullable_);
  }
  return this;
}

std::string DateType::toString() const {
  std::stringstream ss;
  ss << "DATE" << (size() * 8) << unitStr() << nullableStr();
  return ss.str();
}

TimeType::TimeType(Context& ctx, int size, TimeUnit unit, bool nullable)
    : DateTimeBaseType(ctx, kTime, size, unit, nullable) {}

const TimeType* TimeType::make(Context& ctx, int size, TimeUnit unit, bool nullable) {
  return ctx.time(size, unit, nullable);
}

const Type* TimeType::withNullable(bool nullable) const {
  return make(ctx_, size_, unit_, nullable);
}

const Type* TimeType::canonicalize() const {
  if (size_ != 8) {
    return ctx_.time64(unit_, nullable_);
  }
  return this;
}

std::string TimeType::toString() const {
  std::stringstream ss;
  ss << "TIME" << (size() * 8) << unitStr() << nullableStr();
  return ss.str();
}

TimestampType::TimestampType(Context& ctx, TimeUnit unit, bool nullable)
    : DateTimeBaseType(ctx, kTimestamp, 8, unit, nullable) {}

const TimestampType* TimestampType::make(Context& ctx, TimeUnit unit, bool nullable) {
  return ctx.timestamp(unit, nullable);
}

const Type* TimestampType::withNullable(bool nullable) const {
  return make(ctx_, unit_, nullable);
}

std::string TimestampType::toString() const {
  std::stringstream ss;
  ss << "TIMESTAMP" << unitStr() << nullableStr();
  return ss.str();
}

IntervalType::IntervalType(Context& ctx, int size, TimeUnit unit, bool nullable)
    : DateTimeBaseType(ctx, kInterval, size, unit, nullable) {}

const IntervalType* IntervalType::make(Context& ctx,
                                       int size,
                                       TimeUnit unit,
                                       bool nullable) {
  return ctx.interval(size, unit, nullable);
}

const Type* IntervalType::withNullable(bool nullable) const {
  return make(ctx_, size_, unit_, nullable);
}

const Type* IntervalType::canonicalize() const {
  if (size_ != 8) {
    return ctx_.interval64(unit_, nullable_);
  }
  return this;
}

std::string IntervalType::toString() const {
  std::stringstream ss;
  ss << "INTERVAL" << (size() * 8) << unitStr() << nullableStr();
  return ss.str();
}

ArrayBaseType::ArrayBaseType(Context& ctx,
                             Id id,
                             int size,
                             const Type* elem_type,
                             bool nullable)
    : Type(ctx, id, size, nullable), elem_type_(elem_type) {}

bool ArrayBaseType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_array_base = static_cast<const ArrayBaseType*>(&other);
  return elem_type_->equal(*other_array_base->elem_type_);
}

FixedLenArrayType::FixedLenArrayType(Context& ctx,
                                     int num_elems,
                                     const Type* elem_type,
                                     bool nullable)
    : ArrayBaseType(ctx,
                    kFixedLenArray,
                    num_elems * elem_type->size(),
                    elem_type,
                    nullable)
    , num_elems_(num_elems) {}

const FixedLenArrayType* FixedLenArrayType::make(Context& ctx,
                                                 int num_elems,
                                                 const Type* elem_type,
                                                 bool nullable) {
  return ctx.arrayFixed(num_elems, elem_type, nullable);
}

const Type* FixedLenArrayType::withNullable(bool nullable) const {
  return make(ctx_, num_elems_, elem_type_, nullable);
}

const ArrayBaseType* FixedLenArrayType::withElemType(const Type* elem_type) const {
  return make(ctx_, num_elems_, elem_type, nullable_);
}

bool FixedLenArrayType::equal(const Type& other) const {
  if (!ArrayBaseType::equal(other)) {
    return false;
  }
  auto other_fixed_array = static_cast<const FixedLenArrayType*>(&other);
  return num_elems_ == other_fixed_array->num_elems_;
}

std::string FixedLenArrayType::toString() const {
  std::stringstream ss;
  ss << "ARRAY(" << elem_type_->toString() << ")(" << num_elems_ << ")" << nullableStr();
  return ss.str();
}

VarLenArrayType::VarLenArrayType(Context& ctx,
                                 const Type* elem_type,
                                 int offs_size,
                                 bool nullable)
    : ArrayBaseType(ctx, kVarLenArray, -1, elem_type, nullable), offs_size_(offs_size) {}

const VarLenArrayType* VarLenArrayType::make(Context& ctx,
                                             const Type* elem_type,
                                             int offs_size,
                                             bool nullable) {
  return ctx.arrayVarLen(elem_type, offs_size, nullable);
}

const Type* VarLenArrayType::withNullable(bool nullable) const {
  return make(ctx_, elem_type_, offs_size_, nullable);
}

const ArrayBaseType* VarLenArrayType::withElemType(const Type* elem_type) const {
  return make(ctx_, elem_type, offs_size_, nullable_);
}

bool VarLenArrayType::equal(const Type& other) const {
  if (!ArrayBaseType::equal(other)) {
    return false;
  }
  auto other_varlen_array = static_cast<const VarLenArrayType*>(&other);
  return offs_size_ == other_varlen_array->offs_size_;
}

std::string VarLenArrayType::toString() const {
  std::stringstream ss;
  ss << "ARRAY" << (offs_size_ * 8) << "(" << elem_type_->toString() << ")"
     << nullableStr();
  return ss.str();
}

ExtDictionaryType::ExtDictionaryType(Context& ctx,
                                     const Type* elem_type,
                                     int dict_id,
                                     int index_size)
    : Type(ctx, kExtDictionary, index_size, elem_type->nullable())
    , elem_type_(elem_type)
    , dict_id_(dict_id) {}

const ExtDictionaryType* ExtDictionaryType::make(Context& ctx,
                                                 const Type* elem_type,
                                                 int dict_id,
                                                 int index_size) {
  return ctx.extDict(elem_type, dict_id, index_size);
}

const Type* ExtDictionaryType::withNullable(bool nullable) const {
  return make(ctx_, elem_type_->withNullable(nullable), dict_id_, size_);
}

const Type* ExtDictionaryType::canonicalize() const {
  if (size_ != 4) {
    return ctx_.extDict(elem_type_, dict_id_, 4);
  }
  return this;
}

bool ExtDictionaryType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_dict = static_cast<const ExtDictionaryType*>(&other);
  return elem_type_->equal(*other_dict->elem_type_) && dict_id_ == other_dict->dict_id_;
}

std::string ExtDictionaryType::toString() const {
  std::stringstream ss;
  ss << "DICT" << (size() * 8) << "(" << elem_type_->toString() << ")[" << dict_id_ << "]"
     << nullableStr();
  return ss.str();
}

ColumnType::ColumnType(Context& ctx, const Type* column_type, bool nullable)
    : Type(ctx, kColumn, column_type->size(), nullable), column_type_(column_type) {}

const ColumnType* ColumnType::make(Context& ctx, const Type* column_type, bool nullable) {
  return ctx.column(column_type, nullable);
}

const Type* ColumnType::withNullable(bool nullable) const {
  return make(ctx_, column_type_, nullable);
}

bool ColumnType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_column = static_cast<const ColumnType*>(&other);
  return column_type_->equal(*other_column->column_type_);
}

std::string ColumnType::toString() const {
  std::stringstream ss;
  ss << "COLUMN(" << column_type_->toString() << ")";
  return ss.str();
}

ColumnListType::ColumnListType(Context& ctx,
                               const Type* column_type,
                               int length,
                               bool nullable)
    : Type(ctx, kColumnList, column_type->size(), nullable)
    , column_type_(column_type)
    , length_(length) {}

const ColumnListType* ColumnListType::make(Context& ctx,
                                           const Type* column_type,
                                           int length,
                                           bool nullable) {
  return ctx.columnList(column_type, length, nullable);
}

const Type* ColumnListType::withNullable(bool nullable) const {
  return make(ctx_, column_type_, length_, nullable);
}

bool ColumnListType::equal(const Type& other) const {
  if (!Type::equal(other)) {
    return false;
  }
  auto other_column_list = static_cast<const ColumnListType*>(&other);
  return length_ == other_column_list->length_ &&
         column_type_->equal(*other_column_list->column_type_);
}

std::string ColumnListType::toString() const {
  std::stringstream ss;
  ss << "COLUMN_LIST" << length_ << "(" << column_type_->toString() << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, hdk::ir::Type::Id type_id) {
  os << toString(type_id);
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         hdk::ir::FloatingPointType::Precision precision) {
  os << toString(precision);
  return os;
}

std::ostream& operator<<(std::ostream& os, const hdk::ir::Type* type) {
  os << type->toString();
  return os;
}

}  // namespace hdk::ir

std::string toString(hdk::ir::Type::Id type_id) {
  switch (type_id) {
    case hdk::ir::Type::kNull:
      return "Null";
    case hdk::ir::Type::kBoolean:
      return "Boolean";
    case hdk::ir::Type::kInteger:
      return "Integer";
    case hdk::ir::Type::kFloatingPoint:
      return "FloatingPoint";
    case hdk::ir::Type::kDecimal:
      return "Decimal";
    case hdk::ir::Type::kVarChar:
      return "VarChar";
    case hdk::ir::Type::kText:
      return "Text";
    case hdk::ir::Type::kDate:
      return "Date";
    case hdk::ir::Type::kTime:
      return "Time";
    case hdk::ir::Type::kTimestamp:
      return "Timestamp";
    case hdk::ir::Type::kInterval:
      return "Interval";
    case hdk::ir::Type::kFixedLenArray:
      return "FixedLenArray";
    case hdk::ir::Type::kVarLenArray:
      return "VarLenArray";
    case hdk::ir::Type::kExtDictionary:
      return "ExtDictionary";
    case hdk::ir::Type::kColumn:
      return "Column";
    case hdk::ir::Type::kColumnList:
      return "ColumnList";
    default:
      return "InvalidTypeId";
  }
}

std::string toString(hdk::ir::FloatingPointType::Precision precision) {
  switch (precision) {
    case hdk::ir::FloatingPointType::kFloat:
      return "Float";
    case hdk::ir::FloatingPointType::kDouble:
      return "Double";
    default:
      return "InvalidPrecision";
  }
}
