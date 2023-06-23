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

#include "ArrowResultSet.h"

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

#include "ArrowStorage/ArrowStorageUtils.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "Shared/ArrowUtil.h"

namespace {

const hdk::ir::Type* type_from_arrow_field(hdk::ir::Context& ctx,
                                           const arrow::Field& field) {
  switch (field.type()->id()) {
    case arrow::Type::INT8:
      return ctx.int8(field.nullable());
    case arrow::Type::INT16:
      return ctx.int16(field.nullable());
    case arrow::Type::INT32:
      return ctx.int32(field.nullable());
    case arrow::Type::INT64:
      return ctx.int64(field.nullable());
    case arrow::Type::FLOAT:
      return ctx.fp32(field.nullable());
    case arrow::Type::DOUBLE:
      return ctx.fp64(field.nullable());
    case arrow::Type::DICTIONARY:
      return ctx.extDict(ctx.text(field.nullable()), 0);
    case arrow::Type::TIMESTAMP: {
      switch (static_cast<const arrow::TimestampType&>(*field.type()).unit()) {
        case arrow::TimeUnit::SECOND:
          return ctx.timestamp(hdk::ir::TimeUnit::kSecond, field.nullable());
        case arrow::TimeUnit::MILLI:
          return ctx.timestamp(hdk::ir::TimeUnit::kMilli, field.nullable());
        case arrow::TimeUnit::MICRO:
          return ctx.timestamp(hdk::ir::TimeUnit::kMicro, field.nullable());
        case arrow::TimeUnit::NANO:
          return ctx.timestamp(hdk::ir::TimeUnit::kNano, field.nullable());
        default:
          UNREACHABLE();
      }
    }
    case arrow::Type::DATE32:
      return ctx.date32(hdk::ir::TimeUnit::kDay, field.nullable());
    case arrow::Type::DATE64:
      return ctx.date64(hdk::ir::TimeUnit::kSecond, field.nullable());
    case arrow::Type::TIME32:
      return ctx.time64(hdk::ir::TimeUnit::kSecond, field.nullable());
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

ArrowResultSet::ArrowResultSet(const std::shared_ptr<ResultSet>& rows,
                               const std::vector<hdk::ir::TargetMetaInfo>& targets_meta,
                               const ExecutorDeviceType device_type)
    : rows_(rows), targets_meta_(targets_meta), crt_row_idx_(0) {
  resultSetArrowLoopback(device_type);
  auto schema = record_batch_->schema();
  for (int i = 0; i < schema->num_fields(); ++i) {
    std::shared_ptr<arrow::Field> field = schema->field(i);
    auto type = type_from_arrow_field(hdk::ir::Context::defaultCtx(), *schema->field(i));
    column_metainfo_.emplace_back(field->name(), type);
    columns_.emplace_back(record_batch_->column(i));
  }
}

ArrowResultSet::ArrowResultSet(
    const std::shared_ptr<ResultSet>& rows,
    const std::vector<hdk::ir::TargetMetaInfo>& targets_meta,
    const ExecutorDeviceType device_type,
    const size_t min_result_size_for_bulk_dictionary_fetch,
    const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch)
    : rows_(rows), targets_meta_(targets_meta), crt_row_idx_(0) {
  resultSetArrowLoopback(device_type,
                         min_result_size_for_bulk_dictionary_fetch,
                         max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
  auto schema = record_batch_->schema();
  for (int i = 0; i < schema->num_fields(); ++i) {
    std::shared_ptr<arrow::Field> field = schema->field(i);
    auto type = type_from_arrow_field(hdk::ir::Context::defaultCtx(), *schema->field(i));
    column_metainfo_.emplace_back(field->name(), type);
    columns_.emplace_back(record_batch_->column(i));
  }
}

template <typename Type, typename ArrayType>
void ArrowResultSet::appendValue(std::vector<TargetValue>& row,
                                 const arrow::Array& column,
                                 const Type null_val,
                                 const size_t idx) const {
  const auto& col = static_cast<const ArrayType&>(column);
  row.emplace_back(col.IsNull(idx) ? null_val : static_cast<Type>(col.Value(idx)));
}

std::vector<std::string> ArrowResultSet::getDictionaryStrings(
    const size_t col_idx) const {
  if (col_idx >= colCount()) {
    throw std::runtime_error("ArrowResultSet::getDictionaryStrings: col_idx is invalid.");
  }
  auto column_type = colType(col_idx);
  if (!column_type->isExtDictionary()) {
    throw std::runtime_error(
        "ArrowResultSet::getDictionaryStrings: col_idx does not refer to column of type "
        "TEXT.");
  }
  const auto& column = *columns_[col_idx];
  CHECK_EQ(arrow::Type::DICTIONARY, column.type_id());
  const auto& dict_column = static_cast<const arrow::DictionaryArray&>(column);
  const auto& dictionary =
      static_cast<const arrow::StringArray&>(*dict_column.dictionary());
  const size_t dictionary_size = dictionary.length();
  std::vector<std::string> dictionary_strings;
  dictionary_strings.reserve(dictionary_size);
  for (size_t d = 0; d < dictionary_size; ++d) {
    dictionary_strings.emplace_back(dictionary.GetString(d));
  }
  return dictionary_strings;
}

std::vector<TargetValue> ArrowResultSet::getRowAt(const size_t index) const {
  if (index >= rowCount()) {
    return {};
  }

  CHECK_LT(index, rowCount());
  std::vector<TargetValue> row;
  for (int i = 0; i < record_batch_->num_columns(); ++i) {
    const auto& column = *columns_[i];
    auto column_type = colType(i);
    switch (column_type->id()) {
      case hdk::ir::Type::kInteger:
        switch (column_type->size()) {
          case 1:
            CHECK_EQ(arrow::Type::INT8, column.type_id());
            appendValue<int64_t, arrow::Int8Array>(
                row, column, inline_int_null_value(column_type), index);
            break;
          case 2:
            CHECK_EQ(arrow::Type::INT16, column.type_id());
            appendValue<int64_t, arrow::Int16Array>(
                row, column, inline_int_null_value(column_type), index);
            break;
          case 4:
            CHECK_EQ(arrow::Type::INT32, column.type_id());
            appendValue<int64_t, arrow::Int32Array>(
                row, column, inline_int_null_value(column_type), index);
            break;
          case 8:
            CHECK_EQ(arrow::Type::INT64, column.type_id());
            appendValue<int64_t, arrow::Int64Array>(
                row, column, inline_int_null_value(column_type), index);
            break;
          default:
            CHECK(false);
        }
        break;
      case hdk::ir::Type::kFloatingPoint:
        switch (column_type->as<hdk::ir::FloatingPointType>()->precision()) {
          case hdk::ir::FloatingPointType::kFloat:
            CHECK_EQ(arrow::Type::FLOAT, column.type_id());
            appendValue<float, arrow::FloatArray>(
                row, column, inline_fp_null_value<float>(), index);
            break;
          case hdk::ir::FloatingPointType::kDouble:
            CHECK_EQ(arrow::Type::DOUBLE, column.type_id());
            appendValue<double, arrow::DoubleArray>(
                row, column, inline_fp_null_value<double>(), index);
            break;
          default:
            CHECK(false);
        }
        break;
      case hdk::ir::Type::kExtDictionary: {
        CHECK_EQ(arrow::Type::DICTIONARY, column.type_id());
        const auto& dict_column = static_cast<const arrow::DictionaryArray&>(column);
        if (dict_column.IsNull(index)) {
          row.emplace_back(NullableString(nullptr));
        } else {
          const auto& indices =
              static_cast<const arrow::Int32Array&>(*dict_column.indices());
          const auto& dictionary =
              static_cast<const arrow::StringArray&>(*dict_column.dictionary());
          row.emplace_back(dictionary.GetString(indices.Value(index)));
        }
        break;
      }
      case hdk::ir::Type::kTimestamp: {
        CHECK_EQ(arrow::Type::TIMESTAMP, column.type_id());
        appendValue<int64_t, arrow::TimestampArray>(
            row, column, inline_int_null_value(column_type), index);
        break;
      }
      case hdk::ir::Type::kDate: {
        // TODO(wamsi): constexpr?
        CHECK(arrow::Type::DATE32 == column.type_id() ||
              arrow::Type::DATE64 == column.type_id());
        column_type->as<hdk::ir::DateType>()->unit() == hdk::ir::TimeUnit::kDay
            ? appendValue<int64_t, arrow::Date32Array>(
                  row, column, inline_int_null_value(column_type), index)
            : appendValue<int64_t, arrow::Date64Array>(
                  row, column, inline_int_null_value(column_type), index);
        break;
      }
      case hdk::ir::Type::kTime: {
        CHECK_EQ(arrow::Type::TIME32, column.type_id());
        appendValue<int64_t, arrow::Time32Array>(
            row, column, inline_int_null_value(column_type), index);
        break;
      }
      default:
        CHECK(false);
    }
  }
  return row;
}

std::vector<TargetValue> ArrowResultSet::getNextRow(const bool translate_strings,
                                                    const bool decimal_to_double) const {
  if (crt_row_idx_ == rowCount()) {
    return {};
  }
  CHECK_LT(crt_row_idx_, rowCount());
  auto row = getRowAt(crt_row_idx_);
  ++crt_row_idx_;
  return row;
}

size_t ArrowResultSet::colCount() const {
  return column_metainfo_.size();
}

const hdk::ir::Type* ArrowResultSet::colType(size_t col_idx) const {
  CHECK_LT(col_idx, column_metainfo_.size());
  return column_metainfo_[col_idx].type();
}

bool ArrowResultSet::definitelyHasNoRows() const {
  return !rowCount();
}

size_t ArrowResultSet::rowCount() const {
  return record_batch_->num_rows();
}

// Function is for parity with ResultSet interface
// and associated tests
size_t ArrowResultSet::entryCount() const {
  return rowCount();
}

// Function is for parity with ResultSet interface
// and associated tests
bool ArrowResultSet::isEmpty() const {
  return rowCount() == static_cast<size_t>(0);
}

void ArrowResultSet::resultSetArrowLoopback(const ExecutorDeviceType device_type) {
  resultSetArrowLoopback(
      device_type,
      ArrowResultSetConverter::default_min_result_size_for_bulk_dictionary_fetch,
      ArrowResultSetConverter::
          default_max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
}

void ArrowResultSet::resultSetArrowLoopback(
    const ExecutorDeviceType device_type,
    const size_t min_result_size_for_bulk_dictionary_fetch,
    const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {
  std::vector<std::string> col_names;

  if (!targets_meta_.empty()) {
    for (auto& meta : targets_meta_) {
      col_names.push_back(meta.get_resname());
    }
  } else {
    for (unsigned int i = 0; i < rows_->colCount(); i++) {
      col_names.push_back("col_" + std::to_string(i));
    }
  }

  // We convert the given rows to arrow, which gets serialized
  // into a buffer by Arrow Wire.
  auto converter = ArrowResultSetConverter(
      rows_,
      col_names,
      -1,
      min_result_size_for_bulk_dictionary_fetch,
      max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
  converter.transport_method_ = ArrowTransport::WIRE;
  converter.device_type_ = device_type;

  // Lifetime of the result buffer is that of ArrowResultSet
  results_ = std::make_shared<ArrowResult>(converter.getArrowResult());

  // Create a reader for reading back serialized
  arrow::io::BufferReader reader(
      reinterpret_cast<const uint8_t*>(results_->df_buffer.data()), results_->df_size);

  ARROW_ASSIGN_OR_THROW(auto batch_reader,
                        arrow::ipc::RecordBatchStreamReader::Open(&reader));

  ARROW_THROW_NOT_OK(batch_reader->ReadNext(&record_batch_));

  // Collect dictionaries from the record batch into the dictionary memo.
  ARROW_THROW_NOT_OK(
      arrow::ipc::internal::CollectDictionaries(*record_batch_, &dictionary_memo_));

  CHECK_EQ(record_batch_->schema()->num_fields(), record_batch_->num_columns());
}

std::unique_ptr<ArrowResultSet> result_set_arrow_loopback(
    const std::shared_ptr<ResultSet>& rows,
    const ExecutorDeviceType device_type,
    const size_t min_result_size_for_bulk_dictionary_fetch,
    const double max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch) {
  std::vector<hdk::ir::TargetMetaInfo> dummy_targets_meta;
  return std::make_unique<ArrowResultSet>(
      rows,
      dummy_targets_meta,
      device_type,
      min_result_size_for_bulk_dictionary_fetch,
      max_dictionary_to_result_size_ratio_for_bulk_dictionary_fetch);
}
