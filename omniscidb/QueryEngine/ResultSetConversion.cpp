#include "ResultSet.h"
#include "Execute.h"

#ifdef ENABLE_ARROW_CONVERTER
#include "arrow/builder.h"
#include "arrow/buffer.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/metadata.h"
#include "arrow/ipc/writer.h"
#include "arrow/memory_pool.h"
#include "arrow/type.h"

#include <future>

typedef boost::variant<std::vector<bool>,
                       std::vector<int8_t>,
                       std::vector<int16_t>,
                       std::vector<int32_t>,
                       std::vector<int64_t>,
                       std::vector<float>,
                       std::vector<double>>
    ValueArray;

namespace {

SQLTypes get_physical_type(const SQLTypeInfo& ti) {
  auto logical_type = ti.get_type();
  if (IS_INTEGER(logical_type)) {
    switch (ti.get_size()) {
      case 2:
        return kSMALLINT;
      case 4:
        return kINT;
      case 8:
        return kBIGINT;
      default:
        CHECK(false);
    }
  }
  return logical_type;
}

template <typename TYPE, typename C_TYPE>
void create_or_append_value(const ScalarTargetValue& val_cty,
                            std::shared_ptr<ValueArray>& values,
                            const size_t max_size) {
  auto pval_cty = boost::get<C_TYPE>(&val_cty);
  CHECK(pval_cty);
  auto val_ty = static_cast<TYPE>(*pval_cty);
  if (!values) {
    values = std::make_shared<ValueArray>(std::vector<TYPE>());
    boost::get<std::vector<TYPE>>(*values).reserve(max_size);
  }
  CHECK(values);
  auto values_ty = boost::get<std::vector<TYPE>>(values.get());
  CHECK(values_ty);
  values_ty->push_back(val_ty);
}

template <typename TYPE>
void create_or_append_validity(const ScalarTargetValue& value,
                               const SQLTypeInfo& col_type,
                               std::shared_ptr<std::vector<bool>>& null_bitmap,
                               const size_t max_size) {
  if (col_type.get_notnull()) {
    CHECK(!null_bitmap);
    return;
  }
  auto pvalue = boost::get<TYPE>(&value);
  CHECK(pvalue);
  bool is_valid = false;
  if (col_type.is_integer()) {
    is_valid = inline_int_null_val(col_type) != static_cast<int64_t>(*pvalue);
  } else {
    CHECK(col_type.is_fp());
    is_valid = inline_fp_null_val(col_type) != static_cast<double>(*pvalue);
  }
  if (!null_bitmap) {
    null_bitmap = std::make_shared<std::vector<bool>>();
    null_bitmap->reserve(max_size);
  }
  CHECK(null_bitmap);
  null_bitmap->push_back(is_valid);
}

}  // namespace

namespace arrow {

#define ASSERT_OK(expr)           \
  do {                            \
    Status s = (expr);            \
    if (!s.ok()) {                \
      LOG(ERROR) << s.ToString(); \
    }                             \
  } while (0)

TypePtr get_arrow_type(const SQLTypeInfo& mapd_type) {
  switch (get_physical_type(mapd_type)) {
    case kBOOLEAN:
      return std::make_shared<BooleanType>();
    case kSMALLINT:
      return std::make_shared<Int16Type>();
    case kINT:
      return std::make_shared<Int32Type>();
    case kBIGINT:
      return std::make_shared<Int64Type>();
    case kFLOAT:
      return std::make_shared<FloatType>();
    case kDOUBLE:
      return std::make_shared<DoubleType>();
    case kDECIMAL:
    case kNUMERIC:
    case kCHAR:
    case kVARCHAR:
    case kTIME:
    case kTIMESTAMP:
    case kTEXT:
    case kDATE:
    case kARRAY:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    default:
      CHECK(false);
  }
  return nullptr;
}
std::shared_ptr<Field> make_field(const std::string name, const SQLTypeInfo& target_type) {
  return std::make_shared<Field>(name, get_arrow_type(target_type), !target_type.get_notnull());
}

// Cited from arrow/test-util.h
template <typename TYPE, typename C_TYPE>
std::shared_ptr<Array> array_from_vector(const std::shared_ptr<std::vector<bool>> is_valid,
                                         const std::vector<C_TYPE>& values) {
  std::shared_ptr<Array> out;
  MemoryPool* pool = default_memory_pool();
  typename TypeTraits<TYPE>::BuilderType builder(pool);
  if (is_valid) {
    for (size_t i = 0; i < values.size(); ++i) {
      if ((*is_valid)[i]) {
        ASSERT_OK(builder.Append(values[i]));
      } else {
        ASSERT_OK(builder.AppendNull());
      }
    }
  } else {
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_OK(builder.Append(values[i]));
    }
  }
  ASSERT_OK(builder.Finish(&out));
  return out;
}

std::shared_ptr<Array> generate_column(const Field& field,
                                       const std::shared_ptr<std::vector<bool>> is_valid,
                                       const ValueArray& values) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto vals_bool = boost::get<std::vector<bool>>(&values);
      CHECK(vals_bool);
      return array_from_vector<BooleanType, bool>(is_valid, *vals_bool);
    }
    case Type::INT8: {
      auto vals_i8 = boost::get<std::vector<int8_t>>(&values);
      CHECK(vals_i8);
      return array_from_vector<Int8Type, int8_t>(is_valid, *vals_i8);
    }
    case Type::INT16: {
      auto vals_i16 = boost::get<std::vector<int16_t>>(&values);
      CHECK(vals_i16);
      return array_from_vector<Int16Type, int16_t>(is_valid, *vals_i16);
    }
    case Type::INT32: {
      auto vals_i32 = boost::get<std::vector<int32_t>>(&values);
      CHECK(vals_i32);
      return array_from_vector<Int32Type, int32_t>(is_valid, *vals_i32);
    }
    case Type::INT64: {
      auto vals_i64 = boost::get<std::vector<int64_t>>(&values);
      CHECK(vals_i64);
      return array_from_vector<Int64Type, int64_t>(is_valid, *vals_i64);
    }
    case Type::FLOAT: {
      auto vals_float = boost::get<std::vector<float>>(&values);
      CHECK(vals_float);
      return array_from_vector<FloatType, float>(is_valid, *vals_float);
    }
    case Type::DOUBLE: {
      auto vals_double = boost::get<std::vector<double>>(&values);
      CHECK(vals_double);
      return array_from_vector<DoubleType, double>(is_valid, *vals_double);
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

std::vector<std::shared_ptr<Array>> generate_columns(
    const std::vector<std::shared_ptr<Field>>& fields,
    const std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmaps,
    const std::vector<std::shared_ptr<ValueArray>>& column_values) {
  const auto col_count = fields.size();
  CHECK_GT(col_count, 0);
  std::vector<std::shared_ptr<arrow::Array>> columns(col_count, nullptr);
  auto generate = [&](const size_t i, std::shared_ptr<arrow::Array>& column) {
    column = generate_column(*fields[i], null_bitmaps[i], *column_values[i]);
  };
  if (col_count > 1) {
    std::vector<std::future<void>> child_threads;
    for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
      child_threads.push_back(std::async(std::launch::async, generate, col_idx, std::ref(columns[col_idx])));
    }
    for (auto& child : child_threads) {
      child.get();
    }
  } else {
    generate(0, columns[0]);
  }
  return columns;
}

std::shared_ptr<ValueArray> create_value_array(const Field& field, const size_t value_count) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto array = std::make_shared<ValueArray>(std::vector<bool>());
      boost::get<std::vector<bool>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT8: {
      auto array = std::make_shared<ValueArray>(std::vector<int8_t>());
      boost::get<std::vector<int8_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT16: {
      auto array = std::make_shared<ValueArray>(std::vector<int16_t>());
      boost::get<std::vector<int16_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT32: {
      auto array = std::make_shared<ValueArray>(std::vector<int32_t>());
      boost::get<std::vector<int32_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::INT64: {
      auto array = std::make_shared<ValueArray>(std::vector<int64_t>());
      boost::get<std::vector<int64_t>>(*array).reserve(value_count);
      return array;
    }
    case Type::FLOAT: {
      auto array = std::make_shared<ValueArray>(std::vector<float>());
      boost::get<std::vector<float>>(*array).reserve(value_count);
      return array;
    }
    case Type::DOUBLE: {
      auto array = std::make_shared<ValueArray>(std::vector<double>());
      boost::get<std::vector<double>>(*array).reserve(value_count);
      return array;
    }
    default:
      CHECK(false);
  }
  return nullptr;
}

void append_value_array(ValueArray& dst, const ValueArray& src, const Field& field) {
  switch (field.type()->id()) {
    case Type::BOOL: {
      auto dst_bool = boost::get<std::vector<bool>>(&dst);
      auto src_bool = boost::get<std::vector<bool>>(&src);
      CHECK(dst_bool && src_bool);
      dst_bool->insert(dst_bool->end(), src_bool->begin(), src_bool->end());
    } break;
    case Type::INT8: {
      auto dst_i8 = boost::get<std::vector<int8_t>>(&dst);
      auto src_i8 = boost::get<std::vector<int8_t>>(&src);
      CHECK(dst_i8 && src_i8);
      dst_i8->insert(dst_i8->end(), src_i8->begin(), src_i8->end());
    } break;
    case Type::INT16: {
      auto dst_i16 = boost::get<std::vector<int16_t>>(&dst);
      auto src_i16 = boost::get<std::vector<int16_t>>(&src);
      CHECK(dst_i16 && src_i16);
      dst_i16->insert(dst_i16->end(), src_i16->begin(), src_i16->end());
    } break;
    case Type::INT32: {
      auto dst_i32 = boost::get<std::vector<int32_t>>(&dst);
      auto src_i32 = boost::get<std::vector<int32_t>>(&src);
      CHECK(dst_i32 && src_i32);
      dst_i32->insert(dst_i32->end(), src_i32->begin(), src_i32->end());
    } break;
    case Type::INT64: {
      auto dst_i64 = boost::get<std::vector<int64_t>>(&dst);
      auto src_i64 = boost::get<std::vector<int64_t>>(&src);
      CHECK(dst_i64 && src_i64);
      dst_i64->insert(dst_i64->end(), src_i64->begin(), src_i64->end());
    } break;
    case Type::FLOAT: {
      auto dst_flt = boost::get<std::vector<float>>(&dst);
      auto src_flt = boost::get<std::vector<float>>(&src);
      CHECK(dst_flt && src_flt);
      dst_flt->insert(dst_flt->end(), src_flt->begin(), src_flt->end());
    } break;
    case Type::DOUBLE: {
      auto dst_dbl = boost::get<std::vector<double>>(&dst);
      auto src_dbl = boost::get<std::vector<double>>(&src);
      CHECK(dst_dbl && src_dbl);
      dst_dbl->insert(dst_dbl->end(), src_dbl->begin(), src_dbl->end());
    } break;
    default:
      CHECK(false);
      break;
  }
}

#undef ASSERT_OK

}  // namespace arrow

std::pair<std::vector<std::shared_ptr<arrow::Array>>, size_t> ResultSet::getArrowColumns(
    const std::vector<std::shared_ptr<arrow::Field>>& fields) const {
  const auto entry_count = entryCount();
  if (!entry_count) {
    return {{}, 0};
  }
  const auto col_count = colCount();
  size_t row_count = 0;

  // TODO(miyu): speed up for columnar buffers
  auto fetch = [&](std::vector<std::shared_ptr<ValueArray>>& value_seg,
                   std::vector<std::shared_ptr<std::vector<bool>>>& null_bitmap_seg,
                   const size_t start_entry,
                   const size_t end_entry) -> size_t {
    CHECK_EQ(value_seg.size(), col_count);
    CHECK_EQ(null_bitmap_seg.size(), col_count);
    const auto entry_count = end_entry - start_entry;
    size_t seg_row_count = 0;
    for (size_t i = start_entry; i < end_entry; ++i) {
      auto row = getRowAt(i);
      if (row.empty()) {
        continue;
      }
      ++seg_row_count;
      for (size_t j = 0; j < col_count; ++j) {
        const auto& col_type = getColType(j);
        auto scalar_value = boost::get<ScalarTargetValue>(&row[j]);
        // TODO(miyu): support more types other than scalar.
        CHECK(scalar_value);
        switch (get_physical_type(col_type)) {
          case kBOOLEAN:
            create_or_append_value<bool, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kSMALLINT:
            create_or_append_value<int16_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kINT:
            create_or_append_value<int32_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kBIGINT:
            create_or_append_value<int64_t, int64_t>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<int64_t>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kFLOAT:
            create_or_append_value<float, float>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<float>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          case kDOUBLE:
            create_or_append_value<double, double>(*scalar_value, value_seg[j], entry_count);
            create_or_append_validity<double>(*scalar_value, col_type, null_bitmap_seg[j], entry_count);
            break;
          default:
            // TODO(miyu): support more scalar types.
            CHECK(false);
        }
      }
    }
    return seg_row_count;
  };

  std::vector<std::shared_ptr<ValueArray>> column_values(col_count, nullptr);
  std::vector<std::shared_ptr<std::vector<bool>>> null_bitmaps(col_count, nullptr);
  const bool multithreaded = entry_count > 10000 && !isTruncated();
  if (multithreaded) {
    const size_t cpu_count = cpu_threads();
    std::vector<std::future<size_t>> child_threads;
    std::vector<std::vector<std::shared_ptr<ValueArray>>> column_value_segs(
        cpu_count, std::vector<std::shared_ptr<ValueArray>>(col_count, nullptr));
    std::vector<std::vector<std::shared_ptr<std::vector<bool>>>> null_bitmap_segs(
        cpu_count, std::vector<std::shared_ptr<std::vector<bool>>>(col_count, nullptr));
    const auto stride = (entry_count + cpu_count - 1) / cpu_count;
    for (size_t i = 0, start_entry = 0; start_entry < entry_count; ++i, start_entry += stride) {
      const auto end_entry = std::min(entry_count, start_entry + stride);
      child_threads.push_back(std::async(std::launch::async,
                                         fetch,
                                         std::ref(column_value_segs[i]),
                                         std::ref(null_bitmap_segs[i]),
                                         start_entry,
                                         end_entry));
    }
    for (auto& child : child_threads) {
      row_count += child.get();
    }
    for (size_t i = 0; i < fields.size(); ++i) {
      column_values[i] = arrow::create_value_array(*fields[i], row_count);
      if (fields[i]->nullable()) {
        null_bitmaps[i] = std::make_shared<std::vector<bool>>();
        null_bitmaps[i]->reserve(row_count);
      }
      for (size_t j = 0; j < cpu_count; ++j) {
        if (!column_value_segs[j][i]) {
          continue;
        }
        arrow::append_value_array(*column_values[i], *column_value_segs[j][i], *fields[i]);
        if (fields[i]->nullable()) {
          CHECK(null_bitmap_segs[j][i]);
          null_bitmaps[i]->insert(
              null_bitmaps[i]->end(), null_bitmap_segs[j][i]->begin(), null_bitmap_segs[j][i]->end());
        }
      }
    }
  } else {
    row_count = fetch(column_values, null_bitmaps, size_t(0), entry_count);
  }

  return {generate_columns(fields, null_bitmaps, column_values), row_count};
}

arrow::RecordBatch ResultSet::convertToArrow() const {
  const auto col_count = colCount();
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (size_t i = 0; i < col_count; ++i) {
    fields.push_back(arrow::make_field("", getColType(i)));
  }
  auto schema = std::make_shared<arrow::Schema>(fields);
  std::vector<std::shared_ptr<arrow::Array>> columns;
  size_t row_count = 0;
  if (col_count > 0) {
    std::tie(columns, row_count) = getArrowColumns(fields);
  }
  return arrow::RecordBatch(schema, row_count, columns);
}

#endif  // ENABLE_ARROW_CONVERTER
