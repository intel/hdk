/*
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

#pragma once

#include "TestHelpers.h"

#include "arrow/api.h"

#include <gtest/gtest.h>

namespace ArrowTestHelpers {

void compare_arrow_array_decimal_impl(
    const std::vector<int64_t>& expected,
    const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  using ArrowColType = arrow::NumericArray<arrow::Decimal128Type>;
  const arrow::ArrayVector& chunks = actual->chunks();

  int64_t null_val = inline_null_value<int64_t>();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    const arrow::Decimal128* chunk_data = chunk->data()->GetValues<arrow::Decimal128>(1);
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        ASSERT_EQ(expected[compared], chunk_data[j].ToInteger<int64_t>().ValueOrDie());
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

template <typename ArrowColType>
void compare_arrow_array_date_impl(const std::vector<int64_t>& expected,
                                   const std::shared_ptr<arrow::ChunkedArray>& actual) {
  using value_type = typename ArrowColType::value_type;
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  const arrow::ArrayVector& chunks = actual->chunks();

  // Time32/Time64/Timestamp all have int64_t canonical type. So we comparing with int64
  // null value.
  const int64_t null_val = inline_null_value<int64_t>();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto arrow_row_array = std::static_pointer_cast<ArrowColType>(chunk);

    const value_type* chunk_data = arrow_row_array->raw_values();
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        ASSERT_EQ(expected[compared], chunk_data[j]);
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

template <typename TYPE>
void compare_arrow_array_impl(const std::vector<TYPE>& expected,
                              const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(actual->type()->ToString(),
            arrow::CTypeTraits<TYPE>::type_singleton()->ToString());
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  using ArrowColType = arrow::NumericArray<typename arrow::CTypeTraits<TYPE>::ArrowType>;
  const arrow::ArrayVector& chunks = actual->chunks();

  TYPE null_val = inline_null_value<TYPE>();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto arrow_row_array = std::static_pointer_cast<ArrowColType>(chunk);

    const TYPE* chunk_data = arrow_row_array->raw_values();
    for (int64_t j = 0; j < arrow_row_array->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        if constexpr (std::is_floating_point_v<TYPE>) {
          ASSERT_NEAR(expected[compared], chunk_data[j], 0.001);
        } else {
          ASSERT_EQ(expected[compared], chunk_data[j]);
        }
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

void compare_arrow_array_dict(const std::vector<std::string>& expected,
                              const std::shared_ptr<arrow::ChunkedArray>& actual) {
  const arrow::ArrayVector& chunks = actual->chunks();

  std::string null_val = "<NULL>";
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
    auto values = std::static_pointer_cast<arrow::StringArray>(dict_array->dictionary());
    auto indices = std::static_pointer_cast<arrow::Int32Array>(dict_array->indices());
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      auto val = chunk->GetScalar(j).ValueOrDie();
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        ASSERT_EQ(values->GetString(indices->Value(j)), expected[compared]);
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

void compare_arrow_array_str(const std::vector<std::string>& expected,
                             const std::shared_ptr<arrow::ChunkedArray>& actual) {
  const arrow::ArrayVector& chunks = actual->chunks();

  std::string null_val = "<NULL>";
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto str_array = std::static_pointer_cast<arrow::StringArray>(chunk);
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      if (expected[compared] == null_val) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        ASSERT_EQ(str_array->GetString(j), expected[compared]);
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

template <>
void compare_arrow_array_impl(const std::vector<std::string>& expected,
                              const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  if (actual->type()->id() == arrow::Type::DICTIONARY) {
    compare_arrow_array_dict(expected, actual);
  } else {
    ASSERT_EQ(actual->type()->id(), arrow::Type::STRING);
    compare_arrow_array_str(expected, actual);
  }
}

template <typename T, typename ARROW_LIST_TYPE>
void compare_arrow_array_list_impl(const std::vector<std::vector<T>>& expected,
                                   const std::shared_ptr<arrow::ChunkedArray>& actual) {
  using ArrowColType = arrow::NumericArray<typename arrow::CTypeTraits<T>::ArrowType>;
  const arrow::ArrayVector& chunks = actual->chunks();
  size_t compared = 0;

  for (int i = 0; i < actual->num_chunks(); i++) {
    auto chunk = chunks[i];
    auto list_array = std::static_pointer_cast<ARROW_LIST_TYPE>(chunk);
    for (int64_t j = 0; j < chunk->length(); j++, compared++) {
      if (expected[compared].size() == (size_t)1 &&
          expected[compared][0] == inline_null_array_value<T>()) {
        ASSERT_TRUE(chunk->IsNull(j));
      } else {
        ASSERT_TRUE(chunk->IsValid(j));
        auto list_elem = list_array->value_slice(j);
        const T* values = std::static_pointer_cast<ArrowColType>(list_elem)->raw_values();
        ASSERT_EQ((size_t)list_elem->length(), expected[compared].size());
        for (int64_t k = 0; k < list_elem->length(); k++) {
          if (expected[compared][k] == inline_null_value<T>()) {
            ASSERT_TRUE(list_elem->IsNull(k));
          } else {
            ASSERT_TRUE(list_elem->IsValid(k));
            if constexpr (std::is_floating_point_v<T>) {
              ASSERT_NEAR(expected[compared][k], values[k], 0.001);
            } else {
              ASSERT_EQ(expected[compared][k], values[k]);
            }
          }
        }
      }
    }
  }

  ASSERT_EQ(compared, expected.size());
}

template <typename T>
void compare_arrow_array_impl(const std::vector<std::vector<T>>& expected,
                              const std::shared_ptr<arrow::ChunkedArray>& actual) {
  ASSERT_EQ(static_cast<size_t>(actual->length()), expected.size());
  if (actual->type()->id() == arrow::Type::LIST) {
    compare_arrow_array_list_impl<T, arrow::ListArray>(expected, actual);
  } else if (actual->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
    compare_arrow_array_list_impl<T, arrow::FixedSizeListArray>(expected, actual);
  } else {
    CHECK(false);
  }
}

template <typename TYPE>
void compare_arrow_array(const std::vector<TYPE>& expected,
                         const std::shared_ptr<arrow::ChunkedArray>& actual) {
  compare_arrow_array_impl(expected, actual);
}

template <>
void compare_arrow_array<int64_t>(const std::vector<int64_t>& expected,
                                  const std::shared_ptr<arrow::ChunkedArray>& actual) {
  if (actual->type()->id() == arrow::Type::DECIMAL) {
    compare_arrow_array_decimal_impl(expected, actual);
  } else if (actual->type()->id() == arrow::Type::DATE64) {
    compare_arrow_array_date_impl<arrow::Date64Array>(expected, actual);
  } else if (actual->type()->id() == arrow::Type::TIME32) {
    compare_arrow_array_date_impl<arrow::Time32Array>(expected, actual);
  } else if (actual->type()->id() == arrow::Type::TIME64) {
    compare_arrow_array_date_impl<arrow::Time64Array>(expected, actual);
  } else if (actual->type()->id() == arrow::Type::TIMESTAMP) {
    compare_arrow_array_date_impl<arrow::TimestampArray>(expected, actual);
  } else {
    compare_arrow_array_impl(expected, actual);
  }
}

void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at, int col_idx) {}

template <typename T, typename... Ts>
void compare_arrow_table_impl(std::shared_ptr<arrow::Table> at,
                              int col_idx,
                              const std::vector<T>& expected,
                              const std::vector<Ts>... expected_rem) {
  ASSERT_LT(static_cast<size_t>(col_idx), at->columns().size());
  auto col = at->column(col_idx);
  compare_arrow_array(expected, at->column(col_idx));
  compare_arrow_table_impl(at, col_idx + 1, expected_rem...);
}

template <typename... Ts>
void compare_arrow_table(std::shared_ptr<arrow::Table> at,
                         const std::vector<Ts>&... expected) {
  ASSERT_EQ(at->columns().size(), sizeof...(Ts));
  compare_arrow_table_impl(at, 0, expected...);
}

std::shared_ptr<arrow::Table> toArrow(const ExecutionResult& res) {
  return res.getToken()->toArrow();
}

template <typename... Ts>
void compare_res_data(const ExecutionResult& res, const std::vector<Ts>&... expected) {
  compare_arrow_table(toArrow(res), expected...);
}

void compareArrowTables(std::shared_ptr<arrow::Table> expected,
                        std::shared_ptr<arrow::Table> actual) {
  ASSERT_EQ(expected->num_columns(), actual->num_columns());
  ASSERT_EQ(expected->num_rows(), actual->num_rows());
  for (int64_t col_idx = 0; col_idx < expected->num_columns(); ++col_idx) {
    auto expected_col = expected->column(col_idx);
    auto actual_col = actual->column(col_idx);
    ASSERT_TRUE(expected_col->type()->Equals(actual_col->type()));
    for (int64_t row_idx = 0; row_idx < expected->num_rows(); ++row_idx) {
      auto expected_val = expected_col->GetScalar(row_idx);
      auto actual_val = actual_col->GetScalar(row_idx);

      ASSERT_TRUE(expected_val.ValueOrDie()->ApproxEquals(*actual_val.ValueOrDie()));
    }
  }
}

}  // namespace ArrowTestHelpers
