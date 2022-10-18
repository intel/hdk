/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HDK.h"

#include <iostream>

#include "Shared/ArrowUtil.h"

#include <arrow/api.h>

int main(void) {
  std::cout << "Hello, world" << std::endl;

  // Load some data
  auto col_x = std::make_shared<arrow::Field>("x", arrow::int32());
  auto col_y = std::make_shared<arrow::Field>("y", arrow::float64());
  auto schema = arrow::schema({col_x, col_y});

  std::shared_ptr<arrow::Array> x_array;
  arrow::NumericBuilder<arrow::Int32Type> int32_builder;
  ARROW_THROW_NOT_OK(int32_builder.AppendValues({7, 7, 7, 7, 7, 8, 8, 8, 8, 9}));
  ARROW_THROW_NOT_OK(int32_builder.Finish(&x_array));

  std::shared_ptr<arrow::Array> y_array;
  arrow::NumericBuilder<arrow::FloatType> float64_builder;
  ARROW_THROW_NOT_OK(
      float64_builder.AppendValues({0.0, 1.1, 1.1, 1.1, 1.1, 2.2, 2.2, 3.3, 4.4, 5.5}));
  ARROW_THROW_NOT_OK(float64_builder.Finish(&y_array));

  auto table = arrow::Table::Make(schema, {x_array, y_array});

  HDK hdk;
  hdk.read(table, "test");
}
