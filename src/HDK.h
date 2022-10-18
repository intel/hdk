/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"  // ExecutionResult

#include <arrow/api.h>

struct Internal;

class HDK {
 public:
  HDK();

  ~HDK();

  void read(std::shared_ptr<arrow::Table>& table, const std::string& table_name);

  ExecutionResult query(const std::string& sql, const bool is_explain = false);

  static HDK init();

 private:
  Internal* internal_{nullptr};
};
