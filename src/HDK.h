/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"  // ExecutionResult

class HDK {
 public:
  void read();

  ExecutionResult query(const std::string& sql, const bool is_explain = false);

  static HDK init();

 private:
  HDK() {}
};
