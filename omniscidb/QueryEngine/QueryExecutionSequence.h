/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "IR/Node.h"

namespace hdk {

class QueryExecutionSequence {
 public:
  QueryExecutionSequence(const ir::Node* root, ConfigPtr config);

  const std::vector<const ir::Node*>& steps() const { return steps_; }
  size_t size() const { return steps_.size(); }
  const ir::Node* step(size_t idx) const { return steps_[idx]; }

 protected:
  std::vector<const ir::Node*> steps_;
};

}  // namespace hdk