/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExprCollector.h"

namespace hdk::ir {

class UnnestDetector : public ExprCollector<bool, UnnestDetector> {
 public:
  UnnestDetector() { result_ = false; }

 protected:
  void visitUOper(const hdk::ir::UOper* uoper) override {
    if (uoper->opType() == OpType::kUnnest) {
      result_ = true;
      return;
    }
    BaseClass::visitUOper(uoper);
  }
};

}  // namespace hdk::ir
