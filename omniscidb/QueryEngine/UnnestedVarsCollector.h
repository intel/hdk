/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "IR/ExprCollector.h"

class UnnestedVarsCollector
    : public hdk::ir::ExprCollector<std::vector<const hdk::ir::Expr*>,
                                    UnnestedVarsCollector> {
 public:
  void visitUOper(const hdk::ir::UOper* uoper) override {
    if (uoper->opType() == hdk::ir::OpType::kUnnest) {
      if (!uoper->operand()->is<hdk::ir::ColumnVar>()) {
        throw std::runtime_error(
            "Unexpected UNNEST context. Only column references are currently supported.");
      }
      auto var = uoper->operand()->as<hdk::ir::ColumnVar>();
      if (var->rteIdx() != 0) {
        throw std::runtime_error("UNNEST is not yet supported in joins.");
      }
      if (!visited_.count(*var->columnInfo())) {
        result_.push_back(uoper->operand());
        visited_.insert(*var->columnInfo());
      }
    }
  }

 private:
  std::unordered_set<ColumnRef> visited_;
};
