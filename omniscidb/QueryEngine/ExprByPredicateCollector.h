/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "IR/ExprCollector.h"

class ExprByPredicateCollector
    : public hdk::ir::ExprCollector<std::list<const hdk::ir::Expr*>,
                                    ExprByPredicateCollector> {
 public:
  ExprByPredicateCollector(bool (*pred)(const hdk::ir::Expr*)) : pred_(pred) {}

  void visit(const hdk::ir::Expr* expr) override {
    if (pred_(expr)) {
      if (std::find(result_.begin(), result_.end(), expr) == result_.end()) {
        result_.push_back(expr);
      }
    } else {
      ExprVisitor::visit(expr);
    }
  }

 protected:
  bool (*pred_)(const hdk::ir::Expr*);
};
