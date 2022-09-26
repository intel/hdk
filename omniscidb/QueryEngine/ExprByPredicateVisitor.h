/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ScalarExprVisitor.h"

class ExprByPredicateVisitor : public ScalarExprVisitor<void*> {
 public:
  ExprByPredicateVisitor(bool (*pred)(const hdk::ir::Expr*)) : pred_(pred) {}

  void* visit(const hdk::ir::Expr* expr) const override {
    if (pred_(expr)) {
      if (std::find(expr_list_.begin(), expr_list_.end(), expr) == expr_list_.end()) {
        expr_list_.push_back(expr);
      }
      return nullptr;
    }
    return ScalarExprVisitor::visit(expr);
  }

  const std::list<const hdk::ir::Expr*>& result() const { return expr_list_; }

  static std::list<const hdk::ir::Expr*> collect(const hdk::ir::Expr* expr,
                                                 bool (*pred)(const hdk::ir::Expr*)) {
    ExprByPredicateVisitor visitor(pred);
    visitor.visit(expr);
    return std::move(visitor.expr_list_);
  }

  static std::list<const hdk::ir::Expr*> collect(const hdk::ir::ExprPtr& expr,
                                                 bool (*pred)(const hdk::ir::Expr*)) {
    return collect(expr.get(), pred);
  }

 protected:
  bool (*pred_)(const hdk::ir::Expr*);
  mutable std::list<const hdk::ir::Expr*> expr_list_;
};
