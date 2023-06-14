/*
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExprVisitor.h"

namespace hdk::ir {

template <typename ResultType, typename CollectorType>
class ExprCollector : public ExprVisitor<void> {
 public:
  template <typename... Ts>
  static ResultType collect(const Expr* expr, Ts&&... args) {
    CollectorType collector(std::forward<Ts>(args)...);
    collector.visit(expr);
    return std::move(collector.result_);
  }

  template <typename... Ts>
  static ResultType collect(const ExprPtr& expr, Ts&&... args) {
    return collect(expr.get(), std::forward<Ts>(args)...);
  }

  template <typename... Ts>
  static ResultType collect(const ExprPtrVector& exprs, Ts&&... args) {
    CollectorType collector(std::forward<Ts>(args)...);
    for (auto& expr : exprs) {
      collector.visit(expr.get());
    }
    return std::move(collector.result_);
  }

  ResultType& result() { return result_; }
  const ResultType& result() const { return result_; }

 protected:
  using BaseClass = ExprCollector<ResultType, CollectorType>;

  ResultType result_;
};

}  // namespace hdk::ir
