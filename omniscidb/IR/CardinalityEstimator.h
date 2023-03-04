/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Expr.h"

namespace hdk::ir {

/*
 * @type  Estimator
 * @brief Infrastructure to define estimators which take an expression tuple, are called
 * for every row and need a buffer to track state.
 */
class Estimator : public Expr {
 public:
  Estimator() : Expr(Context::defaultCtx().int32(false)){};

  // The tuple argument received by the estimator for every row.
  virtual const ExprPtrList& getArgument() const = 0;

  // The size of the working buffer used by the estimator.
  virtual size_t getBufferSize() const = 0;

  // The name for the estimator runtime function which is called for every row.
  // The runtime function will receive four arguments:
  //   uint8_t* the pointer to the beginning of the estimator buffer
  //   uint32_t the size of the estimator buffer, in bytes
  //   uint8_t* the concatenated bytes for the argument tuple
  //   uint32_t the size of the argument tuple, in bytes
  virtual std::string getRuntimeFunctionName() const = 0;

  ExprPtr withType(const Type* new_type) const override {
    CHECK(false);
    return nullptr;
  }

  bool operator==(const Expr& rhs) const override {
    CHECK(false);
    return false;
  }

  std::string toString() const override {
    CHECK(false);
    return "";
  }
};

/*
 * @type  NDVEstimator
 * @brief Provides an estimate for the number of distinct tuples. Not a real
 *        expression, it's only used in execution units synthesized
 *        for the cardinality estimation before running an user-provided query.
 */
class NDVEstimator : public Estimator {
 public:
  NDVEstimator(const hdk::ir::ExprPtrList& expr_tuple, size_t buffer_size_multiplier)
      : expr_tuple_(expr_tuple), buffer_size_multiplier_(buffer_size_multiplier) {}

  const hdk::ir::ExprPtrList& getArgument() const override { return expr_tuple_; }

  size_t getBufferSize() const override { return 1024 * 1024 * buffer_size_multiplier_; }

  std::string getRuntimeFunctionName() const override {
    return "linear_probabilistic_count";
  }

 private:
  const hdk::ir::ExprPtrList expr_tuple_;
  size_t buffer_size_multiplier_;
};

}  // namespace hdk::ir
