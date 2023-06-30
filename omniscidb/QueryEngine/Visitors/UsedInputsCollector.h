/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "IR/ExprCollector.h"

#include <unordered_set>

struct ColumnVarHash {
  size_t operator()(const hdk::ir::ColumnVar& col_var) const { return col_var.hash(); }
};

using ColumnVarSet = std::unordered_set<hdk::ir::ColumnVar, ColumnVarHash>;

class UsedInputsCollector
    : public hdk::ir::ExprCollector<ColumnVarSet, UsedInputsCollector> {
 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override { CHECK(false); }

  void visitColumnVar(const hdk::ir::ColumnVar* col_var) override {
    result_.insert(*col_var);
  }
};
