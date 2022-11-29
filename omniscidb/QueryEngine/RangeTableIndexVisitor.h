/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <set>

#include "IR/ExprCollector.h"

class MaxRangeTableIndexCollector
    : public hdk::ir::ExprCollector<int, MaxRangeTableIndexCollector> {
 public:
  MaxRangeTableIndexCollector() { result_ = 0; }

 protected:
  void visitColumnVar(const hdk::ir::ColumnVar* column) override {
    result_ = std::max(result_, column->rteIdx());
  }
};

class AllRangeTableIndexCollector
    : public hdk::ir::ExprCollector<std::set<int>, AllRangeTableIndexCollector> {
 protected:
  void visitColumnVar(const hdk::ir::ColumnVar* column) override {
    result_.insert(column->rteIdx());
  }
};
