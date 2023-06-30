/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelAlgExecutionUnit.h"
#include "Visitors/UsedInputsCollector.h"

void RelAlgExecutionUnit::calcInputColDescs(const SchemaProvider* schema_provider) {
  // Scan all currently used expressions to determine used columns.
  UsedInputsCollector collector;
  for (auto& expr : simple_quals) {
    collector.visit(expr.get());
  }
  for (auto& expr : quals) {
    collector.visit(expr.get());
  }
  for (auto& expr : groupby_exprs) {
    if (expr) {
      collector.visit(expr.get());
    }
  }
  for (auto& join_qual : join_quals) {
    for (auto& expr : join_qual.quals) {
      collector.visit(expr.get());
    }
  }

  for (auto& expr : target_exprs) {
    collector.visit(expr);
  }

  if (partition_offsets_col) {
    collector.visit(partition_offsets_col.get());
  }

  std::vector<std::shared_ptr<const InputColDescriptor>> col_descs;
  for (auto& col_var : collector.result()) {
    col_descs.push_back(std::make_shared<const InputColDescriptor>(col_var.columnInfo(),
                                                                   col_var.rteIdx()));
  }

  // For UNION we only have column variables for a single table used
  // in target expressions but should mark all columns as used.
  if (union_all && !col_descs.empty()) {
    CHECK_EQ(col_descs.front()->getNestLevel(), 0);
    CHECK_EQ(input_descs.size(), (size_t)2);
    TableRef processed_table_ref(col_descs.front()->getDatabaseId(),
                                 col_descs.front()->getTableId());
    for (auto tdesc : input_descs) {
      if (tdesc.getTableRef() != processed_table_ref) {
        auto columns = schema_provider->listColumns(tdesc.getTableRef());
        for (auto& col_info : columns) {
          if (!col_info->is_rowid) {
            col_descs.push_back(std::make_shared<InputColDescriptor>(col_info, 0));
          }
        }
      }
    }
  }

  std::sort(
      col_descs.begin(),
      col_descs.end(),
      [](std::shared_ptr<const InputColDescriptor> const& lhs,
         std::shared_ptr<const InputColDescriptor> const& rhs) {
        return std::make_tuple(lhs->getNestLevel(), lhs->getColId(), lhs->getTableId()) <
               std::make_tuple(rhs->getNestLevel(), rhs->getColId(), rhs->getTableId());
      });

  input_col_descs.clear();
  input_col_descs.insert(input_col_descs.end(), col_descs.begin(), col_descs.end());
}
