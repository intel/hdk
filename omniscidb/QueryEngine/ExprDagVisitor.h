/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelAlgDagBuilder.h"
#include "ScalarExprVisitor.h"

// TODO: this visitor similar to RelRexDagVisitor can visit the same node
// multiple times.
class ExprDagVisitor : public ScalarExprVisitor<void*> {
 public:
  using ScalarExprVisitor::visit;

  void visit(const hdk::ir::Node* node) {
    if (auto agg = dynamic_cast<const hdk::ir::Aggregate*>(node)) {
      visitAggregate(agg);
    } else if (auto compound = dynamic_cast<const hdk::ir::Compound*>(node)) {
      visitCompound(compound);
    } else if (auto filter = dynamic_cast<const hdk::ir::Filter*>(node)) {
      visitFilter(filter);
    } else if (auto join = dynamic_cast<const hdk::ir::Join*>(node)) {
      visitJoin(join);
    } else if (auto deep_join = dynamic_cast<const hdk::ir::LeftDeepInnerJoin*>(node)) {
      visitLeftDeepInnerJoin(deep_join);
    } else if (auto logical_union = dynamic_cast<const hdk::ir::LogicalUnion*>(node)) {
      visitLogicalUnion(logical_union);
    } else if (auto values = dynamic_cast<const hdk::ir::LogicalValues*>(node)) {
      visitLogicalValues(values);
    } else if (auto project = dynamic_cast<const hdk::ir::Project*>(node)) {
      visitProject(project);
    } else if (auto scan = dynamic_cast<const hdk::ir::Scan*>(node)) {
      visitScan(scan);
    } else if (auto sort = dynamic_cast<const hdk::ir::Sort*>(node)) {
      visitSort(sort);
    } else if (auto table_fn = dynamic_cast<const hdk::ir::TableFunction*>(node)) {
      visitTableFunction(table_fn);
    } else if (auto translated_join =
                   dynamic_cast<const hdk::ir::TranslatedJoin*>(node)) {
      visitTranslatedJoin(translated_join);
    } else {
      LOG(FATAL) << "Unsupported node type: " << node->toString();
    }
    for (size_t i = 0; i < node->inputCount(); ++i) {
      visit(node->getInput(i));
    }
  }

 protected:
  virtual void visitAggregate(const hdk::ir::Aggregate* agg) {
    for (auto& expr : agg->getAggs()) {
      visit(expr.get());
    }
  }

  virtual void visitCompound(const hdk::ir::Compound* compound) {
    if (compound->getFilter()) {
      visit(compound->getFilter().get());
    }
    for (auto& expr : compound->getGroupByExprs()) {
      visit(expr.get());
    }
    for (auto& expr : compound->getExprs()) {
      visit(expr.get());
    }
  }

  virtual void visitFilter(const hdk::ir::Filter* filter) {
    visit(filter->getConditionExpr());
  }

  virtual void visitJoin(const hdk::ir::Join* join) { visit(join->getCondition()); }

  virtual void visitLeftDeepInnerJoin(const hdk::ir::LeftDeepInnerJoin* join) {
    visit(join->getInnerCondition());
    for (size_t level = 1; level < join->inputCount(); ++level) {
      if (auto* outer_condition = join->getOuterCondition(level)) {
        visit(outer_condition);
      }
    }
  }

  virtual void visitLogicalUnion(const hdk::ir::LogicalUnion*) {}

  virtual void visitLogicalValues(const hdk::ir::LogicalValues* logical_values) {
    for (size_t row_idx = 0; row_idx < logical_values->getNumRows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < logical_values->getRowsSize(); ++col_idx) {
        visit(logical_values->getValue(row_idx, col_idx));
      }
    }
  }

  virtual void visitProject(const hdk::ir::Project* proj) {
    for (auto& expr : proj->getExprs()) {
      visit(expr.get());
    }
  }

  virtual void visitScan(const hdk::ir::Scan*) {}
  virtual void visitSort(const hdk::ir::Sort*) {}

  virtual void visitTableFunction(const hdk::ir::TableFunction* table_function) {
    for (size_t i = 0; i < table_function->getTableFuncInputsSize(); ++i) {
      visit(table_function->getTableFuncInputExprAt(i));
    }
  }

  virtual void visitTranslatedJoin(const hdk::ir::TranslatedJoin* translated_join) {
    visit(translated_join->getLHS());
    visit(translated_join->getRHS());
    for (auto& expr : translated_join->getFilterCond()) {
      visit(expr.get());
    }
    if (auto* outer_join_condition = translated_join->getOuterJoinCond()) {
      visit(outer_join_condition);
    }
  }

  void* visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    const_cast<ExprDagVisitor*>(this)->visit(subquery->node());
    return nullptr;
  }

  void* visitInSubquery(const hdk::ir::InSubquery* subquery) const override {
    const_cast<ExprDagVisitor*>(this)->visit(subquery->node());
    return nullptr;
  }
};
