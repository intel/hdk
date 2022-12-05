/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryExecutionSequence.h"
#include "ScalarExprVisitor.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>

#include <unordered_set>

namespace hdk {

namespace {

using DAG = boost::
    adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, const ir::Node*>;
using Vertex = DAG::vertex_descriptor;

class CoalesceSecondaryProjectVisitor : public ScalarExprVisitor<bool> {
 public:
  bool visitColumnRef(const hdk::ir::ColumnRef* col_ref) const final {
    // The top level expression is checked before we apply the visitor. If we get
    // here, this input expr is a child of another expr.
    if (auto agg_node = dynamic_cast<const hdk::ir::Aggregate*>(col_ref->node())) {
      return col_ref->index() == 0 && agg_node->getGroupByCount() > 0;
    }
    return false;
  }

  bool visitConstant(const hdk::ir::Constant*) const final { return false; }

  bool visitInSubquery(const hdk::ir::InSubquery*) const final { return false; }

  bool visitScalarSubquery(const hdk::ir::ScalarSubquery*) const final { return false; }

 protected:
  bool aggregateResult(const bool& aggregate, const bool& next_result) const final {
    return aggregate && next_result;
  }

  bool defaultResult() const final { return true; }
};

class QueryExecutionSequenceImpl {
 public:
  static std::vector<const ir::Node*> buildSteps(const ir::Node* root, ConfigPtr config) {
    QueryExecutionSequenceImpl impl(root, config);
    return std::move(impl.execution_steps_);
  }

 protected:
  QueryExecutionSequenceImpl(const ir::Node* root, ConfigPtr config) : config_(config) {
    buildDagEdges(root);
    for (auto& pr : node_to_vertex_) {
      graph_[pr.second] = pr.first;
    }
    execution_points_.insert(root);
    findExecutionPoints(root);
    mergeExecutionPointsWithSimpleProject();
    // TODO: do not merge sort with other executions steps unless we really can
    // intergrate sort or its part to other execution modules.
    mergeExecutionPointsWithSort();
    removeScanExecutionPoints();
    rebuildDag();
    buildSteps();
  }

  void buildDagEdges(const ir::Node* node) {
    // Already visited node
    if (node_to_vertex_.count(node)) {
      return;
    }

    size_t vertex = node_to_vertex_.size();
    node_to_vertex_.emplace(node, vertex);

    for (size_t i = 0; i < node->inputCount(); ++i) {
      const auto input = node->getInput(i);
      buildDagEdges(input);
      boost::add_edge(vertex, node_to_vertex_[input], graph_);
    }
  }

  bool requireReduction(const ir::Node* node) {
    return node->is<ir::Sort>() || node->is<ir::Aggregate>();
  }

  void findExecutionPoints(const ir::Node* node, bool execute_join = false) {
    if (node->is<ir::Scan>()) {
      return;
    }

    // Reduction is always a separate execution step requiring
    // materialized input.
    if (requireReduction(node)) {
      execution_points_.insert(node);
    }

    // For now, materialize all nodes having more than one user
    // to avoid execution of the same operation multiple times.
    // Optimizers are free to duplicate sub-graphs if it's better
    // to not materizlie such nodes.
    if (boost::in_degree(node_to_vertex_[node], graph_) > 1) {
      execution_points_.insert(node);
    }

    // Due to the current window functions support limitations,
    // both window function result and its input have to be materialized.
    if (node->is<ir::Project>() && hasWindowFunctionExpr(node->as<ir::Project>())) {
      execution_points_.insert(node);
      execution_points_.insert(node->getInput(0));
    }

    // Currently, we cannot merge union code into any other execution
    // module. Therefore, mark it and all its inputs as execution points.
    // TODO: UNION ALL should be able to be merged into other execution
    // modules.
    if (node->is<ir::LogicalUnion>()) {
      execution_points_.insert(node);
      for (size_t i = 0; i < node->inputCount(); ++i) {
        execution_points_.insert(node->getInput(i));
      }
    }

    // Currently, we do not merge TableFunction with anything.
    if (node->is<ir::TableFunction>()) {
      execution_points_.insert(node);
      for (size_t i = 0; i < node->inputCount(); ++i) {
        execution_points_.insert(node->getInput(i));
      }
    }

    // LogicalValues are processed separately.
    if (node->is<ir::LogicalValues>()) {
      execution_points_.insert(node);
    }

    bool is_join = node->is<ir::Join>() || node->is<ir::LeftDeepInnerJoin>();
    if (execute_join && is_join) {
      execution_points_.insert(node);
    }

    for (size_t i = 0; i < node->inputCount(); ++i) {
      if (is_join && i > 0) {
        bool left_join =
            node->is<ir::Join>()
                ? node->as<ir::Join>()->getJoinType() == JoinType::LEFT
                : node->as<ir::LeftDeepInnerJoin>()->getJoinType(i) == JoinType::LEFT;
        // In case of left join, all quals of inner input should be applied
        // before join hash table build. Applying it after the join might
        // filter out additional rows because it shouldn't be applied to
        // rows which don't meet join qual. To prevent possible issues,
        // we simply always execute inner inputs of left joins.
        if (left_join || config_->exec.materialize_inner_join_tables) {
          execution_points_.insert(node->getInput(i));
        }
      }
      // We don't support bushy joins, so allow joins only for outer input of
      // other joins. We also should propagate received execute_join flag but
      // only if the current node is not marked as an execution point already.
      bool exec_joins =
          (is_join && i > 0) || (execute_join && !execution_points_.count(node));
      findExecutionPoints(node->getInput(i), exec_joins);
    }
  }

  void mergeExecutionPointsWithSimpleProject() {
    std::vector<const ir::Node*> simple_projects;
    for (auto input : execution_points_) {
      if (boost::in_degree(node_to_vertex_[input], graph_) > 1) {
        continue;
      }

      // Only aggregations and joins can now be merged with a following
      // projection.
      // TODO: Use recursive search to cover multiple consequent projections?
      if (!input->is<ir::Aggregate>() && !input->is<ir::Join>() &&
          !input->is<ir::LeftDeepInnerJoin>()) {
        continue;
      }

      auto [start, end] = boost::in_edges(node_to_vertex_[input], graph_);
      auto node = graph_[start->m_source];

      if (node->is<ir::Project>() && !hasWindowFunctionExpr(node->as<ir::Project>())) {
        // In case of aggregation we allow only 'simple' projections which
        // don't have complex expressions referencing aggregate exprs.
        bool is_simple = true;
        for (auto& expr : node->as<ir::Project>()->getExprs()) {
          if (!expr->is<ir::ColumnRef>() && input->is<ir::Aggregate>()) {
            CoalesceSecondaryProjectVisitor visitor;
            if (!visitor.visit(expr.get())) {
              is_simple = false;
              break;
            }
          }
        }
        if (is_simple) {
          simple_projects.push_back(node);
        }
      }
    }
    for (auto node : simple_projects) {
      execution_points_.insert(node);
      execution_points_.erase(node->getInput(0));
    }
  }

  void mergeExecutionPointsWithSort() {
    std::vector<const ir::Node*> to_merge;
    for (auto node : execution_points_) {
      if (node->is<ir::Sort>()) {
        auto input = node->getInput(0);
        if (boost::in_degree(node_to_vertex_[input], graph_) == 1) {
          to_merge.push_back(input);
        }
      }
    }
    for (auto node : to_merge) {
      execution_points_.erase(node);
    }
  }

  void removeScanExecutionPoints() {
    std::vector<const ir::Node*> to_remove;
    for (auto node : execution_points_) {
      if (node->is<ir::Scan>()) {
        to_remove.push_back(node);
      }
    }
    for (auto node : to_remove) {
      execution_points_.erase(node);
    }
  }

  void buildExecutionEdges(const ir::Node* orig_source, const ir::Node* intermediate) {
    for (size_t i = 0; i < intermediate->inputCount(); ++i) {
      auto input = intermediate->getInput(i);
      if (execution_points_.count(input)) {
        boost::add_edge(node_to_vertex_[orig_source], node_to_vertex_[input], graph_);
      } else {
        buildExecutionEdges(orig_source, input);
      }
    }
  }

  void rebuildDag() {
    // Build a new graph with execution points only.
    graph_ = DAG(execution_points_.size());
    node_to_vertex_.clear();
    for (auto node : execution_points_) {
      graph_[node_to_vertex_.size()] = node;
      node_to_vertex_.emplace(node, node_to_vertex_.size());
    }
    for (auto node : execution_points_) {
      buildExecutionEdges(node, node);
    }
  }

  void buildSteps() {
    std::vector<size_t> vertexes;
    vertexes.reserve(execution_points_.size());
    boost::topological_sort(graph_, std::back_inserter(vertexes));

    execution_steps_.reserve(vertexes.size());
    for (auto vertex : vertexes) {
      execution_steps_.push_back(graph_[vertex]);
    }
  }

  DAG graph_;
  std::unordered_map<const hdk::ir::Node*, size_t> node_to_vertex_;
  std::unordered_set<const ir::Node*> execution_points_;
  std::vector<const ir::Node*> execution_steps_;
  ConfigPtr config_;
};

}  // namespace

QueryExecutionSequence::QueryExecutionSequence(const ir::Node* root, ConfigPtr config) {
  steps_ = QueryExecutionSequenceImpl::buildSteps(root, config);
}

}  // namespace hdk
