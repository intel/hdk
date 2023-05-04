/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRewrite.h"
#include "RelAlgExecutionUnit.h"

#include "IR/ExprRewriter.h"
#include "IR/Node.h"
#include "SchemaMgr/SchemaProvider.h"
#include "Shared/sqldefs.h"

#include <memory>
#include <unordered_map>
#include <vector>

class Executor;

namespace hdk {

class WorkUnitBuilder {
 public:
  WorkUnitBuilder(const ir::Node* root,
                  const ir::QueryDag* dag,
                  Executor* executor,
                  SchemaProviderPtr schema_provider,
                  DataProvider* data_provider,
                  TemporaryTables& temporary_tables,
                  const ExecutionOptions& eo,
                  const CompilationOptions& co,
                  time_t now,
                  bool just_explain,
                  bool allow_speculative_sort);

  RelAlgExecutionUnit exeUnit() const;
  size_t maxGroupsBufferEntryGuess() const { return max_groups_buffer_entry_guess_; }
  std::unique_ptr<QueryRewriter> releaseQueryRewriter() {
    return std::move(query_rewriter_);
  }
  std::vector<size_t> releaseInputPermutation() { return std::move(input_permutation_); }
  std::vector<size_t> releaseLeftDeepJoinInputSizes() {
    return std::move(left_deep_join_input_sizes_);
  }
  ir::ExprPtrVector&& releaseTargetExprsOwned() { return std::move(target_exprs_[0]); }

  const std::unordered_map<const ir::Node*, int> nestLevels() const {
    return input_nest_levels_;
  }
  const std::vector<JoinType> joinTypes() const { return join_types_; }
  std::optional<unsigned> leftDeepTreeId() const { return left_deep_tree_id_; }

  bool isAgg() const { return is_agg_; }

 protected:
  void build();
  void process(const ir::Node* node);
  void processAggregate(const ir::Aggregate* agg);
  void processProject(const ir::Project* proj);
  void processFilter(const ir::Filter* filter);
  void processSort(const ir::Sort* sort);
  void processUnion(const ir::LogicalUnion* logical_union);
  void processJoin(const ir::Join* join);
  std::list<hdk::ir::ExprPtr> makeJoinQuals(const hdk::ir::Expr* join_condition);
  void reorderTables();
  void computeSimpleQuals();
  int assignNestLevels(const ir::Node* node, int start_idx = 0);
  void computeJoinTypes(const ir::Node* node, bool allow_join = true);
  void computeInputDescs();
  void computeInputColDescs();

  class InputRewriter : public ir::ExprRewriter {
   public:
    InputRewriter() {}

    void addReplacement(const ir::ColumnRef* col_ref, ir::ExprPtr target) {
      addReplacement(col_ref->node(), col_ref->index(), target);
    }

    void addReplacement(const ir::Node* node, unsigned index, ir::ExprPtr target) {
      replacements_.emplace(std::make_pair(node, index), target);
    }

    ir::ExprPtr visitColumnRef(const ir::ColumnRef* col_ref) override {
      if (replacements_.count(std::make_pair(col_ref->node(), col_ref->index()))) {
        return replacements_.at(std::make_pair(col_ref->node(), col_ref->index()));
      }
      return ExprRewriter::visitColumnRef(col_ref);
    }

   private:
    using InputReplacements =
        std::unordered_map<std::pair<const ir::Node*, unsigned>,
                           ir::ExprPtr,
                           boost::hash<std::pair<const ir::Node*, unsigned>>>;

    InputReplacements replacements_;
  };

  const ir::Node* root_;
  const ir::QueryDag* dag_;
  Executor* executor_;
  SchemaProviderPtr schema_provider_;
  DataProvider* data_provider_;
  TemporaryTables& temporary_tables_;
  const ExecutionOptions& eo_;
  const CompilationOptions& co_;
  time_t now_;
  bool just_explain_;
  bool allow_speculative_sort_;
  // Stores nest level for each input (leaf) node.
  std::unordered_map<const ir::Node*, int> input_nest_levels_;
  // Stores nest level for each node to process.
  std::unordered_map<const ir::Node*, int> all_nest_levels_;
  // Stores order for each UNION ALL input node for proper
  // result ordering.
  std::unordered_map<TableRef, int> union_order_;
  std::vector<JoinType> join_types_;
  bool is_agg_ = false;

  std::vector<InputDescriptor> input_descs_;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs_;
  std::list<hdk::ir::ExprPtr> simple_quals_;
  std::list<hdk::ir::ExprPtr> quals_;
  JoinQualsPerNestingLevel join_quals_;
  std::list<hdk::ir::ExprPtr> groupby_exprs_;
  std::vector<ir::ExprPtrVector> target_exprs_;
  std::shared_ptr<hdk::ir::Estimator> estimator_;
  SortInfo sort_info_ = {{}, SortAlgorithm::Default, 0, 0};
  size_t scan_limit_ = 0;
  QueryPlan query_plan_dag_ = EMPTY_QUERY_PLAN;
  HashTableBuildDagMap hash_table_build_plan_dag_;
  TableIdToNodeMap table_id_to_node_map_;
  std::optional<bool> union_all_;

  std::optional<unsigned> left_deep_tree_id_;
  size_t max_groups_buffer_entry_guess_;
  std::unique_ptr<QueryRewriter> query_rewriter_;
  std::vector<size_t> input_permutation_;
  std::vector<size_t> left_deep_join_input_sizes_;

  InputRewriter input_rewriter_;
};

}  // namespace hdk
