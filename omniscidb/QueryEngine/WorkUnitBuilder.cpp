/**
 * Copyright (C) 2022 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "WorkUnitBuilder.h"
// TODO: move used functions here
#include "RelAlgExecutor.h"

#include "IR/Node.h"
#include "QueryEngine/EquiJoinCondition.h"
#include "QueryEngine/ExpressionRewrite.h"
#include "QueryEngine/FromTableReordering.h"
#include "QueryEngine/QueryPlanDagExtractor.h"

namespace hdk {

namespace {

struct ColumnVarHash {
  size_t operator()(const ir::ColumnVar& col_var) const { return col_var.hash(); }
};

using ColumnVarSet = std::unordered_set<ir::ColumnVar, ColumnVarHash>;

class UsedInputsCollector : public ir::ExprCollector<ColumnVarSet, UsedInputsCollector> {
 protected:
  void visitColumnRef(const ir::ColumnRef* col_ref) override { CHECK(false); }

  void visitColumnVar(const ir::ColumnVar* col_var) override { result_.insert(*col_var); }
};

class NestLevelRewriter : public ir::ExprRewriter {
 public:
  NestLevelRewriter(const std::vector<InputDescriptor>& input_descs) {
    for (auto& desc : input_descs) {
      table_to_rte_idx_[std::make_pair(desc.getDatabaseId(), desc.getTableId())] =
          desc.getNestLevel();
    }
  }

 protected:
  ir::ExprPtr visitColumnRef(const ir::ColumnRef* col_ref) override {
    CHECK(false);
    return nullptr;
  }

  ir::ExprPtr visitColumnVar(const ir::ColumnVar* col_var) override {
    auto rte_idx =
        table_to_rte_idx_.at(std::make_pair(col_var->dbId(), col_var->tableId()));
    if (col_var->rteIdx() != rte_idx) {
      return ir::makeExpr<ir::ColumnVar>(col_var->columnInfo(), rte_idx);
    }
    return defaultResult(col_var);
  }

  ir::ExprPtr visitVar(const ir::Var* var) override {
    auto rte_idx = table_to_rte_idx_.at(std::make_pair(var->dbId(), var->tableId()));
    if (var->rteIdx() != rte_idx) {
      return ir::makeExpr<ir::Var>(
          var->columnInfo(), rte_idx, var->whichRow(), var->varNo());
    }
    return defaultResult(var);
  }

  std::unordered_map<std::pair<int, int>, int> table_to_rte_idx_;
};

}  // namespace

WorkUnitBuilder::WorkUnitBuilder(const ir::Node* root,
                                 const ir::QueryDag* dag,
                                 Executor* executor,
                                 SchemaProviderPtr schema_provider,
                                 TemporaryTables& temporary_tables,
                                 const ExecutionOptions& eo,
                                 const CompilationOptions& co,
                                 time_t now,
                                 bool just_explain,
                                 bool allow_speculative_sort)
    : root_(root)
    , dag_(dag)
    , executor_(executor)
    , schema_provider_(schema_provider)
    , temporary_tables_(temporary_tables)
    , eo_(eo)
    , co_(co)
    , now_(now)
    , just_explain_(just_explain)
    , allow_speculative_sort_(allow_speculative_sort)
    , query_hint_(RegisteredQueryHint::fromConfig(executor_->getConfig())) {
  build();
}

RelAlgExecutionUnit WorkUnitBuilder::exeUnit() const {
  std::vector<const ir::Expr*> target_exprs;
  target_exprs.reserve(target_exprs_[0].size());
  for (auto& expr : target_exprs_[0]) {
    target_exprs.push_back(expr.get());
  }
  return {input_descs_,
          input_col_descs_,
          simple_quals_,
          quals_,
          join_quals_,
          groupby_exprs_,
          target_exprs,
          estimator_,
          sort_info_,
          scan_limit_,
          query_hint_,
          query_plan_dag_,
          hash_table_build_plan_dag_,
          table_id_to_node_map_,
          use_bump_allocator_,
          union_all_};
}

void WorkUnitBuilder::build() {
  max_groups_buffer_entry_guess_ =
      executor_->getConfig().exec.group_by.default_max_groups_buffer_entry_guess;
  assignNestLevels(root_);
  computeJoinTypes(root_);
  auto max_rte_idx =
      std::max_element(input_nest_levels_.begin(),
                       input_nest_levels_.end(),
                       [](auto lhs, auto rhs) { return lhs.second < rhs.second; })
          ->second;
  target_exprs_.resize(max_rte_idx + 1);
  // Number of join types must match the max nest level used for input.
  CHECK_EQ(join_types_.size(), max_rte_idx);
  computeInputDescs();
  process(root_);
  if (!join_types_.empty()) {
    reorderTables();
  }
  computeSimpleQuals();
  computeInputColDescs();
}

void WorkUnitBuilder::process(const ir::Node* node) {
  CHECK(all_nest_levels_.count(node));

  if (node->getResult() || node->is<ir::Scan>()) {
    auto scan = node->as<ir::Scan>();
    CHECK(input_nest_levels_.count(node));
    auto rte_idx = input_nest_levels_.at(node);
    // For UNION ALL we have two input nodes with the same nest level.
    // Add targets only for the first of them.
    bool add_targets = target_exprs_[rte_idx].empty();
    for (int i = 0; i < (int)node->size(); ++i) {
      ir::ExprPtr col_var;
      if (scan) {
        col_var = ir::makeExpr<ir::ColumnVar>(scan->getColumnInfo(i), rte_idx);
      } else {
        col_var = ir::makeExpr<ir::ColumnVar>(
            node->getOutputMetainfo()[i].type(), -node->getId(), i, rte_idx, false);
      }

      // RHS of left join is always nullable.
      if (rte_idx > 0 && join_types_[rte_idx - 1] == JoinType::LEFT &&
          !col_var->type()->nullable()) {
        col_var = col_var->withType(col_var->type()->withNullable(true));
      }

      input_rewriter_.addReplacement(node, i, col_var);
      if (add_targets) {
        target_exprs_[rte_idx].push_back(col_var);
      }
    }
    return;
  }

  for (size_t i = 0; i < node->inputCount(); ++i) {
    process(node->getInput(i));
  }

  if (node->as<ir::Aggregate>()) {
    processAggregate(node->as<ir::Aggregate>());
  } else if (node->is<ir::Project>()) {
    processProject(node->as<ir::Project>());
  } else if (node->is<ir::Filter>()) {
    processFilter(node->as<ir::Filter>());
  } else if (node->is<ir::Sort>()) {
    processSort(node->as<ir::Sort>());
  } else if (node->is<ir::LogicalUnion>()) {
    processUnion(node->as<ir::LogicalUnion>());
  } else if (node->is<ir::Join>()) {
    processJoin(node->as<ir::Join>());
  } else {
    CHECK(false) << "Unsupported node: " + node->toString();
  }
}

void WorkUnitBuilder::processAggregate(const ir::Aggregate* agg) {
  RelAlgTranslator translator(
      executor_, input_nest_levels_, join_types_, now_, eo_.just_explain);
  auto rte_idx = all_nest_levels_.at(agg);
  // We don't expect multiple aggregations in a single execution unit.
  if (!groupby_exprs_.empty()) {
    CHECK_EQ(groupby_exprs_.size(), (size_t)1);
    CHECK(groupby_exprs_.front() == nullptr);
    groupby_exprs_.clear();
  }

  ir::ExprPtrVector new_target_exprs;
  for (size_t i = 0; i < agg->getGroupByCount(); ++i) {
    groupby_exprs_.push_back(set_transient_dict(target_exprs_[rte_idx][i]));
    auto target_expr = var_ref(groupby_exprs_.back().get(), ir::Var::kGROUPBY, i + 1);
    new_target_exprs.push_back(target_expr);
  }

  for (auto& expr : agg->getAggs()) {
    auto rewritten_expr = input_rewriter_.visit(expr.get());
    auto target_expr = translator.normalize(rewritten_expr.get());
    target_expr = fold_expr(target_expr.get());
    new_target_exprs.emplace_back(target_expr);
  }

  if (dag_) {
    auto candidate_hint = dag_->getQueryHint(agg);
    if (candidate_hint) {
      query_hint_ = *candidate_hint;
    }
  }

  for (size_t i = 0; i < agg->size(); ++i) {
    input_rewriter_.addReplacement(agg, i, new_target_exprs[i]);
  }
  target_exprs_[rte_idx] = std::move(new_target_exprs);
  is_agg_ = true;
}

void WorkUnitBuilder::processProject(const ir::Project* proj) {
  RelAlgTranslator translator(
      executor_, input_nest_levels_, join_types_, now_, eo_.just_explain);
  auto rte_idx = all_nest_levels_.at(proj);
  ir::ExprPtrVector new_target_exprs;
  for (auto& expr : proj->getExprs()) {
    auto rewritten_expr = input_rewriter_.visit(expr.get());
    auto target_expr = translate(rewritten_expr.get(), translator, eo_.executor_type);
    new_target_exprs.emplace_back(std::move(target_expr));
  }
  if (dag_) {
    auto candidate_hint = dag_->getQueryHint(proj);
    if (candidate_hint) {
      query_hint_ = *candidate_hint;
    }
  }

  for (size_t i = 0; i < proj->size(); ++i) {
    input_rewriter_.addReplacement(proj, i, new_target_exprs[i]);
  }
  target_exprs_[rte_idx] = std::move(new_target_exprs);
  if (groupby_exprs_.empty()) {
    groupby_exprs_.push_back(nullptr);
  }
}

void WorkUnitBuilder::processFilter(const ir::Filter* filter) {
  RelAlgTranslator translator(
      executor_, input_nest_levels_, join_types_, now_, eo_.just_explain);
  auto rte_idx = all_nest_levels_.at(filter);

  // If we filter the result of join then we can merge the filter with
  // join conditions. That maght help to filter-out rows earlier.
  if (!rte_idx && !join_quals_.empty()) {
    auto filter_quals = makeJoinQuals(filter->getConditionExpr());
    for (const auto& qual : filter_quals) {
      const auto qual_rte_idx = MaxRangeTableIndexCollector::collect(qual.get());
      CHECK(static_cast<size_t>(qual_rte_idx) <= join_quals_.size());
      auto qual_level = qual_rte_idx ? qual_rte_idx - 1 : 0;
      // Left join produces a row even if the condition fails, so we either find
      // a proper inner join, or put the filter into quals.
      auto it = std::find_if(join_quals_.begin() + qual_level,
                             join_quals_.end(),
                             [](auto& v) { return v.type == JoinType::INNER; });
      if (it != join_quals_.end()) {
        it->quals.push_back(qual);
      } else {
        quals_.push_back(qual);
      }
    }
  } else {
    auto rewritten_expr = input_rewriter_.visit(filter->getConditionExpr());
    auto filter_expr = translator.normalize(rewritten_expr.get());
    auto qual = fold_expr(filter_expr.get());
    quals_.push_back(qual);
  }

  CHECK_EQ(filter->size(), target_exprs_[rte_idx].size());
  for (size_t i = 0; i < filter->size(); ++i) {
    input_rewriter_.addReplacement(filter, i, target_exprs_[rte_idx][i]);
  }
  if (groupby_exprs_.empty()) {
    groupby_exprs_.push_back(nullptr);
  }
}

void WorkUnitBuilder::processSort(const ir::Sort* sort) {
  sort_info_.algorithm =
      allow_speculative_sort_ ? SortAlgorithm::SpeculativeTopN : SortAlgorithm::Default;
  if (groupby_exprs_.size() == (size_t)1 && !groupby_exprs_.front()) {
    sort_info_.algorithm = SortAlgorithm::StreamingTopN;
  }
  sort_info_.order_entries = get_order_entries(sort);
  sort_info_.limit = sort->getLimit();
  sort_info_.offset = sort->getOffset();

  // Check if selected fields are supported for sort.
  CHECK_EQ(all_nest_levels_.at(sort), 0);
  for (size_t i = 0; i < sort->collationCount(); ++i) {
    auto sort_field = sort->getCollation(i);
    auto& expr = target_exprs_[0][sort_field.getField()];
    if (expr->type()->isArray()) {
      throw std::runtime_error("Columns with array types cannot be used for sorting.");
    }
  }
}

void WorkUnitBuilder::processUnion(const hdk::ir::LogicalUnion* logical_union) {
  auto rte_idx = all_nest_levels_.at(logical_union);
  if (!logical_union->isAll()) {
    throw std::runtime_error("UNION without ALL is not supported yet.");
  }
  // Will throw a std::runtime_error if types don't match.
  logical_union->checkForMatchingMetaInfoTypes();
  // Only Projections and Aggregates from a UNION are supported for now.
  CHECK(dag_);
  dag_->eachNode([logical_union](hdk::ir::Node const* node) {
    if (node->hasInput(logical_union) &&
        !shared::dynamic_castable_to_any<hdk::ir::Project,
                                         hdk::ir::LogicalUnion,
                                         hdk::ir::Aggregate>(node)) {
      throw std::runtime_error("UNION ALL not yet supported in this context.");
    }
  });

  const auto query_infos = get_table_infos(input_descs_, executor_);
  auto const max_num_tuples =
      std::accumulate(query_infos.cbegin(),
                      query_infos.cend(),
                      size_t(0),
                      [](auto max, auto const& query_info) {
                        return std::max(max, query_info.info.getNumTuples());
                      });

  // We expect input ot be either scan or an execution point.
  // In both cases target expressions should already have all
  // required column vars.
  CHECK(logical_union->getInput(0)->is<ir::Scan>() ||
        logical_union->getInput(0)->getResult());
  CHECK_EQ(logical_union->size(), target_exprs_[rte_idx].size());

  scan_limit_ = max_num_tuples;
  union_all_ = logical_union->isAll();
  CHECK(groupby_exprs_.empty());
  groupby_exprs_.push_back(nullptr);
}

void WorkUnitBuilder::processJoin(const ir::Join* join) {
  auto rte_idx = all_nest_levels_.at(join);
  CHECK_EQ(rte_idx, 0);

  if (left_deep_join_input_sizes_.empty()) {
    left_deep_join_input_sizes_.push_back(join->getInput(0)->size());
  }
  left_deep_join_input_sizes_.push_back(join->getInput(1)->size());
  JoinCondition join_cond;
  join_cond.quals = makeJoinQuals(join->getCondition());
  join_cond.type = join->getJoinType();
  join_quals_.emplace_back(join_cond);

  ir::ExprPtrVector new_target_exprs;
  new_target_exprs.reserve(join->size());
  for (size_t i = 0; i < join->inputCount(); ++i) {
    auto input_rte_idx = all_nest_levels_.at(join->getInput(i));
    new_target_exprs.insert(new_target_exprs.end(),
                            target_exprs_[input_rte_idx].begin(),
                            target_exprs_[input_rte_idx].end());
  }
  for (size_t i = 0; i < new_target_exprs.size(); ++i) {
    input_rewriter_.addReplacement(join, i, new_target_exprs[i]);
  }
  target_exprs_[rte_idx] = std::move(new_target_exprs);
  left_deep_tree_id_ = join->getId();
}

std::list<hdk::ir::ExprPtr> WorkUnitBuilder::makeJoinQuals(
    const hdk::ir::Expr* join_condition) {
  RelAlgTranslator translator(
      executor_, input_nest_levels_, join_types_, now_, eo_.just_explain);

  std::list<hdk::ir::ExprPtr> join_condition_quals;
  auto rewritten_condition = input_rewriter_.visit(join_condition);
  auto bw_equals = get_bitwise_equals_conjunction(rewritten_condition.get());
  auto condition_expr =
      translator.normalize(bw_equals ? bw_equals.get() : rewritten_condition.get());
  condition_expr = reverse_logical_distribution(condition_expr);
  auto join_condition_cf = qual_to_conjunctive_form(condition_expr);
  join_condition_quals.insert(join_condition_quals.end(),
                              join_condition_cf.quals.begin(),
                              join_condition_cf.quals.end());
  join_condition_quals.insert(join_condition_quals.end(),
                              join_condition_cf.simple_quals.begin(),
                              join_condition_cf.simple_quals.end());

  return combine_equi_join_conditions(join_condition_quals);
}

void WorkUnitBuilder::reorderTables() {
  if (!executor_->getConfig().opts.from_table_reordering ||
      std::find(join_types_.begin(), join_types_.end(), JoinType::LEFT) !=
          join_types_.end()) {
    return;
  }

  bool do_reordering = false;

  if (executor_->getConfig().opts.from_table_reordering) {
    const auto query_infos = get_table_infos(input_descs_, executor_);
    input_permutation_ = get_node_input_permutation(join_quals_, query_infos, executor_);
    // Adjust nest levels and input descriptors.
    for (size_t i = 0; i < input_permutation_.size(); ++i) {
      if (input_permutation_[i] != i) {
        do_reordering = true;
        break;
      }
    }
  }

  if (do_reordering) {
    for (auto& pr : input_nest_levels_) {
      if (pr.second < (int)input_permutation_.size()) {
        pr.second = input_permutation_[pr.second];
      }
    }
    computeInputDescs();

    NestLevelRewriter rewriter(input_descs_);
    for (auto& expr : target_exprs_[0]) {
      expr = rewriter.visit(expr.get());
    }
    for (auto& expr : simple_quals_) {
      expr = rewriter.visit(expr.get());
    }
    for (auto& expr : quals_) {
      expr = rewriter.visit(expr.get());
    }
    for (auto& expr : groupby_exprs_) {
      if (expr) {
        expr = rewriter.visit(expr.get());
      }
    }
    JoinQualsPerNestingLevel new_join_quals(join_quals_.size());
    for (size_t i = 0; i < join_quals_.size(); ++i) {
      new_join_quals[i].type = join_quals_[i].type;
      for (auto& qual : join_quals_[i].quals) {
        auto new_qual = rewriter.visit(qual.get());

        // Inner join quals might require nest level change.
        auto qual_rte_idx = MaxRangeTableIndexCollector::collect(new_qual.get());
        auto new_level = qual_rte_idx ? qual_rte_idx - 1 : 0;
        new_join_quals[new_level].quals.push_back(new_qual);
      }
    }
    join_quals_ = std::move(new_join_quals);
  }
}

void WorkUnitBuilder::computeSimpleQuals() {
  CHECK(simple_quals_.empty());
  std::list<hdk::ir::ExprPtr> simple_quals;
  std::list<hdk::ir::ExprPtr> quals;
  for (auto qual : quals_) {
    auto qual_cf = qual_to_conjunctive_form(qual);
    simple_quals.insert(
        simple_quals.end(), qual_cf.simple_quals.begin(), qual_cf.simple_quals.end());
    quals.insert(quals.end(), qual_cf.quals.begin(), qual_cf.quals.end());
  }
  simple_quals_ = std::move(simple_quals);
  quals_ = std::move(quals);
}

int WorkUnitBuilder::assignNestLevels(const ir::Node* node, int start_idx) {
  all_nest_levels_.emplace(node, start_idx);

  if (node->getResult() || !node->inputCount()) {
    input_nest_levels_.emplace(node, start_idx);
    return start_idx + 1;
  }

  if (node->is<ir::LogicalUnion>()) {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      assignNestLevels(node->getInput(i), start_idx);
    }
    ++start_idx;
  } else {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      start_idx = assignNestLevels(node->getInput(i), start_idx);
    }
  }

  return start_idx;
}

void WorkUnitBuilder::computeJoinTypes(const ir::Node* node, bool allow_join) {
  if (node->getResult()) {
    return;
  }

  // We expect that joins can have other joins only in its outer
  // input.
  if (node->is<ir::Join>() || node->is<ir::LeftDeepInnerJoin>()) {
    CHECK(allow_join) << "Unsupported bushy join detected";
    computeJoinTypes(node->getInput(0), true);

    if (auto deep_join = node->as<ir::LeftDeepInnerJoin>()) {
      auto join_types = left_deep_join_types(deep_join);
      join_types_.insert(join_types_.end(), join_types.begin(), join_types.end());
    } else {
      join_types_.push_back(node->as<ir::Join>()->getJoinType());
    }

    // That is only to check we don't have bushy joins.
    for (size_t i = 1; i < node->inputCount(); ++i) {
      computeJoinTypes(node->getInput(i), false);
    }
  } else {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      computeJoinTypes(node->getInput(i), allow_join);
    }
  }
}

void WorkUnitBuilder::computeInputDescs() {
  input_descs_.clear();
  for (auto& [node, rte_idx] : input_nest_levels_) {
    if (auto scan = node->as<ir::Scan>()) {
      input_descs_.emplace_back(scan->getDatabaseId(), scan->getTableId(), rte_idx);
    } else {
      input_descs_.emplace_back(-1, -node->getId(), rte_idx);
    }
  }

  std::sort(input_descs_.begin(),
            input_descs_.end(),
            [](const InputDescriptor& lhs, const InputDescriptor& rhs) {
              return lhs.getNestLevel() < rhs.getNestLevel();
            });
}

void WorkUnitBuilder::computeInputColDescs() {
  // Scan all currently used expressions to determine used columns.
  UsedInputsCollector collector;
  for (auto& expr : target_exprs_[0]) {
    collector.visit(expr.get());
  }
  for (auto& expr : simple_quals_) {
    collector.visit(expr.get());
  }
  for (auto& expr : quals_) {
    collector.visit(expr.get());
  }
  for (auto& expr : groupby_exprs_) {
    if (expr) {
      collector.visit(expr.get());
    }
  }
  for (auto& join_qual : join_quals_) {
    for (auto& expr : join_qual.quals) {
      collector.visit(expr.get());
    }
  }

  auto cols = collector.result();
  std::vector<std::shared_ptr<const InputColDescriptor>> col_descs;
  for (auto& col_var : collector.result()) {
    col_descs.push_back(std::make_shared<const InputColDescriptor>(col_var.columnInfo(),
                                                                   col_var.rteIdx()));
  }

  // For UNION we only have column variables for a single table used
  // in target expressions but should mark all columns as used.
  if (union_all_) {
    for (auto& col_var : collector.result()) {
      for (auto tdesc : input_descs_) {
        if (col_var.tableId() != tdesc.getTableId()) {
          auto col_info = std::make_shared<ColumnInfo>(col_var.dbId(),
                                                       tdesc.getTableId(),
                                                       col_var.columnId(),
                                                       "",
                                                       col_var.type(),
                                                       false);
          col_descs.push_back(
              std::make_shared<InputColDescriptor>(col_info, col_var.rteIdx()));
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

  input_col_descs_.clear();
  input_col_descs_.insert(input_col_descs_.end(), col_descs.begin(), col_descs.end());
}

}  // namespace hdk
