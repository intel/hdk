/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CanonizeQuery.h"

#include "IR/Expr.h"
#include "IR/ExprCollector.h"
#include "IR/InputRewriter.h"
#include "IR/Node.h"
#include "QueryBuilder/QueryBuilder.h"

#include <unordered_set>

namespace hdk::ir {

namespace {

bool isCompoundAggregate(const AggExpr* agg) {
  return agg->aggType() == AggType::kStdDevSamp || agg->aggType() == AggType::kCorr;
}

/**
 * Base class holding interface for compound aggregate expansion.
 * Compound aggregate is expanded in three steps.
 *
 * 1. Create an additional projection with new columns required for aggregate
 *    computation. This step is done using addInputExprs method.
 * 2. Replace original compound aggregate expressions with one or more regular
 *    aggregates. This step is done using addAggregates method.
 * 3. Create a reduction projection to compute the final aggregate values.
 *    This step is done using addResult method.
 */
class CompoundAggregate {
 public:
  CompoundAggregate(Aggregate* node,
                    const AggExpr* agg,
                    std::string field,
                    ConfigPtr config)
      : ctx_(agg->ctx())
      , builder_(ctx_, nullptr, config)
      , node_(node)
      , agg_(agg)
      , field_(std::move(field)) {}
  virtual ~CompoundAggregate() = default;

  virtual void addInputExprs(ExprPtrVector& exprs, std::vector<std::string>& fields) = 0;
  virtual void addAggregates(const Project* input,
                             ExprPtrVector& new_aggs,
                             std::vector<std::string>& new_agg_fields) = 0;
  virtual void addResult(ExprPtrVector& reduce_proj_exprs) = 0;

 protected:
  Context& ctx_;
  QueryBuilder builder_;
  Aggregate* node_;
  const AggExpr* agg_;
  std::string field_;
};

/**
 * stddev_samp(x) = sqrt((sum(x * x) - sum(x) * sum(x) / n) / (n - 1))
 */
class StdDevAggregate final : public CompoundAggregate {
 public:
  StdDevAggregate(Aggregate* node,
                  const AggExpr* agg,
                  std::string field,
                  ConfigPtr config)
      : CompoundAggregate(node, agg, field, config) {}

  void addInputExprs(ExprPtrVector& exprs, std::vector<std::string>& fields) override {
    // Add arg * arg column.
    BuilderExpr arg(&builder_, agg_->argShared());
    quad_arg_index_ = static_cast<unsigned>(exprs.size());
    exprs.emplace_back(arg.mul(arg).expr());
    fields.emplace_back(field_ + "__stddev_quad_arg_" + std::to_string(quad_arg_index_));
  }

  void addAggregates(const Project* input,
                     ExprPtrVector& new_aggs,
                     std::vector<std::string>& new_agg_fields) override {
    // Create count(arg), sum(arg), sum(arg * arg)
    BuilderExpr arg(&builder_, agg_->argShared());
    BuilderExpr cnt = arg.count();
    BuilderExpr sum = arg.sum();
    BuilderExpr quad_arg(&builder_, getNodeColumnRef(input, quad_arg_index_));
    BuilderExpr quad_sum = quad_arg.sum();

    auto new_col_idx = static_cast<unsigned>(new_aggs.size() + node_->getGroupByCount());
    new_aggs.emplace_back(cnt.expr());
    new_agg_fields.emplace_back(field_ + "_cnt");
    new_aggs.emplace_back(sum.expr());
    new_agg_fields.emplace_back(field_ + "_sum");
    new_aggs.emplace_back(quad_sum.expr());
    new_agg_fields.emplace_back(field_ + "_quad_sum");

    cnt_ref_ = makeExpr<ir::ColumnRef>(cnt.type(), node_, new_col_idx);
    sum_ref_ = makeExpr<ir::ColumnRef>(sum.type(), node_, new_col_idx + 1);
    quad_sum_ref_ = makeExpr<ir::ColumnRef>(quad_sum.type(), node_, new_col_idx + 2);
  }

  void addResult(ExprPtrVector& reduce_proj_exprs) override {
    BuilderExpr cnt_ref(&builder_, cnt_ref_);
    BuilderExpr sum_ref(&builder_, sum_ref_);
    BuilderExpr quad_sum_ref(&builder_, quad_sum_ref_);
    auto cnt_or_null =
        builder_.ifThenElse(cnt_ref.eq(0), builder_.nullCst(cnt_ref.type()), cnt_ref);
    auto cnt_m_1_or_null = builder_.ifThenElse(
        cnt_ref.eq(1), builder_.nullCst(cnt_ref.type()), cnt_ref.sub(1));
    auto stddev = quad_sum_ref.cast(ctx_.fp64())
                      .sub(sum_ref.mul(sum_ref).cast(ctx_.fp64()).div(cnt_or_null))
                      .div(cnt_m_1_or_null)
                      .pow(0.5);
    reduce_proj_exprs.emplace_back(stddev.expr());
  }

 private:
  unsigned quad_arg_index_ = 0;
  ExprPtr cnt_ref_;
  ExprPtr sum_ref_;
  ExprPtr quad_sum_ref_;
};

/**
 * corr(x, y) = (avg(x * y) * cnt(x) * cnt(y) - sum(x) * sum(y)) /
 * (sqrt(cnt(x) * sum(x * x) - sum(x) * sum(x)) *
 *  sqrt(cnt(y) * sum(y * y) - sum(y) * sum(y)))
 */
class CorrAggregate final : public CompoundAggregate {
 public:
  CorrAggregate(Aggregate* node, const AggExpr* agg, std::string field, ConfigPtr config)
      : CompoundAggregate(node, agg, field, config) {}

  void addInputExprs(ExprPtrVector& exprs, std::vector<std::string>& fields) override {
    // Add x * x, y * y, x * y columns
    BuilderExpr x_arg(&builder_, agg_->argShared());
    BuilderExpr y_arg(&builder_, agg_->arg1Shared());
    first_mul_arg_index_ = static_cast<unsigned>(exprs.size());
    fields.emplace_back(field_ + "__corr_x_mul_x_" + std::to_string(exprs.size()));
    exprs.emplace_back(x_arg.mul(x_arg).expr());
    fields.emplace_back(field_ + "__corr_y_mul_y_" + std::to_string(exprs.size()));
    exprs.emplace_back(y_arg.mul(y_arg).expr());
    fields.emplace_back(field_ + "__corr_x_mul_y_" + std::to_string(exprs.size()));
    exprs.emplace_back(x_arg.mul(y_arg).expr());
  }

  void addAggregates(const Project* input,
                     ExprPtrVector& new_aggs,
                     std::vector<std::string>& new_agg_fields) override {
    // Create count(x), count(y), sum(x), sum(y), sum(x * x), sum(y * y), avg(x * y)
    BuilderExpr x_arg(&builder_, agg_->argShared());
    BuilderExpr y_arg(&builder_, agg_->arg1Shared());
    BuilderExpr quad_x(&builder_, getNodeColumnRef(input, first_mul_arg_index_));
    BuilderExpr quad_y(&builder_, getNodeColumnRef(input, first_mul_arg_index_ + 1));
    BuilderExpr x_mul_y(&builder_, getNodeColumnRef(input, first_mul_arg_index_ + 2));
    auto x_cnt = x_arg.count();
    auto y_cnt = y_arg.count();
    auto sum_x = x_arg.sum();
    auto sum_y = y_arg.sum();
    auto sum_quad_x = quad_x.sum();
    auto sum_quad_y = quad_y.sum();
    auto avg_x_mul_y = x_mul_y.avg();

    // Add created aggregates
    auto new_col_idx = static_cast<unsigned>(new_aggs.size() + node_->getGroupByCount());
    new_aggs.emplace_back(x_cnt.expr());
    new_agg_fields.emplace_back(field_ + "_x_cnt");
    new_aggs.emplace_back(y_cnt.expr());
    new_agg_fields.emplace_back(field_ + "_y_cnt");
    new_aggs.emplace_back(sum_x.expr());
    new_agg_fields.emplace_back(field_ + "_sum_x");
    new_aggs.emplace_back(sum_y.expr());
    new_agg_fields.emplace_back(field_ + "_sum_y");
    new_aggs.emplace_back(sum_quad_x.expr());
    new_agg_fields.emplace_back(field_ + "_sum_quad_x");
    new_aggs.emplace_back(sum_quad_y.expr());
    new_agg_fields.emplace_back(field_ + "_sum_quad_y");
    new_aggs.emplace_back(avg_x_mul_y.expr());
    new_agg_fields.emplace_back(field_ + "_avg_x_mul_y");

    // Create references to new aggregates for the later use (cannot use builder
    // because new aggregates are not in the node yet).
    x_cnt_ref_ = makeExpr<ir::ColumnRef>(x_cnt.type(), node_, new_col_idx);
    y_cnt_ref_ = makeExpr<ir::ColumnRef>(y_cnt.type(), node_, new_col_idx + 1);
    sum_x_ref_ = makeExpr<ir::ColumnRef>(sum_x.type(), node_, new_col_idx + 2);
    sum_y_ref_ = makeExpr<ir::ColumnRef>(sum_y.type(), node_, new_col_idx + 3);
    sum_quad_x_ref_ = makeExpr<ir::ColumnRef>(sum_quad_x.type(), node_, new_col_idx + 4);
    sum_quad_y_ref_ = makeExpr<ir::ColumnRef>(sum_quad_y.type(), node_, new_col_idx + 5);
    avg_x_mul_y_ref_ =
        makeExpr<ir::ColumnRef>(avg_x_mul_y.type(), node_, new_col_idx + 6);
  }

  void addResult(ExprPtrVector& reduce_proj_exprs) override {
    BuilderExpr x_cnt(&builder_, x_cnt_ref_);
    BuilderExpr y_cnt(&builder_, y_cnt_ref_);
    BuilderExpr sum_x(&builder_, sum_x_ref_);
    BuilderExpr sum_y(&builder_, sum_y_ref_);
    BuilderExpr sum_quad_x(&builder_, sum_quad_x_ref_);
    BuilderExpr sum_quad_y(&builder_, sum_quad_y_ref_);
    BuilderExpr avg_x_mul_y(&builder_, avg_x_mul_y_ref_);

    auto x_cnt_or_null =
        builder_.ifThenElse(x_cnt.eq(0), builder_.nullCst(x_cnt.type()), x_cnt);
    auto y_cnt_or_null =
        builder_.ifThenElse(y_cnt.eq(0), builder_.nullCst(y_cnt.type()), y_cnt);
    // In case of integer input, we compute sums in INT64 for better precision
    // but move to FP64 here to avoid overflows in multiplications.
    auto sum_x_fp = sum_x.cast(ctx_.fp64());
    auto sum_y_fp = sum_y.cast(ctx_.fp64());
    auto sum_quad_x_fp = sum_quad_x.cast(ctx_.fp64());
    auto sum_quad_y_fp = sum_quad_y.cast(ctx_.fp64());

    auto den1 = x_cnt_or_null * sum_quad_x_fp - sum_x_fp * sum_x_fp;
    auto den2 = y_cnt_or_null * sum_quad_y_fp - sum_y_fp * sum_y_fp;
    auto den = den1.mul(den2).pow(0.5);
    auto corr = (avg_x_mul_y * x_cnt_or_null * y_cnt_or_null - sum_x_fp * sum_y_fp) / den;

    reduce_proj_exprs.emplace_back(corr.expr());
  }

 private:
  unsigned first_mul_arg_index_ = 0;
  ExprPtr x_cnt_ref_;
  ExprPtr y_cnt_ref_;
  ExprPtr sum_x_ref_;
  ExprPtr sum_y_ref_;
  ExprPtr sum_quad_x_ref_;
  ExprPtr sum_quad_y_ref_;
  ExprPtr avg_x_mul_y_ref_;
};

std::unique_ptr<CompoundAggregate> createHandler(Aggregate* node,
                                                 const AggExpr* agg,
                                                 std::string field,
                                                 ConfigPtr config) {
  if (agg->aggType() == AggType::kStdDevSamp) {
    return std::make_unique<StdDevAggregate>(node, agg, field, config);
  } else if (agg->aggType() == AggType::kCorr) {
    return std::make_unique<CorrAggregate>(node, agg, field, config);
  }
  CHECK(false) << "Unsupported compound aggregate: " << agg->toString();
  return nullptr;
}

void expandCompoundAggregates(std::shared_ptr<Aggregate> node,
                              ConfigPtr config,
                              std::vector<NodePtr>& new_nodes) {
  auto source = node->getAndOwnInput(0);
  ExprPtrVector proj_exprs = getNodeColumnRefs(source.get());
  std::vector<std::string> proj_fields;
  for (size_t i = 0; i < proj_exprs.size(); ++i) {
    proj_fields.push_back("tmp_col_" + std::to_string(i));
  }
  std::unordered_map<const AggExpr*, std::unique_ptr<CompoundAggregate>> handlers;
  for (size_t i = 0; i < node->getAggsCount(); ++i) {
    auto agg = node->getAgg(i)->as<AggExpr>();
    if (isCompoundAggregate(agg)) {
      handlers[agg] = createHandler(
          node.get(), agg, node->getFieldName(i + node->getGroupByCount()), config);
      handlers[agg]->addInputExprs(proj_exprs, proj_fields);
    }
  }

  auto proj =
      std::make_shared<Project>(std::move(proj_exprs), std::move(proj_fields), source);
  new_nodes.push_back(proj);

  ExprPtrVector new_aggs;
  std::vector<std::string> new_agg_fields;
  ExprPtrVector reduce_proj_exprs;
  std::vector<std::string> reduce_proj_fields;
  // Add fields for group keys.
  for (size_t i = 0; i < node->getGroupByCount(); ++i) {
    reduce_proj_exprs.emplace_back(
        getNodeColumnRef(node.get(), static_cast<unsigned>(i)));
    reduce_proj_fields.emplace_back(node->getFieldName(i));
    new_agg_fields.emplace_back(node->getFieldName(i));
  }
  // Add fields for aggregates.
  for (size_t i = 0; i < node->getAggsCount(); ++i) {
    auto agg = node->getAgg(i)->as<AggExpr>();
    auto field = node->getFieldName(i + node->getGroupByCount());
    if (handlers.count(agg)) {
      handlers.at(agg)->addAggregates(proj.get(), new_aggs, new_agg_fields);
      handlers.at(agg)->addResult(reduce_proj_exprs);
      reduce_proj_fields.emplace_back(field);
    } else {
      auto col_idx = static_cast<unsigned>(new_aggs.size() + node->getGroupByCount());
      reduce_proj_exprs.emplace_back(
          makeExpr<ir::ColumnRef>(agg->type(), node.get(), col_idx));
      reduce_proj_fields.emplace_back(field);
      new_aggs.emplace_back(agg->shared());
      new_agg_fields.emplace_back(field);
    }
  }

  node->setAggExprs(std::move(new_aggs));
  node->setFields(std::move(new_agg_fields));
  node->replaceInput(source, proj);
  new_nodes.push_back(node);
  NodePtr reduce_proj = std::make_shared<Project>(
      std::move(reduce_proj_exprs), std::move(reduce_proj_fields), node);
  new_nodes.push_back(reduce_proj);
}

void expandCompoundAggregates(QueryDag& dag) {
  auto& nodes = dag.getNodes();
  std::vector<NodePtr> new_nodes;
  InputRewriter rewriter;
  std::unordered_map<const Node*, NodePtr> expanded_aggs;
  for (auto& node : nodes) {
    for (size_t i = 0; i < node->inputCount(); ++i) {
      // TODO: should we apply rewriter everywhere due to subqueries?
      if (expanded_aggs.count(node->getInput(i))) {
        node->replaceInput(
            node->getAndOwnInput(i), expanded_aggs.at(node->getInput(i)), rewriter);
      }
    }

    if (node->is<Aggregate>()) {
      std::unordered_set<const AggExpr*> aggs;
      auto agg_node = std::dynamic_pointer_cast<Aggregate>(node);
      bool has_compound_agg =
          std::any_of(agg_node->getAggs().begin(),
                      agg_node->getAggs().end(),
                      [](const ExprPtr& expr) {
                        CHECK(expr->is<AggExpr>());
                        return isCompoundAggregate(expr->as<AggExpr>());
                      });

      if (has_compound_agg) {
        expandCompoundAggregates(agg_node, dag.config(), new_nodes);
        rewriter.addNodeMapping(agg_node.get(), new_nodes.back().get());
        expanded_aggs.insert(std::make_pair(agg_node.get(), new_nodes.back()));
      } else {
        new_nodes.push_back(node);
      }
    } else {
      new_nodes.push_back(node);
    }
  }

  if (!expanded_aggs.empty()) {
    dag.setNodes(std::move(new_nodes));
    if (expanded_aggs.count(dag.getRootNode())) {
      dag.setRootNode(expanded_aggs.at(dag.getRootNode()));
    }
  }
}

using InputSet =
    std::unordered_set<std::pair<const hdk::ir::Node*, unsigned>,
                       boost::hash<std::pair<const hdk::ir::Node*, unsigned>>>;

class InputCollector : public hdk::ir::ExprCollector<InputSet, InputCollector> {
 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    result_.emplace(col_ref->node(), col_ref->index());
  }
};

/**
 * Inserts a simple project before any project containing a window function node. Forces
 * all window function inputs into a single contiguous buffer for centralized processing
 * (e.g. in distributed mode). This is also needed when a window function node is preceded
 * by a filter node, both for correctness (otherwise a window operator will be coalesced
 * with its preceding filter node and be computer over unfiltered results, and for
 * performance, as currently filter nodes that are not coalesced into projects keep all
 * columns from the table as inputs, and hence bring everything in memory.
 * Once the new project has been created, the inputs in the
 * window function project must be rewritten to read from the new project, and to index
 * off the projected exprs in the new project.
 */
void addWindowFunctionPreProject(
    QueryDag& dag,
    const bool always_add_project_if_first_project_is_window_expr = false) {
  auto nodes = dag.getNodes();
  std::list<std::shared_ptr<hdk::ir::Node>> node_list(nodes.begin(), nodes.end());
  size_t project_node_counter{0};
  for (auto node_itr = node_list.begin(); node_itr != node_list.end(); ++node_itr) {
    const auto node = *node_itr;

    auto window_func_project_node = std::dynamic_pointer_cast<hdk::ir::Project>(node);
    if (!window_func_project_node) {
      continue;
    }
    project_node_counter++;
    if (!window_func_project_node->hasWindowFunctionExpr()) {
      // this projection node does not have a window function
      // expression -- skip to the next node in the DAG.
      continue;
    }

    const auto prev_node_itr = std::prev(node_itr);
    const auto prev_node = *prev_node_itr;
    CHECK(prev_node);

    auto filter_node = std::dynamic_pointer_cast<hdk::ir::Filter>(prev_node);

    auto scan_node = std::dynamic_pointer_cast<hdk::ir::Scan>(prev_node);
    const bool has_multi_fragment_scan_input =
        (scan_node && scan_node->getNumFragments() > 1) ? true : false;

    // We currently add a preceding project node in one of two conditions:
    // 1. always_add_project_if_first_project_is_window_expr = true, which
    // we currently only set for distributed, but could also be set to support
    // multi-frag window function inputs, either if we can detect that an input table
    // is multi-frag up front, or using a retry mechanism like we do for join filter
    // push down.
    // TODO(todd): Investigate a viable approach for the above.
    // 2. Regardless of #1, if the window function project node is preceded by a
    // filter node. This is required both for correctness and to avoid pulling
    // all source input columns into memory since non-coalesced filter node
    // inputs are currently not pruned or eliminated via dead column elimination.
    // TODO(todd): Investigate whether the shotgun filter node issue affects other
    // query plans, i.e. filters before joins, and whether there is a more general
    // approach to solving this (will still need the preceding project node for
    // window functions preceded by filter nodes for correctness though)

    if (!((always_add_project_if_first_project_is_window_expr &&
           project_node_counter == 1) ||
          filter_node || has_multi_fragment_scan_input)) {
      continue;
    }

    InputCollector input_collector;
    for (size_t i = 0; i < window_func_project_node->size(); i++) {
      input_collector.visit(window_func_project_node->getExpr(i).get());
    }
    const InputSet& inputs = input_collector.result();

    // Note: Technically not required since we are mapping old inputs to new input
    // indices, but makes the re-mapping of inputs easier to follow.
    std::vector<std::pair<const hdk::ir::Node*, unsigned>> sorted_inputs(inputs.begin(),
                                                                         inputs.end());
    std::sort(sorted_inputs.begin(),
              sorted_inputs.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    hdk::ir::ExprPtrVector exprs;
    std::vector<std::string> fields;
    std::unordered_map<unsigned, unsigned> old_index_to_new_index;
    for (auto& input : sorted_inputs) {
      CHECK_EQ(input.first, prev_node.get());
      auto res =
          old_index_to_new_index.insert(std::make_pair(input.second, exprs.size()));
      CHECK(res.second);
      exprs.emplace_back(hdk::ir::makeExpr<hdk::ir::ColumnRef>(
          getColumnType(input.first, input.second), input.first, input.second));
      fields.emplace_back("");
    }

    auto new_project =
        std::make_shared<hdk::ir::Project>(std::move(exprs), fields, prev_node);
    node_list.insert(node_itr, new_project);
    window_func_project_node->replaceInput(
        prev_node, new_project, old_index_to_new_index);
  }

  // Any applied transformation always increases number of nodes.
  if (node_list.size() != nodes.size()) {
    nodes.assign(node_list.begin(), node_list.end());
    dag.setNodes(std::move(nodes));
  }
}

}  // namespace

void canonizeQuery(QueryDag& dag) {
  expandCompoundAggregates(dag);
  addWindowFunctionPreProject(dag);
}

}  // namespace hdk::ir
