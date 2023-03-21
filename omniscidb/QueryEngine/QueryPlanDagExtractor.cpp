/*
 * Copyright 2021 OmniSci, Inc.
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

#include "QueryPlanDagExtractor.h"
#include "Visitors/QueryPlanDagChecker.h"

#include <boost/algorithm/cxx11/any_of.hpp>

namespace {
struct IsEquivBinOp {
  bool operator()(hdk::ir::ExprPtr const& qual) {
    if (auto oper = std::dynamic_pointer_cast<const hdk::ir::BinOper>(qual)) {
      return oper->isEquivalence();
    }
    return false;
  }
};
}  // namespace

std::vector<InnerOuterOrLoopQual> QueryPlanDagExtractor::normalizeColumnsPair(
    const hdk::ir::BinOper* condition) {
  std::vector<InnerOuterOrLoopQual> result;
  const auto lhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(condition->leftOperand());
  const auto rhs_tuple_expr =
      dynamic_cast<const hdk::ir::ExpressionTuple*>(condition->rightOperand());

  CHECK_EQ(static_cast<bool>(lhs_tuple_expr), static_cast<bool>(rhs_tuple_expr));
  auto do_normalize_inner_outer_pair = [this, &result](
                                           const hdk::ir::Expr* lhs,
                                           const hdk::ir::Expr* rhs,
                                           const TemporaryTables* temporary_table) {
    try {
      auto inner_outer_pair =
          HashJoin::normalizeColumnPair(lhs, rhs, schema_provider_, temporary_table);
      InnerOuterOrLoopQual valid_qual{
          std::make_pair(inner_outer_pair.first, inner_outer_pair.second), false};
      result.push_back(valid_qual);
    } catch (HashJoinFail& e) {
      InnerOuterOrLoopQual invalid_qual{std::make_pair(lhs, rhs), true};
      result.push_back(invalid_qual);
    }
  };
  if (lhs_tuple_expr) {
    const auto& lhs_tuple = lhs_tuple_expr->tuple();
    const auto& rhs_tuple = rhs_tuple_expr->tuple();
    CHECK_EQ(lhs_tuple.size(), rhs_tuple.size());
    for (size_t i = 0; i < lhs_tuple.size(); ++i) {
      do_normalize_inner_outer_pair(
          lhs_tuple[i].get(), rhs_tuple[i].get(), &temporary_tables_);
    }
  } else {
    do_normalize_inner_outer_pair(
        condition->leftOperand(), condition->rightOperand(), &temporary_tables_);
  }
  return result;
}

// To extract query plan DAG, we call this function with root node of the query plan
// and some objects required while extracting DAG
// We consider a DAG representation of a query plan as a series of "unique" rel node ids
// We decide each rel node's node id by searching the cached plan DAG first,
// and assign a new id iff there exists no duplicated rel node that can reuse
ExtractedPlanDag QueryPlanDagExtractor::extractQueryPlanDag(
    const hdk::ir::Node*
        node, /* the root node of the query plan tree we want to extract its DAG */
    SchemaProviderPtr schema_provider,
    std::optional<unsigned> left_deep_tree_id,
    std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
    const TemporaryTables& temporary_tables,
    Executor* executor,
    const RelAlgTranslator& rel_alg_translator) {
  // check if this plan tree has not supported pattern for DAG extraction
  auto dag_checker_res =
      QueryPlanDagChecker::hasNonSupportedNodeInDag(node, rel_alg_translator);
  if (dag_checker_res.first) {
    VLOG(1) << "Stop DAG extraction (" << dag_checker_res.second << ")";
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }

  return extractQueryPlanDagImpl(node,
                                 schema_provider,
                                 left_deep_tree_id,
                                 left_deep_tree_infos,
                                 temporary_tables,
                                 executor);
}

ExtractedPlanDag QueryPlanDagExtractor::extractQueryPlanDagImpl(
    const hdk::ir::Node*
        node, /* the root node of the query plan tree we want to extract its DAG */
    SchemaProviderPtr schema_provider,
    std::optional<unsigned> left_deep_tree_id,
    std::unordered_map<unsigned, JoinQualsPerNestingLevel>& left_deep_tree_infos,
    const TemporaryTables& temporary_tables,
    Executor* executor) {
  mapd_unique_lock<mapd_shared_mutex> lock(executor->getDataRecyclerLock());

  auto& cached_dag = executor->getQueryPlanDagCache();
  QueryPlanDagExtractor dag_extractor(
      cached_dag, schema_provider, left_deep_tree_infos, temporary_tables, executor);

  // add the root node of this query plan DAG
  auto res = cached_dag.addNodeIfAbsent(node);
  if (!res) {
    VLOG(1) << "Stop DAG extraction (Query plan dag cache reaches the maximum capacity)";
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }
  CHECK(res.has_value());
  node->setRelNodeDagId(res.value());
  dag_extractor.extracted_dag_.push_back(res.value());

  // visit child node if necessary
  auto num_child_node = node->inputCount();
  switch (num_child_node) {
    case 1:  // unary op
      dag_extractor.visit(node, node->getInput(0));
      break;
    case 2:  // binary op
      if (auto trans_join_node = dynamic_cast<const hdk::ir::TranslatedJoin*>(node)) {
        dag_extractor.visit(trans_join_node, trans_join_node->getLHS());
        dag_extractor.visit(trans_join_node, trans_join_node->getRHS());
        break;
      }
      VLOG(1) << "Visit an invalid rel node while extracting query plan DAG: "
              << ::toString(node);
      return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
    case 0:  // leaf node
      break;
    default:
      // since we replace RelLeftDeepJoin as a set of hdk::ir::TranslatedJoin
      // which is a binary op, # child nodes for every rel node should be <= 2
      UNREACHABLE();
  }

  // check whether extracted DAG is available to use
  if (dag_extractor.extracted_dag_.empty() || dag_extractor.isDagExtractionAvailable()) {
    return {node, EMPTY_QUERY_PLAN, nullptr, nullptr, {}, {}, true};
  }

  return {node,
          dag_extractor.getExtractedQueryPlanDagStr(),
          dag_extractor.getTranslatedJoinInfo(),
          dag_extractor.getPerNestingJoinQualInfo(left_deep_tree_id),
          dag_extractor.getHashTableBuildDag(),
          dag_extractor.getTableIdToNodeMap(),
          false};
}

std::string QueryPlanDagExtractor::getExtractedQueryPlanDagStr() {
  std::ostringstream oss;
  if (extracted_dag_.empty() || contain_not_supported_rel_node_) {
    oss << "N/A";
  } else {
    for (auto& dag_node_id : extracted_dag_) {
      oss << dag_node_id << "|";
    }
  }
  return oss.str();
}

bool QueryPlanDagExtractor::validateNodeId(const hdk::ir::Node* node,
                                           std::optional<RelNodeId> retrieved_node_id) {
  if (!retrieved_node_id) {
    VLOG(1) << "Stop DAG extraction (Detect an invalid dag id)";
    clearInternaStatus();
    return false;
  }
  CHECK(retrieved_node_id.has_value());
  node->setRelNodeDagId(retrieved_node_id.value());
  return true;
}

bool QueryPlanDagExtractor::registerNodeToDagCache(
    const hdk::ir::Node* parent_node,
    const hdk::ir::Node* child_node,
    std::optional<RelNodeId> retrieved_node_id) {
  CHECK(parent_node);
  CHECK(child_node);
  CHECK(retrieved_node_id.has_value());
  auto parent_node_id = parent_node->getRelNodeDagId();
  global_dag_.connectNodes(parent_node_id, retrieved_node_id.value());
  extracted_dag_.push_back(retrieved_node_id.value());
  return true;
}

// we recursively visit each rel node starting from the root
// and collect assigned rel node ids and return them as query plan DAG
// for join operations we additionally generate additional information
// to recycle each hashtable that needs to process a given query
void QueryPlanDagExtractor::visit(const hdk::ir::Node* parent_node,
                                  const hdk::ir::Node* child_node) {
  if (!child_node || contain_not_supported_rel_node_) {
    return;
  }
  auto register_and_visit = [this](const hdk::ir::Node* parent_node,
                                   const hdk::ir::Node* child_node) {
    // This function takes a responsibility for all rel nodes
    // except 1) RelLeftDeepJoinTree and 2) hdk::ir::TranslatedJoin
    auto res = global_dag_.addNodeIfAbsent(child_node);
    if (validateNodeId(child_node, res) &&
        registerNodeToDagCache(parent_node, child_node, res)) {
      for (size_t i = 0; i < child_node->inputCount(); i++) {
        visit(child_node, child_node->getInput(i));
      }
    }
  };
  if (auto translated_join_node =
          dynamic_cast<const hdk::ir::TranslatedJoin*>(child_node)) {
    handleTranslatedJoin(parent_node, translated_join_node);
  } else {
    register_and_visit(parent_node, child_node);
  }
}

void QueryPlanDagExtractor::handleTranslatedJoin(
    const hdk::ir::Node* parent_node,
    const hdk::ir::TranslatedJoin* rel_trans_join) {
  // when left-deep tree has multiple joins this rel_trans_join can be revisited
  // but we need to mark the child query plan to accurately catch the query plan dag
  // here we do not create new dag id since all rel nodes are visited already
  CHECK(parent_node);
  CHECK(rel_trans_join);

  auto res = global_dag_.addNodeIfAbsent(rel_trans_join);
  if (!validateNodeId(rel_trans_join, res) ||
      !registerNodeToDagCache(parent_node, rel_trans_join, res)) {
    return;
  }

  // To extract an access path (query plan DAG) for hashtable is to use a difference of
  // two query plan DAGs 1) query plan DAG after visiting RHS node and 2) query plan DAG
  // after visiting LHS node so by comparing 1) and 2) we can extract which query plan DAG
  // is necessary to project join cols that are used to build a hashtable and we use it as
  // hashtable access path
  QueryPlan current_plan_dag, after_rhs_visited, after_lhs_visited;
  current_plan_dag = getExtractedQueryPlanDagStr();
  auto rhs_node = rel_trans_join->getRHS();
  if (rhs_node) {
    visit(rel_trans_join, rhs_node);
    after_rhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink(rhs_node->getId(), rhs_node);
  }
  auto lhs_node = rel_trans_join->getLHS();
  if (rel_trans_join->getLHS()) {
    visit(rel_trans_join, lhs_node);
    after_lhs_visited = getExtractedQueryPlanDagStr();
    addTableIdToNodeLink(lhs_node->getId(), lhs_node);
  }
  if (isEmptyQueryPlanDag(after_lhs_visited) || isEmptyQueryPlanDag(after_rhs_visited)) {
    VLOG(1) << "Stop DAG extraction (Detect invalid query plan dag of join col(s))";
    clearInternaStatus();
    return;
  }
  // after visiting new node, we have added node id(s) which can be used as an access path
  // so, we extract that node id(s) by splitting the new plan dag by the current plan dag
  auto outer_table_identifier = split(after_rhs_visited, current_plan_dag)[1];
  auto hash_table_identfier = split(after_lhs_visited, after_rhs_visited)[1];

  if (!rel_trans_join->isNestedLoopQual()) {
    std::ostringstream oss;
    std::vector<std::string> join_cols_info;
    auto inner_join_cols = rel_trans_join->getJoinCols(true);
    auto inner_join_col_info =
        global_dag_.translateColVarsToInfoString(inner_join_cols, false);
    join_cols_info.push_back(inner_join_col_info);
    auto outer_join_cols = rel_trans_join->getJoinCols(false);
    auto outer_join_col_info =
        global_dag_.translateColVarsToInfoString(outer_join_cols, false);
    join_cols_info.push_back(outer_join_col_info);
    auto join_qual_info = boost::join(join_cols_info, "|");
    // hash table join cols info | hash table build plan dag (hashtable identifier or
    // hashtable access path)
    auto it = hash_table_query_plan_dag_.find(join_qual_info);
    if (it == hash_table_query_plan_dag_.end()) {
      VLOG(2) << "Add hashtable access path"
              << ", inner join col info: " << inner_join_col_info
              << " (access path: " << hash_table_identfier << ")"
              << ", outer join col info: " << outer_join_col_info
              << " (access path: " << outer_table_identifier << ")";
      hash_table_query_plan_dag_.emplace(join_qual_info,
                                         HashTableBuildDag(inner_join_col_info,
                                                           outer_join_col_info,
                                                           hash_table_identfier,
                                                           outer_table_identifier));
    }
  } else {
    VLOG(2) << "Add loop join access path, for LHS: " << outer_table_identifier
            << ", for RHS: " << hash_table_identfier << "\n";
  }
}

hdk::ir::ColumnVar const* QueryPlanDagExtractor::getColVar(
    hdk::ir::Expr const* col_info) {
  auto col_var = dynamic_cast<const hdk::ir::ColumnVar*>(col_info);
  if (!col_var) {
    auto visited_cols = global_dag_.collectColVars(col_info);
    if (visited_cols.size() == 1) {
      col_var = dynamic_cast<const hdk::ir::ColumnVar*>(visited_cols[0]);
    }
  }
  return col_var;
}
