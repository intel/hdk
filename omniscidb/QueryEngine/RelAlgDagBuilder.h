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

/** Notes:
 *  * All copy constuctors of child classes of hdk::ir::Node are deep copies,
 *    and are invoked by the the hdk::ir::Node::deepCopy() overloads.
 */

#pragma once

#include <atomic>
#include <iterator>
#include <memory>
#include <unordered_map>

#include <rapidjson/document.h>
#include <boost/core/noncopyable.hpp>
#include <boost/functional/hash.hpp>
#include <boost/variant.hpp>

#include "Analyzer/Analyzer.h"
#include "Descriptors/InputDescriptors.h"
#include "IR/Node.h"
#include "QueryEngine/QueryHint.h"
#include "SchemaMgr/ColumnInfo.h"
#include "SchemaMgr/SchemaProvider.h"
#include "SchemaMgr/TableInfo.h"
#include "Shared/Config.h"
#include "Shared/TypePunning.h"
#include "Shared/toString.h"

/**
 * Builder class to create an in-memory, easy-to-navigate relational algebra DAG
 * interpreted from a JSON representation from Calcite. Also, applies high level
 * optimizations which can be expressed through relational algebra extended with
 * Compound. The Compound node is an equivalent representation for sequences of
 * Filter, Project and Aggregate nodes. This coalescing minimizes the amount of
 * intermediate buffers required to evaluate a query. Lower level optimizations are
 * taken care by lower levels, mainly RelAlgTranslator and the IR code generation.
 */
class RelAlgDagBuilder : public hdk::ir::QueryDag, public boost::noncopyable {
 public:
  RelAlgDagBuilder() = delete;

  /**
   * Constructs a RelAlg DAG from a JSON representation.
   * @param query_ra A JSON string representation of an RA tree from Calcite.
   * @param db_id ID of the current database.
   * @param schema_provider The source of schema information.
   */
  RelAlgDagBuilder(const std::string& query_ra,
                   int db_id,
                   SchemaProviderPtr schema_provider,
                   ConfigPtr config,
                   bool coalesce = true);

  RelAlgDagBuilder(const rapidjson::Value& query_ast,
                   int db_id,
                   SchemaProviderPtr schema_provider,
                   ConfigPtr config,
                   bool coalesce);

  /**
   * Constructs a sub-DAG for any subqueries. Should only be called during DAG
   * building.
   * @param root_dag_builder The root DAG builder. The root stores pointers to all
   * subqueries.
   * @param query_ast The current JSON node to build a DAG for.
   * @param db_id ID of the current database.
   * @param schema_provider The source of schema information.
   */
  RelAlgDagBuilder(RelAlgDagBuilder& root_dag_builder,
                   const rapidjson::Value& query_ast,
                   int db_id,
                   SchemaProviderPtr schema_provider);

  const Config& config() const { return *config_; }

  std::unique_ptr<RelAlgDagBuilder> not_coalesced;

 private:
  void build(const rapidjson::Value& query_ast, RelAlgDagBuilder& root_dag_builder);

  int db_id_;
  SchemaProviderPtr schema_provider_;
  bool coalesce_ = false;
};

std::string tree_string(const hdk::ir::Node*, const size_t depth = 0);

inline InputColDescriptor column_var_to_descriptor(const hdk::ir::ColumnVar* var) {
  return InputColDescriptor(var->columnInfo(), var->rteIdx());
}

hdk::ir::ExprPtrVector getInputExprsForAgg(const hdk::ir::Node* node);

bool hasWindowFunctionExpr(const hdk::ir::Project* node);

void insert_join_projections(std::vector<hdk::ir::NodePtr>& nodes);
