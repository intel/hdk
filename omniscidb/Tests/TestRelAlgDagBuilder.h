/*
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

#include "IR/LeftDeepInnerJoin.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgOptimizer.h"

class TestRelAlgDagBuilder : public hdk::ir::QueryDag {
 public:
  struct AggDesc {
    AggDesc(hdk::ir::AggType agg_,
            bool distinct_,
            const hdk::ir::Type* type_,
            const std::vector<size_t>& operands_)
        : agg(agg_), distinct(distinct_), type(type_), operands(operands_) {}

    AggDesc(hdk::ir::AggType agg_,
            bool distinct_,
            const hdk::ir::Type* type_,
            size_t operand)
        : agg(agg_), distinct(distinct_), type(type_), operands({operand}) {}

    AggDesc(hdk::ir::AggType agg_,
            const hdk::ir::Type* type_,
            const std::vector<size_t>& operands_)
        : agg(agg_), distinct(false), type(type_), operands(operands_) {}

    AggDesc(hdk::ir::AggType agg_, const hdk::ir::Type* type_, size_t operand)
        : agg(agg_), distinct(false), type(type_), operands({operand}) {}

    AggDesc(hdk::ir::AggType agg_)
        : agg(agg_), distinct(false), type(hdk::ir::Context::defaultCtx().int32()) {
      CHECK(agg == hdk::ir::AggType::kCount);
    }

    hdk::ir::AggType agg;
    bool distinct;
    const hdk::ir::Type* type;
    std::vector<size_t> operands;
  };

  TestRelAlgDagBuilder(SchemaProviderPtr schema_provider, ConfigPtr config)
      : hdk::ir::QueryDag(config), schema_provider_(schema_provider) {}
  ~TestRelAlgDagBuilder() override = default;

  void addNode(hdk::ir::NodePtr node);

  hdk::ir::NodePtr addScan(const TableRef& table);
  hdk::ir::NodePtr addScan(int db_id, int table_id);
  hdk::ir::NodePtr addScan(int db_id, const std::string& table_name);

  hdk::ir::NodePtr addProject(hdk::ir::NodePtr input,
                              const std::vector<std::string>& fields,
                              const std::vector<int>& cols);
  hdk::ir::NodePtr addProject(hdk::ir::NodePtr input,
                              const std::vector<std::string>& fields,
                              hdk::ir::ExprPtrVector exprs);

  hdk::ir::NodePtr addProject(hdk::ir::NodePtr input, const std::vector<int>& cols);
  hdk::ir::NodePtr addProject(hdk::ir::NodePtr input, hdk::ir::ExprPtrVector exprs);

  hdk::ir::NodePtr addFilter(hdk::ir::NodePtr input, hdk::ir::ExprPtr expr);

  hdk::ir::NodePtr addAgg(hdk::ir::NodePtr input,
                          const std::vector<std::string>& fields,
                          size_t group_size,
                          hdk::ir::ExprPtrVector aggs);
  hdk::ir::NodePtr addAgg(hdk::ir::NodePtr input,
                          const std::vector<std::string>& fields,
                          size_t group_size,
                          std::vector<AggDesc> aggs);

  hdk::ir::NodePtr addAgg(hdk::ir::NodePtr input,
                          size_t group_size,
                          hdk::ir::ExprPtrVector aggs);
  hdk::ir::NodePtr addAgg(hdk::ir::NodePtr input,
                          size_t group_size,
                          std::vector<AggDesc> aggs);

  hdk::ir::NodePtr addSort(hdk::ir::NodePtr input,
                           const std::vector<hdk::ir::SortField>& collation,
                           const size_t limit = 0,
                           const size_t offset = 0);

  hdk::ir::NodePtr addJoin(hdk::ir::NodePtr lhs,
                           hdk::ir::NodePtr rhs,
                           const JoinType join_type,
                           hdk::ir::ExprPtr condition);

  hdk::ir::NodePtr addEquiJoin(hdk::ir::NodePtr lhs,
                               hdk::ir::NodePtr rhs,
                               const JoinType join_type,
                               size_t lhs_col_idx,
                               size_t rhs_col_idx);

  void setRoot(hdk::ir::NodePtr root);

  void finalize() {
    if (config_->exec.use_legacy_work_unit_builder) {
      hdk::ir::create_left_deep_join(nodes_);
    } else {
      insert_join_projections(nodes_);
      eliminate_dead_columns(nodes_);
    }
    setRoot(nodes_.back());
  }

 private:
  hdk::ir::NodePtr addScan(TableInfoPtr table_info);

  std::vector<std::string> buildFieldNames(size_t count) const;

  SchemaProviderPtr schema_provider_;
};
