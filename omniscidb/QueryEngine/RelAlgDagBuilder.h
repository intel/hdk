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
 *  * All copy constuctors of child classes of RelAlgNode are deep copies,
 *    and are invoked by the the RelAlgNode::deepCopy() overloads.
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
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "QueryEngine/TypePunning.h"
#include "QueryHint.h"
#include "SchemaMgr/ColumnInfo.h"
#include "SchemaMgr/SchemaProvider.h"
#include "SchemaMgr/TableInfo.h"
#include "Shared/Config.h"
#include "Shared/toString.h"

enum class SortDirection { Ascending, Descending };

enum class NullSortedPosition { First, Last };

class SortField {
 public:
  SortField(const size_t field,
            const SortDirection sort_dir,
            const NullSortedPosition nulls_pos)
      : field_(field), sort_dir_(sort_dir), nulls_pos_(nulls_pos) {}

  bool operator==(const SortField& that) const {
    return field_ == that.field_ && sort_dir_ == that.sort_dir_ &&
           nulls_pos_ == that.nulls_pos_;
  }

  size_t getField() const { return field_; }

  SortDirection getSortDir() const { return sort_dir_; }

  NullSortedPosition getNullsPosition() const { return nulls_pos_; }

  std::string toString() const {
    return cat(::typeName(this),
               "(",
               std::to_string(field_),
               ", sort_dir=",
               (sort_dir_ == SortDirection::Ascending ? "asc" : "desc"),
               ", null_pos=",
               (nulls_pos_ == NullSortedPosition::First ? "nulls_first" : "nulls_last"),
               ")");
  }

  size_t toHash() const {
    auto hash = boost::hash_value(field_);
    boost::hash_combine(hash, sort_dir_ == SortDirection::Ascending ? "a" : "d");
    boost::hash_combine(hash, nulls_pos_ == NullSortedPosition::First ? "f" : "l");
    return hash;
  }

 private:
  const size_t field_;
  const SortDirection sort_dir_;
  const NullSortedPosition nulls_pos_;
};

using RelAlgInputs = std::vector<std::shared_ptr<const RelAlgNode>>;

class RaExecutionDesc;
class ExecutionResult;

class RelAlgNode {
 public:
  RelAlgNode(RelAlgInputs inputs = {})
      : inputs_(std::move(inputs))
      , id_(crt_id_++)
      , context_data_(nullptr)
      , is_nop_(false) {}

  virtual ~RelAlgNode() {}

  void resetQueryExecutionState() {
    context_data_ = nullptr;
    targets_metainfo_ = {};
  }

  void setContextData(const RaExecutionDesc* context_data) const {
    CHECK(!context_data_);
    context_data_ = context_data;
  }

  void setOutputMetainfo(const std::vector<TargetMetaInfo>& targets_metainfo) const {
    targets_metainfo_ = targets_metainfo;
  }

  const std::vector<TargetMetaInfo>& getOutputMetainfo() const {
    return targets_metainfo_;
  }

  unsigned getId() const { return id_; }

  std::string getIdString() const { return "#" + std::to_string(id_); }

  bool hasContextData() const { return !(context_data_ == nullptr); }

  const RaExecutionDesc* getContextData() const { return context_data_; }

  const size_t inputCount() const { return inputs_.size(); }

  const RelAlgNode* getInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx].get();
  }

  std::shared_ptr<const RelAlgNode> getAndOwnInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx];
  }

  void addManagedInput(std::shared_ptr<const RelAlgNode> input) {
    inputs_.push_back(input);
  }

  bool hasInput(const RelAlgNode* needle) const {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == needle) {
        return true;
      }
    }
    return false;
  }

  virtual void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                            std::shared_ptr<const RelAlgNode> input) {
    for (auto& input_ptr : inputs_) {
      if (input_ptr == old_input) {
        input_ptr = input;
        break;
      }
    }
  }

  // to keep an assigned DAG node id for data recycler
  void setRelNodeDagId(const size_t id) const { dag_node_id_ = id; }

  size_t getRelNodeDagId() const { return dag_node_id_; }

  bool isNop() const { return is_nop_; }

  void markAsNop() { is_nop_ = true; }

  virtual std::string toString() const = 0;

  void print() const;

  // return hashed value of a string representation of this rel node
  virtual size_t toHash() const = 0;

  virtual size_t size() const = 0;

  virtual std::shared_ptr<RelAlgNode> deepCopy() const = 0;

  static void resetRelAlgFirstId() noexcept;

  /**
   * Clears the ptr to the result for this descriptor. Is only used for overriding step
   * results in distributed mode.
   */
  void clearContextData() const { context_data_ = nullptr; }

  std::shared_ptr<const ExecutionResult> getResult() const { return result_; }

  void setResult(std::shared_ptr<const ExecutionResult> result) const {
    result_ = result;
  }

 protected:
  RelAlgInputs inputs_;
  const unsigned id_;
  mutable std::optional<size_t> hash_;

 private:
  mutable const RaExecutionDesc* context_data_;
  bool is_nop_;
  mutable std::vector<TargetMetaInfo> targets_metainfo_;
  static thread_local unsigned crt_id_;
  mutable size_t dag_node_id_;
  mutable std::shared_ptr<const ExecutionResult> result_;
};

inline std::string inputsToString(const RelAlgInputs& inputs) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i) {
      ss << ", ";
    }
    ss << inputs[i]->getIdString();
  }
  ss << "]";
  return ss.str();
}

using RelAlgNodePtr = std::shared_ptr<RelAlgNode>;

class RelScan : public RelAlgNode {
 public:
  RelScan(TableInfoPtr tinfo, std::vector<ColumnInfoPtr> column_infos)
      : table_info_(std::move(tinfo))
      , column_infos_(std::move(column_infos))
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {}

  size_t size() const override { return column_infos_.size(); }

  TableInfoPtr getTableInfo() const { return table_info_; }

  const size_t getNumFragments() const { return table_info_->fragments; }

  const std::string& getFieldName(const size_t i) const { return column_infos_[i]->name; }

  int32_t getDatabaseId() const { return table_info_->db_id; }

  int32_t getTableId() const { return table_info_->table_id; }

  bool isVirtualCol(int col_idx) const {
    CHECK_LT(static_cast<size_t>(col_idx), column_infos_.size());
    return column_infos_[col_idx]->is_rowid;
  }

  const hdk::ir::Type* getColumnType(int col_idx) const {
    CHECK_LT(static_cast<size_t>(col_idx), column_infos_.size());
    return column_infos_[col_idx]->type;
  }

  ColumnInfoPtr getColumnInfo(int col_idx) const {
    CHECK_LT(static_cast<size_t>(col_idx), column_infos_.size());
    return column_infos_[col_idx];
  }

  std::string toString() const override {
    std::vector<std::string_view> field_names;
    field_names.reserve(column_infos_.size());
    for (auto& info : column_infos_) {
      field_names.emplace_back(info->name);
    }
    return cat(::typeName(this),
               getIdString(),
               "(",
               table_info_->name,
               ", ",
               ::toString(field_names),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelScan).hash_code();
      boost::hash_combine(*hash_, table_info_->table_id);
      boost::hash_combine(*hash_, table_info_->name);
      for (auto& info : column_infos_) {
        boost::hash_combine(*hash_, info->name);
      }
    }
    return *hash_;
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  };

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(const QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

 private:
  TableInfoPtr table_info_;
  const std::vector<ColumnInfoPtr> column_infos_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelProject : public RelAlgNode {
 public:
  // Takes memory ownership of the expressions.
  RelProject(hdk::ir::ExprPtrVector exprs,
             const std::vector<std::string>& fields,
             std::shared_ptr<const RelAlgNode> input)
      : exprs_(std::move(exprs))
      , fields_(fields)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    inputs_.push_back(input);
  }

  RelProject(RelProject const&);

  void setExpressions(hdk::ir::ExprPtrVector exprs) const { exprs_ = std::move(exprs); }

  // True iff all the projected expressions are inputs. If true,
  // this node can be elided and merged into the previous node
  // since it's just a subset and / or permutation of its outputs.
  bool isSimple() const {
    for (const auto& expr : exprs_) {
      if (!dynamic_cast<const hdk::ir::ColumnRef*>(expr.get())) {
        return false;
      }
    }
    return true;
  }

  bool isIdentity() const;

  bool isRenaming() const;

  size_t size() const override { return exprs_.size(); }

  hdk::ir::ExprPtr getExpr(size_t idx) const { return exprs_[idx]; }

  const hdk::ir::ExprPtrVector& getExprs() const { return exprs_; }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override {
    replaceInput(old_input, input, std::nullopt);
  }

  void replaceInput(
      std::shared_ptr<const RelAlgNode> old_input,
      std::shared_ptr<const RelAlgNode> input,
      std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map);

  void appendInput(std::string new_field_name, hdk::ir::ExprPtr expr);

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               ::toString(exprs_),
               ", ",
               ::toString(fields_),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelProject).hash_code();
      for (auto& expr : exprs_) {
        boost::hash_combine(*hash_, expr->hash());
      }
      boost::hash_combine(*hash_, ::toString(fields_));
    }
    return *hash_;
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelProject>(*this);
  }

  bool hasWindowFunctionExpr() const;

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

 private:
  mutable hdk::ir::ExprPtrVector exprs_;
  mutable std::vector<std::string> fields_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelAggregate : public RelAlgNode {
 public:
  // Takes ownership of the aggregate expressions.
  RelAggregate(const size_t groupby_count,
               hdk::ir::ExprPtrVector aggs,
               const std::vector<std::string>& fields,
               std::shared_ptr<const RelAlgNode> input)
      : groupby_count_(groupby_count)
      , aggs_(aggs)
      , fields_(fields)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    inputs_.push_back(input);
  }

  RelAggregate(RelAggregate const&);

  size_t size() const override { return groupby_count_ + aggs_.size(); }

  const size_t getGroupByCount() const { return groupby_count_; }

  const size_t getAggsCount() const { return aggs_.size(); }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& new_fields) {
    fields_ = std::move(new_fields);
  }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  // TODO: rename to getAggExprs when Rex version is removed.
  const hdk::ir::ExprPtrVector& getAggs() const { return aggs_; }
  hdk::ir::ExprPtr getAgg(size_t i) const { return aggs_[i]; }

  void setAggExprs(hdk::ir::ExprPtrVector new_aggs) { aggs_ = std::move(new_aggs); }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               std::to_string(groupby_count_),
               ", aggs=",
               ::toString(aggs_),
               ", fields=",
               ::toString(fields_),
               ", inputs=",
               inputsToString(inputs_),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelAggregate).hash_code();
      boost::hash_combine(*hash_, groupby_count_);
      for (auto& agg : aggs_) {
        boost::hash_combine(*hash_, agg->hash());
      }
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
      boost::hash_combine(*hash_, ::toString(fields_));
    }
    return *hash_;
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelAggregate>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

 private:
  const size_t groupby_count_;
  hdk::ir::ExprPtrVector aggs_;
  std::vector<std::string> fields_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelJoin : public RelAlgNode {
 public:
  RelJoin(std::shared_ptr<const RelAlgNode> lhs,
          std::shared_ptr<const RelAlgNode> rhs,
          hdk::ir::ExprPtr condition,
          const JoinType join_type)
      : condition_(std::move(condition))
      , join_type_(join_type)
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    inputs_.push_back(lhs);
    inputs_.push_back(rhs);
  }

  RelJoin(RelJoin const&);

  JoinType getJoinType() const { return join_type_; }

  const hdk::ir::Expr* getCondition() const { return condition_.get(); }
  hdk::ir::ExprPtr getConditionShared() const { return condition_; }

  void setCondition(hdk::ir::ExprPtr new_condition) {
    condition_ = std::move(new_condition);
  }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               ::inputsToString(inputs_),
               ", condition=",
               (condition_ ? condition_->toString() : "null"),
               ", join_type=",
               ::toString(join_type_));
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelJoin).hash_code();
      boost::hash_combine(*hash_,
                          condition_ ? condition_->hash() : boost::hash_value("n"));
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
      boost::hash_combine(*hash_, ::toString(getJoinType()));
    }
    return *hash_;
  }

  size_t size() const override { return inputs_[0]->size() + inputs_[1]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelJoin>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

 private:
  hdk::ir::ExprPtr condition_;
  const JoinType join_type_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

// a helper node that contains detailed information of each level of join qual
// which is used when extracting query plan DAG
class RelTranslatedJoin : public RelAlgNode {
 public:
  RelTranslatedJoin(const RelAlgNode* lhs,
                    const RelAlgNode* rhs,
                    std::vector<const hdk::ir::ColumnVar*> lhs_join_cols,
                    std::vector<const hdk::ir::ColumnVar*> rhs_join_cols,
                    hdk::ir::ExprPtrVector filter_ops,
                    hdk::ir::ExprPtr outer_join_cond,
                    const bool nested_loop,
                    const JoinType join_type,
                    const std::string& op_type,
                    const std::string& qualifier,
                    const std::string& op_typeinfo)
      : lhs_(lhs)
      , rhs_(rhs)
      , lhs_join_cols_(std::move(lhs_join_cols))
      , rhs_join_cols_(std::move(rhs_join_cols))
      , filter_ops_(std::move(filter_ops))
      , outer_join_cond_(std::move(outer_join_cond))
      , nested_loop_(nested_loop)
      , join_type_(join_type)
      , op_type_(op_type)
      , qualifier_(qualifier)
      , op_typeinfo_(op_typeinfo) {
    CHECK_EQ(lhs_join_cols_.size(), rhs_join_cols_.size());
  }

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "( join_quals { lhs: ",
               ::toString(lhs_join_cols_),
               ", rhs: ",
               ::toString(rhs_join_cols_),
               " }, filter_quals: { ",
               ::toString(filter_ops_),
               " }, outer_join_cond: { ",
               ::toString(outer_join_cond_),
               " }, loop_join: ",
               ::toString(nested_loop_),
               ", join_type: ",
               ::toString(join_type_),
               ", op_type: ",
               ::toString(op_type_),
               ", qualifier: ",
               ::toString(qualifier_),
               ", op_type_info: ",
               ::toString(op_typeinfo_),
               ")");
  }
  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelTranslatedJoin).hash_code();
      boost::hash_combine(*hash_, lhs_->toHash());
      boost::hash_combine(*hash_, rhs_->toHash());
      boost::hash_combine(
          *hash_, outer_join_cond_ ? outer_join_cond_->hash() : boost::hash_value("n"));
      boost::hash_combine(*hash_, nested_loop_);
      boost::hash_combine(*hash_, ::toString(join_type_));
      boost::hash_combine(*hash_, op_type_);
      boost::hash_combine(*hash_, qualifier_);
      boost::hash_combine(*hash_, op_typeinfo_);
      for (auto& filter_op : filter_ops_) {
        boost::hash_combine(*hash_, filter_op->toString());
      }
    }
    return *hash_;
  }
  const RelAlgNode* getLHS() const { return lhs_; }
  const RelAlgNode* getRHS() const { return rhs_; }
  size_t getFilterCondSize() const { return filter_ops_.size(); }
  const std::vector<hdk::ir::ExprPtr>& getFilterCond() const { return filter_ops_; }
  const hdk::ir::Expr* getOuterJoinCond() const { return outer_join_cond_.get(); }
  std::string getOpType() const { return op_type_; }
  std::string getQualifier() const { return qualifier_; }
  std::string getOpTypeInfo() const { return op_typeinfo_; }
  size_t size() const override { return 0; }
  JoinType getJoinType() const { return join_type_; }
  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override {
    CHECK(false);
  }
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    CHECK(false);
    return nullptr;
  }
  std::string getFieldName(const size_t i) const;
  std::vector<const hdk::ir::ColumnVar*> getJoinCols(bool lhs) const {
    if (lhs) {
      return lhs_join_cols_;
    }
    return rhs_join_cols_;
  }
  bool isNestedLoopQual() const { return nested_loop_; }

 private:
  const RelAlgNode* lhs_;
  const RelAlgNode* rhs_;
  const std::vector<const hdk::ir::ColumnVar*> lhs_join_cols_;
  const std::vector<const hdk::ir::ColumnVar*> rhs_join_cols_;
  const std::vector<hdk::ir::ExprPtr> filter_ops_;
  hdk::ir::ExprPtr outer_join_cond_;
  const bool nested_loop_;
  const JoinType join_type_;
  const std::string op_type_;
  const std::string qualifier_;
  const std::string op_typeinfo_;
};

class RelFilter : public RelAlgNode {
 public:
  RelFilter(hdk::ir::ExprPtr condition, std::shared_ptr<const RelAlgNode> input)
      : condition_(std::move(condition)) {
    CHECK(condition_);
    inputs_.push_back(input);
  }

  // for dummy filter node for data recycler
  RelFilter(hdk::ir::ExprPtr condition) : condition_(std::move(condition)) {
    CHECK(condition_);
  }

  RelFilter(RelFilter const&);

  const hdk::ir::Expr* getConditionExpr() const { return condition_.get(); }
  hdk::ir::ExprPtr getConditionExprShared() const { return condition_; }

  void setCondition(hdk::ir::ExprPtr new_condition) {
    CHECK(new_condition);
    condition_ = std::move(new_condition);
  }

  size_t size() const override { return inputs_[0]->size(); }

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               condition_->toString(),
               ", ",
               ::inputsToString(inputs_) + ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelFilter).hash_code();
      boost::hash_combine(*hash_, condition_->hash());
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelFilter>(*this);
  }

 private:
  hdk::ir::ExprPtr condition_;
};

// Synthetic node to assist execution of left-deep join relational algebra.
class RelLeftDeepInnerJoin : public RelAlgNode {
 public:
  RelLeftDeepInnerJoin(const std::shared_ptr<RelFilter>& filter,
                       RelAlgInputs inputs,
                       std::vector<std::shared_ptr<const RelJoin>>& original_joins);

  const hdk::ir::Expr* getInnerCondition() const;
  hdk::ir::ExprPtr getInnerConditionShared() const;

  const hdk::ir::Expr* getOuterCondition(const size_t nesting_level) const;
  hdk::ir::ExprPtr getOuterConditionShared(const size_t nesting_level) const;

  const JoinType getJoinType(const size_t nesting_level) const;

  std::string toString() const override;

  size_t toHash() const override;

  size_t size() const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override;

  bool coversOriginalNode(const RelAlgNode* node) const;

  const RelFilter* getOriginalFilter() const;

  std::vector<std::shared_ptr<const RelJoin>> getOriginalJoins() const;

 private:
  hdk::ir::ExprPtr condition_;
  std::vector<hdk::ir::ExprPtr> outer_conditions_per_level_;
  const std::shared_ptr<RelFilter> original_filter_;
  const std::vector<std::shared_ptr<const RelJoin>> original_joins_;
};

// The 'RelCompound' node combines filter and on the fly aggregate computation.
// It's the result of combining a sequence of 'RelFilter' (optional), 'RelProject',
// 'RelAggregate' (optional) and a simple 'RelProject' (optional) into a single node
// which can be efficiently executed with no intermediate buffers.
class RelCompound : public RelAlgNode {
 public:
  RelCompound(hdk::ir::ExprPtr filter,
              hdk::ir::ExprPtrVector exprs,
              const size_t groupby_count,
              hdk::ir::ExprPtrVector groupby_exprs,
              const std::vector<std::string>& fields,
              const bool is_agg)
      : filter_(std::move(filter))
      , groupby_count_(groupby_count)
      , fields_(fields)
      , is_agg_(is_agg)
      , groupby_exprs_(std::move(groupby_exprs))
      , exprs_(std::move(exprs))
      , hint_applied_(false)
      , hints_(std::make_unique<Hints>()) {
    CHECK_EQ(fields.size(), exprs_.size());
  }

  RelCompound(RelCompound const&);

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  size_t size() const override { return exprs_.size(); }

  hdk::ir::ExprPtr getFilter() const { return filter_; }

  void setFilterExpr(hdk::ir::ExprPtr new_filter) { filter_ = new_filter; }

  hdk::ir::ExprPtrVector getGroupByExprs() const { return groupby_exprs_; }
  hdk::ir::ExprPtr getGroupByExpr(size_t i) const { return groupby_exprs_[i]; }

  hdk::ir::ExprPtrVector getExprs() const { return exprs_; }
  hdk::ir::ExprPtr getExpr(size_t i) const { return exprs_[i]; }

  const std::vector<std::string>& getFields() const { return fields_; }

  const std::string getFieldName(const size_t i) const { return fields_[i]; }

  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  const size_t getGroupByCount() const { return groupby_count_; }

  bool isAggregate() const { return is_agg_; }

  std::string toString() const override;

  size_t toHash() const override;

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelCompound>(*this);
  }

  void addHint(const ExplainedQueryHint& hint_explained) {
    if (!hint_applied_) {
      hint_applied_ = true;
    }
    hints_->emplace(hint_explained.getHint(), hint_explained);
  }

  const bool hasHintEnabled(QueryHint candidate_hint) const {
    if (hint_applied_ && !hints_->empty()) {
      return hints_->find(candidate_hint) != hints_->end();
    }
    return false;
  }

  const ExplainedQueryHint& getHintInfo(QueryHint hint) const {
    CHECK(hint_applied_);
    CHECK(!hints_->empty());
    CHECK(hasHintEnabled(hint));
    return hints_->at(hint);
  }

  bool hasDeliveredHint() { return !hints_->empty(); }

  Hints* getDeliveredHints() { return hints_.get(); }

 private:
  hdk::ir::ExprPtr filter_;
  const size_t groupby_count_;
  std::vector<std::string> fields_;
  const bool is_agg_;
  hdk::ir::ExprPtrVector groupby_exprs_;
  hdk::ir::ExprPtrVector exprs_;
  bool hint_applied_;
  std::unique_ptr<Hints> hints_;
};

class RelSort : public RelAlgNode {
 public:
  RelSort(const std::vector<SortField>& collation,
          const size_t limit,
          const size_t offset,
          std::shared_ptr<const RelAlgNode> input)
      : collation_(collation), limit_(limit), offset_(offset), empty_result_(false) {
    inputs_.push_back(input);
  }

  bool operator==(const RelSort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ &&
           empty_result_ == that.empty_result_ && hasEquivCollationOf(that);
  }

  size_t collationCount() const { return collation_.size(); }

  SortField getCollation(const size_t i) const {
    CHECK_LT(i, collation_.size());
    return collation_[i];
  }

  void setCollation(std::vector<SortField>&& collation) {
    collation_ = std::move(collation);
  }

  void setEmptyResult(bool emptyResult) { empty_result_ = emptyResult; }

  bool isEmptyResult() const { return empty_result_; }

  size_t getLimit() const { return limit_; }

  size_t getOffset() const { return offset_; }

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               "empty_result: ",
               ::toString(empty_result_),
               ", collation=",
               ::toString(collation_),
               ", limit=",
               std::to_string(limit_),
               ", offset",
               std::to_string(offset_),
               ", inputs=",
               inputsToString(inputs_),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelSort).hash_code();
      for (auto& collation : collation_) {
        boost::hash_combine(*hash_, collation.toHash());
      }
      boost::hash_combine(*hash_, empty_result_);
      boost::hash_combine(*hash_, limit_);
      boost::hash_combine(*hash_, offset_);
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  size_t size() const override { return inputs_[0]->size(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelSort>(*this);
  }

 private:
  std::vector<SortField> collation_;
  const size_t limit_;
  const size_t offset_;
  bool empty_result_;

  bool hasEquivCollationOf(const RelSort& that) const;
};

class RelTableFunction : public RelAlgNode {
 public:
  RelTableFunction(const std::string& function_name,
                   RelAlgInputs inputs,
                   std::vector<std::string>& fields,
                   hdk::ir::ExprPtrVector col_input_exprs,
                   hdk::ir::ExprPtrVector table_func_input_exprs,
                   std::vector<TargetMetaInfo> tuple_type)
      : RelAlgNode(std::move(inputs))
      , function_name_(function_name)
      , fields_(fields)
      , col_input_exprs_(col_input_exprs)
      , table_func_input_exprs_(table_func_input_exprs)
      , tuple_type_(std::move(tuple_type)) {}

  RelTableFunction(RelTableFunction const&);

  void replaceInput(std::shared_ptr<const RelAlgNode> old_input,
                    std::shared_ptr<const RelAlgNode> input) override;

  std::string getFunctionName() const { return function_name_; }

  size_t size() const override { return tuple_type_.size(); }

  const std::vector<TargetMetaInfo>& getTupleType() const { return tuple_type_; }

  size_t getTableFuncInputsSize() const { return table_func_input_exprs_.size(); }

  size_t getColInputsSize() const { return col_input_exprs_.size(); }

  int32_t countConstantArgs() const;

  const hdk::ir::Expr* getTableFuncInputExprAt(size_t idx) const {
    CHECK_LT(idx, table_func_input_exprs_.size());
    return table_func_input_exprs_[idx].get();
  }

  const hdk::ir::ExprPtrVector& getTableFuncInputExprs() const {
    return table_func_input_exprs_;
  }

  void setTableFuncInputs(hdk::ir::ExprPtrVector new_inputs) {
    table_func_input_exprs_ = std::move(new_inputs);
  }

  std::string getFieldName(const size_t idx) const {
    CHECK_LT(idx, fields_.size());
    return fields_[idx];
  }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string>&& fields) { fields_ = std::move(fields); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelTableFunction>(*this);
  }

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               function_name_,
               ", inputs=",
               inputsToString(inputs_),
               ", fields=",
               ::toString(fields_),
               ", table_func_input_exprs=",
               ::toString(table_func_input_exprs_),
               ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelTableFunction).hash_code();
      for (auto& table_func_input : table_func_input_exprs_) {
        boost::hash_combine(*hash_, table_func_input->hash());
      }
      boost::hash_combine(*hash_, function_name_);
      boost::hash_combine(*hash_, ::toString(fields_));
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

 private:
  std::string function_name_;
  std::vector<std::string> fields_;

  hdk::ir::ExprPtrVector col_input_exprs_;
  hdk::ir::ExprPtrVector table_func_input_exprs_;

  const std::vector<TargetMetaInfo> tuple_type_;
};

class RelLogicalValues : public RelAlgNode {
 public:
  RelLogicalValues(const std::vector<TargetMetaInfo>& tuple_type,
                   std::vector<hdk::ir::ExprPtrVector> values)
      : tuple_type_(tuple_type), values_(std::move(values)) {}

  RelLogicalValues(RelLogicalValues const&);

  const std::vector<TargetMetaInfo>& getTupleType() const { return tuple_type_; }

  std::string toString() const override {
    std::string ret = ::typeName(this) + getIdString() + "(";
    for (const auto& target_meta_info : tuple_type_) {
      ret += " (" + target_meta_info.get_resname() + " " +
             target_meta_info.type()->toString() + ")";
    }
    ret += ")";
    return ret;
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(RelLogicalValues).hash_code();
      for (auto& target_meta_info : tuple_type_) {
        boost::hash_combine(*hash_, target_meta_info.get_resname());
        boost::hash_combine(*hash_, target_meta_info.type()->toString());
      }
    }
    return *hash_;
  }

  const hdk::ir::Expr* getValue(const size_t row_idx, const size_t col_idx) const {
    CHECK_LT(row_idx, values_.size());
    const auto& row = values_[row_idx];
    CHECK_LT(col_idx, row.size());
    return row[col_idx].get();
  }

  size_t getRowsSize() const {
    if (values_.empty()) {
      return 0;
    } else {
      return values_.front().size();
    }
  }

  size_t getNumRows() const { return values_.size(); }

  size_t size() const override { return tuple_type_.size(); }

  bool hasRows() const { return !values_.empty(); }

  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalValues>(*this);
  }

 private:
  std::vector<TargetMetaInfo> tuple_type_;
  std::vector<hdk::ir::ExprPtrVector> values_;
};

class RelLogicalUnion : public RelAlgNode {
 public:
  RelLogicalUnion(RelAlgInputs, bool is_all);
  std::shared_ptr<RelAlgNode> deepCopy() const override {
    return std::make_shared<RelLogicalUnion>(*this);
  }
  size_t size() const override;
  std::string toString() const override;
  size_t toHash() const override;

  std::string getFieldName(const size_t i) const;

  inline bool isAll() const { return is_all_; }
  // Will throw a std::runtime_error if MetaInfo types don't match.
  void checkForMatchingMetaInfoTypes() const;

 private:
  bool const is_all_;
};

class QueryNotSupported : public std::runtime_error {
 public:
  QueryNotSupported(const std::string& reason) : std::runtime_error(reason) {}
};

class RelAlgDag {
 public:
  RelAlgDag(ConfigPtr config) : config_(config) { time(&now_); }
  RelAlgDag(ConfigPtr config, time_t now) : config_(config), now_(now) {}
  virtual ~RelAlgDag() = default;

  void eachNode(std::function<void(RelAlgNode const*)> const& callback) const;

  /**
   * Returns the root node of the DAG.
   */
  const RelAlgNode& getRootNode() const {
    CHECK(root_);
    return *root_;
  }

  std::shared_ptr<const RelAlgNode> getRootNodeShPtr() const { return root_; }

  /**
   * Registers a subquery with a root DAG builder. Should only be called during DAG
   * building and registration should only occur on the root.
   */
  void registerSubquery(std::shared_ptr<const hdk::ir::ScalarSubquery> subquery_expr) {
    subqueries_.emplace_back(std::move(subquery_expr));
  }

  void registerQueryHints(RelAlgNodePtr node, Hints* hints_delivered);

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  const std::vector<std::shared_ptr<const hdk::ir::ScalarSubquery>>& getSubqueries()
      const {
    return subqueries_;
  }

  std::optional<RegisteredQueryHint> getQueryHint(const RelAlgNode* node) const {
    auto it = query_hint_.find(node->toHash());
    return it != query_hint_.end() ? std::make_optional(it->second) : std::nullopt;
  }

  const std::unordered_map<size_t, RegisteredQueryHint>& getQueryHints() const {
    return query_hint_;
  }

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  void resetQueryExecutionState();

  time_t now() const { return now_; }

 protected:
  ConfigPtr config_;
  time_t now_;
  // Root node of the query.
  RelAlgNodePtr root_;
  // All nodes including the root one.
  std::vector<RelAlgNodePtr> nodes_;
  std::vector<std::shared_ptr<const hdk::ir::ScalarSubquery>> subqueries_;
  std::unordered_map<size_t, RegisteredQueryHint> query_hint_;
};

/**
 * Builder class to create an in-memory, easy-to-navigate relational algebra DAG
 * interpreted from a JSON representation from Calcite. Also, applies high level
 * optimizations which can be expressed through relational algebra extended with
 * RelCompound. The RelCompound node is an equivalent representation for sequences of
 * RelFilter, RelProject and RelAggregate nodes. This coalescing minimizes the amount of
 * intermediate buffers required to evaluate a query. Lower level optimizations are
 * taken care by lower levels, mainly RelAlgTranslator and the IR code generation.
 */
class RelAlgDagBuilder : public RelAlgDag, public boost::noncopyable {
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
                   ConfigPtr config);

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

 private:
  void build(const rapidjson::Value& query_ast, RelAlgDagBuilder& root_dag_builder);

  int db_id_;
  SchemaProviderPtr schema_provider_;
};

std::string tree_string(const RelAlgNode*, const size_t depth = 0);

inline InputColDescriptor column_var_to_descriptor(const hdk::ir::ColumnVar* var) {
  return InputColDescriptor(var->columnInfo(), var->rteIdx());
}

const hdk::ir::Type* getColumnType(const RelAlgNode* node, size_t col_idx);

hdk::ir::ExprPtr getNodeColumnRef(const RelAlgNode* node, unsigned index);
hdk::ir::ExprPtrVector getNodeColumnRefs(const RelAlgNode* node);
size_t getNodeColumnCount(const RelAlgNode* node);

hdk::ir::ExprPtrVector getInputExprsForAgg(const RelAlgNode* node);
