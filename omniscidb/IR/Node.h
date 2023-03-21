/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Expr.h"

#include "QueryEngine/TargetMetaInfo.h"
#include "SchemaMgr/TableInfo.h"
#include "Shared/Config.h"

class RaExecutionDesc;
class ExecutionResult;

namespace hdk::ir {

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

using NodeInputs = std::vector<std::shared_ptr<const Node>>;

class Node {
 public:
  Node(NodeInputs inputs = {});

  virtual ~Node() {}

  template <typename T>
  bool is() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }

  template <typename T>
  const T* as() const {
    return dynamic_cast<const T*>(this);
  }

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

  const Node* getInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx].get();
  }

  std::shared_ptr<const Node> getAndOwnInput(const size_t idx) const {
    CHECK_LT(idx, inputs_.size());
    return inputs_[idx];
  }

  void addManagedInput(std::shared_ptr<const Node> input) { inputs_.push_back(input); }

  bool hasInput(const Node* needle) const {
    for (auto& input_ptr : inputs_) {
      if (input_ptr.get() == needle) {
        return true;
      }
    }
    return false;
  }

  virtual void replaceInput(std::shared_ptr<const Node> old_input,
                            std::shared_ptr<const Node> input) {
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

  virtual std::shared_ptr<Node> deepCopy() const = 0;

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
  NodeInputs inputs_;
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

inline std::string inputsToString(const NodeInputs& inputs) {
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

using NodePtr = std::shared_ptr<Node>;

class Scan : public Node {
 public:
  Scan(TableInfoPtr tinfo, std::vector<ColumnInfoPtr> column_infos)
      : table_info_(std::move(tinfo)), column_infos_(std::move(column_infos)) {}

  size_t size() const override { return column_infos_.size(); }

  TableInfoPtr getTableInfo() const { return table_info_; }

  const size_t getNumFragments() const { return table_info_->fragments; }

  const std::string& getFieldName(const size_t i) const {
    CHECK_LT(i, column_infos_.size());
    return column_infos_[i]->name;
  }

  int32_t getDatabaseId() const { return table_info_->db_id; }

  int32_t getTableId() const { return table_info_->table_id; }

  bool isVirtualCol(int col_idx) const {
    CHECK_LT(static_cast<size_t>(col_idx), column_infos_.size());
    return column_infos_[col_idx]->is_rowid;
  }

  const Type* getColumnType(int col_idx) const {
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
      hash_ = typeid(Scan).hash_code();
      boost::hash_combine(*hash_, table_info_->table_id);
      boost::hash_combine(*hash_, table_info_->name);
      for (auto& info : column_infos_) {
        boost::hash_combine(*hash_, info->name);
      }
    }
    return *hash_;
  }

  std::shared_ptr<Node> deepCopy() const override {
    CHECK(false);
    return nullptr;
  };

 private:
  TableInfoPtr table_info_;
  const std::vector<ColumnInfoPtr> column_infos_;
};

class Project : public Node {
 public:
  // Takes memory ownership of the expressions.
  Project(ExprPtrVector exprs,
          std::vector<std::string> fields,
          std::shared_ptr<const Node> input)
      : exprs_(std::move(exprs)), fields_(std::move(fields)) {
    inputs_.push_back(input);
  }

  Project(Project const&);

  void setExpressions(ExprPtrVector exprs) const { exprs_ = std::move(exprs); }

  // True iff all the projected expressions are inputs. If true,
  // this node can be elided and merged into the previous node
  // since it's just a subset and / or permutation of its outputs.
  bool isSimple() const {
    for (const auto& expr : exprs_) {
      if (!dynamic_cast<const ColumnRef*>(expr.get())) {
        return false;
      }
    }
    return true;
  }

  bool isIdentity() const;

  bool isRenaming() const;

  size_t size() const override { return exprs_.size(); }

  ExprPtr getExpr(size_t idx) const {
    CHECK_LT(idx, exprs_.size());
    return exprs_[idx];
  }

  const ExprPtrVector& getExprs() const { return exprs_; }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string> fields) { fields_ = std::move(fields); }

  const std::string getFieldName(const size_t i) const {
    CHECK_LT(i, fields_.size());
    return fields_[i];
  }

  void replaceInput(std::shared_ptr<const Node> old_input,
                    std::shared_ptr<const Node> input) override {
    replaceInput(old_input, input, std::nullopt);
  }

  void replaceInput(
      std::shared_ptr<const Node> old_input,
      std::shared_ptr<const Node> input,
      std::optional<std::unordered_map<unsigned, unsigned>> old_to_new_index_map);

  void appendInput(std::string new_field_name, ExprPtr expr);

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
      hash_ = typeid(Project).hash_code();
      for (auto& expr : exprs_) {
        boost::hash_combine(*hash_, expr->hash());
      }
      boost::hash_combine(*hash_, ::toString(fields_));
    }
    return *hash_;
  }

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<Project>(*this);
  }

 private:
  mutable ExprPtrVector exprs_;
  mutable std::vector<std::string> fields_;
};

class Aggregate : public Node {
 public:
  // Takes ownership of the aggregate expressions.
  Aggregate(const size_t groupby_count,
            ExprPtrVector aggs,
            std::vector<std::string> fields,
            std::shared_ptr<const Node> input)
      : groupby_count_(groupby_count)
      , aggs_(std::move(aggs))
      , fields_(std::move(fields)) {
    inputs_.emplace_back(std::move(input));
  }

  Aggregate(Aggregate const&);

  size_t size() const override { return groupby_count_ + aggs_.size(); }

  const size_t getGroupByCount() const { return groupby_count_; }

  const size_t getAggsCount() const { return aggs_.size(); }

  const std::vector<std::string>& getFields() const { return fields_; }
  void setFields(std::vector<std::string> new_fields) { fields_ = std::move(new_fields); }

  const std::string getFieldName(const size_t i) const {
    CHECK_LT(i, fields_.size());
    return fields_[i];
  }

  // TODO: rename to getAggExprs when Rex version is removed.
  const ExprPtrVector& getAggs() const { return aggs_; }
  ExprPtr getAgg(size_t i) const {
    CHECK_LT(i, aggs_.size());
    return aggs_[i];
  }

  void setAggExprs(ExprPtrVector new_aggs) { aggs_ = std::move(new_aggs); }

  void replaceInput(std::shared_ptr<const Node> old_input,
                    std::shared_ptr<const Node> input) override;

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
      hash_ = typeid(Aggregate).hash_code();
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

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<Aggregate>(*this);
  }

 private:
  const size_t groupby_count_;
  ExprPtrVector aggs_;
  std::vector<std::string> fields_;
};

class Join : public Node {
 public:
  Join(std::shared_ptr<const Node> lhs,
       std::shared_ptr<const Node> rhs,
       ExprPtr condition,
       const JoinType join_type)
      : condition_(std::move(condition)), join_type_(join_type) {
    inputs_.emplace_back(std::move(lhs));
    inputs_.emplace_back(std::move(rhs));
  }

  Join(Join const&);

  JoinType getJoinType() const { return join_type_; }

  const Expr* getCondition() const { return condition_.get(); }
  ExprPtr getConditionShared() const { return condition_; }

  void setCondition(ExprPtr new_condition) { condition_ = std::move(new_condition); }

  void replaceInput(std::shared_ptr<const Node> old_input,
                    std::shared_ptr<const Node> input) override;

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               inputsToString(inputs_),
               ", condition=",
               (condition_ ? condition_->toString() : "null"),
               ", join_type=",
               ::toString(join_type_));
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(Join).hash_code();
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

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<Join>(*this);
  }

 private:
  ExprPtr condition_;
  const JoinType join_type_;
};

// a helper node that contains detailed information of each level of join qual
// which is used when extracting query plan DAG
class TranslatedJoin : public Node {
 public:
  TranslatedJoin(const Node* lhs,
                 const Node* rhs,
                 std::vector<const ColumnVar*> lhs_join_cols,
                 std::vector<const ColumnVar*> rhs_join_cols,
                 ExprPtrVector filter_ops,
                 ExprPtr outer_join_cond,
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
      hash_ = typeid(TranslatedJoin).hash_code();
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
  const Node* getLHS() const { return lhs_; }
  const Node* getRHS() const { return rhs_; }
  size_t getFilterCondSize() const { return filter_ops_.size(); }
  const std::vector<ExprPtr>& getFilterCond() const { return filter_ops_; }
  const Expr* getOuterJoinCond() const { return outer_join_cond_.get(); }
  std::string getOpType() const { return op_type_; }
  std::string getQualifier() const { return qualifier_; }
  std::string getOpTypeInfo() const { return op_typeinfo_; }
  size_t size() const override { return 0; }
  JoinType getJoinType() const { return join_type_; }
  void replaceInput(std::shared_ptr<const Node> old_input,
                    std::shared_ptr<const Node> input) override {
    CHECK(false);
  }
  std::shared_ptr<Node> deepCopy() const override {
    CHECK(false);
    return nullptr;
  }
  std::string getFieldName(const size_t i) const;
  std::vector<const ColumnVar*> getJoinCols(bool lhs) const {
    if (lhs) {
      return lhs_join_cols_;
    }
    return rhs_join_cols_;
  }
  bool isNestedLoopQual() const { return nested_loop_; }

 private:
  const Node* lhs_;
  const Node* rhs_;
  const std::vector<const ColumnVar*> lhs_join_cols_;
  const std::vector<const ColumnVar*> rhs_join_cols_;
  const std::vector<ExprPtr> filter_ops_;
  ExprPtr outer_join_cond_;
  const bool nested_loop_;
  const JoinType join_type_;
  const std::string op_type_;
  const std::string qualifier_;
  const std::string op_typeinfo_;
};

class Filter : public Node {
 public:
  Filter(ExprPtr condition, std::shared_ptr<const Node> input)
      : condition_(std::move(condition)) {
    CHECK(condition_);
    inputs_.emplace_back(std::move(input));
  }

  // for dummy filter node for data recycler
  Filter(ExprPtr condition) : condition_(std::move(condition)) { CHECK(condition_); }

  Filter(Filter const&);

  const Expr* getConditionExpr() const { return condition_.get(); }
  ExprPtr getConditionExprShared() const { return condition_; }

  void setCondition(ExprPtr new_condition) {
    CHECK(new_condition);
    condition_ = std::move(new_condition);
  }

  size_t size() const override {
    CHECK(!inputs_.empty());
    return inputs_[0]->size();
  }

  void replaceInput(std::shared_ptr<const Node> old_input,
                    std::shared_ptr<const Node> input) override;

  std::string toString() const override {
    return cat(::typeName(this),
               getIdString(),
               "(",
               condition_->toString(),
               ", ",
               inputsToString(inputs_) + ")");
  }

  size_t toHash() const override {
    if (!hash_) {
      hash_ = typeid(Filter).hash_code();
      boost::hash_combine(*hash_, condition_->hash());
      for (auto& node : inputs_) {
        boost::hash_combine(*hash_, node->toHash());
      }
    }
    return *hash_;
  }

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<Filter>(*this);
  }

 private:
  ExprPtr condition_;
};

class Sort : public Node {
 public:
  Sort(std::vector<SortField> collation,
       const size_t limit,
       const size_t offset,
       std::shared_ptr<const Node> input)
      : collation_(std::move(collation))
      , limit_(limit)
      , offset_(offset)
      , empty_result_(false) {
    inputs_.emplace_back(std::move(input));
  }

  bool operator==(const Sort& that) const {
    return limit_ == that.limit_ && offset_ == that.offset_ &&
           empty_result_ == that.empty_result_ && hasEquivCollationOf(that);
  }

  size_t collationCount() const { return collation_.size(); }

  SortField getCollation(const size_t i) const {
    CHECK_LT(i, collation_.size());
    return collation_[i];
  }

  void setCollation(std::vector<SortField> collation) {
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
      hash_ = typeid(Sort).hash_code();
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

  size_t size() const override {
    CHECK(inputs_[0]);
    return inputs_[0]->size();
  }

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<Sort>(*this);
  }

 private:
  std::vector<SortField> collation_;
  const size_t limit_;
  const size_t offset_;
  bool empty_result_;

  bool hasEquivCollationOf(const Sort& that) const;
};

class LogicalValues : public Node {
 public:
  LogicalValues(std::vector<TargetMetaInfo> tuple_type, std::vector<ExprPtrVector> values)
      : tuple_type_(std::move(tuple_type)), values_(std::move(values)) {}

  LogicalValues(LogicalValues const&);

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
      hash_ = typeid(LogicalValues).hash_code();
      for (auto& target_meta_info : tuple_type_) {
        boost::hash_combine(*hash_, target_meta_info.get_resname());
        boost::hash_combine(*hash_, target_meta_info.type()->toString());
      }
    }
    return *hash_;
  }

  const Expr* getValue(const size_t row_idx, const size_t col_idx) const {
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

  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<LogicalValues>(*this);
  }

 private:
  std::vector<TargetMetaInfo> tuple_type_;
  std::vector<ExprPtrVector> values_;
};

class LogicalUnion : public Node {
 public:
  LogicalUnion(NodeInputs, bool is_all);
  std::shared_ptr<Node> deepCopy() const override {
    return std::make_shared<LogicalUnion>(*this);
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

class QueryDag {
 public:
  QueryDag(ConfigPtr config) : config_(config) { time(&now_); }
  QueryDag(ConfigPtr config, time_t now) : config_(config), now_(now) {}
  QueryDag(ConfigPtr config, NodePtr root);
  virtual ~QueryDag() = default;

  void eachNode(std::function<void(Node const*)> const& callback) const;

  void setRootNode(NodePtr node) { root_ = node; }

  /**
   * Returns the root node of the DAG.
   */
  const Node* getRootNode() const {
    CHECK(root_);
    return root_.get();
  }

  std::shared_ptr<const Node> getRootNodeShPtr() const { return root_; }

  /**
   * Registers a subquery with a root DAG builder. Should only be called during DAG
   * building and registration should only occur on the root.
   */
  void registerSubquery(std::shared_ptr<const ScalarSubquery> subquery_expr) {
    subqueries_.emplace_back(std::move(subquery_expr));
  }

  /**
   * Gets all registered subqueries. Only the root DAG can contain subqueries.
   */
  const std::vector<std::shared_ptr<const ScalarSubquery>>& getSubqueries() const {
    return subqueries_;
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
  NodePtr root_;
  // All nodes including the root one.
  std::vector<NodePtr> nodes_;
  std::vector<std::shared_ptr<const ScalarSubquery>> subqueries_;
};

const Type* getColumnType(const Node* node, size_t col_idx);

ExprPtr getNodeColumnRef(const Node* node, unsigned index);
ExprPtrVector getNodeColumnRefs(const Node* node);
size_t getNodeColumnCount(const Node* node);
ExprPtr getJoinInputColumnRef(const ColumnRef* col_ref);

}  // namespace hdk::ir
