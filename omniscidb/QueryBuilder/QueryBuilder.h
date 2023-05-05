/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "IR/Exception.h"
#include "IR/Node.h"
#include "SchemaMgr/SchemaProvider.h"
#include "Shared/Config.h"

namespace hdk::ir {

class InvalidQueryError : public Error {
 public:
  InvalidQueryError() {}
  InvalidQueryError(std::string desc) : Error(std::move(desc)) {}
};

class QueryBuilder;
class BuilderNode;
class ExprRewriter;

class BuilderExpr {
 public:
  BuilderExpr();
  BuilderExpr(const BuilderExpr& other) = default;
  BuilderExpr(BuilderExpr&& other) = default;
  BuilderExpr(const QueryBuilder* builder,
              ExprPtr expr,
              const std::string& name = "",
              bool auto_name = true);

  BuilderExpr& operator=(const BuilderExpr& other) = default;
  BuilderExpr& operator=(BuilderExpr&& other) = default;

  BuilderExpr rename(const std::string& name) const;
  const std::string& name() const { return name_; }
  bool isAutoNamed() const { return auto_name_; }

  ExprPtr expr() const { return expr_; }
  const Type* type() const { return expr_->type(); }

  BuilderExpr avg() const;
  BuilderExpr min() const;
  BuilderExpr max() const;
  BuilderExpr sum() const;
  BuilderExpr count(bool is_distinct = false) const;
  BuilderExpr approxCountDist() const;
  BuilderExpr approxQuantile(double val) const;
  BuilderExpr sample() const;
  BuilderExpr singleValue() const;
  BuilderExpr stdDev() const;
  BuilderExpr corr(const BuilderExpr& arg) const;

  BuilderExpr agg(const std::string& agg_str, const BuilderExpr& arg) const;
  BuilderExpr agg(const std::string& agg_str, double val = HUGE_VAL) const;
  BuilderExpr agg(AggType agg_kind, const BuilderExpr& arg) const;
  BuilderExpr agg(AggType agg_kind, double val) const;
  BuilderExpr agg(AggType agg_kind, bool is_dinstinct, const BuilderExpr& arg) const;
  BuilderExpr agg(AggType agg_kind,
                  bool is_dinstinct = false,
                  double val = HUGE_VAL) const;

  BuilderExpr extract(DateExtractField field) const;
  BuilderExpr extract(const std::string& field) const;

  BuilderExpr cast(const Type* new_type) const;
  BuilderExpr cast(const std::string& new_type) const;

  BuilderExpr logicalNot() const;
  BuilderExpr uminus() const;
  BuilderExpr isNull() const;
  BuilderExpr unnest() const;

  BuilderExpr add(const BuilderExpr& rhs) const;
  BuilderExpr add(int val) const;
  BuilderExpr add(int64_t val) const;
  BuilderExpr add(float val) const;
  BuilderExpr add(double val) const;

  BuilderExpr add(const BuilderExpr& rhs, DateAddField field) const;
  BuilderExpr add(int64_t val, DateAddField field) const;
  BuilderExpr add(const BuilderExpr& rhs, const std::string& field) const;
  BuilderExpr add(int64_t val, const std::string& field) const;

  BuilderExpr sub(const BuilderExpr& rhs) const;
  BuilderExpr sub(int val) const;
  BuilderExpr sub(int64_t val) const;
  BuilderExpr sub(float val) const;
  BuilderExpr sub(double val) const;

  BuilderExpr sub(const BuilderExpr& rhs, DateAddField field) const;
  BuilderExpr sub(int val, DateAddField field) const;
  BuilderExpr sub(int64_t val, DateAddField field) const;
  BuilderExpr sub(const BuilderExpr& rhs, const std::string& field) const;
  BuilderExpr sub(int val, const std::string& field) const;
  BuilderExpr sub(int64_t val, const std::string& field) const;

  BuilderExpr mul(const BuilderExpr& rhs) const;
  BuilderExpr mul(int val) const;
  BuilderExpr mul(int64_t val) const;
  BuilderExpr mul(float val) const;
  BuilderExpr mul(double val) const;

  BuilderExpr div(const BuilderExpr& rhs) const;
  BuilderExpr div(int val) const;
  BuilderExpr div(int64_t val) const;
  BuilderExpr div(float val) const;
  BuilderExpr div(double val) const;

  BuilderExpr mod(const BuilderExpr& rhs) const;
  BuilderExpr mod(int val) const;
  BuilderExpr mod(int64_t val) const;

  BuilderExpr ceil() const;
  BuilderExpr floor() const;

  BuilderExpr pow(const BuilderExpr& rhs) const;
  BuilderExpr pow(int val) const;
  BuilderExpr pow(int64_t val) const;
  BuilderExpr pow(float val) const;
  BuilderExpr pow(double val) const;

  BuilderExpr logicalAnd(const BuilderExpr& rhs) const;
  BuilderExpr logicalOr(const BuilderExpr& rhs) const;

  BuilderExpr eq(const BuilderExpr& rhs) const;
  BuilderExpr eq(int val) const;
  BuilderExpr eq(int64_t val) const;
  BuilderExpr eq(float val) const;
  BuilderExpr eq(double val) const;
  BuilderExpr eq(const std::string& val) const;

  BuilderExpr ne(const BuilderExpr& rhs) const;
  BuilderExpr ne(int val) const;
  BuilderExpr ne(int64_t val) const;
  BuilderExpr ne(float val) const;
  BuilderExpr ne(double val) const;
  BuilderExpr ne(const std::string& val) const;

  BuilderExpr lt(const BuilderExpr& rhs) const;
  BuilderExpr lt(int val) const;
  BuilderExpr lt(int64_t val) const;
  BuilderExpr lt(float val) const;
  BuilderExpr lt(double val) const;
  BuilderExpr lt(const std::string& val) const;

  BuilderExpr le(const BuilderExpr& rhs) const;
  BuilderExpr le(int val) const;
  BuilderExpr le(int64_t val) const;
  BuilderExpr le(float val) const;
  BuilderExpr le(double val) const;
  BuilderExpr le(const std::string& val) const;

  BuilderExpr gt(const BuilderExpr& rhs) const;
  BuilderExpr gt(int val) const;
  BuilderExpr gt(int64_t val) const;
  BuilderExpr gt(float val) const;
  BuilderExpr gt(double val) const;
  BuilderExpr gt(const std::string& val) const;

  BuilderExpr ge(const BuilderExpr& rhs) const;
  BuilderExpr ge(int val) const;
  BuilderExpr ge(int64_t val) const;
  BuilderExpr ge(float val) const;
  BuilderExpr ge(double val) const;
  BuilderExpr ge(const std::string& val) const;

  BuilderExpr at(const BuilderExpr& idx) const;
  BuilderExpr at(int idx) const;
  BuilderExpr at(int64_t idx) const;

  BuilderExpr rewrite(ExprRewriter& rewriter) const;

  BuilderExpr operator!() const;
  BuilderExpr operator-() const;
  BuilderExpr operator[](const BuilderExpr& idx) const;
  BuilderExpr operator[](int idx) const;
  BuilderExpr operator[](int64_t idx) const;

  const QueryBuilder& builder() const;
  Context& ctx() const;

 protected:
  friend class QueryBuilder;
  friend class BuilderNode;

  const QueryBuilder* builder_;
  ExprPtr expr_;
  std::string name_;
  bool auto_name_;
};

class BuilderSortField {
 public:
  BuilderSortField();
  BuilderSortField(int col_idx,
                   SortDirection dir,
                   NullSortedPosition null_pos = NullSortedPosition::Last);
  BuilderSortField(int col_idx,
                   const std::string& dir,
                   const std::string& null_pos = "last");
  BuilderSortField(const std::string& col_name,
                   SortDirection dir,
                   NullSortedPosition null_pos = NullSortedPosition::Last);
  BuilderSortField(const std::string& col_name,
                   const std::string& dir,
                   const std::string& null_pos = "last");
  BuilderSortField(BuilderExpr expr,
                   SortDirection dir,
                   NullSortedPosition null_pos = NullSortedPosition::Last);
  BuilderSortField(BuilderExpr expr,
                   const std::string& dir,
                   const std::string& null_pos = "last");

  bool hasColIdx() const { return std::holds_alternative<int>(field_); }
  bool hasColName() const { return std::holds_alternative<std::string>(field_); }
  bool hasExpr() const { return std::holds_alternative<BuilderExpr>(field_); }

  int colIdx() const { return std::get<int>(field_); }
  std::string colName() const { return std::get<std::string>(field_); }
  ExprPtr expr() const { return std::get<BuilderExpr>(field_).expr(); }
  SortDirection dir() const { return dir_; }
  NullSortedPosition nullsPosition() const { return null_pos_; }

  static SortDirection parseSortDirection(const std::string& val);
  static NullSortedPosition parseNullPosition(const std::string& val);

 protected:
  std::variant<int, std::string, BuilderExpr> field_;
  SortDirection dir_;
  NullSortedPosition null_pos_;
};

class BuilderNode {
 public:
  BuilderNode();

  BuilderExpr ref(int col_idx) const;
  BuilderExpr ref(const std::string& col_name) const;
  std::vector<BuilderExpr> ref(std::initializer_list<int> col_indices) const;
  std::vector<BuilderExpr> ref(std::vector<int> col_indices) const;
  std::vector<BuilderExpr> ref(std::initializer_list<std::string> col_names) const;
  std::vector<BuilderExpr> ref(std::vector<std::string> col_names) const;

  BuilderExpr count() const;
  BuilderExpr count(int col_idx, bool is_distinct = false) const;
  BuilderExpr count(const std::string& col_name, bool is_distinct = false) const;
  BuilderExpr count(BuilderExpr col_ref, bool is_distinct = false) const;

  BuilderNode proj(std::initializer_list<int> col_indices) const;
  BuilderNode proj(std::initializer_list<int> col_indices,
                   const std::vector<std::string>& fields) const;
  BuilderNode proj(std::initializer_list<std::string> col_names) const;
  BuilderNode proj(std::initializer_list<std::string> col_names,
                   const std::vector<std::string>& fields) const;
  BuilderNode proj(int col_idx) const;
  BuilderNode proj(int col_idx, const std::string& field_name) const;
  BuilderNode proj(const std::vector<int> col_indices) const;
  BuilderNode proj(const std::vector<int> col_indices,
                   const std::vector<std::string>& fields) const;
  BuilderNode proj(const std::string& col_name) const;
  BuilderNode proj(const std::string& col_name, const std::string& field_name) const;
  BuilderNode proj(const std::vector<std::string>& col_names) const;
  BuilderNode proj(const std::vector<std::string>& col_names,
                   const std::vector<std::string>& fields) const;
  BuilderNode proj(const BuilderExpr& expr) const;
  BuilderNode proj(const BuilderExpr& expr, const std::string& field) const;
  BuilderNode proj(const std::vector<BuilderExpr>& exprs) const;
  BuilderNode proj(const std::vector<BuilderExpr>& exprs,
                   const std::vector<std::string>& fields) const;

  BuilderNode filter(const BuilderExpr& condition) const;

  BuilderNode agg(int group_key, const std::string& agg_str) const;
  BuilderNode agg(int group_key, std::initializer_list<std::string> aggs) const;
  BuilderNode agg(int group_key, const std::vector<std::string>& aggs) const;
  BuilderNode agg(int group_key, BuilderExpr agg_expr) const;
  BuilderNode agg(int group_key, const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(const std::string& group_key, const std::string& agg_str) const;
  BuilderNode agg(const std::string& group_key,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(const std::string& group_key,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(const std::string& group_key, BuilderExpr agg_expr) const;
  BuilderNode agg(const std::string& group_key,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(BuilderExpr group_key, const std::string& agg_str) const;
  BuilderNode agg(BuilderExpr group_key, std::initializer_list<std::string> aggs) const;
  BuilderNode agg(BuilderExpr group_key, const std::vector<std::string>& aggs) const;
  BuilderNode agg(BuilderExpr group_key, BuilderExpr agg_expr) const;
  BuilderNode agg(BuilderExpr group_key, const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(std::initializer_list<int> group_keys,
                  const std::string& agg_str) const;
  BuilderNode agg(std::initializer_list<int> group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(std::initializer_list<int> group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(std::initializer_list<int> group_keys, BuilderExpr agg_expr) const;
  BuilderNode agg(std::initializer_list<int> group_keys,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(std::initializer_list<std::string> group_keys,
                  const std::string& agg_str) const;
  BuilderNode agg(std::initializer_list<std::string> group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(std::initializer_list<std::string> group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(std::initializer_list<std::string> group_keys,
                  BuilderExpr agg_expr) const;
  BuilderNode agg(std::initializer_list<std::string> group_keys,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(std::initializer_list<BuilderExpr> group_keys,
                  const std::string& agg_str) const;
  BuilderNode agg(std::initializer_list<BuilderExpr> group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(std::initializer_list<BuilderExpr> group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(std::initializer_list<BuilderExpr> group_keys,
                  BuilderExpr agg_expr) const;
  BuilderNode agg(std::initializer_list<BuilderExpr> group_keys,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(const std::vector<int>& group_keys, const std::string& agg_str) const;
  BuilderNode agg(const std::vector<int>& group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(const std::vector<int>& group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(const std::vector<int>& group_keys, BuilderExpr agg_expr) const;
  BuilderNode agg(const std::vector<int>& group_keys,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(const std::vector<std::string>& group_keys,
                  const std::string& agg_str) const;
  BuilderNode agg(const std::vector<std::string>& group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(const std::vector<std::string>& group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(const std::vector<std::string>& group_keys, BuilderExpr agg_expr) const;
  BuilderNode agg(const std::vector<std::string>& group_keys,
                  const std::vector<BuilderExpr>& aggs) const;
  BuilderNode agg(const std::vector<BuilderExpr>& group_keys,
                  const std::string& agg_str) const;
  BuilderNode agg(const std::vector<BuilderExpr>& group_keys,
                  std::initializer_list<std::string> aggs) const;
  BuilderNode agg(const std::vector<BuilderExpr>& group_keys,
                  const std::vector<std::string>& aggs) const;
  BuilderNode agg(const std::vector<BuilderExpr>& group_keys, BuilderExpr agg_expr) const;
  BuilderNode agg(const std::vector<BuilderExpr>& group_keys,
                  const std::vector<BuilderExpr>& aggs) const;

  BuilderExpr parseAggString(const std::string& agg_str) const;

  BuilderNode sort(int col_idx,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(int col_idx,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<int> col_indexes,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<int> col_indexes,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<int>& col_indexes,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<int>& col_indexes,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::string& col_name,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::string& col_name,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<std::string> col_names,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<std::string> col_names,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<std::string>& col_names,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<std::string>& col_names,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(BuilderExpr col_ref,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(BuilderExpr col_ref,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<BuilderExpr> col_refs,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(std::initializer_list<BuilderExpr> col_refs,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<BuilderExpr>& col_refs,
                   SortDirection dir = SortDirection::Ascending,
                   NullSortedPosition null_pos = NullSortedPosition::Last,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<BuilderExpr>& col_refs,
                   const std::string& dir,
                   const std::string& null_pos = "last",
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const BuilderSortField& field,
                   size_t limit = 0,
                   size_t offset = 0) const;
  BuilderNode sort(const std::vector<BuilderSortField>& fields,
                   size_t limit = 0,
                   size_t offset = 0) const;

  BuilderNode join(const BuilderNode& rhs, JoinType join_type = JoinType::INNER) const;
  BuilderNode join(const BuilderNode& rhs, const std::string& join_type) const;
  BuilderNode join(const BuilderNode& rhs,
                   const std::vector<std::string>& col_names,
                   JoinType join_type = JoinType::INNER) const;
  BuilderNode join(const BuilderNode& rhs,
                   const std::vector<std::string>& col_names,
                   const std::string& join_type) const;
  BuilderNode join(const BuilderNode& rhs,
                   const std::vector<std::string>& lhs_col_names,
                   const std::vector<std::string>& rhs_col_names,
                   JoinType join_type = JoinType::INNER) const;
  BuilderNode join(const BuilderNode& rhs,
                   const std::vector<std::string>& lhs_col_names,
                   const std::vector<std::string>& rhs_col_names,
                   const std::string& join_type) const;
  BuilderNode join(const BuilderNode& rhs,
                   const BuilderExpr& cond,
                   JoinType join_type = JoinType::INNER) const;
  BuilderNode join(const BuilderNode& rhs,
                   const BuilderExpr& cond,
                   const std::string& join_type) const;

  BuilderExpr operator[](int col_idx) const;
  BuilderExpr operator[](const std::string& col_name) const;

  std::unique_ptr<QueryDag> finalize() const;

  NodePtr node() const { return node_; }
  int size() const { return node_->size(); }
  ColumnInfoPtr columnInfo(int col_index) const;
  ColumnInfoPtr columnInfo(const std::string& col_name) const;

 protected:
  friend class QueryBuilder;

  BuilderNode(const QueryBuilder* builder, NodePtr node);

  BuilderNode proj() const;
  BuilderNode proj(const ExprPtrVector& exprs,
                   const std::vector<std::string>& fields) const;
  std::vector<BuilderExpr> parseAggString(const std::vector<std::string>& aggs) const;

  const QueryBuilder* builder_;
  NodePtr node_;
};

class QueryBuilder {
 public:
  QueryBuilder(Context& ctx, SchemaProviderPtr schema_provider, ConfigPtr config);

  BuilderNode scan(const std::string& table_name) const;
  BuilderNode scan(int db_id, const std::string& table_name) const;
  BuilderNode scan(int db_id, int table_id) const;
  BuilderNode scan(const TableRef& table_ref) const;

  BuilderExpr count() const;

  BuilderExpr cst(int val) const;
  BuilderExpr cst(int val, const Type* type) const;
  BuilderExpr cst(int val, const std::string& type) const;
  BuilderExpr cst(int64_t val) const;
  BuilderExpr cst(int64_t val, const Type* type) const;
  BuilderExpr cst(int64_t val, const std::string& type) const;
  BuilderExpr cst(double val) const;
  BuilderExpr cst(double val, const Type* type) const;
  BuilderExpr cst(double val, const std::string& type) const;
  BuilderExpr cst(const std::string& val) const;
  BuilderExpr cst(const std::string& val, const Type* type) const;
  BuilderExpr cst(const std::string& val, const std::string& type) const;
  BuilderExpr cst(std::initializer_list<int>& vals) const;
  BuilderExpr cst(std::initializer_list<int> vals, const Type* type) const;
  BuilderExpr cst(std::initializer_list<int> vals, const std::string& type) const;
  BuilderExpr cst(std::initializer_list<double> vals) const;
  BuilderExpr cst(std::initializer_list<double> vals, const Type* type) const;
  BuilderExpr cst(std::initializer_list<double> vals, const std::string& type) const;
  BuilderExpr cst(const std::vector<int>& vals) const;
  BuilderExpr cst(const std::vector<int>& vals, const Type* type) const;
  BuilderExpr cst(const std::vector<int>& vals, const std::string& type) const;
  BuilderExpr cst(const std::vector<double>& vals) const;
  BuilderExpr cst(const std::vector<double>& vals, const Type* type) const;
  BuilderExpr cst(const std::vector<double>& vals, const std::string& type) const;
  BuilderExpr cst(const std::vector<std::string>& vals, const Type* type) const;
  BuilderExpr cst(const std::vector<std::string>& vals, const std::string& type) const;
  BuilderExpr cst(const std::vector<BuilderExpr>& vals, const Type* type) const;
  BuilderExpr cst(const std::vector<BuilderExpr>& vals, const std::string& type) const;
  BuilderExpr cstNoScale(int64_t val, const Type* type) const;
  BuilderExpr cstNoScale(int64_t val, const std::string& type) const;
  BuilderExpr trueCst() const;
  BuilderExpr falseCst() const;
  BuilderExpr nullCst() const;
  BuilderExpr nullCst(const Type* type) const;
  BuilderExpr nullCst(const std::string& type) const;
  BuilderExpr date(const std::string& val) const;
  BuilderExpr time(const std::string& val) const;
  BuilderExpr timestamp(const std::string& val) const;

  BuilderExpr ifThenElse(const BuilderExpr& cond,
                         const BuilderExpr& if_val,
                         const BuilderExpr& else_val);

 protected:
  friend class BuilderExpr;
  friend class BuilderNode;

  BuilderNode scan(TableInfoPtr table_info) const;

  Context& ctx_;
  SchemaProviderPtr schema_provider_;
  ConfigPtr config_;
};

BuilderExpr operator+(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator+(const BuilderExpr& lhs, int rhs);
BuilderExpr operator+(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator+(const BuilderExpr& lhs, float rhs);
BuilderExpr operator+(const BuilderExpr& lhs, double rhs);
BuilderExpr operator+(int lhs, const BuilderExpr& rhs);
BuilderExpr operator+(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator+(float lhs, const BuilderExpr& rhs);
BuilderExpr operator+(double lhs, const BuilderExpr& rhs);

BuilderExpr operator-(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator-(const BuilderExpr& lhs, int rhs);
BuilderExpr operator-(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator-(const BuilderExpr& lhs, float rhs);
BuilderExpr operator-(const BuilderExpr& lhs, double rhs);
BuilderExpr operator-(int lhs, const BuilderExpr& rhs);
BuilderExpr operator-(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator-(float lhs, const BuilderExpr& rhs);
BuilderExpr operator-(double lhs, const BuilderExpr& rhs);

BuilderExpr operator*(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator*(const BuilderExpr& lhs, int rhs);
BuilderExpr operator*(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator*(const BuilderExpr& lhs, float rhs);
BuilderExpr operator*(const BuilderExpr& lhs, double rhs);
BuilderExpr operator*(int lhs, const BuilderExpr& rhs);
BuilderExpr operator*(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator*(float lhs, const BuilderExpr& rhs);
BuilderExpr operator*(double lhs, const BuilderExpr& rhs);

BuilderExpr operator/(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator/(const BuilderExpr& lhs, int rhs);
BuilderExpr operator/(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator/(const BuilderExpr& lhs, float rhs);
BuilderExpr operator/(const BuilderExpr& lhs, double rhs);
BuilderExpr operator/(int lhs, const BuilderExpr& rhs);
BuilderExpr operator/(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator/(float lhs, const BuilderExpr& rhs);
BuilderExpr operator/(double lhs, const BuilderExpr& rhs);

BuilderExpr operator%(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator%(const BuilderExpr& lhs, int rhs);
BuilderExpr operator%(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator%(int lhs, const BuilderExpr& rhs);
BuilderExpr operator%(int64_t lhs, const BuilderExpr& rhs);

BuilderExpr operator&&(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator||(const BuilderExpr& lhs, const BuilderExpr& rhs);

BuilderExpr operator==(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator==(const BuilderExpr& lhs, int rhs);
BuilderExpr operator==(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator==(const BuilderExpr& lhs, float rhs);
BuilderExpr operator==(const BuilderExpr& lhs, double rhs);
BuilderExpr operator==(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator==(int lhs, const BuilderExpr& rhs);
BuilderExpr operator==(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator==(float lhs, const BuilderExpr& rhs);
BuilderExpr operator==(double lhs, const BuilderExpr& rhs);
BuilderExpr operator==(const std::string& lhs, const BuilderExpr& rhs);

BuilderExpr operator!=(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator!=(const BuilderExpr& lhs, int rhs);
BuilderExpr operator!=(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator!=(const BuilderExpr& lhs, float rhs);
BuilderExpr operator!=(const BuilderExpr& lhs, double rhs);
BuilderExpr operator!=(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator!=(int lhs, const BuilderExpr& rhs);
BuilderExpr operator!=(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator!=(float lhs, const BuilderExpr& rhs);
BuilderExpr operator!=(double lhs, const BuilderExpr& rhs);
BuilderExpr operator!=(const std::string& lhs, const BuilderExpr& rhs);

BuilderExpr operator<(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator<(const BuilderExpr& lhs, int rhs);
BuilderExpr operator<(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator<(const BuilderExpr& lhs, float rhs);
BuilderExpr operator<(const BuilderExpr& lhs, double rhs);
BuilderExpr operator<(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator<(int lhs, const BuilderExpr& rhs);
BuilderExpr operator<(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator<(float lhs, const BuilderExpr& rhs);
BuilderExpr operator<(double lhs, const BuilderExpr& rhs);
BuilderExpr operator<(const std::string& lhs, const BuilderExpr& rhs);

BuilderExpr operator<=(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator<=(const BuilderExpr& lhs, int rhs);
BuilderExpr operator<=(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator<=(const BuilderExpr& lhs, float rhs);
BuilderExpr operator<=(const BuilderExpr& lhs, double rhs);
BuilderExpr operator<=(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator<=(int lhs, const BuilderExpr& rhs);
BuilderExpr operator<=(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator<=(float lhs, const BuilderExpr& rhs);
BuilderExpr operator<=(double lhs, const BuilderExpr& rhs);
BuilderExpr operator<=(const std::string& lhs, const BuilderExpr& rhs);

BuilderExpr operator>(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator>(const BuilderExpr& lhs, int rhs);
BuilderExpr operator>(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator>(const BuilderExpr& lhs, float rhs);
BuilderExpr operator>(const BuilderExpr& lhs, double rhs);
BuilderExpr operator>(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator>(int lhs, const BuilderExpr& rhs);
BuilderExpr operator>(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator>(float lhs, const BuilderExpr& rhs);
BuilderExpr operator>(double lhs, const BuilderExpr& rhs);
BuilderExpr operator>(const std::string& lhs, const BuilderExpr& rhs);

BuilderExpr operator>=(const BuilderExpr& lhs, const BuilderExpr& rhs);
BuilderExpr operator>=(const BuilderExpr& lhs, int rhs);
BuilderExpr operator>=(const BuilderExpr& lhs, int64_t rhs);
BuilderExpr operator>=(const BuilderExpr& lhs, float rhs);
BuilderExpr operator>=(const BuilderExpr& lhs, double rhs);
BuilderExpr operator>=(const BuilderExpr& lhs, const std::string& rhs);
BuilderExpr operator>=(int lhs, const BuilderExpr& rhs);
BuilderExpr operator>=(int64_t lhs, const BuilderExpr& rhs);
BuilderExpr operator>=(float lhs, const BuilderExpr& rhs);
BuilderExpr operator>=(double lhs, const BuilderExpr& rhs);
BuilderExpr operator>=(const std::string& lhs, const BuilderExpr& rhs);

}  // namespace hdk::ir
