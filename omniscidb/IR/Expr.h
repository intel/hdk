/**
 * Copyright 2021 OmniSci, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Context.h"
#include "Type.h"

#include "SchemaMgr/ColumnInfo.h"
#include "Shared/sqltypes.h"

#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <vector>

class RelAlgNode;

namespace hdk::ir {

class Type;
class Expr;

using ExprPtr = std::shared_ptr<const Expr>;
using ExprPtrList = std::list<ExprPtr>;
using ExprPtrVector = std::vector<ExprPtr>;

template <typename Tp, typename... Args>
inline
    typename std::enable_if<std::is_base_of<Expr, Tp>::value, std::shared_ptr<Tp>>::type
    makeExpr(Args&&... args) {
  return std::make_shared<Tp>(std::forward<Args>(args)...);
}

class ColumnVar;
class TargetEntry;
using DomainSet = std::list<const Expr*>;

/*
 * @type Expr
 * @brief super class for all expressions in parse trees and in query plans
 */
class Expr : public std::enable_shared_from_this<Expr> {
 public:
  Expr(const Type* type, bool has_agg = false);
  virtual ~Expr() {}

  const Type* type() const { return type_; }
  Context& ctx() const { return type_->ctx(); }
  bool containsAgg() const { return contains_agg_; }

  virtual ExprPtr cast(const Type* new_type, bool is_dict_intersection = false) const;

  // Make a deep copy of self
  virtual ExprPtr deep_copy() const = 0;
  // Make a deep copy of self replacing its type with the specified one.
  virtual ExprPtr withType(const Type* type) const;

  virtual bool operator==(const Expr& rhs) const = 0;
  virtual std::string toString() const = 0;
  virtual void print() const;

  /*
   * @brief decompress adds cast operator to decompress encoded result
   */
  ExprPtr decompress() const;

  virtual size_t hash() const;

  template <typename T>
  const T* as() const {
    return dynamic_cast<const T*>(this);
  }

  template <typename T>
  bool is() const {
    return as<T>() != nullptr;
  }

 protected:
  const Type* type_;
  bool contains_agg_;
  mutable std::optional<size_t> hash_;
};

/*
 * Reference to an output node column.
 */
class ColumnRef : public Expr {
 public:
  ColumnRef(const Type* type, const RelAlgNode* node, unsigned idx)
      : Expr(type), node_(node), idx_(idx) {}

  ExprPtr deep_copy() const override { return makeExpr<ColumnRef>(type_, node_, idx_); }

  bool operator==(const Expr& rhs) const override {
    const ColumnRef* rhsp = dynamic_cast<const ColumnRef*>(&rhs);
    return rhsp && node_ == rhsp->node_ && idx_ == rhsp->idx_;
  }

  std::string toString() const override;

  const RelAlgNode* node() const { return node_; }

  unsigned index() const { return idx_; }

  size_t hash() const override;

 protected:
  const RelAlgNode* node_;
  unsigned idx_;
};

/*
 * Used in Compound nodes to referene group by keys columns in target
 * expressions. Numbering starts with 1 to be consistent with RexRef.
 */
class GroupColumnRef : public Expr {
 public:
  GroupColumnRef(const Type* type, unsigned idx) : Expr(type), idx_(idx) {}

  ExprPtr deep_copy() const override { return makeExpr<GroupColumnRef>(type_, idx_); }

  bool operator==(const Expr& rhs) const override {
    const GroupColumnRef* rhsp = dynamic_cast<const GroupColumnRef*>(&rhs);
    return rhsp && idx_ == rhsp->idx_;
  }

  std::string toString() const override {
    return "(GroupColumnRef idx=" + std::to_string(idx_) + ")";
  }

  unsigned index() const { return idx_; }

  size_t hash() const override;

 protected:
  unsigned idx_;
};

/*
 * @type ColumnVar
 * @brief expression that evaluates to the value of a column in a given row from a base
 * table. It is used in parse trees and is only used in Scan nodes in a query plan for
 * scanning a table while Var nodes are used for all other plans.
 */
class ColumnVar : public Expr {
 public:
  ColumnVar(ColumnInfoPtr col_info, int nest_level)
      : Expr(col_info->type), rte_idx_(nest_level), col_info_(std::move(col_info)) {}
  ColumnVar(const hdk::ir::Type* type)
      : Expr(type)
      , rte_idx_(-1)
      , col_info_(std::make_shared<ColumnInfo>(-1, 0, 0, "", type_, false)) {}
  ColumnVar(const hdk::ir::Type* type,
            int table_id,
            int col_id,
            int nest_level,
            bool is_virtual = false)
      : Expr(type)
      , rte_idx_(nest_level)
      , col_info_(
            std::make_shared<ColumnInfo>(-1, table_id, col_id, "", type_, is_virtual)) {}
  int dbId() const { return col_info_->db_id; }
  int tableId() const { return col_info_->table_id; }
  int columnId() const { return col_info_->column_id; }
  int rteIdx() const { return rte_idx_; }
  ColumnInfoPtr columnInfo() const { return col_info_; }
  bool isVirtual() const { return col_info_->is_rowid; }

  ExprPtr deep_copy() const override;
  ExprPtr withType(const Type* type) const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 protected:
  int rte_idx_;  // 0-based range table index, used for table ordering in multi-joins
  ColumnInfoPtr col_info_;
};

/*
 * @type ExpressionTuple
 * @brief A tuple of expressions on the side of an equi-join on multiple columns.
 * Not to be used in any other context.
 */
class ExpressionTuple : public Expr {
 public:
  ExpressionTuple(const ExprPtrVector& tuple)
      : Expr(hdk::ir::Context::defaultCtx().null()), tuple_(tuple){};

  const ExprPtrVector& tuple() const { return tuple_; }

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const ExprPtrVector tuple_;
};

/*
 * @type Var
 * @brief expression that evaluates to the value of a column in a given row generated
 * from a query plan node.  It is only used in plan nodes above Scan nodes.
 * The row can be produced by either the inner or the outer plan in case of a join.
 * It inherits from ColumnVar to keep track of the lineage through the plan nodes.
 * The table_id will be set to 0 if the Var does not correspond to an original column
 * value.
 */
class Var : public ColumnVar {
 public:
  enum WhichRow { kINPUT_OUTER, kINPUT_INNER, kOUTPUT, kGROUPBY };
  Var(ColumnInfoPtr col_info, int i, WhichRow o, int v)
      : ColumnVar(col_info, i), which_row_(o), var_no_(v) {}
  Var(const hdk::ir::Type* type, WhichRow o, int v)
      : ColumnVar(type), which_row_(o), var_no_(v) {}
  WhichRow whichRow() const { return which_row_; }
  int varNo() const { return var_no_; }
  ExprPtr deep_copy() const override;
  ExprPtr withType(const Type* type) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  WhichRow which_row_;  // indicate which row this Var should project from.  It can be
                        // from the outer input plan or the inner input plan (for joins)
                        // or the output row in the current plan.
  int var_no_;          // the column number in the row.  1-based
};

/*
 * @type Constant
 * @brief expression for a constant value
 */
class Constant : public Expr {
 public:
  Constant(const Type* type, bool is_null, Datum v, bool cacheable = true)
      : Expr(type), is_null_(is_null), cacheable_(cacheable), value_(v) {
    if (is_null) {
      setNullValue();
    } else {
      type_ = type_->withNullable(false);
    }
  }
  Constant(const Type* type, bool is_null, const ExprPtrList& l, bool cacheable = true)
      : Expr(type)
      , is_null_(is_null)
      , cacheable_(cacheable)
      , value_(Datum{0})
      , value_list_(l) {}
  ~Constant() override;
  bool isNull() const { return is_null_; }
  bool cacheable() const { return cacheable_; }
  Datum value() const { return value_; }
  int64_t intVal() const { return extract_int_type_from_datum(value_, type_); }
  double fpVal() const { return extract_fp_type_from_datum(value_, type_); }
  const ExprPtrList& valueList() const { return value_list_; }
  ExprPtr deep_copy() const override;
  ExprPtr cast(const Type* new_type, bool is_dict_intersection = false) const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

  static ExprPtr make(const Type* type, int64_t val, bool cacheable = true);

 protected:
  // Constant is NULL
  bool is_null_;
  // A hint for DAG caches. Cache hit is unlikely, when set to true
  // (e.g. constant expression represents NOW datetime).
  bool cacheable_;
  Datum value_;  // the constant value
  const ExprPtrList value_list_;
  ExprPtr castNumber(const Type* new_type) const;
  ExprPtr castString(const Type* new_type) const;
  ExprPtr castFromString(const Type* new_type) const;
  ExprPtr castToString(const Type* new_type) const;
  ExprPtr doCast(const Type* new_type) const;
  void setNullValue();
};

/*
 * @type UOper
 * @brief represents unary operator expressions.  operator types include
 * kUMINUS, kISNULL, kEXISTS, kCAST
 */
class UOper : public Expr {
 public:
  UOper(const Type* type,
        bool has_agg,
        SQLOps o,
        ExprPtr p,
        bool is_dict_intersection = false)
      : Expr(type, has_agg)
      , op_type_(o)
      , operand_(p)
      , is_dict_intersection_(is_dict_intersection) {}
  UOper(const Type* type, SQLOps o, ExprPtr p)
      : Expr(type), op_type_(o), operand_(p), is_dict_intersection_(false) {}

  SQLOps opType() const { return op_type_; }

  bool isNot() const { return op_type_ == SQLOps::kNOT; }
  bool isUMinus() const { return op_type_ == SQLOps::kUMINUS; }
  bool isIsNull() const { return op_type_ == SQLOps::kISNULL; }
  bool isIsNotNull() const { return op_type_ == SQLOps::kISNOTNULL; }
  bool isExists() const { return op_type_ == SQLOps::kEXISTS; }
  bool isCast() const { return op_type_ == SQLOps::kCAST; }
  bool isUnnest() const { return op_type_ == SQLOps::kUNNEST; }

  const Expr* operand() const { return operand_.get(); }
  ExprPtr operandShared() const { return operand_; }
  bool isDictIntersection() const { return is_dict_intersection_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  ExprPtr cast(const Type* new_type, bool is_dict_intersection = false) const override;

  size_t hash() const override;

 protected:
  SQLOps op_type_;   // operator type, e.g., kUMINUS, kISNULL, kEXISTS
  ExprPtr operand_;  // operand expression
  bool is_dict_intersection_;
};

/*
 * @type BinOper
 * @brief represents binary operator expressions.  it includes all
 * comparison, arithmetic and boolean binary operators.  it handles ANY/ALL qualifiers
 * in case the right_operand is a subquery.
 */
class BinOper : public Expr {
 public:
  BinOper(const Type* type, bool has_agg, SQLOps o, SQLQualifier q, ExprPtr l, ExprPtr r)
      : Expr(type, has_agg)
      , op_type_(o)
      , qualifier_(q)
      , left_operand_(l)
      , right_operand_(r) {}
  BinOper(const Type* type, SQLOps o, SQLQualifier q, ExprPtr l, ExprPtr r)
      : Expr(type), op_type_(o), qualifier_(q), left_operand_(l), right_operand_(r) {}

  SQLOps opType() const { return op_type_; }

  bool isEq() const { return op_type_ == SQLOps::kEQ; }
  bool isBwEq() const { return op_type_ == SQLOps::kBW_EQ; }
  bool isNe() const { return op_type_ == SQLOps::kNE; }
  bool isLt() const { return op_type_ == SQLOps::kLT; }
  bool isGt() const { return op_type_ == SQLOps::kGT; }
  bool isLe() const { return op_type_ == SQLOps::kLE; }
  bool isGe() const { return op_type_ == SQLOps::kGE; }
  bool isAnd() const { return op_type_ == SQLOps::kAND; }
  bool isOr() const { return op_type_ == SQLOps::kOR; }
  bool isMinus() const { return op_type_ == SQLOps::kMINUS; }
  bool isPlus() const { return op_type_ == SQLOps::kPLUS; }
  bool isMul() const { return op_type_ == SQLOps::kMULTIPLY; }
  bool isDivide() const { return op_type_ == SQLOps::kDIVIDE; }
  bool isModulo() const { return op_type_ == SQLOps::kMODULO; }
  bool isArrayAt() const { return op_type_ == SQLOps::kARRAY_AT; }

  bool isEquivalence() const { return isEq() || isBwEq(); }
  bool isComparison() const { return IS_COMPARISON(op_type_); }
  bool isLogic() const { return IS_LOGIC(op_type_); }
  bool isArithmetic() const { return IS_ARITHMETIC(op_type_); }

  SQLQualifier qualifier() const { return qualifier_; }
  const Expr* leftOperand() const { return left_operand_.get(); }
  const Expr* rightOperand() const { return right_operand_.get(); }
  ExprPtr leftOperandShared() const { return left_operand_; }
  ExprPtr rightOperandShared() const { return right_operand_; }

  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  SQLOps op_type_;          // operator type, e.g., kLT, kAND, kPLUS, etc.
  SQLQualifier qualifier_;  // qualifier kANY, kALL or kONE.  Only relevant with
                            // right_operand is Subquery
  ExprPtr left_operand_;    // the left operand expression
  ExprPtr right_operand_;   // the right operand expression
};

/**
 * @type RangeOper
 * @brief
 */
class RangeOper : public Expr {
 public:
  RangeOper(const bool l_inclusive, const bool r_inclusive, ExprPtr l, ExprPtr r)
      : Expr(hdk::ir::Context::defaultCtx().null())
      , left_inclusive_(l_inclusive)
      , right_inclusive_(r_inclusive)
      , left_operand_(l)
      , right_operand_(r) {
    CHECK(left_operand_);
    CHECK(right_operand_);
  }

  const Expr* leftOperand() const { return left_operand_.get(); }
  const Expr* rightOperand() const { return right_operand_.get(); }

  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  // build a range between these two operands
  bool left_inclusive_;
  bool right_inclusive_;
  ExprPtr left_operand_;
  ExprPtr right_operand_;
};

class ScalarSubquery : public Expr {
 public:
  ScalarSubquery(const hdk::ir::Type* type, std::shared_ptr<const RelAlgNode> node)
      : Expr(type), node_(node) {}
  ExprPtr deep_copy() const override { return makeExpr<ScalarSubquery>(type_, node_); }

  bool operator==(const Expr& rhs) const override {
    const ScalarSubquery* rhsp = dynamic_cast<const ScalarSubquery*>(&rhs);
    return rhsp && node_ == rhsp->node_;
  }

  std::string toString() const override;

  const RelAlgNode* node() const { return node_.get(); }
  std::shared_ptr<const RelAlgNode> nodeShared() const { return node_; }

  size_t hash() const override;

 private:
  std::shared_ptr<const RelAlgNode> node_;
};

/*
 * @type InValues
 * @brief represents predicate expr IN (v1, v2, ...)
 * v1, v2, ... are can be either Constant or Parameter.
 */
class InValues : public Expr {
 public:
  InValues(ExprPtr a, const ExprPtrList& l);
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  const ExprPtrList& valueList() const { return value_list_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;                   // the argument left of IN
  const ExprPtrList value_list_;  // the list of values right of IN
};

/*
 * @type InIntegerSet
 * @brief represents predicate expr IN (v1, v2, ...) for the case where the right
 *        hand side is a list of integers or dictionary-encoded strings generated
 *        by a IN subquery. Avoids the overhead of storing a list of shared pointers
 *        to Constant objects, making it more suitable for IN sub-queries usage.
 * v1, v2, ... are integers
 */
class InIntegerSet : public Expr {
 public:
  InIntegerSet(const std::shared_ptr<const Expr> a,
               const std::vector<int64_t>& values,
               const bool not_null);

  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }

  const std::vector<int64_t>& valueList() const { return value_list_; }

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const std::shared_ptr<const Expr> arg_;  // the argument left of IN
  const std::vector<int64_t> value_list_;  // the list of values right of IN
};

class InSubquery : public Expr {
 public:
  InSubquery(const hdk::ir::Type* type,
             hdk::ir::ExprPtr arg,
             std::shared_ptr<const RelAlgNode> node)
      : Expr(type), arg_(std::move(arg)), node_(std::move(node)) {}

  ExprPtr deep_copy() const override {
    return makeExpr<InSubquery>(type_, arg_->deep_copy(), node_);
  }

  bool operator==(const Expr& rhs) const override {
    const InSubquery* rhsp = dynamic_cast<const InSubquery*>(&rhs);
    return rhsp && node_ == rhsp->node_;
  }

  std::string toString() const override;

  const hdk::ir::Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }

  const RelAlgNode* node() const { return node_.get(); }
  std::shared_ptr<const RelAlgNode> nodeShared() const { return node_; }

  size_t hash() const override;

 private:
  hdk::ir::ExprPtr arg_;
  std::shared_ptr<const RelAlgNode> node_;
};

/*
 * @type CharLengthExpr
 * @brief expression for the CHAR_LENGTH expression.
 * arg must evaluate to char, varchar or text.
 */
class CharLengthExpr : public Expr {
 public:
  CharLengthExpr(ExprPtr a, bool e)
      : Expr(a->ctx().int32(a->type()->nullable())), arg_(a), calc_encoded_length_(e) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  bool calcEncodedLength() const { return calc_encoded_length_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;
  bool calc_encoded_length_;
};

/*
 * @type KeyForStringExpr
 * @brief expression for the KEY_FOR_STRING expression.
 * arg must be a dict encoded column, not str literal.
 */
class KeyForStringExpr : public Expr {
 public:
  KeyForStringExpr(ExprPtr a) : Expr(a->ctx().int32(a->type()->nullable())), arg_(a) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;
};

/*
 * @type SampleRatioExpr
 * @brief expression for the SAMPLE_RATIO expression. Argument range is expected to be
 * between 0 and 1.
 */
class SampleRatioExpr : public Expr {
 public:
  SampleRatioExpr(ExprPtr a) : Expr(a->ctx().boolean(a->type()->nullable())), arg_(a) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;
};

/**
 * @brief Expression class for the LOWER (lowercase) string function.
 * The "arg" constructor parameter must be an expression that resolves to a string
 * datatype (e.g. TEXT).
 */
class LowerExpr : public Expr {
 public:
  LowerExpr(ExprPtr arg) : Expr(arg->type()), arg_(arg) {}

  const Expr* arg() const { return arg_.get(); }

  ExprPtr argShared() const { return arg_; }

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;

  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;
};

/*
 * @type CardinalityExpr
 * @brief expression for the CARDINALITY expression.
 * arg must evaluate to array (or multiset when supported).
 */
class CardinalityExpr : public Expr {
 public:
  CardinalityExpr(ExprPtr a) : Expr(a->ctx().int32(a->type()->nullable())), arg_(a) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;
};

/*
 * @type LikeExpr
 * @brief expression for the LIKE predicate.
 * arg must evaluate to char, varchar or text.
 */
class LikeExpr : public Expr {
 public:
  LikeExpr(ExprPtr a, ExprPtr l, ExprPtr e, bool i, bool s)
      : Expr(a->ctx().boolean(a->type()->nullable()))
      , arg_(a)
      , like_expr_(l)
      , escape_expr_(e)
      , is_ilike_(i)
      , is_simple_(s) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  const Expr* likeExpr() const { return like_expr_.get(); }
  const Expr* escapeExpr() const { return escape_expr_.get(); }
  bool isIlike() const { return is_ilike_; }
  bool isSimple() const { return is_simple_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;          // the argument to the left of LIKE
  ExprPtr like_expr_;    // expression that evaluates to like string
  ExprPtr escape_expr_;  // expression that evaluates to escape string, can be nullptr
  bool is_ilike_;        // is this ILIKE?
  bool is_simple_;  // is this simple, meaning we can use fast path search (fits '%str%'
                    // pattern with no inner '%','_','[',']'
};

/*
 * @type RegexpExpr
 * @brief expression for REGEXP.
 * arg must evaluate to char, varchar or text.
 */
class RegexpExpr : public Expr {
 public:
  RegexpExpr(ExprPtr a, ExprPtr p, ExprPtr e)
      : Expr(a->ctx().boolean(a->type()->nullable()))
      , arg_(a)
      , pattern_expr_(p)
      , escape_expr_(e) {}
  const Expr* arg() const { return arg_.get(); }
  const ExprPtr argShared() const { return arg_; }
  const Expr* patternExpr() const { return pattern_expr_.get(); }
  const Expr* escapeExpr() const { return escape_expr_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;           // the argument to the left of REGEXP
  ExprPtr pattern_expr_;  // expression that evaluates to pattern string
  ExprPtr escape_expr_;   // expression that evaluates to escape string, can be nullptr
};

/*
 * @type WidthBucketExpr
 * @brief expression for width_bucket functions.
 */
class WidthBucketExpr : public Expr {
 public:
  WidthBucketExpr(const ExprPtr target_value,
                  const ExprPtr lower_bound,
                  const ExprPtr upper_bound,
                  const ExprPtr partition_count)
      : Expr(target_value->ctx().int32(target_value->type()->nullable()))
      , target_value_(target_value)
      , lower_bound_(lower_bound)
      , upper_bound_(upper_bound)
      , partition_count_(partition_count) {}
  const Expr* targetValue() const { return target_value_.get(); }
  const Expr* lowerBound() const { return lower_bound_.get(); }
  const Expr* upperBound() const { return upper_bound_.get(); }
  const Expr* partitionCount() const { return partition_count_.get(); }
  ExprPtr deep_copy() const override;
  double boundVal(const Expr* bound_expr) const;
  int32_t partitionCountVal() const;
  template <typename T>
  int32_t computeBucket(T target_const_val, const hdk::ir::Type* type) const {
    // this utility function is useful for optimizing expression range decision
    // for an expression depending on width_bucket expr
    T null_val =
        type->isInteger() ? inline_int_null_value(type) : inline_fp_null_value(type);
    double lower_bound_val = boundVal(lower_bound_.get());
    double upper_bound_val = boundVal(upper_bound_.get());
    auto partition_count_val = partitionCountVal();
    if (target_const_val == null_val) {
      return INT32_MIN;
    }
    float res;
    if (lower_bound_val < upper_bound_val) {
      if (target_const_val < lower_bound_val) {
        return 0;
      } else if (target_const_val >= upper_bound_val) {
        return partition_count_val + 1;
      }
      double dividend = upper_bound_val - lower_bound_val;
      res = ((partition_count_val * (target_const_val - lower_bound_val)) / dividend) + 1;
    } else {
      if (target_const_val > lower_bound_val) {
        return 0;
      } else if (target_const_val <= upper_bound_val) {
        return partition_count_val + 1;
      }
      double dividend = lower_bound_val - upper_bound_val;
      res = ((partition_count_val * (lower_bound_val - target_const_val)) / dividend) + 1;
    }
    return res;
  }
  // Returns true if lower, upper and partition count exprs are constant
  bool isConstantExpr() const;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr target_value_;     // target value expression
  ExprPtr lower_bound_;      // lower_bound
  ExprPtr upper_bound_;      // upper_bound
  ExprPtr partition_count_;  // partition_count
};

/*
 * @type LikelihoodExpr
 * @brief expression for LIKELY and UNLIKELY boolean identity functions.
 */
class LikelihoodExpr : public Expr {
 public:
  LikelihoodExpr(ExprPtr a, float l = 0.5)
      : Expr(a->ctx().boolean(a->type()->nullable())), arg_(a), likelihood_(l) {}
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  float likelihood() const { return likelihood_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExprPtr arg_;  // the argument to LIKELY, UNLIKELY
  float likelihood_;
};

/*
 * @type AggExpr
 * @brief expression for builtin SQL aggregates.
 */
class AggExpr : public Expr {
 public:
  AggExpr(const Type* type,
          SQLAgg a,
          ExprPtr g,
          bool d,
          std::shared_ptr<const Constant> e)
      : Expr(type, true), agg_type_(a), arg_(g), is_distinct_(d), arg1_(e) {}
  SQLAgg aggType() const { return agg_type_; }
  const Expr* arg() const { return arg_.get(); }
  ExprPtr argShared() const { return arg_; }
  bool isDistinct() const { return is_distinct_; }
  std::shared_ptr<const Constant> arg1() const { return arg1_; }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  SQLAgg agg_type_;   // aggregate type: kAVG, kMIN, kMAX, kSUM, kCOUNT
  ExprPtr arg_;       // argument to aggregate
  bool is_distinct_;  // true only if it is for COUNT(DISTINCT x)
  // APPROX_COUNT_DISTINCT error_rate, APPROX_QUANTILE quantile
  std::shared_ptr<const Constant> arg1_;
};

/*
 * @type CaseExpr
 * @brief the CASE-WHEN-THEN-ELSE expression
 */
class CaseExpr : public Expr {
 public:
  CaseExpr(const hdk::ir::Type* type,
           bool has_agg,
           std::list<std::pair<ExprPtr, ExprPtr>> expr_pairs,
           ExprPtr e)
      : Expr(type, has_agg), expr_pairs_(std::move(expr_pairs)), else_expr_(e) {}
  const std::list<std::pair<ExprPtr, ExprPtr>>& exprPairs() const { return expr_pairs_; }
  const Expr* elseExpr() const { return else_expr_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
  ExprPtr cast(const Type* new_type, bool is_dict_intersection) const override;

  size_t hash() const override;

 private:
  std::list<std::pair<ExprPtr, ExprPtr>>
      expr_pairs_;     // a pair of expressions for each WHEN expr1 THEN expr2.  expr1
                       // must be of boolean type.  all expr2's must be of compatible
                       // types and will be promoted to the common type.
  ExprPtr else_expr_;  // expression for ELSE.  nullptr if omitted.
};

/*
 * @type ExtractExpr
 * @brief the EXTRACT expression
 */
class ExtractExpr : public Expr {
 public:
  ExtractExpr(const hdk::ir::Type* type, bool has_agg, ExtractField f, ExprPtr e)
      : Expr(type, has_agg), field_(f), from_expr_(e) {}
  ExtractField field() const { return field_; }
  const Expr* from() const { return from_expr_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  ExtractField field_;
  ExprPtr from_expr_;
};

/*
 * @type DateAddExpr
 * @brief the DATEADD expression
 */
class DateAddExpr : public Expr {
 public:
  DateAddExpr(const hdk::ir::Type* type,
              const DateAddField f,
              const ExprPtr number,
              const ExprPtr datetime)
      : Expr(type, false), field_(f), number_(number), datetime_(datetime) {}
  DateAddField field() const { return field_; }
  const Expr* number() const { return number_.get(); }
  const Expr* datetime() const { return datetime_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const DateAddField field_;
  const ExprPtr number_;
  const ExprPtr datetime_;
};

/*
 * @type DateDiffExpr
 * @brief the DATEDIFF expression
 */
class DateDiffExpr : public Expr {
 public:
  DateDiffExpr(const hdk::ir::Type* type,
               const DateTruncField f,
               const ExprPtr start,
               const ExprPtr end)
      : Expr(type, false), field_(f), start_(start), end_(end) {}
  DateTruncField field() const { return field_; }
  const Expr* start() const { return start_.get(); }
  const Expr* end() const { return end_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const DateTruncField field_;
  const ExprPtr start_;
  const ExprPtr end_;
};

/*
 * @type DateTruncExpr
 * @brief the DATE_TRUNC expression
 */
class DateTruncExpr : public Expr {
 public:
  DateTruncExpr(const hdk::ir::Type* type, bool has_agg, DateTruncField f, ExprPtr e)
      : Expr(type, has_agg), field_(f), from_expr_(e) {}
  DateTruncField field() const { return field_; }
  const Expr* from() const { return from_expr_.get(); }
  ExprPtr deep_copy() const override;
  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  DateTruncField field_;
  ExprPtr from_expr_;
};

class FunctionOper : public Expr {
 public:
  FunctionOper(const hdk::ir::Type* type,
               const std::string& name,
               const ExprPtrVector& args)
      : Expr(type, false), name_(name), args_(args) {}

  const std::string& name() const { return name_; }

  size_t arity() const { return args_.size(); }

  const Expr* arg(const size_t i) const {
    CHECK_LT(i, args_.size());
    return args_[i].get();
  }

  ExprPtr argShared(const size_t i) const {
    CHECK_LT(i, args_.size());
    return args_[i];
  }

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  size_t hash() const override;

 private:
  const std::string name_;
  const ExprPtrVector args_;
};

class FunctionOperWithCustomTypeHandling : public FunctionOper {
 public:
  FunctionOperWithCustomTypeHandling(const hdk::ir::Type* type,
                                     const std::string& name,
                                     const ExprPtrVector& args)
      : FunctionOper(type, name, args) {}

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
};

/*
 * @type OffsetInFragment
 * @brief The offset of a row in the current fragment. To be used by updates.
 */
class OffsetInFragment : public Expr {
 public:
  OffsetInFragment() : Expr(hdk::ir::Context::defaultCtx().int64(false)){};

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;
};

/*
 * @type OrderEntry
 * @brief represents an entry in ORDER BY clause.
 */
struct OrderEntry {
  OrderEntry(int t, bool d, bool nf) : tle_no(t), is_desc(d), nulls_first(nf){};
  ~OrderEntry() {}

  std::string toString() const;
  void print() const;

  size_t hash() const {
    size_t res = 0;
    boost::hash_combine(res, tle_no);
    boost::hash_combine(res, is_desc);
    boost::hash_combine(res, nulls_first);
    return res;
  }

  int tle_no;       /* targetlist entry number: 1-based */
  bool is_desc;     /* true if order is DESC */
  bool nulls_first; /* true if nulls are ordered first.  otherwise last. */
};

/*
 * @type WindowFunction
 * @brief A window function.
 */
class WindowFunction : public Expr {
 public:
  WindowFunction(const hdk::ir::Type* type,
                 const SqlWindowFunctionKind kind,
                 const ExprPtrVector& args,
                 const ExprPtrVector& partition_keys,
                 const ExprPtrVector& order_keys,
                 const std::vector<OrderEntry>& collation)
      : Expr(type)
      , kind_(kind)
      , args_(args)
      , partition_keys_(partition_keys)
      , order_keys_(order_keys)
      , collation_(collation){};

  ExprPtr deep_copy() const override;

  bool operator==(const Expr& rhs) const override;
  std::string toString() const override;

  SqlWindowFunctionKind kind() const { return kind_; }

  const ExprPtrVector& args() const { return args_; }

  const ExprPtrVector& getPartitionKeys() const { return partition_keys_; }

  const ExprPtrVector& getOrderKeys() const { return order_keys_; }

  const std::vector<OrderEntry>& getCollation() const { return collation_; }

  size_t hash() const override;

 private:
  const SqlWindowFunctionKind kind_;
  const ExprPtrVector args_;
  const ExprPtrVector partition_keys_;
  const ExprPtrVector order_keys_;
  const std::vector<OrderEntry> collation_;
};

/*
 * @type ArrayExpr
 * @brief Corresponds to ARRAY[] statements in SQL
 */

class ArrayExpr : public Expr {
 public:
  ArrayExpr(const hdk::ir::Type* array_type,
            ExprPtrVector const& array_exprs,
            bool is_null = false,
            bool local_alloc = false)
      : Expr(array_type)
      , contained_expressions_(array_exprs)
      , local_alloc_(local_alloc)
      , is_null_(is_null) {}

  ExprPtr deep_copy() const override;
  std::string toString() const override;
  bool operator==(Expr const& rhs) const override;
  size_t getElementCount() const { return contained_expressions_.size(); }
  bool isLocalAlloc() const { return local_alloc_; }
  bool isNull() const { return is_null_; }

  const Expr* getElement(const size_t i) const {
    CHECK_LT(i, contained_expressions_.size());
    return contained_expressions_[i].get();
  }

  size_t hash() const override;

 private:
  ExprPtrVector contained_expressions_;
  bool local_alloc_;
  bool is_null_;  // constant is NULL
};

/*
 * @type TargetEntry
 * @brief Target list defines a relational projection.  It is a list of TargetEntry's.
 */
class TargetEntry {
 public:
  TargetEntry(const std::string& n, ExprPtr e, bool u) : resname(n), expr(e), unnest(u) {}
  virtual ~TargetEntry() {}
  const std::string& get_resname() const { return resname; }
  void set_resname(const std::string& name) { resname = name; }
  const Expr* get_expr() const { return expr.get(); }
  ExprPtr get_own_expr() const { return expr; }
  void set_expr(ExprPtr e) { expr = e; }
  bool get_unnest() const { return unnest; }
  std::string toString() const;
  void print() const;

  size_t hash() const;

 private:
  std::string resname;  // alias name, e.g., SELECT salary + bonus AS compensation,
  ExprPtr expr;         // expression to evaluate for the value
  bool unnest;          // unnest a collection type
};

// Returns true iff the two expression lists are equal (same size and each element are
// equal).
bool expr_list_match(const ExprPtrVector& lhs, const ExprPtrVector& rhs);

}  // namespace hdk::ir
