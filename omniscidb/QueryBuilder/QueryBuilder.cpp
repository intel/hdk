/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryBuilder.h"

#include "Analyzer/Analyzer.h"
#include "IR/ExprCollector.h"
#include "IR/ExprRewriter.h"
#include "Shared/SqlTypesLayout.h"

#include <boost/algorithm/string.hpp>
#include <numeric>

namespace hdk::ir {

namespace {

int normalizeColIndex(const Node* node, int col_idx) {
  int size = static_cast<int>(node->size());
  if (col_idx >= size || col_idx < -size) {
    throw InvalidQueryError() << "Out-of-border column index.\n"
                              << "  Column index: " << col_idx << "\n"
                              << "  Node: " << node->toString() << "\n";
  }
  return col_idx < 0 ? size + col_idx : col_idx;
}

std::unordered_set<std::string> getColNames(const Node* node) {
  std::unordered_set<std::string> res;
  res.reserve(node->size());
  auto scan = node->as<Scan>();
  for (size_t col_idx = 0; col_idx < node->size(); ++col_idx) {
    if (!scan || !scan->isVirtualCol(col_idx)) {
      res.insert(node->getFieldName(col_idx));
    }
  }
  return res;
}

std::string getFieldName(const Node* node, int col_idx) {
  col_idx = normalizeColIndex(node, col_idx);
  return node->getFieldName(col_idx);
}

ExprPtr getRefByName(const Node* node,
                     const std::string& col_name,
                     bool allow_null_res = false);

ExprPtr getRefByName(const Join* join, const std::string& col_name) {
  auto lhs_input_ref = getRefByName(join->getInput(0), col_name, true);
  auto rhs_input_ref = getRefByName(join->getInput(1), col_name, true);
  if (lhs_input_ref && rhs_input_ref) {
    throw InvalidQueryError() << "ambiguous column name '" << col_name
                              << "' for Join node: " << join->toString();
  }
  return lhs_input_ref ? lhs_input_ref : rhs_input_ref;
}

ExprPtr getRefByName(const Node* node, const std::string& col_name, bool allow_null_res) {
  ExprPtr res = nullptr;
  if (auto join = node->as<Join>()) {
    res = getRefByName(join, col_name);
  } else {
    for (size_t i = 0; i < node->size(); ++i) {
      if (node->getFieldName(i) == col_name) {
        res = getNodeColumnRef(node, (unsigned)i);
      }
    }
  }

  if (!res && !allow_null_res) {
    throw InvalidQueryError() << "getRefByName error: unknown column name.\n"
                              << "  Column name: " << col_name << "\n"
                              << "  Node: " << node->toString() << "\n";
  }

  return res;
}

ExprPtr getRefByIndex(const Node* node, int col_idx) {
  col_idx = normalizeColIndex(node, col_idx);
  return getNodeColumnRef(node, col_idx);
}

std::string chooseName(const std::string& name,
                       const std::unordered_set<std::string>& names) {
  std::string prefix;
  if (name.empty()) {
    prefix = "expr_";
  } else if (names.count(name)) {
    prefix = name + "_";
  } else {
    return name;
  }

  size_t idx = 1;
  std::string candidate;
  do {
    candidate = prefix + std::to_string(idx++);
  } while (names.count(candidate));

  return candidate;
}

std::vector<std::string> buildFieldNames(const std::vector<BuilderExpr>& exprs) {
  std::unordered_set<std::string> names;
  // First check all manually set names are unique.
  for (auto& expr : exprs) {
    if (!expr.isAutoNamed()) {
      if (expr.name().empty()) {
        throw InvalidQueryError() << "Empty field names are not allowed";
      }
      auto pr = names.insert(expr.name());
      if (!pr.second) {
        throw InvalidQueryError() << "Duplicated field name: " << expr.name();
      }
    }
  }

  // Build the resulting vector adding suffixes to auto-names when needed.
  std::vector<std::string> res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    if (expr.isAutoNamed()) {
      auto name = chooseName(expr.name(), names);
      auto pr = names.insert(name);
      CHECK(pr.second);
      res.emplace_back(std::move(name));
    } else {
      res.emplace_back(expr.name());
    }
  }

  return res;
}

ExprPtrVector collectExprs(const std::vector<BuilderExpr>& exprs) {
  ExprPtrVector res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    res.emplace_back(expr.expr());
  }
  return res;
}

class InputNodesCollector
    : public ExprCollector<std::unordered_set<const Node*>, InputNodesCollector> {
 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    result_.insert(col_ref->node());
  }
};

class InputColIndexesCollector
    : public ExprCollector<std::unordered_set<int>, InputColIndexesCollector> {
 protected:
  void visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    result_.insert(col_ref->index());
  }
};

void expandAllowedInput(const Node* node,
                        std::unordered_set<const Node*>& allowed_nodes) {
  allowed_nodes.insert(node);
  if (node->is<Join>()) {
    expandAllowedInput(node->getInput(0), allowed_nodes);
    expandAllowedInput(node->getInput(1), allowed_nodes);
  }
}

std::unordered_set<const Node*> expandAllowedInput(
    const std::vector<const Node*>& nodes) {
  std::unordered_set<const Node*> allowed;
  for (auto node : nodes) {
    expandAllowedInput(node, allowed);
  }
  return allowed;
}

void checkExprInput(const ExprPtr& expr,
                    const std::unordered_set<const Node*>& allowed_nodes,
                    const std::string& node_name) {
  auto expr_nodes = InputNodesCollector::collect(expr);
  for (auto& node : expr_nodes) {
    if (!allowed_nodes.count(node)) {
      std::stringstream ss;
      ss << "Wrong expression in a " << node_name
         << ": non-input node is referenced by an expression." << std::endl
         << "  Expression: " << expr->toString() << std::endl
         << "  Referenced node: " << node->toString() << std::endl
         << "  Input nodes:" << std::endl;
      for (auto node : allowed_nodes) {
        ss << "    " << node->toString() << std::endl;
      }
      throw InvalidQueryError(ss.str());
    }
  }
}

void checkExprInput(const ExprPtrVector& exprs,
                    const std::vector<const Node*>& nodes,
                    const std::string& node_name) {
  std::unordered_set<const Node*> allowed = expandAllowedInput(nodes);
  for (auto& expr : exprs) {
    checkExprInput(expr, allowed, node_name);
  }
}

void checkExprInput(const std::vector<BuilderExpr>& exprs,
                    const std::vector<const Node*>& nodes,
                    const std::string& node_name) {
  std::unordered_set<const Node*> allowed = expandAllowedInput(nodes);
  for (auto& expr : exprs) {
    checkExprInput(expr.expr(), allowed, node_name);
  }
}

void checkExprInput(const BuilderExpr& expr,
                    const std::vector<const Node*>& nodes,
                    const std::string& node_name) {
  std::unordered_set<const Node*> allowed = expandAllowedInput(nodes);
  checkExprInput(expr.expr(), allowed, node_name);
}

bool isIdentShuffle(const std::vector<int>& shuffle) {
  for (int i = 0; i < (int)shuffle.size(); ++i) {
    if (i != shuffle[i]) {
      return false;
    }
  }
  return true;
}

class InputRewriter : public ExprRewriter {
 public:
  InputRewriter(const hdk::ir::Node* new_base, const std::vector<int>& new_indexes)
      : new_base_(new_base), new_indexes_(new_indexes) {}

  hdk::ir::ExprPtr visitColumnRef(const hdk::ir::ColumnRef* col_ref) override {
    CHECK_LT((size_t)col_ref->index(), new_indexes_.size());
    return hdk::ir::makeExpr<hdk::ir::ColumnRef>(
        col_ref->type(), new_base_, new_indexes_[col_ref->index()]);
  }

 private:
  const hdk::ir::Node* new_base_;
  const std::vector<int>& new_indexes_;
};

std::vector<BuilderExpr> replaceInput(const std::vector<BuilderExpr>& exprs,
                                      const Node* new_base,
                                      const std::vector<int> new_indexes) {
  InputRewriter rewriter(new_base, new_indexes);
  std::vector<BuilderExpr> res;
  res.reserve(exprs.size());
  for (auto& expr : exprs) {
    res.emplace_back(expr.rewrite(rewriter));
  }
  return res;
}

void checkCstArrayType(const Type* type, size_t elems) {
  if (!type->isArray()) {
    throw InvalidQueryError()
        << "Only array types can be used to translate a vector to a literal. Provided: "
        << type->toString();
  }
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  // Only few types are actually supported by codegen.
  if (!elem_type->isInt8() && !elem_type->isInt32() && !elem_type->isFp64()) {
    throw InvalidQueryError() << "Only int8, int32, and fp64 elements are supported in "
                                 "array literals. Requested: "
                              << elem_type->toString();
  }
  if (type->isFixedLenArray()) {
    auto num_elems = type->as<FixedLenArrayType>()->numElems();
    if (static_cast<size_t>(num_elems) != elems) {
      throw InvalidQueryError()
          << "Literal array elements count mismatch. Expected " << num_elems
          << " elements, provided " << elems << " elements.";
    }
  }
}

DateAddField timeUnitToDateAddField(TimeUnit unit) {
  switch (unit) {
    case TimeUnit::kMonth:
      return DateAddField::kMonth;
    case TimeUnit::kDay:
      return DateAddField::kDay;
    case TimeUnit::kSecond:
      return DateAddField::kSecond;
    case TimeUnit::kMilli:
      return DateAddField::kMilli;
    case TimeUnit::kMicro:
      return DateAddField::kMicro;
    case TimeUnit::kNano:
      return DateAddField::kNano;
    default:
      break;
  }
  throw InvalidQueryError() << "unknown time unit: " << unit;
}

DateAddField parseDateAddField(const std::string& field) {
  static const std::unordered_map<std::string, DateAddField> field_names = {
      {"year", DateAddField::kYear},
      {"years", DateAddField::kYear},
      {"quarter", DateAddField::kQuarter},
      {"quarters", DateAddField::kQuarter},
      {"month", DateAddField::kMonth},
      {"months", DateAddField::kMonth},
      {"day", DateAddField::kDay},
      {"days", DateAddField::kDay},
      {"hour", DateAddField::kHour},
      {"hours", DateAddField::kHour},
      {"min", DateAddField::kMinute},
      {"mins", DateAddField::kMinute},
      {"minute", DateAddField::kMinute},
      {"minutes", DateAddField::kMinute},
      {"sec", DateAddField::kSecond},
      {"secs", DateAddField::kSecond},
      {"second", DateAddField::kSecond},
      {"seconds", DateAddField::kSecond},
      {"millennium", DateAddField::kMillennium},
      {"millenniums", DateAddField::kMillennium},
      {"century", DateAddField::kCentury},
      {"centuries", DateAddField::kCentury},
      {"decade", DateAddField::kDecade},
      {"decades", DateAddField::kDecade},
      {"ms", DateAddField::kMilli},
      {"milli", DateAddField::kMilli},
      {"millisecond", DateAddField::kMilli},
      {"milliseconds", DateAddField::kMilli},
      {"us", DateAddField::kMicro},
      {"micro", DateAddField::kMicro},
      {"microsecond", DateAddField::kMicro},
      {"microseconds", DateAddField::kMicro},
      {"ns", DateAddField::kNano},
      {"nano", DateAddField::kNano},
      {"nanosecond", DateAddField::kNano},
      {"nanoseconds", DateAddField::kNano},
      {"week", DateAddField::kWeek},
      {"weeks", DateAddField::kWeek},
      {"quarterday", DateAddField::kQuarterDay},
      {"quarter_day", DateAddField::kQuarterDay},
      {"quarter day", DateAddField::kQuarterDay},
      {"quarterdays", DateAddField::kQuarterDay},
      {"quarter_days", DateAddField::kQuarterDay},
      {"quarter days", DateAddField::kQuarterDay},
      {"weekday", DateAddField::kWeekDay},
      {"week_day", DateAddField::kWeekDay},
      {"week day", DateAddField::kWeekDay},
      {"weekdays", DateAddField::kWeekDay},
      {"week_days", DateAddField::kWeekDay},
      {"week days", DateAddField::kWeekDay},
      {"dayofyear", DateAddField::kDayOfYear},
      {"day_of_year", DateAddField::kDayOfYear},
      {"day of year", DateAddField::kDayOfYear},
      {"doy", DateAddField::kDayOfYear}};
  auto canonical = boost::trim_copy(boost::to_lower_copy(field));
  if (!field_names.count(canonical)) {
    throw InvalidQueryError() << "Cannot parse date add field: '" << field << "'";
  }
  return field_names.at(canonical);
}

JoinType parseJoinType(const std::string& join_type) {
  static const std::unordered_map<std::string, JoinType> join_types = {
      {"left", JoinType::LEFT},
      {"inner", JoinType::INNER},
      {"semi", JoinType::SEMI},
      {"anti", JoinType::ANTI}};
  auto canonical = boost::trim_copy(boost::to_lower_copy(join_type));
  if (!join_types.count(canonical)) {
    throw InvalidQueryError() << "Cannot parse join type: '" << join_type << "'";
  }
  return join_types.at(canonical);
}

void collectNodes(NodePtr node,
                  std::vector<NodePtr>& nodes,
                  std::unordered_set<const Node*>& visited) {
  for (size_t i = 0; i < node->inputCount(); ++i) {
    collectNodes(node->getAndOwnInput(i), nodes, visited);
  }

  if (!visited.count(node.get())) {
    nodes.push_back(node);
    visited.insert(node.get());
  }
}

std::vector<NodePtr> collectNodes(NodePtr node) {
  std::vector<NodePtr> nodes;
  std::unordered_set<const Node*> visited;
  collectNodes(node, nodes, visited);
  return nodes;
}

/**
 * Check if expr is WindowFunction or AggExpr that can be transformed
 * into corresponding WindowFunction.
 */
std::shared_ptr<const WindowFunction> checkOrGetWindowFn(ExprPtr expr) {
  if (expr->is<WindowFunction>()) {
    return std::dynamic_pointer_cast<const WindowFunction>(expr);
  }

  if (auto agg = expr->as<AggExpr>()) {
    switch (agg->aggType()) {
      case AggType::kCount:
        if (agg->arg()) {
          return std::shared_ptr<const WindowFunction>(new WindowFunction(
              agg->type(), WindowFunctionKind::Count, {agg->argShared()}, {}, {}, {}));
        } else {
          return std::shared_ptr<const WindowFunction>(
              new WindowFunction(agg->type(), WindowFunctionKind::Count, {}, {}, {}, {}));
        }
      case AggType::kAvg:
        return std::shared_ptr<const WindowFunction>(new WindowFunction(
            agg->type(), WindowFunctionKind::Avg, {agg->argShared()}, {}, {}, {}));
      case AggType::kMin:
        return std::shared_ptr<const WindowFunction>(new WindowFunction(
            agg->type(), WindowFunctionKind::Min, {agg->argShared()}, {}, {}, {}));
      case AggType::kMax:
        return std::shared_ptr<const WindowFunction>(new WindowFunction(
            agg->type(), WindowFunctionKind::Max, {agg->argShared()}, {}, {}, {}));
      case AggType::kSum:
        return std::shared_ptr<const WindowFunction>(new WindowFunction(
            agg->type(), WindowFunctionKind::Sum, {agg->argShared()}, {}, {}, {}));
      default:
        break;
    }
  }

  return nullptr;
}

}  // namespace

BuilderOrderByKey::BuilderOrderByKey()
    : expr_(nullptr)
    , dir_(SortDirection::Ascending)
    , null_pos_(NullSortedPosition::Last) {}

BuilderOrderByKey::BuilderOrderByKey(const BuilderExpr& expr,
                                     SortDirection dir,
                                     NullSortedPosition null_pos)
    : expr_(expr.expr()), dir_(dir), null_pos_(null_pos) {}

BuilderOrderByKey::BuilderOrderByKey(const BuilderExpr& expr,
                                     const std::string& dir,
                                     const std::string& null_pos)
    : expr_(expr.expr())
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {}

SortDirection BuilderOrderByKey::parseSortDirection(const std::string& val) {
  return BuilderSortField::parseSortDirection(val);
}

NullSortedPosition BuilderOrderByKey::parseNullPosition(const std::string& val) {
  return BuilderSortField::parseNullPosition(val);
}

BuilderExpr::BuilderExpr() : builder_(nullptr) {}

BuilderExpr::BuilderExpr(const QueryBuilder* builder,
                         ExprPtr expr,
                         const std::string& name,
                         bool auto_name)
    : builder_(builder), expr_(expr), name_(name), auto_name_(auto_name) {}

BuilderExpr BuilderExpr::rename(const std::string& name) const {
  return {builder_, expr_, name, false};
}

BuilderExpr BuilderExpr::avg() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for avg aggregate: "
                              << expr_->type()->toString();
  }
  auto agg =
      makeExpr<AggExpr>(builder_->ctx_.fp64(), AggType::kAvg, expr_, false, nullptr);
  auto name = name_.empty() ? "avg" : name_ + "_avg";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::min() const {
  if (!expr_->type()->isNumber() && !expr_->type()->isDateTime()) {
    throw InvalidQueryError() << "Unsupported type for min aggregate: "
                              << expr_->type()->toString();
  }
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kMin, expr_, false, nullptr);
  auto name = name_.empty() ? "min" : name_ + "_min";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::max() const {
  if (!expr_->type()->isNumber() && !expr_->type()->isDateTime()) {
    throw InvalidQueryError() << "Unsupported type for max aggregate: "
                              << expr_->type()->toString();
  }
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kMax, expr_, false, nullptr);
  auto name = name_.empty() ? "max" : name_ + "_max";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::sum() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for sum aggregate: "
                              << expr_->type()->toString();
  }
  auto res_type = expr_->type();
  if (res_type->isInteger() && res_type->size() < 8) {
    res_type = builder_->ctx_.int64(res_type->nullable());
  }
  auto agg = makeExpr<AggExpr>(res_type, AggType::kSum, expr_, false, nullptr);
  auto name = name_.empty() ? "sum" : name_ + "_sum";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::count(bool is_distinct) const {
  if (!expr_->is<hdk::ir::ColumnRef>()) {
    throw InvalidQueryError()
        << "Count method is valid for column references only. Used for: "
        << expr_->toString();
  }
  auto count_type = builder_->config_->exec.group_by.bigint_count
                        ? builder_->ctx_.int64(false)
                        : builder_->ctx_.int32(false);
  auto agg = makeExpr<AggExpr>(count_type, AggType::kCount, expr_, is_distinct, nullptr);
  auto name = name_.empty() ? "count" : name_ + "_count";
  if (is_distinct) {
    name += "_dist";
  }
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::approxCountDist() const {
  if (!expr_->is<hdk::ir::ColumnRef>()) {
    throw InvalidQueryError()
        << "ApproxCountDist method is valid for column references only. Used for: "
        << expr_->toString();
  }
  auto count_type = builder_->config_->exec.group_by.bigint_count
                        ? builder_->ctx_.int64(false)
                        : builder_->ctx_.int32(false);
  auto agg =
      makeExpr<AggExpr>(count_type, AggType::kApproxCountDistinct, expr_, true, nullptr);
  auto name = name_.empty() ? "approx_count_dist" : name_ + "_approx_count_dist";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::approxQuantile(double val) const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Unsupported type for sum aggregate: "
                              << expr_->type()->toString();
  }
  if (val < 0.0 || val > 1.0) {
    throw InvalidQueryError()
        << "ApproxQuantile expects argument between 0.0 and 1.0 but got " << val;
  }
  Datum d;
  d.doubleval = val;
  auto cst = makeExpr<Constant>(builder_->ctx_.fp64(), false, d);
  auto agg = makeExpr<AggExpr>(
      builder_->ctx_.fp64(), AggType::kApproxQuantile, expr_, false, cst);
  auto name = name_.empty() ? "approx_quantile" : name_ + "_approx_quantile";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::sample() const {
  auto agg = makeExpr<AggExpr>(expr_->type(), AggType::kSample, expr_, false, nullptr);
  auto name = name_.empty() ? "sample" : name_ + "_sample";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::singleValue() const {
  if (expr_->type()->isVarLen()) {
    throw InvalidQueryError() << "Varlen type " << expr_->type()->toString()
                              << " is not suported for single value aggregate.";
  }
  auto agg =
      makeExpr<AggExpr>(expr_->type(), AggType::kSingleValue, expr_, false, nullptr);
  auto name = name_.empty() ? "single_value" : name_ + "_single_value";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::stdDev() const {
  if (!expr_->type()->isNumber()) {
    throw InvalidQueryError() << "Non-numeric type " << expr_->type()->toString()
                              << " is not allowed for STDDEV aggregate.";
  }
  auto agg = makeExpr<AggExpr>(
      builder_->ctx_.fp64(), AggType::kStdDevSamp, expr_, false, nullptr);
  auto name = name_.empty() ? "stddev" : name_ + "_stddev";
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::corr(const BuilderExpr& arg) const {
  if (!type()->isNumber() || !arg.type()->isNumber()) {
    throw InvalidQueryError()
        << "Cannot apply CORR aggregate to non-numeric types. Provided: "
        << type()->toString() << " and " << arg.type()->toString();
  }
  if (expr()->is<Constant>() || arg.expr()->is<Constant>()) {
    throw InvalidQueryError()
        << "Literals are not allowed in CORR aggregate. Provided arguments: "
        << expr_->toString() << " and " << arg.expr()->toString();
  }
  auto agg =
      makeExpr<AggExpr>(builder_->ctx_.fp64(), AggType::kCorr, expr_, false, arg.expr());
  auto name = name_.empty() ? "corr" : name_ + "_corr";
  if (!arg.name_.empty()) {
    name += "_" + arg.name_;
  }
  return {builder_, agg, name, true};
}

BuilderExpr BuilderExpr::lag(int n) const {
  ExprPtr expr{new WindowFunction(expr_->type(),
                                  WindowFunctionKind::Lag,
                                  {expr_, builder_->cst(n).expr()},
                                  {},
                                  {},
                                  {})};
  auto name = name_.empty() ? "lag" : name_ + "_lag";
  return {builder_, expr, name, true};
}

BuilderExpr BuilderExpr::lead(int n) const {
  ExprPtr expr{new WindowFunction(expr_->type(),
                                  WindowFunctionKind::Lead,
                                  {expr_, builder_->cst(n).expr()},
                                  {},
                                  {},
                                  {})};
  auto name = name_.empty() ? "lead" : name_ + "_lead";
  return {builder_, expr, name, true};
}

BuilderExpr BuilderExpr::firstValue() const {
  ExprPtr expr{new WindowFunction(
      expr_->type(), WindowFunctionKind::FirstValue, {expr_}, {}, {}, {})};
  auto name = name_.empty() ? "first_value" : name_ + "_first_value";
  return {builder_, expr, name, true};
}

BuilderExpr BuilderExpr::lastValue() const {
  ExprPtr expr{new WindowFunction(
      expr_->type(), WindowFunctionKind::LastValue, {expr_}, {}, {}, {})};
  auto name = name_.empty() ? "last_value" : name_ + "_last_value";
  return {builder_, expr, name, true};
}

BuilderExpr BuilderExpr::agg(const std::string& agg_str, const BuilderExpr& arg) const {
  static const std::unordered_map<std::string, AggType> agg_names = {
      {"count", AggType::kCount},
      {"count_dist", AggType::kCount},
      {"count_distinct", AggType::kCount},
      {"count dist", AggType::kCount},
      {"count distinct", AggType::kCount},
      {"sum", AggType::kSum},
      {"min", AggType::kMin},
      {"max", AggType::kMax},
      {"avg", AggType::kAvg},
      {"approx_count_dist", AggType::kApproxCountDistinct},
      {"approx_count_distinct", AggType::kApproxCountDistinct},
      {"approx count dist", AggType::kApproxCountDistinct},
      {"approx count distinct", AggType::kApproxCountDistinct},
      {"approx_quantile", AggType::kApproxQuantile},
      {"approx quantile", AggType::kApproxQuantile},
      {"sample", AggType::kSample},
      {"single_value", AggType::kSingleValue},
      {"single value", AggType::kSingleValue},
      {"stddev", AggType::kStdDevSamp},
      {"stddev_samp", AggType::kStdDevSamp},
      {"stddev samp", AggType::kStdDevSamp},
      {"corr", AggType::kCorr}};
  static const std::unordered_set<std::string> distinct_names = {
      "count_dist", "count_distinct", "count dist", "count distinct"};
  auto agg_str_lower = boost::algorithm::to_lower_copy(agg_str);
  if (!agg_names.count(agg_str_lower)) {
    throw InvalidQueryError() << "Unknown aggregate name: " << agg_str;
  }

  auto kind = agg_names.at(agg_str_lower);
  if (kind == AggType::kApproxQuantile && !arg.expr()) {
    throw InvalidQueryError("Missing argument for approximate quantile aggregate.");
  }
  if (kind == AggType::kCorr && !arg.expr()) {
    throw InvalidQueryError("Missing argument for corr aggregate.");
  }

  auto is_distinct = distinct_names.count(agg_str_lower);
  return agg(kind, is_distinct, arg);
}

BuilderExpr BuilderExpr::agg(const std::string& agg_str, double val) const {
  BuilderExpr arg;
  if (val != HUGE_VAL) {
    arg = builder_->cst(val);
  }
  return agg(agg_str, arg);
}

BuilderExpr BuilderExpr::agg(AggType agg_kind, const BuilderExpr& arg) const {
  return agg(agg_kind, false, arg);
}

BuilderExpr BuilderExpr::agg(AggType agg_kind, double val) const {
  return agg(agg_kind, false, val);
}

BuilderExpr BuilderExpr::agg(AggType agg_kind,
                             bool is_distinct,
                             const BuilderExpr& arg) const {
  if (is_distinct && agg_kind != AggType::kCount) {
    throw InvalidQueryError() << "Distinct property cannot be set to true for "
                              << agg_kind << " aggregate.";
  }
  if (arg.expr() && agg_kind != AggType::kApproxQuantile && agg_kind != AggType::kCorr) {
    throw InvalidQueryError() << "Aggregate argument is supported for approximate "
                                 "quantile and corr only but provided for "
                              << agg_kind;
  }
  if (agg_kind == AggType::kApproxQuantile) {
    if (!arg.expr()->is<Constant>() || !arg.type()->isFloatingPoint()) {
      throw InvalidQueryError() << "Expected fp constant argumnt for approximate "
                                   "quantile. Provided: "
                                << arg.expr()->toString();
    }
  }

  switch (agg_kind) {
    case AggType::kAvg:
      return avg();
    case AggType::kMin:
      return min();
    case AggType::kMax:
      return max();
    case AggType::kSum:
      return sum();
    case AggType::kCount:
      return count(is_distinct);
    case AggType::kApproxCountDistinct:
      return approxCountDist();
    case AggType::kApproxQuantile:
      return approxQuantile(arg.expr()->as<Constant>()->fpVal());
    case AggType::kSample:
      return sample();
    case AggType::kSingleValue:
      return singleValue();
    case AggType::kStdDevSamp:
      return stdDev();
    case AggType::kCorr:
      return corr(arg);
    default:
      break;
  }
  throw InvalidQueryError() << "Unsupported aggregate type: " << agg_kind;
}

BuilderExpr BuilderExpr::agg(AggType agg_kind, bool is_distinct, double val) const {
  BuilderExpr arg;
  if (val != HUGE_VAL) {
    arg = builder_->cst(val);
  }
  return agg(agg_kind, is_distinct, arg);
}

BuilderExpr BuilderExpr::extract(DateExtractField field) const {
  if (!expr_->type()->isDateTime()) {
    throw InvalidQueryError()
        << "Only datetime types are allowed for extract operation. Actual type: "
        << expr_->type()->toString();
  }
  auto extract_expr =
      makeExpr<ExtractExpr>(builder_->ctx_.int64(expr_->type()->nullable()),
                            expr_->containsAgg(),
                            field,
                            expr_->decompress());
  return {builder_, extract_expr, "", true};
}

BuilderExpr BuilderExpr::extract(const std::string& field) const {
  static const std::unordered_map<std::string, DateExtractField> allowed_values = {
      {"year", DateExtractField::kYear},
      {"quarter", DateExtractField::kQuarter},
      {"month", DateExtractField::kMonth},
      {"day", DateExtractField::kDay},
      {"hour", DateExtractField::kHour},
      {"min", DateExtractField::kMinute},
      {"minute", DateExtractField::kMinute},
      {"sec", DateExtractField::kSecond},
      {"second", DateExtractField::kSecond},
      {"milli", DateExtractField::kMilli},
      {"millisecond", DateExtractField::kMilli},
      {"micro", DateExtractField::kMicro},
      {"microsecond", DateExtractField::kMicro},
      {"nano", DateExtractField::kNano},
      {"nanosecond", DateExtractField::kNano},
      {"dow", DateExtractField::kDayOfWeek},
      {"dayofweek", DateExtractField::kDayOfWeek},
      {"day_of_week", DateExtractField::kDayOfWeek},
      {"day of week", DateExtractField::kDayOfWeek},
      {"isodow", DateExtractField::kIsoDayOfWeek},
      {"isodayofweek", DateExtractField::kIsoDayOfWeek},
      {"iso_day_of_week", DateExtractField::kIsoDayOfWeek},
      {"iso day of week", DateExtractField::kIsoDayOfWeek},
      {"doy", DateExtractField::kDayOfYear},
      {"dayofyear", DateExtractField::kDayOfYear},
      {"day_of_year", DateExtractField::kDayOfYear},
      {"day of year", DateExtractField::kDayOfYear},
      {"epoch", DateExtractField::kEpoch},
      {"quarterday", DateExtractField::kQuarterDay},
      {"quarter_day", DateExtractField::kQuarterDay},
      {"quarter day", DateExtractField::kQuarterDay},
      {"week", DateExtractField::kWeek},
      {"weeksunday", DateExtractField::kWeekSunday},
      {"week_sunday", DateExtractField::kWeekSunday},
      {"week sunday", DateExtractField::kWeekSunday},
      {"weeksaturday", DateExtractField::kWeekSaturday},
      {"week_saturday", DateExtractField::kWeekSaturday},
      {"week saturday", DateExtractField::kWeekSaturday},
      {"dateepoch", DateExtractField::kDateEpoch},
      {"date_epoch", DateExtractField::kDateEpoch},
      {"date epoch", DateExtractField::kDateEpoch}};
  auto canonical = boost::trim_copy(boost::to_lower_copy(field));
  if (!allowed_values.count(canonical)) {
    throw InvalidQueryError() << "Cannot parse date extract field: '" << field << "'";
  }
  return extract(allowed_values.at(canonical));
}

BuilderExpr BuilderExpr::cast(const Type* new_type) const {
  if (expr_->type()->isInteger()) {
    if (new_type->isNumber() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isBoolean()) {
      return ne(builder_->cst(0, expr_->type()));
    }
  } else if (expr_->type()->isFloatingPoint()) {
    if (new_type->isInteger() || new_type->isFloatingPoint()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isDecimal()) {
    if (new_type->isNumber() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isBoolean()) {
      return ne(builder_->cst(0, expr_->type()));
    }
  } else if (expr_->type()->isBoolean()) {
    if (new_type->isInteger() || new_type->isDecimal()) {
      std::list<std::pair<ExprPtr, ExprPtr>> expr_list;
      expr_list.emplace_back(expr_, builder_->cst(1, new_type).expr());
      auto case_expr = std::make_shared<CaseExpr>(
          new_type, expr_->containsAgg(), expr_list, builder_->cst(0, new_type).expr());
      if (expr_->type()->nullable()) {
        auto is_null = std::make_shared<UOper>(
            builder_->ctx_.boolean(false), OpType::kIsNull, expr_);
        auto is_not_null =
            std::make_shared<UOper>(builder_->ctx_.boolean(false), OpType::kNot, is_null);
        auto null_cst = std::make_shared<Constant>(new_type, true, Datum{});
        std::list<std::pair<ExprPtr, ExprPtr>> expr_list;
        expr_list.emplace_back(is_not_null, case_expr);
        case_expr =
            makeExpr<CaseExpr>(new_type, expr_->containsAgg(), expr_list, null_cst);
      }
      return {builder_, case_expr, "", true};
    } else if (new_type->isFloatingPoint() || new_type->isBoolean()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isString()) {
    if (new_type->equal(expr_->type()->withNullable(new_type->nullable()))) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isExtDictionary()) {
      if (new_type->as<ExtDictionaryType>()->dictId() <= TRANSIENT_DICT_ID &&
          !expr_->is<Constant>()) {
        throw InvalidQueryError(
            "Cannot apply transient dictionary encoding to non-literal expression.");
      }
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isNumber() || new_type->isDateTime() || new_type->isBoolean() ||
               new_type->isString()) {
      if (expr_->is<Constant>()) {
        try {
          return {builder_, expr_->cast(new_type), "", true};
        } catch (std::runtime_error& e) {
          throw InvalidQueryError(e.what());
        }
      } else {
        throw InvalidQueryError(
            "String conversions for non-literals are not yet supported.");
      }
    }
  } else if (expr_->type()->isExtDictionary()) {
    if (new_type->isText()) {
      return {builder_, expr_->cast(new_type), "", true};
    } else if (new_type->isExtDictionary()) {
      if (new_type->as<ExtDictionaryType>()->dictId() <= TRANSIENT_DICT_ID &&
          !expr_->is<Constant>()) {
        throw InvalidQueryError(
            "Cannot apply transient dictionary encoding to non-literal expression.");
      }
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isDate()) {
    LOG(ERROR) << "Conversion date: " << expr_->type() << " new_type: " << new_type;
    if (new_type->isInteger() || new_type->isDate() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isTime()) {
    if (new_type->isTime()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  } else if (expr_->type()->isTimestamp()) {
    if (new_type->isInteger() || new_type->isDate() || new_type->isTimestamp()) {
      return {builder_, expr_->cast(new_type), "", true};
    }
  }

  throw InvalidQueryError() << "Conversion from " << expr_->type()->toString() << " to "
                            << new_type->toString() << " is not supported.";
}

BuilderExpr BuilderExpr::cast(const std::string& new_type) const {
  return cast(builder_->ctx_.typeFromString(new_type));
}

BuilderExpr BuilderExpr::logicalNot() const {
  if (!expr_->type()->isBoolean()) {
    throw InvalidQueryError("Only boolean expressions are allowed for NOT operation.");
  }
  if (expr_->is<Constant>()) {
    return builder_->cst(!expr_->as<Constant>()->intVal(), expr_->type());
  }
  auto uoper = makeExpr<UOper>(builder_->ctx_.boolean(expr_->type()->nullable()),
                               expr_->containsAgg(),
                               OpType::kNot,
                               expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::uminus() const {
  if (!expr_->type()->isNumber() && !expr_->type()->isInterval()) {
    throw InvalidQueryError("Only numeric expressions are allowed for UMINUS operation.");
  }
  if (expr_->is<Constant>()) {
    auto cst_expr = expr_->as<Constant>();
    if (cst_expr->type()->isInteger()) {
      return builder_->cst(-cst_expr->intVal(), cst_expr->type());
    } else if (cst_expr->type()->isFloatingPoint()) {
      return builder_->cst(-cst_expr->fpVal(), cst_expr->type());
    } else if (cst_expr->type()->isDecimal()) {
      return builder_->cstNoScale(-cst_expr->intVal(), cst_expr->type());
    } else {
      CHECK(cst_expr->type()->isInterval());
      return builder_->cst(-cst_expr->intVal(), cst_expr->type());
    }
  }
  auto uoper =
      makeExpr<UOper>(expr_->type(), expr_->containsAgg(), OpType::kUMinus, expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::isNull() const {
  if (expr_->type()->isNull()) {
    return builder_->trueCst();
  } else if (!expr_->type()->nullable()) {
    return builder_->falseCst();
  } else if (expr_->is<Constant>()) {
    return builder_->cst((int)expr_->as<Constant>()->isNull(),
                         builder_->ctx_.boolean(false));
  }
  auto uoper = makeExpr<UOper>(
      builder_->ctx_.boolean(false), expr_->containsAgg(), OpType::kIsNull, expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::unnest() const {
  if (!expr_->type()->isArray()) {
    throw InvalidQueryError("Only array expressions are allowed for UNNEST operation.");
  }
  auto elem_type = expr_->type()->as<ArrayBaseType>()->elemType();
  auto uoper = makeExpr<UOper>(elem_type, expr_->containsAgg(), OpType::kUnnest, expr_);
  return {builder_, uoper, "", true};
}

BuilderExpr BuilderExpr::add(const BuilderExpr& rhs) const {
  if ((expr_->type()->isDate() || expr_->type()->isTimestamp()) &&
      rhs.expr()->type()->isInterval()) {
    // Generate DATEADD expression.
    auto res_size = std::max(expr_->type()->size(), rhs.expr()->type()->size());
    auto date_unit = expr_->type()->as<DateTimeBaseType>()->unit();
    auto interval_unit = rhs.expr()->type()->as<IntervalType>()->unit();
    auto res_unit = std::max(date_unit, interval_unit);
    auto res_nullable = expr_->type()->nullable() || rhs.expr()->type()->nullable();
    const Type* res_type =
        expr_->type()->isDate()
            ? (const Type*)builder_->ctx_.date(res_size, res_unit, res_nullable)
            : (const Type*)builder_->ctx_.timestamp(res_unit, res_nullable);
    auto add_expr = std::make_shared<DateAddExpr>(
        res_type, timeUnitToDateAddField(interval_unit), rhs.expr(), expr_);
    return {builder_, add_expr, "", true};
  }
  if ((rhs.expr()->type()->isDate() || rhs.expr()->type()->isTimestamp()) &&
      expr_->type()->isInterval()) {
    return rhs.add(*this);
  }
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kPlus, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply PLUS operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::add(int val) const {
  return add(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::add(int64_t val) const {
  return add(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::add(float val) const {
  return add(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::add(double val) const {
  return add(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::add(const BuilderExpr& rhs, DateAddField field) const {
  if (!expr_->type()->isDate() && !expr_->type()->isTimestamp()) {
    throw InvalidQueryError()
        << "Left operand of DATE_ADD operation should be DATE or TIMESTAMP. Actual type: "
        << expr_->type()->toString();
  }
  if (!rhs.expr()->type()->isInteger()) {
    throw InvalidQueryError()
        << "Right operand of DATE_ADD operation should be INTEGER. Actual type: "
        << rhs.expr()->type()->toString();
  }
  auto number_expr = rhs.expr();
  if (!number_expr->type()->isInt64()) {
    number_expr =
        rhs.cast(number_expr->ctx().int64(number_expr->type()->nullable())).expr();
  }
  auto res_type = expr_->type()->withNullable(expr_->type()->nullable() ||
                                              number_expr->type()->nullable());
  if (res_type->isDate()) {
    auto unit = res_type->as<DateType>()->unit();
    unit = unit == TimeUnit::kDay ? TimeUnit::kSecond : unit;
    res_type = res_type->ctx().timestamp(unit, res_type->nullable());
  }
  auto date_add_expr = makeExpr<DateAddExpr>(res_type, field, number_expr, expr_);
  return {builder_, date_add_expr};
}

BuilderExpr BuilderExpr::add(int64_t val, DateAddField field) const {
  return add(builder_->cst(val, builder_->ctx_.int64(false)), field);
}

BuilderExpr BuilderExpr::add(const BuilderExpr& rhs, const std::string& field) const {
  return add(rhs, parseDateAddField(field));
}

BuilderExpr BuilderExpr::add(int64_t val, const std::string& field) const {
  return add(builder_->cst(val, builder_->ctx_.int64(false)), parseDateAddField(field));
}

BuilderExpr BuilderExpr::sub(const BuilderExpr& rhs) const {
  if ((expr_->type()->isDate() || expr_->type()->isTimestamp()) &&
      rhs.expr()->type()->isInterval()) {
    return add(rhs.uminus());
  }
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kMinus, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply MINUS operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::sub(int val) const {
  return sub(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::sub(int64_t val) const {
  return sub(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::sub(float val) const {
  return sub(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::sub(double val) const {
  return sub(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::sub(const BuilderExpr& rhs, DateAddField field) const {
  return add(rhs.uminus(), field);
}

BuilderExpr BuilderExpr::sub(int val, DateAddField field) const {
  return sub(builder_->cst(val, builder_->ctx_.int32(false)), field);
}

BuilderExpr BuilderExpr::sub(int64_t val, DateAddField field) const {
  return sub(builder_->cst(val, builder_->ctx_.int64(false)), field);
}

BuilderExpr BuilderExpr::sub(const BuilderExpr& rhs, const std::string& field) const {
  return sub(rhs, parseDateAddField(field));
}

BuilderExpr BuilderExpr::sub(int val, const std::string& field) const {
  return sub(builder_->cst(val, builder_->ctx_.int32(false)), parseDateAddField(field));
}

BuilderExpr BuilderExpr::sub(int64_t val, const std::string& field) const {
  return sub(builder_->cst(val, builder_->ctx_.int64(false)), parseDateAddField(field));
}

BuilderExpr BuilderExpr::mul(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kMul, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply MUL operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::mul(int val) const {
  return mul(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::mul(int64_t val) const {
  return mul(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::mul(float val) const {
  return mul(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::mul(double val) const {
  return mul(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::div(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kDiv, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply DIV operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::div(int val) const {
  return div(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::div(int64_t val) const {
  return div(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::div(float val) const {
  return div(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::div(double val) const {
  return div(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::mod(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kMod, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply MOD operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::mod(int val) const {
  return mod(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::mod(int64_t val) const {
  return mod(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::ceil() const {
  if (expr_->type()->isInteger()) {
    return *this;
  }
  if (expr_->type()->isNumber()) {
    ExprPtr op_expr = makeExpr<FunctionOperWithCustomTypeHandling>(
        expr_->type(), "CEIL", ExprPtrVector{expr_});
    return {builder_, op_expr, "", true};
  }
  throw InvalidQueryError() << "Cannot apply CEIL operation for operand type "
                            << expr_->type()->toString();
}

BuilderExpr BuilderExpr::floor() const {
  if (expr_->type()->isInteger()) {
    return *this;
  }
  if (expr_->type()->isNumber()) {
    ExprPtr op_expr = makeExpr<FunctionOperWithCustomTypeHandling>(
        expr_->type(), "FLOOR", ExprPtrVector{expr_});
    return {builder_, op_expr, "", true};
  }
  throw InvalidQueryError() << "Cannot apply CEIL operation for operand type "
                            << expr_->type()->toString();
}

BuilderExpr BuilderExpr::pow(const BuilderExpr& rhs) const {
  if (!expr_->type()->isNumber() || !rhs.type()->isNumber()) {
    throw InvalidQueryError() << "Cannot apply POW operation for operand types: "
                              << type()->toString() << " and " << rhs.type()->toString();
  }
  ExprPtrVector args;
  args.push_back(cast(ctx().fp64(type()->nullable())).expr());
  args.push_back(rhs.cast(ctx().fp64(rhs.type()->nullable())).expr());
  auto res_type = ctx().fp64(type()->nullable() || rhs.type()->nullable());
  auto pow_expr =
      hdk::ir::makeExpr<hdk::ir::FunctionOper>(res_type, "POWER", std::move(args));
  return {builder_, pow_expr, "", true};
}

BuilderExpr BuilderExpr::pow(int val) const {
  return pow(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::pow(int64_t val) const {
  return pow(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::pow(float val) const {
  return pow(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::pow(double val) const {
  return pow(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::logicalAnd(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kAnd, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply AND operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::logicalOr(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kOr, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply OR operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::eq(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kEq, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply EQ operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::eq(int val) const {
  return eq(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::eq(int64_t val) const {
  return eq(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::eq(float val) const {
  return eq(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::eq(double val) const {
  return eq(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::eq(const std::string& val) const {
  return eq(builder_->cst(val));
}

BuilderExpr BuilderExpr::ne(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kNe, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply NE operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::ne(int val) const {
  return ne(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::ne(int64_t val) const {
  return ne(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::ne(float val) const {
  return ne(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::ne(double val) const {
  return ne(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::ne(const std::string& val) const {
  return ne(builder_->cst(val));
}

BuilderExpr BuilderExpr::lt(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kLt, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply LT operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::lt(int val) const {
  return lt(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::lt(int64_t val) const {
  return lt(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::lt(float val) const {
  return lt(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::lt(double val) const {
  return lt(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::lt(const std::string& val) const {
  return lt(builder_->cst(val));
}

BuilderExpr BuilderExpr::le(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kLe, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply LE operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::le(int val) const {
  return le(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::le(int64_t val) const {
  return le(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::le(float val) const {
  return le(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::le(double val) const {
  return le(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::le(const std::string& val) const {
  return le(builder_->cst(val));
}

BuilderExpr BuilderExpr::gt(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kGt, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply GT operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::gt(int val) const {
  return gt(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::gt(int64_t val) const {
  return gt(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::gt(float val) const {
  return gt(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::gt(double val) const {
  return gt(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::gt(const std::string& val) const {
  return gt(builder_->cst(val));
}

BuilderExpr BuilderExpr::ge(const BuilderExpr& rhs) const {
  try {
    auto bin_oper = Analyzer::normalizeOperExpr(
        OpType::kGe, Qualifier::kOne, expr_, rhs.expr(), nullptr);
    return {builder_, bin_oper, "", true};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError() << "Cannot apply GE operation for operand types "
                              << expr_->type()->toString() << " and "
                              << rhs.expr()->type()->toString();
  }
}

BuilderExpr BuilderExpr::ge(int val) const {
  return ge(builder_->cst(val, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::ge(int64_t val) const {
  return ge(builder_->cst(val, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::ge(float val) const {
  return ge(builder_->cst(val, builder_->ctx_.fp32(false)));
}

BuilderExpr BuilderExpr::ge(double val) const {
  return ge(builder_->cst(val, builder_->ctx_.fp64(false)));
}

BuilderExpr BuilderExpr::ge(const std::string& val) const {
  return ge(builder_->cst(val));
}

BuilderExpr BuilderExpr::at(const BuilderExpr& idx) const {
  if (!expr_->type()->isArray()) {
    throw InvalidQueryError() << "Cannot use ARRAY_AT for " << expr_->type()->toString();
  }
  if (!idx.expr()->type()->isInteger()) {
    throw InvalidQueryError()
        << "Only integer indexes can be used in ARRAY_AT. Provided: "
        << idx.expr()->type()->toString();
  }
  auto res_type = expr_->type()->as<ArrayBaseType>()->elemType();
  auto at_expr = makeExpr<BinOper>(res_type,
                                   expr_->containsAgg() || idx.expr()->containsAgg(),
                                   OpType::kArrayAt,
                                   Qualifier::kOne,
                                   expr_,
                                   idx.expr());
  return {builder_, at_expr, "", true};
}

BuilderExpr BuilderExpr::at(int idx) const {
  return at(builder_->cst(idx, builder_->ctx_.int32(false)));
}

BuilderExpr BuilderExpr::at(int64_t idx) const {
  return at(builder_->cst(idx, builder_->ctx_.int64(false)));
}

BuilderExpr BuilderExpr::over() const {
  return over(std::vector<BuilderExpr>());
}

BuilderExpr BuilderExpr::over(const BuilderExpr& key) const {
  return over(std::vector<BuilderExpr>({key}));
}

BuilderExpr BuilderExpr::over(const std::vector<BuilderExpr>& keys) const {
  auto wnd_fn = checkOrGetWindowFn(expr_);
  if (!wnd_fn) {
    throw InvalidQueryError()
        << "Expected window function or supported aggregate (COUNT, AVG, MIN, MAX, SUM) "
           "for OVER. Provided: "
        << expr_->toString();
  }

  if (!keys.empty()) {
    for (auto& key : keys) {
      if (!key.expr()->is<ColumnRef>()) {
        throw InvalidQueryError() << "Currently, only column references can be used as a "
                                     "partition key. Provided: "
                                  << key.expr()->toString();
      }
    }

    ExprPtrVector new_part_keys;
    new_part_keys.reserve(wnd_fn->partitionKeys().size() + keys.size());
    new_part_keys.insert(new_part_keys.end(),
                         wnd_fn->partitionKeys().begin(),
                         wnd_fn->partitionKeys().end());
    for (auto& expr : keys) {
      new_part_keys.push_back(expr.expr());
    }
    wnd_fn = makeExpr<WindowFunction>(wnd_fn->type(),
                                      wnd_fn->kind(),
                                      wnd_fn->args(),
                                      new_part_keys,
                                      wnd_fn->orderKeys(),
                                      wnd_fn->collation());
  }

  return {builder_, wnd_fn, name_, auto_name_};
}

BuilderExpr BuilderExpr::orderBy(BuilderExpr key,
                                 SortDirection dir,
                                 NullSortedPosition null_pos) const {
  return orderBy(std::vector<BuilderExpr>({key}), dir, null_pos);
}

BuilderExpr BuilderExpr::orderBy(BuilderExpr key,
                                 const std::string& dir,
                                 const std::string& null_pos) const {
  return orderBy(std::vector<BuilderExpr>({key}), dir, null_pos);
}

BuilderExpr BuilderExpr::orderBy(std::initializer_list<BuilderExpr> keys,
                                 SortDirection dir,
                                 NullSortedPosition null_pos) const {
  return orderBy(std::vector<BuilderExpr>(keys), dir, null_pos);
}

BuilderExpr BuilderExpr::orderBy(std::initializer_list<BuilderExpr> keys,
                                 const std::string& dir,
                                 const std::string& null_pos) const {
  return orderBy(std::vector<BuilderExpr>(keys), dir, null_pos);
}

BuilderExpr BuilderExpr::orderBy(const std::vector<BuilderExpr>& keys,
                                 SortDirection dir,
                                 NullSortedPosition null_pos) const {
  std::vector<BuilderOrderByKey> order_keys;
  order_keys.reserve(keys.size());
  for (auto& key : keys) {
    order_keys.emplace_back(key, dir, null_pos);
  }
  return orderBy(order_keys);
}

BuilderExpr BuilderExpr::orderBy(const std::vector<BuilderExpr>& keys,
                                 const std::string& dir,
                                 const std::string& null_pos) const {
  std::vector<BuilderOrderByKey> order_keys;
  order_keys.reserve(keys.size());
  for (auto& key : keys) {
    order_keys.emplace_back(key, dir, null_pos);
  }
  return orderBy(order_keys);
}

BuilderExpr BuilderExpr::orderBy(const BuilderOrderByKey& key) const {
  return orderBy(std::vector<BuilderOrderByKey>({key}));
}

BuilderExpr BuilderExpr::orderBy(const std::vector<BuilderOrderByKey>& keys) const {
  auto wnd_fn = expr_->as<WindowFunction>();
  if (!wnd_fn) {
    throw InvalidQueryError() << "Expected window function for ORDER BY. Provided: "
                              << expr_->toString();
  }

  for (auto& key : keys) {
    if (!key.expr()->is<ColumnRef>()) {
      throw InvalidQueryError()
          << "Currently, only column references can be used in ORDER BY. Provided: "
          << key.expr()->toString();
    }
  }

  ExprPtrVector new_order_keys = wnd_fn->orderKeys();
  new_order_keys.reserve(wnd_fn->orderKeys().size() + keys.size());
  new_order_keys.insert(
      new_order_keys.end(), wnd_fn->orderKeys().begin(), wnd_fn->orderKeys().end());
  for (auto& key : keys) {
    new_order_keys.push_back(key.expr());
  }

  std::vector<OrderEntry> new_collation;
  new_collation.reserve(wnd_fn->collation().size() + keys.size());
  new_collation.insert(
      new_collation.end(), wnd_fn->collation().begin(), wnd_fn->collation().end());
  for (auto& key : keys) {
    new_collation.emplace_back(static_cast<int>(new_collation.size()),
                               key.dir() == SortDirection::Descending,
                               key.nullsPosition() == NullSortedPosition::First);
  }

  auto res = makeExpr<WindowFunction>(wnd_fn->type(),
                                      wnd_fn->kind(),
                                      wnd_fn->args(),
                                      wnd_fn->partitionKeys(),
                                      new_order_keys,
                                      new_collation);

  return {builder_, res, name_, auto_name_};
}

BuilderExpr BuilderExpr::rewrite(ExprRewriter& rewriter) const {
  return {builder_, rewriter.visit(expr_.get()), name_, auto_name_};
}

BuilderExpr BuilderExpr::operator!() const {
  return logicalNot();
}

BuilderExpr BuilderExpr::operator-() const {
  return uminus();
}

BuilderExpr BuilderExpr::operator[](const BuilderExpr& idx) const {
  return at(idx);
}

BuilderExpr BuilderExpr::operator[](int idx) const {
  return at(idx);
}

BuilderExpr BuilderExpr::operator[](int64_t idx) const {
  return at(idx);
}

const QueryBuilder& BuilderExpr::builder() const {
  return *builder_;
}

Context& BuilderExpr::ctx() const {
  return const_cast<Context&>(builder_->ctx_);
}

BuilderSortField::BuilderSortField()
    : field_(0), dir_(SortDirection::Ascending), null_pos_(NullSortedPosition::Last) {}

BuilderSortField::BuilderSortField(int col_idx,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(col_idx), dir_(dir), null_pos_(null_pos) {}

BuilderSortField::BuilderSortField(int col_idx,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(col_idx)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {}

BuilderSortField::BuilderSortField(const std::string& col_name,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(col_name), dir_(dir), null_pos_(null_pos) {}

BuilderSortField::BuilderSortField(const std::string& col_name,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(col_name)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {}

BuilderSortField::BuilderSortField(BuilderExpr expr,
                                   SortDirection dir,
                                   NullSortedPosition null_pos)
    : field_(expr), dir_(dir), null_pos_(null_pos) {
  if (!expr.expr()->is<ColumnRef>()) {
    throw InvalidQueryError() << "Only column references are allowed for "
                                 "sort operation. Provided expression: "
                              << expr.expr()->toString();
  }
}

BuilderSortField::BuilderSortField(BuilderExpr expr,
                                   const std::string& dir,
                                   const std::string& null_pos)
    : field_(expr)
    , dir_(parseSortDirection(dir))
    , null_pos_(parseNullPosition(null_pos)) {
  if (!expr.expr()->is<ColumnRef>()) {
    throw InvalidQueryError() << "Only column references are allowed for "
                                 "sort operation. Provided expression: "
                              << expr.expr()->toString();
  }
}

SortDirection BuilderSortField::parseSortDirection(const std::string& val) {
  auto val_lowered = boost::trim_copy(boost::to_lower_copy(val));
  if (val_lowered == "asc" || val_lowered == "ascending") {
    return SortDirection::Ascending;
  } else if (val_lowered == "desc" || val_lowered == "descending") {
    return SortDirection::Descending;
  }
  throw InvalidQueryError() << "Cannot parse sort direction (use 'asc' or 'desc'): '"
                            << val << "'";
}

NullSortedPosition BuilderSortField::parseNullPosition(const std::string& val) {
  auto val_lowered = boost::trim_copy(boost::to_lower_copy(val));
  if (val_lowered == "first") {
    return NullSortedPosition::First;
  } else if (val_lowered == "last") {
    return NullSortedPosition::Last;
  }
  throw InvalidQueryError() << "Cannot parse nulls position (use 'first' or 'last'): '"
                            << val << "'";
}

BuilderNode::BuilderNode() : builder_(nullptr) {}

BuilderNode::BuilderNode(const QueryBuilder* builder, NodePtr node)
    : builder_(builder), node_(node) {}

BuilderExpr BuilderNode::ref(int col_idx) const {
  auto expr = getRefByIndex(node_.get(), col_idx);
  auto name = getFieldName(node_.get(), col_idx);
  return {builder_, expr, name};
}

BuilderExpr BuilderNode::ref(const std::string& col_name) const {
  auto expr = getRefByName(node_.get(), col_name);
  return {builder_, expr, col_name};
}

std::vector<BuilderExpr> BuilderNode::ref(std::initializer_list<int> col_indices) const {
  return ref(std::vector<int>(col_indices));
}

std::vector<BuilderExpr> BuilderNode::ref(std::vector<int> col_indices) const {
  std::vector<BuilderExpr> res;
  res.reserve(col_indices.size());
  for (auto col_idx : col_indices) {
    res.emplace_back(ref(col_idx));
  }
  return res;
}

std::vector<BuilderExpr> BuilderNode::ref(
    std::initializer_list<std::string> col_names) const {
  return ref(std::vector<std::string>(col_names));
}

std::vector<BuilderExpr> BuilderNode::ref(std::vector<std::string> col_names) const {
  std::vector<BuilderExpr> res;
  res.reserve(col_names.size());
  for (auto col_name : col_names) {
    res.emplace_back(ref(col_name));
  }
  return res;
}

BuilderExpr BuilderNode::count() const {
  return builder_->count();
}

BuilderExpr BuilderNode::count(int col_idx, bool is_distinct) const {
  return ref(col_idx).count(is_distinct);
}

BuilderExpr BuilderNode::count(const std::string& col_name, bool is_distinct) const {
  return ref(col_name).count(is_distinct);
}

BuilderExpr BuilderNode::count(BuilderExpr col_ref, bool is_distinct) const {
  return col_ref.count(is_distinct);
}

BuilderNode BuilderNode::proj(std::initializer_list<int> col_indices) const {
  return proj(ref(col_indices));
}

BuilderNode BuilderNode::proj(std::initializer_list<int> col_indices,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_indices), fields);
}

BuilderNode BuilderNode::proj(std::initializer_list<std::string> col_names) const {
  return proj(ref(col_names));
}

BuilderNode BuilderNode::proj(std::initializer_list<std::string> col_names,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_names), fields);
}

BuilderNode BuilderNode::proj(int col_idx) const {
  return proj(ref({col_idx}));
}

BuilderNode BuilderNode::proj(int col_idx, const std::string& field_name) const {
  return proj(ref(col_idx), field_name);
}

BuilderNode BuilderNode::proj(const std::vector<int> col_indices) const {
  return proj(ref(col_indices));
}

BuilderNode BuilderNode::proj(const std::vector<int> col_indices,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_indices), fields);
}

BuilderNode BuilderNode::proj(const std::string& col_name) const {
  return proj(ref(col_name));
}

BuilderNode BuilderNode::proj(const std::string& col_name,
                              const std::string& field_name) const {
  return proj(ref(col_name), field_name);
}

BuilderNode BuilderNode::proj(const std::vector<std::string>& col_names) const {
  return proj(ref(col_names));
}

BuilderNode BuilderNode::proj(const std::vector<std::string>& col_names,
                              const std::vector<std::string>& fields) const {
  return proj(ref(col_names), fields);
}

BuilderNode BuilderNode::proj(const BuilderExpr& expr) const {
  return proj(std::vector<BuilderExpr>({expr}));
}

BuilderNode BuilderNode::proj(const BuilderExpr& expr, const std::string& field) const {
  return proj(std::vector<BuilderExpr>({expr}), {field});
}

BuilderNode BuilderNode::proj(const std::vector<BuilderExpr>& exprs) const {
  auto fields = buildFieldNames(exprs);
  auto expr_ptrs = collectExprs(exprs);
  return proj(expr_ptrs, fields);
}

BuilderNode BuilderNode::proj(const std::vector<BuilderExpr>& exprs,
                              const std::vector<std::string>& fields) const {
  std::unordered_set<std::string> names;
  for (auto& name : fields) {
    auto pr = names.insert(name);
    if (!pr.second) {
      throw InvalidQueryError() << "Duplicated field name: " << name;
    }
  }
  auto expr_ptrs = collectExprs(exprs);
  return proj(expr_ptrs, fields);
}

BuilderNode BuilderNode::proj() const {
  std::vector<int> col_indexes(node_->size());
  std::iota(col_indexes.begin(), col_indexes.end(), 0);
  return proj(col_indexes);
}

BuilderNode BuilderNode::proj(const ExprPtrVector& exprs,
                              const std::vector<std::string>& fields) const {
  if (exprs.empty()) {
    throw InvalidQueryError(
        "Empty projections are not allowed. At least one expression is required.");
  }
  if (exprs.size() != fields.size()) {
    throw InvalidQueryError() << "Mismathed number of expressions (" << exprs.size()
                              << ") and field names (" << fields.size() << ")";
  }
  checkExprInput(exprs, {node_.get()}, "projection");
  auto proj = std::make_shared<Project>(exprs, fields, node_);
  return {builder_, proj};
}

BuilderNode BuilderNode::filter(const BuilderExpr& condition) const {
  checkExprInput(condition, {node_.get()}, "filter");
  // Filter-out virtual column if it is not used on the filter.,
  auto base = node_;
  if (node_->is<Scan>()) {
    auto used_cols = InputColIndexesCollector::collect(condition.expr());
    int cols_to_proj =
        used_cols.count((int)(node_->size() - 1)) ? node_->size() : node_->size() - 1;
    std::vector<int> col_indices(cols_to_proj);
    std::iota(col_indices.begin(), col_indices.end(), 0);
    base = proj(col_indices).node();
  }
  auto filter = std::make_shared<Filter>(condition.expr(), base);
  return {builder_, filter};
}

BuilderExpr BuilderNode::parseAggString(const std::string& agg_str) const {
  auto agg_str_lower = boost::trim_copy(boost::algorithm::to_lower_copy(agg_str));
  if (agg_str_lower == "count") {
    return count();
  }

  // Parse string like <agg_name>(<col_name>[, <agg_param>]).
  auto pos = agg_str_lower.find('(');
  if (agg_str_lower.back() == ')' && pos != std::string::npos) {
    auto agg_name = boost::trim_copy(agg_str_lower.substr(0, pos));
    auto col_name =
        boost::trim_copy(agg_str_lower.substr(pos + 1, agg_str_lower.size() - pos - 2));

    if (agg_name == "count" && (col_name.empty() || col_name == "1" || col_name == "*")) {
      return count();
    }

    BuilderExpr arg;
    auto comma_pos = col_name.find(',');
    if (comma_pos != std::string::npos) {
      auto val_str = boost::trim_copy(
          col_name.substr(comma_pos + 1, col_name.size() - comma_pos - 1));
      char* end = nullptr;
      auto val = std::strtod(val_str.c_str(), &end);
      // Require value string to be fully interpreted to avoid silent errors like
      // 1..1 interpreted as 1.
      if (val == HUGE_VAL || end == val_str.c_str() ||
          end != (val_str.c_str() + val_str.size())) {
        // If value is not decimal then assume it is a column name (for corr aggregate).
        auto ref = getRefByName(node_.get(), val_str, true);
        if (!ref) {
          throw InvalidQueryError()
              << "Cannot parse aggregate parameter (decimal or column name expected): "
              << val_str;
        }
        arg = BuilderExpr(builder_, ref, val_str);
      } else {
        arg = builder_->cst(val);
      }
      col_name = boost::trim_copy(col_name.substr(0, comma_pos));
    }
    return ref(col_name).agg(agg_name, arg);
  }

  throw InvalidQueryError() << "Cannot parse aggregate string: '" << agg_str << "'";
}

std::vector<BuilderExpr> BuilderNode::parseAggString(
    const std::vector<std::string>& aggs) const {
  std::vector<BuilderExpr> res;
  res.reserve(aggs.size());
  for (auto& agg_str : aggs) {
    res.emplace_back(parseAggString(agg_str));
  }
  return res;
}

BuilderNode BuilderNode::agg(int group_key, const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_key), std::vector<BuilderExpr>());
  }
  return agg(ref(group_key), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(int group_key,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(int group_key, const std::vector<std::string>& aggs) const {
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(int group_key, BuilderExpr agg_expr) const {
  return agg(ref(group_key), agg_expr);
}

BuilderNode BuilderNode::agg(int group_key, const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_key), aggs);
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::string& agg_str) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(agg_str));
  }
  if (agg_str.empty()) {
    return agg(ref(group_key), std::vector<BuilderExpr>());
  }
  return agg(ref(group_key), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             std::initializer_list<std::string> aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(aggs));
  }
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::vector<std::string>& aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), parseAggString(aggs));
  }
  return agg(ref(group_key), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::string& group_key, BuilderExpr agg_expr) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), agg_expr);
  }
  return agg(ref(group_key), agg_expr);
}

BuilderNode BuilderNode::agg(const std::string& group_key,
                             const std::vector<BuilderExpr>& aggs) const {
  if (group_key.empty()) {
    return agg(std::vector<std::string>(), aggs);
  }
  return agg(ref(group_key), aggs);
}

BuilderNode BuilderNode::agg(BuilderExpr group_key, const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(std::vector<BuilderExpr>({group_key}), std::vector<BuilderExpr>());
  }
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             std::initializer_list<std::string> aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             const std::vector<std::string>& aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key, BuilderExpr agg_expr) const {
  return agg(std::vector<BuilderExpr>({group_key}), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(BuilderExpr group_key,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(std::vector<BuilderExpr>({group_key}), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(std::initializer_list<int> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}
BuilderNode BuilderNode::agg(std::initializer_list<std::string> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(std::vector<BuilderExpr>(group_keys), std::vector<BuilderExpr>());
  }
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             BuilderExpr agg_expr) const {
  return agg(std::vector<BuilderExpr>(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(std::initializer_list<BuilderExpr> group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(std::vector<BuilderExpr>(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<int>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(ref(group_keys), std::vector<BuilderExpr>());
  }
  return agg(ref(group_keys), parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(ref(group_keys), parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(ref(group_keys), std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<std::string>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  return agg(ref(group_keys), aggs);
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::string& agg_str) const {
  if (agg_str.empty()) {
    return agg(group_keys, std::vector<BuilderExpr>());
  }
  return agg(group_keys, parseAggString(agg_str));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             std::initializer_list<std::string> aggs) const {
  return agg(group_keys, parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::vector<std::string>& aggs) const {
  return agg(group_keys, parseAggString(aggs));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             BuilderExpr agg_expr) const {
  return agg(group_keys, std::vector<BuilderExpr>({agg_expr}));
}

BuilderNode BuilderNode::agg(const std::vector<BuilderExpr>& group_keys,
                             const std::vector<BuilderExpr>& aggs) const {
  if (group_keys.empty() && aggs.empty()) {
    throw InvalidQueryError(
        "Empty aggregations are not allowed. At least one group key or aggregate is "
        "required.");
  }

  checkExprInput(group_keys, {node_.get()}, "aggregation");
  checkExprInput(aggs, {node_.get()}, "aggregation");

  std::vector<int> shuffle;
  std::vector<int> rev_shuffle(node_->size(), -1);
  shuffle.reserve(node_->size());
  for (auto& key : group_keys) {
    if (!key.expr_->is<ir::ColumnRef>()) {
      throw InvalidQueryError()
          << "Aggregation group key should be a column reference. Passed expression: "
          << key.expr_->toString();
    }
    auto col_idx = key.expr_->as<ir::ColumnRef>()->index();
    if (rev_shuffle[col_idx] == -1) {
      rev_shuffle[col_idx] = shuffle.size();
    }
    shuffle.push_back(col_idx);
  }

  for (auto& agg_expr : aggs) {
    if (!agg_expr.expr()->is<AggExpr>()) {
      throw InvalidQueryError() << "Non-aggregte expression is used as an aggregate: "
                                << agg_expr.expr()->toString();
    }
  }

  // Aggregate node requires all key columns to be first in the list
  // of input columns. Make additional projection to achieve that.
  // We also add a projection when aggregate over a scan because
  // such construction is not supported.
  if (!isIdentShuffle(shuffle) || node_->is<Scan>()) {
    for (size_t i = 0; i < rev_shuffle.size(); ++i) {
      if (rev_shuffle[i] == -1) {
        rev_shuffle[i] = shuffle.size();
        shuffle.push_back(i);
      }
    }

    auto base = proj(shuffle);
    std::vector<BuilderExpr> new_keys;
    new_keys.reserve(group_keys.size());
    for (int i = 0; i < (int)group_keys.size(); ++i) {
      if (group_keys[i].isAutoNamed()) {
        new_keys.emplace_back(base.ref(i));
      } else {
        new_keys.emplace_back(base.ref(i).rename(group_keys[i].name()));
      }
    }
    return base.agg(new_keys, replaceInput(aggs, base.node_.get(), rev_shuffle));
  }

  auto all_exprs = group_keys;
  for (auto& agg : aggs) {
    all_exprs.emplace_back(agg);
  }
  auto agg_node = std::make_shared<Aggregate>(
      group_keys.size(), collectExprs(aggs), buildFieldNames(all_exprs), node_);
  return {builder_, agg_node};
}

namespace {
template <typename ColsType, typename... Ts>
std::vector<BuilderSortField> toSortFields(ColsType&& cols, Ts... args) {
  std::vector<BuilderSortField> res;
  for (auto& col : cols) {
    res.emplace_back(col, args...);
  }
  return res;
}

}  // namespace

BuilderNode BuilderNode::sort(int col_idx,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_idx, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(int col_idx,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_idx, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<int> col_indexes,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<int> col_indexes,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<int>& col_indexes,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<int>& col_indexes,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_indexes, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::string& col_name,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_name, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(const std::string& col_name,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_name, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<std::string> col_names,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<std::string> col_names,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<std::string>& col_names,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<std::string>& col_names,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_names, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(BuilderExpr col_ref,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_ref, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(BuilderExpr col_ref,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort({col_ref, dir, null_pos}, limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<BuilderExpr> col_refs,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(std::initializer_list<BuilderExpr> col_refs,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderExpr>& col_refs,
                              SortDirection dir,
                              NullSortedPosition null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderExpr>& col_refs,
                              const std::string& dir,
                              const std::string& null_pos,
                              size_t limit,
                              size_t offset) const {
  return sort(toSortFields(col_refs, dir, null_pos), limit, offset);
}

BuilderNode BuilderNode::sort(const BuilderSortField& field,
                              size_t limit,
                              size_t offset) const {
  return sort(std::vector<BuilderSortField>({field}), limit, offset);
}

BuilderNode BuilderNode::sort(const std::vector<BuilderSortField>& fields,
                              size_t limit,
                              size_t offset) const {
  std::vector<SortField> collation;
  for (auto& field : fields) {
    ExprPtr col_ref;
    if (field.hasColIdx()) {
      col_ref = ref(field.colIdx()).expr();
    } else if (field.hasColName()) {
      col_ref = ref(field.colName()).expr();
    } else {
      CHECK(field.hasExpr());
      col_ref = field.expr();
      if (field.expr()->as<ColumnRef>()->node() != node_.get()) {
        throw InvalidQueryError()
            << "Sort field refers non-input column: " << field.expr()->toString();
      }
    }
    if (col_ref->type()->isArray()) {
      throw InvalidQueryError() << "Cannot sort on array column: " << col_ref->toString();
    }
    collation.emplace_back(
        col_ref->as<ColumnRef>()->index(), field.dir(), field.nullsPosition());
  }
  // Sort over scan is not supported.
  auto base = node_;
  if (node_->is<Scan>()) {
    // Filter out rowid column if it's not used in the sort.
    auto scan = node_->as<Scan>();
    bool uses_rowid =
        std::any_of(collation.begin(), collation.end(), [&](const SortField& field) {
          return scan->isVirtualCol(field.getField());
        });
    int cols_to_proj = uses_rowid ? node_->size() : node_->size() - 1;
    std::vector<int> col_indices(cols_to_proj);
    std::iota(col_indices.begin(), col_indices.end(), 0);
    base = proj(col_indices).node();
  }
  auto sort_node = std::make_shared<Sort>(std::move(collation), limit, offset, base);
  return {builder_, sort_node};
}

BuilderNode BuilderNode::join(const BuilderNode& rhs, JoinType join_type) const {
  auto lhs_cols = getColNames(node_.get());
  auto rhs_cols = getColNames(rhs.node().get());
  std::vector<std::string> common_cols;
  for (auto& rhs_col : rhs_cols) {
    if (lhs_cols.count(rhs_col)) {
      common_cols.push_back(rhs_col);
    }
  }
  if (common_cols.empty()) {
    throw InvalidQueryError()
        << "Cannot find common columns to generate default equi join."
        << "\n  LHS columns: " << toString(lhs_cols)
        << "\n  RHS columns: " << toString(rhs_cols);
  }
  return join(rhs, common_cols, join_type);
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const std::string& join_type) const {
  return join(rhs, parseJoinType(join_type));
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const std::vector<std::string>& col_names,
                              JoinType join_type) const {
  return join(rhs, col_names, col_names, join_type);
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const std::vector<std::string>& col_names,
                              const std::string& join_type) const {
  return join(rhs, col_names, parseJoinType(join_type));
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const std::vector<std::string>& lhs_col_names,
                              const std::vector<std::string>& rhs_col_names,
                              JoinType join_type) const {
  if (lhs_col_names.size() != rhs_col_names.size()) {
    throw InvalidQueryError() << "Mismatched number of key columns for equi join: "
                              << lhs_col_names.size() << " vs. " << rhs_col_names.size();
  }
  if (lhs_col_names.empty()) {
    throw InvalidQueryError("Columns set for equi join should not be empty.");
  }
  auto cond = ref(lhs_col_names[0]).eq(rhs.ref(rhs_col_names[0]));
  for (size_t col_idx = 1; col_idx < lhs_col_names.size(); ++col_idx) {
    cond =
        cond.logicalAnd(ref(lhs_col_names[col_idx]).eq(rhs.ref(rhs_col_names[col_idx])));
  }
  NodePtr join_node = std::make_shared<Join>(node_, rhs.node(), cond.expr(), join_type);

  // Create a projection to filter out virtual columns and RHS key
  // columns.
  std::vector<int> proj_cols;
  proj_cols.reserve(join_node->size());
  auto lhs_scan = node_->as<Scan>();
  for (int i = 0; i < (int)node_->size(); ++i) {
    if (!lhs_scan || !lhs_scan->isVirtualCol(i)) {
      proj_cols.emplace_back(i);
    }
  }
  std::unordered_set<std::string> exclude_cols(rhs_col_names.begin(),
                                               rhs_col_names.end());
  auto rhs_scan = rhs.node()->as<Scan>();
  for (int i = 0; i < (int)rhs.node()->size(); ++i) {
    if (!exclude_cols.count(getFieldName(rhs.node().get(), i)) &&
        (!rhs_scan || !rhs_scan->isVirtualCol(i))) {
      proj_cols.emplace_back(i + node_->size());
    }
  }
  return BuilderNode(builder_, join_node).proj(proj_cols);
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const std::vector<std::string>& lhs_col_names,
                              const std::vector<std::string>& rhs_col_names,
                              const std::string& join_type) const {
  return join(rhs, lhs_col_names, rhs_col_names, parseJoinType(join_type));
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const BuilderExpr& cond,
                              JoinType join_type) const {
  checkExprInput(cond, {node_.get(), rhs.node().get()}, "join");
  NodePtr join_node = std::make_shared<Join>(node_, rhs.node(), cond.expr(), join_type);

  // Create a projection to filter out virtual columns if any
  // This will also assign new names to matching LHS and RHS
  // column names.
  std::vector<int> proj_cols;
  auto lhs_scan = node_->as<Scan>();
  for (int i = 0; i < (int)node_->size(); ++i) {
    if (!lhs_scan || !lhs_scan->isVirtualCol(i)) {
      proj_cols.push_back(i);
    }
  }
  auto rhs_scan = rhs.node()->as<Scan>();
  for (int i = 0; i < (int)rhs.node()->size(); ++i) {
    if (!rhs_scan || !rhs_scan->isVirtualCol(i)) {
      proj_cols.push_back(i + node_->size());
    }
  }
  return BuilderNode(builder_, join_node).proj(proj_cols);
}

BuilderNode BuilderNode::join(const BuilderNode& rhs,
                              const BuilderExpr& cond,
                              const std::string& join_type) const {
  return join(rhs, cond, parseJoinType(join_type));
}

BuilderExpr BuilderNode::operator[](int col_idx) const {
  return ref(col_idx);
}

BuilderExpr BuilderNode::operator[](const std::string& col_name) const {
  return ref(col_name);
}

std::unique_ptr<QueryDag> BuilderNode::finalize() const {
  // Scan and join nodes are not supposed to be a root of a query DAG.
  // Add a projection in such cases.
  if (auto scan = node_->as<Scan>()) {
    std::vector<int> cols;
    cols.reserve(scan->size());
    for (int i = 0; i < (int)node_->size(); ++i) {
      if (!scan->isVirtualCol(i)) {
        cols.push_back(i);
      }
    }
    return proj(cols).finalize();
  }
  // For join of scan nodes, we exclude virtual columns from the final
  // projection.
  if (auto join = node_->as<Join>()) {
    std::vector<int> cols;
    cols.reserve(join->size());
    for (int i = 0; i < (int)node_->size(); ++i) {
      auto col_ref = std::dynamic_pointer_cast<const ir::ColumnRef>(ref(i).expr());
      if (!col_ref->node()->is<Scan>() ||
          !col_ref->node()->as<Scan>()->isVirtualCol(col_ref->index())) {
        cols.push_back(i);
      }
    }
    return proj(cols).finalize();
  }

  auto res = std::make_unique<QueryDag>(builder_->config_, node_);
  res->setNodes(collectNodes(node_));
  return res;
}

ColumnInfoPtr BuilderNode::columnInfo(int col_index) const {
  auto ref = getRefByIndex(node_.get(), col_index);
  int real_idx = ref->as<ir::ColumnRef>()->index();
  if (node_->is<Scan>()) {
    return node_->as<Scan>()->getColumnInfo(real_idx);
  }
  return std::make_shared<ColumnInfo>(-1,
                                      -node_->getId(),
                                      real_idx,
                                      getFieldName(node_.get(), real_idx),
                                      ref->type(),
                                      false);
}

ColumnInfoPtr BuilderNode::columnInfo(const std::string& col_name) const {
  auto ref = getRefByName(node_.get(), col_name);
  int real_idx = ref->as<ir::ColumnRef>()->index();
  if (node_->is<Scan>()) {
    return node_->as<Scan>()->getColumnInfo(real_idx);
  }
  return std::make_shared<ColumnInfo>(-1,
                                      -node_->getId(),
                                      real_idx,
                                      getFieldName(node_.get(), real_idx),
                                      ref->type(),
                                      false);
}

size_t BuilderNode::rowCount() const {
  if (auto scan = node_->as<hdk::ir::Scan>()) {
    return scan->getTableInfo()->row_count;
  }
  throw InvalidQueryError("Row count is available for scan nodes only.");
}

QueryBuilder::QueryBuilder(Context& ctx,
                           SchemaProviderPtr schema_provider,
                           ConfigPtr config)
    : ctx_(ctx), schema_provider_(schema_provider), config_(config) {}

BuilderNode QueryBuilder::scan(const std::string& table_name) const {
  auto db_ids = schema_provider_->listDatabases();
  TableInfoPtr found_table = nullptr;
  for (auto db_id : db_ids) {
    auto table_info = schema_provider_->getTableInfo(db_id, table_name);
    if (table_info) {
      if (found_table) {
        throw InvalidQueryError() << "Ambiguous table name: " << table_name;
      }
      found_table = table_info;
    }
  }
  if (!found_table) {
    throw InvalidQueryError() << "Unknown table: " << table_name;
  }
  return scan(found_table);
}

BuilderNode QueryBuilder::scan(int db_id, const std::string& table_name) const {
  auto table_info = schema_provider_->getTableInfo(db_id, table_name);
  if (!table_info) {
    throw InvalidQueryError() << "Unknown table: " << table_name << " (db_id=" << db_id
                              << ")";
  }
  return scan(table_info);
}

BuilderNode QueryBuilder::scan(int db_id, int table_id) const {
  auto table_info = schema_provider_->getTableInfo(db_id, table_id);
  if (!table_info) {
    throw InvalidQueryError() << "Unknown table reference: db_id=" << db_id
                              << " table_id=" << table_id;
  }
  return scan(table_info);
}

BuilderNode QueryBuilder::scan(const TableRef& table_ref) const {
  return scan(table_ref.db_id, table_ref.table_id);
}

BuilderNode QueryBuilder::scan(TableInfoPtr table_info) const {
  auto scan =
      std::make_shared<Scan>(table_info, schema_provider_->listColumns(*table_info));
  return {this, scan};
}

BuilderExpr QueryBuilder::count() const {
  auto count_type =
      config_->exec.group_by.bigint_count ? ctx_.int64(false) : ctx_.int32(false);
  auto agg = makeExpr<AggExpr>(count_type, AggType::kCount, nullptr, false, nullptr);
  return {this, agg, "count", true};
}

BuilderExpr QueryBuilder::rowNumber() const {
  ExprPtr expr{new WindowFunction(
      ctx_.int64(false), WindowFunctionKind::RowNumber, {}, {}, {}, {})};
  return {this, expr, "row_number", true};
}

BuilderExpr QueryBuilder::rank() const {
  ExprPtr expr{
      new WindowFunction(ctx_.int64(false), WindowFunctionKind::Rank, {}, {}, {}, {})};
  return {this, expr, "rank", true};
}

BuilderExpr QueryBuilder::denseRank() const {
  ExprPtr expr{new WindowFunction(
      ctx_.int64(false), WindowFunctionKind::DenseRank, {}, {}, {}, {})};
  return {this, expr, "dense_rank", true};
}

BuilderExpr QueryBuilder::percentRank() const {
  ExprPtr expr{new WindowFunction(
      ctx_.fp64(false), WindowFunctionKind::PercentRank, {}, {}, {}, {})};
  return {this, expr, "percent_rank", true};
}

BuilderExpr QueryBuilder::nTile(int tile_count) const {
  if (tile_count <= 0) {
    throw InvalidQueryError()
        << "Expected positive integer for tile count argument. Provided: " << tile_count;
  }
  ExprPtr expr{new WindowFunction(ctx_.int64(false),
                                  WindowFunctionKind::NTile,
                                  {cst(tile_count).expr()},
                                  {},
                                  {},
                                  {})};
  return {this, expr, "ntile", true};
}

BuilderExpr QueryBuilder::cst(int val) const {
  return cst(static_cast<int64_t>(val));
}

BuilderExpr QueryBuilder::cst(int val, const Type* type) const {
  return cst(static_cast<int64_t>(val), type);
}

BuilderExpr QueryBuilder::cst(int val, const std::string& type) const {
  return cst(static_cast<int64_t>(val), type);
}

BuilderExpr QueryBuilder::cst(int64_t val) const {
  return cst(val, ctx_.int64(false));
}

BuilderExpr QueryBuilder::cst(int64_t val, const Type* type) const {
  if (!type->isNumber() && !type->isDateTime() && !type->isInterval() &&
      !type->isBoolean()) {
    throw InvalidQueryError()
        << "Cannot create a literal from an integer value for type: " << type->toString();
  }
  if (type->isDate() && type->as<DateType>()->unit() == TimeUnit::kDay) {
    throw InvalidQueryError("Literals of date type with DAY time unit are not allowed.");
  }
  auto cst_expr = Constant::make(type, val);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(int64_t val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(double val) const {
  return cst(val, ctx_.fp64());
}

BuilderExpr QueryBuilder::cst(double val, const Type* type) const {
  Datum d;
  if (type->isFp32()) {
    d.floatval = static_cast<float>(val);
  } else if (type->isFp64()) {
    d.doubleval = val;
  } else if (type->isDecimal()) {
    d.bigintval =
        static_cast<int64_t>(val * exp_to_scale(type->as<DecimalType>()->scale()));
  } else {
    throw InvalidQueryError() << "Cannot create a literal from a double value for type: "
                              << type->toString();
  }
  auto cst_expr = std::make_shared<Constant>(type, false, d);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(double val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::string& val) const {
  return cst(val, ctx_.text());
}

BuilderExpr QueryBuilder::cst(const std::string& val, const Type* type) const {
  if (type->isDate() && type->as<DateType>()->unit() == TimeUnit::kDay) {
    throw InvalidQueryError("Literals of date type with DAY time unit are not allowed.");
  }
  if (type->isArray()) {
    throw InvalidQueryError("Cannot parse string to an array literal.");
  }
  try {
    Datum d;
    d.stringval = new std::string(val);
    auto cst_expr = std::make_shared<Constant>(ctx_.text(), false, d);
    return {this, cst_expr->cast(type)};
  } catch (std::runtime_error& e) {
    throw InvalidQueryError(e.what());
  }
}

BuilderExpr QueryBuilder::cst(const std::string& val, const std::string& type) const {
  return cst(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cstNoScale(int64_t val, const Type* type) const {
  if (!type->isDecimal()) {
    throw InvalidQueryError()
        << "Only decimal types are allowed for QueryBuilder::cstNoScale. Provided: "
        << type->toString();
  }
  Datum d;
  d.bigintval = val;
  auto cst_expr = std::make_shared<Constant>(type, false, d);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cstNoScale(int64_t val, const std::string& type) const {
  return cstNoScale(val, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::trueCst() const {
  Datum d;
  d.boolval = true;
  auto cst_expr = std::make_shared<Constant>(ctx_.boolean(false), false, d);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::falseCst() const {
  Datum d;
  d.boolval = false;
  auto cst_expr = std::make_shared<Constant>(ctx_.boolean(false), false, d);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::nullCst() const {
  return nullCst(ctx_.null());
}

BuilderExpr QueryBuilder::nullCst(const Type* type) const {
  auto cst_expr = std::make_shared<Constant>(type, true, Datum{});
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::nullCst(const std::string& type) const {
  return nullCst(ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::date(const std::string& val) const {
  return cst(val, "date");
}

BuilderExpr QueryBuilder::time(const std::string& val) const {
  return cst(val, "time");
}

BuilderExpr QueryBuilder::timestamp(const std::string& val) const {
  return cst(val, "timestamp");
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int>& vals) const {
  return cst(std::vector<int>(vals));
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int> vals, const Type* type) const {
  return cst(std::vector<int>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<int> vals,
                              const std::string& type) const {
  return cst(std::vector<int>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals) const {
  return cst(std::vector<double>(vals));
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals,
                              const Type* type) const {
  return cst(std::vector<double>(vals), type);
}

BuilderExpr QueryBuilder::cst(std::initializer_list<double> vals,
                              const std::string& type) const {
  return cst(std::vector<double>(vals), type);
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals) const {
  return cst(vals, ctx_.arrayVarLen(ctx_.int32()));
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals, const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<int>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals) const {
  return cst(vals, ctx_.arrayVarLen(ctx_.fp64()));
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals, const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<double>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::vector<std::string>& vals,
                              const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    exprs.emplace_back(cst(val, elem_type).expr());
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<std::string>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::cst(const std::vector<BuilderExpr>& vals,
                              const Type* type) const {
  checkCstArrayType(type, vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType();
  ExprPtrList exprs;
  for (auto val : vals) {
    auto elem_val = val.cast(elem_type).expr();
    exprs.emplace_back(elem_val);
  }
  auto cst_expr = std::make_shared<Constant>(type, false, exprs);
  return {this, cst_expr};
}

BuilderExpr QueryBuilder::cst(const std::vector<BuilderExpr>& vals,
                              const std::string& type) const {
  return cst(vals, ctx_.typeFromString(type));
}

BuilderExpr QueryBuilder::ifThenElse(const BuilderExpr& cond,
                                     const BuilderExpr& if_val,
                                     const BuilderExpr& else_val) {
  bool nullable = if_val.type()->nullable() || else_val.type()->nullable();
  bool has_agg = cond.expr()->containsAgg() || if_val.expr()->containsAgg() ||
                 else_val.expr()->containsAgg();
  auto res_type = if_val.type()->withNullable(nullable);
  if (!res_type->equal(else_val.type()->withNullable(nullable))) {
    throw InvalidQueryError() << "Mismatched type for if-then-else values: "
                              << if_val.type()->toString() << " and "
                              << else_val.type()->toString();
  }
  std::list<std::pair<ExprPtr, ExprPtr>> expr_pairs;
  expr_pairs.emplace_back(std::make_pair(cond.expr(), if_val.expr()));
  auto case_expr =
      std::make_shared<CaseExpr>(res_type, has_agg, expr_pairs, else_val.expr());
  return {this, case_expr};
}

hdk::ir::BuilderExpr operator+(const hdk::ir::BuilderExpr& lhs,
                               const hdk::ir::BuilderExpr& rhs) {
  return lhs.add(rhs);
}

hdk::ir::BuilderExpr operator+(const hdk::ir::BuilderExpr& lhs, int rhs) {
  return lhs.add(rhs);
}

hdk::ir::BuilderExpr operator+(const hdk::ir::BuilderExpr& lhs, int64_t rhs) {
  return lhs.add(rhs);
}

hdk::ir::BuilderExpr operator+(const hdk::ir::BuilderExpr& lhs, float rhs) {
  return lhs.add(rhs);
}

hdk::ir::BuilderExpr operator+(const hdk::ir::BuilderExpr& lhs, double rhs) {
  return lhs.add(rhs);
}

hdk::ir::BuilderExpr operator+(int lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).add(rhs);
}

hdk::ir::BuilderExpr operator+(int64_t lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).add(rhs);
}

hdk::ir::BuilderExpr operator+(float lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).add(rhs);
}

hdk::ir::BuilderExpr operator+(double lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).add(rhs);
}

hdk::ir::BuilderExpr operator-(const hdk::ir::BuilderExpr& lhs,
                               const hdk::ir::BuilderExpr& rhs) {
  return lhs.sub(rhs);
}

hdk::ir::BuilderExpr operator-(const hdk::ir::BuilderExpr& lhs, int rhs) {
  return lhs.sub(rhs);
}

hdk::ir::BuilderExpr operator-(const hdk::ir::BuilderExpr& lhs, int64_t rhs) {
  return lhs.sub(rhs);
}

hdk::ir::BuilderExpr operator-(const hdk::ir::BuilderExpr& lhs, float rhs) {
  return lhs.sub(rhs);
}

hdk::ir::BuilderExpr operator-(const hdk::ir::BuilderExpr& lhs, double rhs) {
  return lhs.sub(rhs);
}

hdk::ir::BuilderExpr operator-(int lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).sub(rhs);
}

hdk::ir::BuilderExpr operator-(int64_t lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).sub(rhs);
}

hdk::ir::BuilderExpr operator-(float lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).sub(rhs);
}

hdk::ir::BuilderExpr operator-(double lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).sub(rhs);
}

hdk::ir::BuilderExpr operator*(const hdk::ir::BuilderExpr& lhs,
                               const hdk::ir::BuilderExpr& rhs) {
  return lhs.mul(rhs);
}

hdk::ir::BuilderExpr operator*(const hdk::ir::BuilderExpr& lhs, int rhs) {
  return lhs.mul(rhs);
}

hdk::ir::BuilderExpr operator*(const hdk::ir::BuilderExpr& lhs, int64_t rhs) {
  return lhs.mul(rhs);
}

hdk::ir::BuilderExpr operator*(const hdk::ir::BuilderExpr& lhs, float rhs) {
  return lhs.mul(rhs);
}

hdk::ir::BuilderExpr operator*(const hdk::ir::BuilderExpr& lhs, double rhs) {
  return lhs.mul(rhs);
}

hdk::ir::BuilderExpr operator*(int lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).mul(rhs);
}

hdk::ir::BuilderExpr operator*(int64_t lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).mul(rhs);
}

hdk::ir::BuilderExpr operator*(float lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).mul(rhs);
}

hdk::ir::BuilderExpr operator*(double lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).mul(rhs);
}

hdk::ir::BuilderExpr operator/(const hdk::ir::BuilderExpr& lhs,
                               const hdk::ir::BuilderExpr& rhs) {
  return lhs.div(rhs);
}

hdk::ir::BuilderExpr operator/(const hdk::ir::BuilderExpr& lhs, int rhs) {
  return lhs.div(rhs);
}

hdk::ir::BuilderExpr operator/(const hdk::ir::BuilderExpr& lhs, int64_t rhs) {
  return lhs.div(rhs);
}

hdk::ir::BuilderExpr operator/(const hdk::ir::BuilderExpr& lhs, float rhs) {
  return lhs.div(rhs);
}

hdk::ir::BuilderExpr operator/(const hdk::ir::BuilderExpr& lhs, double rhs) {
  return lhs.div(rhs);
}

hdk::ir::BuilderExpr operator/(int lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).div(rhs);
}

hdk::ir::BuilderExpr operator/(int64_t lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).div(rhs);
}

hdk::ir::BuilderExpr operator/(float lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).div(rhs);
}

hdk::ir::BuilderExpr operator/(double lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).div(rhs);
}

hdk::ir::BuilderExpr operator%(const hdk::ir::BuilderExpr& lhs,
                               const hdk::ir::BuilderExpr& rhs) {
  return lhs.mod(rhs);
}

hdk::ir::BuilderExpr operator%(const hdk::ir::BuilderExpr& lhs, int rhs) {
  return lhs.mod(rhs);
}

hdk::ir::BuilderExpr operator%(const hdk::ir::BuilderExpr& lhs, int64_t rhs) {
  return lhs.mod(rhs);
}

hdk::ir::BuilderExpr operator%(int lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).mod(rhs);
}

hdk::ir::BuilderExpr operator%(int64_t lhs, const hdk::ir::BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).mod(rhs);
}

hdk::ir::BuilderExpr operator&&(const hdk::ir::BuilderExpr& lhs,
                                const hdk::ir::BuilderExpr& rhs) {
  return lhs.logicalAnd(rhs);
}

hdk::ir::BuilderExpr operator||(const hdk::ir::BuilderExpr& lhs,
                                const hdk::ir::BuilderExpr& rhs) {
  return lhs.logicalOr(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, int rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, float rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, double rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.eq(rhs);
}

BuilderExpr operator==(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).eq(rhs);
}

BuilderExpr operator==(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).eq(rhs);
}

BuilderExpr operator==(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).eq(rhs);
}

BuilderExpr operator==(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).eq(rhs);
}

BuilderExpr operator==(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).eq(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, int rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, float rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, double rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.ne(rhs);
}

BuilderExpr operator!=(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).ne(rhs);
}

BuilderExpr operator!=(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).ne(rhs);
}

BuilderExpr operator!=(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).ne(rhs);
}

BuilderExpr operator!=(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).ne(rhs);
}

BuilderExpr operator!=(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).ne(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, int rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, float rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, double rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.lt(rhs);
}

BuilderExpr operator<(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).lt(rhs);
}

BuilderExpr operator<(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).lt(rhs);
}

BuilderExpr operator<(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).lt(rhs);
}

BuilderExpr operator<(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).lt(rhs);
}

BuilderExpr operator<(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).lt(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, int rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, float rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, double rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.le(rhs);
}

BuilderExpr operator<=(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).le(rhs);
}

BuilderExpr operator<=(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).le(rhs);
}

BuilderExpr operator<=(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).le(rhs);
}

BuilderExpr operator<=(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).le(rhs);
}

BuilderExpr operator<=(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).le(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, int rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, float rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, double rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.gt(rhs);
}

BuilderExpr operator>(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).gt(rhs);
}

BuilderExpr operator>(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).gt(rhs);
}

BuilderExpr operator>(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).gt(rhs);
}

BuilderExpr operator>(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).gt(rhs);
}

BuilderExpr operator>(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).gt(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, const BuilderExpr& rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, int rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, int64_t rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, float rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, double rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(const BuilderExpr& lhs, const std::string& rhs) {
  return lhs.ge(rhs);
}

BuilderExpr operator>=(int lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int32(false)).ge(rhs);
}

BuilderExpr operator>=(int64_t lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().int64(false)).ge(rhs);
}

BuilderExpr operator>=(float lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp32(false)).ge(rhs);
}

BuilderExpr operator>=(double lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs, rhs.ctx().fp64(false)).ge(rhs);
}

BuilderExpr operator>=(const std::string& lhs, const BuilderExpr& rhs) {
  return rhs.builder().cst(lhs).ge(rhs);
}

}  // namespace hdk::ir
