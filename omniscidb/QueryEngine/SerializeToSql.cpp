/*
 * Copyright 2019 OmniSci, Inc.
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

#include "SerializeToSql.h"
#include "ExternalExecutor.h"

ScalarExprToSql::ScalarExprToSql(const RelAlgExecutionUnit* ra_exe_unit,
                                 SchemaProviderPtr schema_provider)
    : ra_exe_unit_(ra_exe_unit), schema_provider_(schema_provider) {}

std::string ScalarExprToSql::visitVar(const hdk::ir::Var* var) const {
  auto it = ra_exe_unit_->groupby_exprs.begin();
  std::advance(it, var->varNo() - 1);
  return visit(it->get());
}

std::string ScalarExprToSql::visitColumnVar(const hdk::ir::ColumnVar* col_var) const {
  return serialize_table_ref(col_var->dbId(), col_var->tableId(), schema_provider_) +
         "." +
         serialize_column_ref(
             col_var->dbId(), col_var->tableId(), col_var->columnId(), schema_provider_);
}

std::string ScalarExprToSql::visitConstant(const hdk::ir::Constant* constant) const {
  if (constant->isNull()) {
    return "NULL";
  }
  auto constant_type = constant->type();
  const auto result = DatumToString(constant->value(), constant_type);
  if (constant_type->isString() || constant_type->isExtDictionary()) {
    return "'" + result + "'";
  } else {
    return result;
  }
}

std::string ScalarExprToSql::visitUOper(const hdk::ir::UOper* uoper) const {
  const auto operand = uoper->operand();
  const auto operand_str = visit(operand);
  const auto optype = uoper->opType();
  switch (optype) {
    case kNOT: {
      return "NOT (" + operand_str + ")";
    }
    case kUMINUS: {
      return "-" + operand_str;
    }
    case kISNULL: {
      return operand_str + " IS NULL";
    }
    case kCAST: {
      auto operand_type = operand->type();
      auto target_type = uoper->type();
      if (!is_supported_type_for_extern_execution(target_type)) {
        throw std::runtime_error("Type not supported yet for extern execution: " +
                                 target_type->toString());
      }
      if ((operand_type->id() == target_type->id() &&
           operand_type->size() == target_type->size()) ||
          ((operand_type->isString() || operand_type->isExtDictionary()) &&
           (target_type->isString() && target_type->isExtDictionary()))) {
        return operand_str;
      }
      return "CAST(" + operand_str + " AS " + hdk::ir::sqlTypeName(target_type) + ")";
    }
    default: {
      throw std::runtime_error("Unary operator type: " + std::to_string(optype) +
                               " not supported");
    }
  }
}

std::string ScalarExprToSql::visitBinOper(const hdk::ir::BinOper* bin_oper) const {
  return visit(bin_oper->leftOperand()) + " " + binOpTypeToString(bin_oper->opType()) +
         " " + visit(bin_oper->rightOperand());
}

std::string ScalarExprToSql::visitInValues(const hdk::ir::InValues* in_values) const {
  const auto needle = visit(in_values->arg());
  const auto haystack = visitList(in_values->valueList());
  return needle + " IN (" + boost::algorithm::join(haystack, ", ") + ")";
}

std::string ScalarExprToSql::visitLikeExpr(const hdk::ir::LikeExpr* like) const {
  const auto str = visit(like->arg());
  const auto pattern = visit(like->likeExpr());
  const auto result = str + " LIKE " + pattern;
  if (like->escapeExpr()) {
    const auto escape = visit(like->escapeExpr());
    return result + " ESCAPE " + escape;
  }
  return result;
}

std::string ScalarExprToSql::visitCaseExpr(const hdk::ir::CaseExpr* case_) const {
  std::string case_str = "CASE ";
  const auto& expr_pair_list = case_->exprPairs();
  for (const auto& expr_pair : expr_pair_list) {
    const auto when = "WHEN " + visit(expr_pair.first.get());
    const auto then = " THEN " + visit(expr_pair.second.get());
    case_str += when + then;
  }
  return case_str + " ELSE " + visit(case_->elseExpr()) + " END";
}

namespace {

std::string agg_to_string(const hdk::ir::AggExpr* agg_expr,
                          const RelAlgExecutionUnit* ra_exe_unit,
                          SchemaProviderPtr schema_provider) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, schema_provider);
  const auto agg_type = ::toString(agg_expr->aggType());
  const auto arg = agg_expr->arg() ? scalar_expr_to_sql.visit(agg_expr->arg()) : "*";
  const auto distinct = agg_expr->isDistinct() ? "DISTINCT " : "";
  return agg_type + "(" + distinct + arg + ")";
}

}  // namespace

std::string ScalarExprToSql::visitFunctionOper(
    const hdk::ir::FunctionOper* func_oper) const {
  std::string result = func_oper->name();
  if (result == "||") {
    CHECK_EQ(func_oper->arity(), size_t(2));
    return visit(func_oper->arg(0)) + "||" + visit(func_oper->arg(1));
  }
  if (result == "SUBSTRING") {
    result = "SUBSTR";
  }
  std::vector<std::string> arg_strs;
  for (size_t i = 0; i < func_oper->arity(); ++i) {
    arg_strs.push_back(visit(func_oper->arg(i)));
  }
  return result + "(" + boost::algorithm::join(arg_strs, ",") + ")";
}

std::string ScalarExprToSql::visitWindowFunction(
    const hdk::ir::WindowFunction* window_func) const {
  std::string result = ::toString(window_func->kind());
  {
    const auto arg_strs = visitList(window_func->args());
    result += "(" + boost::algorithm::join(arg_strs, ",") + ")";
  }
  result += " OVER (";
  {
    const auto partition_strs = visitList(window_func->partitionKeys());
    if (!partition_strs.empty()) {
      result += "PARTITION BY " + boost::algorithm::join(partition_strs, ",");
    }
  }
  {
    std::vector<std::string> order_strs;
    const auto& order_keys = window_func->orderKeys();
    const auto& collation = window_func->collation();
    CHECK_EQ(order_keys.size(), collation.size());
    for (size_t i = 0; i < order_keys.size(); ++i) {
      std::string order_str = visit(order_keys[i].get());
      order_str += collation[i].is_desc ? " DESC" : " ASC";
      // TODO: handle nulls first / last
      order_strs.push_back(order_str);
    }
    if (!order_strs.empty()) {
      result += " ORDER BY " + boost::algorithm::join(order_strs, ",");
    }
  }
  result += ")";
  return result;
}

std::string ScalarExprToSql::visitAggExpr(const hdk::ir::AggExpr* agg) const {
  return agg_to_string(agg, ra_exe_unit_, schema_provider_);
}

std::string ScalarExprToSql::binOpTypeToString(const SQLOps op_type) {
  switch (op_type) {
    case kEQ:
      return "=";
    case kNE:
      return "<>";
    case kLT:
      return "<";
    case kLE:
      return "<=";
    case kGT:
      return ">";
    case kGE:
      return ">=";
    case kAND:
      return "AND";
    case kOR:
      return "OR";
    case kMINUS:
      return "-";
    case kPLUS:
      return "+";
    case kMULTIPLY:
      return "*";
    case kDIVIDE:
      return "/";
    case kMODULO:
      return "%";
    case kARRAY_AT:
      return "[]";
    default:
      LOG(FATAL) << "Invalid operator type: " << op_type;
      return "";
  }
}

template <typename List>
std::vector<std::string> ScalarExprToSql::visitList(const List& expressions) const {
  std::vector<std::string> result;
  for (const auto& expression : expressions) {
    result.push_back(visit(expression.get()));
  }
  return result;
}

namespace {

std::string where_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                            SchemaProviderPtr schema_provider) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, schema_provider);
  auto qual_strings = scalar_expr_to_sql.visitList(ra_exe_unit->quals);
  const auto simple_qual_strings =
      scalar_expr_to_sql.visitList(ra_exe_unit->simple_quals);
  qual_strings.insert(
      qual_strings.end(), simple_qual_strings.begin(), simple_qual_strings.end());
  return boost::algorithm::join(qual_strings, " AND ");
}

std::string join_condition_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                                     SchemaProviderPtr schema_provider) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, schema_provider);
  std::vector<std::string> qual_strings;
  for (const auto& join_level_quals : ra_exe_unit->join_quals) {
    const auto level_qual_strings = scalar_expr_to_sql.visitList(join_level_quals.quals);
    qual_strings.insert(
        qual_strings.end(), level_qual_strings.begin(), level_qual_strings.end());
  }
  return boost::algorithm::join(qual_strings, " AND ");
}

std::string targets_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                              SchemaProviderPtr schema_provider) {
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, schema_provider);
  std::vector<std::string> target_strings;
  for (const auto target : ra_exe_unit->target_exprs) {
    target_strings.push_back(scalar_expr_to_sql.visit(target));
  }
  return boost::algorithm::join(target_strings, ", ");
}

std::string group_by_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                               SchemaProviderPtr schema_provider) {
  if (ra_exe_unit->groupby_exprs.size() == 1 || !ra_exe_unit->groupby_exprs.front()) {
    return "";
  }
  ScalarExprToSql scalar_expr_to_sql(ra_exe_unit, schema_provider);
  const auto group_by_strings = scalar_expr_to_sql.visitList(ra_exe_unit->groupby_exprs);
  return boost::algorithm::join(group_by_strings, ", ");
}

std::string from_to_string(const RelAlgExecutionUnit* ra_exe_unit,
                           SchemaProviderPtr schema_provider) {
  std::vector<std::string> from_strings;
  for (const auto& input_desc : ra_exe_unit->input_descs) {
    const auto table_ref = serialize_table_ref(
        input_desc.getDatabaseId(), input_desc.getTableId(), schema_provider);
    from_strings.push_back(table_ref);
  }
  return boost::algorithm::join(from_strings, ", ");
}

std::string maybe(const std::string& prefix, const std::string& clause) {
  return clause.empty() ? "" : " " + prefix + " " + clause;
}

}  // namespace

std::string serialize_table_ref(int db_id,
                                const int table_id,
                                SchemaProviderPtr schema_provider) {
  if (table_id >= 0) {
    const auto table_info = schema_provider->getTableInfo(db_id, table_id);
    CHECK(table_info);
    return table_info->name;
  }
  return "\"#temp" + std::to_string(table_id) + "\"";
}

std::string serialize_column_ref(int db_id,
                                 const int table_id,
                                 const int column_id,
                                 SchemaProviderPtr schema_provider) {
  if (table_id >= 0) {
    const auto col_info = schema_provider->getColumnInfo(db_id, table_id, column_id);
    CHECK(col_info);
    return col_info->name;
  }
  return "col" + std::to_string(column_id);
}

ExecutionUnitSql serialize_to_sql(const RelAlgExecutionUnit* ra_exe_unit,
                                  SchemaProviderPtr schema_provider) {
  const auto targets = targets_to_string(ra_exe_unit, schema_provider);
  const auto from = from_to_string(ra_exe_unit, schema_provider);
  const auto join_on = join_condition_to_string(ra_exe_unit, schema_provider);
  const auto where = where_to_string(ra_exe_unit, schema_provider);
  const auto group = group_by_to_string(ra_exe_unit, schema_provider);
  return {"SELECT " + targets + " FROM " + from + maybe("ON", join_on) +
              maybe("WHERE", where) + maybe("GROUP BY", group),
          from};
}
