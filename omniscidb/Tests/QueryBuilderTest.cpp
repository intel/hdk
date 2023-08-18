/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArrowTestHelpers.h"
#include "TestHelpers.h"

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "ConfigBuilder/ConfigBuilder.h"
#include "IR/Expr.h"
#include "IR/Node.h"
#include "QueryBuilder/QueryBuilder.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/QueryExecutionSequence.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SchemaMgr.h"
#include "SchemaMgr/SimpleSchemaProvider.h"
#include "Shared/DateTimeParser.h"

#include <gtest/gtest.h>
#include <boost/numeric/conversion/cast.hpp>
#include <typeinfo>

using namespace std::string_literals;
using namespace ArrowTestHelpers;
using namespace TestHelpers::ArrowSQLRunner;
using namespace hdk;
using namespace hdk::ir;

constexpr int TEST_TABLE_ID1 = 1;
constexpr int TEST_TABLE_ID2 = 2;
constexpr int TEST_TABLE_ID3 = 3;

EXTERN extern bool g_enable_table_functions;

namespace {

std::string getFilePath(const std::string& file_name) {
  return TEST_SOURCE_PATH + "/ArrowStorageDataFiles/"s + file_name;
}

template <typename R, hdk::ir::Type::Id TYPE>
R dateTimeParse(std::string_view const s, hdk::ir::TimeUnit unit) {
  if (auto const time = dateTimeParseOptional<TYPE>(s, unit)) {
    try {
      return boost::numeric_cast<R>(*time);
    } catch (const std::bad_cast& e) {
      throw std::runtime_error(
          cat("numeric_cast<", typeid(R).name(), ">() failed with ", e.what()));
    }
  } else {
    throw std::runtime_error(cat(
        "Invalid date/time (templated) (", std::to_string(TYPE), ") string (", s, ')'));
  }
}

class TestSuite : public ::testing::Test {
 public:
  ExecutionResult runQuery(std::unique_ptr<QueryDag> dag) {
    auto ra_executor = RelAlgExecutor(getExecutor(), getStorage(), std::move(dag));
    auto eo = ExecutionOptions::fromConfig(config());
    return ra_executor.executeRelAlgQuery(
        getCompilationOptions(ExecutorDeviceType::CPU), eo, false);
  }
};

void checkRef(const BuilderExpr& expr,
              NodePtr node,
              unsigned int col_idx,
              const std::string& col_name,
              bool auto_named = true) {
  ASSERT_TRUE(expr.expr()->is<hdk::ir::ColumnRef>());
  auto col_ref = expr.expr()->as<hdk::ir::ColumnRef>();
  ASSERT_EQ(col_ref->node(), node.get());
  ASSERT_EQ(col_ref->index(), col_idx);
  ASSERT_EQ(expr.name(), col_name);
  ASSERT_EQ(expr.isAutoNamed(), auto_named);
}

void checkAgg(const BuilderExpr& expr,
              const Type* type,
              AggType kind,
              bool is_distinct,
              const std::string& name,
              double val = HUGE_VAL,
              Interpolation interpolation = Interpolation::kLinear) {
  ASSERT_TRUE(expr.expr()->is<hdk::ir::AggExpr>());
  auto agg = expr.expr()->as<hdk::ir::AggExpr>();
  ASSERT_EQ(agg->type()->toString(), type->toString());
  ASSERT_EQ(agg->aggType(), kind);
  ASSERT_EQ(agg->isDistinct(), is_distinct);
  ASSERT_EQ(expr.name(), name);
  if (val != HUGE_VAL) {
    ASSERT_TRUE(agg->arg1());
    ASSERT_NEAR(agg->arg1()->as<hdk::ir::Constant>()->fpVal(), val, 0.001);

    if (kind == AggType::kQuantile) {
      ASSERT_EQ(agg->interpolation(), interpolation);
    }
  }
}

void checkAgg(const BuilderExpr& expr,
              const Type* type,
              AggType kind,
              bool is_distinct,
              const std::string& name,
              int val) {
  checkAgg(expr, type, kind, is_distinct, name);
  auto agg = expr.expr()->as<hdk::ir::AggExpr>();
  ASSERT_TRUE(agg->arg1());
  ASSERT_TRUE(agg->arg1()->type()->isInteger());
  ASSERT_NEAR(agg->arg1()->as<hdk::ir::Constant>()->intVal(), val, 0.001);
}

void checkExtract(const BuilderExpr& expr, DateExtractField field, bool cast = false) {
  ASSERT_TRUE(expr.expr()->is<hdk::ir::ExtractExpr>());
  auto extract = expr.expr()->as<ExtractExpr>();
  ASSERT_TRUE(extract->type()->isInt64());
  ASSERT_EQ(extract->type()->nullable(), extract->from()->type()->nullable());
  ASSERT_EQ(extract->field(), field);
  if (cast) {
    ASSERT_TRUE(extract->from()->is<UOper>());
    ASSERT_TRUE(extract->from()->as<UOper>()->isCast());
  }
}

void checkCast(const BuilderExpr& expr, const Type* type) {
  ASSERT_TRUE(expr.expr()->is<UOper>());
  ASSERT_TRUE(expr.expr()->as<UOper>()->isCast());
  ASSERT_TRUE(expr.expr()->type()->equal(type));
}

void checkBoolCastThroughCase(const BuilderExpr& expr,
                              const Type* type,
                              bool nullable = true) {
  ASSERT_TRUE(expr.expr()->is<CaseExpr>());
  auto case_expr = expr.expr()->as<CaseExpr>();
  if (nullable) {
    auto& pairs = case_expr->exprPairs();
    auto else_expr = case_expr->elseExpr();
    ASSERT_EQ(pairs.size(), (size_t)1);
    ASSERT_TRUE(pairs.front().first->is<UOper>());
    ASSERT_TRUE(pairs.front().first->as<UOper>()->isNot());
    ASSERT_TRUE(pairs.front().first->as<UOper>()->operand()->is<UOper>());
    ASSERT_TRUE(pairs.front().first->as<UOper>()->operand()->as<UOper>()->isIsNull());
    ASSERT_TRUE(else_expr->is<Constant>());
    ASSERT_TRUE(else_expr->as<Constant>()->isNull());
    ASSERT_TRUE(else_expr->type()->equal(type));
    ASSERT_TRUE(pairs.front().second->is<CaseExpr>());
    case_expr = pairs.front().second->as<CaseExpr>();
  }
  auto& pairs = case_expr->exprPairs();
  auto else_expr = case_expr->elseExpr();
  ASSERT_EQ(pairs.size(), (size_t)1);
  ASSERT_TRUE(pairs.front().first->is<hdk::ir::ColumnRef>());
  ASSERT_TRUE(pairs.front().second->is<Constant>());
  ASSERT_TRUE(pairs.front().second->type()->equal(type->withNullable(false)));
  ASSERT_TRUE(else_expr->is<Constant>());
  ASSERT_TRUE(else_expr->type()->equal(type->withNullable(false)));
  if (type->isInteger()) {
    ASSERT_EQ(pairs.front().second->as<Constant>()->intVal(), 1);
    ASSERT_EQ(else_expr->as<Constant>()->intVal(), 0);
  } else if (type->isDecimal()) {
    ASSERT_EQ(pairs.front().second->as<Constant>()->intVal(),
              (int64_t)exp_to_scale(type->as<DecimalType>()->scale()));
    ASSERT_EQ(else_expr->as<Constant>()->intVal(), 0);
  } else {
    UNREACHABLE();
  }
}

void checkUOper(const BuilderExpr& expr, const Type* type, OpType op_type) {
  ASSERT_TRUE(expr.expr()->is<UOper>());
  auto uoper = expr.expr()->as<UOper>();
  ASSERT_TRUE(uoper->type()->equal(type));
  ASSERT_EQ(uoper->opType(), op_type);
}

void checkUOper(const BuilderExpr& expr,
                const Type* type,
                OpType op_type,
                const BuilderExpr& op) {
  checkUOper(expr, type, op_type);
  ASSERT_EQ(expr.expr()->as<UOper>()->operand()->toString(), op.expr()->toString());
}

void checkBinOper(const BuilderExpr& expr,
                  const Type* type,
                  OpType op_type,
                  const BuilderExpr& lhs,
                  const BuilderExpr& rhs) {
  ASSERT_TRUE(expr.expr()->is<BinOper>());
  auto bin_oper = expr.expr()->as<BinOper>();
  ASSERT_TRUE(bin_oper->type()->equal(type))
      << bin_oper->type()->toString() << " vs. " << type->toString();
  ASSERT_EQ(bin_oper->opType(), op_type) << bin_oper->opType() << " vs. " << op_type;
  ASSERT_EQ(bin_oper->qualifier(), Qualifier::kOne) << bin_oper->qualifier();
  ASSERT_EQ(bin_oper->leftOperand()->toString(), lhs.expr()->toString());
  ASSERT_EQ(bin_oper->rightOperand()->toString(), rhs.expr()->toString());
}

void checkDateAdd(const BuilderExpr& expr,
                  const Type* type,
                  DateAddField field,
                  const BuilderExpr& number,
                  const BuilderExpr& date) {
  ASSERT_TRUE(expr.expr()->is<DateAddExpr>());
  auto add_expr = expr.expr()->as<DateAddExpr>();
  ASSERT_TRUE(add_expr->type()->equal(type));
  ASSERT_EQ(add_expr->field(), field);
  ASSERT_EQ(add_expr->number()->toString(), number.expr()->toString());
  ASSERT_EQ(add_expr->datetime()->toString(), date.expr()->toString());
}

void checkCst(const ExprPtr& expr, int64_t val, const Type* type) {
  ASSERT_TRUE(expr->is<Constant>());
  ASSERT_EQ(expr->type()->toString(), type->toString());
  ASSERT_TRUE(type->isInteger() || type->isDecimal() || type->isBoolean() ||
              type->isDateTime() || type->isInterval())
      << type->toString();
  ASSERT_EQ(expr->as<Constant>()->intVal(), val);
}

void checkCst(const BuilderExpr& expr, int64_t val, const Type* type) {
  checkCst(expr.expr(), val, type);
}

void checkCst(const ExprPtr& expr, int val, const Type* type) {
  checkCst(expr, static_cast<int64_t>(val), type);
}

void checkCst(const BuilderExpr& expr, int val, const Type* type) {
  checkCst(expr, static_cast<int64_t>(val), type);
}

void checkCst(const ExprPtr& expr, double val, const Type* type) {
  ASSERT_TRUE(expr->is<Constant>());
  ASSERT_EQ(expr->type()->toString(), type->toString());
  ASSERT_TRUE(type->isFloatingPoint());
  ASSERT_NEAR(expr->as<Constant>()->fpVal(), val, 0.0001);
}

void checkCst(const BuilderExpr& expr, double val, const Type* type) {
  checkCst(expr.expr(), val, type);
}

void checkCst(const BuilderExpr& expr, const std::string& val, const Type* type) {
  ASSERT_TRUE(expr.expr()->is<Constant>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_TRUE(type->isString());
  ASSERT_EQ(*expr.expr()->as<Constant>()->value().stringval, val);
}

void checkCst(const BuilderExpr& expr, bool val, const Type* type) {
  checkCst(expr, (int64_t)val, type);
}

void checkNullCst(const BuilderExpr& expr, const Type* type) {
  ASSERT_TRUE(expr.expr()->is<Constant>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_TRUE(expr.expr()->as<Constant>()->isNull());
}

template <typename T>
void checkCst(const BuilderExpr& expr, std::initializer_list<T> vals, const Type* type) {
  ASSERT_TRUE(type->isArray());
  ASSERT_TRUE(expr.expr()->is<Constant>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_FALSE(expr.expr()->as<Constant>()->isNull());
  auto& exprs = expr.expr()->as<Constant>()->valueList();
  ASSERT_EQ(exprs.size(), vals.size());
  auto elem_type = type->as<ArrayBaseType>()->elemType()->withNullable(false);
  auto val_idx = 0;
  for (auto& elem_expr : exprs) {
    checkCst(elem_expr, std::data(vals)[val_idx], elem_type);
    ++val_idx;
  }
}

void checkFunctionOper(const BuilderExpr& expr,
                       const std::string& fn_name,
                       size_t arity,
                       const Type* type) {
  ASSERT_TRUE(expr.expr()->is<FunctionOper>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_EQ(expr.expr()->as<FunctionOper>()->arity(), arity);
  ASSERT_EQ(expr.expr()->as<FunctionOper>()->name(), fn_name);
}

void checkWindowFunction(const BuilderExpr& expr,
                         const std::string& name,
                         const Type* type,
                         WindowFunctionKind kind,
                         size_t args,
                         size_t part_keys,
                         size_t order_keys) {
  CHECK_EQ(expr.name(), name);
  CHECK_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_TRUE(expr.expr()->is<WindowFunction>());
  auto wnd_fn = expr.expr()->as<WindowFunction>();
  ASSERT_EQ(wnd_fn->kind(), kind);
  ASSERT_EQ(wnd_fn->args().size(), args);
  ASSERT_EQ(wnd_fn->partitionKeys().size(), part_keys);
  ASSERT_EQ(wnd_fn->orderKeys().size(), order_keys);
  ASSERT_EQ(wnd_fn->collation().size(), order_keys);
}

void checkWindowCollation(const BuilderExpr& expr,
                          size_t idx,
                          const BuilderExpr& key,
                          SortDirection dir,
                          NullSortedPosition null_pos) {
  auto wnd_fn = expr.expr()->as<WindowFunction>();
  ASSERT_TRUE(wnd_fn);
  ASSERT_LT(idx, wnd_fn->orderKeys().size());
  ASSERT_TRUE(wnd_fn->orderKeys()[idx]->equal(key.expr().get()));
  ASSERT_EQ(wnd_fn->collation()[idx].is_desc, dir == SortDirection::Descending);
  ASSERT_EQ(wnd_fn->collation()[idx].nulls_first, null_pos == NullSortedPosition::First);
}

void checkCardinality(const BuilderExpr& expr, const BuilderExpr& op) {
  auto cardinality_expr = expr.expr()->as<CardinalityExpr>();
  ASSERT_TRUE(cardinality_expr);
  ASSERT_TRUE(cardinality_expr->arg()->equal(op.expr().get()));
}

}  // anonymous namespace

class QueryBuilderTest : public TestSuite {
 protected:
  static constexpr int TEST_SCHEMA_ID2 = 2;
  static constexpr int TEST_DB_ID2 = (TEST_SCHEMA_ID2 << 24) + 1;

  static void SetUpTestSuite() {
    auto data_mgr = getDataMgr();
    auto ps_mgr = data_mgr->getPersistentStorageMgr();
    storage2_ = std::make_shared<ArrowStorage>(
        TEST_SCHEMA_ID2, "test2", TEST_DB_ID2, configPtr());
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID2, storage2_);
    schema_mgr_ = std::make_shared<SchemaMgr>();
    schema_mgr_->registerProvider(TEST_SCHEMA_ID, getStorage());
    schema_mgr_->registerProvider(TEST_SCHEMA_ID2, storage2_);
    schema_mgr_->registerProvider(hdk::ResultSetRegistry::SCHEMA_ID,
                                  getResultSetRegistry());

    createTable("test1",
                {{"col_bi", ctx().int64()},
                 {"col_i", ctx().int32()},
                 {"col_f", ctx().fp32()},
                 {"col_d", ctx().fp64()}});
    insertCsvValues("test1", "1,11,1.1,11.11\n2,22,2.2,22.22\n3,33,3.3,33.33");
    insertCsvValues("test1", "4,44,4.4,44.44\n5,55,5.5,55.55");

    createTable("test2",
                {{"id1", ctx().int32()},
                 {"id2", ctx().int32()},
                 {"val1", ctx().int32()},
                 {"val2", ctx().int32()}});
    insertCsvValues("test2", "1,1,10,20\n1,2,11,21\n1,2,12,22\n2,1,13,23\n2,2,14,24");
    insertCsvValues("test2", "1,1,15,25\n1,2,,26\n1,2,17,27\n2,1,,28\n2,2,19,29");

    createTable("test3",
                {
                    {"col_bi", ctx().int64()},
                    {"col_i", ctx().int32()},
                    {"col_f", ctx().fp32()},
                    {"col_d", ctx().fp64()},
                    {"col_dec", ctx().decimal64(10, 2)},
                    {"col_b", ctx().boolean()},
                    {"col_str", ctx().text()},
                    {"col_dict", ctx().extDict(ctx().text(), 0)},
                    {"col_date", ctx().date32(hdk::ir::TimeUnit::kDay)},
                    {"col_time", ctx().time64(hdk::ir::TimeUnit::kSecond)},
                    {"col_timestamp", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
                    {"col_arr_i32", ctx().arrayVarLen(ctx().int32())},
                    {"col_arr_i64", ctx().arrayVarLen(ctx().int64())},
                    {"col_arr_i32_2", ctx().arrayVarLen(ctx().int32())},
                    {"col_arr_i32x3", ctx().arrayFixed(3, ctx().int32())},
                    {"col_arr_i32x3_2", ctx().arrayFixed(3, ctx().int32())},
                    {"col_dec2", ctx().decimal64(5, 1)},
                    {"col_dec3", ctx().decimal64(14, 4)},
                    {"col_dict2", ctx().extDict(ctx().text(), -1)},
                    {"col_dict3", ctx().extDict(ctx().text(), -1)},
                    {"col_date2", ctx().date16(hdk::ir::TimeUnit::kDay)},
                    {"col_date3", ctx().date32(hdk::ir::TimeUnit::kSecond)},
                    {"col_date4", ctx().date64(hdk::ir::TimeUnit::kSecond)},
                    {"col_time2", ctx().time64(hdk::ir::TimeUnit::kMilli)},
                    {"col_timestamp2", ctx().timestamp(hdk::ir::TimeUnit::kMilli)},
                    {"col_timestamp3", ctx().timestamp(hdk::ir::TimeUnit::kMicro)},
                    {"col_timestamp4", ctx().timestamp(hdk::ir::TimeUnit::kNano)},
                    {"col_si", ctx().int16()},
                    {"col_ti", ctx().int8()},
                    {"col_b_nn", ctx().boolean(false)},
                    {"col_vc_10", ctx().varChar(10)},
                });

    createTable("test_str", {{"id", ctx().int32()}, {"str", ctx().text()}});
    insertCsvValues("test_str", ",\n1,str1\n,\n3,str333\n,\n5,str55555");

    createTable("test_varr",
                {{"id", ctx().int32()},
                 {"arr1", ctx().arrayVarLen(ctx().int32())},
                 {"arr2", ctx().arrayVarLen(ctx().fp64())}});
    insertJsonValues("test_varr",
                     R"___({"id": 1, "arr1":[1, null, 3], "arr2" : [4.0, null]}
                 {"id": 2, "arr1":null, "arr2" : []}
                 {"id": 3, "arr1":[], "arr2" : null}
                 {"id": 4, "arr1":[null, 2, null, 4], "arr2" : [null, 5.0, 6.0]})___");

    createTable("test_arr",
                {{"id", ctx().int32()},
                 {"arr1", ctx().arrayFixed(2, ctx().int32())},
                 {"arr2", ctx().arrayFixed(3, ctx().fp64())}});
    insertJsonValues("test_arr",
                     R"___({"id": 1, "arr1": null, "arr2": [4.0, null, 6.0]}
                 {"id": 2, "arr1":[null, 2], "arr2" : null}
                 {"id": 3, "arr1":[1, null], "arr2" : [null, 5.0, null]}
                 {"id": 4, "arr1":[1, 2], "arr2" : [4.0, 5.0, 6.0]})___");

    createTable("sort",
                {{"x", ctx().int32()}, {"y", ctx().int32()}, {"z", ctx().int32()}});
    insertCsvValues("sort",
                    "1,1,1\n2,1,2\n3,1,\n4,3,\n5,3,1\n,3,2\n9,2,1\n8,2,\n7,2,3\n6,2,2");

    createTable("ambiguous", {{"x", ctx().int32()}});
    storage2_->createTable("ambiguous", {{"x", ctx().int32()}});

    createTable("join1", {{"id", ctx().int32()}, {"val1", ctx().int32()}});
    insertCsvValues("join1", "1,101\n2,102\n,103\n4,104\n5,105");

    createTable("join2", {{"id", ctx().int32()}, {"val2", ctx().int32()}});
    insertCsvValues("join2", "2,101\n3,102\n4,103\n,104\n6,105");

    createTable("withNull", {{"a", ctx().int64()}});
    insertCsvValues("withNull", "1\nNULL");

    createTable("test_tmstmp",
                {{"col_bi", ctx().int64()},
                 {"col_tmstp", ctx().timestamp(hdk::ir::TimeUnit::kSecond, false)},
                 {"col_tmstp_ms", ctx().timestamp(hdk::ir::TimeUnit::kMilli, false)},
                 {"col_tmstp_ns", ctx().timestamp(hdk::ir::TimeUnit::kNano, false)}});
    insertCsvValues(
        "test_tmstmp",
        "1,1990-01-01 12:03:17,1990-01-01 12:03:17.123,1990-01-01 12:03:17.001002003\n"
        "2,1950-03-18 22:23:15,1950-03-18 22:23:15.456,1950-03-18 22:23:15.041052063\n");

    createTable("test_tmstmp_nullable",
                {{"col_bi", ctx().int64()},
                 {"col_tmstp", ctx().timestamp(hdk::ir::TimeUnit::kSecond, true)},
                 {"col_tmstp_ms", ctx().timestamp(hdk::ir::TimeUnit::kMilli, true)},
                 {"col_tmstp_ns", ctx().timestamp(hdk::ir::TimeUnit::kNano, true)}});
    insertCsvValues(
        "test_tmstmp_nullable",
        "1,1990-01-01 12:03:17,1990-01-01 12:03:17.123,1990-01-01 12:03:17.001002003\n"
        "2,1950-03-18 22:23:15,1950-03-18 22:23:15.456,1950-03-18 22:23:15.041052063\n"
        "3,NULL,NULL,NULL\n");

    createTable("test_bitwise",
                {{"col_i8_1", ctx().int8()},
                 {"col_i8_2", ctx().int8()},
                 {"col_i8nn_1", ctx().int8(false)},
                 {"col_i8nn_2", ctx().int8(false)},
                 {"col_i16_1", ctx().int16()},
                 {"col_i16_2", ctx().int16()},
                 {"col_i16nn_1", ctx().int16(false)},
                 {"col_i16nn_2", ctx().int16(false)},
                 {"col_i32_1", ctx().int32()},
                 {"col_i32_2", ctx().int32()},
                 {"col_i32nn_1", ctx().int32(false)},
                 {"col_i32nn_2", ctx().int32(false)},
                 {"col_i64_1", ctx().int64()},
                 {"col_i64_2", ctx().int64()},
                 {"col_i64nn_1", ctx().int64(false)},
                 {"col_i64nn_2", ctx().int64(false)},
                 {"col_f", ctx().fp32()},
                 {"col_d", ctx().fp64()},
                 {"col_dec", ctx().decimal64(10, 2)},
                 {"col_b", ctx().boolean()},
                 {"col_str", ctx().text()},
                 {"col_dict", ctx().extDict(ctx().text(), 0)},
                 {"col_date", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"col_time", ctx().time64(hdk::ir::TimeUnit::kSecond)}});
    insertCsvValues("test_bitwise",
                    "1,,1,4,1,,1,4,1,,1,4,1,,1,4,,,,,,,,\n"
                    "2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,,,,,,,,\n"
                    "3,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,,,,,,,,\n"
                    ",1,4,1,,1,4,1,,1,4,1,,1,4,1,,,,,,,,\n");

    createTable("test_cardinality",
                {{"col_arr_i32", ctx().arrayVarLen(ctx().int32())},
                 {"col_arr_i32x2", ctx().arrayFixed(2, ctx().int32())},
                 {"col_arr_i32nn", ctx().arrayVarLen(ctx().int32(), 4, false)},
                 {"col_arr_i32x2nn", ctx().arrayFixed(2, ctx().int32(), false)}});
    insertJsonValues(
        "test_cardinality",
        R"___({"col_arr_i32": [1], "col_arr_i32x2": [0, 1], "col_arr_i32nn": [1, 2], "col_arr_i32x2nn": [3, 4]}
              {"col_arr_i32": null, "col_arr_i32x2": null, "col_arr_i32nn": [1], "col_arr_i32x2nn": [3, 4]}
              {"col_arr_i32": [], "col_arr_i32x2": [0, 1], "col_arr_i32nn": [], "col_arr_i32x2nn": [3, 4]}
              {"col_arr_i32": [1, 2], "col_arr_i32x2": null, "col_arr_i32nn": [null], "col_arr_i32x2nn": [3, 4]})___");

    createTable("test_unnest",
                {{"col_i", ctx().int32()},
                 {"col_arr_i32", ctx().arrayVarLen(ctx().int32())},
                 {"col_arr_i32x2", ctx().arrayFixed(2, ctx().int32())},
                 {"col_arr_i32nn", ctx().arrayVarLen(ctx().int32(), 4, false)},
                 {"col_arr_i32x2nn", ctx().arrayFixed(2, ctx().int32(), false)}});
    insertJsonValues(
        "test_unnest",
        R"___({"col_i": 1, "col_arr_i32": [1], "col_arr_i32x2": [0, 1], "col_arr_i32nn": [1, 2], "col_arr_i32x2nn": [3, 4]}
              {"col_i": 2, "col_arr_i32": null, "col_arr_i32x2": [null, null], "col_arr_i32nn": [1], "col_arr_i32x2nn": [5, 6]}
              {"col_i": 3, "col_arr_i32": [], "col_arr_i32x2": [0, 1], "col_arr_i32nn": [], "col_arr_i32x2nn": [7, 8]}
              {"col_i": 4, "col_arr_i32": [1, 2], "col_arr_i32x2": null, "col_arr_i32nn": [null], "col_arr_i32x2nn": [9, 10]})___");

    createTable("test_topk",
                {{"id1", ctx().int32()},
                 {"id2", ctx().int64()},
                 {"i8", ctx().int8()},
                 {"i8nn", ctx().int8()},
                 {"i16", ctx().int16()},
                 {"i16nn", ctx().int16()},
                 {"i32", ctx().int32()},
                 {"i32nn", ctx().int32()},
                 {"i64", ctx().int64()},
                 {"i64nn", ctx().int64()},
                 {"f32", ctx().fp32()},
                 {"f32nn", ctx().fp32()},
                 {"f64", ctx().fp64()},
                 {"f64nn", ctx().fp64()}},
                {8});
    insertCsvValues("test_topk",
                    "1,10000000000,2,0,20,00,200,000,2000,0000,2.2,0.0,2.22,0.00\n"
                    "1,10000000000,1,2,10,20,100,200,1000,2000,1.1,2.2,1.11,2.22\n"
                    "1,10000000000,4,1,40,10,400,100,4000,1000,4.4,1.1,4.44,1.11\n"
                    "2,20000000000,,0,,00,,000,,0000,,0.0,,0.00\n"
                    "2,20000000000,4,4,40,40,400,400,4000,4000,4.4,4.4,4.44,4.44\n"
                    "3,30000000000,1,5,10,50,100,500,1000,5000,1.1,5.5,1.11,5.55\n"
                    "3,30000000000,3,0,30,00,300,000,3000,0000,3.3,0.0,3.33,0.00\n"
                    "4,40000000000,2,1,20,10,200,100,2000,1000,2.2,1.1,2.22,1.11\n"
                    "1,10000000000,,1,,10,,100,,1000,,1.1,,1.11\n"
                    "1,10000000000,5,2,50,20,500,200,5000,2000,5.5,2.2,5.55,2.22\n"
                    "2,20000000000,5,5,50,50,500,500,5000,5000,5.5,5.5,5.55,5.55\n"
                    "2,20000000000,1,1,10,10,100,100,1000,1000,1.1,1.1,1.11,1.11\n"
                    "3,30000000000,4,4,40,40,400,400,4000,4000,4.4,4.4,4.44,4.44\n"
                    "3,30000000000,,3,,30,,300,,3000,,3.3,,3.33\n"
                    "5,50000000000,3,1,30,10,300,100,3000,1000,3.3,1.1,3.33,1.11\n"
                    "6,60000000000,,2,,20,,200,,2000,,2.2,,2.22\n");

    createTable("test_quantile",
                {{"id1", ctx().int32()},
                 {"id2", ctx().int64()},
                 {"i8", ctx().int8()},
                 {"i8nn", ctx().int8(false)},
                 {"i16", ctx().int16()},
                 {"i16nn", ctx().int16(false)},
                 {"i32", ctx().int32()},
                 {"i32nn", ctx().int32(false)},
                 {"i64", ctx().int64()},
                 {"i64nn", ctx().int64(false)},
                 {"f32", ctx().fp32()},
                 {"f32nn", ctx().fp32(false)},
                 {"f64", ctx().fp64()},
                 {"f64nn", ctx().fp64(false)},
                 {"dec", ctx().decimal64(10, 2)},
                 {"decnn", ctx().decimal64(10, 2, false)}},
                {2});
    insertCsvValues(
        "test_quantile",
        "1,10000000000,1,1,10,10,100,100,1000,1000,1.0,1.0,10.0,10.0,100.0,100.0\n"
        "1,10000000000,2,2,20,20,200,200,2000,2000,2.0,2.0,20.0,20.0,200.0,200.0\n"
        "1,10000000000,3,3,30,30,300,300,3000,3000,3.0,3.0,30.0,30.0,300.0,300.0\n"
        "1,10000000000,4,4,40,40,400,400,4000,4000,4.0,4.0,40.0,40.0,400.0,400.0\n"
        "1,10000000000,5,5,50,50,500,500,5000,5000,5.0,5.0,50.0,50.0,500.0,500.0\n"
        "2,20000000000,1,1,10,10,100,100,1000,1000,1.0,1.0,10.0,10.0,100.0,100.0\n"
        "2,20000000000,2,2,20,20,200,200,2000,2000,2.0,2.0,20.0,20.0,200.0,200.0\n"
        "2,20000000000,3,3,30,30,300,300,3000,3000,3.0,3.0,30.0,30.0,300.0,300.0\n"
        "2,20000000000,,4,,40,,400,,4000,,4.0,,40.0,,400.0\n"
        "1,10000000000,6,6,60,60,600,600,6000,6000,6.0,6.0,60.0,60.0,600.0,600.0\n"
        "1,10000000000,7,7,70,70,700,700,7000,7000,7.0,7.0,70.0,70.0,700.0,700.0\n"
        "1,10000000000,8,8,80,80,800,800,8000,8000,8.0,8.0,80.0,80.0,800.0,800.0\n"
        "1,10000000000,9,9,90,90,900,900,9000,9000,9.0,9.0,90.0,90.0,900.0,900.0\n"
        "3,30000000000,1,1,10,10,100,100,1000,1000,1.0,1.0,10.0,10.0,100.0,100.0\n"
        "3,30000000000,2,2,20,20,200,200,2000,2000,2.0,2.0,20.0,20.0,200.0,200.0\n"
        "3,30000000000,,3,,30,,300,,3000,,3.0,,30.0,,300.0\n"
        "4,40000000000,,1,,10,,100,,1000,,1.0,,10.0,,100.0\n");

    createTable("test_quantile_dt",
                {{"id1", ctx().int32()},
                 {"id2", ctx().int64()},
                 {"t64s", ctx().time64(hdk::ir::TimeUnit::kSecond)},
                 {"t64s2", ctx().time64(hdk::ir::TimeUnit::kSecond)},
                 {"d32", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"d64", ctx().date64(hdk::ir::TimeUnit::kDay)},
                 {"ts64", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
                 {"ts64nn", ctx().timestamp(hdk::ir::TimeUnit::kMilli, false)}},
                {2});
    insertCsvValues("test_quantile_dt",
                    "1,10000000000,00:00:01,00:00:10,1970-01-02,1970-01-11,1970-01-01 "
                    "00:00:01,1970-01-01 00:00:10\n"
                    "1,10000000000,00:00:02,00:00:20,1970-01-03,1970-01-21,1970-01-01 "
                    "00:00:02,1970-01-01 00:00:20\n"
                    "1,10000000000,00:00:03,00:00:30,1970-01-04,1970-01-31,1970-01-01 "
                    "00:00:03,1970-01-01 00:00:30\n"
                    "1,10000000000,00:00:04,00:00:40,1970-01-05,1970-02-10,1970-01-01 "
                    "00:00:04,1970-01-01 00:00:40\n"
                    "1,10000000000,00:00:05,00:00:50,1970-01-06,1970-02-20,1970-01-01 "
                    "00:00:05,1970-01-01 00:00:50\n"
                    "2,20000000000,00:00:01,00:00:10,1970-01-02,1970-01-11,1970-01-01 "
                    "00:00:01,1970-01-01 00:00:10\n"
                    "2,20000000000,00:00:02,00:00:20,1970-01-03,1970-01-21,1970-01-01 "
                    "00:00:02,1970-01-01 00:00:20\n"
                    "2,20000000000,00:00:03,00:00:30,1970-01-04,1970-01-31,1970-01-01 "
                    "00:00:03,1970-01-01 00:00:30\n"
                    "2,20000000000,,00:00:40,,1970-02-10,,1970-01-01 00:00:40\n"
                    "1,10000000000,00:00:06,00:01:00,1970-01-07,1970-03-02,1970-01-01 "
                    "00:00:06,1970-01-01 00:01:00\n"
                    "1,10000000000,00:00:07,00:01:10,1970-01-08,1970-03-12,1970-01-01 "
                    "00:00:07,1970-01-01 00:01:10\n"
                    "1,10000000000,00:00:08,00:01:20,1970-01-09,1970-03-22,1970-01-01 "
                    "00:00:08,1970-01-01 00:01:20\n"
                    "1,10000000000,00:00:09,00:01:30,1970-01-10,1970-04-01,1970-01-01 "
                    "00:00:09,1970-01-01 00:01:30\n"
                    "3,30000000000,00:00:01,00:00:10,1970-01-02,1970-01-11,1970-01-01 "
                    "00:00:01,1970-01-01 00:00:10\n"
                    "3,30000000000,00:00:02,00:00:20,1970-01-03,1970-01-21,1970-01-01 "
                    "00:00:02,1970-01-01 00:00:20\n"
                    "3,30000000000,,00:00:30,,1970-01-31,,1970-01-01 00:00:30\n"
                    "4,40000000000,,00:00:10,,1970-01-11,,1970-01-01 00:00:10\n");
  }

  static void TearDownTestSuite() {
    dropTable("test1");
    dropTable("test2");
    dropTable("test3");
    dropTable("test_str");
    dropTable("test_varr");
    dropTable("test_arr");
    dropTable("sort");
    dropTable("ambiguous");
    dropTable("join1");
    dropTable("join2");
    dropTable("withNull");
    dropTable("test_tmstmp");
    dropTable("test_tmstmp_nullable");
    dropTable("test_bitwise");
    dropTable("test_cardinality");
  }

  void compare_res_fields(const ExecutionResult& res,
                          const std::vector<std::string> fields) {
    auto& meta = res.getTargetsMeta();
    ASSERT_EQ(fields.size(), meta.size());
    for (size_t i = 0; i < meta.size(); ++i) {
      ASSERT_EQ(meta[i].get_resname(), fields[i]);
    }
  }

  void compare_test1_data(BuilderNode&& root,
                          const std::vector<int>& cols,
                          const std::vector<std::string>& fields) {
    auto res = runQuery(root.finalize());
    compare_res_fields(res, fields);

    auto at = toArrow(res);
    for (size_t i = 0; i < cols.size(); ++i) {
      switch (cols[i]) {
        case 0:
          compare_arrow_array(std::vector<int64_t>({1, 2, 3, 4, 5}), at->column(i));
          break;
        case 1:
          compare_arrow_array(std::vector<int32_t>({11, 22, 33, 44, 55}), at->column(i));
          break;
        case 2:
          compare_arrow_array(std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                              at->column(i));
          break;
        case 3:
          compare_arrow_array(std::vector<double>({11.11, 22.22, 33.33, 44.44, 55.55}),
                              at->column(i));
          break;
        case 4:
          compare_arrow_array(std::vector<int64_t>({0, 1, 2, 3, 4}), at->column(i));
          break;
        default:
          ASSERT_TRUE(false);
      }
    }
  }

  void compare_test1_data(BuilderNode&& root, std::vector<int> cols = {0, 1, 2, 3}) {
    const std::vector<std::string> names = {"col_bi", "col_i", "col_f", "col_d", "rowid"};
    std::vector<std::string> fields;
    for (auto col_idx : cols) {
      fields.push_back(names[col_idx]);
    }
    compare_test1_data(std::move(root), cols, fields);
  }

  void compare_test2_agg(BuilderNode&& root,
                         const std::vector<std::string>& keys,
                         const std::vector<std::string>& aggs,
                         const std::vector<std::string>& fields) {
    std::stringstream ss;
    ss << "SELECT ";
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i) {
        ss << ", ";
      }
      ss << keys[i] << " AS " << fields[i];
    }
    for (size_t i = 0; i < aggs.size(); ++i) {
      if (i || !keys.empty()) {
        ss << ", ";
      }
      ss << aggs[i];
    }
    ss << " FROM test2";
    if (!keys.empty()) {
      ss << " GROUP BY ";
      for (size_t i = 0; i < keys.size(); ++i) {
        if (i) {
          ss << ", ";
        }
        ss << fields[i];
      }
    }
    ss << ";";
    auto expected_res = runSqlQuery(ss.str(), ExecutorDeviceType::CPU, false);
    auto expected_at = toArrow(expected_res);
    auto actual_res = runQuery(root.finalize());
    auto actual_at = toArrow(actual_res);

    ASSERT_EQ(actual_res.getTargetsMeta().size(), fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      ASSERT_EQ(actual_res.getTargetsMeta()[i].get_resname(), fields[i]);
    }

    compareArrowTables(expected_at, actual_at);
  }

  void compare_sort(BuilderNode&& root,
                    const std::vector<SortField>& fields,
                    size_t limit,
                    size_t offset) {
    auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "sort");
    auto col_infos = getStorage()->listColumns(*table_info);
    auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
    auto proj = std::make_shared<Project>(ExprPtrVector{getNodeColumnRef(scan.get(), 0),
                                                        getNodeColumnRef(scan.get(), 1),
                                                        getNodeColumnRef(scan.get(), 2)},
                                          std::vector<std::string>({"x", "y", "z"}),
                                          scan);
    auto sort = std::make_shared<hdk::ir::Sort>(fields, limit, offset, proj);
    auto dag = std::make_unique<QueryDag>(configPtr());
    dag->setRootNode(sort);

    auto expected_res = runQuery(std::move(dag));
    auto actual_res = runQuery(root.finalize());
    compareArrowTables(toArrow(expected_res), toArrow(actual_res));
  }

  static SchemaMgrPtr schema_mgr_;
  static std::shared_ptr<ArrowStorage> storage2_;
};

SchemaMgrPtr QueryBuilderTest::schema_mgr_;
std::shared_ptr<ArrowStorage> QueryBuilderTest::storage2_;

TEST_F(QueryBuilderTest, Scan) {
  auto tinfo = getStorage()->getTableInfo(TEST_DB_ID, "test1");
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  compare_test1_data(builder.scan("test1"));
  compare_test1_data(builder.scan(TEST_DB_ID, "test1"));
  compare_test1_data(builder.scan(TEST_DB_ID, tinfo->table_id));
  compare_test1_data(builder.scan(*tinfo));
}

TEST_F(QueryBuilderTest, Arithmetics) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto tinfo_a = builder.scan("withNull");

  auto dag = tinfo_a.proj(tinfo_a.ref("a").sub(1)).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int64_t>({0, NULL_BIGINT}));

  dag = tinfo_a.proj(tinfo_a.ref("a").sub(tinfo_a.ref("a"))).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int64_t>({0, NULL_BIGINT}));
}

TEST_F(QueryBuilderTest, TimestampToTime) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto tinfo_a = builder.scan("test_tmstmp");

  auto dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("int32")).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>({dateTimeParse<int32_t, hdk::ir::Type::kTimestamp>(
                                "1990-01-01 12:03:17", TimeUnit::kSecond),
                            dateTimeParse<int32_t, hdk::ir::Type::kTimestamp>(
                                "1950-03-18 22:23:15", TimeUnit::kSecond)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time64[s]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<hdk::ir::Type::kTime>("12:03:17", TimeUnit::kSecond),
           dateTimeParse<hdk::ir::Type::kTime>("22:23:15", TimeUnit::kSecond)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time64[s]").cast("int64")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<hdk::ir::Type::kTime>("12:03:17", TimeUnit::kSecond),
           dateTimeParse<hdk::ir::Type::kTime>("22:23:15", TimeUnit::kSecond)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[s]").cast("int32")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>(
          {dateTimeParse<int32_t, hdk::ir::Type::kTime>("12:03:17", TimeUnit::kSecond),
           dateTimeParse<int32_t, hdk::ir::Type::kTime>("22:23:15", TimeUnit::kSecond)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[ms]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<hdk::ir::Type::kTime>("12:03:17", TimeUnit::kMilli),
           dateTimeParse<hdk::ir::Type::kTime>("22:23:15", TimeUnit::kMilli)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ms").cast("time64[ns]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<hdk::ir::Type::kTime>("12:03:17.123000000", TimeUnit::kNano),
           dateTimeParse<hdk::ir::Type::kTime>("22:23:15.456000000", TimeUnit::kNano)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ms").cast("time64[ns]").cast("int64"))
            .finalize();
  res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "12:03:17.123000000", TimeUnit::kNano),
                                         dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "22:23:15.456000000", TimeUnit::kNano)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ns").cast("time64[ms]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<hdk::ir::Type::kTime>("12:03:17.001", TimeUnit::kMilli),
           dateTimeParse<hdk::ir::Type::kTime>("22:23:15.041", TimeUnit::kMilli)}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ns").cast("time64[ms]").cast("int64"))
            .finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<int64_t, hdk::ir::Type::kTime>("12:03:17.001", TimeUnit::kMilli),
           dateTimeParse<int64_t, hdk::ir::Type::kTime>(
               "22:23:15.041", TimeUnit::kMilli)}));  // 22:23:15.041

  dag =
      tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[ms]").cast("int32")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({dateTimeParse<int32_t, hdk::ir::Type::kTime>(
                                             "12:03:17.000", TimeUnit::kMilli),
                                         dateTimeParse<int32_t, hdk::ir::Type::kTime>(
                                             "22:23:15.000", TimeUnit::kMilli)}));
}

TEST_F(QueryBuilderTest, TimestampToTimeNullable) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto tinfo_a = builder.scan("test_tmstmp_nullable");

  auto dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("int32")).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>({dateTimeParse<int32_t, hdk::ir::Type::kTimestamp>(
                                "1990-01-01 12:03:17", TimeUnit::kSecond),
                            dateTimeParse<int32_t, hdk::ir::Type::kTimestamp>(
                                "1950-03-18 22:23:15", TimeUnit::kSecond),
                            NULL_INT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[s]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<int64_t, hdk::ir::Type::kTime>("12:03:17", TimeUnit::kSecond),
           dateTimeParse<int64_t, hdk::ir::Type::kTime>("22:23:15", TimeUnit::kSecond),
           NULL_BIGINT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[s]").cast("int32")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>(
          {dateTimeParse<int32_t, hdk::ir::Type::kTime>("12:03:17", TimeUnit::kSecond),
           dateTimeParse<int32_t, hdk::ir::Type::kTime>("22:23:15", TimeUnit::kSecond),
           NULL_INT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ms").cast("time64[ns]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "12:03:17.123000000", TimeUnit::kNano),
                                         dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "22:23:15.456000000", TimeUnit::kNano),
                                         NULL_BIGINT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ms").cast("time64[ns]").cast("int64"))
            .finalize();
  res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "12:03:17.123000000", TimeUnit::kNano),
                                         dateTimeParse<int64_t, hdk::ir::Type::kTime>(
                                             "22:23:15.456000000", TimeUnit::kNano),
                                         NULL_BIGINT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp_ns").cast("time64[ms]").cast("int64"))
            .finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<int64_t, hdk::ir::Type::kTime>("12:03:17.001", TimeUnit::kMilli),
           dateTimeParse<int64_t, hdk::ir::Type::kTime>(
               "22:23:15.041", TimeUnit::kMilli),  // 22:23:15.041
           NULL_BIGINT}));

  dag = tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[ms]")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>(
          {dateTimeParse<int64_t, hdk::ir::Type::kTime>("12:03:17.000", TimeUnit::kMilli),
           dateTimeParse<int64_t, hdk::ir::Type::kTime>("22:23:15.000", TimeUnit::kMilli),
           NULL_BIGINT}));

  dag =
      tinfo_a.proj(tinfo_a.ref("col_tmstp").cast("time32[ms]").cast("int32")).finalize();
  res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>(
          {dateTimeParse<int32_t, hdk::ir::Type::kTime>("12:03:17.000", TimeUnit::kMilli),
           dateTimeParse<int32_t, hdk::ir::Type::kTime>("22:23:15.000", TimeUnit::kMilli),
           NULL_INT}));
}

TEST_F(QueryBuilderTest, Arithmetics2) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto tinfo_test2 = builder.scan("test2");

  auto dag = tinfo_test2.proj(tinfo_test2.ref("val1").sub(1)).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res, std::vector<int32_t>({9, 10, 11, 12, 13, 14, NULL_INT, 16, NULL_INT, 18}));
}

TEST_F(QueryBuilderTest, ScanErrors) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  EXPECT_THROW(builder.scan(TEST_DB_ID2, "test1"), hdk::ir::InvalidQueryError);
  EXPECT_THROW(builder.scan(TEST_DB_ID, "unknown"), InvalidQueryError);
  EXPECT_THROW(builder.scan(TEST_DB_ID2, "unknown"), InvalidQueryError);
  EXPECT_THROW(builder.scan("ambiguous"), InvalidQueryError);
  EXPECT_NO_THROW(builder.scan(TEST_DB_ID, "ambiguous"));
  EXPECT_NO_THROW(builder.scan(TEST_DB_ID2, "ambiguous"));
  EXPECT_THROW(builder.scan(TEST_DB_ID, -1), InvalidQueryError);
  EXPECT_THROW(builder.scan(TableRef{TEST_DB_ID, -1}), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ScanRef) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test1");
  checkRef(scan.ref("col_bi"), scan.node(), 0, "col_bi");
  checkRef(scan.ref("col_i"), scan.node(), 1, "col_i");
  checkRef(scan.ref(0), scan.node(), 0, "col_bi");
  checkRef(scan.ref(2), scan.node(), 2, "col_f");
  checkRef(scan.ref(-1), scan.node(), 4, "rowid");
  checkRef(scan.ref(-4), scan.node(), 1, "col_i");
  checkRef(scan["col_i"], scan.node(), 1, "col_i");
  checkRef(scan[2], scan.node(), 2, "col_f");
  checkRef(scan[-1], scan.node(), 4, "rowid");
  auto refs1 = scan.ref({2, 0, -2});
  checkRef(refs1[0], scan.node(), 2, "col_f");
  checkRef(refs1[1], scan.node(), 0, "col_bi");
  checkRef(refs1[2], scan.node(), 3, "col_d");
  auto refs2 = scan.ref({"col_f", "col_bi", "col_d"});
  checkRef(refs2[0], scan.node(), 2, "col_f");
  checkRef(refs2[1], scan.node(), 0, "col_bi");
  checkRef(refs2[2], scan.node(), 3, "col_d");
  EXPECT_THROW(scan.ref(10), InvalidQueryError);
  EXPECT_THROW(scan.ref(-10), InvalidQueryError);
  EXPECT_THROW(scan.ref("unknown"), InvalidQueryError);
  EXPECT_THROW(scan.ref({0, 10}), InvalidQueryError);
  EXPECT_THROW(scan.ref({"col_bi", "unknown"}), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ProjectRef) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto proj = builder.scan("test1").proj({0, 1, 2, 3, 4});
  checkRef(proj.ref("col_bi"), proj.node(), 0, "col_bi");
  checkRef(proj.ref("col_i"), proj.node(), 1, "col_i");
  checkRef(proj.ref(0), proj.node(), 0, "col_bi");
  checkRef(proj.ref(2), proj.node(), 2, "col_f");
  checkRef(proj.ref(-1), proj.node(), 4, "rowid");
  checkRef(proj.ref(-4), proj.node(), 1, "col_i");
  auto refs1 = proj.ref({2, 0, -2});
  checkRef(refs1[0], proj.node(), 2, "col_f");
  checkRef(refs1[1], proj.node(), 0, "col_bi");
  checkRef(refs1[2], proj.node(), 3, "col_d");
  auto refs2 = proj.ref({"col_f", "col_bi", "col_d"});
  checkRef(refs2[0], proj.node(), 2, "col_f");
  checkRef(refs2[1], proj.node(), 0, "col_bi");
  checkRef(refs2[2], proj.node(), 3, "col_d");
  EXPECT_THROW(proj.ref(10), InvalidQueryError);
  EXPECT_THROW(proj.ref(-10), InvalidQueryError);
  EXPECT_THROW(proj.ref("unknown"), InvalidQueryError);
  EXPECT_THROW(proj.ref({0, 10}), InvalidQueryError);
  EXPECT_THROW(proj.ref({"col_bi", "unknown"}), InvalidQueryError);
}

TEST_F(QueryBuilderTest, AggExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto ref_i32 = scan.ref("col_i");
  auto ref_i64 = scan.ref("col_bi");
  auto ref_f32 = scan.ref("col_f");
  auto ref_f64 = scan.ref("col_d");
  auto ref_dec = scan.ref("col_dec");
  auto ref_b = scan.ref("col_b");
  auto ref_str = scan.ref("col_str");
  auto ref_dict = scan.ref("col_dict");
  auto ref_date = scan.ref("col_date");
  auto ref_time = scan.ref("col_time");
  auto ref_timestamp = scan.ref("col_timestamp");
  auto ref_arr = scan.ref("col_arr_i32");
  auto ref_arr_3 = scan.ref("col_arr_i32x3");
  // COUNT
  checkAgg(scan.count(), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(scan.count(0), ctx().int32(false), AggType::kCount, false, "col_bi_count");
  checkAgg(scan.count(0, true),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_bi_count_dist");
  checkAgg(
      scan.count("col_i"), ctx().int32(false), AggType::kCount, false, "col_i_count");
  checkAgg(scan.count("col_i", true),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_i_count_dist");
  checkAgg(scan.count(scan.ref(0)),
           ctx().int32(false),
           AggType::kCount,
           false,
           "col_bi_count");
  checkAgg(scan.count(scan.ref(0), true),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_bi_count_dist");
  checkAgg(ref_i32.count(), ctx().int32(false), AggType::kCount, false, "col_i_count");
  checkAgg(ref_i64.count(true),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_bi_count_dist");
  checkAgg(ref_f32.count(), ctx().int32(false), AggType::kCount, false, "col_f_count");
  checkAgg(
      ref_f64.count(true), ctx().int32(false), AggType::kCount, true, "col_d_count_dist");
  checkAgg(ref_dec.agg(AggType::kCount),
           ctx().int32(false),
           AggType::kCount,
           false,
           "col_dec_count");
  checkAgg(ref_b.agg(AggType::kCount, true),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_b_count_dist");
  checkAgg(
      ref_str.agg("count"), ctx().int32(false), AggType::kCount, false, "col_str_count");
  checkAgg(ref_dict.agg("count_dist"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_dict_count_dist");
  checkAgg(ref_date.agg("count_distinct"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_date_count_dist");
  checkAgg(ref_time.agg("count dist"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_time_count_dist");
  checkAgg(ref_timestamp.agg("count distinct"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_timestamp_count_dist");
  // AVG
  checkAgg(ref_i32.avg(), ctx().fp64(), AggType::kAvg, false, "col_i_avg");
  checkAgg(ref_i64.avg(), ctx().fp64(), AggType::kAvg, false, "col_bi_avg");
  checkAgg(ref_f32.avg(), ctx().fp64(), AggType::kAvg, false, "col_f_avg");
  checkAgg(ref_f64.agg("AVG"), ctx().fp64(), AggType::kAvg, false, "col_d_avg");
  checkAgg(ref_dec.agg("avg"), ctx().fp64(), AggType::kAvg, false, "col_dec_avg");
  EXPECT_THROW(ref_b.avg(), InvalidQueryError);
  EXPECT_THROW(ref_str.avg(), InvalidQueryError);
  EXPECT_THROW(ref_dict.avg(), InvalidQueryError);
  EXPECT_THROW(ref_date.avg(), InvalidQueryError);
  EXPECT_THROW(ref_time.avg(), InvalidQueryError);
  EXPECT_THROW(ref_timestamp.avg(), InvalidQueryError);
  EXPECT_THROW(ref_arr.avg(), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.avg(), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("average"), InvalidQueryError);
  // MIN
  checkAgg(ref_i32.min(), ctx().int32(), AggType::kMin, false, "col_i_min");
  checkAgg(ref_i64.min(), ctx().int64(), AggType::kMin, false, "col_bi_min");
  checkAgg(ref_f32.min(), ctx().fp32(), AggType::kMin, false, "col_f_min");
  checkAgg(ref_f64.agg("MIN"), ctx().fp64(), AggType::kMin, false, "col_d_min");
  checkAgg(
      ref_dec.agg("min"), ctx().decimal64(10, 2), AggType::kMin, false, "col_dec_min");
  checkAgg(
      ref_date.min(), ctx().date32(TimeUnit::kDay), AggType::kMin, false, "col_date_min");
  checkAgg(ref_time.min(),
           ctx().time64(TimeUnit::kSecond),
           AggType::kMin,
           false,
           "col_time_min");
  checkAgg(ref_timestamp.min(),
           ctx().timestamp(TimeUnit::kSecond),
           AggType::kMin,
           false,
           "col_timestamp_min");
  EXPECT_THROW(ref_b.min(), InvalidQueryError);
  EXPECT_THROW(ref_str.min(), InvalidQueryError);
  EXPECT_THROW(ref_dict.min(), InvalidQueryError);
  EXPECT_THROW(ref_arr.min(), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.min(), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("minimum"), InvalidQueryError);
  // MAX
  checkAgg(ref_i32.max(), ctx().int32(), AggType::kMax, false, "col_i_max");
  checkAgg(ref_i64.max(), ctx().int64(), AggType::kMax, false, "col_bi_max");
  checkAgg(ref_f32.max(), ctx().fp32(), AggType::kMax, false, "col_f_max");
  checkAgg(ref_f64.agg("MAX"), ctx().fp64(), AggType::kMax, false, "col_d_max");
  checkAgg(
      ref_dec.agg("max"), ctx().decimal64(10, 2), AggType::kMax, false, "col_dec_max");
  checkAgg(
      ref_date.max(), ctx().date32(TimeUnit::kDay), AggType::kMax, false, "col_date_max");
  checkAgg(ref_time.max(),
           ctx().time64(TimeUnit::kSecond),
           AggType::kMax,
           false,
           "col_time_max");
  checkAgg(ref_timestamp.max(),
           ctx().timestamp(TimeUnit::kSecond),
           AggType::kMax,
           false,
           "col_timestamp_max");
  EXPECT_THROW(ref_b.max(), InvalidQueryError);
  EXPECT_THROW(ref_str.max(), InvalidQueryError);
  EXPECT_THROW(ref_dict.max(), InvalidQueryError);
  EXPECT_THROW(ref_arr.max(), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.max(), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("maximum"), InvalidQueryError);
  // SUM
  checkAgg(ref_i32.sum(), ctx().int64(), AggType::kSum, false, "col_i_sum");
  checkAgg(ref_i64.sum(), ctx().int64(), AggType::kSum, false, "col_bi_sum");
  checkAgg(ref_f32.sum(), ctx().fp32(), AggType::kSum, false, "col_f_sum");
  checkAgg(ref_f64.agg("sum"), ctx().fp64(), AggType::kSum, false, "col_d_sum");
  checkAgg(
      ref_dec.agg("SUM"), ctx().decimal64(10, 2), AggType::kSum, false, "col_dec_sum");
  EXPECT_THROW(ref_b.sum(), InvalidQueryError);
  EXPECT_THROW(ref_str.sum(), InvalidQueryError);
  EXPECT_THROW(ref_dict.sum(), InvalidQueryError);
  EXPECT_THROW(ref_date.sum(), InvalidQueryError);
  EXPECT_THROW(ref_time.sum(), InvalidQueryError);
  EXPECT_THROW(ref_timestamp.sum(), InvalidQueryError);
  EXPECT_THROW(ref_arr.sum(), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.sum(), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("total"), InvalidQueryError);
  // APPROX COUNT DISTINCT
  checkAgg(ref_i32.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_i_approx_count_dist");
  checkAgg(ref_i64.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_bi_approx_count_dist");
  checkAgg(ref_f32.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_f_approx_count_dist");
  checkAgg(ref_f64.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_d_approx_count_dist");
  checkAgg(ref_dec.agg(AggType::kApproxCountDistinct),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_dec_approx_count_dist");
  checkAgg(ref_b.agg("approx_count_distinct"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_b_approx_count_dist");
  checkAgg(ref_str.agg("approx_count_dist"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_str_approx_count_dist");
  checkAgg(ref_dict.agg("approx count dist"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_dict_approx_count_dist");
  checkAgg(ref_date.agg("approx count distinct"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_date_approx_count_dist");
  checkAgg(ref_time.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_time_approx_count_dist");
  checkAgg(ref_timestamp.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_timestamp_approx_count_dist");
  checkAgg(ref_arr.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_arr_i32_approx_count_dist");
  checkAgg(ref_arr_3.approxCountDist(),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_arr_i32x3_approx_count_dist");
  EXPECT_THROW(ref_i32.agg("approximate count disctinct"), InvalidQueryError);
  // APPROX QUANTILE
  checkAgg(ref_i32.approxQuantile(0.0),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_i_approx_quantile",
           0.0);
  checkAgg(ref_i64.approxQuantile(0.5),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_bi_approx_quantile",
           0.5);
  checkAgg(ref_f32.approxQuantile(1.0),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_f_approx_quantile",
           1.0);
  checkAgg(ref_f64.agg("approx quantile", 0.4),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_d_approx_quantile",
           0.4);
  checkAgg(ref_dec.agg("approx_quantile", 0.4),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_dec_approx_quantile",
           0.4);
  EXPECT_THROW(ref_b.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_str.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_dict.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_date.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_time.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_timestamp.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_arr.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.approxQuantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("approximate quantile"), InvalidQueryError);
  EXPECT_THROW(ref_i32.approxQuantile(-1.0), InvalidQueryError);
  EXPECT_THROW(ref_i32.approxQuantile(1.5), InvalidQueryError);
  // QUANTILE
  checkAgg(ref_i32.quantile(0.0),
           ctx().fp64(),
           AggType::kQuantile,
           false,
           "col_i_quantile",
           0.0,
           Interpolation::kLinear);
  checkAgg(ref_i32.quantile(0.5, Interpolation::kLower),
           ctx().int32(),
           AggType::kQuantile,
           false,
           "col_i_quantile",
           0.5,
           Interpolation::kLower);
  checkAgg(ref_i64.quantile(0.5, Interpolation::kHigher),
           ctx().int64(),
           AggType::kQuantile,
           false,
           "col_bi_quantile",
           0.5,
           Interpolation::kHigher);
  checkAgg(ref_i64.quantile(0.5, Interpolation::kNearest),
           ctx().int64(),
           AggType::kQuantile,
           false,
           "col_bi_quantile",
           0.5,
           Interpolation::kNearest);
  checkAgg(ref_i64.quantile(0.5, Interpolation::kMidpoint),
           ctx().fp64(),
           AggType::kQuantile,
           false,
           "col_bi_quantile",
           0.5,
           Interpolation::kMidpoint);
  for (auto interpolation : {Interpolation::kLower,
                             Interpolation::kHigher,
                             Interpolation::kNearest,
                             Interpolation::kMidpoint,
                             Interpolation::kLinear}) {
    checkAgg(ref_date.quantile(0.5, interpolation),
             ctx().date64(TimeUnit::kSecond),
             AggType::kQuantile,
             false,
             "col_date_quantile",
             0.5,
             interpolation);
    checkAgg(ref_time.quantile(0.5, interpolation),
             ctx().time64(TimeUnit::kSecond),
             AggType::kQuantile,
             false,
             "col_time_quantile",
             0.5,
             interpolation);
    checkAgg(ref_timestamp.quantile(0.5, interpolation),
             ctx().timestamp(TimeUnit::kSecond),
             AggType::kQuantile,
             false,
             "col_timestamp_quantile",
             0.5,
             interpolation);
  }
  EXPECT_THROW(ref_b.quantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_str.quantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_dict.quantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_arr.quantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.quantile(0.5), InvalidQueryError);
  EXPECT_THROW(ref_i32.agg("quantile"), InvalidQueryError);
  EXPECT_THROW(ref_i32.quantile(-1.0), InvalidQueryError);
  EXPECT_THROW(ref_i32.quantile(1.5), InvalidQueryError);
  // SAMPLE
  checkAgg(
      ref_i32.sample(), ref_i32.expr()->type(), AggType::kSample, false, "col_i_sample");
  checkAgg(
      ref_i64.sample(), ref_i64.expr()->type(), AggType::kSample, false, "col_bi_sample");
  checkAgg(
      ref_f32.sample(), ref_f32.expr()->type(), AggType::kSample, false, "col_f_sample");
  checkAgg(
      ref_f64.sample(), ref_f64.expr()->type(), AggType::kSample, false, "col_d_sample");
  checkAgg(ref_b.sample(), ref_b.expr()->type(), AggType::kSample, false, "col_b_sample");
  checkAgg(ref_dec.sample(),
           ref_dec.expr()->type(),
           AggType::kSample,
           false,
           "col_dec_sample");
  checkAgg(ref_str.sample(),
           ref_str.expr()->type(),
           AggType::kSample,
           false,
           "col_str_sample");
  checkAgg(ref_dict.sample(),
           ref_dict.expr()->type(),
           AggType::kSample,
           false,
           "col_dict_sample");
  checkAgg(ref_time.sample(),
           ref_time.expr()->type(),
           AggType::kSample,
           false,
           "col_time_sample");
  checkAgg(ref_date.sample(),
           ref_date.expr()->type(),
           AggType::kSample,
           false,
           "col_date_sample");
  checkAgg(ref_timestamp.sample(),
           ref_timestamp.expr()->type(),
           AggType::kSample,
           false,
           "col_timestamp_sample");
  checkAgg(ref_arr.sample(),
           ref_arr.expr()->type(),
           AggType::kSample,
           false,
           "col_arr_i32_sample");
  checkAgg(ref_arr_3.sample(),
           ref_arr_3.expr()->type(),
           AggType::kSample,
           false,
           "col_arr_i32x3_sample");
  // SINGLE VALUE
  checkAgg(ref_i32.singleValue(),
           ref_i32.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_i_single_value");
  checkAgg(ref_i64.singleValue(),
           ref_i64.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_bi_single_value");
  checkAgg(ref_f32.singleValue(),
           ref_f32.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_f_single_value");
  checkAgg(ref_f64.singleValue(),
           ref_f64.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_d_single_value");
  checkAgg(ref_b.singleValue(),
           ref_b.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_b_single_value");
  checkAgg(ref_dec.singleValue(),
           ref_dec.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_dec_single_value");
  checkAgg(ref_dict.singleValue(),
           ref_dict.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_dict_single_value");
  checkAgg(ref_time.singleValue(),
           ref_time.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_time_single_value");
  checkAgg(ref_date.singleValue(),
           ref_date.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_date_single_value");
  checkAgg(ref_timestamp.singleValue(),
           ref_timestamp.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_timestamp_single_value");
  checkAgg(ref_arr_3.singleValue(),
           ref_arr_3.expr()->type(),
           AggType::kSingleValue,
           false,
           "col_arr_i32x3_single_value");
  EXPECT_THROW(ref_str.singleValue(), InvalidQueryError);
  EXPECT_THROW(ref_arr.singleValue(), InvalidQueryError);
  // TOP_K
  checkAgg(ref_i32.topK(2),
           ctx().arrayVarLen(ref_i32.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_i_top_2",
           2);
  checkAgg(ref_i64.topK(1),
           ctx().arrayVarLen(ref_i64.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_bi_top_1",
           1);
  checkAgg(ref_f32.topK(2),
           ctx().arrayVarLen(ref_f32.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_f_top_2");
  checkAgg(ref_f64.topK(3),
           ctx().arrayVarLen(ref_f64.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_d_top_3");
  checkAgg(ref_dec.topK(2),
           ctx().arrayVarLen(ref_dec.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_dec_top_2");
  checkAgg(ref_time.topK(-5),
           ctx().arrayVarLen(ref_time.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_time_bottom_5");
  checkAgg(ref_date.topK(6),
           ctx().arrayVarLen(ref_date.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_date_top_6");
  checkAgg(ref_timestamp.topK(3),
           ctx().arrayVarLen(ref_timestamp.expr()->type(), 4, false),
           AggType::kTopK,
           false,
           "col_timestamp_top_3");
  EXPECT_THROW(ref_i32.topK(0), InvalidQueryError);
  EXPECT_THROW(ref_b.topK(2), InvalidQueryError);
  EXPECT_THROW(ref_dict.topK(2), InvalidQueryError);
  EXPECT_THROW(ref_str.topK(2), InvalidQueryError);
  EXPECT_THROW(ref_arr.topK(2), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.topK(2), InvalidQueryError);
  // STDDEV_SAMP
  checkAgg(ref_i32.stdDev(), ctx().fp64(), AggType::kStdDevSamp, false, "col_i_stddev");
  checkAgg(ref_i64.stdDev(), ctx().fp64(), AggType::kStdDevSamp, false, "col_bi_stddev");
  checkAgg(ref_f32.stdDev(), ctx().fp64(), AggType::kStdDevSamp, false, "col_f_stddev");
  checkAgg(ref_f64.stdDev(), ctx().fp64(), AggType::kStdDevSamp, false, "col_d_stddev");
  checkAgg(ref_dec.stdDev(), ctx().fp64(), AggType::kStdDevSamp, false, "col_dec_stddev");
  EXPECT_THROW(ref_b.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_str.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_dict.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_date.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_time.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_timestamp.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_arr.stdDev(), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.stdDev(), InvalidQueryError);
  // CORR
  checkAgg(
      ref_i32.corr(ref_i64), ctx().fp64(), AggType::kCorr, false, "col_i_corr_col_bi");
  checkAgg(
      ref_f32.corr(ref_f64), ctx().fp64(), AggType::kCorr, false, "col_f_corr_col_d");
  checkAgg(
      ref_dec.corr(ref_f64), ctx().fp64(), AggType::kCorr, false, "col_dec_corr_col_d");
  EXPECT_THROW(ref_b.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_str.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_dict.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_date.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_time.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_timestamp.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_arr.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_arr_3.corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(builder.cst(4.5).corr(ref_f64), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_b), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_str), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_dict), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_date), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_time), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_timestamp), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_arr), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(ref_arr_3), InvalidQueryError);
  EXPECT_THROW(ref_f64.corr(builder.cst(3.5)), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ParseAgg) {
  class TestNode : public BuilderNode {
   public:
    TestNode(const BuilderNode& node) : BuilderNode(node) {}

    BuilderExpr parseAgg(const std::string& agg_str) { return parseAggString(agg_str); }
  };

  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  TestNode node(builder.scan("test3"));
  // COUNT
  checkAgg(node.parseAgg("count"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(
      node.parseAgg(" count() "), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(
      node.parseAgg("count(1)"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(node.parseAgg(" count ( 1 ) "),
           ctx().int32(false),
           AggType::kCount,
           false,
           "count");
  checkAgg(
      node.parseAgg("count(*)"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(
      node.parseAgg("count( *)"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(node.parseAgg(" COUNT"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(
      node.parseAgg("CoUnT (1)"), ctx().int32(false), AggType::kCount, false, "count");
  checkAgg(node.parseAgg("count(col_bi)"),
           ctx().int32(false),
           AggType::kCount,
           false,
           "col_bi_count");
  checkAgg(node.parseAgg(" count ( col_i ) "),
           ctx().int32(false),
           AggType::kCount,
           false,
           "col_i_count");
  checkAgg(node.parseAgg("count_dist(col_i)"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_i_count_dist");
  checkAgg(node.parseAgg("count dist (col_i)"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_i_count_dist");
  checkAgg(node.parseAgg("count_distinct( col_i )"),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_i_count_dist");
  checkAgg(node.parseAgg(" count distinct (col_i ) "),
           ctx().int32(false),
           AggType::kCount,
           true,
           "col_i_count_dist");
  EXPECT_THROW(node.parseAgg("count(col_bi, 0.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("count(col_bi, col_i)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("count(2)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("count("), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("cnt"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("dist_count"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("distinct count"), InvalidQueryError);
  // AVG
  checkAgg(node.parseAgg("avg(col_i)"), ctx().fp64(), AggType::kAvg, false, "col_i_avg");
  checkAgg(
      node.parseAgg(" aVg ( col_i)"), ctx().fp64(), AggType::kAvg, false, "col_i_avg");
  EXPECT_THROW(node.parseAgg("avg"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("avg(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("avg(col_i, 0.5)"), InvalidQueryError);
  // MIN
  checkAgg(node.parseAgg("min(col_i)"), ctx().int32(), AggType::kMin, false, "col_i_min");
  checkAgg(
      node.parseAgg(" MiN ( col_i)"), ctx().int32(), AggType::kMin, false, "col_i_min");
  EXPECT_THROW(node.parseAgg("min"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("min(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("min(col_i, 0.5)"), InvalidQueryError);
  // MAX
  checkAgg(node.parseAgg("max(col_i)"), ctx().int32(), AggType::kMax, false, "col_i_max");
  checkAgg(
      node.parseAgg(" mAx ( col_i)"), ctx().int32(), AggType::kMax, false, "col_i_max");
  EXPECT_THROW(node.parseAgg("max"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("max(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("max(col_i, 0.5)"), InvalidQueryError);
  // SUM
  checkAgg(node.parseAgg("sum(col_i)"), ctx().int64(), AggType::kSum, false, "col_i_sum");
  checkAgg(
      node.parseAgg(" SuM ( col_i)"), ctx().int64(), AggType::kSum, false, "col_i_sum");
  EXPECT_THROW(node.parseAgg("sum"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("sum(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("sum(col_i, 0.5)"), InvalidQueryError);
  // APPROX COUNT
  checkAgg(node.parseAgg("approx_count_dist(col_i)"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_i_approx_count_dist");
  checkAgg(node.parseAgg("approx COUNT dist (col_i)"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_i_approx_count_dist");
  checkAgg(node.parseAgg("APPROX_count_distinct( col_i )"),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_i_approx_count_dist");
  checkAgg(node.parseAgg(" approx count DISTINCT (col_i ) "),
           ctx().int32(false),
           AggType::kApproxCountDistinct,
           true,
           "col_i_approx_count_dist");
  EXPECT_THROW(node.parseAgg("approx_count_dist(col_bi, 0.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_count dist(col_bi)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_count_dist"), InvalidQueryError);
  // APPROX QUANTILE
  checkAgg(node.parseAgg("approx_quantile(col_i, 0.0)"),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_i_approx_quantile",
           0.0);
  checkAgg(node.parseAgg(" approx quanTILE ( col_i  ,  0.5 ) "),
           ctx().fp64(),
           AggType::kApproxQuantile,
           false,
           "col_i_approx_quantile",
           0.5);
  EXPECT_THROW(node.parseAgg("approx_quantile"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, col_f)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, -0.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 1.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 1..5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 0.5, 1.5)"), InvalidQueryError);
  // QUANTILE
  checkAgg(node.parseAgg("quantile(col_i, 0.5)"),
           ctx().fp64(),
           AggType::kQuantile,
           false,
           "col_i_quantile",
           0.5,
           Interpolation::kLinear);
  checkAgg(node.parseAgg("  QUANtile (col_i, 0.1)"),
           ctx().fp64(),
           AggType::kQuantile,
           false,
           "col_i_quantile",
           0.1,
           Interpolation::kLinear);
  EXPECT_THROW(node.parseAgg("quantile"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i, col_f)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i, -0.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i, 1.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i, 1..5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("quantile(col_i, 0.5, 1.5)"), InvalidQueryError);
  // SAMPLE
  checkAgg(node.parseAgg("sample(col_i)"),
           ctx().int32(),
           AggType::kSample,
           false,
           "col_i_sample");
  checkAgg(node.parseAgg(" SaMpLe ( col_i)"),
           ctx().int32(),
           AggType::kSample,
           false,
           "col_i_sample");
  EXPECT_THROW(node.parseAgg("sample"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("sample(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("sample(col_i, 0.5)"), InvalidQueryError);
  // SINGLE VALUE
  checkAgg(node.parseAgg("single_value(col_i)"),
           ctx().int32(),
           AggType::kSingleValue,
           false,
           "col_i_single_value");
  checkAgg(node.parseAgg(" SiNgLe value ( col_i)"),
           ctx().int32(),
           AggType::kSingleValue,
           false,
           "col_i_single_value");
  EXPECT_THROW(node.parseAgg("single_value"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("single_value(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("single_value(col_i, 0.5)"), InvalidQueryError);
  // TOP_K
  checkAgg(node.parseAgg("topk(col_i, 2)"),
           ctx().arrayVarLen(ctx().int32(), 4, false),
           AggType::kTopK,
           false,
           "col_i_top_2",
           2);
  checkAgg(node.parseAgg(" top_K ( col_i, -4)"),
           ctx().arrayVarLen(ctx().int32(), 4, false),
           AggType::kTopK,
           false,
           "col_i_bottom_4",
           -4);
  EXPECT_THROW(node.parseAgg("topk"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("topk(col_i)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("topk(col_i, 0.5)"), InvalidQueryError);
  // BOTTOM_K
  checkAgg(node.parseAgg("bottomk(col_i, 2)"),
           ctx().arrayVarLen(ctx().int32(), 4, false),
           AggType::kTopK,
           false,
           "col_i_bottom_2",
           -2);
  checkAgg(node.parseAgg(" bottom_K ( col_i, -4)"),
           ctx().arrayVarLen(ctx().int32(), 4, false),
           AggType::kTopK,
           false,
           "col_i_top_4",
           4);
  EXPECT_THROW(node.parseAgg("bottomk"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("bottomk(col_i)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("bottomk(col_i, 0.5)"), InvalidQueryError);
  // STDDEV_SAMP
  checkAgg(node.parseAgg("stddev(col_i)"),
           ctx().fp64(),
           AggType::kStdDevSamp,
           false,
           "col_i_stddev");
  checkAgg(node.parseAgg("stddev SAMP( col_f)"),
           ctx().fp64(),
           AggType::kStdDevSamp,
           false,
           "col_f_stddev");
  checkAgg(node.parseAgg("stddev_samp( col_d)"),
           ctx().fp64(),
           AggType::kStdDevSamp,
           false,
           "col_d_stddev");
  EXPECT_THROW(node.parseAgg("stddev"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("stddev(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("stddev(col_i, 1)"), InvalidQueryError);
  // CORR
  checkAgg(node.parseAgg("corr(col_i, col_bi)"),
           ctx().fp64(),
           AggType::kCorr,
           false,
           "col_i_corr_col_bi");
  checkAgg(node.parseAgg("corr(col_d, col_f)"),
           ctx().fp64(),
           AggType::kCorr,
           false,
           "col_d_corr_col_f");
  EXPECT_THROW(node.parseAgg("corr"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("corr(1)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("corr(col_f)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("corr(col_f, 1)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ExtractExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kYear),
               DateExtractField::kYear);
  checkExtract(scan.ref("col_timestamp").extract("year"), DateExtractField::kYear);
  checkExtract(scan.ref("col_timestamp").extract(" YEAR "), DateExtractField::kYear);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kQuarter),
               DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract("quarter"), DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract(" QUARTER "),
               DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kQuarter),
               DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract("quarter"), DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract(" QUARTER "),
               DateExtractField::kQuarter);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kMonth),
               DateExtractField::kMonth);
  checkExtract(scan.ref("col_timestamp").extract("month"), DateExtractField::kMonth);
  checkExtract(scan.ref("col_timestamp").extract(" MONTH "), DateExtractField::kMonth);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kDay),
               DateExtractField::kDay);
  checkExtract(scan.ref("col_timestamp").extract("day"), DateExtractField::kDay);
  checkExtract(scan.ref("col_timestamp").extract(" DAY "), DateExtractField::kDay);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kHour),
               DateExtractField::kHour);
  checkExtract(scan.ref("col_timestamp").extract("hour"), DateExtractField::kHour);
  checkExtract(scan.ref("col_timestamp").extract(" HOUR "), DateExtractField::kHour);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kMinute),
               DateExtractField::kMinute);
  checkExtract(scan.ref("col_timestamp").extract("min"), DateExtractField::kMinute);
  checkExtract(scan.ref("col_timestamp").extract(" MINUTE "), DateExtractField::kMinute);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kSecond),
               DateExtractField::kSecond);
  checkExtract(scan.ref("col_timestamp").extract("sec"), DateExtractField::kSecond);
  checkExtract(scan.ref("col_timestamp").extract(" SECOND "), DateExtractField::kSecond);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kMilli),
               DateExtractField::kMilli);
  checkExtract(scan.ref("col_timestamp").extract("milli"), DateExtractField::kMilli);
  checkExtract(scan.ref("col_timestamp").extract(" MILLISECOND "),
               DateExtractField::kMilli);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kMicro),
               DateExtractField::kMicro);
  checkExtract(scan.ref("col_timestamp").extract("micro"), DateExtractField::kMicro);
  checkExtract(scan.ref("col_timestamp").extract(" MICROSECOND "),
               DateExtractField::kMicro);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kNano),
               DateExtractField::kNano);
  checkExtract(scan.ref("col_timestamp").extract("nano"), DateExtractField::kNano);
  checkExtract(scan.ref("col_timestamp").extract(" NANOSECOND "),
               DateExtractField::kNano);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kDayOfWeek),
               DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract("dow"), DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(" dayOfWeek"),
               DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract("Day_Of_Week"),
               DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(" day OF week "),
               DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kIsoDayOfWeek),
               DateExtractField::kIsoDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract("isodow"),
               DateExtractField::kIsoDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(" IsoDayOfWeek"),
               DateExtractField::kIsoDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract("ISO_Day_Of_Week"),
               DateExtractField::kIsoDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(" ISO day OF week "),
               DateExtractField::kIsoDayOfWeek);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kDayOfYear),
               DateExtractField::kDayOfYear);
  checkExtract(scan.ref("col_timestamp").extract("doy"), DateExtractField::kDayOfYear);
  checkExtract(scan.ref("col_timestamp").extract(" dayOfYear"),
               DateExtractField::kDayOfYear);
  checkExtract(scan.ref("col_timestamp").extract("Day_Of_Year"),
               DateExtractField::kDayOfYear);
  checkExtract(scan.ref("col_timestamp").extract(" day OF year "),
               DateExtractField::kDayOfYear);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kEpoch),
               DateExtractField::kEpoch);
  checkExtract(scan.ref("col_timestamp").extract("epoch"), DateExtractField::kEpoch);
  checkExtract(scan.ref("col_timestamp").extract(" EPOCH "), DateExtractField::kEpoch);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kQuarterDay),
               DateExtractField::kQuarterDay);
  checkExtract(scan.ref("col_timestamp").extract("quarterday"),
               DateExtractField::kQuarterDay);
  checkExtract(scan.ref("col_timestamp").extract(" Quarter_Day "),
               DateExtractField::kQuarterDay);
  checkExtract(scan.ref("col_timestamp").extract(" quarter DAY "),
               DateExtractField::kQuarterDay);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kWeek),
               DateExtractField::kWeek);
  checkExtract(scan.ref("col_timestamp").extract("week"), DateExtractField::kWeek);
  checkExtract(scan.ref("col_timestamp").extract(" WEEK "), DateExtractField::kWeek);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kWeekSunday),
               DateExtractField::kWeekSunday);
  checkExtract(scan.ref("col_timestamp").extract("weeksunday"),
               DateExtractField::kWeekSunday);
  checkExtract(scan.ref("col_timestamp").extract(" WEEK SUNDAY "),
               DateExtractField::kWeekSunday);
  checkExtract(scan.ref("col_timestamp").extract(" Week_Sunday "),
               DateExtractField::kWeekSunday);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kWeekSaturday),
               DateExtractField::kWeekSaturday);
  checkExtract(scan.ref("col_timestamp").extract("weeksaturday"),
               DateExtractField::kWeekSaturday);
  checkExtract(scan.ref("col_timestamp").extract(" WEEK SATURDAY "),
               DateExtractField::kWeekSaturday);
  checkExtract(scan.ref("col_timestamp").extract(" Week_Saturday "),
               DateExtractField::kWeekSaturday);
  checkExtract(scan.ref("col_timestamp").extract(DateExtractField::kDateEpoch),
               DateExtractField::kDateEpoch);
  checkExtract(scan.ref("col_timestamp").extract("dateepoch"),
               DateExtractField::kDateEpoch);
  checkExtract(scan.ref("col_timestamp").extract(" Date Epoch "),
               DateExtractField::kDateEpoch);
  checkExtract(scan.ref("col_date").extract("year"), DateExtractField::kYear, true);
  checkExtract(scan.ref("col_date").extract("sec"), DateExtractField::kSecond, true);
  checkExtract(scan.ref("col_date").extract("dow"), DateExtractField::kDayOfWeek, true);
  checkExtract(scan.ref("col_date").extract("micro"), DateExtractField::kMicro, true);
  checkExtract(scan.ref("col_time").extract("year"), DateExtractField::kYear);
  checkExtract(scan.ref("col_time").extract("sec"), DateExtractField::kSecond);
  checkExtract(scan.ref("col_time").extract("dow"), DateExtractField::kDayOfWeek);
  checkExtract(scan.ref("col_time").extract("micro"), DateExtractField::kMicro);

  EXPECT_THROW(scan.ref("col_bi").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32x3").extract("day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").extract("milli second"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").extract(""), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").extract("day").extract("year"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ParseType) {
  // NULLT
  ASSERT_TRUE(ctx().typeFromString("nullt")->isNull());
  ASSERT_TRUE(ctx().typeFromString("NULLT ")->isNull());
  EXPECT_THROW(ctx().typeFromString("NULLT32"), TypeError);
  EXPECT_THROW(ctx().typeFromString("NULLT[nn]"), TypeError);
  // BOOLEAN
  ASSERT_TRUE(ctx().typeFromString("bool")->equal(ctx().boolean(true)));
  ASSERT_TRUE(ctx().typeFromString("bool[nn]")->equal(ctx().boolean(false)));
  EXPECT_THROW(ctx().typeFromString("bool32"), TypeError);
  // INTEGER
  ASSERT_TRUE(ctx().typeFromString("int")->equal(ctx().int64(true)));
  ASSERT_TRUE(ctx().typeFromString("int[nn]")->equal(ctx().int64(false)));
  ASSERT_TRUE(ctx().typeFromString("int8")->equal(ctx().int8(true)));
  ASSERT_TRUE(ctx().typeFromString("int16[nn]")->equal(ctx().int16(false)));
  ASSERT_TRUE(ctx().typeFromString("int32")->equal(ctx().int32(true)));
  ASSERT_TRUE(ctx().typeFromString("INT64[NN]")->equal(ctx().int64(false)));
  EXPECT_THROW(ctx().typeFromString("int128"), TypeError);
  EXPECT_THROW(ctx().typeFromString("int15"), TypeError);
  EXPECT_THROW(ctx().typeFromString("int 32"), TypeError);
  EXPECT_THROW(ctx().typeFromString("integer"), TypeError);
  // FP
  ASSERT_TRUE(ctx().typeFromString("fp")->equal(ctx().fp64(true)));
  ASSERT_TRUE(ctx().typeFromString("fp[nn]")->equal(ctx().fp64(false)));
  ASSERT_TRUE(ctx().typeFromString("fp32")->equal(ctx().fp32(true)));
  ASSERT_TRUE(ctx().typeFromString("fp64[NN]")->equal(ctx().fp64(false)));
  EXPECT_THROW(ctx().typeFromString("fp16"), TypeError);
  EXPECT_THROW(ctx().typeFromString("fp80"), TypeError);
  EXPECT_THROW(ctx().typeFromString("fp128"), TypeError);
  EXPECT_THROW(ctx().typeFromString("fp 32"), TypeError);
  // DECIMAL
  ASSERT_TRUE(ctx().typeFromString("dec(10,2)")->equal(ctx().decimal64(10, 2, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("DECIMAL( 10 , 2 )[NN]")
                  ->equal(ctx().decimal64(10, 2, false)));
  ASSERT_TRUE(ctx().typeFromString("dec64(06,2)")->equal(ctx().decimal64(6, 2, true)));
  ASSERT_TRUE(
      ctx().typeFromString("decimal64(11,2)[nn]")->equal(ctx().decimal64(11, 2, false)));
  EXPECT_THROW(ctx().typeFromString("decimal"), TypeError);
  EXPECT_THROW(ctx().typeFromString("number(10,2)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("decimal128(10,2)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("dec(-10,2)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("dec(10,-2)"), TypeError);
  // VARCHAR
  ASSERT_TRUE(ctx().typeFromString("varchar(10)")->equal(ctx().varChar(10, true)));
  ASSERT_TRUE(ctx().typeFromString("varchar(0)")->equal(ctx().varChar(0)));
  ASSERT_TRUE(
      ctx().typeFromString("VARCHAR(0200)[nn]")->equal(ctx().varChar(200, false)));
  EXPECT_THROW(ctx().typeFromString("varchar"), TypeError);
  EXPECT_THROW(ctx().typeFromString("varchar[10]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("varchar(-10)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("varchar()"), TypeError);
  // TEXT
  ASSERT_TRUE(ctx().typeFromString("text")->equal(ctx().text(true)));
  ASSERT_TRUE(ctx().typeFromString("text[nn]")->equal(ctx().text(false)));
  EXPECT_THROW(ctx().typeFromString("text32"), TypeError);
  // DATE
  ASSERT_TRUE(
      ctx().typeFromString("date")->equal(ctx().date(8, TimeUnit::kSecond, true)));
  ASSERT_TRUE(ctx().typeFromString("date16")->equal(ctx().date(2, TimeUnit::kDay, true)));
  ASSERT_TRUE(
      ctx().typeFromString("date32")->equal(ctx().date(4, TimeUnit::kSecond, true)));
  ASSERT_TRUE(
      ctx().typeFromString("date64[nn]")->equal(ctx().date(8, TimeUnit::kSecond, false)));
  ASSERT_TRUE(
      ctx().typeFromString("date32[s]")->equal(ctx().date(4, TimeUnit::kSecond, true)));
  ASSERT_TRUE(
      ctx().typeFromString("date[D][NN]")->equal(ctx().date(8, TimeUnit::kDay, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("date[s][nn]")
                  ->equal(ctx().date(8, TimeUnit::kSecond, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("date64[s][nn]")
                  ->equal(ctx().date(8, TimeUnit::kSecond, false)));
  EXPECT_THROW(ctx().typeFromString("date32[m]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("date16[s]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("date32[ms]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("date32[us]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("datdate64[ns]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("date[sec]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("date[day]"), TypeError);
  // TIME
  ASSERT_TRUE(ctx().typeFromString("time")->equal(ctx().time(8, TimeUnit::kMicro, true)));
  ASSERT_TRUE(
      ctx().typeFromString("time16")->equal(ctx().time(2, TimeUnit::kSecond, true)));
  ASSERT_TRUE(
      ctx().typeFromString("time32")->equal(ctx().time(4, TimeUnit::kMilli, true)));
  ASSERT_TRUE(
      ctx().typeFromString("time64[nn]")->equal(ctx().time(8, TimeUnit::kMicro, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("time16[s][NN]")
                  ->equal(ctx().time(2, TimeUnit::kSecond, false)));
  ASSERT_TRUE(
      ctx().typeFromString("time32[ms]")->equal(ctx().time(4, TimeUnit::kMilli, true)));
  ASSERT_TRUE(
      ctx().typeFromString("time64[ms]")->equal(ctx().time(8, TimeUnit::kMilli, true)));
  ASSERT_TRUE(
      ctx().typeFromString("time64[us]")->equal(ctx().time(8, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("time64[ns][nn]")
                  ->equal(ctx().time(8, TimeUnit::kNano, false)));
  EXPECT_THROW(ctx().typeFromString("time[m]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time[d]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time16[ms]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time16[us]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time16[ns]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time32[us]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("time32[ns]"), TypeError);
  // TIMESTAMP
  ASSERT_TRUE(
      ctx().typeFromString("timestamp")->equal(ctx().timestamp(TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp64[nn]")
                  ->equal(ctx().timestamp(TimeUnit::kMicro, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp[s]")
                  ->equal(ctx().timestamp(TimeUnit::kSecond, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp64[s]")
                  ->equal(ctx().timestamp(TimeUnit::kSecond, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp[ms]")
                  ->equal(ctx().timestamp(TimeUnit::kMilli, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp[us][NN]")
                  ->equal(ctx().timestamp(TimeUnit::kMicro, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("timestamp[ns]")
                  ->equal(ctx().timestamp(TimeUnit::kNano, true)));
  EXPECT_THROW(ctx().typeFromString("timestamp32"), TypeError);
  EXPECT_THROW(ctx().typeFromString("timestamp[m]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("timestamp[d]"), TypeError);
  // INTERVAL
  ASSERT_TRUE(
      ctx().typeFromString("interval")->equal(ctx().interval(8, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval16")
                  ->equal(ctx().interval(2, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval32")
                  ->equal(ctx().interval(4, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval64")
                  ->equal(ctx().interval(8, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval[d]")
                  ->equal(ctx().interval(8, TimeUnit::kDay, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval32[m][nn]")
                  ->equal(ctx().interval(4, TimeUnit::kMonth, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval[s]")
                  ->equal(ctx().interval(8, TimeUnit::kSecond, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval16[ms]")
                  ->equal(ctx().interval(2, TimeUnit::kMilli, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval32[us]")
                  ->equal(ctx().interval(4, TimeUnit::kMicro, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("interval32[ns][NN]")
                  ->equal(ctx().interval(4, TimeUnit::kNano, false)));
  EXPECT_THROW(ctx().typeFromString("interval8"), TypeError);
  // FIXED LENGTH ARRAY
  ASSERT_TRUE(ctx()
                  .typeFromString("array(int)(2)")
                  ->equal(ctx().arrayFixed(2, ctx().int64(), true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("array(fp64[nn])(8)[NN]")
                  ->equal(ctx().arrayFixed(8, ctx().fp64(false), false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("array(dec(10, 2)[nn])(8)[NN]")
                  ->equal(ctx().arrayFixed(8, ctx().decimal(8, 10, 2, false), false)));
  ASSERT_TRUE(
      ctx()
          .typeFromString("array(dict(text[nn])[10])(4)[nn]")
          ->equal(ctx().arrayFixed(4, ctx().extDict(ctx().text(false), 10, 4), false)));
  EXPECT_THROW(ctx().typeFromString("array()(10)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("array(array(int)(10))(10)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("array(text)(10)"), TypeError);
  // VARLEN ARRAY
  ASSERT_TRUE(ctx()
                  .typeFromString("array(int)")
                  ->equal(ctx().arrayVarLen(ctx().int64(), 4, true)));
  ASSERT_TRUE(ctx()
                  .typeFromString("array(fp64[nn])[nn]")
                  ->equal(ctx().arrayVarLen(ctx().fp64(false), 4, false)));
  ASSERT_TRUE(ctx()
                  .typeFromString("array(dec(10,2)[nn])")
                  ->equal(ctx().arrayVarLen(ctx().decimal(8, 10, 2, false), 4, true)));
  EXPECT_THROW(ctx().typeFromString("array(array(int))"), TypeError);
  EXPECT_THROW(ctx().typeFromString("array(text)"), TypeError);
  // DICT
  ASSERT_TRUE(ctx().typeFromString("dict")->equal(ctx().extDict(ctx().text(), 0, 4)));
  ASSERT_TRUE(
      ctx().typeFromString("dict8(text)")->equal(ctx().extDict(ctx().text(), 0, 1)));
  ASSERT_TRUE(ctx().typeFromString("dict16")->equal(ctx().extDict(ctx().text(), 0, 2)));
  ASSERT_TRUE(
      ctx().typeFromString("dict32[10]")->equal(ctx().extDict(ctx().text(), 10, 4)));
  ASSERT_TRUE(
      ctx().typeFromString("dict[-10]")->equal(ctx().extDict(ctx().text(), -10, 4)));
  ASSERT_TRUE(ctx()
                  .typeFromString("dict(text[nn])[10]")
                  ->equal(ctx().extDict(ctx().text(false), 10, 4)));
  EXPECT_THROW(ctx().typeFromString("dict64"), TypeError);
  EXPECT_THROW(ctx().typeFromString("dict[nn]"), TypeError);
  EXPECT_THROW(ctx().typeFromString("dict(10)"), TypeError);
  EXPECT_THROW(ctx().typeFromString("dict(int)"), TypeError);
}

TEST_F(QueryBuilderTest, CastIntegerExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkCast(scan.ref("col_bi").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_bi").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_bi").cast("int32"), ctx().int32());
  ASSERT_TRUE(scan.ref("col_bi").cast("int64").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_i").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_i").cast("int16"), ctx().int16());
  ASSERT_TRUE(scan.ref("col_i").cast("int32").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_i").cast("int64"), ctx().int64());
  checkCast(scan.ref("col_si").cast("int8"), ctx().int8());
  ASSERT_TRUE(scan.ref("col_si").cast("int16").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_si").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_si").cast("int64"), ctx().int64());
  ASSERT_TRUE(scan.ref("col_ti").cast("int8").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_ti").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_ti").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_ti").cast("int64"), ctx().int64());
  checkCast(scan.ref("col_i").cast("fp32"), ctx().fp32());
  checkCast(scan.ref("col_bi").cast("fp64"), ctx().fp64());
  checkCast(scan.ref("col_si").cast("dec(10,2)"), ctx().decimal(8, 10, 2));
  checkBinOper(scan.ref("col_ti").cast("bool"),
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_ti"),
               builder.cst(0));
  EXPECT_THROW(scan.ref("col_bi").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").cast("dict"), InvalidQueryError);
  // TODO: allow conversion of integer tiypes to time, date and intervals
  // similar to timestamps?
  EXPECT_THROW(scan.ref("col_bi").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_ti").cast("time[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").cast("date32[s]"), InvalidQueryError);
  checkCast(scan.ref("col_bi").cast("timestamp[s]"), ctx().timestamp(TimeUnit::kSecond));
  checkCast(scan.ref("col_i").cast("timestamp[ms]"), ctx().timestamp(TimeUnit::kMilli));
  checkCast(scan.ref("col_si").cast("timestamp[us]"), ctx().timestamp(TimeUnit::kMicro));
  checkCast(scan.ref("col_ti").cast("timestamp[ns]"), ctx().timestamp(TimeUnit::kNano));
  EXPECT_THROW(scan.ref("col_bi").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastFpExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkCast(scan.ref("col_f").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_f").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_f").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_f").cast("int64"), ctx().int64());
  checkCast(scan.ref("col_d").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_d").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_d").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_d").cast("int64"), ctx().int64());
  ASSERT_TRUE(scan.ref("col_f").cast("fp32").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_f").cast("fp64"), ctx().fp64());
  checkCast(scan.ref("col_d").cast("fp32"), ctx().fp32());
  ASSERT_TRUE(scan.ref("col_d").cast("fp64").expr()->is<hdk::ir::ColumnRef>());
  // TODO: support fp -> dec casts?
  EXPECT_THROW(scan.ref("col_f").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("time[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("date32[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("timestamp[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("timestamp[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("timestamp[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("timestamp[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastDecimalExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkCast(scan.ref("col_dec").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_dec").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_dec").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_dec").cast("int64"), ctx().int64());
  checkCast(scan.ref("col_dec").cast("fp32"), ctx().fp32());
  checkCast(scan.ref("col_dec").cast("fp64"), ctx().fp64());
  ASSERT_TRUE(scan.ref("col_dec").cast("dec(10,2)").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_dec").cast("dec(10,3)"), ctx().decimal(8, 10, 3));
  checkCast(scan.ref("col_dec").cast("dec(11,2)"), ctx().decimal(8, 11, 2));
  checkBinOper(scan.ref("col_dec").cast("bool"),
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_dec"),
               builder.cst(0, "dec(10,2)"));
  EXPECT_THROW(scan.ref("col_dec").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("time[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("date32[s]"), InvalidQueryError);
  checkCast(scan.ref("col_dec").cast("timestamp[s]"), ctx().timestamp(TimeUnit::kSecond));
  checkCast(scan.ref("col_dec").cast("timestamp[ms]"), ctx().timestamp(TimeUnit::kMilli));
  checkCast(scan.ref("col_dec").cast("timestamp[us]"), ctx().timestamp(TimeUnit::kMicro));
  checkCast(scan.ref("col_dec").cast("timestamp[ns]"), ctx().timestamp(TimeUnit::kNano));
  EXPECT_THROW(scan.ref("col_dec").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastBooleanExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBoolCastThroughCase(scan.ref("col_b").cast("int8"), ctx().int8());
  checkBoolCastThroughCase(scan.ref("col_b").cast("int16"), ctx().int16());
  checkBoolCastThroughCase(scan.ref("col_b").cast("int32"), ctx().int32());
  checkBoolCastThroughCase(scan.ref("col_b").cast("int64"), ctx().int64());
  checkBoolCastThroughCase(
      scan.ref("col_b_nn").cast("int32[nn]"), ctx().int32(false), false);
  checkCast(scan.ref("col_b").cast("fp32"), ctx().fp32());
  checkCast(scan.ref("col_b").cast("fp64"), ctx().fp64());
  checkBoolCastThroughCase(scan.ref("col_b").cast("decimal(10,2)"),
                           ctx().decimal(8, 10, 2));
  ASSERT_TRUE(scan.ref("col_b").cast("bool").expr()->is<hdk::ir::ColumnRef>());
  EXPECT_THROW(scan.ref("col_b").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("time[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("date32[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("timestamp[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("timestamp[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("timestamp[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("timestamp[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastStringExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  // Unallowed casts for string columns.
  for (auto col_name : {"col_str"s, "col_vc_10"s}) {
    EXPECT_THROW(scan.ref(col_name).cast("int8"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("int16"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("int32"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("int64"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("fp32"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("fp64"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("dec(10,2)"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("bool"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("time[s]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("time[ms]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("time[us]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("time[ns]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("date16"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("date32[d]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("date32[s]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("timestamp[s]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("timestamp[ms]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("timestamp[us]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("timestamp[ns]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[m]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[d]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[s]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[ms]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[us]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("interval[ns]"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("array(int)(2)"), InvalidQueryError);
    EXPECT_THROW(scan.ref(col_name).cast("array(int)"), InvalidQueryError);
  }
  EXPECT_THROW(scan.ref("col_str").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_vc_10").cast("text"), InvalidQueryError);
  // Allowed casts for string columns.
  ASSERT_TRUE(scan.ref("col_str").cast("text").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_str").cast("dict(text)[1]"), ctx().extDict(ctx().text(), 1));
  ASSERT_TRUE(scan.ref("col_vc_10").cast("varchar(10)").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_vc_10").cast("dict(text)[1]"), ctx().extDict(ctx().text(), 1));
  // Casts for string literal.
  for (auto type : {"text"s, "varchar(20)"s}) {
    checkCst(builder.cst("10", type).cast("int8"), (int64_t)10, ctx().int8(false));
    EXPECT_THROW(builder.cst("str", type).cast("int8"), InvalidQueryError);
    checkCst(builder.cst("11", type).cast("int16"), (int64_t)11, ctx().int16(false));
    EXPECT_THROW(builder.cst("str", type).cast("int16"), InvalidQueryError);
    checkCst(builder.cst("12", type).cast("int32"), (int64_t)12, ctx().int32(false));
    EXPECT_THROW(builder.cst("str", type).cast("int32"), InvalidQueryError);
    checkCst(builder.cst("13", type).cast("int64"), (int64_t)13, ctx().int64(false));
    EXPECT_THROW(builder.cst("str", type).cast("int64"), InvalidQueryError);
    checkCst(builder.cst("1.3", type).cast("fp32"), 1.3, ctx().fp32(false));
    EXPECT_THROW(builder.cst("str", type).cast("fp32"), InvalidQueryError);
    checkCst(builder.cst("1.3", type).cast("fp64"), 1.3, ctx().fp64(false));
    EXPECT_THROW(builder.cst("str", type).cast("fp64"), InvalidQueryError);
    checkCst(builder.cst("1.3", type).cast("dec(10,2)"),
             (int64_t)130,
             ctx().decimal(8, 10, 2, false));
    EXPECT_THROW(builder.cst("str", type).cast("dec(10,2)"), InvalidQueryError);
    checkCst(builder.cst("true", type).cast("bool"), true, ctx().boolean(false));
    checkCst(builder.cst("TRUE", type).cast("bool"), true, ctx().boolean(false));
    checkCst(builder.cst("t", type).cast("bool"), true, ctx().boolean(false));
    checkCst(builder.cst("T", type).cast("bool"), true, ctx().boolean(false));
    checkCst(builder.cst("1", type).cast("bool"), true, ctx().boolean(false));
    checkCst(builder.cst("false", type).cast("bool"), false, ctx().boolean(false));
    checkCst(builder.cst("FALSE", type).cast("bool"), false, ctx().boolean(false));
    checkCst(builder.cst("f", type).cast("bool"), false, ctx().boolean(false));
    checkCst(builder.cst("F", type).cast("bool"), false, ctx().boolean(false));
    checkCst(builder.cst("0", type).cast("bool"), false, ctx().boolean(false));
    EXPECT_THROW(builder.cst("str", type).cast("bool"), InvalidQueryError);
    checkCst(builder.cst("str", type).cast("text"), "str"s, ctx().text(false));
    checkCst(
        builder.cst("str", type).cast("varchar(4)"), "str"s, ctx().varChar(4, false));
    checkCst(builder.cst("str", type).cast("varchar(2)"), "st"s, ctx().varChar(2, false));
    checkCast(builder.cst("str", type).cast("dict(text)[0]"),
              ctx().extDict(ctx().text(), 0));
    checkCst(builder.cst("01:10:12", type).cast("time[s]"),
             (int64_t)4212,
             ctx().time64(TimeUnit::kSecond, false));
    EXPECT_THROW(builder.cst("str", type).cast("time[s]"), InvalidQueryError);
    checkCst(builder.cst("01:10:12", type).cast("time[ms]"),
             (int64_t)4212000,
             ctx().time64(TimeUnit::kMilli, false));
    EXPECT_THROW(builder.cst("str", type).cast("time[ms]"), InvalidQueryError);
    checkCst(builder.cst("01:10:12", type).cast("time[us]"),
             (int64_t)4212000000,
             ctx().time64(TimeUnit::kMicro, false));
    EXPECT_THROW(builder.cst("str", type).cast("time[us]"), InvalidQueryError);
    checkCst(builder.cst("01:10:12", type).cast("time[ns]"),
             (int64_t)4212000000000,
             ctx().time64(TimeUnit::kNano, false));
    EXPECT_THROW(builder.cst("str", type).cast("time[ns]"), InvalidQueryError);
    // Day encoded dates are stored in seconds.
    checkCst(builder.cst("1970-01-02", type).cast("date64[d]"),
             (int64_t)86400,
             ctx().date64(TimeUnit::kDay, false));
    EXPECT_THROW(builder.cst("str", type).cast("date64[d]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02", type).cast("date64[s]"),
             (int64_t)86400,
             ctx().date64(TimeUnit::kSecond, false));
    EXPECT_THROW(builder.cst("str", type).cast("date64[s]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02", type).cast("date64[ms]"),
             (int64_t)86400000,
             ctx().date64(TimeUnit::kMilli, false));
    EXPECT_THROW(builder.cst("str", type).cast("date64[ms]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02", type).cast("date64[us]"),
             (int64_t)86400000000,
             ctx().date64(TimeUnit::kMicro, false));
    EXPECT_THROW(builder.cst("str", type).cast("date64[us]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02", type).cast("date64[ns]"),
             (int64_t)86400000000000,
             ctx().date64(TimeUnit::kNano, false));
    EXPECT_THROW(builder.cst("str", type).cast("date64[ns]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02 01:10:12", type).cast("timestamp[s]"),
             (int64_t)90612,
             ctx().timestamp(TimeUnit::kSecond, false));
    EXPECT_THROW(builder.cst("str", type).cast("timestamp[s]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02 01:10:12", type).cast("timestamp[ms]"),
             (int64_t)90612000,
             ctx().timestamp(TimeUnit::kMilli, false));
    EXPECT_THROW(builder.cst("str", type).cast("timestamp[ms]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02 01:10:12", type).cast("timestamp[us]"),
             (int64_t)90612000000,
             ctx().timestamp(TimeUnit::kMicro, false));
    EXPECT_THROW(builder.cst("str", type).cast("timestamp[us]"), InvalidQueryError);
    checkCst(builder.cst("1970-01-02 01:10:12", type).cast("timestamp[ns]"),
             (int64_t)90612000000000,
             ctx().timestamp(TimeUnit::kNano, false));
    EXPECT_THROW(builder.cst("str", type).cast("timestamp[ns]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[m]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[d]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[s]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[ms]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[us]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("1", type).cast("interval[ns]"), InvalidQueryError);
    EXPECT_THROW(builder.cst("[1, 2]", type).cast("array(int)(2)"), InvalidQueryError);
    EXPECT_THROW(builder.cst("[1, 2, 3]", type).cast("array(int)"), InvalidQueryError);
  }
}

TEST_F(QueryBuilderTest, CastDictExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  EXPECT_THROW(scan.ref("col_dict").cast("int8"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("int16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("int32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("int64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("fp32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("fp64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("bool"), InvalidQueryError);
  checkCast(scan.ref("col_dict").cast("text"), ctx().text());
  EXPECT_THROW(scan.ref("col_dict").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("time[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("date32[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("timestamp[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("timestamp[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("timestamp[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("timestamp[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastDateExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  EXPECT_THROW(scan.ref("col_date").cast("int8"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("int16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("int32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("int64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("fp32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("fp64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("time[ns]"), InvalidQueryError);
  ASSERT_TRUE(scan.ref("col_date").cast("date32[d]").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_date").cast("date32[s]"), ctx().date32(TimeUnit::kSecond));
  checkCast(scan.ref("col_date").cast("date64[ms]"), ctx().date64(TimeUnit::kMilli));
  checkCast(scan.ref("col_date").cast("date64[us]"), ctx().date64(TimeUnit::kMicro));
  checkCast(scan.ref("col_date").cast("date64[ns]"), ctx().date64(TimeUnit::kNano));
  checkCast(scan.ref("col_date").cast("timestamp[s]"),
            ctx().timestamp(TimeUnit::kSecond));
  checkCast(scan.ref("col_date").cast("timestamp[ms]"),
            ctx().timestamp(TimeUnit::kMilli));
  checkCast(scan.ref("col_date").cast("timestamp[us]"),
            ctx().timestamp(TimeUnit::kMicro));
  checkCast(scan.ref("col_date").cast("timestamp[ns]"), ctx().timestamp(TimeUnit::kNano));
  EXPECT_THROW(scan.ref("col_date").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastTimeExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkCast(scan.ref("col_time").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_time").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_time").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_time").cast("int64"), ctx().int64());
  EXPECT_THROW(scan.ref("col_time").cast("fp32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("fp64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("varchar(10)"), InvalidQueryError);
  ASSERT_TRUE(scan.ref("col_time").cast("time[s]").expr()->as<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_time").cast("time[ms]"), ctx().time64(TimeUnit::kMilli));
  checkCast(scan.ref("col_time").cast("time[us]"), ctx().time64(TimeUnit::kMicro));
  checkCast(scan.ref("col_time").cast("time[ns]"), ctx().time64(TimeUnit::kNano));
  EXPECT_THROW(scan.ref("col_time").cast("date16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("date32[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("date32[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("timestamp[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("timestamp[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("timestamp[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("timestamp[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CastTimestampExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkCast(scan.ref("col_timestamp").cast("int8"), ctx().int8());
  checkCast(scan.ref("col_timestamp").cast("int16"), ctx().int16());
  checkCast(scan.ref("col_timestamp").cast("int32"), ctx().int32());
  checkCast(scan.ref("col_timestamp").cast("int64"), ctx().int64());
  EXPECT_THROW(scan.ref("col_timestamp").cast("fp32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("fp64"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("dec(10,2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("dict(text)"), InvalidQueryError);
  checkCast(scan.ref("col_timestamp").cast("time32[s]"), ctx().time32(TimeUnit::kSecond));
  checkCast(scan.ref("col_timestamp").cast("time32[ms]"), ctx().time32(TimeUnit::kMilli));
  checkCast(scan.ref("col_timestamp").cast("time[s]"), ctx().time64(TimeUnit::kSecond));
  checkCast(scan.ref("col_timestamp").cast("time[ms]"), ctx().time64(TimeUnit::kMilli));
  checkCast(scan.ref("col_timestamp").cast("time[us]"), ctx().time64(TimeUnit::kMicro));
  checkCast(scan.ref("col_timestamp").cast("time[ns]"), ctx().time64(TimeUnit::kNano));
  checkCast(scan.ref("col_timestamp").cast("date32[d]"), ctx().date32(TimeUnit::kDay));
  checkCast(scan.ref("col_timestamp").cast("date32[s]"), ctx().date32(TimeUnit::kSecond));
  checkCast(scan.ref("col_timestamp").cast("date64[ms]"), ctx().date64(TimeUnit::kMilli));
  checkCast(scan.ref("col_timestamp").cast("date64[us]"), ctx().date64(TimeUnit::kMicro));
  checkCast(scan.ref("col_timestamp").cast("date64[ns]"), ctx().date64(TimeUnit::kNano));
  ASSERT_TRUE(
      scan.ref("col_timestamp").cast("timestamp[s]").expr()->is<hdk::ir::ColumnRef>());
  checkCast(scan.ref("col_timestamp").cast("timestamp[ms]"),
            ctx().timestamp(TimeUnit::kMilli));
  checkCast(scan.ref("col_timestamp").cast("timestamp[us]"),
            ctx().timestamp(TimeUnit::kMicro));
  checkCast(scan.ref("col_timestamp").cast("timestamp[ns]"),
            ctx().timestamp(TimeUnit::kNano));
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[m]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[d]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("interval[ns]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("array(int)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CstExprScalar) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  checkCst(builder.cst(120), 120L, ctx().int64(false));
  checkCst(builder.cst(120, "int8"), 120, ctx().int8(false));
  checkCst(builder.cst("-120", "int8"), -120, ctx().int8(false));
  checkCst(builder.cst(1200, "int16"), 1200, ctx().int16(false));
  checkCst(builder.cst("-1200", "int16"), -1200, ctx().int16(false));
  checkCst(builder.cst(12000, "int32"), 12000, ctx().int32(false));
  checkCst(builder.cst("-12000", "int32"), -12000, ctx().int32(false));
  checkCst(builder.cst(120000, "int64"), 120000, ctx().int64(false));
  checkCst(builder.cst("-120000", "int64"), -120000, ctx().int64(false));
  checkCst(builder.cst(12.34), 12.34, ctx().fp64(false));
  checkCst(builder.cst(12.34, "fp32"), 12.34, ctx().fp32(false));
  checkCst(builder.cst(12, "fp32"), 12.0, ctx().fp32(false));
  checkCst(builder.cst("12.34", "fp32"), 12.34, ctx().fp32(false));
  checkCst(builder.cst(12.3456, "fp64"), 12.3456, ctx().fp64(false));
  checkCst(builder.cst(12, "fp64"), 12.0, ctx().fp64(false));
  checkCst(builder.cst("12.3456", "fp64"), 12.3456, ctx().fp64(false));
  checkCst(builder.cst(1234, "dec(10,2)"), 123400, ctx().decimal(8, 10, 2, false));
  checkCst(builder.cst(12.34, "dec(10,2)"), 1234, ctx().decimal(8, 10, 2, false));
  checkCst(builder.cst("1234.56", "dec(10,2)"), 123456, ctx().decimal(8, 10, 2, false));
  checkCst(builder.cstNoScale(1234, "dec(10,2)"), 1234, ctx().decimal(8, 10, 2, false));
  checkCst(builder.trueCst(), true, ctx().boolean(false));
  checkCst(builder.cst(1, "bool"), true, ctx().boolean(false));
  checkCst(builder.cst("true", "bool"), true, ctx().boolean(false));
  checkCst(builder.cst("1", "bool"), true, ctx().boolean(false));
  checkCst(builder.cst("T", "bool"), true, ctx().boolean(false));
  checkCst(builder.falseCst(), false, ctx().boolean(false));
  checkCst(builder.cst(0, "bool"), false, ctx().boolean(false));
  checkCst(builder.cst("false", "bool"), false, ctx().boolean(false));
  checkCst(builder.cst("0", "bool"), false, ctx().boolean(false));
  checkCst(builder.cst("F", "bool"), false, ctx().boolean(false));
  checkCst(builder.cst("str"), "str"s, ctx().text(false));
  checkCst(builder.cst("str", "text"), "str"s, ctx().text(false));
  checkCst(builder.cst(1234, "time[s]"), 1234, ctx().time64(TimeUnit::kSecond, false));
  checkCst(builder.cst(1234, "time[ms]"), 1234, ctx().time64(TimeUnit::kMilli, false));
  checkCst(builder.cst(1234, "time[us]"), 1234, ctx().time64(TimeUnit::kMicro, false));
  checkCst(builder.cst(1234, "time[ns]"), 1234, ctx().time64(TimeUnit::kNano, false));
  checkCst(
      builder.cst("00:20:34", "time[s]"), 1234, ctx().time64(TimeUnit::kSecond, false));
  checkCst(builder.cst("00:20:34", "time[ms]"),
           1234000,
           ctx().time64(TimeUnit::kMilli, false));
  checkCst(builder.cst("00:20:34", "time[us]"),
           1234000000,
           ctx().time64(TimeUnit::kMicro, false));
  checkCst(builder.cst("00:20:34", "time[ns]"),
           1234000000000,
           ctx().time64(TimeUnit::kNano, false));
  checkCst(builder.time("01:10:11"),
           static_cast<int64_t>(4211000000),
           ctx().time64(TimeUnit::kMicro, false));
  EXPECT_THROW(builder.cst(1234, "date[d]"), InvalidQueryError);
  checkCst(builder.cst(1234, "date[s]"), 1234, ctx().date64(TimeUnit::kSecond, false));
  checkCst(builder.cst(1234, "date[ms]"), 1234, ctx().date64(TimeUnit::kMilli, false));
  checkCst(builder.cst(1234, "date[us]"), 1234, ctx().date64(TimeUnit::kMicro, false));
  checkCst(builder.cst(1234, "date[ns]"), 1234, ctx().date64(TimeUnit::kNano, false));
  EXPECT_THROW(builder.cst("1970-01-02", "date[d]"), InvalidQueryError);
  checkCst(builder.cst("1970-01-02", "date[s]"),
           86400,
           ctx().date64(TimeUnit::kSecond, false));
  checkCst(builder.cst("1970-01-02", "date[ms]"),
           86400000,
           ctx().date64(TimeUnit::kMilli, false));
  checkCst(builder.cst("1970-01-02", "date[us]"),
           86400000000,
           ctx().date64(TimeUnit::kMicro, false));
  checkCst(builder.cst("1970-01-02", "date[ns]"),
           86400000000000,
           ctx().date64(TimeUnit::kNano, false));
  checkCst(builder.date("1970-01-02"), 86400, ctx().date64(TimeUnit::kSecond, false));
  checkCst(
      builder.cst(1234, "timestamp[s]"), 1234, ctx().timestamp(TimeUnit::kSecond, false));
  checkCst(
      builder.cst(1234, "timestamp[ms]"), 1234, ctx().timestamp(TimeUnit::kMilli, false));
  checkCst(
      builder.cst(1234, "timestamp[us]"), 1234, ctx().timestamp(TimeUnit::kMicro, false));
  checkCst(
      builder.cst(1234, "timestamp[ns]"), 1234, ctx().timestamp(TimeUnit::kNano, false));
  checkCst(builder.cst("1970-01-02 01:02:03", "timestamp[s]"),
           90123,
           ctx().timestamp(TimeUnit::kSecond, false));
  checkCst(builder.cst("1970-01-02 01:02:03", "timestamp[ms]"),
           90123000,
           ctx().timestamp(TimeUnit::kMilli, false));
  checkCst(builder.cst("1970-01-02 01:02:03", "timestamp[us]"),
           90123000000,
           ctx().timestamp(TimeUnit::kMicro, false));
  checkCst(builder.cst("1970-01-02 01:02:03", "timestamp[ns]"),
           90123000000000,
           ctx().timestamp(TimeUnit::kNano, false));
  checkCst(builder.timestamp("1970-01-02 01:02:03"),
           90123000000,
           ctx().timestamp(TimeUnit::kMicro, false));
  checkCst(
      builder.cst(1234, "interval[d]"), 1234, ctx().interval64(TimeUnit::kDay, false));
  checkCst(
      builder.cst(1234, "interval[s]"), 1234, ctx().interval64(TimeUnit::kSecond, false));
  checkCst(
      builder.cst(1234, "interval[ms]"), 1234, ctx().interval64(TimeUnit::kMilli, false));
  checkCst(
      builder.cst(1234, "interval[us]"), 1234, ctx().interval64(TimeUnit::kMicro, false));
  checkCst(
      builder.cst(1234, "interval[ns]"), 1234, ctx().interval64(TimeUnit::kNano, false));

  checkNullCst(builder.nullCst(), ctx().null());
  checkNullCst(builder.nullCst("int64"), ctx().int64());
  checkNullCst(builder.nullCst("fp64"), ctx().fp64());
  checkNullCst(builder.nullCst("text"), ctx().text());

  EXPECT_THROW(builder.cst(10, "text"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10, "array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10, "array(int)"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "text"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "date"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "time"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "timestamp"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "interval"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "array(int)(2)"), InvalidQueryError);
  EXPECT_THROW(builder.cst(10.1, "array(int)"), InvalidQueryError);
  EXPECT_THROW(builder.cst("1234", "interval[d]"), InvalidQueryError);
  EXPECT_THROW(builder.cst("[1, 2]", "array(int)(2)").expr()->print(), InvalidQueryError);
  EXPECT_THROW(builder.cst("[1, 2]", "array(int)").expr()->print(), InvalidQueryError);
  EXPECT_THROW(builder.cstNoScale(10, "int64"), InvalidQueryError);
  EXPECT_THROW(builder.cstNoScale(10, "fp32"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, CstExprArray) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  checkCst(builder.cst({1, 2, 3, 4}, "array(int8)"),
           {1, 2, 3, 4},
           ctx().arrayVarLen(ctx().int8()));
  checkCst(builder.cst({"1", "2", "3", "4"}, "array(int8)"),
           {1, 2, 3, 4},
           ctx().arrayVarLen(ctx().int8()));
  EXPECT_THROW(builder.cst({1.1, 2.2, 3.3}, "array(int8)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int8)(2)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int8)(5)"), InvalidQueryError);
  checkCst(builder.cst({1, 2, 3}, "array(int8)(3)"),
           {1, 2, 3},
           ctx().arrayFixed(3, ctx().int8()));
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int16)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int16)(4)"), InvalidQueryError);
  checkCst(builder.cst({1, 2, 3, 4}, "array(int32)"),
           {1, 2, 3, 4},
           ctx().arrayVarLen(ctx().int32()));
  checkCst(builder.cst({"1", "2", "3", "4"}, "array(int32)"),
           {1, 2, 3, 4},
           ctx().arrayVarLen(ctx().int32()));
  EXPECT_THROW(builder.cst({1.1, 2.2, 3.3}, "array(int32)"), InvalidQueryError);
  checkCst(builder.cst({1, 2, 3}, "array(int32)(3)"),
           {1, 2, 3},
           ctx().arrayFixed(3, ctx().int32()));
  checkCst(builder.cst({"1", "2", "3"}, "array(int32)(3)"),
           {1, 2, 3},
           ctx().arrayFixed(3, ctx().int32()));
  EXPECT_THROW(builder.cst({1.1, 2.2, 3.3}, "array(int32)(3)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int64)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(int64)(4)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1.1, 2.2, 3.3}, "array(fp32)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1.1, 2.2, 3.3}, "array(fp32)(3)"), InvalidQueryError);
  checkCst(builder.cst({1, 2, 3, 4}, "array(fp64)"),
           {1.0, 2.0, 3.0, 4.0},
           ctx().arrayVarLen(ctx().fp64()));
  checkCst(builder.cst({1.1, 2.2, 3.3, 4.4}, "array(fp64)"),
           {1.1, 2.2, 3.3, 4.4},
           ctx().arrayVarLen(ctx().fp64()));
  checkCst(builder.cst({"1.1", "2.2", "3.3", "4.4"}, "array(fp64)"),
           {1.1, 2.2, 3.3, 4.4},
           ctx().arrayVarLen(ctx().fp64()));
  checkCst(builder.cst({1, 2, 3}, "array(fp64)(3)"),
           {1.0, 2.0, 3.0},
           ctx().arrayFixed(3, ctx().fp64()));
  checkCst(builder.cst({1.1, 2.2, 3.3}, "array(fp64)(3)"),
           {1.1, 2.2, 3.3},
           ctx().arrayFixed(3, ctx().fp64()));
  checkCst(builder.cst({"1.1", "2.2", "3.3"}, "array(fp64)(3)"),
           {1.1, 2.2, 3.3},
           ctx().arrayFixed(3, ctx().fp64()));
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(bool)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(time)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(date)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(timestamp)"), InvalidQueryError);
  EXPECT_THROW(builder.cst({1, 2, 3, 4}, "array(interval)"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, NotExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  EXPECT_THROW(scan.ref("col_bi").logicalNot(), InvalidQueryError);
  EXPECT_THROW(!scan.ref("col_bi"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_ti").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalNot(), InvalidQueryError);
  EXPECT_THROW(!scan.ref("col_f"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalNot(), InvalidQueryError);
  checkUOper(
      scan.ref("col_b").logicalNot(), ctx().boolean(), OpType::kNot, scan.ref("col_b"));
  checkUOper(!scan.ref("col_b"), ctx().boolean(), OpType::kNot, scan.ref("col_b"));
  EXPECT_THROW(scan.ref("col_str").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").logicalNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32x3").logicalNot(), InvalidQueryError);
  checkCst(builder.cst(1, "bool").logicalNot(), false, ctx().boolean(false));
  checkCst(!builder.cst(1, "bool"), false, ctx().boolean(false));
  checkCst(builder.cst(0, "bool").logicalNot(), true, ctx().boolean(false));
  checkCst(!builder.cst(0, "bool"), true, ctx().boolean(false));
}

TEST_F(QueryBuilderTest, UMinusExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkUOper(
      scan.ref("col_bi").uminus(), ctx().int64(), OpType::kUMinus, scan.ref("col_bi"));
  checkUOper(-scan.ref("col_i"), ctx().int32(), OpType::kUMinus, scan.ref("col_i"));
  checkUOper(
      scan.ref("col_si").uminus(), ctx().int16(), OpType::kUMinus, scan.ref("col_si"));
  checkUOper(-scan.ref("col_ti"), ctx().int8(), OpType::kUMinus, scan.ref("col_ti"));
  checkUOper(
      scan.ref("col_f").uminus(), ctx().fp32(), OpType::kUMinus, scan.ref("col_f"));
  checkUOper(-scan.ref("col_d"), ctx().fp64(), OpType::kUMinus, scan.ref("col_d"));
  checkUOper(scan.ref("col_dec").uminus(),
             ctx().decimal(8, 10, 2),
             OpType::kUMinus,
             scan.ref("col_dec"));
  EXPECT_THROW(scan.ref("col_b").uminus(), InvalidQueryError);
  EXPECT_THROW(-scan.ref("col_b"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").uminus(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").uminus(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").uminus(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").uminus(), InvalidQueryError);
  EXPECT_THROW(-scan.ref("col_time"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").uminus(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").uminus(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32x3").uminus(), InvalidQueryError);
  checkCst(builder.cst(1, "int8").uminus(), -1, ctx().int8(false));
  checkCst(-builder.cst(2, "int16"), -2, ctx().int16(false));
  checkCst(builder.cst(3, "int32").uminus(), -3, ctx().int32(false));
  checkCst(-builder.cst(4, "int64"), -4, ctx().int64(false));
  checkCst(builder.cst(12.34, "fp32").uminus(), -12.34, ctx().fp32(false));
  checkCst(-builder.cst(12.3456, "fp64"), -12.3456, ctx().fp64(false));
  checkCst(
      builder.cst(12.34, "dec(10,2)").uminus(), -1234, ctx().decimal(8, 10, 2, false));
  checkCst(
      -builder.cst(12.34, "dec(10,2)").uminus(), 1234, ctx().decimal(8, 10, 2, false));
}

TEST_F(QueryBuilderTest, IsNullExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  for (auto& col_name : {"col_bi"s,
                         "col_i"s,
                         "col_si"s,
                         "col_ti"s,
                         "col_f"s,
                         "col_d"s,
                         "col_dec"s,
                         "col_b"s,
                         "col_str"s,
                         "col_dict"s,
                         "col_time"s,
                         "col_date"s,
                         "col_timestamp"s,
                         "col_arr_i32"s,
                         "col_arr_i32x3"s}) {
    checkUOper(scan.ref(col_name).isNull(),
               ctx().boolean(false),
               OpType::kIsNull,
               scan.ref(col_name));
  }
  checkCst(scan.ref("col_b_nn").isNull(), false, ctx().boolean(false));
  checkCst(builder.nullCst().isNull(), true, ctx().boolean(false));
  checkCst(builder.nullCst(ctx().int32()).isNull(), true, ctx().boolean(false));
  checkCst(builder.nullCst(ctx().arrayVarLen(ctx().int32())).isNull(),
           true,
           ctx().boolean(false));
}

TEST_F(QueryBuilderTest, UnnestExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  EXPECT_THROW(scan.ref("col_bi").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_ti").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").unnest(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").unnest(), InvalidQueryError);
  checkUOper(scan.ref("col_arr_i32").unnest(),
             ctx().int32(),
             OpType::kUnnest,
             scan.ref("col_arr_i32"));
  checkUOper(scan.ref("col_arr_i32x3").unnest(),
             ctx().int32(),
             OpType::kUnnest,
             scan.ref("col_arr_i32x3"));
}

TEST_F(QueryBuilderTest, PlusExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi").add(scan.ref("col_bi")),
               ctx().int64(),
               OpType::kPlus,
               scan.ref("col_bi"),
               scan.ref("col_bi"));
  checkBinOper(scan.ref("col_bi").add(1),
               ctx().int64(),
               OpType::kPlus,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") + 1,
               ctx().int32(),
               OpType::kPlus,
               scan.ref("col_si"),
               builder.cst(1, "int32"));
  checkBinOper(1 + scan.ref("col_si"),
               ctx().int32(),
               OpType::kPlus,
               builder.cst(1, "int32"),
               scan.ref("col_si"));
  checkBinOper(1L + scan.ref("col_si"),
               ctx().int64(),
               OpType::kPlus,
               builder.cst(1, "int64"),
               scan.ref("col_si"));
  checkBinOper(scan.ref("col_si") + 1L,
               ctx().int64(),
               OpType::kPlus,
               scan.ref("col_si"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_si") + scan.ref("col_f"),
               ctx().fp32(),
               OpType::kPlus,
               scan.ref("col_si"),
               scan.ref("col_f"));
  checkBinOper(scan.ref("col_i").add(2.0),
               ctx().fp64(),
               OpType::kPlus,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(scan.ref("col_i") + 2.0f,
               ctx().fp32(),
               OpType::kPlus,
               scan.ref("col_i"),
               builder.cst(2.0, "fp32"));
  checkBinOper(scan.ref("col_i") + 2.0,
               ctx().fp64(),
               OpType::kPlus,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(2.0f + scan.ref("col_i"),
               ctx().fp32(),
               OpType::kPlus,
               builder.cst(2.0, "fp32"),
               scan.ref("col_i"));
  checkBinOper(2.0 + scan.ref("col_i"),
               ctx().fp64(),
               OpType::kPlus,
               builder.cst(2.0),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_i").add(scan.ref("col_dec")),
               ctx().decimal64(13, 2),
               OpType::kPlus,
               scan.ref("col_i"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_f").add(scan.ref("col_i")),
               ctx().fp32(),
               OpType::kPlus,
               scan.ref("col_f"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_f").add(scan.ref("col_d")),
               ctx().fp64(),
               OpType::kPlus,
               scan.ref("col_f"),
               scan.ref("col_d"));
  checkBinOper(scan.ref("col_f").add(scan.ref("col_dec")),
               ctx().fp32(),
               OpType::kPlus,
               scan.ref("col_f"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").add(scan.ref("col_dec")),
               ctx().decimal64(11, 2),
               OpType::kPlus,
               scan.ref("col_dec"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").add(scan.ref("col_ti")),
               ctx().decimal64(11, 2),
               OpType::kPlus,
               scan.ref("col_dec"),
               scan.ref("col_ti"));
  checkBinOper(scan.ref("col_dec").add(scan.ref("col_f")),
               ctx().fp32(),
               OpType::kPlus,
               scan.ref("col_dec"),
               scan.ref("col_f"));
  checkBinOper(builder.cst(12, "interval[d]").add(builder.cst(15, "interval[s]")),
               ctx().interval64(TimeUnit::kSecond, false),
               OpType::kPlus,
               builder.cst(12, "interval[d]"),
               builder.cst(15, "interval[s]"));
  checkBinOper(builder.cst(12, "interval[d]").add(builder.cst(15, "interval[d]")),
               ctx().interval64(TimeUnit::kDay, false),
               OpType::kPlus,
               builder.cst(12, "interval[d]"),
               builder.cst(15, "interval[d]"));
  checkBinOper(builder.cst(12, "interval[ms]").add(builder.cst(15, "interval[ns]")),
               ctx().interval64(TimeUnit::kNano, false),
               OpType::kPlus,
               builder.cst(12, "interval[ms]"),
               builder.cst(15, "interval[ns]"));
  checkDateAdd(scan.ref("col_date").add(builder.cst(123, "interval32[d]")),
               ctx().date32(TimeUnit::kDay),
               DateAddField::kDay,
               builder.cst(123, "interval32[d]"),
               scan.ref("col_date"));
  checkDateAdd(builder.cst(123, "interval64[s]").add(scan.ref("col_date")),
               ctx().date64(TimeUnit::kSecond),
               DateAddField::kSecond,
               builder.cst(123, "interval64[s]"),
               scan.ref("col_date"));
  checkDateAdd(scan.ref("col_timestamp").add(builder.cst(123, "interval[ms]")),
               ctx().timestamp(TimeUnit::kMilli),
               DateAddField::kMilli,
               builder.cst(123, "interval[ms]"),
               scan.ref("col_timestamp"));
  checkDateAdd(builder.cst(123, "interval[us]").add(scan.ref("col_timestamp")),
               ctx().timestamp(TimeUnit::kMicro),
               DateAddField::kMicro,
               builder.cst(123, "interval[us]"),
               scan.ref("col_timestamp"));
  EXPECT_THROW(builder.cst(123, "interval[us]").add(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").add(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").add(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").add(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").add(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").add(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").add(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").add(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").add(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").add(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").add(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").add(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").add(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").add(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").add(scan.ref("col_arr_i32x3")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, DateAddExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkDateAdd(scan.ref("col_date").add(scan.ref("col_bi"), "year"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kYear,
               scan.ref("col_bi"),
               scan.ref("col_date"));
  checkDateAdd(scan.ref("col_date2").add(scan.ref("col_i"), "YEARS"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kYear,
               scan.ref("col_i").cast("int64"),
               scan.ref("col_date2"));
  checkDateAdd(scan.ref("col_date3").add(scan.ref("col_si"), "quarter"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kQuarter,
               scan.ref("col_si").cast("int64"),
               scan.ref("col_date3"));
  checkDateAdd(scan.ref("col_date4").add(scan.ref("col_ti"), " quarterS "),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kQuarter,
               scan.ref("col_ti").cast("int64"),
               scan.ref("col_date4"));
  checkDateAdd(scan.ref("col_timestamp").add(1, "month"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kMonth,
               builder.cst(1),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp2").add(1, "months"),
               ctx().timestamp(TimeUnit::kMilli),
               DateAddField::kMonth,
               builder.cst(1),
               scan.ref("col_timestamp2"));
  checkDateAdd(scan.ref("col_timestamp3").add(1L, "day"),
               ctx().timestamp(TimeUnit::kMicro),
               DateAddField::kDay,
               builder.cst(1),
               scan.ref("col_timestamp3"));
  checkDateAdd(scan.ref("col_timestamp4").add(1L, "days"),
               ctx().timestamp(TimeUnit::kNano),
               DateAddField::kDay,
               builder.cst(1),
               scan.ref("col_timestamp4"));
  checkDateAdd(scan.ref("col_timestamp").add(1, "hour"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kHour,
               builder.cst(1),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").add(1, "hours"),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kHour,
               builder.cst(1),
               scan.ref("col_timestamp"));
  for (auto& field : {"min"s, "mins"s, "minute"s, "minutes"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kMinute,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"sec"s, "secs"s, "second"s, "SECONDS"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kSecond,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"ms"s, "milli"s, "millisecond"s, "milliseconds"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kMilli,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"us"s, "micro"s, "microsecond"s, "microseconds"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kMicro,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"ns"s, "nano"s, "nanosecond"s, "nanoseconds"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kNano,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"week"s, "weeks"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kWeek,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"quarterday"s,
                      "quarter day"s,
                      "quarter_day"s,
                      "quarterdays"s,
                      "quarter_days"s,
                      "quarter days"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kQuarterDay,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field :
       {"weekday"s, "week day"s, "week_day"s, "weekdays"s, "week_days"s, "week days"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kWeekDay,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"dayofyear"s, "day_of_year"s, "day of year"s, "doy"s}) {
    checkDateAdd(scan.ref("col_timestamp").add(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kDayOfYear,
                 builder.cst(1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"millennium"s, "millenniums"s}) {
    checkDateAdd(scan.ref("col_timestamp").sub(scan.ref("col_i"), field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kMillennium,
                 scan.ref("col_i").uminus().cast("int64"),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"century"s, "centuries"s}) {
    checkDateAdd(scan.ref("col_timestamp").sub(1, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kCentury,
                 builder.cst(-1),
                 scan.ref("col_timestamp"));
  }
  for (auto& field : {"decade"s, "decades"s}) {
    checkDateAdd(scan.ref("col_timestamp").sub(1L, field),
                 ctx().timestamp(TimeUnit::kSecond),
                 DateAddField::kDecade,
                 builder.cst(-1, "int64"),
                 scan.ref("col_timestamp"));
  }
  checkDateAdd(scan.ref("col_timestamp").add(scan.ref("col_i"), DateAddField::kMilli),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kMilli,
               scan.ref("col_i").cast("int64"),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").add(1, DateAddField::kSecond),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kSecond,
               builder.cst(1),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").add(1L, DateAddField::kDay),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kDay,
               builder.cst(1),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").sub(scan.ref("col_i"), DateAddField::kMilli),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kMilli,
               scan.ref("col_i").uminus().cast("int64"),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").sub(1, DateAddField::kSecond),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kSecond,
               builder.cst(-1),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").sub(1L, DateAddField::kDay),
               ctx().timestamp(TimeUnit::kSecond),
               DateAddField::kDay,
               builder.cst(-1),
               scan.ref("col_timestamp"));
  EXPECT_THROW(scan.ref("col_timestamp").add(1, "mss"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(1, "nanos"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(1, "milli seconds"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_ti").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32x3").add(1, "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_f"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_d"), "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_dec"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_b"), "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_str"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_dict"), "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_date"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_time"), "day"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_timestamp"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(scan.ref("col_arr_i32"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").add(scan.ref("col_arr_i32x3"), "day"),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").add(builder.cst(123, "interval[s]"), "day"),
               InvalidQueryError);
}

TEST_F(QueryBuilderTest, MinusExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi").sub(scan.ref("col_bi")),
               ctx().int64(),
               OpType::kMinus,
               scan.ref("col_bi"),
               scan.ref("col_bi"));
  checkBinOper(scan.ref("col_bi").sub(1),
               ctx().int64(),
               OpType::kMinus,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") - 1,
               ctx().int32(),
               OpType::kMinus,
               scan.ref("col_si"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") - 1L,
               ctx().int64(),
               OpType::kMinus,
               scan.ref("col_si"),
               builder.cst(1, "int64"));
  checkBinOper(1 - scan.ref("col_si"),
               ctx().int32(),
               OpType::kMinus,
               builder.cst(1, "int32"),
               scan.ref("col_si"));
  checkBinOper(1L - scan.ref("col_si"),
               ctx().int64(),
               OpType::kMinus,
               builder.cst(1, "int64"),
               scan.ref("col_si"));
  checkBinOper(scan.ref("col_si") - scan.ref("col_f"),
               ctx().fp32(),
               OpType::kMinus,
               scan.ref("col_si"),
               scan.ref("col_f"));
  checkBinOper(scan.ref("col_i").sub(2.0),
               ctx().fp64(),
               OpType::kMinus,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(scan.ref("col_i") - 2.0f,
               ctx().fp32(),
               OpType::kMinus,
               scan.ref("col_i"),
               builder.cst(2.0, "fp32"));
  checkBinOper(scan.ref("col_i") - 2.0,
               ctx().fp64(),
               OpType::kMinus,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(2.0f - scan.ref("col_i"),
               ctx().fp32(),
               OpType::kMinus,
               builder.cst(2.0, "fp32"),
               scan.ref("col_i"));
  checkBinOper(2.0 - scan.ref("col_i"),
               ctx().fp64(),
               OpType::kMinus,
               builder.cst(2.0),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_i").sub(scan.ref("col_dec")),
               ctx().decimal64(13, 2),
               OpType::kMinus,
               scan.ref("col_i"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_f").sub(scan.ref("col_i")),
               ctx().fp32(),
               OpType::kMinus,
               scan.ref("col_f"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_f").sub(scan.ref("col_d")),
               ctx().fp64(),
               OpType::kMinus,
               scan.ref("col_f"),
               scan.ref("col_d"));
  checkBinOper(scan.ref("col_f").sub(scan.ref("col_dec")),
               ctx().fp32(),
               OpType::kMinus,
               scan.ref("col_f"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").sub(scan.ref("col_dec")),
               ctx().decimal64(11, 2),
               OpType::kMinus,
               scan.ref("col_dec"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").sub(scan.ref("col_ti")),
               ctx().decimal64(11, 2),
               OpType::kMinus,
               scan.ref("col_dec"),
               scan.ref("col_ti"));
  checkBinOper(scan.ref("col_dec").sub(scan.ref("col_f")),
               ctx().fp32(),
               OpType::kMinus,
               scan.ref("col_dec"),
               scan.ref("col_f"));
  checkBinOper(builder.cst(12, "interval[d]").sub(builder.cst(15, "interval[s]")),
               ctx().interval64(TimeUnit::kSecond, false),
               OpType::kMinus,
               builder.cst(12, "interval[d]"),
               builder.cst(15, "interval[s]"));
  checkBinOper(builder.cst(12, "interval[d]").sub(builder.cst(15, "interval[d]")),
               ctx().interval64(TimeUnit::kDay, false),
               OpType::kMinus,
               builder.cst(12, "interval[d]"),
               builder.cst(15, "interval[d]"));
  checkBinOper(builder.cst(12, "interval[ms]").sub(builder.cst(15, "interval[ns]")),
               ctx().interval64(TimeUnit::kNano, false),
               OpType::kMinus,
               builder.cst(12, "interval[ms]"),
               builder.cst(15, "interval[ns]"));
  checkDateAdd(scan.ref("col_date").sub(builder.cst(123, "interval32[d]")),
               ctx().date32(TimeUnit::kDay),
               DateAddField::kDay,
               builder.cst(-123, "interval32[d]"),
               scan.ref("col_date"));
  checkDateAdd(scan.ref("col_date").sub(builder.cst(123, "interval64[s]")),
               ctx().date64(TimeUnit::kSecond),
               DateAddField::kSecond,
               builder.cst(-123, "interval64[s]"),
               scan.ref("col_date"));
  checkDateAdd(scan.ref("col_timestamp").sub(builder.cst(123, "interval[ms]")),
               ctx().timestamp(TimeUnit::kMilli),
               DateAddField::kMilli,
               builder.cst(-123, "interval[ms]"),
               scan.ref("col_timestamp"));
  checkDateAdd(scan.ref("col_timestamp").sub(builder.cst(123, "interval[us]")),
               ctx().timestamp(TimeUnit::kMicro),
               DateAddField::kMicro,
               builder.cst(-123, "interval[us]"),
               scan.ref("col_timestamp"));
  EXPECT_THROW(builder.cst(123, "interval64[s]").sub(scan.ref("col_date")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval[us]").sub(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").sub(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").sub(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").sub(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").sub(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").sub(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").sub(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").sub(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").sub(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").sub(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").sub(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").sub(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").sub(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").sub(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").sub(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").sub(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").sub(scan.ref("col_arr_i32x3")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, MulExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi").mul(scan.ref("col_bi")),
               ctx().int64(),
               OpType::kMul,
               scan.ref("col_bi"),
               scan.ref("col_bi"));
  checkBinOper(scan.ref("col_bi").mul(1),
               ctx().int64(),
               OpType::kMul,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") * 1,
               ctx().int32(),
               OpType::kMul,
               scan.ref("col_si"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") * 1L,
               ctx().int64(),
               OpType::kMul,
               scan.ref("col_si"),
               builder.cst(1, "int64"));
  checkBinOper(1 * scan.ref("col_si"),
               ctx().int32(),
               OpType::kMul,
               builder.cst(1, "int32"),
               scan.ref("col_si"));
  checkBinOper(1L * scan.ref("col_si"),
               ctx().int64(),
               OpType::kMul,
               builder.cst(1, "int64"),
               scan.ref("col_si"));
  checkBinOper(scan.ref("col_si") * scan.ref("col_f"),
               ctx().fp32(),
               OpType::kMul,
               scan.ref("col_si"),
               scan.ref("col_f"));
  checkBinOper(scan.ref("col_i").mul(2.0),
               ctx().fp64(),
               OpType::kMul,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(scan.ref("col_i") * 2.0f,
               ctx().fp32(),
               OpType::kMul,
               scan.ref("col_i"),
               builder.cst(2.0, "fp32"));
  checkBinOper(scan.ref("col_i") * 2.0,
               ctx().fp64(),
               OpType::kMul,
               scan.ref("col_i"),
               builder.cst(2.0, "fp64"));
  checkBinOper(2.0f * scan.ref("col_i"),
               ctx().fp32(),
               OpType::kMul,
               builder.cst(2.0, "fp32"),
               scan.ref("col_i"));
  checkBinOper(2.0 * scan.ref("col_i"),
               ctx().fp64(),
               OpType::kMul,
               builder.cst(2.0, "fp64"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_i").mul(scan.ref("col_dec")),
               ctx().decimal64(12, 2),
               OpType::kMul,
               scan.ref("col_i"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_f").mul(scan.ref("col_i")),
               ctx().fp32(),
               OpType::kMul,
               scan.ref("col_f"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_f").mul(scan.ref("col_d")),
               ctx().fp64(),
               OpType::kMul,
               scan.ref("col_f"),
               scan.ref("col_d"));
  checkBinOper(scan.ref("col_f").mul(scan.ref("col_dec")),
               ctx().fp32(),
               OpType::kMul,
               scan.ref("col_f"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").mul(scan.ref("col_dec")),
               ctx().decimal64(20, 4),
               OpType::kMul,
               scan.ref("col_dec"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").mul(scan.ref("col_ti")),
               ctx().decimal64(10, 2),
               OpType::kMul,
               scan.ref("col_dec"),
               scan.ref("col_ti"));
  checkBinOper(scan.ref("col_dec").mul(scan.ref("col_f")),
               ctx().fp32(),
               OpType::kMul,
               scan.ref("col_dec"),
               scan.ref("col_f"));
  checkBinOper(builder.cst(12, "interval[s]").mul(builder.cst(15, "int32")),
               ctx().interval64(TimeUnit::kSecond, false),
               OpType::kMul,
               builder.cst(12, "interval[s]"),
               builder.cst(15, "int32"));
  checkBinOper(builder.cst(15, "int64").mul(builder.cst(12, "interval[d]")),
               ctx().interval64(TimeUnit::kDay, false),
               OpType::kMul,
               builder.cst(15, "int64"),
               builder.cst(12, "interval[d]"));
  EXPECT_THROW(builder.cst(12, "interval[ms]").mul(builder.cst(15.0)), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "dec(10,2)").mul(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").mul(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").mul(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mul(builder.cst(123, "interval32[d]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mul(builder.cst(123, "interval64[s]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mul(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mul(builder.cst(123, "interval[us]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval64[s]").mul(scan.ref("col_date")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval[us]").mul(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").mul(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mul(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mul(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").mul(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").mul(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").mul(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").mul(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").mul(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").mul(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").mul(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mul(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").mul(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").mul(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mul(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").mul(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").mul(scan.ref("col_arr_i32x3")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, DivExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi").div(scan.ref("col_bi")),
               ctx().int64(),
               OpType::kDiv,
               scan.ref("col_bi"),
               scan.ref("col_bi"));
  checkBinOper(scan.ref("col_bi").div(1),
               ctx().int64(),
               OpType::kDiv,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") / 1,
               ctx().int32(),
               OpType::kDiv,
               scan.ref("col_si"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") / 1L,
               ctx().int64(),
               OpType::kDiv,
               scan.ref("col_si"),
               builder.cst(1, "int64"));
  checkBinOper(1 / scan.ref("col_si"),
               ctx().int32(),
               OpType::kDiv,
               builder.cst(1, "int32"),
               scan.ref("col_si"));
  checkBinOper(1L / scan.ref("col_si"),
               ctx().int64(),
               OpType::kDiv,
               builder.cst(1, "int64"),
               scan.ref("col_si"));
  checkBinOper(scan.ref("col_si") / scan.ref("col_f"),
               ctx().fp32(),
               OpType::kDiv,
               scan.ref("col_si"),
               scan.ref("col_f"));
  checkBinOper(scan.ref("col_i").div(2.0),
               ctx().fp64(),
               OpType::kDiv,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(scan.ref("col_i") / 2.0f,
               ctx().fp32(),
               OpType::kDiv,
               scan.ref("col_i"),
               builder.cst(2.0, "fp32"));
  checkBinOper(scan.ref("col_i") / 2.0,
               ctx().fp64(),
               OpType::kDiv,
               scan.ref("col_i"),
               builder.cst(2.0));
  checkBinOper(2.0f / scan.ref("col_i"),
               ctx().fp32(),
               OpType::kDiv,
               builder.cst(2.0, "fp32"),
               scan.ref("col_i"));
  checkBinOper(2.0 / scan.ref("col_i"),
               ctx().fp64(),
               OpType::kDiv,
               builder.cst(2.0),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_i").div(scan.ref("col_dec")),
               ctx().decimal64(12, 2),
               OpType::kDiv,
               scan.ref("col_i"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_f").div(scan.ref("col_i")),
               ctx().fp32(),
               OpType::kDiv,
               scan.ref("col_f"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_f").div(scan.ref("col_d")),
               ctx().fp64(),
               OpType::kDiv,
               scan.ref("col_f"),
               scan.ref("col_d"));
  checkBinOper(scan.ref("col_f").div(scan.ref("col_dec")),
               ctx().fp32(),
               OpType::kDiv,
               scan.ref("col_f"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").div(scan.ref("col_dec")),
               ctx().decimal64(10, 2),
               OpType::kDiv,
               scan.ref("col_dec"),
               scan.ref("col_dec"));
  checkBinOper(scan.ref("col_dec").div(scan.ref("col_ti")),
               ctx().decimal64(10, 2),
               OpType::kDiv,
               scan.ref("col_dec"),
               scan.ref("col_ti"));
  checkBinOper(scan.ref("col_dec").div(scan.ref("col_f")),
               ctx().fp32(),
               OpType::kDiv,
               scan.ref("col_dec"),
               scan.ref("col_f"));
  checkBinOper(builder.cst(12, "interval[s]").div(builder.cst(15, "int32")),
               ctx().interval64(TimeUnit::kSecond, false),
               OpType::kDiv,
               builder.cst(12, "interval[s]"),
               builder.cst(15, "int32"));
  EXPECT_THROW(builder.cst(15, "int64").div(builder.cst(12, "interval[d]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").div(builder.cst(15.0)), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "dec(10,2)").div(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").div(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").div(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").div(builder.cst(123, "interval32[d]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").div(builder.cst(123, "interval64[s]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").div(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").div(builder.cst(123, "interval[us]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval64[s]").div(scan.ref("col_date")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval[us]").div(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").div(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").div(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").div(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").div(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").div(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").div(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").div(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").div(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").div(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").div(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").div(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").div(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").div(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").div(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").div(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").div(scan.ref("col_arr_i32x3")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ModExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi").mod(scan.ref("col_bi")),
               ctx().int64(),
               OpType::kMod,
               scan.ref("col_bi"),
               scan.ref("col_bi"));
  checkBinOper(scan.ref("col_bi").mod(1),
               ctx().int64(),
               OpType::kMod,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_si") % 12,
               ctx().int32(),
               OpType::kMod,
               scan.ref("col_si"),
               builder.cst(12, "int32"));
  checkBinOper(scan.ref("col_si") % 12L,
               ctx().int64(),
               OpType::kMod,
               scan.ref("col_si"),
               builder.cst(12, "int64"));
  checkBinOper(123 % scan.ref("col_si"),
               ctx().int32(),
               OpType::kMod,
               builder.cst(123, "int32"),
               scan.ref("col_si"));
  checkBinOper(123L % scan.ref("col_si"),
               ctx().int64(),
               OpType::kMod,
               builder.cst(123, "int64"),
               scan.ref("col_si"));
  checkBinOper(scan.ref("col_si") % scan.ref("col_ti"),
               ctx().int16(),
               OpType::kMod,
               scan.ref("col_si"),
               scan.ref("col_ti"));
  EXPECT_THROW(scan.ref("col_si").mod(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").mod(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").mod(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").mod(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").mod(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mod(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mod(scan.ref("col_ti")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mod(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[s]").mod(builder.cst(15, "int32")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(15, "int64").mod(builder.cst(12, "interval[d]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").mod(builder.cst(15.0)), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "dec(10,2)").mod(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").mod(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[ms]").mod(builder.cst(15, "interval[ns]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mod(builder.cst(123, "interval32[d]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mod(builder.cst(123, "interval64[s]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mod(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mod(builder.cst(123, "interval[us]")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval64[s]").mod(scan.ref("col_date")),
               InvalidQueryError);
  EXPECT_THROW(builder.cst(123, "interval[us]").mod(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").mod(builder.cst(123, "interval[ms]")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").mod(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").mod(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").mod(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").mod(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").mod(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").mod(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").mod(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").mod(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").mod(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mod(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").mod(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").mod(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").mod(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").mod(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").mod(scan.ref("col_arr_i32x3")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, AndExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_b").logicalAnd(scan.ref("col_b_nn")),
               ctx().boolean(),
               OpType::kAnd,
               scan.ref("col_b"),
               scan.ref("col_b_nn"));
  checkBinOper(scan.ref("col_b_nn") && scan.ref("col_b"),
               ctx().boolean(),
               OpType::kAnd,
               scan.ref("col_b_nn"),
               scan.ref("col_b"));
  EXPECT_THROW(scan.ref("col_bi").logicalAnd(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalAnd(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalAnd(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").logicalAnd(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalAnd(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalAnd(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").logicalAnd(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalAnd(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalAnd(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalAnd(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalAnd(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalAnd(scan.ref("col_ti")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalAnd(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[s]").logicalAnd(builder.cst(15, "int32")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").logicalAnd(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").logicalAnd(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").logicalAnd(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").logicalAnd(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").logicalAnd(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalAnd(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalAnd(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalAnd(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").logicalAnd(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").logicalAnd(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalAnd(scan.ref("col_timestamp")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").logicalAnd(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").logicalAnd(scan.ref("col_arr_i32x3")),
               InvalidQueryError);
}

TEST_F(QueryBuilderTest, OrExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_b").logicalOr(scan.ref("col_b_nn")),
               ctx().boolean(),
               OpType::kOr,
               scan.ref("col_b"),
               scan.ref("col_b_nn"));
  checkBinOper(scan.ref("col_b_nn") || scan.ref("col_b"),
               ctx().boolean(),
               OpType::kOr,
               scan.ref("col_b_nn"),
               scan.ref("col_b"));
  EXPECT_THROW(scan.ref("col_bi").logicalOr(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalOr(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalOr(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").logicalOr(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalOr(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalOr(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").logicalOr(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalOr(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalOr(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalOr(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalOr(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalOr(scan.ref("col_ti")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalOr(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(builder.cst(12, "interval[s]").logicalOr(builder.cst(15, "int32")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").logicalOr(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").logicalOr(scan.ref("col_time")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").logicalOr(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").logicalOr(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").logicalOr(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").logicalOr(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").logicalOr(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalOr(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").logicalOr(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").logicalOr(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").logicalOr(scan.ref("col_timestamp")),
               InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").logicalOr(scan.ref("col_arr_i32")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").logicalOr(scan.ref("col_arr_i32x3")),
               InvalidQueryError);
}

namespace {

struct CmpEq {
  static constexpr OpType op_type = OpType::kEq;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.eq(std::forward<T>(rhs));
  }
};

struct CmpNe {
  static constexpr OpType op_type = OpType::kNe;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.ne(std::forward<T>(rhs));
  }
};

struct CmpLt {
  static constexpr OpType op_type = OpType::kLt;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.lt(std::forward<T>(rhs));
  }
};

struct CmpLe {
  static constexpr OpType op_type = OpType::kLe;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.le(std::forward<T>(rhs));
  }
};

struct CmpGt {
  static constexpr OpType op_type = OpType::kGt;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.gt(std::forward<T>(rhs));
  }
};

struct CmpGe {
  static constexpr OpType op_type = OpType::kGe;

  template <typename T>
  BuilderExpr operator()(const BuilderExpr& lhs, T&& rhs) const {
    return lhs.ge(std::forward<T>(rhs));
  }
};

template <typename CmpT>
void testBuildCmpNumberExpr(const QueryBuilder& builder) {
  CmpT cmp;
  auto scan = builder.scan("test3");
  std::vector<std::string> num_cols = {
      "col_bi"s, "col_i"s, "col_si"s, "col_ti"s, "col_f"s, "col_d"s, "col_dec"s};
  for (auto& lhs_name : num_cols) {
    for (auto& rhs_name : num_cols) {
      checkBinOper(cmp(scan.ref(lhs_name), scan.ref(rhs_name)),
                   ctx().boolean(),
                   cmp.op_type,
                   scan.ref(lhs_name),
                   scan.ref(rhs_name));
    }
    checkBinOper(cmp(scan.ref(lhs_name), 1),
                 ctx().boolean(),
                 cmp.op_type,
                 scan.ref(lhs_name),
                 builder.cst(1, "int32"));
    checkBinOper(cmp(scan.ref(lhs_name), 1L),
                 ctx().boolean(),
                 cmp.op_type,
                 scan.ref(lhs_name),
                 builder.cst(1, "int64"));
    checkBinOper(cmp(scan.ref(lhs_name), 1.0f),
                 ctx().boolean(),
                 cmp.op_type,
                 scan.ref(lhs_name),
                 builder.cst(1.0, "fp32"));
    checkBinOper(cmp(scan.ref(lhs_name), 1.0),
                 ctx().boolean(),
                 cmp.op_type,
                 scan.ref(lhs_name),
                 builder.cst(1.0, "fp64"));
    EXPECT_THROW(cmp(scan.ref(lhs_name), "str"), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_str")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dict")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_date")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_time")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_timestamp")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32x3")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), builder.cst(123, "interval[s]")),
                 InvalidQueryError);
  }
}

}  // namespace

TEST_F(QueryBuilderTest, CmpNumberExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  testBuildCmpNumberExpr<CmpEq>(builder);
  testBuildCmpNumberExpr<CmpNe>(builder);
  testBuildCmpNumberExpr<CmpLt>(builder);
  testBuildCmpNumberExpr<CmpLe>(builder);
  testBuildCmpNumberExpr<CmpGt>(builder);
  testBuildCmpNumberExpr<CmpGe>(builder);
}

template <typename CmpT>
void testBuildCmpStrExpr(const QueryBuilder& builder) {
  CmpT cmp;
  auto scan = builder.scan("test3");
  for (auto& lhs_name : {"col_str"s, "col_dict"s}) {
    for (auto& rhs_name : {"col_str"s, "col_dict"s, "col_dict2"s}) {
      checkBinOper(cmp(scan.ref(lhs_name), scan.ref(rhs_name)),
                   ctx().boolean(),
                   cmp.op_type,
                   scan.ref(lhs_name),
                   scan.ref(rhs_name));
    }
    checkBinOper(cmp(scan.ref(lhs_name), "str"),
                 ctx().boolean(),
                 cmp.op_type,
                 scan.ref(lhs_name),
                 builder.cst("str"));
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1L), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0f), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_bi")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_i")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_si")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_ti")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_f")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_d")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dec")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_date")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_time")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_timestamp")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32x3")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), builder.cst(123, "interval[s]")),
                 InvalidQueryError);
  }
}

TEST_F(QueryBuilderTest, CmpStrExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  testBuildCmpStrExpr<CmpEq>(builder);
  testBuildCmpStrExpr<CmpNe>(builder);
  testBuildCmpStrExpr<CmpLt>(builder);
  testBuildCmpStrExpr<CmpLe>(builder);
  testBuildCmpStrExpr<CmpGt>(builder);
  testBuildCmpStrExpr<CmpGe>(builder);
}

template <typename CmpT>
void testBuildCmpDateTimeExpr(const QueryBuilder& builder) {
  CmpT cmp;
  auto scan = builder.scan("test3");
  for (auto& lhs_name : {"col_date4"s, "col_time"s, "col_timestamp"s}) {
    for (auto& rhs_name : {"col_date3"s, "col_time2"s, "col_timestamp2"s}) {
      auto lhs_ref = scan.ref(lhs_name);
      auto rhs_ref = scan.ref(rhs_name);
      if ((lhs_ref.expr()->type()->isTime() && !rhs_ref.expr()->type()->isTime()) ||
          (!lhs_ref.expr()->type()->isTime() && rhs_ref.expr()->type()->isTime())) {
        EXPECT_THROW(cmp(lhs_ref, rhs_ref), InvalidQueryError);
      } else {
        checkBinOper(
            cmp(lhs_ref, rhs_ref), ctx().boolean(), cmp.op_type, lhs_ref, rhs_ref);
      }
    }
    EXPECT_THROW(cmp(scan.ref(lhs_name), "str"), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1L), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0f), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_bi")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_i")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_si")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_ti")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_f")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_d")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dec")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_str")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dict")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_arr_i32x3")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), builder.cst(123, "interval[s]")),
                 InvalidQueryError);
  }
}

TEST_F(QueryBuilderTest, CmpDateTimeExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  testBuildCmpDateTimeExpr<CmpEq>(builder);
  testBuildCmpDateTimeExpr<CmpNe>(builder);
  testBuildCmpDateTimeExpr<CmpLt>(builder);
  testBuildCmpDateTimeExpr<CmpLe>(builder);
  testBuildCmpDateTimeExpr<CmpGt>(builder);
  testBuildCmpDateTimeExpr<CmpGe>(builder);
}

template <typename CmpT>
void testBuildCmpArrExpr(const QueryBuilder& builder) {
  CmpT cmp;
  auto scan = builder.scan("test3");
  for (auto& lhs_name : {"col_arr_i32"s, "col_arr_i32x3"s}) {
    for (auto& rhs_name : {"col_arr_i32_2"s, "col_arr_i64"s, "col_arr_i32x3_2"s}) {
      auto lhs_ref = scan.ref(lhs_name);
      auto rhs_ref = scan.ref(rhs_name);
      if (lhs_ref.expr()->type()->equal(rhs_ref.expr()->type())) {
        checkBinOper(
            cmp(lhs_ref, rhs_ref), ctx().boolean(), cmp.op_type, lhs_ref, rhs_ref);
      } else {
        EXPECT_THROW(cmp(lhs_ref, rhs_ref), InvalidQueryError);
      }
    }
    EXPECT_THROW(cmp(scan.ref(lhs_name), "str"), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1L), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0f), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), 1.0), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_bi")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_i")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_si")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_ti")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_f")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_d")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dec")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_str")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_dict")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_date")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_time")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), scan.ref("col_timestamp")), InvalidQueryError);
    EXPECT_THROW(cmp(scan.ref(lhs_name), builder.cst(123, "interval[s]")),
                 InvalidQueryError);
  }
}

TEST_F(QueryBuilderTest, CmpArrExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  testBuildCmpArrExpr<CmpEq>(builder);
  testBuildCmpArrExpr<CmpNe>(builder);
  testBuildCmpArrExpr<CmpLt>(builder);
  testBuildCmpArrExpr<CmpLe>(builder);
  testBuildCmpArrExpr<CmpGt>(builder);
  testBuildCmpArrExpr<CmpGe>(builder);
}

TEST_F(QueryBuilderTest, EqOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") == scan.ref("col_i"),
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") == 1,
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") == 1L,
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") == 1.0f,
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") == 1.0,
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") == "str",
               ctx().boolean(),
               OpType::kEq,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 == scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kEq,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L == scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kEq,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f == scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kEq,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 == scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kEq,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" == scan.ref("col_str"),
               ctx().boolean(),
               OpType::kEq,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, NeOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") != scan.ref("col_i"),
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") != 1,
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") != 1L,
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") != 1.0f,
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") != 1.0,
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") != "str",
               ctx().boolean(),
               OpType::kNe,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 != scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kNe,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L != scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kNe,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f != scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kNe,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 != scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kNe,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" != scan.ref("col_str"),
               ctx().boolean(),
               OpType::kNe,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, LtOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") < scan.ref("col_i"),
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") < 1,
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") < 1L,
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") < 1.0f,
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") < 1.0,
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") < "str",
               ctx().boolean(),
               OpType::kLt,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 < scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLt,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L < scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLt,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f < scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLt,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 < scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLt,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" < scan.ref("col_str"),
               ctx().boolean(),
               OpType::kLt,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, LeOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") <= scan.ref("col_i"),
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") <= 1,
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") <= 1L,
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") <= 1.0f,
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") <= 1.0,
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") <= "str",
               ctx().boolean(),
               OpType::kLe,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 <= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLe,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L <= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLe,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f <= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLe,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 <= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kLe,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" <= scan.ref("col_str"),
               ctx().boolean(),
               OpType::kLe,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, GtOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") > scan.ref("col_i"),
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") > 1,
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") > 1L,
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") > 1.0f,
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") > 1.0,
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") > "str",
               ctx().boolean(),
               OpType::kGt,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 > scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGt,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L > scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGt,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f > scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGt,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 > scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGt,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" > scan.ref("col_str"),
               ctx().boolean(),
               OpType::kGt,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, GeOperators) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_bi") >= scan.ref("col_i"),
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_bi"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_bi") >= 1,
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_bi"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_bi") >= 1L,
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_bi"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_bi") >= 1.0f,
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp32"));
  checkBinOper(scan.ref("col_bi") >= 1.0,
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_bi"),
               builder.cst(1.0, "fp64"));
  checkBinOper(scan.ref("col_str") >= "str",
               ctx().boolean(),
               OpType::kGe,
               scan.ref("col_str"),
               builder.cst("str"));
  checkBinOper(1 >= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGe,
               builder.cst(1, "int32"),
               scan.ref("col_bi"));
  checkBinOper(1L >= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGe,
               builder.cst(1, "int64"),
               scan.ref("col_bi"));
  checkBinOper(1.0f >= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGe,
               builder.cst(1.0, "fp32"),
               scan.ref("col_bi"));
  checkBinOper(1.0 >= scan.ref("col_bi"),
               ctx().boolean(),
               OpType::kGe,
               builder.cst(1.0, "fp64"),
               scan.ref("col_bi"));
  checkBinOper("str" >= scan.ref("col_str"),
               ctx().boolean(),
               OpType::kGe,
               builder.cst("str"),
               scan.ref("col_str"));
}

TEST_F(QueryBuilderTest, ArrayAtExpr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkBinOper(scan.ref("col_arr_i64").at(scan.ref("col_i")),
               ctx().int64(),
               OpType::kArrayAt,
               scan.ref("col_arr_i64"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_arr_i32").at(1),
               ctx().int32(),
               OpType::kArrayAt,
               scan.ref("col_arr_i32"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_arr_i32x3").at(1L),
               ctx().int32(),
               OpType::kArrayAt,
               scan.ref("col_arr_i32x3"),
               builder.cst(1, "int64"));
  checkBinOper(scan.ref("col_arr_i64")[scan.ref("col_i")],
               ctx().int64(),
               OpType::kArrayAt,
               scan.ref("col_arr_i64"),
               scan.ref("col_i"));
  checkBinOper(scan.ref("col_arr_i32")[1],
               ctx().int32(),
               OpType::kArrayAt,
               scan.ref("col_arr_i32"),
               builder.cst(1, "int32"));
  checkBinOper(scan.ref("col_arr_i32x3")[1L],
               ctx().int32(),
               OpType::kArrayAt,
               scan.ref("col_arr_i32x3"),
               builder.cst(1, "int64"));
  EXPECT_THROW(scan.ref("col_arr_i64").at(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64")[scan.ref("col_d")], InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64").at(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64")[scan.ref("col_dec")], InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64").at(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64")[scan.ref("col_dict")], InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64").at(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64")[scan.ref("col_time")], InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i64").at(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_si").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_ti").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dec").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").at(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").at(1), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Ceil) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkRef(scan.ref("col_bi").ceil(), scan.node(), 0, "col_bi");
  checkRef(scan.ref("col_i").ceil(), scan.node(), 1, "col_i");
  checkFunctionOper(scan.ref("col_f").ceil(), "CEIL", 1, ctx().fp32());
  checkFunctionOper(scan.ref("col_d").ceil(), "CEIL", 1, ctx().fp64());
  checkFunctionOper(scan.ref("col_dec").ceil(), "CEIL", 1, ctx().decimal64(10, 2));
  EXPECT_THROW(scan.ref("col_b").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").ceil(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").ceil(), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Floor) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkRef(scan.ref("col_bi").floor(), scan.node(), 0, "col_bi");
  checkRef(scan.ref("col_i").floor(), scan.node(), 1, "col_i");
  checkFunctionOper(scan.ref("col_f").floor(), "FLOOR", 1, ctx().fp32());
  checkFunctionOper(scan.ref("col_d").floor(), "FLOOR", 1, ctx().fp64());
  checkFunctionOper(scan.ref("col_dec").floor(), "FLOOR", 1, ctx().decimal64(10, 2));
  EXPECT_THROW(scan.ref("col_b").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").floor(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").floor(), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Pow) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkFunctionOper(scan.ref("col_bi").pow(2), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_i").pow((int64_t)3), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_f").pow(2.3f), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_d").pow(2.0), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_dec").pow(scan.ref("col_d")), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_d").pow(scan.ref("col_bi")), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_d").pow(scan.ref("col_i")), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_d").pow(scan.ref("col_f")), "POWER", 2, ctx().fp64());
  checkFunctionOper(scan.ref("col_d").pow(scan.ref("col_dec")), "POWER", 2, ctx().fp64());
  EXPECT_THROW(scan.ref("col_b").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_arr_i32").pow(2), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_timestamp")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").pow(scan.ref("col_arr_i32")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Over) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto expr1 = builder.rank();
  checkWindowFunction(
      expr1, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 0, 0);

  auto expr2 = expr1.over();
  checkWindowFunction(
      expr2, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 0, 0);

  auto expr3 = expr1.over(scan.ref("col_bi"));
  checkWindowFunction(
      expr3, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 1, 0);
  ASSERT_TRUE(expr3.expr()->as<WindowFunction>()->partitionKeys()[0]->equal(
      scan.ref("col_bi").expr().get()));

  auto expr4 = expr1.over({scan.ref("col_bi"), scan.ref("col_i")});
  checkWindowFunction(
      expr4, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 2, 0);
  ASSERT_TRUE(expr4.expr()->as<WindowFunction>()->partitionKeys()[0]->equal(
      scan.ref("col_bi").expr().get()));
  ASSERT_TRUE(expr4.expr()->as<WindowFunction>()->partitionKeys()[1]->equal(
      scan.ref("col_i").expr().get()));

  auto expr5 = expr1.over(scan.ref("col_bi")).over(scan.ref("col_i"));
  checkWindowFunction(
      expr5, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 2, 0);
  ASSERT_TRUE(expr5.expr()->as<WindowFunction>()->partitionKeys()[0]->equal(
      scan.ref("col_bi").expr().get()));
  ASSERT_TRUE(expr5.expr()->as<WindowFunction>()->partitionKeys()[1]->equal(
      scan.ref("col_i").expr().get()));

  EXPECT_THROW(expr1.over(scan.ref("col_bi").add(1)), InvalidQueryError);
  EXPECT_THROW(expr1.over(builder.cst(1)), InvalidQueryError);
  EXPECT_THROW(builder.cst(1).over(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").stdDev().over(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").over(), InvalidQueryError);
}

TEST_F(QueryBuilderTest, OrderBy) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto expr1 = builder.rank();
  checkWindowFunction(
      expr1, "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 0, 0);

  auto expr2 = expr1.orderBy(scan.ref("col_i"));
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy(
      scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy(scan.ref("col_i"), "asc");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy(scan.ref("col_i"), "desc", "first");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy({scan.ref("col_i"), scan.ref("col_bi")});
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy({scan.ref("col_i"), scan.ref("col_bi")},
                        SortDirection::Descending,
                        NullSortedPosition::First);
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy({scan.ref("col_i"), scan.ref("col_bi")}, "asc");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy({scan.ref("col_i"), scan.ref("col_bi")}, "desc", "first");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Descending, NullSortedPosition::First);

  expr2 =
      expr1.orderBy(std::vector<BuilderExpr>({scan.ref("col_i"), scan.ref("col_bi")}));
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy(std::vector<BuilderExpr>({scan.ref("col_i"), scan.ref("col_bi")}),
                        SortDirection::Descending,
                        NullSortedPosition::First);
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy(std::vector<BuilderExpr>({scan.ref("col_i"), scan.ref("col_bi")}),
                        "asc");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Ascending, NullSortedPosition::Last);

  expr2 = expr1.orderBy(
      std::vector<BuilderExpr>({scan.ref("col_i"), scan.ref("col_bi")}), "desc", "first");
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy(
      {scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First});
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy({scan.ref("col_i"), "desc", "first"});
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Descending, NullSortedPosition::First);

  expr2 = expr1.orderBy(
      {BuilderOrderByKey(
           scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last),
       BuilderOrderByKey(scan.ref("col_bi"), "desc", "first")});
  checkWindowCollation(
      expr2, 0, scan.ref("col_i"), SortDirection::Ascending, NullSortedPosition::Last);
  checkWindowCollation(
      expr2, 1, scan.ref("col_bi"), SortDirection::Descending, NullSortedPosition::First);

  EXPECT_THROW(expr1.orderBy(scan.ref("col_bi").add(1)), InvalidQueryError);
  EXPECT_THROW(expr1.orderBy(builder.cst(1)), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").sum().orderBy(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_bi").orderBy(scan.ref("col_i")), InvalidQueryError);
  EXPECT_THROW(builder.rank().orderBy(scan.ref("col_i"), "descc"), InvalidQueryError);
  EXPECT_THROW(builder.rank().orderBy(scan.ref("col_i"), "desc", "lasttt"),
               InvalidQueryError);
}

TEST_F(QueryBuilderTest, RowNumber) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkWindowFunction(builder.rowNumber(),
                      "row_number",
                      ctx().int64(false),
                      WindowFunctionKind::RowNumber,
                      0,
                      0,
                      0);
  checkWindowFunction(
      builder.rowNumber().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "row_number",
      ctx().int64(false),
      WindowFunctionKind::RowNumber,
      0,
      1,
      1);
}

TEST_F(QueryBuilderTest, Rank) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkWindowFunction(
      builder.rank(), "rank", ctx().int64(false), WindowFunctionKind::Rank, 0, 0, 0);
  checkWindowFunction(builder.rank().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
                      "rank",
                      ctx().int64(false),
                      WindowFunctionKind::Rank,
                      0,
                      1,
                      1);
}

TEST_F(QueryBuilderTest, DenseRank) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkWindowFunction(builder.denseRank(),
                      "dense_rank",
                      ctx().int64(false),
                      WindowFunctionKind::DenseRank,
                      0,
                      0,
                      0);
  checkWindowFunction(
      builder.denseRank().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "dense_rank",
      ctx().int64(false),
      WindowFunctionKind::DenseRank,
      0,
      1,
      1);
}

TEST_F(QueryBuilderTest, PercentRank) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkWindowFunction(builder.percentRank(),
                      "percent_rank",
                      ctx().fp64(false),
                      WindowFunctionKind::PercentRank,
                      0,
                      0,
                      0);
  checkWindowFunction(
      builder.percentRank().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "percent_rank",
      ctx().fp64(false),
      WindowFunctionKind::PercentRank,
      0,
      1,
      1);
}

TEST_F(QueryBuilderTest, NTile) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto ntile1 = builder.nTile(5);
  checkWindowFunction(
      ntile1, "ntile", ctx().int64(false), WindowFunctionKind::NTile, 1, 0, 0);
  checkCst(ntile1.expr()->as<WindowFunction>()->args()[0], 5, ctx().int64(false));
  checkWindowFunction(
      builder.nTile(5).over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "ntile",
      ctx().int64(false),
      WindowFunctionKind::NTile,
      1,
      1,
      1);
  EXPECT_THROW(builder.nTile(0), InvalidQueryError);
  EXPECT_THROW(builder.nTile(-1), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Lag) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto lag1 = scan.ref("col_bi").lag();
  checkWindowFunction(
      lag1, "col_bi_lag", ctx().int64(), WindowFunctionKind::Lag, 2, 0, 0);
  ASSERT_TRUE(lag1.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkCst(lag1.expr()->as<WindowFunction>()->args()[1], 1, ctx().int64(false));

  auto lag5 = scan.ref("col_i").lag(5);
  checkWindowFunction(lag5, "col_i_lag", ctx().int32(), WindowFunctionKind::Lag, 2, 0, 0);
  ASSERT_TRUE(lag5.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(lag5.expr()->as<WindowFunction>()->args()[1], 5, ctx().int64(false));

  auto lag0 = scan.ref("col_i").lag(0);
  checkWindowFunction(lag0, "col_i_lag", ctx().int32(), WindowFunctionKind::Lag, 2, 0, 0);
  ASSERT_TRUE(lag0.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(lag0.expr()->as<WindowFunction>()->args()[1], 0, ctx().int64(false));

  auto lagm5 = scan.ref("col_i").lag(-5);
  checkWindowFunction(
      lagm5, "col_i_lag", ctx().int32(), WindowFunctionKind::Lag, 2, 0, 0);
  ASSERT_TRUE(lagm5.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(lagm5.expr()->as<WindowFunction>()->args()[1], -5, ctx().int64(false));

  checkWindowFunction(
      scan.ref("col_i").lag(5).over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_lag",
      ctx().int32(),
      WindowFunctionKind::Lag,
      2,
      1,
      1);
}

TEST_F(QueryBuilderTest, Lead) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto lead1 = scan.ref("col_bi").lead();
  checkWindowFunction(
      lead1, "col_bi_lead", ctx().int64(), WindowFunctionKind::Lead, 2, 0, 0);
  ASSERT_TRUE(lead1.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkCst(lead1.expr()->as<WindowFunction>()->args()[1], 1, ctx().int64(false));

  auto lead5 = scan.ref("col_i").lead(5);
  checkWindowFunction(
      lead5, "col_i_lead", ctx().int32(), WindowFunctionKind::Lead, 2, 0, 0);
  ASSERT_TRUE(lead5.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(lead5.expr()->as<WindowFunction>()->args()[1], 5, ctx().int64(false));

  auto lead0 = scan.ref("col_i").lead(0);
  checkWindowFunction(
      lead0, "col_i_lead", ctx().int32(), WindowFunctionKind::Lead, 2, 0, 0);
  ASSERT_TRUE(lead0.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(lead0.expr()->as<WindowFunction>()->args()[1], 0, ctx().int64(false));

  auto leadm5 = scan.ref("col_i").lead(-5);
  checkWindowFunction(
      leadm5, "col_i_lead", ctx().int32(), WindowFunctionKind::Lead, 2, 0, 0);
  ASSERT_TRUE(leadm5.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_i").expr().get()));
  checkCst(leadm5.expr()->as<WindowFunction>()->args()[1], -5, ctx().int64(false));

  checkWindowFunction(
      scan.ref("col_i").lead(5).over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_lead",
      ctx().int32(),
      WindowFunctionKind::Lead,
      2,
      1,
      1);
}

TEST_F(QueryBuilderTest, FirstValue) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto first_value = scan.ref("col_bi").firstValue();
  checkWindowFunction(first_value,
                      "col_bi_first_value",
                      ctx().int64(),
                      WindowFunctionKind::FirstValue,
                      1,
                      0,
                      0);
  ASSERT_TRUE(first_value.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").firstValue().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_first_value",
      ctx().int32(),
      WindowFunctionKind::FirstValue,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, LastValueValue) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto last_value = scan.ref("col_bi").lastValue();
  checkWindowFunction(last_value,
                      "col_bi_last_value",
                      ctx().int64(),
                      WindowFunctionKind::LastValue,
                      1,
                      0,
                      0);
  ASSERT_TRUE(last_value.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").lastValue().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_last_value",
      ctx().int32(),
      WindowFunctionKind::LastValue,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, WindowAvg) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto avg = scan.ref("col_bi").avg().over();
  checkWindowFunction(avg, "col_bi_avg", ctx().fp64(), WindowFunctionKind::Avg, 1, 0, 0);
  ASSERT_TRUE(avg.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").avg().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_avg",
      ctx().fp64(),
      WindowFunctionKind::Avg,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, WindowSum) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto sum = scan.ref("col_bi").sum().over();
  checkWindowFunction(sum, "col_bi_sum", ctx().int64(), WindowFunctionKind::Sum, 1, 0, 0);
  ASSERT_TRUE(sum.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").sum().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_sum",
      ctx().int64(),
      WindowFunctionKind::Sum,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, WindowMin) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto min = scan.ref("col_bi").min().over();
  checkWindowFunction(min, "col_bi_min", ctx().int64(), WindowFunctionKind::Min, 1, 0, 0);
  ASSERT_TRUE(min.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").min().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_min",
      ctx().int32(),
      WindowFunctionKind::Min,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, WindowMax) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  auto max = scan.ref("col_bi").max().over();
  checkWindowFunction(max, "col_bi_max", ctx().int64(), WindowFunctionKind::Max, 1, 0, 0);
  ASSERT_TRUE(max.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(
      scan.ref("col_i").max().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
      "col_i_max",
      ctx().int32(),
      WindowFunctionKind::Max,
      1,
      1,
      1);
}

TEST_F(QueryBuilderTest, WindowCount) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test3");
  checkWindowFunction(builder.count().over(),
                      "count",
                      ctx().int32(false),
                      WindowFunctionKind::Count,
                      0,
                      0,
                      0);
  auto count_bi = scan.ref("col_bi").count().over();
  checkWindowFunction(
      count_bi, "col_bi_count", ctx().int32(false), WindowFunctionKind::Count, 1, 0, 0);
  ASSERT_TRUE(count_bi.expr()->as<WindowFunction>()->args()[0]->equal(
      scan.ref("col_bi").expr().get()));
  checkWindowFunction(builder.count().over(scan.ref("col_bi")).orderBy(scan.ref("col_i")),
                      "count",
                      ctx().int32(false),
                      WindowFunctionKind::Count,
                      0,
                      1,
                      1);
}

TEST_F(QueryBuilderTest, BwAnd) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  checkBinOper(scan.ref("col_i8_1").bwAnd(scan.ref("col_i8_2")),
               ctx().int8(),
               OpType::kBwAnd,
               scan.ref("col_i8_1"),
               scan.ref("col_i8_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwAnd(scan.ref("col_i16nn_2")),
               ctx().int16(false),
               OpType::kBwAnd,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i16nn_2"));
  checkBinOper(scan.ref("col_i32nn_1").bwAnd(scan.ref("col_i32_2")),
               ctx().int32(),
               OpType::kBwAnd,
               scan.ref("col_i32nn_1"),
               scan.ref("col_i32_2"));
  checkBinOper(scan.ref("col_i64_1").bwAnd(scan.ref("col_i64nn_2")),
               ctx().int64(),
               OpType::kBwAnd,
               scan.ref("col_i64_1"),
               scan.ref("col_i64nn_2"));
  checkBinOper(scan.ref("col_i8_1").bwAnd(scan.ref("col_i16_2")),
               ctx().int16(),
               OpType::kBwAnd,
               scan.ref("col_i8_1").cast(ctx().int16()),
               scan.ref("col_i16_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwAnd(scan.ref("col_i8nn_2")),
               ctx().int16(false),
               OpType::kBwAnd,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i8nn_2").cast(ctx().int16(false)));
  checkBinOper(scan.ref("col_i32nn_1").bwAnd(scan.ref("col_i64_2")),
               ctx().int64(),
               OpType::kBwAnd,
               scan.ref("col_i32nn_1").cast(ctx().int64(false)),
               scan.ref("col_i64_2"));
  checkBinOper(scan.ref("col_i32_1").bwAnd(scan.ref("col_i16nn_2")),
               ctx().int32(),
               OpType::kBwAnd,
               scan.ref("col_i32_1"),
               scan.ref("col_i16nn_2").cast(ctx().int32(false)));
  checkBinOper(scan.ref("col_i16_1").bwAnd(1),
               ctx().int32(),
               OpType::kBwAnd,
               scan.ref("col_i16_1").cast(ctx().int32()),
               builder.cst(1, ctx().int32(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwAnd((int64_t)1),
               ctx().int64(false),
               OpType::kBwAnd,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwAnd(1),
               ctx().int64(false),
               OpType::kBwAnd,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwAnd(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").bwAnd(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").bwAnd((int64_t)1), InvalidQueryError);
}

TEST_F(QueryBuilderTest, BwAnd_Exec) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  auto dag = scan.proj({scan.ref("col_i8_1").bwAnd(scan.ref("col_i8_2")),
                        scan.ref("col_i16nn_1").bwAnd(scan.ref("col_i16nn_2")),
                        scan.ref("col_i32nn_1").bwAnd(scan.ref("col_i32_2")),
                        scan.ref("col_i64_1").bwAnd(scan.ref("col_i64nn_2")),
                        scan.ref("col_i8_1").bwAnd(scan.ref("col_i16_2")),
                        scan.ref("col_i16nn_1").bwAnd(scan.ref("col_i8nn_2")),
                        scan.ref("col_i32nn_1").bwAnd(scan.ref("col_i64_2")),
                        scan.ref("col_i32_1").bwAnd(scan.ref("col_i16nn_2")),
                        scan.ref("col_i16_1").bwAnd(1),
                        scan.ref("col_i64nn_1").bwAnd((int64_t)1)})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int8_t>(
          {inline_null_value<int8_t>(), 2, 2, inline_null_value<int8_t>()}),
      std::vector<int16_t>({0, 2, 2, 0}),
      std::vector<int32_t>({inline_null_value<int32_t>(), 2, 2, 0}),
      std::vector<int64_t>({0, 2, 2, inline_null_value<int64_t>()}),
      std::vector<int16_t>(
          {inline_null_value<int16_t>(), 2, 2, inline_null_value<int16_t>()}),
      std::vector<int16_t>({0, 2, 2, 0}),
      std::vector<int64_t>({inline_null_value<int64_t>(), 2, 2, 0}),
      std::vector<int32_t>({0, 2, 2, inline_null_value<int32_t>()}),
      std::vector<int32_t>({1, 0, 1, inline_null_value<int32_t>()}),
      std::vector<int64_t>({1, 0, 1, 0}));
}

TEST_F(QueryBuilderTest, BwOr) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  checkBinOper(scan.ref("col_i8_1").bwOr(scan.ref("col_i8_2")),
               ctx().int8(),
               OpType::kBwOr,
               scan.ref("col_i8_1"),
               scan.ref("col_i8_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwOr(scan.ref("col_i16nn_2")),
               ctx().int16(false),
               OpType::kBwOr,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i16nn_2"));
  checkBinOper(scan.ref("col_i32nn_1").bwOr(scan.ref("col_i32_2")),
               ctx().int32(),
               OpType::kBwOr,
               scan.ref("col_i32nn_1"),
               scan.ref("col_i32_2"));
  checkBinOper(scan.ref("col_i64_1").bwOr(scan.ref("col_i64nn_2")),
               ctx().int64(),
               OpType::kBwOr,
               scan.ref("col_i64_1"),
               scan.ref("col_i64nn_2"));
  checkBinOper(scan.ref("col_i8_1").bwOr(scan.ref("col_i16_2")),
               ctx().int16(),
               OpType::kBwOr,
               scan.ref("col_i8_1").cast(ctx().int16()),
               scan.ref("col_i16_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwOr(scan.ref("col_i8nn_2")),
               ctx().int16(false),
               OpType::kBwOr,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i8nn_2").cast(ctx().int16(false)));
  checkBinOper(scan.ref("col_i32nn_1").bwOr(scan.ref("col_i64_2")),
               ctx().int64(),
               OpType::kBwOr,
               scan.ref("col_i32nn_1").cast(ctx().int64(false)),
               scan.ref("col_i64_2"));
  checkBinOper(scan.ref("col_i32_1").bwOr(scan.ref("col_i16nn_2")),
               ctx().int32(),
               OpType::kBwOr,
               scan.ref("col_i32_1"),
               scan.ref("col_i16nn_2").cast(ctx().int32(false)));
  checkBinOper(scan.ref("col_i16_1").bwOr(1),
               ctx().int32(),
               OpType::kBwOr,
               scan.ref("col_i16_1").cast(ctx().int32()),
               builder.cst(1, ctx().int32(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwOr((int64_t)1),
               ctx().int64(false),
               OpType::kBwOr,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwOr(1),
               ctx().int64(false),
               OpType::kBwOr,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwOr(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").bwOr(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").bwOr((int64_t)1), InvalidQueryError);
}

TEST_F(QueryBuilderTest, BwOr_Exec) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  auto dag = scan.proj({scan.ref("col_i8_1").bwOr(scan.ref("col_i8_2")),
                        scan.ref("col_i16nn_1").bwOr(scan.ref("col_i16nn_2")),
                        scan.ref("col_i32nn_1").bwOr(scan.ref("col_i32_2")),
                        scan.ref("col_i64_1").bwOr(scan.ref("col_i64nn_2")),
                        scan.ref("col_i8_1").bwOr(scan.ref("col_i16_2")),
                        scan.ref("col_i16nn_1").bwOr(scan.ref("col_i8nn_2")),
                        scan.ref("col_i32nn_1").bwOr(scan.ref("col_i64_2")),
                        scan.ref("col_i32_1").bwOr(scan.ref("col_i16nn_2")),
                        scan.ref("col_i16_1").bwOr(1),
                        scan.ref("col_i64nn_1").bwOr((int64_t)1)})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int8_t>(
          {inline_null_value<int8_t>(), 3, 3, inline_null_value<int8_t>()}),
      std::vector<int16_t>({5, 3, 3, 5}),
      std::vector<int32_t>({inline_null_value<int32_t>(), 3, 3, 5}),
      std::vector<int64_t>({5, 3, 3, inline_null_value<int64_t>()}),
      std::vector<int16_t>(
          {inline_null_value<int16_t>(), 3, 3, inline_null_value<int16_t>()}),
      std::vector<int16_t>({5, 3, 3, 5}),
      std::vector<int64_t>({inline_null_value<int64_t>(), 3, 3, 5}),
      std::vector<int32_t>({5, 3, 3, inline_null_value<int32_t>()}),
      std::vector<int32_t>({1, 3, 3, inline_null_value<int32_t>()}),
      std::vector<int64_t>({1, 3, 3, 5}));
}

TEST_F(QueryBuilderTest, BwXor) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  checkBinOper(scan.ref("col_i8_1").bwXor(scan.ref("col_i8_2")),
               ctx().int8(),
               OpType::kBwXor,
               scan.ref("col_i8_1"),
               scan.ref("col_i8_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwXor(scan.ref("col_i16nn_2")),
               ctx().int16(false),
               OpType::kBwXor,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i16nn_2"));
  checkBinOper(scan.ref("col_i32nn_1").bwXor(scan.ref("col_i32_2")),
               ctx().int32(),
               OpType::kBwXor,
               scan.ref("col_i32nn_1"),
               scan.ref("col_i32_2"));
  checkBinOper(scan.ref("col_i64_1").bwXor(scan.ref("col_i64nn_2")),
               ctx().int64(),
               OpType::kBwXor,
               scan.ref("col_i64_1"),
               scan.ref("col_i64nn_2"));
  checkBinOper(scan.ref("col_i8_1").bwXor(scan.ref("col_i16_2")),
               ctx().int16(),
               OpType::kBwXor,
               scan.ref("col_i8_1").cast(ctx().int16()),
               scan.ref("col_i16_2"));
  checkBinOper(scan.ref("col_i16nn_1").bwXor(scan.ref("col_i8nn_2")),
               ctx().int16(false),
               OpType::kBwXor,
               scan.ref("col_i16nn_1"),
               scan.ref("col_i8nn_2").cast(ctx().int16(false)));
  checkBinOper(scan.ref("col_i32nn_1").bwXor(scan.ref("col_i64_2")),
               ctx().int64(),
               OpType::kBwXor,
               scan.ref("col_i32nn_1").cast(ctx().int64(false)),
               scan.ref("col_i64_2"));
  checkBinOper(scan.ref("col_i32_1").bwXor(scan.ref("col_i16nn_2")),
               ctx().int32(),
               OpType::kBwXor,
               scan.ref("col_i32_1"),
               scan.ref("col_i16nn_2").cast(ctx().int32(false)));
  checkBinOper(scan.ref("col_i16_1").bwXor(1),
               ctx().int32(),
               OpType::kBwXor,
               scan.ref("col_i16_1").cast(ctx().int32()),
               builder.cst(1, ctx().int32(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwXor((int64_t)1),
               ctx().int64(false),
               OpType::kBwXor,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  checkBinOper(scan.ref("col_i64nn_1").bwXor(1),
               ctx().int64(false),
               OpType::kBwXor,
               scan.ref("col_i64nn_1"),
               builder.cst(1, ctx().int64(false)));
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_b")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_dec")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_f")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_d")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_str")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_dict")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_date")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_i32_1").bwXor(scan.ref("col_time")), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").bwXor(1), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").bwXor((int64_t)1), InvalidQueryError);
}

TEST_F(QueryBuilderTest, BwXor_Exec) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  auto dag = scan.proj({scan.ref("col_i8_1").bwXor(scan.ref("col_i8_2")),
                        scan.ref("col_i16nn_1").bwXor(scan.ref("col_i16nn_2")),
                        scan.ref("col_i32nn_1").bwXor(scan.ref("col_i32_2")),
                        scan.ref("col_i64_1").bwXor(scan.ref("col_i64nn_2")),
                        scan.ref("col_i8_1").bwXor(scan.ref("col_i16_2")),
                        scan.ref("col_i16nn_1").bwXor(scan.ref("col_i8nn_2")),
                        scan.ref("col_i32nn_1").bwXor(scan.ref("col_i64_2")),
                        scan.ref("col_i32_1").bwXor(scan.ref("col_i16nn_2")),
                        scan.ref("col_i16_1").bwXor(1),
                        scan.ref("col_i64nn_1").bwXor((int64_t)1)})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int8_t>(
          {inline_null_value<int8_t>(), 1, 1, inline_null_value<int8_t>()}),
      std::vector<int16_t>({5, 1, 1, 5}),
      std::vector<int32_t>({inline_null_value<int32_t>(), 1, 1, 5}),
      std::vector<int64_t>({5, 1, 1, inline_null_value<int64_t>()}),
      std::vector<int16_t>(
          {inline_null_value<int16_t>(), 1, 1, inline_null_value<int16_t>()}),
      std::vector<int16_t>({5, 1, 1, 5}),
      std::vector<int64_t>({inline_null_value<int64_t>(), 1, 1, 5}),
      std::vector<int32_t>({5, 1, 1, inline_null_value<int32_t>()}),
      std::vector<int32_t>({0, 3, 2, inline_null_value<int32_t>()}),
      std::vector<int64_t>({0, 3, 2, 5}));
}

TEST_F(QueryBuilderTest, BwNot) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  checkUOper(
      scan.ref("col_i8_1").bwNot(), ctx().int8(), OpType::kBwNot, scan.ref("col_i8_1"));
  checkUOper(scan.ref("col_i8nn_1").bwNot(),
             ctx().int8(false),
             OpType::kBwNot,
             scan.ref("col_i8nn_1"));
  checkUOper(scan.ref("col_i16_1").bwNot(),
             ctx().int16(),
             OpType::kBwNot,
             scan.ref("col_i16_1"));
  checkUOper(scan.ref("col_i16nn_1").bwNot(),
             ctx().int16(false),
             OpType::kBwNot,
             scan.ref("col_i16nn_1"));
  checkUOper(scan.ref("col_i32_1").bwNot(),
             ctx().int32(),
             OpType::kBwNot,
             scan.ref("col_i32_1"));
  checkUOper(scan.ref("col_i32nn_1").bwNot(),
             ctx().int32(false),
             OpType::kBwNot,
             scan.ref("col_i32nn_1"));
  checkUOper(scan.ref("col_i64_1").bwNot(),
             ctx().int64(),
             OpType::kBwNot,
             scan.ref("col_i64_1"));
  checkUOper(scan.ref("col_i64nn_1").bwNot(),
             ctx().int64(false),
             OpType::kBwNot,
             scan.ref("col_i64nn_1"));
  checkUOper(scan.ref("col_i64nn_1").bwNot(),
             ctx().int64(false),
             OpType::kBwNot,
             scan.ref("col_i64nn_1"));
  checkCst(
      builder.cst(8, ctx().int32(false)).bwNot(), (int)0xfffffff7, ctx().int32(false));
  checkCst(builder.cst(8, ctx().int64(false)).bwNot(),
           (int64_t)0xfffffffffffffff7LL,
           ctx().int64(false));
  EXPECT_THROW(scan.ref("col_dec").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_b").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_f").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_d").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_str").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_dict").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_date").bwNot(), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").bwNot(), InvalidQueryError);
}

TEST_F(QueryBuilderTest, BwNot_Exec) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_bitwise");
  auto dag = scan.proj({scan.ref("col_i8_1").bwNot(),
                        scan.ref("col_i16nn_1").bwNot(),
                        scan.ref("col_i32nn_1").bwNot(),
                        scan.ref("col_i64_1").bwNot()})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int8_t>(
          {(int8_t)0xfe, (int8_t)0xfd, (int8_t)0xfc, inline_null_value<int8_t>()}),
      std::vector<int16_t>(
          {(int16_t)0xfffe, (int16_t)0xfffd, (int16_t)0xfffc, (int16_t)0xfffb}),
      std::vector<int32_t>({(int32_t)0xfffffffe,
                            (int32_t)0xfffffffd,
                            (int32_t)0xfffffffc,
                            (int32_t)0xfffffffb}),
      std::vector<int64_t>({(int64_t)0xfffffffffffffffeLL,
                            (int64_t)0xfffffffffffffffdLL,
                            (int64_t)0xfffffffffffffffcLL,
                            inline_null_value<int64_t>()}));
}

TEST_F(QueryBuilderTest, Cardinality) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_cardinality");
  checkCardinality(scan.ref("col_arr_i32").cardinality(), scan.ref("col_arr_i32"));
  checkCardinality(scan.ref("col_arr_i32x2").cardinality(), scan.ref("col_arr_i32x2"));
  checkCardinality(scan.ref("col_arr_i32nn").cardinality(), scan.ref("col_arr_i32nn"));
  checkCst(scan.ref("col_arr_i32x2nn").cardinality(), 2, ctx().int32(false));
  checkCst(builder.cst({1, 2, 3, 4}, "array(int8)").cardinality(), 4, ctx().int32(false));
  checkNullCst(builder.nullCst("array(int8)").cardinality(), ctx().int32());
  auto scan2 = builder.scan("test3");
  EXPECT_THROW(scan2.ref("col_i").cardinality(), InvalidQueryError);
  EXPECT_THROW(scan2.ref("col_d").cardinality(), InvalidQueryError);
  EXPECT_THROW(scan2.ref("col_dec").cardinality(), InvalidQueryError);
  EXPECT_THROW(scan2.ref("col_time").cardinality(), InvalidQueryError);
  EXPECT_THROW(scan2.ref("col_str").cardinality(), InvalidQueryError);
  EXPECT_THROW(builder.cst(1).cardinality(), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Cardinality_Exec) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test_cardinality");
  auto dag = scan.proj({scan.ref("col_arr_i32").cardinality(),
                        scan.ref("col_arr_i32x2").cardinality(),
                        scan.ref("col_arr_i32nn").cardinality(),
                        scan.ref("col_arr_i32x2nn").cardinality()})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int32_t>({1, inline_null_value<int32_t>(), 0, 2}),
      std::vector<int32_t>(
          {2, inline_null_value<int32_t>(), 2, inline_null_value<int32_t>()}),
      std::vector<int32_t>({2, 1, 0, 1}),
      std::vector<int32_t>({2, 2, 2, 2}));
}

TEST_F(QueryBuilderTest, TopK_Exec_PerfectHash) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg({"id1"s},
                        {"topk(i8, 2)"s,
                         "topk(i8nn, 2)"s,
                         "topk(i16, 2)"s,
                         "topk(i16nn, 2)"s,
                         "topk(i32, 2)"s,
                         "topk(i32nn, 2)"s,
                         "topk(i64, 2)"s,
                         "topk(i64nn, 2)"s,
                         "topk(f32, 2)"s,
                         "topk(f32nn, 2)"s,
                         "topk(f64, 2)"s,
                         "topk(f64nn, 2)"s})
                   .sort({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(
        res,
        std::vector<int32_t>({1, 2, 3, 4, 5, 6}),
        std::vector<std::vector<int8_t>>({{5, 4}, {5, 4}, {4, 3}, {2}, {3}, {}}),
        std::vector<std::vector<int8_t>>({{2, 2}, {5, 4}, {5, 4}, {1}, {1}, {2}}),
        std::vector<std::vector<int16_t>>({{50, 40}, {50, 40}, {40, 30}, {20}, {30}, {}}),
        std::vector<std::vector<int16_t>>(
            {{20, 20}, {50, 40}, {50, 40}, {10}, {10}, {20}}),
        std::vector<std::vector<int32_t>>(
            {{500, 400}, {500, 400}, {400, 300}, {200}, {300}, {}}),
        std::vector<std::vector<int32_t>>(
            {{200, 200}, {500, 400}, {500, 400}, {100}, {100}, {200}}),
        std::vector<std::vector<int64_t>>(
            {{5000, 4000}, {5000, 4000}, {4000, 3000}, {2000}, {3000}, {}}),
        std::vector<std::vector<int64_t>>(
            {{2000, 2000}, {5000, 4000}, {5000, 4000}, {1000}, {1000}, {2000}}),
        std::vector<std::vector<float>>(
            {{5.5, 4.4}, {5.5, 4.4}, {4.4, 3.3}, {2.2}, {3.3}, {}}),
        std::vector<std::vector<float>>(
            {{2.2, 2.2}, {5.5, 4.4}, {5.5, 4.4}, {1.1}, {1.1}, {2.2}}),
        std::vector<std::vector<double>>(
            {{5.55, 4.44}, {5.55, 4.44}, {4.44, 3.33}, {2.22}, {3.33}, {}}),
        std::vector<std::vector<double>>(
            {{2.22, 2.22}, {5.55, 4.44}, {5.55, 4.44}, {1.11}, {1.11}, {2.22}}));
  }
}

TEST_F(QueryBuilderTest, TopK_Exec_BaselineHash) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg({"id2"s},
                        {"topk(i8, 3)"s,
                         "topk(i8nn, 3)"s,
                         "topk(i16, 3)"s,
                         "topk(i16nn, 3)"s,
                         "topk(i32, 3)"s,
                         "topk(i32nn, 3)"s,
                         "topk(i64, 3)"s,
                         "topk(i64nn, 3)"s,
                         "topk(f32, 3)"s,
                         "topk(f32nn, 3)"s,
                         "topk(f64, 3)"s,
                         "topk(f64nn, 3)"s})
                   .sort({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(
        res,
        std::vector<int64_t>({10000000000ULL,
                              20000000000ULL,
                              30000000000ULL,
                              40000000000ULL,
                              50000000000ULL,
                              60000000000ULL}),
        std::vector<std::vector<int8_t>>({{5, 4, 2}, {5, 4, 1}, {4, 3, 1}, {2}, {3}, {}}),
        std::vector<std::vector<int8_t>>(
            {{2, 2, 1}, {5, 4, 1}, {5, 4, 3}, {1}, {1}, {2}}),
        std::vector<std::vector<int16_t>>(
            {{50, 40, 20}, {50, 40, 10}, {40, 30, 10}, {20}, {30}, {}}),
        std::vector<std::vector<int16_t>>(
            {{20, 20, 10}, {50, 40, 10}, {50, 40, 30}, {10}, {10}, {20}}),
        std::vector<std::vector<int32_t>>(
            {{500, 400, 200}, {500, 400, 100}, {400, 300, 100}, {200}, {300}, {}}),
        std::vector<std::vector<int32_t>>(
            {{200, 200, 100}, {500, 400, 100}, {500, 400, 300}, {100}, {100}, {200}}),
        std::vector<std::vector<int64_t>>({{5000, 4000, 2000},
                                           {5000, 4000, 1000},
                                           {4000, 3000, 1000},
                                           {2000},
                                           {3000},
                                           {}}),
        std::vector<std::vector<int64_t>>({{2000, 2000, 1000},
                                           {5000, 4000, 1000},
                                           {5000, 4000, 3000},
                                           {1000},
                                           {1000},
                                           {2000}}),
        std::vector<std::vector<float>>(
            {{5.5, 4.4, 2.2}, {5.5, 4.4, 1.1}, {4.4, 3.3, 1.1}, {2.2}, {3.3}, {}}),
        std::vector<std::vector<float>>(
            {{2.2, 2.2, 1.1}, {5.5, 4.4, 1.1}, {5.5, 4.4, 3.3}, {1.1}, {1.1}, {2.2}}),
        std::vector<std::vector<double>>({{5.55, 4.44, 2.22},
                                          {5.55, 4.44, 1.11},
                                          {4.44, 3.33, 1.11},
                                          {2.22},
                                          {3.33},
                                          {}}),
        std::vector<std::vector<double>>({{2.22, 2.22, 1.11},
                                          {5.55, 4.44, 1.11},
                                          {5.55, 4.44, 3.33},
                                          {1.11},
                                          {1.11},
                                          {2.22}}));
  }
}

TEST_F(QueryBuilderTest, TopK_Exec_NoGroupBy) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg(std::vector<int>(),
                        {"topk(i8, 5)"s,
                         "topk(i8nn, 5)"s,
                         "topk(i16, 5)"s,
                         "topk(i16nn, 5)"s,
                         "topk(i32, 5)"s,
                         "topk(i32nn, 5)"s,
                         "topk(i64, 5)"s,
                         "topk(i64nn, 5)"s,
                         "topk(f32, 5)"s,
                         "topk(f32nn, 5)"s,
                         "topk(f64, 5)"s,
                         "topk(f64nn, 5)"s})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<std::vector<int8_t>>({{5, 5, 4, 4, 4}}),
                     std::vector<std::vector<int8_t>>({{5, 5, 4, 4, 3}}),
                     std::vector<std::vector<int16_t>>({{50, 50, 40, 40, 40}}),
                     std::vector<std::vector<int16_t>>({{50, 50, 40, 40, 30}}),
                     std::vector<std::vector<int32_t>>({{500, 500, 400, 400, 400}}),
                     std::vector<std::vector<int32_t>>({{500, 500, 400, 400, 300}}),
                     std::vector<std::vector<int64_t>>({{5000, 5000, 4000, 4000, 4000}}),
                     std::vector<std::vector<int64_t>>({{5000, 5000, 4000, 4000, 3000}}),
                     std::vector<std::vector<float>>({{5.5, 5.5, 4.4, 4.4, 4.4}}),
                     std::vector<std::vector<float>>({{5.5, 5.5, 4.4, 4.4, 3.3}}),
                     std::vector<std::vector<double>>({{5.55, 5.55, 4.44, 4.44, 4.44}}),
                     std::vector<std::vector<double>>({{5.55, 5.55, 4.44, 4.44, 3.33}}));
  }
}

TEST_F(QueryBuilderTest, BottomK_Exec_PerfectHash) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg({"id1"s},
                        {"bottomk(i8, 2)"s,
                         "bottomk(i8nn, 2)"s,
                         "bottomk(i16, 2)"s,
                         "bottomk(i16nn, 2)"s,
                         "bottomk(i32, 2)"s,
                         "bottomk(i32nn, 2)"s,
                         "bottomk(i64, 2)"s,
                         "bottomk(i64nn, 2)"s,
                         "bottomk(f32, 2)"s,
                         "bottomk(f32nn, 2)"s,
                         "bottomk(f64, 2)"s,
                         "bottomk(f64nn, 2)"s})
                   .sort({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(
        res,
        std::vector<int32_t>({1, 2, 3, 4, 5, 6}),
        std::vector<std::vector<int8_t>>({{1, 2}, {1, 4}, {1, 3}, {2}, {3}, {}}),
        std::vector<std::vector<int8_t>>({{0, 1}, {0, 1}, {0, 3}, {1}, {1}, {2}}),
        std::vector<std::vector<int16_t>>({{10, 20}, {10, 40}, {10, 30}, {20}, {30}, {}}),
        std::vector<std::vector<int16_t>>({{0, 10}, {0, 10}, {0, 30}, {10}, {10}, {20}}),
        std::vector<std::vector<int32_t>>(
            {{100, 200}, {100, 400}, {100, 300}, {200}, {300}, {}}),
        std::vector<std::vector<int32_t>>(
            {{0, 100}, {0, 100}, {0, 300}, {100}, {100}, {200}}),
        std::vector<std::vector<int64_t>>(
            {{1000, 2000}, {1000, 4000}, {1000, 3000}, {2000}, {3000}, {}}),
        std::vector<std::vector<int64_t>>(
            {{0, 1000}, {0, 1000}, {0, 3000}, {1000}, {1000}, {2000}}),
        std::vector<std::vector<float>>(
            {{1.1, 2.2}, {1.1, 4.4}, {1.1, 3.3}, {2.2}, {3.3}, {}}),
        std::vector<std::vector<float>>(
            {{0.0, 1.1}, {0.0, 1.1}, {0.0, 3.3}, {1.1}, {1.1}, {2.2}}),
        std::vector<std::vector<double>>(
            {{1.11, 2.22}, {1.11, 4.44}, {1.11, 3.33}, {2.22}, {3.33}, {}}),
        std::vector<std::vector<double>>(
            {{0.00, 1.11}, {0.00, 1.11}, {0.00, 3.33}, {1.11}, {1.11}, {2.22}}));
  }
}

TEST_F(QueryBuilderTest, BottomK_Exec_BaselineHash) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg({"id2"s},
                        {"bottomk(i8, 3)"s,
                         "bottomk(i8nn, 3)"s,
                         "bottomk(i16, 3)"s,
                         "bottomk(i16nn, 3)"s,
                         "bottomk(i32, 3)"s,
                         "bottomk(i32nn, 3)"s,
                         "bottomk(i64, 3)"s,
                         "bottomk(i64nn, 3)"s,
                         "bottomk(f32, 3)"s,
                         "bottomk(f32nn, 3)"s,
                         "bottomk(f64, 3)"s,
                         "bottomk(f64nn, 3)"s})
                   .sort({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(
        res,
        std::vector<int64_t>({10000000000ULL,
                              20000000000ULL,
                              30000000000ULL,
                              40000000000ULL,
                              50000000000ULL,
                              60000000000ULL}),
        std::vector<std::vector<int8_t>>({{1, 2, 4}, {1, 4, 5}, {1, 3, 4}, {2}, {3}, {}}),
        std::vector<std::vector<int8_t>>(
            {{0, 1, 1}, {0, 1, 4}, {0, 3, 4}, {1}, {1}, {2}}),
        std::vector<std::vector<int16_t>>(
            {{10, 20, 40}, {10, 40, 50}, {10, 30, 40}, {20}, {30}, {}}),
        std::vector<std::vector<int16_t>>(
            {{0, 10, 10}, {0, 10, 40}, {0, 30, 40}, {10}, {10}, {20}}),
        std::vector<std::vector<int32_t>>(
            {{100, 200, 400}, {100, 400, 500}, {100, 300, 400}, {200}, {300}, {}}),
        std::vector<std::vector<int32_t>>(
            {{0, 100, 100}, {0, 100, 400}, {0, 300, 400}, {100}, {100}, {200}}),
        std::vector<std::vector<int64_t>>({{1000, 2000, 4000},
                                           {1000, 4000, 5000},
                                           {1000, 3000, 4000},
                                           {2000},
                                           {3000},
                                           {}}),
        std::vector<std::vector<int64_t>>(
            {{0, 1000, 1000}, {0, 1000, 4000}, {0, 3000, 4000}, {1000}, {1000}, {2000}}),
        std::vector<std::vector<float>>(
            {{1.1, 2.2, 4.4}, {1.1, 4.4, 5.5}, {1.1, 3.3, 4.4}, {2.2}, {3.3}, {}}),
        std::vector<std::vector<float>>(
            {{0.0, 1.1, 1.1}, {0.0, 1.1, 4.4}, {0.0, 3.3, 4.4}, {1.1}, {1.1}, {2.2}}),
        std::vector<std::vector<double>>({{1.11, 2.22, 4.44},
                                          {1.11, 4.44, 5.55},
                                          {1.11, 3.33, 4.44},
                                          {2.22},
                                          {3.33},
                                          {}}),
        std::vector<std::vector<double>>({{0.00, 1.11, 1.11},
                                          {0.00, 1.11, 4.44},
                                          {0.00, 3.33, 4.44},
                                          {1.11},
                                          {1.11},
                                          {2.22}}));
  }
}

TEST_F(QueryBuilderTest, BottomK_Exec_NoGroupBy) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());
    auto dag = builder.scan("test_topk")
                   .agg(std::vector<int>(),
                        {"bottomk(i8, 5)"s,
                         "bottomk(i8nn, 5)"s,
                         "bottomk(i16, 5)"s,
                         "bottomk(i16nn, 5)"s,
                         "bottomk(i32, 5)"s,
                         "bottomk(i32nn, 5)"s,
                         "bottomk(i64, 5)"s,
                         "bottomk(i64nn, 5)"s,
                         "bottomk(f32, 5)"s,
                         "bottomk(f32nn, 5)"s,
                         "bottomk(f64, 5)"s,
                         "bottomk(f64nn, 5)"s})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<std::vector<int8_t>>({{1, 1, 1, 2, 2}}),
                     std::vector<std::vector<int8_t>>({{0, 0, 0, 1, 1}}),
                     std::vector<std::vector<int16_t>>({{10, 10, 10, 20, 20}}),
                     std::vector<std::vector<int16_t>>({{0, 0, 0, 10, 10}}),
                     std::vector<std::vector<int32_t>>({{100, 100, 100, 200, 200}}),
                     std::vector<std::vector<int32_t>>({{0, 0, 0, 100, 100}}),
                     std::vector<std::vector<int64_t>>({{1000, 1000, 1000, 2000, 2000}}),
                     std::vector<std::vector<int64_t>>({{0, 0, 0, 1000, 1000}}),
                     std::vector<std::vector<float>>({{1.1, 1.1, 1.1, 2.2, 2.2}}),
                     std::vector<std::vector<float>>({{0.0, 0.0, 0.0, 1.1, 1.1}}),
                     std::vector<std::vector<double>>({{1.11, 1.11, 1.11, 2.22, 2.22}}),
                     std::vector<std::vector<double>>({{0.00, 0.00, 0.00, 1.11, 1.11}}));
  }
}

TEST_F(QueryBuilderTest, TopK_Unnest) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto sort = builder.scan("test_topk").agg({"id1"s}, {"topk(i32, 2)"s}).sort({0});
      auto dag = sort.proj({sort.ref("id1"), sort.ref("i32_top_2").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 2, 2, 3, 3, 4, 5}),
                       std::vector<int32_t>({500, 400, 500, 400, 400, 300, 200, 300}));
    }

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto sort = builder.scan("test_topk").agg({"id2"s}, {"topk(i64, 2)"s}).sort({0});
      auto dag = sort.proj({sort.ref("id2"), sort.ref("i64_top_2").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int64_t>({10000000000ULL,
                                10000000000ULL,
                                20000000000ULL,
                                20000000000ULL,
                                30000000000ULL,
                                30000000000ULL,
                                40000000000ULL,
                                50000000000ULL}),
          std::vector<int64_t>({5000, 4000, 5000, 4000, 4000, 3000, 2000, 3000}));
    }

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto agg = builder.scan("test_topk").agg({"id1"s}, {"topk(i32, 2)"s});
      auto dag = agg.proj({agg.ref("id1"), agg.ref("i32_top_2").unnest()})
                     .sort({BuilderSortField{0, "asc"}, BuilderSortField{1, "desc"}})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 2, 2, 3, 3, 4, 5}),
                       std::vector<int32_t>({500, 400, 500, 400, 400, 300, 200, 300}));
    }

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto agg = builder.scan("test_topk").agg({"id2"s}, {"topk(i64, 2)"s});
      auto dag = agg.proj({agg.ref("id2"), agg.ref("i64_top_2").unnest()})
                     .sort({BuilderSortField{0, "asc"}, BuilderSortField{1, "desc"}})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int64_t>({10000000000ULL,
                                10000000000ULL,
                                20000000000ULL,
                                20000000000ULL,
                                30000000000ULL,
                                30000000000ULL,
                                40000000000ULL,
                                50000000000ULL}),
          std::vector<int64_t>({5000, 4000, 5000, 4000, 4000, 3000, 2000, 3000}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Exec) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag = scan.agg({"id1"s},
                          {scan.ref("i8").quantile(0.5),
                           scan.ref("i8nn").quantile(0.5, Interpolation::kLower),
                           scan.ref("i16").quantile(0.4),
                           scan.ref("i16nn").quantile(0.5, Interpolation::kHigher),
                           scan.ref("i32").quantile(0.6),
                           scan.ref("i32nn").quantile(0.4, Interpolation::kNearest),
                           scan.ref("i64").quantile(0.3),
                           scan.ref("i64nn").quantile(0.4, Interpolation::kMidpoint),
                           scan.ref("f32").quantile(0.5),
                           scan.ref("f32nn").quantile(0.5, Interpolation::kLinear),
                           scan.ref("f64").quantile(0.0),
                           scan.ref("f64nn").quantile(1.0),
                           scan.ref("dec").quantile(0.6),
                           scan.ref("decnn").quantile(0.5, Interpolation::kLower)})
                     .sort(0)
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int32_t>({1, 2, 3, 4}),
          std::vector<double>({5.0, 2.0, 1.5, inline_null_value<double>()}),
          std::vector<int8_t>({5, 2, 2, 1}),
          std::vector<double>({42.0, 18.0, 14.0, inline_null_value<double>()}),
          std::vector<int16_t>({50, 30, 20, 10}),
          std::vector<double>({580.0, 220.0, 160.0, inline_null_value<double>()}),
          std::vector<int32_t>({400, 200, 200, 100}),
          std::vector<double>({3400.0, 1600.0, 1300.0, inline_null_value<double>()}),
          std::vector<double>({4500.0, 2500.0, 1500.0, 1000.0}),
          std::vector<float>({5.0, 2.0, 1.5, inline_null_value<float>()}),
          std::vector<float>({5.0, 2.5, 2.0, 1.0}),
          std::vector<double>({10.0, 10.0, 10.0, inline_null_value<double>()}),
          std::vector<double>({90.0, 40.0, 30.0, 10.0}),
          std::vector<int64_t>({58000, 22000, 16000, inline_null_value<int64_t>()}),
          std::vector<int64_t>({50000, 20000, 20000, 10000}));
    }

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag = scan.agg({"id2"s},
                          {scan.ref("i8").quantile(0.5),
                           scan.ref("i8nn").quantile(0.5, Interpolation::kLower),
                           scan.ref("i16").quantile(0.4),
                           scan.ref("i16nn").quantile(0.5, Interpolation::kHigher),
                           scan.ref("i32").quantile(0.6),
                           scan.ref("i32nn").quantile(0.4, Interpolation::kNearest),
                           scan.ref("i64").quantile(0.3),
                           scan.ref("i64nn").quantile(0.4, Interpolation::kMidpoint),
                           scan.ref("f32").quantile(0.5),
                           scan.ref("f32nn").quantile(0.5, Interpolation::kLinear),
                           scan.ref("f64").quantile(0.0),
                           scan.ref("f64nn").quantile(1.0),
                           scan.ref("dec").quantile(0.6),
                           scan.ref("decnn").quantile(0.5, Interpolation::kLower)})
                     .sort(0)
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int64_t>(
              {10000000000LL, 20000000000LL, 30000000000LL, 40000000000LL}),
          std::vector<double>({5.0, 2.0, 1.5, inline_null_value<double>()}),
          std::vector<int8_t>({5, 2, 2, 1}),
          std::vector<double>({42.0, 18.0, 14.0, inline_null_value<double>()}),
          std::vector<int16_t>({50, 30, 20, 10}),
          std::vector<double>({580.0, 220.0, 160.0, inline_null_value<double>()}),
          std::vector<int32_t>({400, 200, 200, 100}),
          std::vector<double>({3400.0, 1600.0, 1300.0, inline_null_value<double>()}),
          std::vector<double>({4500.0, 2500.0, 1500.0, 1000.0}),
          std::vector<float>({5.0, 2.0, 1.5, inline_null_value<float>()}),
          std::vector<float>({5.0, 2.5, 2.0, 1.0}),
          std::vector<double>({10.0, 10.0, 10.0, inline_null_value<double>()}),
          std::vector<double>({90.0, 40.0, 30.0, 10.0}),
          std::vector<int64_t>({58000, 22000, 16000, inline_null_value<int64_t>()}),
          std::vector<int64_t>({50000, 20000, 20000, 10000}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Exec_No_GroupBy) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag = scan.agg(std::vector<std::string>(),
                          {scan.ref("i8").quantile(0.5),
                           scan.ref("i16").quantile(0.4, Interpolation::kLower),
                           scan.ref("i32").quantile(0.6, Interpolation::kHigher),
                           scan.ref("i32").quantile(0.2, Interpolation::kLinear),
                           scan.ref("i64").quantile(0.3, Interpolation::kNearest),
                           scan.ref("f32").quantile(0.7, Interpolation::kMidpoint),
                           scan.ref("f64").quantile(0.0),
                           scan.ref("dec").quantile(1.0)})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<double>({3.0}),
                       std::vector<int16_t>({20}),
                       std::vector<int32_t>({400}),
                       std::vector<double>({160.0}),
                       std::vector<int64_t>({2000}),
                       std::vector<float>({5.5}),
                       std::vector<double>({10.0}),
                       std::vector<int64_t>({90000}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Datetime_Exec) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile_dt");
      auto dag = scan.agg({"id1"s},
                          {scan.ref("t64s").quantile(0.5),
                           scan.ref("t64s2").quantile(0.5, Interpolation::kLower),
                           scan.ref("d32").quantile(0.4, Interpolation::kHigher),
                           scan.ref("d64").quantile(0.4, Interpolation::kMidpoint),
                           scan.ref("ts64").quantile(0.6, Interpolation::kNearest),
                           scan.ref("ts64nn").quantile(0.6, Interpolation::kLinear)})
                     .sort(0)
                     .finalize();
      auto res = runQuery(std::move(dag));
      int64_t ms_per_day = 86400 * 1000;
      compare_res_data(
          res,
          std::vector<int32_t>({1, 2, 3, 4}),
          std::vector<int64_t>({5, 2, 1, inline_null_value<int64_t>()}),
          std::vector<int64_t>({50, 20, 20, 10}),
          std::vector<int64_t>({5 * ms_per_day,
                                2 * ms_per_day,
                                2 * ms_per_day,
                                inline_null_value<int64_t>()}),
          std::vector<int64_t>(
              {45 * ms_per_day, 25 * ms_per_day, 15 * ms_per_day, 10 * ms_per_day}),
          std::vector<int64_t>({6, 2, 2, inline_null_value<int64_t>()}),
          std::vector<int64_t>({58000, 28000, 22000, 10000}));
    }

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile_dt");
      auto dag = scan.agg({"id2"s},
                          {scan.ref("t64s").quantile(0.5),
                           scan.ref("t64s2").quantile(0.5, Interpolation::kLower),
                           scan.ref("d32").quantile(0.4, Interpolation::kHigher),
                           scan.ref("d64").quantile(0.4, Interpolation::kMidpoint),
                           scan.ref("ts64").quantile(0.6, Interpolation::kNearest),
                           scan.ref("ts64nn").quantile(0.6, Interpolation::kLinear)})
                     .sort(0)
                     .finalize();
      auto res = runQuery(std::move(dag));
      int64_t ms_per_day = 86400 * 1000;
      compare_res_data(
          res,
          std::vector<int64_t>(
              {10000000000LL, 20000000000LL, 30000000000LL, 40000000000LL}),
          std::vector<int64_t>({5, 2, 1, inline_null_value<int64_t>()}),
          std::vector<int64_t>({50, 20, 20, 10}),
          std::vector<int64_t>({5 * ms_per_day,
                                2 * ms_per_day,
                                2 * ms_per_day,
                                inline_null_value<int64_t>()}),
          std::vector<int64_t>(
              {45 * ms_per_day, 25 * ms_per_day, 15 * ms_per_day, 10 * ms_per_day}),
          std::vector<int64_t>({6, 2, 2, inline_null_value<int64_t>()}),
          std::vector<int64_t>({58000, 28000, 22000, 10000}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Exec_Sort) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag = scan.agg({"id1"s}, {scan.ref("f32").quantile(0.5)}).sort(1).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({3, 2, 1, 4}),
                       std::vector<float>({1.5, 2.0, 5.0, inline_null_value<float>()}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Exec_Fetch) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag1 = scan.agg({"id1"s}, {scan.ref("f32").quantile(0.5)}).sort(0).finalize();
      auto res1 = runQuery(std::move(dag1));

      auto dag2 = builder.scan(res1.tableName()).proj({0, 1}).finalize();
      auto res2 = runQuery(std::move(dag2));
      compare_res_data(res2,
                       std::vector<int32_t>({1, 2, 3, 4}),
                       std::vector<float>({5.0, 2.0, 1.5, inline_null_value<float>()}));
    }
  }
}

TEST_F(QueryBuilderTest, Quantile_Exec_Stddev) {
  for (bool enable_columnar : {false, true}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    {
      QueryBuilder builder(ctx(), schema_mgr_, configPtr());
      auto scan = builder.scan("test_quantile");
      auto dag =
          scan.agg({"id1"s}, {scan.ref("f32").quantile(0.5), scan.ref("f32").stdDev()})
              .sort(0)
              .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int32_t>({1, 2, 3, 4}),
          std::vector<float>({5.0, 2.0, 1.5, inline_null_value<float>()}),
          std::vector<double>({2.73861, 1.0, 0.707107, inline_null_value<double>()}));
    }
  }
}

TEST_F(QueryBuilderTest, SimpleProjection) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  compare_test1_data(builder.scan("test1").proj({0, 1, 2, 3}));
  compare_test1_data(builder.scan("test1").proj({0, 1, 2, 3, 4}), {0, 1, 2, 3, 4});
  compare_test1_data(builder.scan("test1").proj({"col_bi", "col_i", "col_f", "col_d"}));
  compare_test1_data(
      builder.scan("test1").proj({"col_bi", "col_i", "col_f", "col_d", "rowid"}),
      {0, 1, 2, 3, 4});
  compare_test1_data(builder.scan("test1").proj({1, 0}), {1, 0});
  compare_test1_data(
      builder.scan("test1").proj({1, 0}, {"c1", "c2"}), {1, 0}, {"c1", "c2"});
  compare_test1_data(builder.scan("test1").proj(std::vector<int>({1, 0})), {1, 0});
  compare_test1_data(builder.scan("test1").proj(std::vector<int>({1, 0}), {"c", "d"}),
                     {1, 0},
                     {"c", "d"});
  compare_test1_data(builder.scan("test1").proj({"col_i", "col_bi"}), {1, 0});
  compare_test1_data(builder.scan("test1").proj({"col_i", "col_bi"}, {"c4", "c5"}),
                     {1, 0},
                     {"c4", "c5"});
  compare_test1_data(
      builder.scan("test1").proj(std::vector<std::string>({"col_i", "col_bi"})), {1, 0});
  compare_test1_data(builder.scan("test1").proj(
                         std::vector<std::string>({"col_i", "col_bi"}), {"cc", "cd"}),
                     {1, 0},
                     {"cc", "cd"});
  compare_test1_data(builder.scan("test1").proj({"col_i", "col_bi", "rowid"}), {1, 0, 4});
  compare_test1_data(builder.scan("test1").proj(0), {0});
  compare_test1_data(builder.scan("test1").proj(2), {2});
  compare_test1_data(builder.scan("test1").proj(4), {4});
  compare_test1_data(builder.scan("test1").proj(0, "col"), {0}, {"col"});
  compare_test1_data(builder.scan("test1").proj("col_bi"), {0});
  compare_test1_data(builder.scan("test1").proj("col_i"), {1});
  compare_test1_data(builder.scan("test1").proj("rowid"), {4});
  compare_test1_data(builder.scan("test1").proj("col_bi", "col"), {0}, {"col"});
  compare_test1_data(
      builder.scan("test1").proj({"col_i", "col_i"}), {1, 1}, {"col_i", "col_i_1"});
  compare_test1_data(
      builder.scan("test1").proj({1, 1, 1}), {1, 1, 1}, {"col_i", "col_i_1", "col_i_2"});
  compare_test1_data(builder.scan("test1").proj(-1), {4});
  compare_test1_data(builder.scan("test1").proj(-2), {3});
  compare_test1_data(builder.scan("test1").proj({1, -2}), {1, 3});
  compare_test1_data(builder.scan("test1").proj(std::vector<int>({1, -2})), {1, 3});
  auto scan1 = builder.scan("test1");
  compare_test1_data(scan1.proj(scan1.ref(2)), {2});
  compare_test1_data(scan1.proj(scan1.ref(2).rename("c2")), {2}, {"c2"});
  compare_test1_data(scan1.proj({scan1.ref(1)}), {1});
  compare_test1_data(scan1.proj({scan1.ref(1).rename("c1")}), {1}, {"c1"});
  compare_test1_data(scan1.proj({scan1.ref(-2)}), {3});
  compare_test1_data(scan1.proj({scan1.ref(-2).rename("c")}), {3}, {"c"});
  compare_test1_data(scan1.proj(std::vector<BuilderExpr>({scan1.ref(3)})), {3});
  compare_test1_data(
      scan1.proj(std::vector<BuilderExpr>({scan1.ref(3).rename("c1")})), {3}, {"c1"});
  compare_test1_data(scan1.proj(scan1.ref("col_i")), {1});
  compare_test1_data(
      scan1.proj({scan1.ref("col_i"), scan1.ref("col_i")}), {1, 1}, {"col_i", "col_i_1"});

  EXPECT_THROW(builder.scan("test1").proj("unknown"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({"unknown"}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(20), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({20}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(-10), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({-10}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({1, 2}, {"c1", "c1"}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({"col_bi", "col_i"}, {"c1", "c1"}),
               InvalidQueryError);
  EXPECT_THROW(
      scan1.proj({scan1.ref("col_bi").rename("c"), scan1.ref("col_i").rename("c")}),
      InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(scan1.ref(0)), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({0, 1}, {"c1", "c2", "c3"}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({"col_bi"}, {}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(std::vector<int>(), {}), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ScanFilter) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test1");
  auto dag = scan.filter(scan.ref("col_bi").gt(3)).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({4, 5}),
                   std::vector<int32_t>({44, 55}),
                   std::vector<float>({4.4f, 5.5f}),
                   std::vector<double>({44.44f, 55.55f}));
}

TEST_F(QueryBuilderTest, ProjFilter) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test1");
  auto filter = scan.filter(scan.ref("col_bi").gt(3));
  auto dag = filter.proj({filter.ref("col_i").add(1), filter.ref("col_f")}).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int32_t>({45, 56}), std::vector<float>({4.4f, 5.5f}));
}

TEST_F(QueryBuilderTest, FilterProj) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test1");
  auto proj = scan.proj({scan["col_bi"], (scan["col_i"] + 1).rename("inc")});
  auto dag = proj.filter(proj["inc"] > 40).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int64_t>({4, 5}), std::vector<int32_t>({45, 56}));
}

TEST_F(QueryBuilderTest, Aggregate) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  compare_test2_agg(
      builder.scan("test2").agg(0, "count"), {"id1"}, {"COUNT(*)"}, {"id1", "count"});
  compare_test2_agg(
      builder.scan("test2").agg(0, {"count()"}), {"id1"}, {"COUNT(*)"}, {"id1", "count"});
  compare_test2_agg(builder.scan("test2").agg(0, std::vector<std::string>{"count(*)"}),
                    {"id1"},
                    {"COUNT(*)"},
                    {"id1", "count"});
  auto scan = builder.scan("test2");
  compare_test2_agg(scan.agg(1, scan.count()), {"id2"}, {"COUNT(*)"}, {"id2", "count"});
  compare_test2_agg(
      scan.agg(-4, {scan.count()}), {"id2"}, {"COUNT(*)"}, {"id2", "count"});
  compare_test2_agg(
      builder.scan("test2").agg("id1", "COUNT"), {"id1"}, {"COUNT(*)"}, {"id1", "count"});
  compare_test2_agg(builder.scan("test2").agg("id1", {"COUNT()"}),
                    {"id1"},
                    {"COUNT(*)"},
                    {"id1", "count"});
  compare_test2_agg(
      builder.scan("test2").agg("id1", std::vector<std::string>{"CoUnT(1)"}),
      {"id1"},
      {"COUNT(*)"},
      {"id1", "count"});
  compare_test2_agg(scan.agg("id2", scan.ref("val1").count()),
                    {"id2"},
                    {"COUNT( val1)"},
                    {"id2", "val1_count"});
  compare_test2_agg(scan.agg("id2", {scan.ref("val1").count()}),
                    {"id2"},
                    {"COUNT(val1)"},
                    {"id2", "val1_count"});
  compare_test2_agg(scan.agg(scan.ref("id1"), "count(val1)"),
                    {"id1"},
                    {"COUNT(val1)"},
                    {"id1", "val1_count"});
  compare_test2_agg(scan.agg(scan.ref("id1"), {"COUNT(val1)"}),
                    {"id1"},
                    {"COUNT(val1)"},
                    {"id1", "val1_count"});
  compare_test2_agg(scan.agg(scan.ref("id1"), std::vector<std::string>{"CoUnT( val1 )"}),
                    {"id1"},
                    {"COUNT(val1)"},
                    {"id1", "val1_count"});
  compare_test2_agg(scan.agg(scan.ref("id2"), scan.ref("val1").count()),
                    {"id2"},
                    {"COUNT(val1)"},
                    {"id2", "val1_count"});
  compare_test2_agg(
      scan.agg(scan.ref("id2"), {scan.count()}), {"id2"}, {"COUNT(*)"}, {"id2", "count"});
  compare_test2_agg(builder.scan("test2").agg({0, 1}, "min(val1)"),
                    {"id1", "id2"},
                    {"MIN(val1)"},
                    {"id1", "id2", "val1_min"});
  compare_test2_agg(builder.scan("test2").agg({1, 1}, {"avg(val1)", "max(val2)"}),
                    {"id2", "id2"},
                    {"AVG(val1)", "MAX(val2)"},
                    {"id2", "id2_1", "val1_avg", "val2_max"});
  compare_test2_agg(builder.scan("test2").agg(
                        {0, 0}, std::vector<std::string>{"avg( val2)", "sum( val2)"}),
                    {"id1", "id1"},
                    {"AVG(val2)", "SUM(val2)"},
                    {"id1", "id1_1", "val2_avg", "val2_sum"});
  compare_test2_agg(scan.agg({1, 0}, scan.ref("val2").avg()),
                    {"id2", "id1"},
                    {"AVG(val2)"},
                    {"id2", "id1", "val2_avg"});
  compare_test2_agg(scan.agg({0}, {scan.ref("val2").min(), scan.ref("id2").count(true)}),
                    {"id1"},
                    {"MIN(val2), COUNT(DISTINCT id2)"},
                    {"id1", "val2_min", "id2_count_dist"});
  compare_test2_agg(builder.scan("test2").agg({"id1"}, "count_dist( id2 )"),
                    {"id1"},
                    {"COUNT(DISTINCT id2)"},
                    {"id1", "id2_count_dist"});
  compare_test2_agg(
      builder.scan("test2").agg({"id1"}, {"count_distinct(id2)", "count distinct(id2 )"}),
      {"id1"},
      {"COUNT(DISTINCT id2)", "COUNT(DISTINCT id2)"},
      {"id1", "id2_count_dist", "id2_count_dist_1"});
  compare_test2_agg(
      builder.scan("test2").agg(
          {"id1"}, std::vector<std::string>{"count dist(id2)", "COUNT distinct( id2)"}),
      {"id1"},
      {"COUNT(DISTINCT id2)", "COUNT(DISTINCT id2)"},
      {"id1", "id2_count_dist", "id2_count_dist_1"});
  compare_test2_agg(scan.agg({"id2", "id1"}, scan.ref("val2").avg()),
                    {"id2", "id1"},
                    {"AVG(val2)"},
                    {"id2", "id1", "val2_avg"});
  compare_test2_agg(
      scan.agg({"id1"}, {scan.ref("val2").min(), scan.ref("id2").count(true)}),
      {"id1"},
      {"MIN(val2), COUNT(DISTINCT id2)"},
      {"id1", "val2_min", "id2_count_dist"});
  compare_test2_agg(scan.agg({scan.ref("id1")}, "approx_count_dist( id2)"),
                    {"id1"},
                    {"COUNT(DISTINCT id2)"},
                    {"id1", "id2_approx_count_dist"});
  compare_test2_agg(scan.agg({scan.ref("id1")},
                             {"approx_count_distinct(id2)", "approx count dist( id2 )"}),
                    {"id1"},
                    {"COUNT(DISTINCT id2)", "COUNT(DISTINCT id2)"},
                    {"id1", "id2_approx_count_dist", "id2_approx_count_dist_1"});
  compare_test2_agg(scan.agg({scan.ref("id1")},
                             std::vector<std::string>{"approx count distinct(id2)",
                                                      "approx COUNT distinct(id2)"}),
                    {"id1"},
                    {"COUNT(DISTINCT id2)", "COUNT(DISTINCT id2)"},
                    {"id1", "id2_approx_count_dist", "id2_approx_count_dist_1"});
  compare_test2_agg(scan.agg({scan.ref("id2"), scan.ref("id1")}, scan.ref("val2").avg()),
                    {"id2", "id1"},
                    {"AVG(val2)"},
                    {"id2", "id1", "val2_avg"});
  compare_test2_agg(
      scan.agg({scan.ref("id1")}, {scan.ref("val2").min(), scan.ref("id2").count(true)}),
      {"id1"},
      {"MIN(val2)", "COUNT(DISTINCT id2)"},
      {"id1", "val2_min", "id2_count_dist"});
  compare_test2_agg(
      scan.agg(std::vector<int>{}, {"count(val1)"}), {}, {"COUNT(val1)"}, {"val1_count"});
  compare_test2_agg(
      builder.scan("test2").agg(std::vector<int>{0, 1}, std::vector<std::string>{}),
      {"id1", "id2"},
      {},
      {"id1", "id2"});
  compare_test2_agg(builder.scan("test2").agg("", "approx_quantile(val2,0.1)"),
                    {},
                    {"APPROX_QUANTILE(val2, 0.1)"},
                    {"val2_approx_quantile"});
  compare_test2_agg(builder.scan("test2").agg("", {"approx quantile(val2, 0.5)"}),
                    {},
                    {"APPROX_QUANTILE( val2 , 0.5 )"},
                    {"val2_approx_quantile"});
  compare_test2_agg(builder.scan("test2").agg("", {"approx quantile (  val2,  1.0)"}),
                    {},
                    {"APPROX_QUANTILE(val2 ,1.0)"},
                    {"val2_approx_quantile"});
  compare_test2_agg(builder.scan("test2").agg("", {"sample(id1)"}),
                    {},
                    {"SAMPLE(id1)"},
                    {"id1_sample"});
  compare_test2_agg(builder.scan("test2").agg("val2", {"single_value(id1)"}),
                    {"val2"},
                    {"SINGLE_VALUE(id1)"},
                    {"val2", "id1_single_value"});
  compare_test2_agg(builder.scan("test2").agg("val2", {"single value(id1)"}),
                    {"val2"},
                    {"SINGLE_VALUE(id1)"},
                    {"val2", "id1_single_value"});
  compare_test2_agg(builder.scan("test2").agg("id1", {"stddev(val1)", "stddev(val2)"}),
                    {"id1"},
                    {"STDDEV_SAMP(val1)", "STDDEV_SAMP(val2)"},
                    {"id1", "val1_stddev", "val2_stddev"});
  compare_test2_agg(
      builder.scan("test2").agg("id1", {"stddev(val1)", "corr(val1, val2)"}),
      {"id1"},
      {"STDDEV_SAMP(val1)", "CORR(val1, val2)"},
      {"id1", "val1_stddev", "val1_corr_val2"});
  compare_test2_agg(scan.agg(0, ""), {"id1"}, {}, {"id1"});
  compare_test2_agg(scan.agg({0}, ""), {"id1"}, {}, {"id1"});
  compare_test2_agg(scan.agg("id1", ""), {"id1"}, {}, {"id1"});
  compare_test2_agg(scan.agg({"id1"}, ""), {"id1"}, {}, {"id1"});
  compare_test2_agg(scan.agg(scan.ref("id1"), ""), {"id1"}, {}, {"id1"});
  compare_test2_agg(scan.agg({scan.ref("id1")}, ""), {"id1"}, {}, {"id1"});

  EXPECT_THROW(builder.scan("test2").agg(15, "count"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test2").agg("unknown", "count"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test2").agg(0, "count(val34)"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test2").agg(0, "count(val1))"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test2").agg(0, "count(2)"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test2").agg(std::vector<int>(), std::vector<std::string>()),
               InvalidQueryError);
  EXPECT_THROW(scan.agg({scan.ref(0).rename("id1"), scan.ref(0).rename("id1")}, "count"),
               InvalidQueryError);
  EXPECT_THROW(scan.agg("", "approx_quantile(val2,1.1)"), InvalidQueryError);
  EXPECT_THROW(scan.agg("", "approx_quantile(val2,1..1)").node()->print(),
               InvalidQueryError);
  EXPECT_THROW(scan.agg("", "approx_quantile(val2,--1.1)"), InvalidQueryError);
  EXPECT_THROW(scan.agg("", ""), InvalidQueryError);
  EXPECT_THROW(scan.agg(std::vector<std::string>(), ""), InvalidQueryError);
  EXPECT_THROW(scan.agg("", std::vector<std::string>()), InvalidQueryError);
  EXPECT_THROW(scan.agg({0}, {scan.ref("val2").min(), scan.ref("id2")}),
               InvalidQueryError);
  EXPECT_THROW(scan.agg({0}, scan.ref("id2")), InvalidQueryError);
}

TEST_F(QueryBuilderTest, ParseSort) {
  ASSERT_EQ(BuilderSortField::parseSortDirection("asc"), SortDirection::Ascending);
  ASSERT_EQ(BuilderSortField::parseSortDirection("ascending"), SortDirection::Ascending);
  ASSERT_EQ(BuilderSortField::parseSortDirection(" aScenDing  "),
            SortDirection::Ascending);
  ASSERT_EQ(BuilderSortField::parseSortDirection("desc"), SortDirection::Descending);
  ASSERT_EQ(BuilderSortField::parseSortDirection(" descending"),
            SortDirection::Descending);
  ASSERT_EQ(BuilderSortField::parseSortDirection("DESC  "), SortDirection::Descending);
  ASSERT_EQ(BuilderSortField::parseNullPosition("first"), NullSortedPosition::First);
  ASSERT_EQ(BuilderSortField::parseNullPosition(" FIRST "), NullSortedPosition::First);
  ASSERT_EQ(BuilderSortField::parseNullPosition("last"), NullSortedPosition::Last);
  ASSERT_EQ(BuilderSortField::parseNullPosition("laST "), NullSortedPosition::Last);
  EXPECT_THROW(BuilderSortField::parseSortDirection("ascc"), InvalidQueryError);
  EXPECT_THROW(BuilderSortField::parseSortDirection("disc"), InvalidQueryError);
  EXPECT_THROW(BuilderSortField::parseNullPosition("begin"), InvalidQueryError);
  EXPECT_THROW(BuilderSortField::parseNullPosition("end"), InvalidQueryError);
}

TEST_F(QueryBuilderTest, Sort) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("sort");
  compare_sort(builder.scan("sort").sort(0),
               {{0, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort({1, 2}),
               {{1, SortDirection::Ascending, NullSortedPosition::Last},
                {2, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort("x"),
               {{0, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort({"y", "z"}),
               {{1, SortDirection::Ascending, NullSortedPosition::Last},
                {2, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(scan.sort(scan.ref(0)),
               {{0, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(scan.sort({scan.ref(1), scan.ref(2)}),
               {{1, SortDirection::Ascending, NullSortedPosition::Last},
                {2, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort({0, SortDirection::Ascending}),
               {{0, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort(BuilderSortField{"x", "desc"}, 4),
               {{0, SortDirection::Descending, NullSortedPosition::Last}},
               4,
               0);
  compare_sort(builder.scan("sort").sort({"x", "desc", "first"}, 4, 5),
               {{0, SortDirection::Descending, NullSortedPosition::First}},
               4,
               5);
  compare_sort(builder.scan("sort").sort({0, SortDirection::Ascending}),
               {{0, SortDirection::Ascending, NullSortedPosition::Last}},
               0,
               0);
  compare_sort(builder.scan("sort").sort(BuilderSortField{"x", "desc"}, 4),
               {{0, SortDirection::Descending, NullSortedPosition::Last}},
               4,
               0);
  compare_sort(builder.scan("sort").sort({"x", "desc", "first"}, 4, 5),
               {{0, SortDirection::Descending, NullSortedPosition::First}},
               4,
               5);
  compare_sort(
      builder.scan("sort").sort({{1, SortDirection::Descending}, {"z", "asc", "first"}}),
      {{1, SortDirection::Descending, NullSortedPosition::Last},
       {2, SortDirection::Ascending, NullSortedPosition::First}},
      0,
      0);
  compare_sort(builder.scan("sort").sort(
                   {{1, SortDirection::Descending}, {"z", "asc", "first"}}, 5),
               {{1, SortDirection::Descending, NullSortedPosition::Last},
                {2, SortDirection::Ascending, NullSortedPosition::First}},
               5,
               0);
  compare_sort(builder.scan("sort").sort(
                   {{1, SortDirection::Descending}, {"z", "asc", "first"}}, 5, 2),
               {{1, SortDirection::Descending, NullSortedPosition::Last},
                {2, SortDirection::Ascending, NullSortedPosition::First}},
               5,
               2);
  for (auto dir : {SortDirection::Ascending, SortDirection::Descending}) {
    compare_sort(
        builder.scan("sort").sort(0, dir), {{0, dir, NullSortedPosition::Last}}, 0, 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(
        builder.scan("sort").sort("x", dir), {{0, dir, NullSortedPosition::Last}}, 0, 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(scan.sort(scan.ref(0), dir), {{0, dir, NullSortedPosition::Last}}, 0, 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
  }
  for (auto [dir, null_pos] :
       std::initializer_list<std::pair<SortDirection, NullSortedPosition>>{
           {SortDirection::Ascending, NullSortedPosition::First},
           {SortDirection::Ascending, NullSortedPosition::Last},
           {SortDirection::Descending, NullSortedPosition::First},
           {SortDirection::Descending, NullSortedPosition::Last}}) {
    compare_sort(builder.scan("sort").sort(0, dir, null_pos), {{0, dir, null_pos}}, 0, 0);
    compare_sort(
        builder.scan("sort").sort(0, dir, null_pos, 5), {{0, dir, null_pos}}, 5, 0);
    compare_sort(
        builder.scan("sort").sort(0, dir, null_pos, 5, 2), {{0, dir, null_pos}}, 5, 2);
    compare_sort(builder.scan("sort").sort({2, 1}, dir, null_pos),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir, null_pos, 5),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 5,
                 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir, null_pos, 5, 2),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 5,
                 2);
    compare_sort(
        builder.scan("sort").sort("x", dir, null_pos), {{0, dir, null_pos}}, 0, 0);
    compare_sort(
        builder.scan("sort").sort("x", dir, null_pos, 6), {{0, dir, null_pos}}, 6, 0);
    compare_sort(
        builder.scan("sort").sort("x", dir, null_pos, 6, 1), {{0, dir, null_pos}}, 6, 1);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir, null_pos),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir, null_pos, 7),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 7,
                 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir, null_pos, 7, 2),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 7,
                 2);
    compare_sort(scan.sort(scan.ref(0), dir, null_pos), {{0, dir, null_pos}}, 0, 0);
    compare_sort(scan.sort(scan.ref(0), dir, null_pos, 5), {{0, dir, null_pos}}, 5, 0);
    compare_sort(scan.sort(scan.ref(0), dir, null_pos, 5, 3), {{0, dir, null_pos}}, 5, 3);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir, null_pos),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir, null_pos, 4),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 4,
                 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir, null_pos, 4, 4),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 4,
                 4);
  }
  for (auto dir_str : {"asc", "desc"}) {
    auto dir = BuilderSortField::parseSortDirection(dir_str);
    compare_sort(builder.scan("sort").sort(0, dir_str),
                 {{0, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir_str),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort("x", dir_str),
                 {{0, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir_str),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
    compare_sort(
        scan.sort(scan.ref(0), dir_str), {{0, dir, NullSortedPosition::Last}}, 0, 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir_str),
                 {{2, dir, NullSortedPosition::Last}, {1, dir, NullSortedPosition::Last}},
                 0,
                 0);
  }
  for (auto [dir_str, null_pos_str] :
       std::initializer_list<std::pair<std::string, std::string>>{
           {"ASC", "FIRST"}, {" asc", " last"}, {"DESC", "first "}, {"desc ", "LAST "}}) {
    auto dir = BuilderSortField::parseSortDirection(dir_str);
    auto null_pos = BuilderSortField::parseNullPosition(null_pos_str);
    compare_sort(
        builder.scan("sort").sort(0, dir_str, null_pos_str), {{0, dir, null_pos}}, 0, 0);
    compare_sort(builder.scan("sort").sort(0, dir_str, null_pos_str, 5),
                 {{0, dir, null_pos}},
                 5,
                 0);
    compare_sort(builder.scan("sort").sort(0, dir_str, null_pos_str, 5, 2),
                 {{0, dir, null_pos}},
                 5,
                 2);
    compare_sort(builder.scan("sort").sort({2, 1}, dir_str, null_pos_str),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir_str, null_pos_str, 5),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 5,
                 0);
    compare_sort(builder.scan("sort").sort({2, 1}, dir_str, null_pos_str, 5, 2),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 5,
                 2);
    compare_sort(builder.scan("sort").sort("x", dir_str, null_pos_str),
                 {{0, dir, null_pos}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort("x", dir_str, null_pos_str, 6),
                 {{0, dir, null_pos}},
                 6,
                 0);
    compare_sort(builder.scan("sort").sort("x", dir_str, null_pos_str, 6, 1),
                 {{0, dir, null_pos}},
                 6,
                 1);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir_str, null_pos_str),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir_str, null_pos_str, 7),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 7,
                 0);
    compare_sort(builder.scan("sort").sort({"z", "y"}, dir_str, null_pos_str, 7, 2),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 7,
                 2);
    compare_sort(
        scan.sort(scan.ref(0), dir_str, null_pos_str), {{0, dir, null_pos}}, 0, 0);
    compare_sort(
        scan.sort(scan.ref(0), dir_str, null_pos_str, 5), {{0, dir, null_pos}}, 5, 0);
    compare_sort(
        scan.sort(scan.ref(0), dir_str, null_pos_str, 5, 3), {{0, dir, null_pos}}, 5, 3);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir_str, null_pos_str),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 0,
                 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir_str, null_pos_str, 4),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 4,
                 0);
    compare_sort(scan.sort({scan.ref(2), scan.ref(1)}, dir_str, null_pos_str, 4, 4),
                 {{2, dir, null_pos}, {1, dir, null_pos}},
                 4,
                 4);
  }

  EXPECT_THROW(builder.scan("sort").sort(10), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort(0, "asccc"), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort(0, "asc", "firstt"), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort(-10), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort("unknown"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test3").sort("col_arr_i32"), InvalidQueryError);
  EXPECT_THROW(builder.scan("test3").sort("col_arr_i32x3"), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort(builder.scan("test3").ref(0)),
               InvalidQueryError);
}

TEST_F(QueryBuilderTest, SortAggFilterProj) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());
  auto scan = builder.scan("test2");
  auto proj = scan.proj(
      {scan["id1"], scan["id2"], scan["val1"], (scan["val2"] + 10).rename("val2")});
  auto dag = proj.filter(!proj["val1"].isNull())
                 .agg({"id1", "id2"}, "sum(val2)")
                 .sort({"id1", "id2"})
                 .finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({1, 1, 2, 2}),
                   std::vector<int32_t>({1, 2, 1, 2}),
                   std::vector<int64_t>({65, 100, 33, 73}));
}

TEST_F(QueryBuilderTest, Join) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  {
    auto dag = builder.scan("join1").join(builder.scan("join2")).finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({2, 4}),
                     std::vector<int32_t>({102, 104}),
                     std::vector<int32_t>({101, 103}));
  }

  {
    auto dag = builder.scan("join1").join(builder.scan("join2"), "left").finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({1, 2, inline_null_value<int32_t>(), 4, 5}),
                     std::vector<int32_t>({101, 102, 103, 104, 105}),
                     std::vector<int32_t>({inline_null_value<int32_t>(),
                                           101,
                                           inline_null_value<int32_t>(),
                                           103,
                                           inline_null_value<int32_t>()}));
  }

  {
    auto dag = builder.scan("join1")
                   .join(builder.scan("join2"), std::vector<std::string>{"id"})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({2, 4}),
                     std::vector<int32_t>({102, 104}),
                     std::vector<int32_t>({101, 103}));
  }

  {
    auto dag = builder.scan("join1")
                   .join(builder.scan("join2"),
                         std::vector<std::string>{"val1"},
                         std::vector<std::string>{"val2"})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({1, 2, inline_null_value<int32_t>(), 4, 5}),
                     std::vector<int32_t>({101, 102, 103, 104, 105}),
                     std::vector<int32_t>({2, 3, 4, inline_null_value<int32_t>(), 6}));
  }

  {
    auto scan1 = builder.scan("join1");
    auto scan2 = builder.scan("join2");
    auto dag = scan1.join(scan2, scan1["id"] == scan2["id"]).finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({2, 4}),
                     std::vector<int32_t>({102, 104}),
                     std::vector<int32_t>({2, 4}),
                     std::vector<int32_t>({101, 103}));
  }

  {
    auto scan1 = builder.scan("join1");
    auto scan2 = builder.scan("join2");
    auto dag = scan1.join(scan2, scan1["id"] == scan2["id"], "left").finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int32_t>({1, 2, inline_null_value<int32_t>(), 4, 5}),
                     std::vector<int32_t>({101, 102, 103, 104, 105}),
                     std::vector<int32_t>({inline_null_value<int32_t>(),
                                           2,
                                           inline_null_value<int32_t>(),
                                           4,
                                           inline_null_value<int32_t>()}),
                     std::vector<int32_t>({inline_null_value<int32_t>(),
                                           101,
                                           inline_null_value<int32_t>(),
                                           103,
                                           inline_null_value<int32_t>()}));
  }
}

class Issue355 : public TestSuite {
 protected:
  static void SetUpTestSuite() {
    createTable("frame1",
                {{"col0", ctx().int64()},
                 {"col1", ctx().int64()},
                 {"col2", ctx().int64()},
                 {"col3", ctx().int64()},
                 {"col4", ctx().int64()},
                 {"col5", ctx().int64()},
                 {"col6", ctx().int64()},
                 {"col7", ctx().int64()},
                 {"col8", ctx().int64()},
                 {"col9", ctx().int64()},
                 {"col10", ctx().int64()},
                 {"col11", ctx().int64()},
                 {"col12", ctx().int64()},
                 {"col13", ctx().int64()},
                 {"col14", ctx().int64()},
                 {"col15", ctx().int64()}});
    insertCsvValues("frame1", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16");
    createTable("frame2", {{"col1", ctx().int64()}, {"col2", ctx().int64()}});
    insertCsvValues("frame2", "1,2");
    createTable("frame3", {{"col1", ctx().int64()}, {"col2", ctx().int64()}});
    insertCsvValues("frame3", "1,2");
  }

  static void TearDownTestSuite() {}
};

TEST_F(Issue355, Reproducer) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto step0 = builder.scan("frame1");
  auto step1 = step0.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto step2 = builder.scan("frame1");
  auto step3 = step1.join(step2, step1.ref(0) == step2.ref(0));
  auto step4 = step3.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27});
  auto step5 = builder.scan("frame2");
  auto step6 = step4.join(step5, step4.ref(0) == step5.ref(0));
  auto step7 = builder.scan("frame1");
  auto step8 = step7.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto step9 = builder.scan("frame1");
  auto step10 = step8.join(step9, step8.ref(0) == step9.ref(0));
  auto step11 =
      step10.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 27});
  auto step12 = step6.join(step11, step6.ref(0) == step11.ref(0));
  auto step13 = step12.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18});
  auto step14 = builder.scan("frame3");
  auto step15 = step13.join(step14, step13.ref(0) == step14.ref(0));
  auto step16 = builder.scan("frame1");
  auto step17 = step16.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto step18 = builder.scan("frame1");
  auto step19 = step17.join(step18, step17.ref(0) == step18.ref(0));
  auto step20 = step19.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27});
  auto step21 = builder.scan("frame2");
  auto step22 = step20.join(step21, step20.ref(0) == step21.ref(0));
  auto step23 = builder.scan("frame1");
  auto step24 = step23.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto step25 = builder.scan("frame1");
  auto step26 = step24.join(step25, step24.ref(0) == step25.ref(0));
  auto step27 = step26.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27});
  auto step28 = step22.join(step27, step22.ref(0) == step27.ref(0));
  auto step29 =
      step28.proj({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20});
  auto step30 = step15.join(step29, step15.ref(0) == step29.ref(0));
  auto step31 = step30.proj({0, 11});

  auto dag = step31.finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int64_t>({1}), std::vector<int64_t>({12}));
}

class Issue513 : public TestSuite {
 protected:
  static void SetUpTestSuite() {
    createTable("test_513", {{"A", ctx().int64()}});
    insertCsvValues("test_513", "1\n2\n3");
  }

  static void TearDownTestSuite() { dropTable("test_513"); }
};

TEST_F(Issue513, Reproducer) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("test_513");
  auto proj = scan.proj(scan.ref("A").isNull());
  auto dag = proj.agg(std::vector<std::string>(), {proj.ref(0).count()}).finalize();
  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int32_t>({3}));
}

class Issue588 : public TestSuite {
 protected:
  static void SetUpTestSuite() {
    createTable("test_588_1", {{"id", ctx().int64()}, {"A", ctx().int64()}});
    insertCsvValues("test_588_1", "1,1\n2,2\n3,3");
    createTable("test_588_2", {{"id", ctx().int64()}, {"B", ctx().int64()}});
    insertCsvValues("test_588_2", "1,2\n2,3\n3,3");
    createTable("test_588_3", {{"id", ctx().int64()}, {"C", ctx().int64()}});
    insertCsvValues("test_588_3", "1,1\n2,2\n3,3");
    createTable("test_588_4", {{"id", ctx().int64()}, {"D", ctx().int64()}});
    insertCsvValues("test_588_4", "1,2\n2,3\n3,3");
  }

  static void TearDownTestSuite() {
    dropTable("test_588_1");
    dropTable("test_588_2");
    dropTable("test_588_3");
    dropTable("test_588_4");
  }
};

TEST_F(Issue588, Reproducer1) {
  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan1 = builder.scan("test_588_1");
  auto scan2 = builder.scan("test_588_2");
  auto scan3 = builder.scan("test_588_3");
  auto scan4 = builder.scan("test_588_4");

  auto dag1 = scan1.proj({0, 1}).join(scan2.proj({0, 1})).finalize();
  auto res1 = runQuery(std::move(dag1));

  auto dag2 =
      builder.scan(res1.tableName()).proj({0, 1}).join(scan3.proj({0, 1})).finalize();
  auto res2 = runQuery(std::move(dag2));

  auto dag3 =
      builder.scan(res2.tableName()).proj({0, 1}).join(scan4.proj({0, 1})).finalize();
  auto res3 = runQuery(std::move(dag3));
}

TEST_F(QueryBuilderTest, RunOnResult) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  {
    auto dag1 = builder.scan("test1").proj({0, 1}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(res1,
                     std::vector<int64_t>({1, 2, 3, 4, 5}),
                     std::vector<int32_t>({11, 22, 33, 44, 55}));

    auto dag2 = builder.scan(res1.tableName()).proj({1, 0}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(res2,
                     std::vector<int32_t>({11, 22, 33, 44, 55}),
                     std::vector<int64_t>({1, 2, 3, 4, 5}));

    auto scan = builder.scan(res2.tableName());
    auto dag3 = scan.proj({scan["col_i"] + 1, scan["col_bi"] + 2}).finalize();
    auto res3 = runQuery(std::move(dag3));
    compare_res_data(res3,
                     std::vector<int32_t>({12, 23, 34, 45, 56}),
                     std::vector<int64_t>({3, 4, 5, 6, 7}));
  }

  {
    auto scan1 = builder.scan("test1");
    auto dag1 = scan1.proj({scan1["col_bi"] + 1, scan1["col_f"]}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(res1,
                     std::vector<int64_t>({2, 3, 4, 5, 6}),
                     std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}));

    auto scan2 = builder.scan("test1");
    auto dag2 = scan2.proj({scan2["col_bi"] + 2, scan2["col_d"]}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(res2,
                     std::vector<int64_t>({3, 4, 5, 6, 7}),
                     std::vector<double>({11.11, 22.22, 33.33, 44.44, 55.55}));

    auto dag3 =
        builder.scan(res1.tableName()).join(builder.scan(res2.tableName())).finalize();
    auto res3 = runQuery(std::move(dag3));
    compare_res_data(res3,
                     std::vector<int64_t>({3, 4, 5, 6}),
                     std::vector<float>({2.2, 3.3, 4.4, 5.5}),
                     std::vector<double>({11.11, 22.22, 33.33, 44.44}));
  }
}

TEST_F(QueryBuilderTest, RowidOnResult) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  {
    auto dag1 = builder.scan("test1").proj({0, 1}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(res1,
                     std::vector<int64_t>({1, 2, 3, 4, 5}),
                     std::vector<int32_t>({11, 22, 33, 44, 55}));

    auto dag2 =
        builder.scan(res1.tableName()).proj({"col_i", "col_bi", "rowid"}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(res2,
                     std::vector<int32_t>({11, 22, 33, 44, 55}),
                     std::vector<int64_t>({1, 2, 3, 4, 5}),
                     std::vector<int64_t>({0, 1, 2, 3, 4}));
  }
}

TEST_F(QueryBuilderTest, SqlOnResult) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto res1 =
      runSqlQuery("SELECT col_bi, col_i FROM test1;", ExecutorDeviceType::CPU, false);
  compare_res_data(res1,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int32_t>({11, 22, 33, 44, 55}));

  auto dag = builder.scan(res1.tableName()).proj({1, 0}).finalize();
  auto res2 = runQuery(std::move(dag));
  compare_res_data(res2,
                   std::vector<int32_t>({11, 22, 33, 44, 55}),
                   std::vector<int64_t>({1, 2, 3, 4, 5}));

  auto res3 = runSqlQuery("SELECT col_bi + 1, col_i - 1 FROM " + res2.tableName() + ";",
                          ExecutorDeviceType::CPU,
                          false);
  compare_res_data(res3,
                   std::vector<int64_t>({2, 3, 4, 5, 6}),
                   std::vector<int32_t>({10, 21, 32, 43, 54}));
}

TEST_F(QueryBuilderTest, NoneEncodedStringInRes) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());

    auto dag1 = builder.scan("test_str").proj({0, 1}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(
        res1,
        std::vector<int32_t>({inline_null_value<int32_t>(),
                              1,
                              inline_null_value<int32_t>(),
                              3,
                              inline_null_value<int32_t>(),
                              5}),
        std::vector<std::string>(
            {"<NULL>"s, "str1"s, "<NULL>"s, "str333"s, "<NULL>"s, "str55555"s}));

    auto dag2 = builder.scan(res1.tableName()).proj({0, 1}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(
        res2,
        std::vector<int32_t>({inline_null_value<int32_t>(),
                              1,
                              inline_null_value<int32_t>(),
                              3,
                              inline_null_value<int32_t>(),
                              5}),
        std::vector<std::string>(
            {"<NULL>"s, "str1"s, "<NULL>"s, "str333"s, "<NULL>"s, "str55555"s}));

    auto scan = builder.scan(res2.tableName());
    auto dag3 = scan.filter(scan.ref(0) > 1).finalize();
    auto res3 = runQuery(std::move(dag3));
    compare_res_data(res3,
                     std::vector<int32_t>({3, 5}),
                     std::vector<std::string>({"str333"s, "str55555"s}));
  }
}

TEST_F(QueryBuilderTest, VarlenArrayInRes) {
  std::vector<std::pair<bool, bool>> variants = {
      {true, false}, {true, true}, {false, false}, {false, true}};
  for (auto [enable_columnar, lazy_fetch] : variants) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    auto orig_enable_lazy_fetch = config().rs.enable_lazy_fetch;
    ScopeGuard guard([&]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
      config().rs.enable_lazy_fetch = orig_enable_lazy_fetch;
    });
    config().rs.enable_columnar_output = enable_columnar;
    config().rs.enable_lazy_fetch = lazy_fetch;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());

    auto dag1 = builder.scan("test_varr").proj({0, 1, 2}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(
        res1,
        std::vector<int32_t>({1, 2, 3, 4}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({1, inline_null_value<int32_t>(), 3}),
             std::vector<int32_t>({inline_null_array_value<int32_t>()}),
             std::vector<int32_t>({}),
             std::vector<int32_t>(
                 {inline_null_value<int32_t>(), 2, inline_null_value<int32_t>(), 4})}),
        std::vector<std::vector<double>>(
            {std::vector<double>({4.0, inline_null_value<double>()}),
             std::vector<double>({}),
             std::vector<double>({inline_null_array_value<double>()}),
             std::vector<double>({inline_null_value<double>(), 5.0, 6.0})}));

    auto dag2 = builder.scan(res1.tableName()).proj({2, 1, 0}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(
        res2,
        std::vector<std::vector<double>>(
            {std::vector<double>({4.0, inline_null_value<double>()}),
             std::vector<double>({}),
             std::vector<double>({inline_null_array_value<double>()}),
             std::vector<double>({inline_null_value<double>(), 5.0, 6.0})}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({1, inline_null_value<int32_t>(), 3}),
             std::vector<int32_t>({inline_null_array_value<int32_t>()}),
             std::vector<int32_t>({}),
             std::vector<int32_t>(
                 {inline_null_value<int32_t>(), 2, inline_null_value<int32_t>(), 4})}),
        std::vector<int32_t>({1, 2, 3, 4}));

    auto scan = builder.scan(res2.tableName());
    auto dag3 = scan.filter(scan.ref(2) > 2).finalize();
    auto res3 = runQuery(std::move(dag3));
    compare_res_data(
        res3,
        std::vector<std::vector<double>>(
            {std::vector<double>({inline_null_array_value<double>()}),
             std::vector<double>({inline_null_value<double>(), 5.0, 6.0})}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({}),
             std::vector<int32_t>(
                 {inline_null_value<int32_t>(), 2, inline_null_value<int32_t>(), 4})}),
        std::vector<int32_t>({3, 4}));
  }
}

TEST_F(QueryBuilderTest, FixedArrayInRes) {
  std::vector<std::pair<bool, bool>> variants = {
      {true, false}, {true, true}, {false, false}, {false, true}};
  for (auto [enable_columnar, lazy_fetch] : variants) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    auto orig_enable_lazy_fetch = config().rs.enable_lazy_fetch;
    ScopeGuard guard([&]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
      config().rs.enable_lazy_fetch = orig_enable_lazy_fetch;
    });
    config().rs.enable_columnar_output = enable_columnar;
    config().rs.enable_lazy_fetch = lazy_fetch;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());

    auto dag1 = builder.scan("test_arr").proj({0, 1, 2}).finalize();
    auto res1 = runQuery(std::move(dag1));
    compare_res_data(
        res1,
        std::vector<int32_t>({1, 2, 3, 4}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({inline_null_array_value<int32_t>()}),
             std::vector<int32_t>({inline_null_value<int32_t>(), 2}),
             std::vector<int32_t>({1, inline_null_value<int32_t>()}),
             std::vector<int32_t>({1, 2})}),
        std::vector<std::vector<double>>(
            {std::vector<double>({4.0, inline_null_value<double>(), 6.0}),
             std::vector<double>({inline_null_array_value<double>()}),
             std::vector<double>(
                 {inline_null_value<double>(), 5.0, inline_null_value<double>()}),
             std::vector<double>({4.0, 5.0, 6.0})}));

    auto dag2 = builder.scan(res1.tableName()).proj({2, 1, 0}).finalize();
    auto res2 = runQuery(std::move(dag2));
    compare_res_data(
        res2,
        std::vector<std::vector<double>>(
            {std::vector<double>({4.0, inline_null_value<double>(), 6.0}),
             std::vector<double>({inline_null_array_value<double>()}),
             std::vector<double>(
                 {inline_null_value<double>(), 5.0, inline_null_value<double>()}),
             std::vector<double>({4.0, 5.0, 6.0})}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({inline_null_array_value<int32_t>()}),
             std::vector<int32_t>({inline_null_value<int32_t>(), 2}),
             std::vector<int32_t>({1, inline_null_value<int32_t>()}),
             std::vector<int32_t>({1, 2})}),
        std::vector<int32_t>({1, 2, 3, 4}));

    auto scan = builder.scan(res2.tableName());
    auto dag3 = scan.filter(scan.ref(2) > 2).finalize();
    auto res3 = runQuery(std::move(dag3));
    compare_res_data(
        res3,
        std::vector<std::vector<double>>(
            {std::vector<double>(
                 {inline_null_value<double>(), 5.0, inline_null_value<double>()}),
             std::vector<double>({4.0, 5.0, 6.0})}),
        std::vector<std::vector<int32_t>>(
            {std::vector<int32_t>({1, inline_null_value<int32_t>()}),
             std::vector<int32_t>({1, 2})}),
        std::vector<int32_t>({3, 4}));
  }
}

TEST_F(QueryBuilderTest, ProjUnnest) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());

    auto scan = builder.scan("test_unnest");
    {
      auto dag =
          scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res, std::vector<int32_t>({1, 4, 4}), std::vector<int32_t>({1, 1, 2}));
    }

    {
      auto dag =
          scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32x2").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int32_t>({1, 1, 2, 2, 3, 3}),
          std::vector<int32_t>(
              {0, 1, inline_null_value<int32_t>(), inline_null_value<int32_t>(), 0, 1}));
    }

    {
      auto dag =
          scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32nn").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 2, 4}),
                       std::vector<int32_t>({1, 2, 1, inline_null_value<int32_t>()}));
    }

    {
      auto dag =
          scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32x2nn").unnest()}).finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 2, 2, 3, 3, 4, 4}),
                       std::vector<int32_t>({3, 4, 5, 6, 7, 8, 9, 10}));
    }
  }
}

TEST_F(QueryBuilderTest, ProjUnnestMultiCol) {
  for (bool enable_columnar : {true, false}) {
    auto orig_enable_columnar = config().rs.enable_columnar_output;
    ScopeGuard guard([orig_enable_columnar]() {
      config().rs.enable_columnar_output = orig_enable_columnar;
    });
    config().rs.enable_columnar_output = enable_columnar;

    QueryBuilder builder(ctx(), schema_mgr_, configPtr());

    auto scan = builder.scan("test_unnest");
    {
      auto dag = scan.proj({scan.ref("col_i"),
                            scan.ref("col_arr_i32").unnest(),
                            scan.ref("col_arr_i32nn").unnest()})
                     .sort({0, 1, 2})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int32_t>({1, 1, 4, 4}),
          std::vector<int32_t>({1, 1, 1, 2}),
          std::vector<int32_t>(
              {1, 2, inline_null_value<int32_t>(), inline_null_value<int32_t>()}));
    }

    {
      auto dag = scan.proj({scan.ref("col_i"),
                            scan.ref("col_arr_i32x2").unnest(),
                            scan.ref("col_arr_i32x2nn").unnest()})
                     .sort({0, 1, 2})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
                       std::vector<int32_t>({0,
                                             0,
                                             1,
                                             1,
                                             inline_null_value<int32_t>(),
                                             inline_null_value<int32_t>(),
                                             inline_null_value<int32_t>(),
                                             inline_null_value<int32_t>(),
                                             0,
                                             0,
                                             1,
                                             1}),
                       std::vector<int32_t>({3, 4, 3, 4, 5, 5, 6, 6, 7, 8, 7, 8}));
    }

    {
      auto dag = scan.proj({scan.ref("col_i"),
                            scan.ref("col_arr_i32nn").unnest(),
                            scan.ref("col_arr_i32x2").unnest()})
                     .sort({0, 1, 2})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(
          res,
          std::vector<int32_t>({1, 1, 1, 1, 2, 2}),
          std::vector<int32_t>({1, 1, 2, 2, 1, 1}),
          std::vector<int32_t>(
              {0, 1, 0, 1, inline_null_value<int32_t>(), inline_null_value<int32_t>()}));
    }

    {
      auto dag = scan.proj({scan.ref("col_i"),
                            scan.ref("col_arr_i32x2nn").unnest(),
                            scan.ref("col_arr_i32nn").unnest()})
                     .sort({0, 1, 2})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 1, 1, 2, 2, 4, 4}),
                       std::vector<int32_t>({3, 3, 4, 4, 5, 6, 9, 10}),
                       std::vector<int32_t>({1,
                                             2,
                                             1,
                                             2,
                                             1,
                                             1,
                                             inline_null_value<int32_t>(),
                                             inline_null_value<int32_t>()}));
    }

    {
      auto dag = scan.proj({scan.ref("col_i"),
                            scan.ref("col_arr_i32").unnest(),
                            scan.ref("col_arr_i32x2").unnest(),
                            scan.ref("col_arr_i32nn").unnest(),
                            scan.ref("col_arr_i32x2nn").unnest()})
                     .sort({0, 1, 2, 3, 4})
                     .finalize();
      auto res = runQuery(std::move(dag));
      compare_res_data(res,
                       std::vector<int32_t>({1, 1, 1, 1, 1, 1, 1, 1}),
                       std::vector<int32_t>({1, 1, 1, 1, 1, 1, 1, 1}),
                       std::vector<int32_t>({0, 0, 0, 0, 1, 1, 1, 1}),
                       std::vector<int32_t>({1, 1, 2, 2, 1, 1, 2, 2}),
                       std::vector<int32_t>({3, 4, 3, 4, 3, 4, 3, 4}));
    }
  }
}

TEST_F(QueryBuilderTest, ProjUnnestMultiStep) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  auto scan = builder.scan("test_unnest");
  {
    auto dag = scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32").unnest()})
                   .proj({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res, std::vector<int32_t>({1, 4, 4}));
  }

  {
    auto dag = scan.proj({scan.ref("col_i"), scan.ref("col_arr_i32").unnest()})
                   .agg({0}, "count"s)
                   .sort({0})
                   .finalize();
    auto res = runQuery(std::move(dag));
    compare_res_data(res, std::vector<int32_t>({1, 4}), std::vector<int32_t>({1, 2}));
  }
}

TEST_F(QueryBuilderTest, Head) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  {
    auto dag = builder.scan("test1").proj({0, 1}).finalize();
    auto res = runQuery(std::move(dag));

    auto res1 = res.head(10);
    compare_res_data(res1,
                     std::vector<int64_t>({1, 2, 3, 4, 5}),
                     std::vector<int32_t>({11, 22, 33, 44, 55}));

    auto res2 = res.head(3);
    compare_res_data(
        res2, std::vector<int64_t>({1, 2, 3}), std::vector<int32_t>({11, 22, 33}));

    auto res3 = res.head(0);
    compare_res_data(res3, std::vector<int64_t>({}), std::vector<int32_t>({}));
  }

  {
    auto dag = builder.scan("test1")
                   .proj({0, 1})
                   .sort(std::vector<BuilderSortField>(), 3, 1)
                   .finalize();
    auto res = runQuery(std::move(dag));

    auto res1 = res.head(10);
    compare_res_data(
        res1, std::vector<int64_t>({2, 3, 4}), std::vector<int32_t>({22, 33, 44}));

    auto res2 = res.head(2);
    compare_res_data(res2, std::vector<int64_t>({2, 3}), std::vector<int32_t>({22, 33}));

    auto res3 = res.head(0);
    compare_res_data(res3, std::vector<int64_t>({}), std::vector<int32_t>({}));
  }
}

TEST_F(QueryBuilderTest, Tail) {
  QueryBuilder builder(ctx(), schema_mgr_, configPtr());

  {
    auto dag = builder.scan("test1").proj({0, 1}).finalize();
    auto res = runQuery(std::move(dag));

    auto res1 = res.tail(10);
    compare_res_data(res1,
                     std::vector<int64_t>({1, 2, 3, 4, 5}),
                     std::vector<int32_t>({11, 22, 33, 44, 55}));

    auto res2 = res.tail(3);
    compare_res_data(
        res2, std::vector<int64_t>({3, 4, 5}), std::vector<int32_t>({33, 44, 55}));

    auto res3 = res.tail(0);
    compare_res_data(res3, std::vector<int64_t>({}), std::vector<int32_t>({}));
  }

  {
    auto dag = builder.scan("test1")
                   .proj({0, 1})
                   .sort(std::vector<BuilderSortField>(), 3, 1)
                   .finalize();
    auto res = runQuery(std::move(dag));

    auto res1 = res.tail(10);
    compare_res_data(
        res1, std::vector<int64_t>({2, 3, 4}), std::vector<int32_t>({22, 33, 44}));

    auto res2 = res.tail(2);
    compare_res_data(res2, std::vector<int64_t>({3, 4}), std::vector<int32_t>({33, 44}));

    auto res3 = res.tail(0);
    compare_res_data(res3, std::vector<int64_t>({}), std::vector<int32_t>({}));
  }
}

class Taxi : public TestSuite {
 protected:
  static void SetUpTestSuite() {
    ArrowStorage::TableOptions table_options;
    table_options.fragment_size = 2;
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    getStorage()->importCsvFile(
        getFilePath("taxi_sample.csv"),
        "trips",
        {{"trip_id", ctx().int32()},
         {"vendor_id", ctx().extDict(ctx().text(), 0)},
         {"pickup_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
         {"dropoff_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
         {"store_and_fwd_flag", ctx().extDict(ctx().text(), 0)},
         {"rate_code_id", ctx().int16()},
         {"pickup_longitude", ctx().fp64()},
         {"pickup_latitude", ctx().fp64()},
         {"dropoff_longitude", ctx().fp64()},
         {"dropoff_latitude", ctx().fp64()},
         {"passenger_count", ctx().int16()},
         {"trip_distance", ctx().decimal64(14, 2)},
         {"fare_amount", ctx().decimal64(14, 2)},
         {"extra", ctx().decimal64(14, 2)},
         {"mta_tax", ctx().decimal64(14, 2)},
         {"tip_amount", ctx().decimal64(14, 2)},
         {"tolls_amount", ctx().decimal64(14, 2)},
         {"ehail_fee", ctx().decimal64(14, 2)},
         {"improvement_surcharge", ctx().decimal64(14, 2)},
         {"total_amount", ctx().decimal64(14, 2)},
         {"payment_type", ctx().extDict(ctx().text(), 0)},
         {"trip_type", ctx().int16()},
         {"pickup", ctx().extDict(ctx().text(), 0)},
         {"dropoff", ctx().extDict(ctx().text(), 0)},
         {"cab_type", ctx().extDict(ctx().text(), 0)},
         {"precipitation", ctx().decimal64(14, 2)},
         {"snow_depth", ctx().int16()},
         {"snowfall", ctx().decimal64(14, 2)},
         {"max_temperature", ctx().int16()},
         {"min_temperature", ctx().int16()},
         {"average_wind_speed", ctx().decimal64(14, 2)},
         {"pickup_nyct2010_gid", ctx().int16()},
         {"pickup_ctlabel", ctx().extDict(ctx().text(), 0)},
         {"pickup_borocode", ctx().int16()},
         {"pickup_boroname", ctx().extDict(ctx().text(), 0)},
         {"pickup_ct2010", ctx().extDict(ctx().text(), 0)},
         {"pickup_boroct2010", ctx().extDict(ctx().text(), 0)},
         {"pickup_cdeligibil", ctx().extDict(ctx().text(), 0)},
         {"pickup_ntacode", ctx().extDict(ctx().text(), 0)},
         {"pickup_ntaname", ctx().extDict(ctx().text(), 0)},
         {"pickup_puma", ctx().extDict(ctx().text(), 0)},
         {"dropoff_nyct2010_gid", ctx().int16()},
         {"dropoff_ctlabel", ctx().extDict(ctx().text(), 0)},
         {"dropoff_borocode", ctx().int16()},
         {"dropoff_boroname", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ct2010", ctx().extDict(ctx().text(), 0)},
         {"dropoff_boroct2010", ctx().extDict(ctx().text(), 0)},
         {"dropoff_cdeligibil", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ntacode", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ntaname", ctx().extDict(ctx().text(), 0)},
         {"dropoff_puma", ctx().extDict(ctx().text(), 0)}},
        table_options,
        parse_options);
  }

  static void TearDownTestSuite() {}

  void run_compare_q1(std::unique_ptr<QueryDag> dag) {
    auto res = runQuery(std::move(dag));
    compare_res_data(
        res, std::vector<std::string>({"green"s}), std::vector<int32_t>({20}));
  }

  void run_compare_q2(std::unique_ptr<QueryDag> dag) {
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int16_t>({1, 2, 5}),
                     std::vector<double>({98.19f / 16, 75.0f, 13.58f / 3}));
  }

  void run_compare_q3(std::unique_ptr<QueryDag> dag) {
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int16_t>({1, 2, 5}),
                     std::vector<int64_t>({2013, 2013, 2013}),
                     std::vector<int32_t>({16, 1, 3}));
  }

  void run_compare_q4(std::unique_ptr<QueryDag> dag) {
    auto res = runQuery(std::move(dag));
    compare_res_data(res,
                     std::vector<int16_t>({1, 5, 2}),
                     std::vector<int64_t>({2013, 2013, 2013}),
                     std::vector<int32_t>({0, 0, 0}),
                     std::vector<int32_t>({16, 3, 1}));
  }
};

namespace {

hdk::ir::ExprPtr createColumnRef(const std::shared_ptr<hdk::ir::Scan>& scan,
                                 const std::string& col_name) {
  for (size_t i = 0; i < scan->size(); ++i) {
    if (scan->getFieldName(i) == col_name) {
      return hdk::ir::getNodeColumnRef(scan.get(), (unsigned)i);
    }
  }
  CHECK(false);
  return nullptr;
}

}  // namespace

TEST_F(Taxi, Q1_NoBuilder) {
  // SELECT cab_type, count(*) FROM trips GROUP BY cab_type;
  // Create 'trips' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "trips");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with `cab_type` field at the front because it is
  // required for aggregation.
  auto cab_type_ref = createColumnRef(scan, "cab_type");
  auto proj = std::make_shared<Project>(
      ExprPtrVector{cab_type_ref}, std::vector<std::string>{"cab_type"}, scan);
  // Create aggregation.
  auto count_agg =
      makeExpr<AggExpr>(ctx().int32(), AggType::kCount, nullptr, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      1, ExprPtrVector{count_agg}, std::vector<std::string>{"cab_type", "count"}, proj);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(agg);

  run_compare_q1(std::move(dag));
}

TEST_F(Taxi, Q1_1) {
  // SELECT cab_type, count(*) FROM trips GROUP BY cab_type;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto dag = builder.scan("trips").agg("cab_type", "count").finalize();

  run_compare_q1(std::move(dag));
}

TEST_F(Taxi, Q1_2) {
  // SELECT cab_type, count(*) FROM trips GROUP BY cab_type;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto dag = builder.scan("trips").agg({"cab_type"}, builder.count()).finalize();

  run_compare_q1(std::move(dag));
}

TEST_F(Taxi, Q2_NoBuilder) {
  // SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count
  // ORDER BY passenger_count;
  // Create 'trips' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "trips");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with `passenger_count` and `total_amount` fields.
  auto passenger_count_ref = createColumnRef(scan, "passenger_count");
  auto total_amount_ref = createColumnRef(scan, "total_amount");
  auto proj = std::make_shared<Project>(
      ExprPtrVector{passenger_count_ref, total_amount_ref},
      std::vector<std::string>{"passenger_count", "total_amount"},
      scan);
  // Create aggregation.
  auto total_amount_proj_ref =
      makeExpr<hdk::ir::ColumnRef>(total_amount_ref->type(), proj.get(), 1);
  auto avg_agg = makeExpr<AggExpr>(
      ctx().int32(), AggType::kAvg, total_amount_proj_ref, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      1,
      ExprPtrVector{avg_agg},
      std::vector<std::string>{"passenger_count", "total_amount"},
      proj);
  // Create sort.
  auto sort = std::make_shared<hdk::ir::Sort>(
      std::vector<hdk::ir::SortField>{
          {0, SortDirection::Ascending, NullSortedPosition::First}},
      0,
      0,
      agg);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(sort);

  run_compare_q2(std::move(dag));
}

TEST_F(Taxi, Q2_1) {
  // SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count
  // ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.agg({"passenger_count"}, {"avg(total_amount)"})
                 .sort("passenger_count")
                 .finalize();

  run_compare_q2(std::move(dag));
}

TEST_F(Taxi, Q2_2) {
  // SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count
  // ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.agg({"passenger_count"}, {scan.ref("total_amount").avg()})
                 .sort({{0, SortDirection::Ascending}})
                 .finalize();

  run_compare_q2(std::move(dag));
}

TEST_F(Taxi, Q2_3) {
  // SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count
  // ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto agg = scan.agg({scan.ref("passenger_count")}, {scan.ref("total_amount").avg()});
  auto dag = agg.sort({{agg.ref(0), "asc"}}).finalize();

  run_compare_q2(std::move(dag));
}

TEST_F(Taxi, Q3_NoBuilder) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*)
  // FROM trips GROUP BY passenger_count, pickup_year ORDER BY passenger_count;
  // Create 'trips' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "trips");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with `passenger_count` and `total_amount` fields.
  auto passenger_count_ref = createColumnRef(scan, "passenger_count");
  auto pickup_datetime_ref = createColumnRef(scan, "pickup_datetime");
  auto pickup_year = makeExpr<ExtractExpr>(
      ctx().int64(), false, DateExtractField::kYear, pickup_datetime_ref);
  auto proj = std::make_shared<Project>(
      ExprPtrVector{passenger_count_ref, pickup_year},
      std::vector<std::string>{"passenger_count", "pickup_year"},
      scan);
  // Create aggregation.
  auto count_agg =
      makeExpr<AggExpr>(ctx().int32(), AggType::kCount, nullptr, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      2,
      ExprPtrVector{count_agg},
      std::vector<std::string>{"passenger_count", "pickup_year", "count"},
      proj);
  // Create sort.
  auto sort = std::make_shared<hdk::ir::Sort>(
      std::vector<hdk::ir::SortField>{
          {0, SortDirection::Ascending, NullSortedPosition::First}},
      0,
      0,
      agg);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(sort);

  run_compare_q3(std::move(dag));
}

TEST_F(Taxi, Q3_1) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*)
  // FROM trips GROUP BY passenger_count, pickup_year ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract(DateExtractField::kYear)},
                       {"passenger_count", "pickup_year"})
                 .agg({"passenger_count", "pickup_year"}, {builder.count()})
                 .sort("passenger_count")
                 .finalize();

  run_compare_q3(std::move(dag));
}

TEST_F(Taxi, Q3_2) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*)
  // FROM trips GROUP BY passenger_count, pickup_year ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract("year")},
                       {"passenger_count", "pickup_year"})
                 .agg({0, 1}, "count")
                 .sort(0)
                 .finalize();

  run_compare_q3(std::move(dag));
}

TEST_F(Taxi, Q3_3) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*)
  // FROM trips GROUP BY passenger_count, pickup_year ORDER BY passenger_count;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag =
      scan.proj({scan.ref("passenger_count"),
                 scan.ref("pickup_datetime").extract("year").rename("pickup_year")})
          .agg({0, 1}, {"count"})
          .sort({"passenger_count", SortDirection::Ascending})
          .finalize();

  run_compare_q3(std::move(dag));
}

TEST_F(Taxi, Q4_NoBuilder) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year,
  // cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP BY
  // passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
  // Create 'trips' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "trips");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with `passenger_count` and `total_amount` fields.
  auto passenger_count_ref = createColumnRef(scan, "passenger_count");
  auto pickup_datetime_ref = createColumnRef(scan, "pickup_datetime");
  auto pickup_year = makeExpr<ExtractExpr>(
      ctx().int64(), false, DateExtractField::kYear, pickup_datetime_ref);
  auto trip_distance_ref = createColumnRef(scan, "trip_distance");
  auto distance = trip_distance_ref->cast(ctx().int32());
  auto proj = std::make_shared<Project>(
      ExprPtrVector{passenger_count_ref, pickup_year, distance},
      std::vector<std::string>{"passenger_count", "pickup_year", "distance"},
      scan);
  // Create aggregation.
  auto count_agg =
      makeExpr<AggExpr>(ctx().int32(), AggType::kCount, nullptr, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      3,
      ExprPtrVector{count_agg},
      std::vector<std::string>{"passenger_count", "pickup_year", "distance", "count"},
      proj);
  // Create sort.
  auto sort = std::make_shared<hdk::ir::Sort>(
      std::vector<hdk::ir::SortField>{
          {1, SortDirection::Ascending, NullSortedPosition::First},
          {3, SortDirection::Descending, NullSortedPosition::First}},
      0,
      0,
      agg);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(sort);

  run_compare_q4(std::move(dag));
}

TEST_F(Taxi, Q4_1) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year,
  // cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP BY
  // passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract(DateExtractField::kYear),
                        scan.ref("trip_distance").cast(ctx().int32())},
                       {"passenger_count", "pickup_year", "distance"})
                 .agg({"passenger_count", "pickup_year", "distance"}, {builder.count()})
                 .sort({{"pickup_year", SortDirection::Ascending},
                        {"count", SortDirection::Descending}})
                 .finalize();

  run_compare_q4(std::move(dag));
}

TEST_F(Taxi, Q4_2) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year,
  // cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP BY
  // passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract("year").rename("pickup_year"),
                        scan.ref("trip_distance").cast("int32").rename("distance")})
                 .agg({0, 1, 2}, {"count"})
                 .sort({{1, SortDirection::Ascending}, {3, SortDirection::Descending}})
                 .finalize();

  run_compare_q4(std::move(dag));
}

TEST_F(Taxi, Q4_3) {
  // SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year,
  // cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP BY
  // passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("trips");
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract("year").rename("pickup_year"),
                        scan.ref("trip_distance").cast("int32").rename("distance")})
                 .agg({0, 1, 2}, "count(*)")
                 .sort({{"pickup_year"s, "asc"s}, {"count"s, "desc"s}})
                 .finalize();

  run_compare_q4(std::move(dag));
}

class DISABLED_TPCH : public TestSuite {
 protected:
  static void SetUpTestSuite() {
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    parse_options.delimiter = '|';
    getStorage()->importCsvFile("part.tbl",
                                "part",
                                {{"p_partkey", ctx().int32(false)},
                                 {"p_name", ctx().extDict(ctx().text(), 0)},
                                 {"p_mfgr", ctx().extDict(ctx().text(), 0)},
                                 {"p_brand", ctx().extDict(ctx().text(), 0)},
                                 {"p_type", ctx().extDict(ctx().text(), 0)},
                                 {"p_size", ctx().int32()},
                                 {"p_container", ctx().extDict(ctx().text(), 0)},
                                 {"p_retailprice", ctx().decimal64(13, 2)},
                                 {"p_comment", ctx().text()}},
                                {20},
                                parse_options);
    getStorage()->importCsvFile("supplier.tbl",
                                "supplier",
                                {{"s_suppkey", ctx().int32(false)},
                                 {"s_name", ctx().extDict(ctx().text(), 0)},
                                 {"s_address", ctx().extDict(ctx().text(), 0)},
                                 {"s_nationkey", ctx().int32(false)},
                                 {"s_phone", ctx().extDict(ctx().text(), 0)},
                                 {"s_acctbal", ctx().decimal64(13, 2)},
                                 {"s_comment", ctx().text()}},
                                {5},
                                parse_options);
    getStorage()->importCsvFile("partsupp.tbl",
                                "partsupp",
                                {{"ps_partkey", ctx().int32(false)},
                                 {"ps_suppkey", ctx().int32(false)},
                                 {"ps_availqty", ctx().int32()},
                                 {"ps_supplycost", ctx().decimal64(13, 2)},
                                 {"ps_comment", ctx().text()}},
                                {80},
                                parse_options);
    getStorage()->importCsvFile("customer.tbl",
                                "customer",
                                {{"c_custkey", ctx().int32(false)},
                                 {"c_name", ctx().extDict(ctx().text(), 0)},
                                 {"c_address", ctx().extDict(ctx().text(), 0)},
                                 {"c_nationkey", ctx().int32(false)},
                                 {"c_phone", ctx().extDict(ctx().text(), 0)},
                                 {"c_acctbal", ctx().decimal64(13, 2)},
                                 {"c_mktsegment", ctx().extDict(ctx().text(), 0)},
                                 {"c_comment", ctx().text()}},
                                {15},
                                parse_options);
    getStorage()->importCsvFile("orders.tbl",
                                "orders",
                                {{"o_orderkey", ctx().int32(false)},
                                 {"o_custkey", ctx().int32(false)},
                                 {"o_orderstatus", ctx().extDict(ctx().text(), 0)},
                                 {"o_totalprice", ctx().decimal64(13, 2)},
                                 {"o_orderdate", ctx().date32()},
                                 {"o_orderpriority", ctx().extDict(ctx().text(), 0)},
                                 {"o_clerk", ctx().extDict(ctx().text(), 0)},
                                 {"o_shippriority", ctx().int32()},
                                 {"o_comment", ctx().text()}},
                                {150},
                                parse_options);
    getStorage()->importCsvFile("lineitem.tbl",
                                "lineitem",
                                {{"l_orderkey", ctx().int32(false)},
                                 {"l_partkey", ctx().int32(false)},
                                 {"l_suppkey", ctx().int32(false)},
                                 {"l_linenumber", ctx().int32()},
                                 {"l_quantity", ctx().decimal64(13, 2)},
                                 {"l_extendedprice", ctx().decimal64(13, 2)},
                                 {"l_discount", ctx().decimal64(13, 2)},
                                 {"l_tax", ctx().decimal64(13, 2)},
                                 {"l_returnflag", ctx().extDict(ctx().text(), 0)},
                                 {"l_linestatus", ctx().extDict(ctx().text(), 0)},
                                 {"l_shipdate", ctx().date32()},
                                 {"l_commitdate", ctx().date32()},
                                 {"l_receiptdate", ctx().date32()},
                                 {"l_shipinstruct", ctx().extDict(ctx().text(), 0)},
                                 {"l_shipmode", ctx().extDict(ctx().text(), 0)},
                                 {"l_comment", ctx().text()}},
                                {600},
                                parse_options);
    getStorage()->importCsvFile("nation.tbl",
                                "nation",
                                {{"n_nationkey", ctx().int32(false)},
                                 {"n_name", ctx().extDict(ctx().text(), 0)},
                                 {"n_regionkey", ctx().int32(false)},
                                 {"n_comment", ctx().text()}},
                                {5},
                                parse_options);
    getStorage()->importCsvFile("region.tbl",
                                "region",
                                {{"r_regionkey", ctx().int32(false)},
                                 {"r_name", ctx().extDict(ctx().text(), 0)},
                                 {"r_comment", ctx().text()}},
                                {1},
                                parse_options);
  }

  static void TearDownTestSuite() {}

  void compare_q1(const ExecutionResult& res) {
    compare_res_data(
        res,
        std::vector<std::string>({"A"s, "N"s, "N"s, "R"s}),
        std::vector<std::string>({"F"s, "F"s, "O"s, "F"s}),
        std::vector<int64_t>({3747400, 104100, 7516800, 3651100}),
        std::vector<int64_t>({3756962464, 104130107, 7538495537, 3657084124}),
        std::vector<int64_t>({356761920970, 9990608980, 716531663034, 347384728758}),
        std::vector<int64_t>(
            {37101416222424, 1036450802280, 74498798133073, 36169060112193}),
        std::vector<double>({25.354533152909337,
                             27.394736842105264,
                             25.558653519211152,
                             25.059025394646532}),
        std::vector<double>({25419.231826792962,
                             27402.659736842106,
                             25632.42277116627,
                             25100.09693891558}),
        std::vector<double>({0.0508660351826793,
                             0.04289473684210526,
                             0.049697381842910573,
                             0.05002745367192862}),
        std::vector<int32_t>({1478, 38, 2941, 1457}));
  }

  void compare_q3(const ExecutionResult& res) {
    compare_res_data(
        res,
        std::vector<int32_t>({1637, 5191, 742, 3492, 2883, 998, 3430, 4423}),
        std::vector<int64_t>({1642249253,
                              493783094,
                              437280480,
                              437160724,
                              366669612,
                              117855486,
                              47266775,
                              30559365}),
        std::vector<int64_t>(
            {dateTimeParse<hdk::ir::Type::kDate>("1995-02-08", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1994-12-11", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1994-12-23", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1994-11-24", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1995-01-23", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1994-11-26", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1994-12-12", TimeUnit::kMilli),
             dateTimeParse<hdk::ir::Type::kDate>("1995-02-17", TimeUnit::kMilli)}),
        std::vector<int32_t>({0, 0, 0, 0, 0, 0, 0, 0}));
  }
};

TEST_F(DISABLED_TPCH, Q1_SQL) {
  auto query = R"""(
    select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
    from
        lineitem
    where
        l_shipdate <= date '1998-12-01' - interval '90' day (3)
    group by
        l_returnflag,
        l_linestatus
    order by
        l_returnflag,
        l_linestatus;
  )""";
  auto res = runSqlQuery(query, ExecutorDeviceType::CPU, false);
  compare_q1(res);
}

TEST_F(DISABLED_TPCH, Q1_NoBuilder) {
  // Create 'lineitem' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "lineitem");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with required fields.
  auto returnflag_info = getStorage()->getColumnInfo(*table_info, "l_returnflag");
  auto returnflag_ref = makeExpr<hdk::ir::ColumnRef>(
      returnflag_info->type, scan.get(), returnflag_info->column_id - 1);
  auto linestatus_info = getStorage()->getColumnInfo(*table_info, "l_linestatus");
  auto linestatus_ref = makeExpr<hdk::ir::ColumnRef>(
      linestatus_info->type, scan.get(), linestatus_info->column_id - 1);
  auto quantity_info = getStorage()->getColumnInfo(*table_info, "l_quantity");
  auto quantity_ref = makeExpr<hdk::ir::ColumnRef>(
      quantity_info->type, scan.get(), quantity_info->column_id - 1);
  auto extendedprice_info = getStorage()->getColumnInfo(*table_info, "l_extendedprice");
  auto extendedprice_ref = makeExpr<hdk::ir::ColumnRef>(
      extendedprice_info->type, scan.get(), extendedprice_info->column_id - 1);
  auto discount_info = getStorage()->getColumnInfo(*table_info, "l_discount");
  auto discount_ref = makeExpr<hdk::ir::ColumnRef>(
      discount_info->type, scan.get(), discount_info->column_id - 1);
  auto tax_info = getStorage()->getColumnInfo(*table_info, "l_tax");
  auto tax_ref =
      makeExpr<hdk::ir::ColumnRef>(tax_info->type, scan.get(), tax_info->column_id - 1);
  auto shipdate_info = getStorage()->getColumnInfo(*table_info, "l_shipdate");
  auto shipdate_ref = makeExpr<hdk::ir::ColumnRef>(
      shipdate_info->type, scan.get(), shipdate_info->column_id - 1);
  auto proj = std::make_shared<Project>(ExprPtrVector{returnflag_ref,
                                                      linestatus_ref,
                                                      quantity_ref,
                                                      extendedprice_ref,
                                                      discount_ref,
                                                      tax_ref,
                                                      shipdate_ref},
                                        std::vector<std::string>{"l_returnflag",
                                                                 "l_linestatus",
                                                                 "l_quantity",
                                                                 "l_extendedprice",
                                                                 "l_discount",
                                                                 "l_tax",
                                                                 "l_shipdate"},
                                        scan);
  // Create filter.
  auto proj_shipdate_ref =
      makeExpr<hdk::ir::ColumnRef>(shipdate_info->type, proj.get(), 6);
  Datum d1;
  d1.stringval = new std::string("1998-12-01");
  auto date_cst = makeExpr<Constant>(ctx().text(), false, d1)->cast(ctx().date32());
  auto interval_cst = Constant::make(ctx().int64(), -90);
  auto date_cst2 =
      makeExpr<DateAddExpr>(ctx().date32(), DateAddField::kDay, interval_cst, date_cst);
  auto cond = makeExpr<BinOper>(
      ctx().boolean(), OpType::kLe, Qualifier::kOne, proj_shipdate_ref, date_cst2);
  auto filter = std::make_shared<Filter>(cond, proj);
  // Create aggregation.
  auto filter_discount =
      makeExpr<hdk::ir::ColumnRef>(discount_info->type, filter.get(), 4);
  auto mdiscount = makeExpr<BinOper>(discount_info->type,
                                     OpType::kMinus,
                                     Qualifier::kOne,
                                     Constant::make(ctx().int32(), 1),
                                     filter_discount);
  auto filter_tax = makeExpr<hdk::ir::ColumnRef>(tax_info->type, filter.get(), 5);
  auto inc_tax = makeExpr<BinOper>(tax_info->type,
                                   OpType::kPlus,
                                   Qualifier::kOne,
                                   Constant::make(ctx().int32(), 1),
                                   filter_tax);
  auto filter_extendedprice =
      makeExpr<hdk::ir::ColumnRef>(extendedprice_info->type, filter.get(), 3);
  auto mul1 = makeExpr<BinOper>(ctx().decimal64(26, 4),
                                OpType::kMul,
                                Qualifier::kOne,
                                filter_extendedprice,
                                mdiscount);
  auto mul2 = makeExpr<BinOper>(
      ctx().decimal64(39, 6), OpType::kMul, Qualifier::kOne, mul1, inc_tax);
  auto filter_quantity =
      makeExpr<hdk::ir::ColumnRef>(quantity_info->type, filter.get(), 2);
  auto sum1 = makeExpr<AggExpr>(
      filter_quantity->type(), AggType::kSum, filter_quantity, false, nullptr);
  auto sum2 = makeExpr<AggExpr>(
      filter_extendedprice->type(), AggType::kSum, filter_extendedprice, false, nullptr);
  auto sum3 = makeExpr<AggExpr>(mul1->type(), AggType::kSum, mul1, false, nullptr);
  auto sum4 = makeExpr<AggExpr>(mul2->type(), AggType::kSum, mul2, false, nullptr);
  auto avg1 =
      makeExpr<AggExpr>(ctx().fp64(), AggType::kAvg, filter_quantity, false, nullptr);
  auto avg2 = makeExpr<AggExpr>(
      ctx().fp64(), AggType::kAvg, filter_extendedprice, false, nullptr);
  auto avg3 =
      makeExpr<AggExpr>(ctx().fp64(), AggType::kAvg, filter_discount, false, nullptr);
  auto count = makeExpr<AggExpr>(ctx().int32(), AggType::kCount, nullptr, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      2,
      ExprPtrVector{sum1, sum2, sum3, sum4, avg1, avg2, avg3, count},
      std::vector<std::string>{"l_returnflag",
                               "l_linestatus",
                               "sum_qty",
                               "sum_base_price",
                               "sum_disc_price",
                               "sum_charge",
                               "avg_qty",
                               "avg_price",
                               "avg_disc",
                               "count_order"},
      filter);
  // Create sort.
  auto sort = std::make_shared<hdk::ir::Sort>(
      std::vector<hdk::ir::SortField>{
          {0, SortDirection::Ascending, NullSortedPosition::First},
          {1, SortDirection::Ascending, NullSortedPosition::First}},
      0,
      0,
      agg);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(sort);

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}

TEST_F(DISABLED_TPCH, Q1_1) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("lineitem");
  auto filter = scan.filter(scan.ref("l_shipdate")
                                .le(builder.cst("1998-12-01")
                                        .cast(ctx().date32())
                                        .sub(90, DateAddField::kDay)));
  auto disc_price =
      filter.ref("l_extendedprice").mul(builder.cst(1).sub(filter.ref("l_discount")));
  auto dag = filter
                 .agg({"l_returnflag", "l_linestatus"},
                      {filter.ref("l_quantity").sum().rename("sum_qty"),
                       filter.ref("l_extendedprice").sum().rename("sum_base_price"),
                       disc_price.sum().rename("sum_disc_price"),
                       disc_price.mul(builder.cst(1).add(filter.ref("l_tax")))
                           .sum()
                           .rename("sum_charge"),
                       filter.ref("l_quantity").avg().rename("avg_qty"),
                       filter.ref("l_extendedprice").avg().rename("avg_price"),
                       filter.ref("l_discount").avg().rename("avg_disc"),
                       builder.count().rename("count_order")})
                 .sort({{"l_returnflag"}, {"l_linestatus"}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}

TEST_F(DISABLED_TPCH, Q1_2) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("lineitem");
  auto filter =
      scan.filter(scan.ref("l_shipdate").le(builder.date("1998-12-01").sub(90, "day")));
  auto disc_price =
      filter.ref("l_extendedprice").mul(builder.cst(1).sub(filter.ref("l_discount")));
  auto charge = disc_price.mul(builder.cst(1).add(filter.ref("l_tax")));
  auto dag = filter
                 .agg({"l_returnflag", "l_linestatus"},
                      {filter.ref("l_quantity").sum().rename("sum_qty"),
                       filter.ref("l_extendedprice").sum().rename("sum_base_price"),
                       disc_price.sum().rename("sum_disc_price"),
                       charge.sum().rename("sum_charge"),
                       filter.ref("l_quantity").avg().rename("avg_qty"),
                       filter.ref("l_extendedprice").avg().rename("avg_price"),
                       filter.ref("l_discount").avg().rename("avg_disc"),
                       builder.count().rename("count_order")})
                 .sort({"l_returnflag", "l_linestatus"})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}

TEST_F(DISABLED_TPCH, Q1_3) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto scan = builder.scan("lineitem");
  auto filter =
      scan.filter(scan["l_shipdate"] <= builder.date("1998-12-01").sub(90, "day"));
  auto disc_price = filter["l_extendedprice"] * (1 - filter["l_discount"]);
  auto charge = disc_price * (1 + filter["l_tax"]);
  auto dag = filter
                 .agg({"l_returnflag", "l_linestatus"},
                      {filter.ref("l_quantity").sum().rename("sum_qty"),
                       filter.ref("l_extendedprice").sum().rename("sum_base_price"),
                       disc_price.sum().rename("sum_disc_price"),
                       charge.sum().rename("sum_charge"),
                       filter.ref("l_quantity").avg().rename("avg_qty"),
                       filter.ref("l_extendedprice").avg().rename("avg_price"),
                       filter.ref("l_discount").avg().rename("avg_disc"),
                       builder.count().rename("count_order")})
                 .sort({"l_returnflag", "l_linestatus"})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}

TEST_F(DISABLED_TPCH, Q3_SQL) {
  auto query = R"""(
    select
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        o_orderdate,
        o_shippriority
    from
        customer,
        orders,
        lineitem
    where
        c_mktsegment = 'BUILDING'
        and c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15'
        and l_shipdate > date '1995-03-15'
    group by
        l_orderkey,
        o_orderdate,
        o_shippriority
    order by
        revenue desc,
        o_orderdate
    limit 10;
  )""";
  auto res = runSqlQuery(query, ExecutorDeviceType::CPU, false);
  compare_q3(res);
}

TEST_F(DISABLED_TPCH, Q3_NoBuilder) {
  // Create scans.
  auto c_table_info = getStorage()->getTableInfo(TEST_DB_ID, "customer");
  auto c_col_infos = getStorage()->listColumns(*c_table_info);
  auto customer = std::make_shared<Scan>(c_table_info, std::move(c_col_infos));
  auto o_table_info = getStorage()->getTableInfo(TEST_DB_ID, "orders");
  auto o_col_infos = getStorage()->listColumns(*o_table_info);
  auto orders = std::make_shared<Scan>(o_table_info, std::move(o_col_infos));
  auto l_table_info = getStorage()->getTableInfo(TEST_DB_ID, "lineitem");
  auto l_col_infos = getStorage()->listColumns(*l_table_info);
  auto lineitem = std::make_shared<Scan>(l_table_info, std::move(l_col_infos));
  // Create joins.
  auto l_orderkey_info = getStorage()->getColumnInfo(*l_table_info, "l_orderkey");
  auto l_orderkey_ref = makeExpr<hdk::ir::ColumnRef>(
      l_orderkey_info->type, lineitem.get(), l_orderkey_info->column_id - 1);
  auto o_orderkey_info = getStorage()->getColumnInfo(*o_table_info, "o_orderkey");
  auto o_orderkey_ref = makeExpr<hdk::ir::ColumnRef>(
      o_orderkey_info->type, orders.get(), o_orderkey_info->column_id - 1);
  auto join1 = std::make_shared<Join>(
      lineitem,
      orders,
      makeExpr<BinOper>(
          ctx().boolean(), OpType::kEq, Qualifier::kOne, l_orderkey_ref, o_orderkey_ref),
      JoinType::INNER);

  auto o_custkey_info = getStorage()->getColumnInfo(*o_table_info, "o_custkey");
  auto o_custkey_ref =
      makeExpr<hdk::ir::ColumnRef>(o_custkey_info->type,
                                   join1.get(),
                                   o_custkey_info->column_id - 1 + lineitem->size());
  auto c_custkey_info = getStorage()->getColumnInfo(*c_table_info, "c_custkey");
  auto c_custkey_ref = makeExpr<hdk::ir::ColumnRef>(
      c_custkey_info->type, customer.get(), c_custkey_info->column_id - 1);
  auto join2 = std::make_shared<Join>(
      join1,
      customer,
      makeExpr<BinOper>(
          ctx().boolean(), OpType::kEq, Qualifier::kOne, o_custkey_ref, c_custkey_ref),
      JoinType::INNER);

  // Create filter.
  auto c_mktsegment_info = getStorage()->getColumnInfo(*c_table_info, "c_mktsegment");
  auto c_mktsegment_ref = makeExpr<hdk::ir::ColumnRef>(
      c_mktsegment_info->type,
      join2.get(),
      c_mktsegment_info->column_id - 1 + lineitem->size() + orders->size());
  Datum d1;
  d1.stringval = new std::string("BUILDING");
  auto str_cst = makeExpr<Constant>(ctx().text(), false, d1);
  auto c_cond = makeExpr<BinOper>(
      ctx().boolean(), OpType::kEq, Qualifier::kOne, c_mktsegment_ref, str_cst);
  // Create orders filter.
  auto o_orderdate_info = getStorage()->getColumnInfo(*o_table_info, "o_orderdate");
  auto o_orderdate_ref =
      makeExpr<hdk::ir::ColumnRef>(o_orderdate_info->type,
                                   join2.get(),
                                   o_orderdate_info->column_id - 1 + lineitem->size());
  Datum d2;
  d2.stringval = new std::string("1995-03-15");
  auto date_cst = makeExpr<Constant>(ctx().text(), false, d2)->cast(ctx().date32());
  auto o_cond = makeExpr<BinOper>(
      ctx().boolean(), OpType::kLt, Qualifier::kOne, o_orderdate_ref, date_cst);
  // Create lineitem filter.
  auto l_shipdate_info = getStorage()->getColumnInfo(*l_table_info, "l_shipdate");
  auto l_shipdate_ref = makeExpr<hdk::ir::ColumnRef>(
      l_shipdate_info->type, join2.get(), l_shipdate_info->column_id - 1);
  auto l_cond = makeExpr<BinOper>(
      ctx().boolean(), OpType::kGt, Qualifier::kOne, l_shipdate_ref, date_cst);
  auto cond =
      makeExpr<BinOper>(ctx().boolean(), OpType::kAnd, Qualifier::kOne, c_cond, o_cond);
  cond = makeExpr<BinOper>(ctx().boolean(), OpType::kAnd, Qualifier::kOne, cond, l_cond);
  auto filter = std::make_shared<Filter>(cond, join2);
  // Create projection with required fields.
  l_orderkey_ref = makeExpr<hdk::ir::ColumnRef>(
      l_orderkey_info->type, filter.get(), l_orderkey_info->column_id - 1);
  auto l_extendedprice_info =
      getStorage()->getColumnInfo(*l_table_info, "l_extendedprice");
  auto l_extendedprice_ref = makeExpr<hdk::ir::ColumnRef>(
      l_extendedprice_info->type, filter.get(), l_extendedprice_info->column_id - 1);
  auto l_discount_info = getStorage()->getColumnInfo(*l_table_info, "l_discount");
  auto l_discount_ref = makeExpr<hdk::ir::ColumnRef>(
      l_discount_info->type, filter.get(), l_discount_info->column_id - 1);
  o_orderdate_ref =
      makeExpr<hdk::ir::ColumnRef>(o_orderdate_info->type,
                                   filter.get(),
                                   o_orderdate_info->column_id - 1 + lineitem->size());
  auto o_shippriority_info = getStorage()->getColumnInfo(*o_table_info, "o_shippriority");
  auto o_shippriority_ref =
      makeExpr<hdk::ir::ColumnRef>(o_shippriority_info->type,
                                   filter.get(),
                                   o_shippriority_info->column_id - 1 + lineitem->size());
  auto proj = std::make_shared<Project>(
      ExprPtrVector{l_orderkey_ref,
                    o_orderdate_ref,
                    o_shippriority_ref,
                    l_extendedprice_ref,
                    l_discount_ref},
      std::vector<std::string>{
          "l_orderkey", "o_orderdate", "o_shippriority", "l_extendedprice", "l_discount"},
      filter);
  // Create aggregation.
  l_extendedprice_ref =
      makeExpr<hdk::ir::ColumnRef>(l_extendedprice_info->type, proj.get(), 3);
  l_discount_ref = makeExpr<hdk::ir::ColumnRef>(l_discount_info->type, proj.get(), 4);
  auto mdiscount = makeExpr<BinOper>(l_discount_info->type,
                                     OpType::kMinus,
                                     Qualifier::kOne,
                                     Constant::make(ctx().int32(), 1),
                                     l_discount_ref);
  auto mul1 = makeExpr<BinOper>(ctx().decimal64(26, 4),
                                OpType::kMul,
                                Qualifier::kOne,
                                l_extendedprice_ref,
                                mdiscount);
  auto sum1 = makeExpr<AggExpr>(mul1->type(), AggType::kSum, mul1, false, nullptr);
  auto agg = std::make_shared<Aggregate>(
      3,
      ExprPtrVector{sum1},
      std::vector<std::string>{"l_orderkey", "o_orderdate", "o_shippriority", "revenue"},
      proj);
  // Create projection to reorder columns.
  auto proj2 = std::make_shared<Project>(
      ExprPtrVector{getNodeColumnRef(agg.get(), 0),
                    getNodeColumnRef(agg.get(), 3),
                    getNodeColumnRef(agg.get(), 1),
                    getNodeColumnRef(agg.get(), 2)},
      std::vector<std::string>{"l_orderkey", "revenue", "o_orderdate", "o_shippriority"},
      agg);
  // Create sort.
  auto sort = std::make_shared<hdk::ir::Sort>(
      std::vector<hdk::ir::SortField>{
          {1, SortDirection::Descending, NullSortedPosition::First},
          {2, SortDirection::Ascending, NullSortedPosition::First}},
      0,
      0,
      proj2);
  // Create DAG.
  auto dag = std::make_unique<QueryDag>(configPtr());
  dag->setRootNode(sort);

  auto res = runQuery(std::move(dag));
  compare_q3(res);
}

TEST_F(DISABLED_TPCH, Q3_1) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto lineitem = builder.scan("lineitem");
  auto orders = builder.scan("orders");
  auto customer = builder.scan("customer");
  auto join1 =
      lineitem.join(orders, lineitem.ref("l_orderkey").eq(orders.ref("o_orderkey")));
  auto join2 = join1.join(customer, join1.ref("o_custkey").eq(customer.ref("c_custkey")));
  auto filter = join2.filter(
      join2.ref("c_mktsegment")
          .eq(builder.cst("BUILDING"))
          .logicalAnd(join2.ref("o_orderdate").lt(builder.date("1995-03-15")))
          .logicalAnd(join2.ref("l_shipdate").gt(builder.date("1995-03-15"))));
  auto revenue =
      filter.ref("l_extendedprice").mul(builder.cst(1).sub(filter.ref("l_discount")));
  auto dag = filter
                 .agg({"l_orderkey", "o_orderdate", "o_shippriority"},
                      {revenue.sum().rename("revenue")})
                 .proj({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
                 .sort({{"revenue"s, "desc"s}, {"o_orderdate"s, "asc"s}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q3(res);
}

TEST_F(DISABLED_TPCH, Q3_2) {
  QueryBuilder builder(ctx(), getStorage(), configPtr());
  auto join = builder.scan("lineitem")
                  .join(builder.scan("orders"), {"l_orderkey"}, {"o_orderkey"}, "inner")
                  .join(builder.scan("customer"), {"o_custkey"}, {"c_custkey"}, "inner");
  auto filter = join.filter(join["c_mktsegment"] == builder.cst("BUILDING") &&
                            join["o_orderdate"] < builder.date("1995-03-15") &&
                            join["l_shipdate"] > builder.date("1995-03-15"));
  auto revenue = filter["l_extendedprice"] * (1 - filter["l_discount"]);
  auto dag = filter
                 .agg({"l_orderkey", "o_orderdate", "o_shippriority"},
                      {revenue.sum().rename("revenue")})
                 .proj({0, 3, 1, 2})
                 .sort({{1, "desc"}, {2, "asc"}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q3(res);
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  ConfigBuilder builder;
  builder.parseCommandLineArgs(argc, argv, true);
  auto config = builder.config();

  // Enable table function. Must be done before init.
  g_enable_table_functions = true;

  int err{0};
  try {
    init(config);
    err = RUN_ALL_TESTS();
    reset();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  return err;
}
