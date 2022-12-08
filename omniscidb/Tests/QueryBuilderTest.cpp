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
#include "IR/QueryBuilder.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/QueryExecutionSequence.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SchemaMgr.h"
#include "SchemaMgr/SimpleSchemaProvider.h"
#include "Shared/DateTimeParser.h"

#include <gtest/gtest.h>

using namespace std::string_literals;
using namespace ArrowTestHelpers;
using namespace TestHelpers::ArrowSQLRunner;
using namespace hdk;
using namespace hdk::ir;

constexpr int TEST_TABLE_ID1 = 1;
constexpr int TEST_TABLE_ID2 = 2;
constexpr int TEST_TABLE_ID3 = 3;

extern bool g_enable_table_functions;

namespace {

std::string getFilePath(const std::string& file_name) {
  return std::string("../../Tests/ArrowStorageDataFiles/") + file_name;
}

class TestSuite : public ::testing::Test {
 public:
  ExecutionResult runQuery(std::unique_ptr<QueryDag> dag) {
    auto ra_executor = RelAlgExecutor(
        getExecutor(), getStorage(), getDataMgr()->getDataProvider(), std::move(dag));
    auto eo = ExecutionOptions::fromConfig(config());
    return ra_executor.executeRelAlgQuery(CompilationOptions(), eo, false);
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
              double val = HUGE_VAL) {
  ASSERT_TRUE(expr.expr()->is<hdk::ir::AggExpr>());
  auto agg = expr.expr()->as<hdk::ir::AggExpr>();
  ASSERT_EQ(agg->type()->toString(), type->toString());
  ASSERT_EQ(agg->aggType(), kind);
  ASSERT_EQ(agg->isDistinct(), is_distinct);
  ASSERT_EQ(expr.name(), name);
  if (val != HUGE_VAL) {
    ASSERT_TRUE(agg->arg1());
    ASSERT_NEAR(agg->arg1()->fpVal(), val, 0.001);
  }
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

void checkBinOper(const BuilderExpr& expr,
                  const Type* type,
                  OpType op_type,
                  const BuilderExpr& lhs,
                  const BuilderExpr& rhs) {
  ASSERT_TRUE(expr.expr()->is<BinOper>());
  auto bin_oper = expr.expr()->as<BinOper>();
  ASSERT_TRUE(bin_oper->type()->equal(type));
  ASSERT_EQ(bin_oper->opType(), op_type);
  ASSERT_EQ(bin_oper->qualifier(), Qualifier::kOne);
  ASSERT_EQ(bin_oper->leftOperand()->toString(), lhs.expr()->toString());
  ASSERT_EQ(bin_oper->rightOperand()->toString(), rhs.expr()->toString());
}

void checkCst(const BuilderExpr& expr, int64_t val, const Type* type) {
  ASSERT_TRUE(expr.expr()->is<Constant>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_TRUE(type->isInteger() || type->isDecimal() || type->isBoolean() ||
              type->isDateTime())
      << type->toString();
  ASSERT_EQ(expr.expr()->as<Constant>()->intVal(), val);
}

void checkCst(const BuilderExpr& expr, double val, const Type* type) {
  ASSERT_TRUE(expr.expr()->is<Constant>());
  ASSERT_EQ(expr.expr()->type()->toString(), type->toString());
  ASSERT_TRUE(type->isFloatingPoint());
  ASSERT_NEAR(expr.expr()->as<Constant>()->fpVal(), val, 0.0001);
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

}  // anonymous namespace

class QueryBuilderTest : public TestSuite {
 protected:
  static constexpr int TEST_SCHEMA_ID2 = 2;
  static constexpr int TEST_DB_ID2 = (TEST_SCHEMA_ID2 << 24) + 1;

  static void SetUpTestSuite() {
    auto data_mgr = getDataMgr();
    auto ps_mgr = data_mgr->getPersistentStorageMgr();
    storage2_ = std::make_shared<ArrowStorage>(TEST_SCHEMA_ID2, "test2", TEST_DB_ID2);
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID2, storage2_);
    schema_mgr_ = std::make_shared<SchemaMgr>();
    schema_mgr_->registerProvider(TEST_SCHEMA_ID, getStorage());
    schema_mgr_->registerProvider(TEST_SCHEMA_ID2, storage2_);

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
                    {"col_arr_i32_3", ctx().arrayFixed(3, ctx().int32())},
                    {"col_dec2", ctx().decimal64(5, 1)},
                    {"col_dec3", ctx().decimal64(14, 4)},
                    {"col_dict2", ctx().extDict(ctx().text(), -1)},
                    {"col_dict3", ctx().extDict(ctx().text(), -1)},
                    {"col_date2", ctx().date16(hdk::ir::TimeUnit::kDay)},
                    {"col_date3", ctx().date32(hdk::ir::TimeUnit::kSecond)},
                    {"col_date4", ctx().date64(hdk::ir::TimeUnit::kSecond)},
                    {"col_timestamp2", ctx().timestamp(hdk::ir::TimeUnit::kMilli)},
                    {"col_timestamp3", ctx().timestamp(hdk::ir::TimeUnit::kMicro)},
                    {"col_timestamp4", ctx().timestamp(hdk::ir::TimeUnit::kNano)},
                    {"col_si", ctx().int16()},
                    {"col_ti", ctx().int8()},
                    {"col_b_nn", ctx().boolean(false)},
                    {"col_vc_10", ctx().varChar(10)},
                });

    createTable("test4",
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
                });
    insertCsvValues("test4",
                    "10,10,2.2,4.4,12.34,true,str1,dict1,1970-01-02,15:00:11,"
                    "2022-02-23 15:00:11");
    insertCsvValues("test4",
                    "10,10,2.2,4.4,12.34,false,str1,dict1,2022-02-23,15:00:11,"
                    "2022-02-23 15:00:11");
    insertCsvValues("test4",
                    "10,10,2.2,4.4,12.34,,str1,dict1,2022-02-23,15:00:11,2022-"
                    "02-23 15:00:11");

    createTable("sort",
                {{"x", ctx().int32()}, {"y", ctx().int32()}, {"z", ctx().int32()}});
    insertCsvValues("sort",
                    "1,1,1\n2,1,2\n3,1,\n4,3,\n5,3,1\n,3,2\n9,2,1\n8,2,\n7,2,3\n6,2,2");

    createTable("ambiguous", {{"x", ctx().int32()}});
    storage2_->createTable("ambiguous", {{"x", ctx().int32()}});
  }

  static void TearDownTestSuite() {}

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
    auto proj =
        std::make_shared<Project>(getNodeColumnRefs(scan.get()),
                                  std::vector<std::string>({"x", "y", "z", "rowid"}),
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
  auto ref_arr_3 = scan.ref("col_arr_i32_3");
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
           "col_arr_i32_3_approx_count_dist");
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
           "col_arr_i32_3_sample");
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
           "col_arr_i32_3_single_value");
  EXPECT_THROW(ref_str.singleValue(), InvalidQueryError);
  EXPECT_THROW(ref_arr.singleValue(), InvalidQueryError);
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
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, -0.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 1.5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 1..5)"), InvalidQueryError);
  EXPECT_THROW(node.parseAgg("approx_quantile(col_i, 0.5, 1.5)"), InvalidQueryError);
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
  EXPECT_THROW(scan.ref("col_arr_i32_3").extract("day"), InvalidQueryError);
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
  ASSERT_TRUE(ctx().typeFromString("date")->equal(ctx().date(8, TimeUnit::kDay, true)));
  ASSERT_TRUE(ctx().typeFromString("date16")->equal(ctx().date(2, TimeUnit::kDay, true)));
  ASSERT_TRUE(ctx().typeFromString("date32")->equal(ctx().date(4, TimeUnit::kDay, true)));
  ASSERT_TRUE(
      ctx().typeFromString("date64[nn]")->equal(ctx().date(8, TimeUnit::kDay, false)));
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
  EXPECT_THROW(scan.ref("col_time").cast("int8"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("int16"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("int32"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_time").cast("int64"), InvalidQueryError);
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
  checkCast(scan.ref("col_timestamp").cast("fp32"), ctx().fp32());
  checkCast(scan.ref("col_timestamp").cast("fp64"), ctx().fp64());
  checkCast(scan.ref("col_timestamp").cast("dec(10,2)"), ctx().decimal(8, 10, 2));
  EXPECT_THROW(scan.ref("col_timestamp").cast("bool"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("text"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("varchar(10)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("dict(text)"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("time[s]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("time[ms]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("time[us]"), InvalidQueryError);
  EXPECT_THROW(scan.ref("col_timestamp").cast("time[ns]"), InvalidQueryError);
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

TEST_F(QueryBuilderTest, UserSql) {
  if (!config().debug.sql.empty()) {
    config().debug.dump = true;
    auto res = runSqlQuery(config().debug.sql, ExecutorDeviceType::CPU, false);
    auto at = toArrow(res);
    std::cout << at->ToString() << std::endl;
    config().debug.dump = false;
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
  compare_test1_data(scan1.proj(scan1.ref(2).name("c2")), {2}, {"c2"});
  compare_test1_data(scan1.proj({scan1.ref(1)}), {1});
  compare_test1_data(scan1.proj({scan1.ref(1).name("c1")}), {1}, {"c1"});
  compare_test1_data(scan1.proj({scan1.ref(-2)}), {3});
  compare_test1_data(scan1.proj({scan1.ref(-2).name("c")}), {3}, {"c"});
  compare_test1_data(scan1.proj(std::vector<BuilderExpr>({scan1.ref(3)})), {3});
  compare_test1_data(
      scan1.proj(std::vector<BuilderExpr>({scan1.ref(3).name("c1")})), {3}, {"c1"});
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
  EXPECT_THROW(scan1.proj({scan1.ref("col_bi").name("c"), scan1.ref("col_i").name("c")}),
               InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(scan1.ref(0)), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({0, 1}, {"c1", "c2", "c3"}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj({"col_bi"}, {}), InvalidQueryError);
  EXPECT_THROW(builder.scan("test1").proj(std::vector<int>(), {}), InvalidQueryError);
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
  EXPECT_THROW(scan.agg({scan.ref(0).name("id1"), scan.ref(0).name("id1")}, "count"),
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
  EXPECT_THROW(builder.scan("test3").sort("col_arr_i32_3"), InvalidQueryError);
  EXPECT_THROW(builder.scan("sort").sort(builder.scan("test3").ref(0)),
               InvalidQueryError);
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

TEST_F(Taxi, Q1_NoBuilder) {
  // SELECT cab_type, count(*) FROM trips GROUP BY cab_type;
  // Create 'trips' scan.
  auto table_info = getStorage()->getTableInfo(TEST_DB_ID, "trips");
  auto col_infos = getStorage()->listColumns(*table_info);
  auto scan = std::make_shared<Scan>(table_info, std::move(col_infos));
  // Create a projection with `cab_type` field at the front because it is
  // required for aggregation.
  auto cab_type_info = getStorage()->getColumnInfo(*table_info, "cab_type");
  auto cab_type_ref = makeExpr<hdk::ir::ColumnRef>(
      cab_type_info->type, scan.get(), cab_type_info->column_id - 1);
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
  auto passenger_count_info = getStorage()->getColumnInfo(*table_info, "passenger_count");
  auto passenger_count_ref = makeExpr<hdk::ir::ColumnRef>(
      passenger_count_info->type, scan.get(), passenger_count_info->column_id - 1);
  auto total_amount_info = getStorage()->getColumnInfo(*table_info, "total_amount");
  auto total_amount_ref = makeExpr<hdk::ir::ColumnRef>(
      total_amount_info->type, scan.get(), total_amount_info->column_id - 1);
  auto proj = std::make_shared<Project>(
      ExprPtrVector{passenger_count_ref, total_amount_ref},
      std::vector<std::string>{"passenger_count", "total_amount"},
      scan);
  // Create aggregation.
  auto total_amount_proj_ref =
      makeExpr<hdk::ir::ColumnRef>(total_amount_info->type, proj.get(), 1);
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
  auto passenger_count_info = getStorage()->getColumnInfo(*table_info, "passenger_count");
  auto passenger_count_ref = makeExpr<hdk::ir::ColumnRef>(
      passenger_count_info->type, scan.get(), passenger_count_info->column_id - 1);
  auto pickup_datetime_info = getStorage()->getColumnInfo(*table_info, "pickup_datetime");
  auto pickup_datetime_ref = makeExpr<hdk::ir::ColumnRef>(
      pickup_datetime_info->type, scan.get(), pickup_datetime_info->column_id - 1);
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
  auto dag = scan.proj({scan.ref("passenger_count"),
                        scan.ref("pickup_datetime").extract("year").name("pickup_year")})
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
  auto passenger_count_info = getStorage()->getColumnInfo(*table_info, "passenger_count");
  auto passenger_count_ref = makeExpr<hdk::ir::ColumnRef>(
      passenger_count_info->type, scan.get(), passenger_count_info->column_id - 1);
  auto pickup_datetime_info = getStorage()->getColumnInfo(*table_info, "pickup_datetime");
  auto pickup_datetime_ref = makeExpr<hdk::ir::ColumnRef>(
      pickup_datetime_info->type, scan.get(), pickup_datetime_info->column_id - 1);
  auto pickup_year = makeExpr<ExtractExpr>(
      ctx().int64(), false, DateExtractField::kYear, pickup_datetime_ref);
  auto trip_distance_info = getStorage()->getColumnInfo(*table_info, "trip_distance");
  auto trip_distance_ref = makeExpr<hdk::ir::ColumnRef>(
      trip_distance_info->type, scan.get(), trip_distance_info->column_id - 1);
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
                        scan.ref("pickup_datetime").extract("year").name("pickup_year"),
                        scan.ref("trip_distance").cast("int32").name("distance")})
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
                        scan.ref("pickup_datetime").extract("year").name("pickup_year"),
                        scan.ref("trip_distance").cast("int32").name("distance")})
                 .agg({0, 1, 2}, "count(*)")
                 .sort({{"pickup_year"s, "asc"s}, {"count"s, "desc"s}})
                 .finalize();

  run_compare_q4(std::move(dag));
}

class TPCH : public TestSuite {
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

TEST_F(TPCH, Q1_SQL) {
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

TEST_F(TPCH, Q1) {
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

#if 0
TEST_F(TPCH, Q1WithBuilder1) {
  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("lineitem");
  auto filter = scan.filter(scan.ref("l_shipdate")
                                .le(builder.cst("1998-12-01")
                                        .cast(ctx().date32())
                                        .minus(90, DateAddField::kDay)));
  auto disc_price =
      filter.ref("l_extendedprice").mul(builder.cst(1).minus(filter.ref("l_discount")));
  auto dag = filter
                 .agg({"l_returnflag", "l_linestatus"},
                      {filter.ref("l_quantity").sum(),
                       filter.ref("l_extendedprice").sum(),
                       disc_price.sum(),
                       disc_price.mul(builder.cst(1).plus(filter.ref("l_tax"))).sum(),
                       filter.ref("l_quantity").avg(),
                       filter.ref("l_extendedprice").avg(),
                       filter.ref("l_discount").avg(),
                       builder.count()},
                      {"l_returnflag",
                       "l_linestatus",
                       "sum_qty",
                       "sum_base_price",
                       "sum_disc_price",
                       "sum_charge",
                       "avg_qty",
                       "avg_price",
                       "avg_disc",
                       "count_order"})
                 .sort({{"l_returnflag"}, {"l_linestatus"}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}

TEST_F(TPCH, Q1WithBuilder2) {
  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("lineitem");
  auto filter =
      scan.filter(scan.ref("l_shipdate").le(builder.date("1998-12-01").minus(90, "day")));
  auto disc_price =
      filter.ref("l_extendedprice").mul(builder.cst(1).minus(filter.ref("l_discount")));
  auto charge = disc_price.mul(builder.cst(1).plus(filter.ref("l_tax")));
  auto dag = filter
                 .agg({"l_returnflag", "l_linestatus"},
                      {filter.ref("l_quantity").sum().name("sum_qty"),
                       filter.ref("l_extendedprice").sum().name("sum_base_price"),
                       disc_price.sum().name("sum_disc_price"),
                       charge.sum().name("sum_charge"),
                       filter.ref("l_quantity").avg().name("avg_qty"),
                       filter.ref("l_extendedprice").avg().name("avg_price"),
                       filter.ref("l_discount").avg().name("avg_disc"),
                       builder.count().name("count_order")})
                 .sort({"l_returnflag", "l_linestatus"})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q1(res);
}
#endif

TEST_F(TPCH, Q3_SQL) {
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

TEST_F(TPCH, Q3) {
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

#if 0
TEST_F(TPCH, Q3WithBuilder1) {
  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto lineitem = builder.scan("lineitem");
  auto orders = builder.scan("orders");
  auto customer = builder.scan("customer");
  auto join1 =
      lineitem.join(orders, lineitem.ref("l_orderkey").eq(orders.ref("o_orderkey")));
  auto join2 = join1.join(customer, join1.ref("o_custkey").eq(customer.ref("c_custkey")));
  auto filter = join2.filter(
      {join2.ref("c_mktsegment").eq(builder.cst("BUILDING")),
       join2.ref("o_orderdate").lt(builder.cst("1995-03-15").cast(ctx().date32())),
       join2.ref("l_shipdate").gt(builder.cst("1995-03-15").cast(ctx().date32()))});
  auto revenue =
      filter.ref("l_extendedprice").mul(builder.cst(1).minus(filter.ref("l_discount")));
  auto dag = filter
                 .agg({"l_orderkey", "o_orderdate", "o_shippriority"},
                      {revenue.sum()},
                      {"l_orderkey", "o_orderdate", "o_shippriority", "revenue"})
                 .proj({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
                 .sort({{"revenue", SortDirection::Descending}, {"o_orderdate"}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q3(res);
}

TEST_F(TPCH, Q3WithBuilder2) {
  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto join = builder.scan("lineitem")
                  .join(builder.scan("orders"), {"l_orderkey"}, {"o_orderkey"})
                  .join(builder.scan("customer"), {"o_custkey"}, {"c_custkey"});
  auto filter = join.filter({join.ref("c_mktsegment").eq(builder.cst("BUILDING")),
                             join.ref("o_orderdate").lt(builder.date("1995-03-15")),
                             join.ref("l_shipdate").gt(builder.date("1995-03-15"))});
  auto revenue =
      filter.ref("l_extendedprice").mul(builder.cst(1).minus(filter.ref("l_discount")));
  auto dag = filter
                 .agg({"l_orderkey", "o_orderdate", "o_shippriority"},
                      {revenue.sum().name("revenue")})
                 .proj({0, 3, 1, 2})
                 .sort({{1, SortDirection::Descending}, {2}})
                 .finalize();

  auto res = runQuery(std::move(dag));
  compare_q3(res);
}
#endif

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  ConfigBuilder builder;
  builder.parseCommandLineArgs(argc, argv, true);
  auto config = builder.config();

  // Avoid Calcite initialization for this suite.
  // config->debug.use_ra_cache = "dummy";
  // Enable table function. Must be done before init.
  g_enable_table_functions = true;

  // config->debug.dump = true;

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
