/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArrowTestHelpers.h"
#include "TestHelpers.h"
#include "TestRelAlgDagBuilder.h"

#include "ConfigBuilder/ConfigBuilder.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/QueryExecutionSequence.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SimpleSchemaProvider.h"

#include "ArrowSQLRunner/ArrowSQLRunner.h"

#include <gtest/gtest.h>

using namespace ArrowTestHelpers;
using namespace TestHelpers::ArrowSQLRunner;
using namespace hdk;
using namespace hdk::ir;

constexpr int TEST_TABLE_ID1 = 1;
constexpr int TEST_TABLE_ID2 = 2;
constexpr int TEST_TABLE_ID3 = 3;

extern bool g_enable_table_functions;

class ExecutionSequenceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    createTable("test1",
                {{"col_bi", ctx().int64()},
                 {"col_i", ctx().int32()},
                 {"col_f", ctx().fp32()},
                 {"col_d", ctx().fp64()}},
                {2});
    insertCsvValues("test1", "1,11,1.1,11.11\n2,22,2.2,22.22\n3,33,3.3,33.33");
    insertCsvValues("test1", "4,44,4.4,44.44\n5,55,5.5,55.55");

    createTable("test2",
                {{"id1", ctx().int32()},
                 {"id2", ctx().int32()},
                 {"val1", ctx().int32()},
                 {"val2", ctx().int32()}},
                {2});
    insertCsvValues("test2", "1,1,10,20\n1,2,11,21\n1,2,12,22\n2,1,13,23\n2,2,14,24");
    insertCsvValues("test2", "1,1,15,25\n1,2,,26\n1,2,17,27\n2,1,,28\n2,2,19,29");

    createTable("test3",
                {{"col_bi", ctx().int64()},
                 {"col_i", ctx().int32()},
                 {"col_f", ctx().fp32()},
                 {"col_d", ctx().fp64()}},
                {2});
    insertCsvValues("test3", "1,11,1.1,11.11\n2,22,2.2,22.22\n3,33,3.3,33.33");

    createTable("test4", {{"col_bi", ctx().int64()}, {"col_i", ctx().int32()}}, {2});
    insertCsvValues("test4", "2,122\n3,133\n4,144");

    createTable("test5",
                {{"col_bi", ctx().int64()},
                 {"col_i", ctx().int32()},
                 {"col_f", ctx().fp32()},
                 {"col_d", ctx().fp64()}},
                {2});
    insertCsvValues("test5", "1,11,1.1,11.11\n2,22,2.2,22.22\n3,33,3.3,33.33");

    createTable("array_test", {{"arr_float", ctx().arrayVarLen(ctx().fp32())}});
    insertJsonValues("array_test",
                     R"___({"arr_float": [0.0, 1.0]}
{"arr_float": [1.0, 0.0]}
{"arr_float": [0.0, 1.0]})___");

    createTable("test_str1",
                {{"i", ctx().int32()}, {"str", ctx().extDict(ctx().text(), 0)}});
    insertCsvValues("test_str1", "1,str1\n2,str2\n3,str3");

    createTable("test_str2",
                {{"i", ctx().int32()}, {"str", ctx().extDict(ctx().text(), 0)}});
    insertCsvValues("test_str2", "1,str1\n2,str2\n3,str3");
  }

  static void TearDownTestSuite() {}

  ExecutionResult runQuery(std::unique_ptr<QueryDag> dag,
                           bool legacy_work_units = false,
                           bool just_explain = false) {
    auto orig_use_legacy_work_unit_builder = config().exec.use_legacy_work_unit_builder;
    ScopeGuard g([&]() {
      config().exec.use_legacy_work_unit_builder = orig_use_legacy_work_unit_builder;
    });
    config().exec.use_legacy_work_unit_builder = legacy_work_units;

    auto ra_executor = RelAlgExecutor(
        getExecutor(), getStorage(), getDataMgr()->getDataProvider(), std::move(dag));
    auto eo = ExecutionOptions::fromConfig(config());
    eo.just_explain = just_explain;
    eo.allow_loop_joins = true;
    return ra_executor.executeRelAlgQuery(
        CompilationOptions::defaults(ExecutorDeviceType::CPU), eo, false);
  }
};

TEST_F(ExecutionSequenceTest, ProjectSequence) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(scan.get(), 1),
                                getNodeColumnRef(scan.get(), 2),
                                getNodeColumnRef(scan.get(), 3)});
  auto proj2 = dag->addProject(proj1,
                               {getNodeColumnRef(proj1.get(), 0),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(proj1.get(), 1),
                                                  Constant::make(ctx().int32(), 2)),
                                getNodeColumnRef(proj1.get(), 2),
                                getNodeColumnRef(proj1.get(), 3)});
  auto proj3 = dag->addProject(proj2,
                               {getNodeColumnRef(proj2.get(), 0),
                                getNodeColumnRef(proj2.get(), 1),
                                makeExpr<BinOper>(ctx().fp32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(proj2.get(), 2),
                                                  Constant::make(ctx().fp32(), 1)),
                                getNodeColumnRef(proj2.get(), 3)});
  auto proj4 = dag->addProject(proj3,
                               {getNodeColumnRef(proj3.get(), 0),
                                getNodeColumnRef(proj3.get(), 1),
                                getNodeColumnRef(proj3.get(), 2),
                                makeExpr<BinOper>(ctx().fp64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(proj3.get(), 3),
                                                  Constant::make(ctx().fp64(), 2))});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({2, 3, 4, 5, 6}),
                   std::vector<int>({13, 24, 35, 46, 57}),
                   std::vector<float>({2.1, 3.2, 4.3, 5.4, 6.5}),
                   std::vector<double>({13.11, 24.22, 35.33, 46.44, 57.55}));
}

TEST_F(ExecutionSequenceTest, FilterScan) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test1");
  auto filter1 = dag->addFilter(scan,
                                makeExpr<BinOper>(ctx().boolean(),
                                                  OpType::kEq,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 0),
                                                  Constant::make(ctx().int64(), 2)));
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({2}),
                   std::vector<int>({22}),
                   std::vector<float>({2.2}),
                   std::vector<double>({22.22}),
                   std::vector<int64_t>({1}));
}

TEST_F(ExecutionSequenceTest, ProjectFilterSequence) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(scan.get(), 1),
                                getNodeColumnRef(scan.get(), 2),
                                getNodeColumnRef(scan.get(), 3)});
  auto filter1 = dag->addFilter(proj1,
                                makeExpr<BinOper>(ctx().boolean(),
                                                  OpType::kNe,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(proj1.get(), 0),
                                                  Constant::make(ctx().int64(), 2)));
  auto proj2 = dag->addProject(filter1,
                               {getNodeColumnRef(filter1.get(), 0),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(filter1.get(), 1),
                                                  Constant::make(ctx().int32(), 2)),
                                getNodeColumnRef(filter1.get(), 2),
                                getNodeColumnRef(filter1.get(), 3)});
  auto filter2 = dag->addFilter(proj2,
                                makeExpr<BinOper>(ctx().boolean(),
                                                  OpType::kLe,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(proj2.get(), 1),
                                                  Constant::make(ctx().int32(), 50)));
  auto proj3 = dag->addProject(filter2,
                               {getNodeColumnRef(filter2.get(), 0),
                                getNodeColumnRef(filter2.get(), 1),
                                makeExpr<BinOper>(ctx().fp32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(filter2.get(), 2),
                                                  Constant::make(ctx().fp32(), 1)),
                                getNodeColumnRef(filter2.get(), 3)});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({3, 4, 5}),
                   std::vector<int>({24, 35, 46}),
                   std::vector<float>({3.2, 4.3, 5.4}),
                   std::vector<double>({22.22, 33.33, 44.44}));
}

TEST_F(ExecutionSequenceTest, ProjectSort) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto proj1 = dag->addProject(scan,
                               {getNodeColumnRef(scan.get(), 0),
                                getNodeColumnRef(scan.get(), 1),
                                getNodeColumnRef(scan.get(), 2),
                                getNodeColumnRef(scan.get(), 3)});
  auto sort = dag->addSort(
      proj1,
      {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last},
       {2, hdk::ir::SortDirection::Descending, hdk::ir::NullSortedPosition::First}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({1, 1, 1, 1, 1, 1, 2, 2, 2, 2}),
                   std::vector<int32_t>({2, 2, 1, 2, 2, 1, 1, 2, 2, 1}),
                   std::vector<int32_t>({inline_null_value<int32_t>(),
                                         17,
                                         15,
                                         12,
                                         11,
                                         10,
                                         inline_null_value<int32_t>(),
                                         19,
                                         14,
                                         13}),
                   std::vector<int32_t>({26, 27, 25, 22, 21, 20, 28, 29, 24, 23}));
}

TEST_F(ExecutionSequenceTest, SortByArray) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "array_test");
  auto proj1 = dag->addProject(scan, {getNodeColumnRef(scan.get(), 0)});
  auto sort = dag->addSort(
      proj1, {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  EXPECT_THROW(runQuery(std::move(dag)), std::runtime_error);
}

TEST_F(ExecutionSequenceTest, FilterNoGroupAggregate) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto filter1 =
      dag->addFilter(scan,
                     makeExpr<BinOper>(ctx().boolean(),
                                       OpType::kEq,
                                       Qualifier::kOne,
                                       makeExpr<BinOper>(ctx().int32(),
                                                         OpType::kPlus,
                                                         Qualifier::kOne,
                                                         getNodeColumnRef(scan.get(), 0),
                                                         getNodeColumnRef(scan.get(), 1)),
                                       Constant::make(ctx().int32(), 3)));
  auto proj1 = dag->addProject(
      filter1,
      {makeExpr<BinOper>(ctx().int32(),
                         OpType::kPlus,
                         Qualifier::kOne,
                         getNodeColumnRef(filter1.get(), 0),
                         makeExpr<BinOper>(ctx().int32(),
                                           OpType::kPlus,
                                           Qualifier::kOne,
                                           getNodeColumnRef(filter1.get(), 1),
                                           getNodeColumnRef(filter1.get(), 2)))});
  auto agg1 = dag->addAgg(proj1, 0, {{AggType::kSum, ctx().int32(), 0}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int32_t>({65}));
}

TEST_F(ExecutionSequenceTest, ProjectGroupAgg) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto proj1 = dag->addProject(scan,
                               {makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 0),
                                                  Constant::make(ctx().int32(), 1)),
                                getNodeColumnRef(scan.get(), 1),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 2),
                                                  Constant::make(ctx().int32(), 100)),
                                getNodeColumnRef(scan.get(), 3)});
  auto agg1 = dag->addAgg(proj1,
                          2,
                          {{AggType::kMax, ctx().int32(), 2},
                           {AggType::kSum, ctx().int64(), 3},
                           {AggType::kCount},
                           {AggType::kCount, ctx().int32(), 2}});
  auto sort = dag->addSort(
      agg1,
      {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last},
       {1, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({2, 2, 3, 3}),
                   std::vector<int32_t>({1, 2, 1, 2}),
                   std::vector<int32_t>({115, 117, 113, 119}),
                   std::vector<int64_t>({45, 96, 51, 53}),
                   std::vector<int32_t>({2, 4, 2, 2}),
                   std::vector<int32_t>({2, 3, 1, 2}));
}

TEST_F(ExecutionSequenceTest, AggSimpleProject) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto proj1 = dag->addProject(scan,
                               {makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 0),
                                                  Constant::make(ctx().int32(), 1)),
                                getNodeColumnRef(scan.get(), 1),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan.get(), 2),
                                                  Constant::make(ctx().int32(), 100)),
                                getNodeColumnRef(scan.get(), 3)});
  auto agg1 = dag->addAgg(proj1,
                          2,
                          {{AggType::kMax, ctx().int32(), 2},
                           {AggType::kSum, ctx().int64(), 3},
                           {AggType::kCount},
                           {AggType::kCount, ctx().int32(), 2}});
  auto proj2 = dag->addProject(agg1,
                               {getNodeColumnRef(agg1.get(), 1),
                                getNodeColumnRef(agg1.get(), 0),
                                getNodeColumnRef(agg1.get(), 3),
                                getNodeColumnRef(agg1.get(), 2),
                                getNodeColumnRef(agg1.get(), 5),
                                getNodeColumnRef(agg1.get(), 4)});
  auto sort = dag->addSort(
      proj2,
      {{1, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last},
       {0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({1, 2, 1, 2}),
                   std::vector<int32_t>({2, 2, 3, 3}),
                   std::vector<int64_t>({45, 96, 51, 53}),
                   std::vector<int32_t>({115, 117, 113, 119}),
                   std::vector<int32_t>({2, 3, 1, 2}),
                   std::vector<int32_t>({2, 4, 2, 2}));
}

TEST_F(ExecutionSequenceTest, AggScan1) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto agg1 = dag->addAgg(scan, 2, {{AggType::kMax, ctx().int32(), 2}});
  auto sort = dag->addSort(
      agg1,
      {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last},
       {1, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({1, 1, 2, 2}),
                   std::vector<int32_t>({1, 2, 1, 2}),
                   std::vector<int32_t>({15, 17, 13, 19}));
}

TEST_F(ExecutionSequenceTest, AggScan2) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test2");
  auto agg = dag->addAgg(scan, 2, {{AggType::kMax, ctx().int32(), 2}});
  auto proj = dag->addProject(agg, {getNodeColumnRef(agg.get(), 1)});
  auto sort = dag->addSort(
      proj, {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int32_t>({1, 1, 2, 2}));
}

TEST_F(ExecutionSequenceTest, LogicalValues) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  std::vector<TargetMetaInfo> tuple_type = {{"id", ctx().int32()},
                                            {"val", ctx().int64()}};
  std::vector<ExprPtrVector> values = {
      {Constant::make(ctx().int32(), 1), Constant::make(ctx().int64(), 11)},
      {Constant::make(ctx().int32(), 2), Constant::make(ctx().int64(), 22)},
      {Constant::make(ctx().int32(), 3), Constant::make(ctx().int64(), 33)}};
  auto logical_values = std::make_shared<LogicalValues>(tuple_type, values);
  dag->addNode(logical_values);
  dag->addProject(logical_values, {0, 1});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)2);

  auto res = runQuery(std::move(dag));
  compare_res_data(
      res, std::vector<int32_t>({1, 2, 3}), std::vector<int64_t>({11, 22, 33}));
}

TEST_F(ExecutionSequenceTest, LogicalUnion) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan1, {0, 1, 2, 3});
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto proj2 = dag->addProject(scan2, {0, 1, 2, 3});
  auto logical_union = std::make_shared<LogicalUnion>(NodeInputs{proj1, proj2}, true);
  dag->addNode(logical_union);
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)3);

  auto res = runQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>({1, 2, 3, 4, 5, 1, 2, 3}),
      std::vector<int32_t>({11, 22, 33, 44, 55, 11, 22, 33}),
      std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5, 1.1, 2.2, 3.3}),
      std::vector<double>({11.11, 22.22, 33.33, 44.44, 55.55, 11.11, 22.22, 33.33}));
}

TEST_F(ExecutionSequenceTest, TableFunction) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan1, {3, 3});
  std::vector<TargetMetaInfo> tuple_type = {{"val", ctx().fp64()}};
  auto table_fn = std::shared_ptr<TableFunction>(new TableFunction(
      "row_adder",
      NodeInputs{proj1},
      {""},
      {getNodeColumnRef(proj1.get(), 0), getNodeColumnRef(proj1.get(), 1)},
      {Constant::make(ctx().int32(), 1),
       getNodeColumnRef(proj1.get(), 0),
       getNodeColumnRef(proj1.get(), 1)},
      tuple_type));
  dag->addNode(table_fn);
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)2);

  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<double>({22.22, 44.44, 66.66, 88.88, 111.1}));
}

TEST_F(ExecutionSequenceTest, InnerJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto join = dag->addEquiJoin(scan1, scan2, JoinType::INNER, 0, 0);
  auto proj1 = dag->addProject(join, {0, 1, 2, 3, 6, 7, 8});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3}),
                   std::vector<int32_t>({11, 22, 33}),
                   std::vector<float>({1.1, 2.2, 3.3}),
                   std::vector<double>({11.11, 22.22, 33.33}),
                   std::vector<int32_t>({11, 22, 33}),
                   std::vector<float>({1.1, 2.2, 3.3}),
                   std::vector<double>({11.11, 22.22, 33.33}));
}

TEST_F(ExecutionSequenceTest, NestedInnerLeftJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto scan3 = dag->addScan(TEST_DB_ID, "test4");
  auto join1 = dag->addEquiJoin(scan1, scan2, JoinType::INNER, 0, 0);
  auto join2 = dag->addEquiJoin(join1, scan3, JoinType::LEFT, 0, 0);
  auto proj1 = dag->addProject(join2, {0, 1, 2, 3, 6, 7, 8, 11});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3}),
                   std::vector<int32_t>({11, 22, 33}),
                   std::vector<float>({1.1, 2.2, 3.3}),
                   std::vector<double>({11.11, 22.22, 33.33}),
                   std::vector<int32_t>({11, 22, 33}),
                   std::vector<float>({1.1, 2.2, 3.3}),
                   std::vector<double>({11.11, 22.22, 33.33}),
                   std::vector<int32_t>({inline_null_value<int32_t>(), 122, 133}));
}

TEST_F(ExecutionSequenceTest, JoinProjectJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto scan3 = dag->addScan(TEST_DB_ID, "test4");
  auto join1 = dag->addEquiJoin(scan1, scan2, JoinType::INNER, 0, 0);
  auto proj1 = dag->addProject(join1,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(join1.get(), 0),
                                                  Constant::make(ctx().int64(), 2)),
                                getNodeColumnRef(join1.get(), 1),
                                getNodeColumnRef(join1.get(), 7),
                                getNodeColumnRef(join1.get(), 8)});
  auto join2 = dag->addEquiJoin(proj1, scan3, JoinType::LEFT, 0, 0);
  auto proj2 = dag->addProject(join2, {0, 1, 2, 3, 5});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({3, 4, 5}),
                   std::vector<int32_t>({11, 22, 33}),
                   std::vector<float>({1.1, 2.2, 3.3}),
                   std::vector<double>({11.11, 22.22, 33.33}),
                   std::vector<int32_t>({133, 144, inline_null_value<int32_t>()}));
}

TEST_F(ExecutionSequenceTest, ProjectJoinProjectJoinProject) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan1,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan1.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(scan1.get(), 1),
                                getNodeColumnRef(scan1.get(), 2),
                                getNodeColumnRef(scan1.get(), 3)});
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto join1 = dag->addEquiJoin(proj1, scan2, JoinType::INNER, 0, 0);
  auto proj2 = dag->addProject(join1,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(join1.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(join1.get(), 1),
                                getNodeColumnRef(join1.get(), 6),
                                getNodeColumnRef(join1.get(), 7)});
  auto scan3 = dag->addScan(TEST_DB_ID, "test4");
  auto join2 = dag->addEquiJoin(proj2, scan3, JoinType::LEFT, 0, 0);
  auto proj3 = dag->addProject(join2,
                               {getNodeColumnRef(join2.get(), 0),
                                getNodeColumnRef(join2.get(), 1),
                                getNodeColumnRef(join2.get(), 2),
                                getNodeColumnRef(join2.get(), 3),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(join2.get(), 5),
                                                  Constant::make(ctx().int32(), 100))});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({3, 4}),
                   std::vector<int32_t>({11, 22}),
                   std::vector<float>({2.2, 3.3}),
                   std::vector<double>({22.22, 33.33}),
                   std::vector<int32_t>({233, 244}));
}

TEST_F(ExecutionSequenceTest, JoinsAggregate) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan1,
                               {getNodeColumnRef(scan1.get(), 0),
                                makeExpr<BinOper>(ctx().int32(),
                                                  OpType::kMinus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan1.get(), 1),
                                                  Constant::make(ctx().int32(), 10)),
                                getNodeColumnRef(scan1.get(), 2),
                                getNodeColumnRef(scan1.get(), 3)});
  auto scan2 = dag->addScan(TEST_DB_ID, "test2");
  auto join1 = dag->addEquiJoin(proj1, scan2, JoinType::INNER, 0, 0);
  auto proj2 = dag->addProject(join1,
                               {getNodeColumnRef(join1.get(), 0),
                                getNodeColumnRef(join1.get(), 5),
                                getNodeColumnRef(join1.get(), 1),
                                getNodeColumnRef(join1.get(), 6)});
  auto agg1 = dag->addAgg(proj2,
                          2,
                          {{AggType::kSum, ctx().int32(), 2},
                           {AggType::kMax, ctx().int32(), 3},
                           {AggType::kCount},
                           {AggType::kCount, ctx().int32(), 3}});
  auto sort = dag->addSort(
      agg1,
      {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last},
       {1, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();
  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 1, 2, 2}),
                   std::vector<int32_t>({1, 2, 1, 2}),
                   std::vector<int32_t>({2, 4, 24, 24}),
                   std::vector<int32_t>({15, 17, 13, 19}),
                   std::vector<int32_t>({2, 4, 2, 2}),
                   std::vector<int32_t>({2, 3, 1, 2}));
}

TEST_F(ExecutionSequenceTest, JoinOfProjections) {
  auto orig_materialize_inner_join_tables = config().exec.materialize_inner_join_tables;
  config().exec.materialize_inner_join_tables = false;
  ScopeGuard g([&]() {
    config().exec.materialize_inner_join_tables = orig_materialize_inner_join_tables;
  });

  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan1,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kPlus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan1.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(scan1.get(), 1)});
  auto scan2 = dag->addScan(TEST_DB_ID, "test4");
  auto proj2 = dag->addProject(scan2,
                               {makeExpr<BinOper>(ctx().int64(),
                                                  OpType::kMinus,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(scan2.get(), 0),
                                                  Constant::make(ctx().int64(), 1)),
                                getNodeColumnRef(scan2.get(), 1)});
  auto join1 = dag->addEquiJoin(proj1, proj2, JoinType::INNER, 0, 0);
  auto proj3 = dag->addProject(join1,
                               {getNodeColumnRef(join1.get(), 0),
                                getNodeColumnRef(join1.get(), 1),
                                getNodeColumnRef(join1.get(), 3)});
  dag->finalize();
  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({2, 3}),
                   std::vector<int32_t>({11, 22}),
                   std::vector<int32_t>({133, 144}));
}

TEST_F(ExecutionSequenceTest, FilteredJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test4");
  auto join1 = dag->addEquiJoin(scan1, scan2, JoinType::INNER, 0, 0);
  auto filter1 = dag->addFilter(join1,
                                makeExpr<BinOper>(ctx().boolean(),
                                                  OpType::kGt,
                                                  Qualifier::kOne,
                                                  getNodeColumnRef(join1.get(), 6),
                                                  getNodeColumnRef(join1.get(), 1)));
  auto proj3 = dag->addProject(filter1,
                               {getNodeColumnRef(filter1.get(), 0),
                                getNodeColumnRef(filter1.get(), 1),
                                getNodeColumnRef(filter1.get(), 2),
                                getNodeColumnRef(filter1.get(), 3),
                                getNodeColumnRef(filter1.get(), 6)});
  dag->finalize();
  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({2, 3, 4}),
                   std::vector<int32_t>({22, 33, 44}),
                   std::vector<float>({2.2, 3.3, 4.4}),
                   std::vector<double>({22.22, 33.33, 44.44}),
                   std::vector<int32_t>({122, 133, 144}));
}

TEST_F(ExecutionSequenceTest, TwoStepExplain) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test1");
  auto proj1 = dag->addProject(scan, {getNodeColumnRef(scan.get(), 0)});
  auto agg1 = dag->addAgg(proj1, 1, {{AggType::kCount}});
  auto proj2 = dag->addProject(
      agg1, {getNodeColumnRef(agg1.get(), 1), getNodeColumnRef(agg1.get(), 0)});
  auto agg2 = dag->addAgg(proj2, 1, {{AggType::kCount}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)2);

  auto res = runQuery(std::move(dag), false, true);
  EXPECT_EQ(res.getRows()->rowCount(), (size_t)1);
}

TEST_F(ExecutionSequenceTest, JoinThreeScansFilterAggregate) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test_str1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto scan3 = dag->addScan(TEST_DB_ID, "test_str2");
  auto join1 = dag->addJoin(scan1,
                            scan2,
                            JoinType::LEFT,
                            makeExpr<BinOper>(ctx().boolean(),
                                              OpType::kGt,
                                              Qualifier::kOne,
                                              getNodeColumnRef(scan1.get(), 0),
                                              getNodeColumnRef(scan2.get(), 0)));
  auto join2 = dag->addJoin(join1,
                            scan3,
                            JoinType::INNER,
                            makeExpr<BinOper>(ctx().boolean(),
                                              OpType::kNe,
                                              Qualifier::kOne,
                                              getNodeColumnRef(join1.get(), 1),
                                              getNodeColumnRef(scan3.get(), 1)));
  auto proj = dag->addProject(join2, {4, 8});
  auto agg = dag->addAgg(proj, 2, {{AggType::kCount}});
  auto sort = dag->addSort(
      agg,
      {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::First},
       {1, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::First}});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  auto inull = inline_null_value<int32_t>();
  compare_res_data(res,
                   std::vector<int32_t>({inull, inull, 11, 11, 11, 22, 22}),
                   std::vector<int32_t>({2, 3, 1, 2, 3, 1, 2}),
                   std::vector<int32_t>({1, 1, 2, 1, 1, 1, 1}));
}

TEST_F(ExecutionSequenceTest, Limit0) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan = dag->addScan(TEST_DB_ID, "test1");
  auto proj = dag->addProject(scan, {getNodeColumnRef(scan.get(), 0)});
  auto sort = dag->addSort(proj, {});
  std::dynamic_pointer_cast<Sort>(sort)->setEmptyResult(true);
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)1);

  auto res = runQuery(std::move(dag));
  compare_res_data(res, std::vector<int64_t>());
}

TEST_F(ExecutionSequenceTest, JoinOfJoins) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(getStorage(), configPtr());
  auto scan1 = dag->addScan(TEST_DB_ID, "test1");
  auto scan2 = dag->addScan(TEST_DB_ID, "test3");
  auto scan3 = dag->addScan(TEST_DB_ID, "test5");
  auto join1 = dag->addEquiJoin(scan1, scan2, JoinType::INNER, 1, 1);
  auto join2 = dag->addEquiJoin(scan1, scan3, JoinType::INNER, 1, 1);
  auto join3 = dag->addJoin(
      join1,
      join2,
      JoinType::INNER,
      makeExpr<BinOper>(ctx().boolean(),
                        OpType::kAnd,
                        Qualifier::kOne,
                        makeExpr<BinOper>(ctx().boolean(),
                                          OpType::kEq,
                                          Qualifier::kOne,
                                          getNodeColumnRef(join1.get(), 1),
                                          getNodeColumnRef(join2.get(), 1)),
                        makeExpr<BinOper>(ctx().boolean(),
                                          OpType::kEq,
                                          Qualifier::kOne,
                                          getNodeColumnRef(join1.get(), 5),
                                          getNodeColumnRef(join2.get(), 5))));

  auto proj3 = dag->addProject(join3, std::vector<int>{1, 5});
  dag->finalize();

  QueryExecutionSequence new_seq(dag->getRootNode(), configPtr());
  CHECK_EQ(new_seq.size(), (size_t)2);

  auto res = runQuery(std::move(dag));
  compare_res_data(
      res, std::vector<int32_t>({11, 22, 33}), std::vector<int64_t>({1, 2, 3}));
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  ConfigBuilder builder;
  builder.parseCommandLineArgs(argc, argv, true);
  auto config = builder.config();

  // Avoid Calcite initialization for this suite.
  config->debug.use_ra_cache = "dummy";
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
