/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArrowTestHelpers.h"
#include "TestHelpers.h"

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "ConfigBuilder/ConfigBuilder.h"
#include "QueryBuilder/QueryBuilder.h"
#include "Shared/scope.h"

using namespace std::string_literals;
using namespace ArrowTestHelpers;
using namespace TestHelpers::ArrowSQLRunner;
using namespace hdk;
using namespace hdk::ir;

EXTERN extern bool g_enable_table_functions;

class PartitionedGroupByTest : public ::testing::Test {
 protected:
  static constexpr int TEST_SCHEMA_ID2 = 2;
  static constexpr int TEST_DB_ID2 = (TEST_SCHEMA_ID2 << 24) + 1;
  static constexpr size_t row_count = 20;
  static std::vector<int64_t> id1_vals;
  static std::vector<int32_t> id2_vals;
  static std::vector<int16_t> id3_vals;
  static std::vector<std::string> id4_vals;
  static std::vector<int32_t> v1_vals;
  static std::vector<int32_t> v2_vals;
  static std::vector<int64_t> v1_sums;
  static std::vector<int64_t> v2_sums;

  static void SetUpTestSuite() {
    createTable("test1",
                {{"id1", ctx().int64()},
                 {"id2", ctx().int32()},
                 {"id3", ctx().int16()},
                 {"id4", ctx().extDict(ctx().text(), 0)},
                 {"v1", ctx().int32()},
                 {"v2", ctx().int32()}},
                {row_count / 5});
    std::stringstream ss;
    for (size_t i = 1; i <= row_count; ++i) {
      auto val = i == row_count ? 1000000000000 : i;  // to avoid perfect hash
      id1_vals.push_back(val);
      id2_vals.push_back(i * 10);
      id3_vals.push_back(i * 100);
      id4_vals.push_back("str"s + ::std::to_string(i));
      v1_vals.push_back(i * 3);
      v2_vals.push_back(i * 111);
      v1_sums.push_back(i * 3);
      v2_sums.push_back(i * 111);
      ss << id1_vals.back() << "," << id2_vals.back() << "," << id3_vals.back() << ","
         << id4_vals.back() << "," << v1_vals.back() << "," << v2_vals.back()
         << std::endl;
    }
    insertCsvValues("test1", ss.str());
  }

  static void TearDownTestSuite() { dropTable("test1"); }
};

std::vector<int64_t> PartitionedGroupByTest::id1_vals;
std::vector<int32_t> PartitionedGroupByTest::id2_vals;
std::vector<int16_t> PartitionedGroupByTest::id3_vals;
std::vector<std::string> PartitionedGroupByTest::id4_vals;
std::vector<int32_t> PartitionedGroupByTest::v1_vals;
std::vector<int32_t> PartitionedGroupByTest::v2_vals;
std::vector<int64_t> PartitionedGroupByTest::v1_sums;
std::vector<int64_t> PartitionedGroupByTest::v2_sums;

TEST_F(PartitionedGroupByTest, SingleKey) {
  auto old_exec_groupby = config().exec.group_by;
  ScopeGuard g([&old_exec_groupby]() { config().exec.group_by = old_exec_groupby; });

  config().exec.group_by.default_max_groups_buffer_entry_guess = 1;
  config().exec.group_by.big_group_threshold = 1;
  config().exec.group_by.enable_cpu_partitioned_groupby = true;
  config().exec.group_by.partitioning_buffer_size_threshold = 10;
  config().exec.group_by.partitioning_group_size_threshold = 1.5;
  config().exec.group_by.min_partitions = 2;
  config().exec.group_by.max_partitions = 8;
  config().exec.group_by.partitioning_buffer_target_size = 200;
  config().exec.enable_multifrag_execution_result = true;

  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("test1");
  auto dag1 = scan.agg({"id1"s}, {"sum(v1)"s}).finalize();
  auto res1 = runQuery(std::move(dag1));
  // Check the result has 4 fragments (partitions).
  ASSERT_EQ(res1.getToken()->resultSetCount(), (size_t)4);
  auto dag2 = builder.scan(res1.tableName()).sort({0}).finalize();
  auto res2 = runQuery(std::move(dag2));
  compare_res_data(res2, id1_vals, v1_sums);
}

TEST_F(PartitionedGroupByTest, MultipleKeys) {
  auto old_exec = config().exec;
  ScopeGuard g([&old_exec]() { config().exec = old_exec; });

  config().exec.group_by.default_max_groups_buffer_entry_guess = 1;
  config().exec.group_by.big_group_threshold = 1;
  config().exec.group_by.enable_cpu_partitioned_groupby = true;
  config().exec.group_by.partitioning_buffer_size_threshold = 10;
  config().exec.group_by.partitioning_group_size_threshold = 1.5;
  config().exec.group_by.min_partitions = 2;
  config().exec.group_by.max_partitions = 8;
  config().exec.group_by.partitioning_buffer_target_size = 612;
  config().exec.enable_multifrag_execution_result = true;

  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("test1");
  auto dag1 =
      scan.agg({"id1"s, "id2"s, "id3"s, "id4"s}, {"sum(v1)"s, "sum(v2)"s}).finalize();
  auto res1 = runQuery(std::move(dag1));
  // Check the result has 4 fragments (partitions).
  ASSERT_EQ(res1.getToken()->resultSetCount(), (size_t)4);
  auto dag2 = builder.scan(res1.tableName()).sort({0, 1, 2, 3}).finalize();
  auto res2 = runQuery(std::move(dag2));
  compare_res_data(res2, id1_vals, id2_vals, id3_vals, id4_vals, v1_sums, v2_sums);
}

TEST_F(PartitionedGroupByTest, ReorderedKeys) {
  auto old_exec = config().exec;
  ScopeGuard g([&old_exec]() { config().exec = old_exec; });

  config().exec.group_by.default_max_groups_buffer_entry_guess = 1;
  config().exec.group_by.big_group_threshold = 1;
  config().exec.group_by.enable_cpu_partitioned_groupby = true;
  config().exec.group_by.partitioning_buffer_size_threshold = 10;
  config().exec.group_by.partitioning_group_size_threshold = 1.5;
  config().exec.group_by.min_partitions = 2;
  config().exec.group_by.max_partitions = 8;
  config().exec.group_by.partitioning_buffer_target_size = 612;
  config().exec.enable_multifrag_execution_result = true;

  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("test1");
  auto dag1 =
      scan.agg({"id4"s, "id2"s, "id1"s, "id3"s}, {"sum(v1)"s, "sum(v2)"s}).finalize();
  auto res1 = runQuery(std::move(dag1));
  // Check the result has 4 fragments (partitions).
  ASSERT_EQ(res1.getToken()->resultSetCount(), (size_t)4);
  auto dag2 = builder.scan(res1.tableName()).sort({2}).finalize();
  auto res2 = runQuery(std::move(dag2));
  compare_res_data(res2, id4_vals, id2_vals, id1_vals, id3_vals, v1_sums, v2_sums);
}

TEST_F(PartitionedGroupByTest, AggregationWithSort) {
  auto old_exec = config().exec;
  ScopeGuard g([&old_exec]() { config().exec = old_exec; });

  config().exec.group_by.default_max_groups_buffer_entry_guess = 1;
  config().exec.group_by.big_group_threshold = 1;
  config().exec.group_by.enable_cpu_partitioned_groupby = true;
  config().exec.group_by.partitioning_buffer_size_threshold = 10;
  config().exec.group_by.partitioning_group_size_threshold = 1.5;
  config().exec.group_by.min_partitions = 2;
  config().exec.group_by.max_partitions = 8;
  config().exec.group_by.partitioning_buffer_target_size = 612;
  config().exec.enable_multifrag_execution_result = true;

  QueryBuilder builder(ctx(), getSchemaProvider(), configPtr());
  auto scan = builder.scan("test1");
  auto dag1 = scan.agg({"id1"s, "id2"s, "id3"s, "id4"s}, {"sum(v1)"s, "sum(v2)"s})
                  .sort({0, 1, 2, 3})
                  .finalize();
  auto res = runQuery(std::move(dag1));
  compare_res_data(res, id1_vals, id2_vals, id3_vals, id4_vals, v1_sums, v2_sums);
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
