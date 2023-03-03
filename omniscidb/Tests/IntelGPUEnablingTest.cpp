/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "ArrowSQLRunner/SQLiteComparator.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

#include <string>

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

const size_t g_num_rows{10};
const ExecutorDeviceType g_dt{ExecutorDeviceType::GPU};

// todo: move to a separate file
struct ExecuteTestBase {
  static void createTestInnerTable() {
    createTable("test_inner",
                {{"x", ctx().int32(false)},
                 {"y", ctx().int32()},
                 {"xx", ctx().int16()},
                 {"str", ctx().extDict(ctx().text(), 0)},
                 {"dt", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"dt32", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"dt16", ctx().date16(hdk::ir::TimeUnit::kDay)},
                 {"ts", ctx().timestamp(hdk::ir::TimeUnit::kSecond)}},
                {2});
    run_sqlite_query("DROP TABLE IF EXISTS test_inner;");
    run_sqlite_query(
        "CREATE TABLE test_inner(x int not null, y int, xx smallint, str text, dt "
        "DATE, "
        "dt32 DATE, dt16 DATE, ts DATETIME);");
    {
      const std::string insert_query{
          "INSERT INTO test_inner VALUES(7, 43, 7, 'foo', '1999-09-09', '1999-09-09', "
          "'1999-09-09', '2014-12-13 22:23:15');"};
      insertCsvValues("test_inner",
                      "7,43,7,foo,1999-09-09,1999-09-09,1999-09-09,2014-12-13 22:23:15");
      run_sqlite_query(insert_query);
    }
    {
      const std::string insert_query{
          "INSERT INTO test_inner VALUES(-9, 72, -9, 'bars', '2014-12-13', '2014-12-13', "
          "'2014-12-13', '1999-09-09 14:15:16');"};
      insertCsvValues(
          "test_inner",
          "-9,72,-9,bars,2014-12-13,2014-12-13,2014-12-13,1999-09-09 14:15:16");
      run_sqlite_query(insert_query);
    }
  }

  static void createTestInnerLoopJoinTable() {
    createTable(
        "test_inner_loop_join",
        {{"x", ctx().int32(false)}, {"y", ctx().int32(false)}, {"xx", ctx().int16()}},
        {2});
    run_sqlite_query("DROP TABLE IF EXISTS test_inner_loop_join;");
    run_sqlite_query(
        "CREATE TABLE test_inner_loop_join(x int not null, y int not null, xx "
        "smallint);");

    run_sqlite_query("INSERT INTO test_inner_loop_join VALUES(7, 43, 12);");
    run_sqlite_query("INSERT INTO test_inner_loop_join VALUES(8, 2, 11);");
    run_sqlite_query("INSERT INTO test_inner_loop_join VALUES(9, 7, 10);");
    insertCsvValues("test_inner_loop_join", "7,43,2");
    insertCsvValues("test_inner_loop_join", "8,2,11");
    insertCsvValues("test_inner_loop_join", "9,7,10");
  }

  static void createSmallTestsTable() {
    createTable(
        "small_tests",
        {{"x", ctx().int32(false)}, {"y", ctx().int32(false)}, {"z", ctx().int32(false)}},
        {5});
    run_sqlite_query("DROP TABLE IF EXISTS small_tests;");
    run_sqlite_query(
        "CREATE TABLE small_tests(x int not null, y int not null, z int not null);");
    {
      const std::string insert_query{"INSERT INTO small_tests VALUES(7, 43, 2);"};
      run_sqlite_query(insert_query);
      insertCsvValues("small_tests", "7,43,2");
    }
    {
      const std::string insert_query{"INSERT INTO small_tests VALUES(9, 72, 1);"};
      run_sqlite_query(insert_query);
      insertCsvValues("small_tests", "9,72,1");
    }
    {
      const std::string insert_query{"INSERT INTO small_tests VALUES(0, 0, 2);"};
      run_sqlite_query(insert_query);
      insertCsvValues("small_tests", "0,0,2");
    }
    {
      const std::string insert_query{"INSERT INTO small_tests VALUES(1, 1, 1);"};
      run_sqlite_query(insert_query);
      insertCsvValues("small_tests", "1,1,1");
    }
    {
      const std::string insert_query{"INSERT INTO small_tests VALUES(2, 2, 2);"};
      run_sqlite_query(insert_query);
      insertCsvValues("small_tests", "2,2,2");
    }
  }

  static void createTestTable() {
    auto test_inner = getStorage()->getTableInfo(TEST_DB_ID, "test_inner");
    auto test_inner_str = getStorage()->getColumnInfo(*test_inner, "str");
    auto test_inner_str_type = test_inner_str->type;

    createTable("test",
                {{"x", ctx().int32(false)},
                 {"w", ctx().int8()},
                 {"y", ctx().int32()},
                 {"z", ctx().int16()},
                 {"t", ctx().int64()},
                 {"b", ctx().boolean()},
                 {"f", ctx().fp32()},
                 {"ff", ctx().fp32()},
                 {"fn", ctx().fp32()},
                 {"d", ctx().fp64()},
                 {"dn", ctx().fp64()},
                 {"str", test_inner_str_type},
                 {"null_str", ctx().extDict(ctx().text(), 0)},
                 {"fixed_str", ctx().extDict(ctx().text(), 0, 2)},
                 {"fixed_null_str", ctx().extDict(ctx().text(), 0, 2)},
                 {"real_str", ctx().text()},
                 {"shared_dict", test_inner_str_type},
                 {"m", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
                 {"m_3", ctx().timestamp(hdk::ir::TimeUnit::kMilli)},
                 {"m_6", ctx().timestamp(hdk::ir::TimeUnit::kMicro)},
                 {"m_9", ctx().timestamp(hdk::ir::TimeUnit::kNano)},
                 {"n", ctx().time64(hdk::ir::TimeUnit::kSecond)},
                 {"o", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"o1", ctx().date16(hdk::ir::TimeUnit::kDay)},
                 {"o2", ctx().date32(hdk::ir::TimeUnit::kDay)},
                 {"fx", ctx().int16()},
                 {"dd", ctx().decimal64(10, 2)},
                 {"dd_notnull", ctx().decimal64(10, 2, false)},
                 {"ss", ctx().extDict(ctx().text(), 0)},
                 {"u", ctx().int32()},
                 {"ofd", ctx().int32()},
                 {"ufd", ctx().int32(false)},
                 {"ofq", ctx().int64()},
                 {"ufq", ctx().int64(false)},
                 {"smallint_nulls", ctx().int16()},
                 {"bn", ctx().boolean(false)}},
                {2});
    run_sqlite_query("DROP TABLE IF EXISTS test;");
    run_sqlite_query(
        "CREATE TABLE test(x int not null, w tinyint, y int, z smallint, t bigint, b "
        "boolean, f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), m_3 timestamp(3), m_6 timestamp(6), m_9 timestamp(9), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null, "
        "smallint_nulls smallint, bn boolean not null);");

    CHECK_EQ(g_num_rows % 2, size_t(0));
    for (size_t i = 0; i < g_num_rows; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(7, -8, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
          "'foo', null, 'foo', null, "
          "'real_foo', 'foo',"
          "'2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '1999-07-11 "
          "14:02:53.874533', "
          "'2006-04-26 "
          "03:49:04.607435125', "
          "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, "
          "'fish', "
          "null, "
          "2147483647, -2147483648, null, -1, 32767, 't');"};
      insertCsvValues(
          "test",
          "7,-8,42,101,1001,true,1.1,1.1,,2.2,,foo,,foo,,real_foo,foo,2014-12-13 "
          "22:23:15,2014-12-13 22:23:15.323,1999-07-11 14:02:53.874533,2006-04-26 "
          "03:49:04.607435125,15:13:14,1999-09-09,1999-09-09,1999-09-09,9,111.1,111.1,"
          "fish,,2147483647,-2147483648,,-1,32767,true");
      run_sqlite_query(insert_query);
    }
    for (size_t i = 0; i < g_num_rows / 2; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(8, -7, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
          "-2002.4, 'bar', null, 'bar', null, "
          "'real_bar', NULL, '2014-12-13 22:23:15', '2014-12-13 22:23:15.323', "
          "'2014-12-13 "
          "22:23:15.874533', "
          "'2014-12-13 22:23:15.607435763', '15:13:14', NULL, NULL, NULL, NULL, 222.2, "
          "222.2, "
          "null, null, null, "
          "-2147483647, "
          "9223372036854775807, -9223372036854775808, null, 'f');"};
      insertCsvValues(
          "test",
          "8,-7,43,-78,1002,false,1.2,101.2,-101.2,2.4,-2002.4,bar,,bar,,real_bar,,2014-"
          "12-13 22:23:15,2014-12-13 22:23:15.323,2014-12-13 22:23:15.874533,2014-12-13 "
          "22:23:15.607435763,15:13:14,,,,,222.2,222.2,,,,-2147483647,"
          "9223372036854775807,-9223372036854775808,,false");
      run_sqlite_query(insert_query);
    }
    for (size_t i = 0; i < g_num_rows / 2; ++i) {
      const std::string insert_query{
          "INSERT INTO test VALUES(7, -7, 43, 102, 1002, null, 1.3, 1000.3, -1000.3, "
          "2.6, "
          "-220.6, 'baz', null, null, null, "
          "'real_baz', 'baz', '2014-12-14 22:23:15', '2014-12-14 22:23:15.750', "
          "'2014-12-14 22:23:15.437321', "
          "'2014-12-14 22:23:15.934567401', '15:13:14', '1999-09-09', '1999-09-09', "
          "'1999-09-09', 11, "
          "333.3, 333.3, "
          "'boat', null, 1, "
          "-1, 1, -9223372036854775808, 1, 't');"};
      insertCsvValues(
          "test",
          "7,-7,43,102,1002,,1.3,1000.3,-1000.3,2.6,-220.6,baz,,,,real_baz,baz,2014-12-"
          "14 22:23:15,2014-12-14 22:23:15.750,2014-12-14 22:23:15.437321,2014-12-14 "
          "22:23:15.934567401,15:13:14,1999-09-09,1999-09-09,1999-09-09,11,333.3,333.3,"
          "boat,,1,-1,1,-9223372036854775808,1,true");
      run_sqlite_query(insert_query);
    }
  }

  static void createAndPopulateTestTables() {
    createTestInnerTable();
    createTestTable();
    createSmallTestsTable();
    createTestInnerLoopJoinTable();
  }
};

class JoinTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(JoinTest, SimpleJoin) {
  c("SELECT a.x FROM test_inner_loop_join as a, test_inner_loop_join as b WHERE a.x > "
    "b.y ",
    g_dt);
}

class AggregationTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(AggregationTest, StandaloneCount) {
  c("SELECT COUNT(*) FROM test;", g_dt);
}

TEST_F(AggregationTest, StandaloneCountFilter) {
  c("SELECT COUNT(*) FROM small_tests WHERE x > y;", g_dt);
  c("SELECT COUNT(*) FROM small_tests WHERE y > z;", g_dt);
}

TEST_F(AggregationTest, StandaloneCountWithProjection) {
  c("SELECT COUNT(x) FROM test;", g_dt);
}

TEST_F(AggregationTest, ConsequentCount) {
  c("SELECT COUNT(*) FROM test;", g_dt);
  c("SELECT COUNT(*) FROM test;", g_dt);
}

TEST_F(AggregationTest, ConsequentCountWithProjection) {
  c("SELECT COUNT(x) FROM test;", g_dt);
  c("SELECT COUNT(x) FROM test;", g_dt);
}

TEST_F(AggregationTest, CountStarAfterCountWithProjection) {
  c("SELECT COUNT(x) FROM test;", g_dt);
  c("SELECT COUNT(*) FROM test;", g_dt);
}

TEST_F(AggregationTest, CountWithProjectionAfterCountStar) {
  c("SELECT COUNT(*) FROM test;", g_dt);
  c("SELECT COUNT(x) FROM test;", g_dt);
}

TEST_F(AggregationTest, Sum) {
  c("SELECT SUM(x) FROM test;", g_dt);
}

TEST_F(AggregationTest, Sum2) {
  c("SELECT SUM(x + y) FROM test;", g_dt);
}

TEST_F(AggregationTest, ConsequentSum) {
  c("SELECT SUM(x) FROM test;", g_dt);
  c("SELECT SUM(y) FROM test;", g_dt);
}

class GroupByAggTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(GroupByAggTest, GroupByCount) {
  c("SELECT COUNT(*) FROM small_tests GROUP BY z;", g_dt);
  c("SELECT COUNT(x) FROM small_tests GROUP BY z;", g_dt);
}

TEST_F(GroupByAggTest, GroupBySum) {
  GTEST_SKIP();
  c("SELECT SUM(x) FROM small_tests GROUP BY z;", g_dt);
  c("SELECT SUM(x + y) FROM small_tests GROUP BY z;", g_dt);
}

class BasicTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(BasicTest, SimpleFilter) {
  GTEST_SKIP();
  c("SELECT * FROM test WHERE x > 0;", g_dt);
}

int main(int argc, char* argv[]) {
  auto config = std::make_shared<Config>();
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::INFO;
  log_options.set_options();
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  logger::init(log_options);

  config->exec.heterogeneous.allow_query_step_cpu_retry = false;
  config->exec.heterogeneous.allow_cpu_retry = false;

  init(config);

  int result = 0;
  try {
    ExecuteTestBase::createAndPopulateTestTables();
    result = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Exception: " << e.what();
    return 1;
  }

  reset();
  return result;
}
