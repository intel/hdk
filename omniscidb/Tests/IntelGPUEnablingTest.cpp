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

  static void createGpuSortTestTable() {
    createTable("gpu_sort_test",
                {{"x", ctx().int64()},
                 {"y", ctx().int32()},
                 {"z", ctx().int16()},
                 {"t", ctx().int8()}},
                {2});
    run_sqlite_query("DROP TABLE IF EXISTS gpu_sort_test;");
    run_sqlite_query(
        "CREATE TABLE gpu_sort_test (x bigint, y int, z smallint, t tinyint);");
    TestHelpers::ValuesGenerator gen("gpu_sort_test");
    for (size_t i = 0; i < 4; ++i) {
      insertCsvValues("gpu_sort_test", "2,2,2,2");
      run_sqlite_query(gen(2, 2, 2, 2));
    }
    for (size_t i = 0; i < 6; ++i) {
      insertCsvValues("gpu_sort_test", "16000,16000,16000,127");
      run_sqlite_query(gen(16000, 16000, 16000, 127));
    }
  }

  static void createAndPopulateTestTables() {
    createTestInnerTable();
    createTestTable();
    createSmallTestsTable();
    createTestInnerLoopJoinTable();
    createGpuSortTestTable();
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

TEST_F(AggregationTest, StandaloneSum) {
  c("SELECT SUM(x) FROM test;", g_dt);
}

TEST_F(AggregationTest, SimpleAggregations) {
  c("SELECT COUNT(*) FROM test;", g_dt);
  c("SELECT COUNT(f) FROM test;", g_dt);
  c("SELECT COUNT(smallint_nulls), COUNT(*), COUNT(fn) FROM test;", g_dt);
  c("SELECT MIN(x) FROM test;", g_dt);
  c("SELECT MAX(x) FROM test;", g_dt);
  c("SELECT MIN(z) FROM test;", g_dt);
  c("SELECT MAX(z) FROM test;", g_dt);
  c("SELECT MIN(t) FROM test;", g_dt);
  c("SELECT MAX(t) FROM test;", g_dt);
  c("SELECT MIN(ff) FROM test;", g_dt);
  c("SELECT MIN(fn) FROM test;", g_dt);
  c("SELECT SUM(ff) FROM test;", g_dt);
  c("SELECT SUM(fn) FROM test;", g_dt);
  c("SELECT SUM(x + y) FROM test;", g_dt);
  c("SELECT SUM(x + y + z) FROM test;", g_dt);
  c("SELECT SUM(x + y + z + t) FROM test;", g_dt);
}

TEST_F(AggregationTest, FilterAndCount) {
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
    "1000 AND t < 1002;",
    g_dt);
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > "
    "1000 AND t < 1003);",
    g_dt);
  c("SELECT COUNT(*) FROM test WHERE x <> 7;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE z <> 102;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE t <> 1002;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x + y = 49;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x + y + z = 150;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x + y + z + t = 1151;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE CAST(x as TINYINT) + CAST(y as TINYINT) < CAST(z "
    "as TINYINT);",
    g_dt);
  c("SELECT COUNT(*) FROM test WHERE CAST(y as TINYINT) / CAST(x as TINYINT) = 6", g_dt);
  c("SELECT SUM(x + y) FROM test WHERE x + y = 49;", g_dt);
  c("SELECT SUM(x + y + z) FROM test WHERE x + y = 49;", g_dt);
  c("SELECT SUM(x + y + z + t) FROM test WHERE x + y = 49;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x - y = -35;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x - y + z = 66;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x - y + z + t = 1067;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE y - x = 35;", g_dt);
  c("SELECT 'Hello', 'World', 7 FROM test WHERE x <> 7;", g_dt);
  c("SELECT 'Total', COUNT(*) FROM test WHERE x <> 7;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 100;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 200;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 300;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 111.0;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 111.1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > 222.2;", g_dt);
}

TEST_F(AggregationTest, ComplexFilterAndCount) {
  c("SELECT COUNT(*) FROM test WHERE dd > CAST(111.0 AS decimal(10, 2));", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > CAST(222.0 AS decimal(10, 2));", g_dt);
  c("SELECT COUNT(*) FROM test WHERE dd > CAST(333.0 AS decimal(10, 2));", g_dt);
  c("SELECT COUNT(*) FROM test WHERE 1<>2;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE 1=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE 22 > 33;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE ff < 23.0/4.0 AND 22 < 33;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x + 3*8/2 < 35 + y - 20/5;", g_dt);
  c("SELECT x + 2 * 10/4 + 3 AS expr FROM test WHERE x + 3*8/2 < 35 + y - 20/5 ORDER "
    "BY expr ASC;",
    g_dt);
  c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 20.0/5;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < y AND 0=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < y AND 1=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < y OR 1<1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < y OR 1=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < 35 AND x < y AND 1=1 AND 0=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE 1>2 AND x < 35 AND x < y AND y < 10;", g_dt);
  c("SELECT COUNT(*) AS val FROM test WHERE (test.dd = 0.5 OR test.dd = 3);", g_dt);
}

TEST_F(AggregationTest, Sum) {
  c("SELECT SUM(x) FROM test;", g_dt);
  c("SELECT SUM(y) FROM test;", g_dt);
  c("SELECT SUM(x + y) FROM test;", g_dt);
  c("SELECT SUM(dd * x) FROM test;", g_dt);
  c("SELECT SUM(dd * y) FROM test;", g_dt);
  c("SELECT SUM(dd * w) FROM test;", g_dt);
  c("SELECT SUM(dd * z) FROM test;", g_dt);
  c("SELECT SUM(dd * t) FROM test;", g_dt);
}

TEST_F(AggregationTest, FilterAndSum) {
  c("SELECT SUM(x * dd) FROM test;", g_dt);
  c("SELECT SUM(y * dd) FROM test;", g_dt);
  c("SELECT SUM(w * dd) FROM test;", g_dt);
  c("SELECT SUM(z * dd) FROM test;", g_dt);
  c("SELECT SUM(t * dd) FROM test;", g_dt);
  c("SELECT SUM(dd * ufd) FROM test;", g_dt);
  c("SELECT SUM(dd * d) FROM test;", g_dt);
  c("SELECT SUM(dd * dn) FROM test;", g_dt);
  c("SELECT SUM(x * dd_notnull) FROM test;", g_dt);
  c("SELECT SUM(2 * x) FROM test WHERE x = 7;", g_dt);
  c("SELECT SUM(2 * x + z) FROM test WHERE x = 7;", g_dt);
  c("SELECT SUM(x + y) FROM test WHERE x - y = -35;", g_dt);
  c("SELECT SUM(x + y) FROM test WHERE y - x = 35;", g_dt);
  c("SELECT SUM(x + y - z) FROM test WHERE y - x = 35;", g_dt);
  c("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;", g_dt);
  c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", g_dt);
  c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", g_dt);
  c("SELECT SUM(z) FROM test WHERE z IS NOT NULL;", g_dt);
  c("SELECT SUM(dd) FROM test;", g_dt);
  c("SELECT SUM(dd * 0.99) FROM test;", g_dt);
}

TEST_F(AggregationTest, MinMax) {
  c("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;", g_dt);
  c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", g_dt);
  c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", g_dt);
  c("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;", g_dt);
  c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", g_dt);
  c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", g_dt);
  c("SELECT MIN(x) FROM test WHERE x = 7;", g_dt);
  c("SELECT MIN(z) FROM test WHERE z = 101;", g_dt);
  c("SELECT MIN(t) FROM test WHERE t = 1001;", g_dt);
  c("SELECT MIN(dd) FROM test;", g_dt);
  c("SELECT MAX(dd) FROM test;", g_dt);
  c("SELECT MAX(x + dd) FROM test;", g_dt);
  c("SELECT MAX(x + 2 * dd), MIN(x + 2 * dd) FROM test;", g_dt);
  c("SELECT MIN(dd * dd) FROM test;", g_dt);
  c("SELECT MAX(dd * dd) FROM test;", g_dt);
  c("SELECT MAX(dd_notnull * 1) FROM test;", g_dt);
}

TEST_F(AggregationTest, Average) {
  c("SELECT AVG(x + y) FROM test;", g_dt);
  c("SELECT AVG(x + y + z) FROM test;", g_dt);
  c("SELECT AVG(x + y + z + t) FROM test;", g_dt);
  c("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;", g_dt);
  c("SELECT AVG(y) FROM test WHERE z > 100 AND z < 102;", g_dt);
  c("SELECT AVG(y) FROM test WHERE t > 1000 AND t < 1002;", g_dt);
  c("SELECT AVG(dd) FROM test;", g_dt);
  c("SELECT AVG(dd) FROM test WHERE x > 6 AND x < 8;", g_dt);
  c("SELECT AVG(u * f) FROM test;", g_dt);
  c("SELECT AVG(u * d) FROM test;", g_dt);
}

TEST_F(AggregationTest, NegateSum) {
  c("SELECT SUM(-y) FROM test;", g_dt);
  c("SELECT SUM(-z) FROM test;", g_dt);
  c("SELECT SUM(-t) FROM test;", g_dt);
  c("SELECT SUM(-dd) FROM test;", g_dt);
  c("SELECT SUM(-f) FROM test;", g_dt);
  c("SELECT SUM(-d) FROM test;", g_dt);
}

TEST_F(AggregationTest, NullHandling) {
  c("SELECT COUNT(*) FROM test WHERE u IS NOT NULL;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE ofq >= 0 OR ofq IS NULL;", g_dt);
}

class OverflowTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(OverflowTest, OverflowAndUnderFlow) {
  c("SELECT dd * ufd FROM test;", g_dt);
  c("SELECT dd * -1 FROM test;", g_dt);
  c("SELECT -1 * dd FROM test;", g_dt);
  c("SELECT ofq * -1 FROM test;", g_dt);
  c("SELECT 1 * ofq FROM test;", g_dt);
  c("SELECT 566 * 244;", g_dt);
  EXPECT_THROW(run_multiple_agg("SELECT ofd * ofd FROM test;", g_dt), std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT 9223372036854775807 * 2;", g_dt),
               std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT -9223372036854775808 * 2;", g_dt),
               std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT 2*9223372036854775807;", g_dt),
               std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT -2*-9223372036854775808;", g_dt),
               std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT ofq * 2 FROM test;", g_dt), std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT ofq * -2 FROM test;", g_dt), std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT 2* ofq FROM test;", g_dt), std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT -2* ofq FROM test;", g_dt), std::runtime_error);
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

TEST_F(GroupByAggTest, AggHaving) {
  GTEST_SKIP();
  c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 0=1;", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 1=1;", g_dt);
  c("SELECT x, COUNT(*) AS n FROM test GROUP BY x, ufd ORDER BY x, n;", g_dt);
  c("SELECT MIN(x), MAX(x) FROM test WHERE real_str LIKE '%nope%';", g_dt);
  c("SELECT COUNT(*) FROM test WHERE (x > 7 AND y / (x - 7) < 44);", g_dt);
  c("SELECT x, AVG(ff) AS val FROM test GROUP BY x ORDER BY val;", g_dt);
  c("SELECT x, MAX(fn) as val FROM test WHERE fn IS NOT NULL GROUP BY x ORDER BY val;",
    g_dt);
  c("SELECT MAX(dn) FROM test WHERE dn IS NOT NULL;", g_dt);
  c("SELECT x, MAX(dn) as val FROM test WHERE dn IS NOT NULL GROUP BY x ORDER BY val;",
    g_dt);
  c("SELECT COUNT(*) as val FROM test GROUP BY x, y, ufd ORDER BY val;", g_dt);
}

class GroupByTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(GroupByTest, Basic) {
  c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", g_dt);
}

TEST_F(GroupByTest, WithFilter) {
  c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", g_dt);
  c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + "
    "y;",
    g_dt);
  c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", g_dt);
}

TEST_F(GroupByTest, WithOrdering) {
  GTEST_SKIP();
  c("SELECT x, dd, COUNT(*) FROM test GROUP BY x, dd ORDER BY x, dd;", g_dt);
  c("SELECT 'literal_string' AS key0 FROM test GROUP BY key0;", g_dt);
  c("SELECT str, MIN(y) FROM test WHERE y IS NOT NULL GROUP BY str ORDER BY str DESC;",
    g_dt);
  c("SELECT x, AVG(u), COUNT(*) AS n FROM test GROUP BY x ORDER BY n DESC;", g_dt);
  c("SELECT f, ss FROM test GROUP BY f, ss ORDER BY f DESC;", g_dt);
  c("SELECT fx, COUNT(*) n FROM test GROUP BY fx ORDER BY n DESC, fx IS NULL DESC;",
    g_dt);
}

TEST_F(GroupByTest, WithFilterHaving) {
  c("SELECT dd AS key1, COUNT(*) AS value1 FROM test GROUP BY key1 HAVING key1 IS NOT "
    "NULL ORDER BY key1, value1 "
    "DESC "
    "LIMIT 12;",
    g_dt);
  c("SELECT x, MAX(z) FROM test WHERE z IS NOT NULL GROUP BY x HAVING x > 7;", g_dt);
  c("SELECT CAST((dd - 0.5) * 2.0 AS int) AS key0, COUNT(*) AS val FROM test WHERE (dd "
    ">= 100.0 AND dd < 400.0) "
    "GROUP "
    "BY key0 HAVING key0 >= 0 AND key0 < 400 ORDER BY val DESC LIMIT 50 OFFSET 0;",
    g_dt);
  c("SELECT fx, COUNT(*) FROM test GROUP BY fx HAVING COUNT(*) > 5;", g_dt);
}

TEST_F(GroupByTest, WithCase) {
  c("SELECT y, AVG(CASE WHEN x BETWEEN 6 AND 7 THEN x END) FROM test GROUP BY y ORDER "
    "BY y;",
    g_dt);
  c("SELECT CASE WHEN x > 8 THEN 100000000 ELSE 42 END AS c, COUNT(*) FROM test GROUP "
    "BY c;",
    g_dt);
}

class BasicTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(BasicTest, SimpleFilterWithLiteral) {
  c("SELECT * FROM test WHERE x > 0;", g_dt);
}

TEST_F(BasicTest, Time) {
  ASSERT_EQ(
      static_cast<int64_t>(g_num_rows + g_num_rows / 2),
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", g_dt)));
  ASSERT_EQ(
      static_cast<int64_t>(g_num_rows + g_num_rows / 2),
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE CAST('10/09/1999' AS DATE) > o;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('10/09/99' AS DATE) > o;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('10-Sep-99' AS DATE) > o;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('9/10/99' AS DATE) > o;", g_dt)));
  ASSERT_EQ(
      static_cast<int64_t>(g_num_rows + g_num_rows / 2),
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE CAST('31/Oct/2013' AS DATE) > o;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('10/31/13' AS DATE) > o;", g_dt)));
  // check TIME FORMATS
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('151315' AS TIME) > n;", g_dt)));
}

TEST_F(BasicTest, Time2) {
  GTEST_SKIP();
  ASSERT_EQ(
      static_cast<int64_t>(g_num_rows + g_num_rows / 2),
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", g_dt)));
  ASSERT_EQ(
      0,
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) <= o;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", g_dt)));
  ASSERT_EQ(0,
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) <= n;", g_dt)));
  cta("SELECT NOW() FROM test limit 1;",
      "SELECT DATETIME('NOW') FROM test LIMIT 1;",
      g_dt);
  EXPECT_ANY_THROW(run_simple_agg("SELECT DATETIME(NULL) FROM test LIMIT 1;", g_dt));
  // these next tests work because all dates are before now 2015-12-8 17:00:00
  ASSERT_EQ(
      static_cast<int64_t>(2 * g_num_rows),
      v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m < NOW();", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE o IS NULL OR o < CURRENT_DATE;", g_dt)));
  ASSERT_EQ(
      static_cast<int64_t>(2 * g_num_rows),
      v<int64_t>(run_simple_agg(
          "SELECT COUNT(*) FROM test WHERE o IS NULL OR o < CURRENT_DATE();", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE m < CURRENT_TIMESTAMP;", g_dt)));
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE m < CURRENT_TIMESTAMP();", g_dt)));
  ASSERT_TRUE(v<int64_t>(
      run_simple_agg("SELECT CURRENT_DATE = CAST(CURRENT_TIMESTAMP AS DATE);", g_dt)));
  ASSERT_TRUE(
      v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, CURRENT_TIMESTAMP) < "
                                "CURRENT_DATE AND CURRENT_DATE <= CURRENT_TIMESTAMP;",
                                g_dt)));
  ASSERT_TRUE(v<int64_t>(run_simple_agg(
      "SELECT CAST(CURRENT_DATE AS TIMESTAMP) <= CURRENT_TIMESTAMP;", g_dt)));
  ASSERT_TRUE(v<int64_t>(run_simple_agg(
      "SELECT EXTRACT(YEAR FROM CURRENT_DATE) = EXTRACT(YEAR FROM CURRENT_TIMESTAMP)"
      " AND EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM CURRENT_TIMESTAMP)"
      " AND EXTRACT(DAY FROM CURRENT_DATE) = EXTRACT(DAY FROM CURRENT_TIMESTAMP)"
      " AND EXTRACT(HOUR FROM CURRENT_DATE) = 0"
      " AND EXTRACT(MINUTE FROM CURRENT_DATE) = 0"
      " AND EXTRACT(SECOND FROM CURRENT_DATE) = 0;",
      g_dt)));
  ASSERT_TRUE(v<int64_t>(run_simple_agg(
      "SELECT EXTRACT(HOUR FROM CURRENT_TIME()) = EXTRACT(HOUR FROM CURRENT_TIMESTAMP)"
      " AND EXTRACT(MINUTE FROM CURRENT_TIME) = EXTRACT(MINUTE FROM CURRENT_TIMESTAMP)"
      " AND EXTRACT(SECOND FROM CURRENT_TIME) = EXTRACT(SECOND FROM CURRENT_TIMESTAMP)"
      ";",
      g_dt)));
}

TEST_F(BasicTest, Timestamp) {
  ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';",
                g_dt)));
  ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
            v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                      "TIMESTAMP) > timestamp(0) '1999-09-08T160000';",
                                      g_dt)));
  ASSERT_EQ(0,
            v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                      "TIMESTAMP) > timestamp(0) '1999-09-10T160000';",
                                      g_dt)));
}

TEST_F(BasicTest, TimeExtract) {
  ASSERT_EQ(14185957950LL,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;",
                                      g_dt)));
  ASSERT_EQ(14185152000LL,
            v<int64_t>(run_simple_agg(
                "SELECT MAX(EXTRACT(DATEEPOCH FROM m) * 10) FROM test;", g_dt)));
  ASSERT_EQ(20140,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;",
                                      g_dt)));
  ASSERT_EQ(120,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;",
                                      g_dt)));
  ASSERT_EQ(140,
            v<int64_t>(
                run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", g_dt)));
  ASSERT_EQ(
      22,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", g_dt)));
  ASSERT_EQ(
      23,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", g_dt)));
  ASSERT_EQ(
      15,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", g_dt)));
  ASSERT_EQ(
      6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", g_dt)));
  ASSERT_EQ(
      348,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", g_dt)));
  ASSERT_EQ(
      15,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", g_dt)));
  ASSERT_EQ(
      13,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", g_dt)));
  ASSERT_EQ(
      14,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", g_dt)));
  ASSERT_EQ(
      1999,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", g_dt)));
  ASSERT_EQ(
      9,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", g_dt)));
  ASSERT_EQ(
      9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", g_dt)));
  ASSERT_EQ(4,
            v<int64_t>(run_simple_agg(
                "SELECT EXTRACT(DOW FROM o) FROM test WHERE o IS NOT NULL;", g_dt)));
  ASSERT_EQ(252,
            v<int64_t>(run_simple_agg(
                "SELECT EXTRACT(DOY FROM o) FROM test WHERE o IS NOT NULL;", g_dt)));
  ASSERT_EQ(
      936835200LL,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", g_dt)));
  ASSERT_EQ(936835200LL,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DATEEPOCH FROM o)) FROM test;",
                                      g_dt)));
  ASSERT_EQ(52LL,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2012-01-01 "
                                      "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(
      1LL,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SUNDAY FROM CAST('2012-01-01 "
                                "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                g_dt)));
  ASSERT_EQ(
      1LL,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SATURDAY FROM CAST('2012-01-01 "
                                "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                g_dt)));
  ASSERT_EQ(10LL,
            v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2008-03-03 "
                                      "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(
      10LL,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SUNDAY FROM CAST('2008-03-03 "
                                "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                g_dt)));
  ASSERT_EQ(
      10LL,
      v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK_SATURDAY FROM CAST('2008-03-03 "
                                "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                g_dt)));
  // Monday
  ASSERT_EQ(1LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-03 "
                                      "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                      g_dt)));
  // Monday
  ASSERT_EQ(1LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-03 "
                                      "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                      g_dt)));
  // Sunday
  ASSERT_EQ(0LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-02 "
                                      "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                      g_dt)));
  // Sunday
  ASSERT_EQ(7LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-02 "
                                      "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(15000000000LL,
            v<int64_t>(run_simple_agg(
                "SELECT EXTRACT(nanosecond from m) FROM test limit 1;", g_dt)));
  ASSERT_EQ(15000000LL,
            v<int64_t>(run_simple_agg(
                "SELECT EXTRACT(microsecond from m) FROM test limit 1;", g_dt)));
  ASSERT_EQ(15000LL,
            v<int64_t>(run_simple_agg(
                "SELECT EXTRACT(millisecond from m) FROM test limit 1;", g_dt)));
  ASSERT_EQ(56000000000LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(nanosecond from TIMESTAMP(0) "
                                      "'1999-03-14 23:34:56') FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(56000000LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(microsecond from TIMESTAMP(0) "
                                      "'1999-03-14 23:34:56') FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(56000LL,
            v<int64_t>(run_simple_agg("SELECT EXTRACT(millisecond from TIMESTAMP(0) "
                                      "'1999-03-14 23:34:56') FROM test limit 1;",
                                      g_dt)));
  ASSERT_EQ(2005,
            v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2005-12-31 "
                                      "23:59:59') from test limit 1;",
                                      g_dt)));
  ASSERT_EQ(1997,
            v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '1997-01-01 "
                                      "23:59:59') from test limit 1;",
                                      g_dt)));
  ASSERT_EQ(2006,
            v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2006-01-01 "
                                      "00:0:00') from test limit 1;",
                                      g_dt)));
  ASSERT_EQ(2014,
            v<int64_t>(run_simple_agg("select EXTRACT(year from TIMESTAMP '2014-01-01 "
                                      "00:00:00') from test limit 1;",
                                      g_dt)));
}

TEST_F(BasicTest, In) {
  c("SELECT COUNT(*) FROM test WHERE x IN (7, 8);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x IN (9, 10);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE z IN (101, 102);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE z IN (201, 202);", g_dt);
  c("SELECT COUNT(*) FROM test WHERE x IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "
    "14, 15, 16, 17, 18, 19, 20);",
    g_dt);
}

TEST_F(BasicTest, SumAndAverage) {
  c("SELECT AVG(ff) FROM test;", g_dt);
  c("SELECT AVG(w) FROM test;", g_dt);
  c("SELECT AVG(z) FROM test;", g_dt);
  c("SELECT SUM(CAST(x AS FLOAT)) FROM test GROUP BY y;", g_dt);
  // c("SELECT AVG(CAST(x AS FLOAT)) FROM test GROUP BY y;", g_dt);
  c("SELECT count(*) FROM test GROUP BY x;", g_dt);
  c("SELECT count(ff) FROM test GROUP BY x;", g_dt);
  c("SELECT x, AVG(ff) AS val FROM test GROUP BY x;", g_dt);
  c("SELECT SUM(x) FROM test GROUP BY z;", g_dt);
  c("SELECT SUM(CAST(x AS FLOAT)) FROM test GROUP BY z;", g_dt);
  c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 "
    "GROUP BY x, y;",
    g_dt);
  c("SELECT t + x, AVG(x) AS avg_x FROM test WHERE z <= 50 and t < 2000 GROUP BY t + x "
    "ORDER BY avg_x DESC",
    g_dt);
}

TEST_F(BasicTest, Distinct) {
  GTEST_SKIP();
  c("SELECT COUNT(distinct x) FROM test;", g_dt);
}

TEST_F(BasicTest, Sort) {
  c("SELECT x from test ORDER BY x ASC;", g_dt);
  c("SELECT x from test ORDER BY x DESC;", g_dt);
  c("SELECT COUNT(*) as val from test GROUP BY x ORDER BY val ASC;", g_dt);
  c("SELECT COUNT(*) as val from test GROUP BY x ORDER BY val DESC;", g_dt);
  c("SELECT x, COUNT(*) as val from test GROUP BY x ORDER BY val DESC;", g_dt);
  c("SELECT COUNT(*) as val from test GROUP BY x ORDER BY val ASC LIMIT 2;", g_dt);
  c("SELECT x, COUNT(*) AS val FROM gpu_sort_test GROUP BY x ORDER BY val DESC;", g_dt);
}

class FallbackTest : public ExecuteTestBase, public ::testing::Test {
 protected:
  void SetUp() override {
    config().exec.heterogeneous.allow_query_step_cpu_retry = true;
    config().exec.heterogeneous.allow_cpu_retry = true;
  }
  void TearDown() override {
    config().exec.heterogeneous.allow_query_step_cpu_retry = false;
    config().exec.heterogeneous.allow_cpu_retry = false;
  }
};

TEST_F(FallbackTest, InWithStrings) {
  c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar');", g_dt);
  c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar', 'real_baz', "
    "'foo');",
    g_dt);
  c("SELECT COUNT(*) FROM test WHERE str IN ('foo', 'bar', 'real_foo');", g_dt);
}

TEST_F(FallbackTest, Stddev) {
  // stddev_pop
  ASSERT_NEAR(static_cast<double>(0.5),
              v<double>(run_simple_agg("SELECT STDDEV_SAMP(x) FROM test;", g_dt)),
              static_cast<double>(0.2));

  ASSERT_NEAR(static_cast<double>(0.58),  // corr expansion
              v<double>(run_simple_agg("SELECT (avg(x * y) - avg(x) * avg(y)) /"
                                       "(stddev_pop(x) * stddev_pop(y)) FROM test;",
                                       g_dt)),
              static_cast<double>(0.01));
}

TEST_F(FallbackTest, PowerCorr) {
  ASSERT_NEAR(static_cast<double>(0.33),
              v<double>(run_simple_agg("SELECT POWER(CORR(x, y), 2) FROM test;", g_dt)),
              static_cast<double>(0.01));
}

int main(int argc, char* argv[]) {
  auto config = std::make_shared<Config>();
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  desc.add_options()("dump-ir",
                     po::value<bool>()->default_value(false)->implicit_value(true),
                     "Dump IR and PTX for all executed queries to file."
                     " Currently only supports single node tests.");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm["dump-ir"].as<bool>()) {
    // Only log IR, PTX channels to file with no rotation size.
    log_options.channels_ = {logger::Channel::IR, logger::Channel::PTX};
    log_options.rotation_size_ = std::numeric_limits<size_t>::max();
  }

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
