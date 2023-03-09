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
  GTEST_SKIP();
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

class BasicTest : public ExecuteTestBase, public ::testing::Test {};

TEST_F(BasicTest, SimpleFilterWithLiteral) {
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
