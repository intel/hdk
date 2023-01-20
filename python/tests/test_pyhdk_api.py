#!/usr/bin/env python3

#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0


import json
import pandas
import pyarrow
import pytest
import pyhdk


class BaseTest:
    @staticmethod
    def check_taxi_q1_res(res):
        df = res.to_arrow().to_pandas()
        assert df["cab_type"].tolist() == ["green"]
        assert df["cnt"].tolist() == [20]

    @staticmethod
    def check_taxi_q2_res(res):
        df = res.to_arrow().to_pandas()
        assert df["passenger_count"].tolist() == [1, 2, 5]
        assert df["total_amount_avg"].tolist() == [98.19 / 16, 75.0, 13.58 / 3]

    @staticmethod
    def check_taxi_q3_res(res):
        df = res.to_arrow().to_pandas()
        assert df["passenger_count"].tolist() == [1, 2, 5]
        assert df["pickup_year"].tolist() == [2013, 2013, 2013]
        assert df["cnt"].tolist() == [16, 1, 3]

    @staticmethod
    def check_taxi_q4_res(res):
        df = res.to_arrow().to_pandas()
        assert df["passenger_count"].tolist() == [1, 5, 2]
        assert df["pickup_year"].tolist() == [2013, 2013, 2013]
        assert df["distance"].tolist() == [0, 0, 0]
        assert df["cnt"].tolist() == [16, 3, 1]


class TestTaxiSql(BaseTest):
    def test_taxi_over_csv_modular(self):
        # Initialize HDK components
        config = pyhdk.buildConfig()
        storage = pyhdk.storage.ArrowStorage(1)
        data_mgr = pyhdk.storage.DataMgr(config)
        data_mgr.registerDataProvider(storage)
        calcite = pyhdk.sql.Calcite(storage, config)
        executor = pyhdk.Executor(data_mgr, config)

        # Import data
        storage.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv",
            "trips",
        )

        # Run Taxi Q1 SQL query
        ra = calcite.process(
            "SELECT cab_type, count(*) as cnt FROM trips GROUP BY cab_type;"
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, ra)
        res = rel_alg_executor.execute()
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 SQL query
        ra = calcite.process(
            """SELECT passenger_count, AVG(total_amount) as total_amount_avg
               FROM trips
               GROUP BY passenger_count
               ORDER BY passenger_count;"""
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, ra)
        res = rel_alg_executor.execute()
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 SQL query
        ra = calcite.process(
            """SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) as cnt
               FROM trips
               GROUP BY passenger_count, pickup_year
               ORDER BY passenger_count;"""
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, ra)
        res = rel_alg_executor.execute()
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 SQL query
        ra = calcite.process(
            """SELECT
                 passenger_count,
                 extract(year from pickup_datetime) AS pickup_year,
                 cast(trip_distance as int) AS distance,
                 count(*) AS cnt
               FROM trips
               GROUP BY passenger_count, pickup_year, distance
               ORDER BY pickup_year, cnt desc;"""
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, ra)
        res = rel_alg_executor.execute()
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_explicit_instance(self):
        # Initialize HDK components wrapped into a single object
        hdk = pyhdk.init()

        # Import data
        # import for all cases?
        # globs?
        hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv",
            "trips",
        )

        # Run Taxi Q1 SQL query
        res = hdk.sql("SELECT cab_type, count(*) as cnt FROM trips GROUP BY cab_type;")
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 SQL query
        res = hdk.sql(
            """SELECT passenger_count, AVG(total_amount) as total_amount_avg
               FROM trips
               GROUP BY passenger_count
               ORDER BY passenger_count;"""
        )
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 SQL query
        res = hdk.sql(
            """SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) as cnt
               FROM trips
               GROUP BY passenger_count, pickup_year
               ORDER BY passenger_count;"""
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 SQL query
        res = hdk.sql(
            """SELECT
                 passenger_count,
                 extract(year from pickup_datetime) AS pickup_year,
                 cast(trip_distance as int) AS distance,
                 count(*) AS cnt
               FROM trips
               GROUP BY passenger_count, pickup_year, distance
               ORDER BY pickup_year, cnt desc;"""
        )
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_explicit_aliases(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        # Tables are referenced through the resulting object
        # Might allow to unify work with imported tables and temporary
        # tables (results of other queries)
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv"
        )

        # Run Taxi Q1 SQL query
        res = hdk.sql(
            "SELECT cab_type, count(*) as cnt FROM trips GROUP BY cab_type;",
            trips=trips,
        )
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 SQL query
        res = hdk.sql(
            """SELECT passenger_count, AVG(total_amount) as total_amount_avg
               FROM trips
               GROUP BY passenger_count
               ORDER BY passenger_count;""",
            trips=trips,
        )
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 SQL query
        res = hdk.sql(
            """SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) as cnt
               FROM trips
               GROUP BY passenger_count, pickup_year
               ORDER BY passenger_count;""",
            trips=trips,
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 SQL query
        res = hdk.sql(
            """SELECT
                 passenger_count,
                 extract(year from pickup_datetime) AS pickup_year,
                 cast(trip_distance as int) AS distance,
                 count(*) AS cnt
               FROM trips
               GROUP BY passenger_count, pickup_year, distance
               ORDER BY pickup_year, cnt desc;""",
            trips=trips,
        )
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_multistep(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv"
        )

        # Run Taxi Q3 SQL query in 2 steps
        tmp = hdk.sql(
            """SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year
               FROM trips""",
            trips=trips,
        )
        res = hdk.sql(
            """SELECT passenger_count, pickup_year, count(*) as cnt
               FROM trips
               GROUP BY passenger_count, pickup_year
               ORDER BY passenger_count;""",
            trips=tmp,
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 SQL query in 3 steps
        tmp = hdk.sql(
            """SELECT
                 passenger_count,
                 extract(year from pickup_datetime) AS pickup_year,
                 cast(trip_distance as int) AS distance
               FROM trips;""",
            trips=trips,
        )
        tmp = hdk.sql(
            """SELECT passenger_count, pickup_year, distance, count(*) AS cnt
               FROM trips
               GROUP BY passenger_count, pickup_year, distance;""",
            trips=tmp,
        )
        res = hdk.sql("SELET * FROM trips ORDER BY pickup_year, cnt desc;", trips=tmp)
        self.check_taxi_q4_res(res)


class TestTaxiIR(BaseTest):
    @pytest.mark.skip(reason="QueryBuilder API is not yet available in PyHDK")
    def test_taxi_over_csv_modular(self):
        # Initialize HDK components
        config = pyhdk.buildConfig()
        storage = pyhdk.storage.ArrowStorage(1)
        data_mgr = pyhdk.storage.DataMgr(config)
        data_mgr.registerDataProvider(storage)
        executor = pyhdk.Executor(data_mgr, config)

        # Import data
        storage.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv",
            "trips",
        )

        # Run Taxi Q1 IR query
        builder = pyhdk.QueryBuilder(config, storage)
        dag = builder.scan("trips").agg("cab_type", "count").finalize()
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, dag)
        res = rel_alg_executor.execute()
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 IR query
        builder = pyhdk.QueryBuilder(config, storage)
        dag = (
            builder.scan("trips")
            .agg("passenger_count", "avg(total_amount)")
            .sort("passenger_count")
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, dag)
        res = rel_alg_executor.execute()
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 IR query
        builder = pyhdk.QueryBuilder(config, storage)
        trips = builder.scan("trips")
        dag = (
            trips.proj(
                [
                    trips["passenger_count"],
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                ]
            )
            .agg(["passenger_count", "pickup_year"], "count")
            .sort("passenger_count")
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, dag)
        res = rel_alg_executor.execute()
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 IR query
        builder = pyhdk.QueryBuilder(config, storage)
        trips = builder.scan("trips")
        dag = (
            trips.proj(
                [
                    trips["passenger_count"],
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                    trips["trip_distance"].cast("int32").name("distance"),
                ]
            )
            .agg(["passenger_count", "pickup_year", "distance"], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(executor, storage, data_mgr, dag)
        res = rel_alg_executor.execute()
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_explicit_instance(self):
        # Initialize HDK components wrapped into a single object
        hdk = pyhdk.init()

        # Import data
        hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv",
            "trips",
        )

        # Run Taxi Q1 IR query
        res = hdk.scan("trips").agg("cab_type", "count").run()
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 IR query
        res = (
            hdk.scan("trips")
            .agg("passenger_count", "avg(total_amount)")
            .sort("passenger_count")
            .run()
        )
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 IR query
        trips = hdk.scan("trips")
        res = (
            trips.proj(
                [
                    trips["passenger_count"],
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                ]
            )
            .agg(["passenger_count", "pickup_year"], "count")
            .sort("passenger_count")
            .run()
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 IR query
        trips = hdk.scan("trips")
        res = (
            trips.proj(
                [
                    trips["passenger_count"],
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                    trips["trip_distance"].cast("int32").name("distance"),
                ]
            )
            .agg(["passenger_count", "pickup_year", "distance"], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .run()
        )
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_implicit_scan(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        # How to reference it in SQL? Use file name as a table name? Use the same approach as in Modin?
        # When is it deleted from the storage? Only exlicit tables drop for simplicity?
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv"
        )

        # Run Taxi Q1 IR query
        res = trips.agg("cab_type", "count").run()
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 IR query
        res = (
            trips.agg("passenger_count", "avg(total_amount)")
            .sort("passenger_count")
            .run()
        )
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 IR query
        res = (
            trips.proj(
                [
                    "passenger_count",
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                ]
            )
            .agg(["passenger_count", "pickup_year"], "count")
            .sort("passenger_count")
            .run()
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 IR query
        res = (
            trips.proj(
                [
                    "passenger_count",
                    trips["pickup_datetime"].extract("year").name("pickup_year"),
                    trips["trip_distance"].cast("int32").name("distance"),
                ]
            )
            .agg([0, 1, 2], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .run()
        )
        self.check_taxi_q4_res(res)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_run_query_on_results(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header_no_null.csv"
        )

        # Run a part of Taxi Q2 IR query
        res = trips.agg("passenger_count", "avg(total_amount)").run()
        # Now sort it to get the final result
        # Can we make it without transforming to Arrow with the following import to ArrowStorage?
        res = res.sort("passenger_count").run()
        self.check_taxi_q2_res(res)
