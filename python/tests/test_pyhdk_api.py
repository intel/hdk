#!/usr/bin/env python3

#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0


import pandas
import pytest
import pyhdk
import numpy as np


class BaseTest:
    @staticmethod
    def check_schema(schema, expected):
        assert len(schema) == len(expected)
        assert schema.keys() == expected.keys()
        for key in schema.keys():
            assert str(schema[key].type) == expected[key]

    @staticmethod
    def check_cst(cst, val, type):
        assert str(cst) == f"(Const {val})"
        assert str(cst.type) == str(type)

    @staticmethod
    def check_ref(ref, idx):
        assert ref.is_ref
        assert ref.index == idx

    @staticmethod
    def check_res(res, expected):
        df = res.to_arrow().to_pandas()
        expected_cols = list(expected.keys())
        actual_cols = df.columns.to_list()
        assert actual_cols == expected_cols
        for col in actual_cols:
            assert df[col].fillna("null").to_list() == expected[col]


class TestImport(BaseTest):
    def test_create_table(self):
        hdk = pyhdk.init()

        ht = hdk.create_table("test1", [("a", "int"), ("b", hdk.type("fp"))])
        self.check_schema(ht.schema, {"a": "INT64", "b": "FP64"})
        hdk.drop_table(ht)

        ht = hdk.create_table(
            "test2", {"a": "int", "b": hdk.type("fp")}, fragment_size=4
        )
        self.check_schema(ht.schema, {"a": "INT64", "b": "FP64"})
        hdk.drop_table(ht)

        with pytest.raises(TypeError) as e:
            hdk.create_table("test1", [(1, "int")])
        with pytest.raises(TypeError) as e:
            hdk.create_table("test1", [("a", 12)])
        with pytest.raises(TypeError) as e:
            hdk.create_table("test1", [("a", "int", "int")])
        with pytest.raises(TypeError) as e:
            hdk.create_table("test1", "a: int")
        with pytest.raises(RuntimeError) as e:
            hdk.create_table("test1", [])
        with pytest.raises(RuntimeError) as e:
            hdk.create_table("test1", dict())
        with pytest.raises(TypeError) as e:
            hdk.create_table("test2", {"a": "int"}, fragment_size="4")

    def test_import_pydict(self):
        hdk = pyhdk.init()
        table_name = "table_test_import_pydict"
        ht = hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, table_name)
        assert ht.is_scan
        assert ht.table_name == table_name
        assert ht.size == 2
        self.check_schema(ht.schema, {"a": "INT64", "b": "FP64"})
        hdk.drop_table(table_name)

    @pytest.mark.parametrize("header", [True, False])
    @pytest.mark.parametrize("create_table", [True, False])
    @pytest.mark.parametrize("full_schema", [None, True, False])
    @pytest.mark.parametrize("glob", [True, False])
    def test_import_csv(self, header, create_table, full_schema, glob):
        hdk = pyhdk.init()
        table_name = "table_import_csv"

        if create_table:
            hdk.create_table(table_name, (("col1", "int32"), ("col2", "fp32")))
            real_schema = {"col1": "INT32", "col2": "FP32"}

        if glob:
            file_name = "omniscidb/Tests/ArrowStorageDataFiles/numbers_header*.csv"
            ref_data = {
                "col1": [*range(1, 10), *range(10, 19)],
                "col2": [*np.arange(10.0, 100.0, 10), *np.arange(100.0, 190.0, 10)],
            }
        else:
            file_name = "omniscidb/Tests/ArrowStorageDataFiles/numbers_header.csv"
            ref_data = {"col1": [*range(1, 10)], "col2": [*np.arange(10.0, 100.0, 10)]}

        if full_schema is None:
            if header:
                schema = None
            else:
                schema = ["col1", "col2"]
            if not create_table:
                real_schema = {"col1": "INT64", "col2": "FP64"}
        elif full_schema:
            if not create_table:
                real_schema = {"col1": "INT32", "col2": "FP64"}
            schema = real_schema
        else:
            if header:
                schema = {"col2": "fp32"}
            else:
                schema = {"col1": None, "col2": "fp32"}
            if not create_table:
                real_schema = {"col1": "INT64", "col2": "FP32"}

        skip_rows = 0 if header else 1
        ht = hdk.import_csv(
            file_name, table_name, schema=schema, header=header, skip_rows=skip_rows
        )

        self.check_schema(ht.schema, real_schema)
        self.check_res(ht.proj(0, 1).run(), ref_data)

        hdk.drop_table(table_name)

    # TODO(dmitiim) Support more types. Decimal(128/256), Time32 (ms/us),
    # Timestamp, Text have several conversion issues in arrow, pandas,
    # parquet, hdk processing conversions.
    # e.g. parquet replacing [s] with [ms] on save of Time32.
    @pytest.mark.parametrize("create_table", [True, False])
    @pytest.mark.parametrize("glob", [True, False])
    def test_import_parquet(self, create_table, glob):
        hdk = pyhdk.init()
        table_name = "table_parquet"

        if glob:
            if create_table:
                hdk.create_table(table_name, {"fpD": "fp64", "intD": "int32"})
            real_schema = {"fpD": "FP64", "intD": "INT32"}
            file_name = "omniscidb/Tests/ArrowStorageDataFiles/fp_int*.parquet"
            ref_data = {
                "fpD": [1.00001, 2.00002, 3.00003, 4.00004, 5.00005, 6.00006, 7.00007],
                "intD": [43, 45, 47, 48, 49, 51, 53],
            }
        else:
            if create_table:
                hdk.create_table(table_name, (("a", "int64"), ("b", "fp64")))
                real_schema = {"a": "INT64", "b": "FP64"}
            else:
                real_schema = {"a": "INT64", "b": "FP64"}
            file_name = "omniscidb/Tests/ArrowStorageDataFiles/int_float.parquet"
            ref_data = {"a": [*range(1, 6)], "b": [1.1, 2.2, 3.3, 4.4, 5.5]}

        ht = hdk.import_parquet(file_name, table_name)
        self.check_schema(ht.schema, real_schema)
        self.check_res(ht.proj(0, 1).run(), ref_data)

        hdk.drop_table(table_name)


class TestBuilder(BaseTest):
    def test_scan(self):
        hdk = pyhdk.init()
        table_name = "table_test_scan"
        hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, table_name)
        ht = hdk.scan(table_name)
        assert ht.is_scan
        assert ht.table_name == table_name
        with pytest.raises(RuntimeError) as e:
            _ = hdk.scan("unknown_table")
        with pytest.raises(TypeError) as e:
            _ = hdk.scan(1)
        hdk.drop_table(table_name)

    def test_drop_table(self):
        hdk = pyhdk.init()
        table_name = "table_test_drop_table"

        hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, table_name)
        ht = hdk.scan(table_name)
        hdk.drop_table(table_name)
        with pytest.raises(RuntimeError) as e:
            _ = hdk.scan(table_name)

        hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, table_name)
        ht = hdk.scan(table_name)
        hdk.drop_table(ht)
        with pytest.raises(RuntimeError) as e:
            _ = hdk.scan(table_name)

        with pytest.raises(TypeError) as e:
            hdk.drop_table(1)

    def test_type_from_str(self):
        hdk = pyhdk.init()
        assert str(hdk.type("int")) == "INT64"
        assert str(hdk.type("fp")) == "FP64"
        assert str(hdk.type("text[NN] ")) == "TEXT[NN]"
        with pytest.raises(TypeError) as e:
            hdk.type(1)

    def test_cst(self):
        hdk = pyhdk.init()
        self.check_cst(hdk.cst(None), "NULL", "NULLT")
        self.check_cst(hdk.cst(None, "int"), "NULL", "INT64")
        self.check_cst(hdk.cst(None, hdk.type("fp")), "NULL", "FP64")

        self.check_cst(hdk.cst(True), "t", "BOOL[NN]")
        self.check_cst(hdk.cst(False), "f", "BOOL[NN]")
        self.check_cst(hdk.cst(True, "int"), "1", "INT64[NN]")
        self.check_cst(hdk.cst(False, hdk.type("bool")), "f", "BOOL[NN]")

        self.check_cst(hdk.cst(123), "123", "INT64[NN]")
        self.check_cst(hdk.cst(123, "int32"), "123", "INT32[NN]")
        self.check_cst(hdk.cst(123, "dec(10,2)"), "123.00", "DEC64(10,2)[NN]")
        self.check_cst(hdk.cst(123, "dec(10,2)", False), "1.23", "DEC64(10,2)[NN]")

        self.check_cst(hdk.cst(1.23), "1.230000", "FP64[NN]")
        self.check_cst(hdk.cst(1.23, "fp32"), "1.230000", "FP32[NN]")
        self.check_cst(hdk.cst(1.23, "dec(10,2)"), "1.23", "DEC64(10,2)[NN]")

        self.check_cst(hdk.cst("str1"), "str1", "TEXT[NN]")
        self.check_cst(hdk.cst("123", "int32"), "123", "INT32[NN]")
        self.check_cst(hdk.cst("1.23", "dec(10,2)"), "1.23", "DEC64(10,2)[NN]")
        self.check_cst(hdk.cst("1983-12-01", "date"), "1983-12-01", "DATE64[s][NN]")
        self.check_cst(
            hdk.cst("2020-10-04 15:00:00", "timestamp"),
            "2020-10-04 15:00:00.000000",
            "TIMESTAMP[us][NN]",
        )

        with pytest.raises(TypeError) as e:
            hdk.cst(dict())

        self.check_cst(hdk.date("1983-12-01"), "1983-12-01", "DATE64[s][NN]")
        self.check_cst(hdk.time("15:00:00"), "15:00:00", "TIME64[us][NN]")
        self.check_cst(
            hdk.timestamp("2020-10-04 15:00:00"),
            "2020-10-04 15:00:00.000000",
            "TIMESTAMP[us][NN]",
        )

        self.check_cst(
            hdk.cst([1, 2, 3], "array(int32)"), "[1, 2, 3]", "ARRAY32(INT32)"
        )
        self.check_cst(
            hdk.cst([1.1, 2.2, 3.3]),
            "[1.100000, 2.200000, 3.300000]",
            "ARRAY32(FP64[NN])[NN]",
        )
        self.check_cst(hdk.cst([], "array(int32)"), "[]", "ARRAY32(INT32)")
        with pytest.raises(RuntimeError) as e:
            hdk.cst([])

    def test_ref(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        self.check_ref(ht.ref(0), 0)
        self.check_ref(ht.ref(1), 1)
        self.check_ref(ht.ref(-1), 1)
        self.check_ref(ht.ref(-2), 0)
        self.check_ref(ht.ref("a"), 0)
        self.check_ref(ht.ref("b"), 1)
        self.check_ref(ht.ref("rowid"), 2)

        with pytest.raises(TypeError) as e:
            ht.ref([])
        with pytest.raises(TypeError) as e:
            ht.ref(None)
        with pytest.raises(RuntimeError) as e:
            ht.ref(-3)
        with pytest.raises(RuntimeError) as e:
            ht.ref(4)
        with pytest.raises(RuntimeError) as e:
            ht.ref("x")

        hdk.drop_table(ht)

    def test_proj(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]})
        proj = ht.proj(
            0,
            -1,
            "a",
            "b",
            ht.ref("a").rename("c"),
            exprs={"d": 0, "e": "b", "f": ht.ref("a")},
            g=0,
            h="b",
            a=ht.ref("b"),
        )
        self.check_res(
            proj.run(),
            {
                "a_1": [1, 2, 3],
                "b": [1.1, 2.2, 3.3],
                "a_2": [1, 2, 3],
                "b_1": [1.1, 2.2, 3.3],
                "c": [1, 2, 3],
                "d": [1, 2, 3],
                "e": [1.1, 2.2, 3.3],
                "f": [1, 2, 3],
                "g": [1, 2, 3],
                "h": [1.1, 2.2, 3.3],
                "a": [1.1, 2.2, 3.3],
            },
        )

        hdk.drop_table(ht)

    def test_sort(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {
                "a": [1, 2, 1, 2, 1, 2, 1, None, 1, 2],
                "b": [1, 1, None, 1, 1, 2, 2, 2, 2, 2],
            }
        )

        self.check_res(
            ht.sort("a", -1).run(),
            {
                "a": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, "null"],
                "b": [1.0, 1.0, 2.0, 2.0, "null", 1.0, 1.0, 2.0, 2.0, 2.0],
            },
        )

        self.check_res(
            ht.sort(("a", "desc"), (ht.ref("b"), "asc")).run(),
            {
                "a": [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, "null"],
                "b": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, "null", 2.0],
            },
        )

        self.check_res(
            ht.sort(("a", "desc", "first"), ht.ref("b")).run(),
            {
                "a": ["null", 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "b": [2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, "null"],
            },
        )

        self.check_res(
            ht.sort(("a", "desc"), fields={"b": "asc"}).run(),
            {
                "a": [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, "null"],
                "b": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, "null", 2.0],
            },
        )

        self.check_res(
            ht.sort(fields={"b": ("asc", "first")}, a="desc").run(),
            {
                "a": [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, "null"],
                "b": ["null", 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
        )

        self.check_res(
            ht.sort(b=("asc", "first"), a=("desc", "first")).run(),
            {
                "a": [1.0, 2.0, 2.0, 1.0, 1.0, "null", 2.0, 2.0, 1.0, 1.0],
                "b": ["null", 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
        )

        # TODO: test offset when result set conversion to arrow is fixed
        # and respects offset option.
        self.check_res(
            ht.sort("a", "b", limit=5, offset=0).run(),
            {"a": [1, 1, 1, 1, 1], "b": [1.0, 1.0, 2.0, 2.0, "null"]},
        )

        with pytest.raises(RuntimeError):
            ht.sort("c")
        with pytest.raises(RuntimeError):
            ht.sort(fields={"c": "asc"})
        with pytest.raises(RuntimeError):
            ht.sort(c="asc")
        with pytest.raises(TypeError):
            ht.sort(fields="a")
        with pytest.raises(TypeError):
            ht.sort(fields=["a"])
        with pytest.raises(RuntimeError):
            ht.sort(("a", "aasc"))
        with pytest.raises(RuntimeError):
            ht.sort(fields={"a": "aasc"})
        with pytest.raises(RuntimeError):
            ht.sort(a="aasc")
        with pytest.raises(RuntimeError):
            ht.sort(("a", "asc", "ffirst"))
        with pytest.raises(RuntimeError):
            ht.sort(fields={"a": ("asc", "ffirst")})
        with pytest.raises(RuntimeError):
            ht.sort(a=("asc", "ffirst"))
        with pytest.raises(TypeError):
            ht.sort((1.0, "asc"))
        with pytest.raises(TypeError):
            ht.sort(fields={1.0: "asc"})
        with pytest.raises(TypeError):
            ht.sort(())
        with pytest.raises(TypeError):
            ht.sort(fields={"a": ()})
        with pytest.raises(TypeError):
            ht.sort(a=())
        with pytest.raises(TypeError):
            ht.sort(("a", "asc", "first", "fast"))
        with pytest.raises(TypeError):
            ht.sort(fields={"a": ("asc", "first", "fast")})
        with pytest.raises(TypeError):
            ht.sort(a=("asc", "first", "fast"))
        with pytest.raises(TypeError):
            ht.sort(fields={"a": 1.0})
        with pytest.raises(TypeError):
            ht.sort(a=(1.0))
        with pytest.raises(TypeError):
            ht.sort(fields={"a": ("asc", 1.0)})
        with pytest.raises(TypeError):
            ht.sort(a=("asc", 1.0))
        with pytest.raises(TypeError):
            ht.sort("a", limit=1.5)
        with pytest.raises(ValueError):
            ht.sort("a", limit=-1)
        with pytest.raises(TypeError):
            ht.sort("a", offset=1.5)
        with pytest.raises(ValueError):
            ht.sort("a", offset=-1)

        hdk.drop_table(ht)

    def test_agg(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {
                "a": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                "b": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        self.check_res(
            ht.agg(["a", -2], "sum(c)", ht.ref("c").min(), hdk.count())
            .sort("a", "b")
            .run(),
            {
                "a": [1, 1, 2, 2],
                "b": [1, 2, 1, 2],
                "c_sum": [9, 16, 6, 24],
                "c_min": [1, 7, 2, 6],
                "count": [3, 2, 2, 3],
            },
        )

        self.check_res(
            ht.agg(
                ht.ref("a"),
                aggs={"bc": "count(b)", "cmx": ht.ref("c").max()},
                cmn="min(c)",
                cv=ht.ref("c").avg(),
            )
            .sort("a")
            .run(),
            {
                "a": [1, 2],
                "bc": [5, 5],
                "cmx": [9, 10],
                "cmn": [1, 2],
                "cv": [5.0, 6.0],
            },
        )

        self.check_res(
            ht.agg(ht.ref("b"), cd=ht.ref("a").count(True)).sort("b").run(),
            {"b": [1, 2], "cd": [2, 2]},
        )

        self.check_res(
            ht.agg(
                ht.ref("b"),
                a1=ht.ref("c").approx_quantile(0),
                a2=ht.ref("c").approx_quantile(0.5),
                a3=ht.ref("c").approx_quantile(1),
            )
            .sort("b")
            .run(),
            {"b": [1, 2], "a1": [1, 6], "a2": [3, 8], "a3": [5, 10]},
        )

        with pytest.raises(RuntimeError):
            ht.agg("e")
        with pytest.raises(TypeError):
            ht.agg(1.5)
        with pytest.raises(TypeError):
            ht.agg("a", 1.5)
        with pytest.raises(RuntimeError):
            ht.agg("a", "min(e)")
        with pytest.raises(RuntimeError):
            ht.agg("a", "select")
        with pytest.raises(TypeError):
            ht.agg("a", aggs="min(c)")
        with pytest.raises(TypeError):
            ht.agg("a", aggs={1.5: "min(b)"})
        with pytest.raises(TypeError):
            ht.agg("a", aggs={"m": 1})
        with pytest.raises(TypeError):
            ht.agg("a", h=1.5)
        with pytest.raises(RuntimeError):
            ht.agg("a", h="min(f)")

        hdk.drop_table(ht)

    def test_filter(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        )

        self.check_res(
            ht.filter(ht["a"] > 5).run(), {"a": [6, 7, 8, 9, 10], "b": [5, 4, 3, 2, 1]}
        )

        self.check_res(ht.filter(ht["a"] >= 6, ht["b"] > 4).run(), {"a": [6], "b": [5]})

        self.check_res(
            ht.filter((ht["a"] < 2).logical_or(ht["b"] <= 2)).run(),
            {"a": [1, 9, 10], "b": [10, 2, 1]},
        )

        self.check_res(
            ht.filter(
                (ht["a"] == 2).logical_or(ht["a"] == 3).logical_and(ht["b"] != 9)
            ).run(),
            {"a": [3], "b": [8]},
        )

        hdk.drop_table(ht)

        ht = hdk.import_pydict({"a": [1, 2, None, None, 5], "b": [10, 9, 8, 7, 6]})

        self.check_res(ht.filter(ht["a"].is_null()).proj("b").run(), {"b": [8, 7]})

        self.check_res(
            ht.filter(ht["a"].is_not_null()).proj("b").run(), {"b": [10, 9, 6]}
        )

        with pytest.raises(TypeError):
            ht.filter("a")
        with pytest.raises(TypeError):
            ht.filter(0)

        hdk.drop_table(ht)

    def test_join(self):
        hdk = pyhdk.init()
        ht1 = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )
        ht2 = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "y": [5.5, 4.4, 3.3, 2.2, 1.1]}
        )

        self.check_res(
            ht1.join(ht2).run(), {"a": [3], "b": [3], "x": [3.3], "y": [3.3]}
        )

        self.check_res(
            ht1.join(ht2, how="left").run(),
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "y": ["null", "null", 3.3, "null", "null"],
            },
        )

        self.check_res(
            ht1.join(ht2, "a").run(),
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "b_1": [1, 2, 3, 4, 5],
                "y": [5.5, 4.4, 3.3, 2.2, 1.1],
            },
        )

        self.check_res(
            ht1.join(ht2, ["a", "b"]).run(),
            {"a": [3], "b": [3], "x": [3.3], "y": [3.3]},
        )

        self.check_res(
            ht1.join(ht2, "a", "b").run(),
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "a_1": [1, 2, 3, 4, 5],
                "y": [5.5, 4.4, 3.3, 2.2, 1.1],
            },
        )

        self.check_res(
            ht1.join(ht2, ["a", "b"], ["b", "a"]).run(),
            {"a": [3], "b": [3], "x": [3.3], "y": [3.3]},
        )

        self.check_res(
            ht1.join(ht2, cond=ht1["a"].eq(ht2["b"])).run(),
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "a_1": [1, 2, 3, 4, 5],
                "b_1": [1, 2, 3, 4, 5],
                "y": [5.5, 4.4, 3.3, 2.2, 1.1],
            },
        )

        with pytest.raises(TypeError):
            ht1.join("a")
        with pytest.raises(TypeError):
            ht1.join(ht2, 0)
        with pytest.raises(TypeError):
            ht1.join(ht2, ["a", 1])
        with pytest.raises(RuntimeError):
            ht1.join(ht2, "c")
        with pytest.raises(ValueError):
            ht1.join(ht2, None, "a")
        with pytest.raises(ValueError):
            ht1.join(ht2, ["a"], ["a", "b"])
        with pytest.raises(TypeError):
            ht1.join(ht2, how=1)
        with pytest.raises(RuntimeError):
            ht1.join(ht2, how="outer")

        hdk.drop_table(ht1)
        hdk.drop_table(ht2)

    def test_math_ops(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )

        self.check_res(
            ht.proj(ht["a"].uminus()).run(), {"expr_1": [-1, -2, -3, -4, -5]}
        )

        self.check_res(ht.proj(-ht["a"]).run(), {"expr_1": [-1, -2, -3, -4, -5]})

        self.check_res(
            ht.proj(a1=ht["a"] + ht["b"], a2=ht["a"] + 1, a3=ht["a"] + 1.5).run(),
            {
                "a1": [6, 6, 6, 6, 6],
                "a2": [2, 3, 4, 5, 6],
                "a3": [2.5, 3.5, 4.5, 5.5, 6.5],
            },
        )

        self.check_res(
            ht.proj(a1=ht["a"] - ht["b"], a2=ht["a"] - 1, a3=ht["a"] - 1.5).run(),
            {
                "a1": [-4, -2, 0, 2, 4],
                "a2": [0, 1, 2, 3, 4],
                "a3": [-0.5, 0.5, 1.5, 2.5, 3.5],
            },
        )

        self.check_res(
            ht.proj(a1=ht["a"] * ht["b"], a2=ht["a"] * 2, a3=ht["a"] * 1.5).run(),
            {
                "a1": [5, 8, 9, 8, 5],
                "a2": [2, 4, 6, 8, 10],
                "a3": [1.5, 3.0, 4.5, 6.0, 7.5],
            },
        )

        self.check_res(
            ht.proj(a1=ht["a"] / ht["b"], a2=ht["a"] / 2, a3=ht["a"] / 2.0).run(),
            {
                "a1": [0.2, 0.5, 1.0, 2.0, 5.0],
                "a2": [0.5, 1.0, 1.5, 2.0, 2.5],
                "a3": [0.5, 1.0, 1.5, 2.0, 2.5],
            },
        )

        self.check_res(
            ht.proj(a1=ht["a"] // ht["b"], a2=ht["a"] // 2, a3=ht["x"] // 2.0).run(),
            {
                "a1": [0, 0, 1, 2, 5],
                "a2": [0, 1, 1, 2, 2],
                "a3": [0.0, 1.0, 1.0, 2.0, 2.0],
            },
        )

        self.check_res(
            ht.proj(
                a1=ht["a"].div(ht["b"]),
                a2=ht["a"].div(2),
                a3=ht["a"].div(2.0),
                a4=ht["x"].div(2.0),
            ).run(),
            {
                "a1": [0, 0, 1, 2, 5],
                "a2": [0, 1, 1, 2, 2],
                "a3": [0.5, 1.0, 1.5, 2.0, 2.5],
                "a4": [0.55, 1.1, 1.65, 2.2, 2.75],
            },
        )

        self.check_res(
            ht.proj(a1=ht["a"] % ht["b"], a2=ht["a"] % 2).run(),
            {"a1": [1, 2, 0, 0, 0], "a2": [1, 0, 1, 0, 1]},
        )

        hdk.drop_table(ht)

    def test_cast(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [1.1, 2.2, 3.3, 4.4, 5.5]})

        self.check_res(
            ht.proj(c1=ht["a"].cast("fp64"), c2=ht["b"].cast("int")).run(),
            {"c1": [1.0, 2.0, 3.0, 4.0, 5.0], "c2": [1, 2, 3, 4, 6]},
        )

        self.check_res(
            ht.proj(
                c1=hdk.cst("1970-01-01 01:00:00").cast("timestamp[ms]").cast("int")
            ).run(),
            {"c1": [3600000, 3600000, 3600000, 3600000, 3600000]},
        )

        hdk.drop_table(ht)

    def test_extract(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {
                "a": pandas.to_datetime(
                    ["20230207", "20220308", "20210409"], format="%Y%m%d"
                )
            }
        )

        self.check_res(
            ht.proj(
                r1=ht["a"].extract("year"),
                r2=ht["a"].extract("month"),
                r3=ht["a"].extract("day"),
            ).run(),
            {"r1": [2023, 2022, 2021], "r2": [2, 3, 4], "r3": [7, 8, 9]},
        )

        with pytest.raises(TypeError):
            ht["a"].extract(1)

        hdk.drop_table(ht)

    def test_date_add(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict(
            {
                "a": pandas.to_datetime(
                    ["20230207", "20220308", "20210409"], format="%Y%m%d"
                ),
                "b": [1, 2, 3],
            }
        )

        self.check_res(
            ht.proj(
                d1=ht["a"].add(ht["b"], "year"),
                d2=ht["a"].add(1, "month"),
                d3=ht["a"].sub(ht["b"], "day"),
                d4=ht["a"].sub(1, "hour"),
            ).run(),
            {
                "d1": [
                    pandas.Timestamp("2024-02-07 00:00:00"),
                    pandas.Timestamp("2024-03-08 00:00:00"),
                    pandas.Timestamp("2024-04-09 00:00:00"),
                ],
                "d2": [
                    pandas.Timestamp("2023-03-07 00:00:00"),
                    pandas.Timestamp("2022-04-08 00:00:00"),
                    pandas.Timestamp("2021-05-09 00:00:00"),
                ],
                "d3": [
                    pandas.Timestamp("2023-02-06 00:00:00"),
                    pandas.Timestamp("2022-03-06 00:00:00"),
                    pandas.Timestamp("2021-04-06 00:00:00"),
                ],
                "d4": [
                    pandas.Timestamp("2023-02-06 23:00:00"),
                    pandas.Timestamp("2022-03-07 23:00:00"),
                    pandas.Timestamp("2021-04-08 23:00:00"),
                ],
            },
        )

        with pytest.raises(TypeError):
            ht["a"].add(1, 1)
        with pytest.raises(RuntimeError):
            ht["a"].add(1, "dayss")
        with pytest.raises(TypeError):
            ht["a"].sub(1, 1)
        with pytest.raises(RuntimeError):
            ht["a"].sub(1, "dayss")
        with pytest.raises(RuntimeError):
            ht["b"].add(1, "day")
        with pytest.raises(RuntimeError):
            ht["b"].sub(1, "day")

        hdk.drop_table(ht)

    def test_unnest(self):
        hdk = pyhdk.init()
        ht = hdk.create_table("test1", [("a", "array(int)")])
        hdk.import_pydict({"a": [[1, 2], [1, 2, 3, 4]]}, ht)

        self.check_res(
            ht.proj(a=ht["a"].unnest()).agg(["a"], "count").sort("a").run(),
            {"a": [1, 2, 3, 4], "count": [2, 2, 1, 1]},
        )

        hdk.drop_table(ht)

    def test_at(self):
        hdk = pyhdk.init()
        ht = hdk.create_table("test1", [("a", "array(int)"), ("b", "int")])
        hdk.import_pydict({"a": [[1, 2], [2, 3, 4]], "b": [2, 3]}, ht)

        self.check_res(
            ht.proj(a1=ht["a"].at(1), a2=ht["a"][ht["b"]], a3=ht["a"].at(-1)).run(),
            {"a1": [1, 2], "a2": [2, 4], "a3": ["null", "null"]},
        )

        hdk.drop_table(ht)

    def test_run_on_res(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        res1 = ht.proj("b", "a").run()
        self.check_res(res1, {"b": [10, 20, 30, 40, 50], "a": [1, 2, 3, 4, 5]})

        res2 = res1.agg(["a"], "count").run()
        self.check_res(res2, {"a": [1, 2, 3, 4, 5], "count": [1, 1, 1, 1, 1]})

        res3 = res2.join(res1, "count", "a").run()
        self.check_res(
            res3,
            {"a": [1, 2, 3, 4, 5], "count": [1, 1, 1, 1, 1], "b": [10, 10, 10, 10, 10]},
        )

        res4 = res2.run()
        self.check_res(res4, {"a": [1, 2, 3, 4, 5], "count": [1, 1, 1, 1, 1]})

        res5 = res4.filter(res4.ref("a") > res4["count"]).run()
        self.check_res(res5, {"a": [2, 3, 4, 5], "count": [1, 1, 1, 1]})

        res6 = res5.run()
        self.check_res(res6, {"a": [2, 3, 4, 5], "count": [1, 1, 1, 1]})

        assert res6.is_scan
        assert res6.size == 2
        self.check_schema(res6.schema, {"a": "INT64", "count": "INT32[NN]"})

        res7 = res6.proj("rowid", "a", "count").run()
        self.check_res(
            res7, {"rowid": [0, 1, 2, 3], "a": [2, 3, 4, 5], "count": [1, 1, 1, 1]}
        )


class TestSql(BaseTest):
    def test_no_alias(self):
        hdk = pyhdk.init()
        ht = hdk.import_pydict({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})

        self.check_res(
            hdk.sql(f"SELECT a, b FROM {ht.table_name};"),
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]},
        )

    def test_alias(self):
        hdk = pyhdk.init()
        ht1 = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )
        ht2 = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "y": [5.5, 4.4, 3.3, 2.2, 1.1]}
        )

        self.check_res(
            hdk.sql("SELECT a, b FROM t1;", t1=ht1),
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]},
        )
        self.check_res(
            hdk.sql("SELECT a, b FROM t1;", t1=ht2),
            {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]},
        )

        self.check_res(
            hdk.sql(
                "SELECT t1.a, t1.b, t1.x, t2.y FROM t1, t2 WHERE t1.b = t2.a ORDER BY t1.a;",
                t1=ht1,
                t2=ht2,
            ),
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "x": [1.1, 2.2, 3.3, 4.4, 5.5],
                "y": [1.1, 2.2, 3.3, 4.4, 5.5],
            },
        )

    def test_run_on_res(self):
        hdk = pyhdk.init()
        ht1 = hdk.import_pydict(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "x": [1.1, 2.2, 3.3, 4.4, 5.5]}
        )

        res1 = hdk.sql("SELECT a, b FROM t1;", t1=ht1)
        self.check_res(res1, {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})

        res2 = hdk.sql("SELECT b + 1 as b, a - 1 as a FROM t1;", t1=res1)
        self.check_res(res2, {"b": [6, 5, 4, 3, 2], "a": [0, 1, 2, 3, 4]})

        res3 = hdk.sql(f"SELECT b - 1 as b, a + 1 as a FROM {res1.table_name};")
        self.check_res(res3, {"b": [4, 3, 2, 1, 0], "a": [2, 3, 4, 5, 6]})


class BaseTaxiTest:
    @staticmethod
    def check_taxi_q1_res(res):
        df = res.to_arrow().to_pandas()
        assert df["cab_type"].tolist() == ["green"]
        if "cnt" in df.columns:
            assert df["cnt"].tolist() == [20]
        else:
            assert df["count"].tolist() == [20]

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
        if "cnt" in df.columns:
            assert df["cnt"].tolist() == [16, 1, 3]
        else:
            assert df["count"].tolist() == [16, 1, 3]

    @staticmethod
    def check_taxi_q4_res(res):
        df = res.to_arrow().to_pandas()
        assert df["passenger_count"].tolist() == [1, 5, 2]
        assert df["pickup_year"].tolist() == [2013, 2013, 2013]
        assert df["distance"].tolist() == [0, 0, 0]
        if "cnt" in df.columns:
            assert df["cnt"].tolist() == [16, 3, 1]
        else:
            assert df["count"].tolist() == [16, 3, 1]


class TestTaxiSql(BaseTaxiTest):
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
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv", "trips"
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

        storage.dropTable("trips")

    def test_taxi_over_csv_explicit_instance(self):
        # Initialize HDK components wrapped into a single object
        hdk = pyhdk.init()

        # Import data
        # import for all cases?
        # globs?
        hdk.import_csv(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv", "trips"
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

        hdk.drop_table("trips")

    def test_taxi_over_csv_explicit_aliases(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        # Tables are referenced through the resulting object
        # Might allow to unify work with imported tables and temporary
        # tables (results of other queries)
        trips = hdk.import_csv(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv"
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

        hdk.drop_table(trips)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_taxi_over_csv_multistep(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv"
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

        hdk.drop_table(trips)


class TestTaxiIR(BaseTaxiTest):
    def test_taxi_over_csv_modular(self):
        # Initialize HDK components
        config = pyhdk.buildConfig()
        storage = pyhdk.storage.ArrowStorage(1)
        data_mgr = pyhdk.storage.DataMgr(config)
        data_mgr.registerDataProvider(storage)
        executor = pyhdk.Executor(data_mgr, config)

        # Import data
        storage.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv", "trips"
        )

        # Run Taxi Q1 IR query
        builder = pyhdk.QueryBuilder(storage, config)
        dag = builder.scan("trips").agg("cab_type", "count").finalize()
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            executor, storage, data_mgr, dag=dag
        )
        res = rel_alg_executor.execute()
        self.check_taxi_q1_res(res)

        # Run Taxi Q2 IR query
        builder = pyhdk.QueryBuilder(storage, config)
        dag = (
            builder.scan("trips")
            .agg("passenger_count", "avg(total_amount)")
            .sort("passenger_count")
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            executor, storage, data_mgr, dag=dag
        )
        res = rel_alg_executor.execute()
        self.check_taxi_q2_res(res)

        # Run Taxi Q3 IR query
        builder = pyhdk.QueryBuilder(storage, config)
        trips = builder.scan("trips")
        dag = (
            trips.proj(
                trips["passenger_count"],
                trips["pickup_datetime"].extract("year").rename("pickup_year"),
            )
            .agg(["passenger_count", "pickup_year"], "count")
            .sort("passenger_count")
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            executor, storage, data_mgr, dag=dag
        )
        res = rel_alg_executor.execute()
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 IR query
        builder = pyhdk.QueryBuilder(storage, config)
        trips = builder.scan("trips")
        dag = (
            trips.proj(
                trips["passenger_count"],
                trips["pickup_datetime"].extract("year").rename("pickup_year"),
                trips["trip_distance"].cast("int32").rename("distance"),
            )
            .agg(["passenger_count", "pickup_year", "distance"], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .finalize()
        )
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            executor, storage, data_mgr, dag=dag
        )
        res = rel_alg_executor.execute()
        self.check_taxi_q4_res(res)

        storage.dropTable("trips")

    def test_taxi_over_csv_explicit_instance(self):
        # Initialize HDK components wrapped into a single object
        hdk = pyhdk.init()

        # Import data
        hdk.import_csv(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv", "trips"
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
                "passenger_count", pickup_year=trips["pickup_datetime"].extract("year")
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
                "passenger_count",
                pickup_year=trips["pickup_datetime"].extract("year"),
                distance=trips["trip_distance"].cast("int32"),
            )
            .agg(["passenger_count", "pickup_year", "distance"], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .run()
        )
        self.check_taxi_q4_res(res)

        hdk.drop_table(trips)

    def test_taxi_over_csv_implicit_scan(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        # How to reference it in SQL? Use file name as a table name? Use the same approach as in Modin?
        # When is it deleted from the storage? Only exlicit tables drop for simplicity?
        trips = hdk.import_csv(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv"
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
                "passenger_count", pickup_year=trips["pickup_datetime"].extract("year")
            )
            .agg(["passenger_count", "pickup_year"], "count")
            .sort("passenger_count")
            .run()
        )
        self.check_taxi_q3_res(res)

        # Run Taxi Q4 IR query
        res = (
            trips.proj(
                "passenger_count",
                pickup_year=trips["pickup_datetime"].extract("year"),
                distance=trips["trip_distance"].cast("int32"),
            )
            .agg([0, 1, 2], "count")
            .sort(("pickup_year", "asc"), ("count", "desc"))
            .run()
        )
        self.check_taxi_q4_res(res)

        hdk.drop_table(trips)

    @pytest.mark.skip(reason="unimplemented concept")
    def test_run_query_on_results(self):
        # Initialize HDK components hidden from users
        hdk = pyhdk.init()

        # Import data
        trips = hdk.importCsvFile(
            "omniscidb/Tests/ArrowStorageDataFiles/taxi_sample_header.csv"
        )

        # Run a part of Taxi Q2 IR query
        res = trips.agg("passenger_count", "avg(total_amount)").run()
        # Now sort it to get the final result
        # Can we make it without transforming to Arrow with the following import to ArrowStorage?
        res = res.sort("passenger_count").run()
        self.check_taxi_q2_res(res)

        hdk.drop_table(trips)
