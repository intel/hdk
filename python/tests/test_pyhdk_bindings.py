#!/usr/bin/env python3

import json
import pandas
import pyarrow
import pyhdk
import pytest


class TestArrowStorage:
    def test_import_arrow_table(self):
        storage = pyhdk.storage.ArrowStorage(123)
        assert storage is not None
        assert storage.getId() == 123

        at = pyarrow.Table.from_pandas(
            pandas.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
        )

        opt = pyhdk.storage.TableOptions(2)
        storage.importArrowTable(at, "test1", opt)

        databases = storage.listDatabases()
        assert len(databases) == 1
        db_id = databases[0]
        assert db_id >> 24 == 123

        tables = storage.listTables(db_id)
        assert len(tables) == 1
        table = tables[0]
        assert table.db_id == db_id
        assert table.name == "test1"
        assert table.fragments == 2
        assert table.is_stream == False

        columns = storage.listColumns(db_id, "test1")
        assert len(columns) == 3

        assert columns[0].db_id == db_id
        assert columns[0].table_id == table.table_id
        assert columns[0].name == "a"
        assert columns[0].type.type == "BIGINT"
        assert columns[0].is_rowid == False

        assert columns[1].db_id == db_id
        assert columns[1].table_id == table.table_id
        assert columns[1].name == "b"
        assert columns[1].type.type == "BIGINT"
        assert columns[1].is_rowid == False

        assert columns[2].db_id == db_id
        assert columns[2].table_id == table.table_id
        assert columns[2].name == "rowid"
        assert columns[2].type.type == "BIGINT"
        assert columns[2].is_rowid == True


class TestCalcite:
    @classmethod
    def setup_class(cls):
        cls.config = pyhdk.buildConfig()
        cls.storage = pyhdk.storage.ArrowStorage(1)
        cls.calcite = pyhdk.sql.Calcite(cls.storage, cls.config)

        at = pyarrow.Table.from_pandas(
            pandas.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        )
        opt = pyhdk.storage.TableOptions(2)
        cls.storage.importArrowTable(at, "test", opt)

    @classmethod
    def teardown_class(cls):
        del cls.calcite
        del cls.storage
        del cls.config

    def test_sql_parsing(self):
        json_ra = self.calcite.process("SELECT a, b FROM test;")
        ra = json.loads(json_ra)
        assert len(ra["rels"]) == 2
        assert ra["rels"][0]["relOp"] == "LogicalTableScan"
        assert ra["rels"][1]["relOp"] == "LogicalProject"
