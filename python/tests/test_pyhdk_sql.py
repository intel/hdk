#!/usr/bin/env python3

import json
import pandas
import pyarrow
import pytest
import pyhdk


class TestSql:
    @classmethod
    def setup_class(cls):
        cls.config = pyhdk.buildConfig()
        cls.storage = pyhdk.storage.ArrowStorage(1)
        cls.data_mgr = pyhdk.storage.DataMgr(cls.config)
        cls.data_mgr.registerDataProvider(cls.storage)

        cls.calcite = pyhdk.sql.Calcite(cls.storage, cls.config)
        cls.executor = pyhdk.Executor(cls.data_mgr, cls.config)

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

    @classmethod
    def execute_sql(cls, sql, **kwargs):
        ra = cls.calcite.process(sql)
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            cls.executor, cls.storage, cls.data_mgr, ra
        )
        return rel_alg_executor.execute(**kwargs)

    def test_simple_projection(self):
        res = self.execute_sql("SELECT * FROM test;")
        df = res.to_arrow().to_pandas()
        assert df.shape == (3, 2)
        assert df["a"].tolist() == [1, 2, 3]
        assert df["b"].tolist() == [10, 20, 30]

    def test_explain(self):
        res = self.execute_sql("SELECT * FROM test;", just_explain=True)
        explain_str = res.to_explain_str()
        assert explain_str[:15] == "IR for the CPU:"
