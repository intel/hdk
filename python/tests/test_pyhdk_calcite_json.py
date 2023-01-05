#!/usr/bin/env python3

# 
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import json
import pandas
import pytest
import pyarrow
import pyhdk


class TestCalciteJson:
    @classmethod
    def setup_class(cls):
        #pyhdk.initLogger(debug_logs=True)
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
    def execute_json_ra(cls, json_ra, **kwargs):
        ra = cls.calcite.process(json_ra)
        rel_alg_executor = pyhdk.sql.RelAlgExecutor(
            cls.executor, cls.storage, cls.data_mgr, ra
        )
        return rel_alg_executor.execute(**kwargs)
        
    def test_filter_json(self):
        json_ra_str = """execute calcite 
        {
  "rels": [
    {
      "id": "0",
      "relOp": "LogicalTableScan",
      "fieldNames": [
        "a",
        "b",
        "rowid"
      ],
      "table": [
        "test-db",
        "test"
      ],
      "inputs": []
    },
    {
      "id": "1",
      "relOp": "LogicalFilter",
      "condition": {
        "op": "AND",
        "operands": [
          {
            "op": ">",
            "operands": [
              {
                "input": 0
              },
              {
                "literal": 1,
                "type": "DECIMAL",
                "target_type": "INTEGER",
                "scale": 0,
                "precision": 1,
                "type_scale": 0,
                "type_precision": 10
              }
            ],
            "type": {
              "type": "BOOLEAN",
              "nullable": false
            }
          },
          {
            "op": "<",
            "operands": [
              {
                "input": 0
              },
              {
                "literal": 3,
                "type": "DECIMAL",
                "target_type": "INTEGER",
                "scale": 0,
                "precision": 1,
                "type_scale": 0,
                "type_precision": 10
              }
            ],
            "type": {
              "type": "BOOLEAN",
              "nullable": false
            }
          }
        ],
        "type": {
          "type": "BOOLEAN",
          "nullable": false
        }
      }
    },
    {
      "id": "2",
      "relOp": "LogicalProject",
      "fields": [
        "$f0"
      ],
      "exprs": [
        {
          "literal": 0,
          "type": "DECIMAL",
          "target_type": "INTEGER",
          "scale": 0,
          "precision": 1,
          "type_scale": 0,
          "type_precision": 10
        }
      ]
    },
    {
      "id": "3",
      "relOp": "LogicalAggregate",
      "fields": [
        "EXPR$0"
      ],
      "group": [],
      "aggs": [
        {
          "agg": "COUNT",
          "type": {
            "type": "BIGINT",
            "nullable": false
          },
          "distinct": false,
          "operands": []
        }
      ]
    }
  ]
}
        """

        res = self.execute_json_ra(json_ra_str)
        df = res.to_arrow().to_pandas()
        assert df.shape == (1,1)
        print(df)
        assert df["$f0"].tolist()[0] == 1