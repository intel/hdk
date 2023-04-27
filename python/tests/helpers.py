#!/usr/bin/env python3

#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0


def check_schema(schema, expected):
    assert len(schema) == len(expected)
    assert schema.keys() == expected.keys()
    for key in schema.keys():
        assert str(schema[key].type) == expected[key]


def check_res(res, expected):
    df = res.to_arrow().to_pandas()
    expected_cols = list(expected.keys())
    actual_cols = df.columns.to_list()
    assert actual_cols == expected_cols
    for col in actual_cols:
        vals = df[col].fillna("null").to_list()
        assert len(vals) == len(expected[col])
        for expected_val, actual_val in zip(expected[col], vals):
            if type(expected_val) is float:
                assert abs(expected_val - actual_val) < 0.0001
            else:
                assert expected_val == actual_val
