#!/usr/bin/env python3

#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
import pandas as pd


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


def compare_tables(
    left_df: pd.DataFrame, right_df: pd.DataFrame, try_to_guess: bool = False
):
    left_cols = left_df.columns.to_list()
    right_cols = right_df.columns.to_list()
    left_cols.sort()
    right_cols.sort()

    diff_idx = [
        idx for idx, col_name in enumerate(right_cols) if col_name != left_cols[idx]
    ]

    print("compare lists: ", diff_idx)
    drop_left = []
    drop_right = []
    for drop_idx in diff_idx:
        drop_left += [left_cols[drop_idx]]
        drop_right += [right_cols[drop_idx]]
    if try_to_guess:
        right_df = right_df.rename(columns=dict(zip(drop_right, drop_left)))
    else:
        left_df = left_df.drop(columns=drop_left)
        right_df = right_df.drop(columns=drop_right)

    left_cols = left_df.columns.to_list()
    right_cols = right_df.columns.to_list()
    left_cols.sort()
    right_cols.sort()

    assert left_cols == right_cols, "Table column names are different"

    left_df.sort_values(by=left_cols, inplace=True)
    right_df.sort_values(by=left_cols, inplace=True)
    for col in left_df.columns:
        if left_df[col].dtype in ["category"]:
            left_df[col] = left_df[col].astype("str")
            right_df[col] = right_df[col].astype("str")

    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)
    if not all(left_df == right_df):
        mask = left_df == right_df
        print("Mismathed left: ")
        print(left_df[mask])
        print("         right: ")
        print(left_df[mask])
        raise RuntimeError("Results mismatched")
