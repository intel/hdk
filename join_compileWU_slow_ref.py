#!/usr/bin/env python

import sys
import pyhdk
import pandas as pd
import numpy as np
import time


def compare_tables(left_df: pd.DataFrame, right_df: pd.DataFrame):
    try_to_guess = True
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
        print("cols: ", left_cols, " drops: ", drop_left)
        print("cols: ", right_cols, " drops: ", drop_right)
        left_df = left_df.drop(columns=drop_left)
        right_df = right_df.drop(columns=drop_right)

    left_cols = left_df.columns.to_list()
    right_cols = right_df.columns.to_list()
    left_cols.sort()
    right_cols.sort()

    print("cols: r - ", right_cols, " l - ", left_cols)

    assert left_cols == right_cols, "Table column names are different"

    left_df.sort_values(by=left_cols, inplace=True)
    right_df.sort_values(by=left_cols, inplace=True)
    for col in left_df.columns:
        if left_df[col].dtype in ["category"]:
            left_df[col] = left_df[col].astype("str")
            right_df[col] = right_df[col].astype("str")
    print("l dtypes \n", left_df.dtypes)
    print("r dtypes \n", right_df.dtypes)

    print("l size: ", left_df.size, " - r size: ", right_df.size)

    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)
    if not all(left_df == right_df):
        mask = left_df == right_df
        print("Mismathed left: ")
        print(left_df[mask])
        print("         right: ")
        print(left_df[mask])
        raise RuntimeError("Results mismatched")


pyhdk_init_args = {}
pyhdk_init_args["enable_debug_timer"] = True
pyhdk_init_args["enable_cpu_groupby_multifrag_kernels"] = False
# pyhdk_init_args["debug_logs"] = True
hdk = pyhdk.init(**pyhdk_init_args)
fragment_size = 4000000

N = 2

np.random.seed(1)
column_list = list()
for num in range(N):
    df_setup = {
        "column_1": np.random.randint(0, 150, size=(15000)),
        "column_3": np.random.randint(0, 6, size=(15000)),
        "column_5": np.random.randint(0, 10, size=(15000)),
        "A" + str(num): np.random.randint(0, 100, size=(15000)),
        "B" + str(num): np.random.randint(0, 100, size=(15000)),
        "C" + str(num): np.random.randint(0, 100, size=(15000)),
        "D" + str(num): np.random.randint(0, 100, size=(15000)),
    }
    column_list.append(df_setup)

df_list = list()
for num in range(N):
    df = pd.DataFrame(column_list[num])
    df_list.append(df)

t1 = time.time()
for idx, df in enumerate(df_list):
    if idx == 0:
        df_base = df.copy()
        df_base = df_base.to_dict("list")
    else:
        if type(df_base) == dict:
            ht_base = hdk.import_pydict(df_base)
        else:
            ht_base = df_base
        df_r = hdk.import_pydict(df.to_dict("list"))
        df_ans = ht_base.join(
            df_r,
            ["column_1", "column_3", "column_5"],
            ["column_1", "column_3", "column_5"],
        ).run()
        df_base = df_ans.to_arrow().to_pandas().to_dict("list")
print("Hdk Time:", time.time() - t1)

t2 = time.time()
for idx, df in enumerate(df_list):
    if idx == 0:
        df_base_pd = df.copy()
    else:
        df_base_pd = pd.merge(
            df_base_pd,
            df,
            left_on=["column_1", "column_3", "column_5"],
            right_on=["column_1", "column_3", "column_5"],
            how="inner",
        )
print("Pandas Time:", time.time() - t2)

print("[hdk] shape: ", pd.DataFrame(df_base).shape)
print("[ pd] shape: ", df_base_pd.shape)
print("compare: ")
compare_tables(pd.DataFrame(df_base), df_base_pd)
