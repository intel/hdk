#!/usr/bin/env python3

#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import pyhdk
import pyarrow

from helpers import check_res


class TestImport:
    def test_null_schema(self):
        table = pyarrow.table(
            [[]], schema=pyarrow.schema([pyarrow.field("A", pyarrow.null())])
        )
        opt = pyhdk.storage.TableOptions(2)
        hdk = pyhdk.init()
        ht = hdk.import_arrow(table)
        hdk.drop_table(ht)

    def test_dict_import(self):
        hdk = pyhdk.init()
        ht = hdk.create_table("table1", {"col1": "dict", "col2": "text"})

        col1 = pyarrow.array(["str1", "str2"])
        at = pyarrow.table([col1, col1], names=["col1", "col2"])
        hdk.import_arrow(at, ht)

        col1 = pyarrow.DictionaryArray.from_arrays([0, 1, 0, 1], ["str3", "str4"])
        at = pyarrow.table([col1, col1], names=["col1", "col2"])
        hdk.import_arrow(at, ht)

        check_res(
            ht.run(),
            {
                "col1": ["str1", "str2", "str3", "str4", "str3", "str4"],
                "col2": ["str1", "str2", "str3", "str4", "str3", "str4"],
            },
        )

        hdk.drop_table(ht)
