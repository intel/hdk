#!/usr/bin/env python3

#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import pyhdk
import pyarrow


class TestImport:
    def test_null_schema(self):
        table = pyarrow.table(
            [[]], schema=pyarrow.schema([pyarrow.field("A", pyarrow.null())])
        )
        opt = pyhdk.storage.TableOptions(2)
        hdk = pyhdk.init()
        ht = hdk.import_arrow(table)
        hdk.drop_table(ht)
