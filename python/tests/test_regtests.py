#!/usr/bin/env python3

#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import pyhdk
import numpy as np
import pyarrow as pa

from helpers import check_res


class TestRegressions:
    def test_issue439(self):
        hdk = pyhdk.init()
        ht = hdk.import_arrow(pa.Table.from_pandas(pd.DataFrame({"a": range(100000)})))

        res = ht.proj("a").run()
        res = res.proj("a").run()
        check_res(res, {"a": [i for i in range(100000)]})

        res = ht.sort(("a", "desc")).run()
        check_res(res, {"a": [i for i in range(99999, -1, -1)]})
        res = res.proj("a").run()
        check_res(res, {"a": [i for i in range(99999, -1, -1)]})
