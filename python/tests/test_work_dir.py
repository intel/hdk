#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pyhdk
import time
import pytest

import pyhdk


class TestWorkDir:
    def test_work_dir(self):
        assert os.path.exists(
            "python/tests/test_dir_token"
        ), "Run pytest from the HDK root dir."

    def test_log_dir(self):
        log_dir = "pyhdk_test_dir"
        pyhdk.initLogger(log_dir=log_dir)
        assert os.path.exists("pyhdk_test_dir")

    # The Calcite initialization test below can fail to detect the log4j.log file when this test is run as part of all other pyhdk tests. Mark it as xfail and expect a failure when run in that context.
    @pytest.mark.xfail
    def test_calcite_log_dir(self):
        log_dir = "pyhdk_test_dir"
        config = pyhdk.buildConfig(log_dir=log_dir)
        storage = pyhdk.storage.ArrowStorage(1, config)

        calcite = pyhdk.sql.Calcite(storage, config)

        assert os.path.exists("pyhdk_test_dir")
        assert os.path.isfile("pyhdk_test_dir/log4j.log")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("pyhdk_test_dir")
