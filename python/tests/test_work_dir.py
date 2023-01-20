#
# Copyright 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import os


class TestWorkDir:
    def test_work_dir(self):
        assert os.path.exists(
            "python/tests/test_dir_token"
        ), "Run pytest from the HDK root dir."
