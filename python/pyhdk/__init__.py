#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import os

# We set these dlopen flags to allow calls from JIT code
# to HDK shared objects. Otherwise, such relocations would
# be unresolved and we would have calls by zero address.
# TODO: Is there a way to avoid this in the Python code?
if sys.platform == "linux":
    prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

from pyhdk._common import TypeInfo, SQLType, setGlobalConfig, initLogger
from pyhdk._execute import Executor
import pyhdk.sql as sql
import pyhdk.storage as storage

if sys.platform == "linux":
    sys.setdlopenflags(prev)
