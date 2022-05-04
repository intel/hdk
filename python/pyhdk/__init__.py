#
# Copyright 2022 Intel Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

# We set these dlopen flags to allow calls from JIT code
# to HDK shared objects. Otherwise, such relocations would
# be unresolved and we would have calls by zero address.
# TODO: Is there a way to avoid this in the Python code?
if sys.platform == "linux":
    prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

from pyhdk._common import TypeInfo, SQLType
from pyhdk._execute import Executor
import pyhdk.sql as sql
import pyhdk.storage as storage

if sys.platform == "linux":
    sys.setdlopenflags(prev)
