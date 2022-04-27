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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyhdk.calcite cimport CalciteJNI
from pyhdk.calcite cimport FilterPushDownInfo

cdef class Calcite:
  cdef shared_ptr[CalciteJNI] calcite

  def __cinit__(self, **kwargs):
    cdef string udf_filename = kwargs.get("udf_filename", "")
    cdef size_t calcite_max_mem_mb = kwargs.get("calcite_max_mem_mb", 1024)
    self.calcite = make_shared[CalciteJNI](udf_filename, calcite_max_mem_mb)

  def process(self, string sql, string schema_json, **kwargs):
    cdef string user = kwargs.get("user", "admin")
    cdef string db_name = kwargs.get("db_name", "test-db")
    cdef vector[FilterPushDownInfo] filter_push_down_info = vector[FilterPushDownInfo]()
    cdef bool legacy_syntax = kwargs.get("legacy_syntax", False)
    cdef bool is_explain = kwargs.get("is_explain", False)
    cdef bool is_view_optimize = kwargs.get("is_view_optimize", False)
    return self.calcite.get().process(user, db_name, sql, schema_json, filter_push_down_info, legacy_syntax, is_explain, is_view_optimize)
