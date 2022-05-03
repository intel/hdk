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
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyhdk._storage cimport CSchemaProvider

cdef extern from "omniscidb/Calcite/CalciteJNI.h":
  cdef cppclass FilterPushDownInfo:
    int input_prev;
    int input_start;
    int input_next;

  cdef cppclass CalciteJNI:
    CalciteJNI(shared_ptr[CSchemaProvider], string, size_t);
    string process(string, string, string, vector[FilterPushDownInfo], bool, bool, bool)
