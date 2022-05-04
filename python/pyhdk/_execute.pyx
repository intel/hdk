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

from pyhdk._storage cimport DataMgr

cdef class Executor:
  def __cinit__(self, DataMgr data_mgr, int id = 0):
    cdef CSystemParameters params = CSystemParameters()
    cdef string debug_dir = "".encode('UTF-8')
    cdef string debug_file = "".encode('UTF-8')
    cdef CBufferProvider *buffer_provider = data_mgr.c_data_mgr.get().getBufferProvider()
    self.c_executor = CExecutor.getExecutor(id, data_mgr.c_data_mgr.get(), buffer_provider, debug_dir, debug_file, params)
