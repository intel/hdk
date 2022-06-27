#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from pyhdk._storage cimport DataMgr

cdef class Executor:
  def __cinit__(self, DataMgr data_mgr, int id = 0):
    cdef CSystemParameters params = CSystemParameters()
    cdef string debug_dir = "".encode('UTF-8')
    cdef string debug_file = "".encode('UTF-8')
    cdef CBufferProvider *buffer_provider = data_mgr.c_data_mgr.get().getBufferProvider()
    self.c_executor = CExecutor.getExecutor(id, data_mgr.c_data_mgr.get(), buffer_provider, debug_dir, debug_file, params)
