#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from pyhdk._common cimport Config
from pyhdk._storage cimport DataMgr

cdef class Executor:
  def __cinit__(self, DataMgr data_mgr, Config config):
    cdef string debug_dir = "".encode('UTF-8')
    cdef string debug_file = "".encode('UTF-8')
    self.c_executor = CExecutor.getExecutor(data_mgr.c_data_mgr.get(), config.c_config, debug_dir, debug_file)
