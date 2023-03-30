#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport shared_ptr, make_shared, static_pointer_cast

from pyhdk._common cimport Config
from pyhdk._storage cimport DataMgr, Storage, CAbstractBufferMgr, CSchemaProvider

cdef class Executor:
  def __cinit__(self, DataMgr data_mgr, Config config):
    cdef string debug_dir = "".encode('UTF-8')
    cdef string debug_file = "".encode('UTF-8')
    self.c_executor = CExecutor.getExecutor(data_mgr.c_data_mgr.get(), config.c_config, debug_dir, debug_file)

cdef class ResultSetRegistry(Storage):
  cdef shared_ptr[CResultSetRegistry] c_registry

  def __cinit__(self, Config config):
    self.c_registry = make_shared[CResultSetRegistry](config.c_config)
    self.c_schema_provider = static_pointer_cast[CSchemaProvider, CResultSetRegistry](self.c_registry)
    self.c_abstract_buffer_mgr = static_pointer_cast[CAbstractBufferMgr, CResultSetRegistry](self.c_registry)
