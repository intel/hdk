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
from libcpp.memory cimport shared_ptr, make_shared, static_pointer_cast
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move
from cython.operator cimport dereference, preincrement

from pyarrow.lib cimport pyarrow_unwrap_table
from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport TypeInfo, Config, CConfig

cdef class TableInfo:
  @property
  def db_id(self):
    return self.c_table_info.get().db_id

  @property
  def table_id(self):
    return self.c_table_info.get().table_id

  @property
  def name(self):
    return self.c_table_info.get().name

  @property
  def is_view(self):
    return self.c_table_info.get().is_view

  @property
  def persistence_level(self):
    return self.c_table_info.get().persistence_level

  @property
  def fragments(self):
    return self.c_table_info.get().fragments

  @property
  def is_stream(self):
    return self.c_table_info.get().is_stream

  def __str__(self):
    return self.c_table_info.get().toString()

  def __repr__(self):
    return self.c_table_info.get().toString()

cdef class ColumnInfo:
  @property
  def db_id(self):
    return self.c_column_info.get().db_id

  @property
  def table_id(self):
    return self.c_column_info.get().table_id

  @property
  def column_id(self):
    return self.c_column_info.get().column_id

  @property
  def name(self):
    return self.c_column_info.get().name

  @property
  def type(self):
    cdef TypeInfo res = TypeInfo()
    res.c_type_info = self.c_column_info.get().type
    return res

  @property
  def is_rowid(self):
    return self.c_column_info.get().is_rowid

  def __str__(self):
    return self.c_column_info.get().toString()

  def __repr__(self):
    return self.c_column_info.get().toString()

cdef class SchemaProvider:
  cpdef getId(self):
    return self.c_schema_provider.get().getId()

  cpdef listDatabases(self):
    return self.c_schema_provider.get().listDatabases()

  cpdef listTables(self, db):
    cdef int db_id = db
    cdef CTableInfoList tables

    tables = self.c_schema_provider.get().listTables(db_id)

    res = []
    cdef vector[CTableInfoPtr].iterator it = tables.begin()
    cdef TableInfo table_info
    while it != tables.end():
      table_info = TableInfo()
      table_info.c_table_info = dereference(it)
      res.append(table_info)
      preincrement(it)
    return res

  cpdef listColumns(self, db, table):
    cdef int db_id = db
    cdef int table_id
    cdef CColumnInfoList columns

    if isinstance(table, str):
      table_id = self.getTableInfo(db, table).table_id
    else:
      table_id = table
    columns = self.c_schema_provider.get().listColumns(db_id, table_id)

    res = []
    cdef vector[CColumnInfoPtr].iterator it = columns.begin()
    cdef ColumnInfo column_info
    while it != columns.end():
      column_info = ColumnInfo()
      column_info.c_column_info = dereference(it)
      res.append(column_info)
      preincrement(it)
    return res

  cpdef getTableInfo(self, db, table):
    cdef int db_id = db
    cdef string table_name
    cdef int table_id
    cdef CTableInfoPtr table_info

    if isinstance(table, str):
      table_name = table
      table_info = self.c_schema_provider.get().getTableInfoByName(db_id, table_name)
    else:
      table_id = table
      table_info = self.c_schema_provider.get().getTableInfo(db_id, table_id)

    cdef TableInfo res = TableInfo()
    res.c_table_info = table_info
    return res

  cpdef getColumnInfo(self, db, table, column):
    cdef int db_id = db
    cdef int table_id
    cdef string column_name
    cdef int column_id
    cdef CColumnInfoPtr column_info

    if isinstance(table, str):
      table_id = self.getTableInfo(db, table).table_id
    else:
      table_id = table

    if isinstance(column, str):
      column_name = column
      column_info = self.c_schema_provider.get().getColumnInfoByName(db_id, table_id, column_name)
    else:
      column_id = column
      column_info = self.c_schema_provider.get().getColumnInfo(db_id, table_id, column_id)

    cdef ColumnInfo res = ColumnInfo()
    res.c_column_info = column_info
    return res

cdef class TableOptions:
  cdef CTableOptions c_options

  def __cinit__(self, int fragment_size):
    self.c_options.fragment_size = fragment_size

cdef class ArrowStorage(Storage):
  cdef shared_ptr[CArrowStorage] c_storage

  def __cinit__(self, int schema_id):
    cdef string schema_name = f"schema_#{schema_id}"
    cdef int db_id = (schema_id << 24) + 1
    self.c_storage = make_shared[CArrowStorage](schema_id, schema_name, db_id)
    self.c_schema_provider = static_pointer_cast[CSchemaProvider, CArrowStorage](self.c_storage)
    self.c_abstract_buffer_mgr = static_pointer_cast[CAbstractBufferMgr, CArrowStorage](self.c_storage)

  def importArrowTable(self, table, name, TableOptions options):
    cdef shared_ptr[CArrowTable] at = pyarrow_unwrap_table(table)
    self.c_storage.get().importArrowTable(at, name, options.c_options)

  def dropTable(self, string name, bool throw_if_not_exist = False):
    self.c_storage.get().dropTable(name, throw_if_not_exist)

cdef class DataMgr:
  def __cinit__(self, Config config):
    cdef CSystemParameters sys_params
    cdef map[CGpuMgrName, unique_ptr[CGpuMgr]] gpuMgrs = move(map[CGpuMgrName, unique_ptr[CGpuMgr]]())
    self.c_data_mgr = make_shared[CDataMgr](dereference(config.c_config), sys_params, move(gpuMgrs), 1 << 27, 0)

  cpdef registerDataProvider(self, Storage storage):
    cdef schema_id = storage.getId()
    cdef shared_ptr[CAbstractBufferMgr] buffer_mgr = storage.c_abstract_buffer_mgr
    self.c_data_mgr.get().getPersistentStorageMgr().registerDataProvider(schema_id, buffer_mgr)
