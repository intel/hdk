#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared, static_pointer_cast
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move
from cython.operator cimport dereference, preincrement

from pyarrow.lib cimport pyarrow_unwrap_table
from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport TypeInfo, Config, CConfig, CContext
from pyhdk._common import buildConfig

from collections.abc import Iterable

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

    if not table_info:
      return None

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

cdef class SchemaMgr:
  def __cinit__(self):
    self.c_schema_mgr = make_shared[CSchemaMgr]()
    self.c_schema_provider = static_pointer_cast[CSchemaProvider, CSchemaMgr](self.c_schema_mgr)

  def registerProvider(self, SchemaProvider provider):
    cdef schema_id = provider.getId()
    self.c_schema_mgr.get().registerProvider(schema_id, provider.c_schema_provider)

cdef class TableOptions:
  cdef CTableOptions c_options

  def __cinit__(self, int fragment_size = 0):
    self.c_options = CTableOptions()
    if fragment_size > 0:
      self.c_options.fragment_size = fragment_size

  @property
  def fragment_size(self):
    return self.c_options.fragment_size

  @fragment_size.setter
  def fragment_size(self, value):
    if not isinstance(value, int):
      raise TypeError("Only integer values are allowed for fragment_size.")
    self.c_options.fragment_size = value

cdef class CsvParseOptions:
  cdef CCsvParseOptions c_options

  def __cinit__(self):
    self.c_options = CCsvParseOptions()

  @property
  def delimiter(self):
    return self.c_options.delimiter

  @delimiter.setter
  def delimiter(self, value):
    if not isinstance(value, str) or not (len(value) == 1):
      raise TypeError("Only single-character strings are allowed for delimiter.")
    self.c_options.delimiter = value.encode('utf8')[0]

  @property
  def header(self):
    return self.c_options.header

  @header.setter
  def header(self, value):
    self.c_options.header = value

  @property
  def skip_rows(self):
    return self.c_options.skip_rows

  @skip_rows.setter
  def skip_rows(self, value):
    self.c_options.skip_rows = value

  @property
  def block_size(self):
    return self.c_options.block_size

  @block_size.setter
  def block_size(self, value):
    self.c_options.block_size = value

cdef class ArrowStorage(Storage):
  cdef shared_ptr[CArrowStorage] c_storage

  def __cinit__(self, int schema_id, Config config = buildConfig()):
    cdef string schema_name = f"schema_#{schema_id}"
    cdef int db_id = (schema_id << 24) + 1
    self.c_storage = make_shared[CArrowStorage](schema_id, schema_name, db_id, config.c_config)
    self.c_schema_provider = static_pointer_cast[CSchemaProvider, CArrowStorage](self.c_storage)
    self.c_abstract_buffer_mgr = static_pointer_cast[CAbstractBufferMgr, CArrowStorage](self.c_storage)

  def createTable(self, table_name, scheme, TableOptions table_opts):
    if not isinstance(table_name, str):
      raise TypeError(f"Expected str for 'table_name' arg. Got: {type(table_name)}.")

    cdef vector[CColumnDescription] col_descs
    cdef CColumnDescription col_desc

    if isinstance(scheme, dict):
      for key, val in scheme.items():
        col_desc.name = self._process_col_name(key)
        col_desc.type = (<TypeInfo>self._process_type(val)).c_type_info
        col_descs.push_back(col_desc)
    elif isinstance(scheme, Iterable):
      for val in scheme:
        if not isinstance(val, tuple) or len(val) != 2:
          raise TypeError(f"Expected tuple of 2 as a column descriptor. Got: {type(val)}.")
        col_desc.name = self._process_col_name(val[0])
        col_desc.type = (<TypeInfo>self._process_type(val[1])).c_type_info
        col_descs.push_back(col_desc)
    else:
      raise TypeError(f"Expected dict or list of tuples for 'scheme' arg. Got: {type(scheme)}.")

    self.c_storage.get().createTable(table_name, col_descs, table_opts.c_options)

  def _process_col_name(self, val):
    if not isinstance(val, str):
      raise TypeError(f"Expected str for column name. Got: {type(val)}.")
    return val

  def _process_type(self, val):
    if isinstance(val, str):
      res = TypeInfo()
      res.c_type_info = CContext.defaultCtx().typeFromString(val)
      return res
    elif isinstance(val, TypeInfo):
      return val
    else:
      raise TypeError(f"Expected TypeInfo or str for column type. Got: {type(val)}.")

  def importArrowTable(self, table, name, TableOptions options):
    cdef shared_ptr[CArrowTable] at = pyarrow_unwrap_table(table)
    self.c_storage.get().importArrowTable(at, name, options.c_options)

  def appendArrowTable(self, table, name):
    cdef shared_ptr[CArrowTable] at = pyarrow_unwrap_table(table)
    self.c_storage.get().appendArrowTable(at, name)

  def importCsvFile(self, file_name, table_name, schema = None, TableOptions table_opts = None, CsvParseOptions csv_opts = None):
    if table_opts is None:
      table_opts = TableOptions()
    if csv_opts is None:
      csv_opts = CsvParseOptions()

    cdef vector[CColumnDescription] c_schema
    cdef CColumnDescription col_desc

    def process_col_type(col_name, col_type):
      if not isinstance(col_name, str):
        raise TypeError(f"Expected str for a column name. Got: {type(col_name)}.")
      col_desc.name = col_name

      if col_type is None or col_type == "":
        col_desc.type = NULL
      elif isinstance(col_type, str):
        col_desc.type = CContext.defaultCtx().typeFromString(col_type)
      elif isinstance(col_type, TypeInfo):
        col_desc.type = (<TypeInfo>col_type).c_type_info
      else:
        raise TypeError(f"Expected None, str or TypeInfo for a column type. Got: {type()}.")

      c_schema.push_back(col_desc)

    if schema is None:
      self.c_storage.get().importCsvFile(file_name, table_name, table_opts.c_options, csv_opts.c_options)
    else:
      if isinstance(schema, dict):
        for col_name, col_type in schema.items():
          process_col_type(col_name, col_type)
      elif isinstance(schema, Iterable):
        for col_info in schema:
          if isinstance(col_info, str):
            process_col_type(col_info, None)
          elif isinstance(col_info, tuple):
            if len(col_info) == 1:
              process_col_type(col_info[0], None)
            elif len(col_info) == 2:
              process_col_type(col_info[0], col_info[1])
            else:
              raise TypeError(f"Expected tuple length for a column descriptor is 1 or 2. Got: {len(col_info)}.")
          else:
            raise TypeError(f"Expected str or tuple for a column descriptor. Got: {type(col_info)}.")

      self.c_storage.get().importCsvFileWithSchema(file_name, table_name, c_schema, table_opts.c_options, csv_opts.c_options)

  def appendCsvFile(self, file_name, table_name, CsvParseOptions csv_opts = None):
    if csv_opts is None:
      csv_opts = CsvParseOptions()
    self.c_storage.get().appendCsvFile(file_name, table_name, csv_opts.c_options)

  def importParquetFile(self, file_name, table_name, TableOptions table_opts = None):
    if table_opts is None:
      table_opts = TableOptions()

    self.c_storage.get().importParquetFile(file_name, table_name, table_opts.c_options)

  def appendParquetFile(self, file_name, table_name):
    self.c_storage.get().appendParquetFile(file_name, table_name)

  def dropTable(self, string name, bool throw_if_not_exist = False):
    self.c_storage.get().dropTable(name, throw_if_not_exist)

  def tableInfo(self, string table_name):
    return self.getTableInfo(self.c_storage.get().dbId(), table_name)

cdef class DataMgr:
  def __cinit__(self, Config config):
    self.c_data_mgr = make_shared[CDataMgr](dereference(config.c_config), 0)

  cpdef registerDataProvider(self, Storage storage):
    cdef schema_id = storage.getId()
    cdef shared_ptr[CAbstractBufferMgr] buffer_mgr = storage.c_abstract_buffer_mgr
    self.c_data_mgr.get().getPersistentStorageMgr().registerDataProvider(schema_id, buffer_mgr)
