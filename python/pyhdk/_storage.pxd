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

from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CSQLTypeInfo

cdef extern from "omniscidb/DataMgr/MemoryLevel.h" namespace "Data_Namespace":
  enum MemoryLevel:
    DISK_LEVEL = 0,
    CPU_LEVEL = 1,
    GPU_LEVEL = 2,


cdef extern from "omniscidb/SchemaMgr/TableInfo.h":
  cdef cppclass CTableRef "TableRef":
    int db_id
    int table_id

    TableRef(int, int)

  cdef cppclass CTableInfo "TableInfo"(CTableRef):
    string name
    bool is_view
    MemoryLevel persistence_level
    size_t fragments
    bool is_stream

    CTableInfo(int, int, string, bool, MemoryLevel, size_t, bool);
    string toString()

ctypedef shared_ptr[CTableInfo] CTableInfoPtr
ctypedef vector[CTableInfoPtr] CTableInfoList

cdef extern from "omniscidb/SchemaMgr/ColumnInfo.h":
  cdef cppclass CColumnRef "ColumnRef":
    int db_id
    int table_id
    int column_id

    CColumnRef(int, int, int)

  cdef cppclass CColumnInfo "ColumnInfo"(CColumnRef):
    string name
    CSQLTypeInfo type
    bool is_rowid

    CColumnInfo(int, int, int, string, CSQLTypeInfo, bool)
    string toString()

ctypedef shared_ptr[CColumnInfo] CColumnInfoPtr
ctypedef vector[CColumnInfoPtr] CColumnInfoList

cdef extern from "omniscidb/SchemaMgr/SchemaProvider.h":
  cdef cppclass CSchemaProvider "SchemaProvider":
    int getId()
    vector[int] listDatabases()
    CTableInfoList listTables(int)
    CColumnInfoList listColumns(int, int)
    CTableInfoPtr getTableInfo(int, int)
    CTableInfoPtr getTableInfoByName "getTableInfo"(int, string&)
    CColumnInfoPtr getColumnInfo(int, int, int)
    CColumnInfoPtr getColumnInfoByName "getColumnInfo"(int, int, string&)

cdef class TableInfo:
  cdef CTableInfoPtr c_table_info

cdef class ColumnInfo:
  cdef CColumnInfoPtr c_column_info

cdef class SchemaProvider:
  cdef shared_ptr[CSchemaProvider] c_schema_provider

  cpdef getId(self)
  cpdef listDatabases(self)
  cpdef listTables(self, db)
  cpdef listColumns(self, db, table)
  cpdef getTableInfo(self, db, table)
  cpdef getColumnInfo(self, db, table, column)

cdef extern from "omniscidb/ArrowStorage/ArrowStorage.h" namespace "ArrowStorage":
  struct CColumnDescription "ArrowStorage::ColumnDescription":
    string name;
    CSQLTypeInfo type;

  struct CTableOptions "ArrowStorage::TableOptions":
    size_t fragment_size;

  struct CCsvParseOptions "ArrowStorage::CsvParseOptions":
    char delimiter;
    bool header;
    size_t skip_rows;
    size_t block_size;

  struct CJsonParseOptions "ArrowStorage::JsonParseOptions":
    size_t skip_rows;
    size_t block_size;

cdef extern from "omniscidb/ArrowStorage/ArrowStorage.h":
  cdef cppclass CArrowStorage "ArrowStorage"(CSchemaProvider):
    CArrowStorage(int, string, int);

    CTableInfoPtr importArrowTable(shared_ptr[CArrowTable], string&, CTableOptions&);
