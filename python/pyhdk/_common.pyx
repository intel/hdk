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

cdef class SQLType:
  cdef CSQLTypes c_val

  def __cinit__(self, int val):
    self.c_val = <CSQLTypes>val

  def __eq__(self, val):
    if isinstance(val, int):
      return <int>self.c_val == val
    if isinstance(val, str):
      return self.__repr__() == val
    if isinstance(val, SQLType):
      return <int>self.c_val == <int>val.c_val
    return False

  def __repr__(self):
    cdef names = {
      <int>kNULLT : "NULLT",
      <int>kBOOLEAN : "BOOLEAN",
      <int>kCHAR : "CHAR",
      <int>kVARCHAR : "VARCHAR",
      <int>kNUMERIC : "NUMERIC",
      <int>kDECIMAL : "DECIMAL",
      <int>kINT : "INT",
      <int>kSMALLINT : "SMALLINT",
      <int>kFLOAT : "FLOAT",
      <int>kDOUBLE : "DOUBLE",
      <int>kTIME : "TIME",
      <int>kTIMESTAMP : "TIMESTAMP",
      <int>kBIGINT : "BIGINT",
      <int>kTEXT : "TEXT",
      <int>kDATE : "DATE",
      <int>kARRAY : "ARRAY",
      <int>kINTERVAL_DAY_TIME : "INTERVAL_DAY_TIME",
      <int>kINTERVAL_YEAR_MONTH : "INTERVAL_YEAR_MONTH",
      <int>kTINYINT : "TINYINT",
      <int>kEVAL_CONTEXT_TYPE : "EVAL_CONTEXT_TYPE",
      <int>kVOID : "VOID",
      <int>kCURSOR : "CURSOR",
      <int>kCOLUMN : "COLUMN",
      <int>kCOLUMN_LIST : "COLUMN_LIST",
      <int>kSQLTYPE_LAST : "LAST",
    }
    return names[<int>self.c_val]

cdef class TypeInfo:
  @property
  def type(self):
    return SQLType(self.c_type_info.get_type())

  @property
  def subtype(self):
    return SQLType(self.c_type_info.get_subtype())

  @property
  def dimension(self):
    return self.c_type_info.get_dimension()

  @property
  def precision(self):
    return self.c_type_info.get_precision()

  @property
  def input_srid(self):
    return self.c_type_info.get_input_srid()

  @property
  def scale(self):
    return self.c_type_info.get_scale()

  @property
  def output_srid(self):
    return self.c_type_info.get_output_srid()

  @property
  def notnull(self):
    return self.c_type_info.get_notnull()

  @property
  def compression(self):
    return self.c_type_info.get_compression()

  @property
  def comp_param(self):
    return self.c_type_info.get_comp_param()

  @property
  def size(self):
    return self.c_type_info.get_size()

  @property
  def logical_size(self):
    return self.c_type_info.get_logical_size()

  def __str__(self):
    return self.c_type_info.toString()

  def __repr__(self):
    return self.c_type_info.toString()
