#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr, make_unique, shared_ptr
from cython.operator cimport dereference

cdef class SQLType:
  cdef CTypeId c_val

  def __cinit__(self, int val):
    self.c_val = <CTypeId>val

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
      <int>kNull : "NULL",
      <int>kBoolean : "BOOLEAN",
      <int>kInteger : "INTEGER",
      <int>kFloatingPoint : "FP",
      <int>kDecimal : "DECIMAL",
      <int>kVarChar : "VARCHAR",
      <int>kText : "TEXT",
      <int>kDate : "DATE",
      <int>kTime : "TIME",
      <int>kTimestamp : "TIMESTAMP",
      <int>kInterval : "INTERVAL",
      <int>kFixedLenArray : "FIX_LEN_ARRAY",
      <int>kVarLenArray : "VAR_LEN_ARRAY",
      <int>kExtDictionary : "EXT_DICTIONARY",
      <int>kColumn : "COLUMN",
      <int>kColumnList : "COLUMN_LIST",
    }
    return names[<int>self.c_val]

cdef class TypeInfo:
  @property
  def type(self):
    return SQLType(self.c_type_info.id())

  @property
  def nullable(self):
    return self.c_type_info.nullable()

  @property
  def size(self):
    return self.c_type_info.size()

  def __str__(self):
    return self.c_type_info.toString()

  def __repr__(self):
    return self.c_type_info.toString()


def buildConfig(*, enable_debug_timer=None, enable_union=False, **kwargs):
  global g_enable_debug_timer
  if enable_debug_timer is not None:
    g_enable_debug_timer = enable_debug_timer

  # Remove legacy params to provide better compatibility with PyOmniSciDbe
  kwargs.pop("enable_union", None)
  kwargs.pop("enable_thrift_logs", None)

  cmd_str = "".join(' --%s %r' % arg for arg in kwargs.iteritems())
  cmd_str = cmd_str.replace("_", "-")
  cdef string app = "modin".encode('UTF-8')
  cdef CConfigBuilder builder
  builder.parseCommandLineArgs(app, cmd_str, False)
  cdef Config config = Config()
  config.c_config = builder.config()
  return config

def initLogger(*, debug_logs=False, **kwargs):
  argv0 = "PyHDK".encode('UTF-8')
  cdef char *cargv0 = argv0
  cdef unique_ptr[CLogOptions] opts = make_unique[CLogOptions](cargv0)
  if debug_logs:
    opts.get().severity_ = CSeverity.DEBUG3
  CInitLogger(dereference(opts))
