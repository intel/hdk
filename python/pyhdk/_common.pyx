#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr, make_unique, shared_ptr
from cython.operator cimport dereference

cdef class TypeId:
  cdef CTypeId c_val

  def __cinit__(self, int val):
    self.c_val = <CTypeId>val

  def __eq__(self, val):
    if isinstance(val, int):
      return <int>self.c_val == val
    if isinstance(val, str):
      return self.__repr__() == val
    if isinstance(val, TypeId):
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
    return TypeId(self.c_type_info.id())

  @property
  def nullable(self):
    return self.c_type_info.nullable()

  @property
  def size(self):
    return self.c_type_info.size()

  @property
  def is_null(self):
    return self.c_type_info.isNull()

  @property
  def is_bool(self):
    return self.c_type_info.isBoolean()

  @property
  def is_int(self):
    return self.c_type_info.isInteger()

  @property
  def is_fp(self):
    return self.c_type_info.isFloatingPoint()

  @property
  def is_decimal(self):
    return self.c_type_info.isDecimal()

  @property
  def is_var_char(self):
    return self.c_type_info.isVarChar()

  @property
  def is_text(self):
    return self.c_type_info.isText()

  @property
  def is_date(self):
    return self.c_type_info.isDate()

  @property
  def is_time(self):
    return self.c_type_info.isTime()

  @property
  def is_timestamp(self):
    return self.c_type_info.isTimestamp()

  @property
  def is_interval(self):
    return self.c_type_info.isInterval()

  @property
  def is_fixed_len_array(self):
    return self.c_type_info.isFixedLenArray()

  @property
  def is_var_len_array(self):
    return self.c_type_info.isVarLenArray()

  @property
  def is_ext_dictionary(self):
    return self.c_type_info.isExtDictionary()

  @property
  def is_column(self):
    return self.c_type_info.isColumn()

  @property
  def is_column_list(self):
    return self.c_type_info.isColumnList()

  @property
  def is_int8(self):
    return self.c_type_info.isInt8()

  @property
  def is_int16(self):
    return self.c_type_info.isInt16()

  @property
  def is_int32(self):
    return self.c_type_info.isInt32()

  @property
  def is_int64(self):
    return self.c_type_info.isInt64()

  @property
  def is_fp32(self):
    return self.c_type_info.isFp32()

  @property
  def is_fp64(self):
    return self.c_type_info.isFp64()

  @property
  def is_number(self):
    return self.c_type_info.isNumber()

  @property
  def is_string(self):
    return self.c_type_info.isString()

  @property
  def is_date_time(self):
    return self.c_type_info.isDateTime()

  @property
  def is_array(self):
    return self.c_type_info.isArray()

  @property
  def is_var_len(self):
    return self.c_type_info.isVarLen()

  @property
  def is_buffer(self):
    return self.c_type_info.isBuffer()

  @property
  def elem_type(self):
    res = TypeInfo()
    if self.is_array:
      res.c_type_info = (<const CArrayBaseType*>self.c_type_info).elemType()
    elif self.is_ext_dictionary:
      res.c_type_info = (<const CExtDictionaryType*>self.c_type_info).elemType()
    else:
      raise TypeError(f"Only arrays and dictionaries provide element type. Actual type: {self}")
    return res

  def __str__(self):
    return self.c_type_info.toString()

  def __repr__(self):
    return self.c_type_info.toString()

cdef class Config:
  @property
  def gpu_prop(self):
    return self.c_config.get().exec.heterogeneous.forced_gpu_proportion
  
  @gpu_prop.setter
  def gpu_prop(self, gpu_prop):
    self.c_config.get().exec.heterogeneous.forced_gpu_proportion=gpu_prop
    self.c_config.get().exec.heterogeneous.forced_cpu_proportion=100-gpu_prop
  
  @property
  def allow_cpu_retry(self):
    return self.c_config.get().exec.heterogeneous.allow_cpu_retry

  @allow_cpu_retry.setter
  def allow_cpu_retry(self, allowed):
    self.c_config.get().exec.heterogeneous.allow_cpu_retry=allowed

  @property
  def allow_query_step_cpu_retry(self):
    return self.c_config.get().exec.heterogeneous.allow_query_step_cpu_retry

  @allow_query_step_cpu_retry.setter
  def allow_query_step_cpu_retry(self, allowed):
    self.c_config.get().exec.heterogeneous.allow_query_step_cpu_retry=allowed
  
  @property
  def enable_heterogeneous_execution(self):
    return self.c_config.get().exec.heterogeneous.enable_heterogeneous_execution

  @enable_heterogeneous_execution.setter
  def enable_heterogeneous_execution(self, enabled):
    self.c_config.get().exec.heterogeneous.enable_heterogeneous_execution=enabled
  
  @property
  def forced_heterogeneous_distribution(self):
    return self.c_config.get().exec.heterogeneous.forced_heterogeneous_distribution

  @forced_heterogeneous_distribution.setter
  def forced_heterogeneous_distribution(self, enabled):
    self.c_config.get().exec.heterogeneous.forced_heterogeneous_distribution=enabled
 
  @property
  def enable_multifrag_heterogeneous_execution(self):
    return self.c_config.get().exec.heterogeneous.enable_multifrag_heterogeneous_execution

  @enable_multifrag_heterogeneous_execution.setter
  def enable_multifrag_heterogeneous_execution(self, enabled):
    self.c_config.get().exec.heterogeneous.enable_multifrag_heterogeneous_execution=enabled

def buildConfig(*, enable_debug_timer=None, enable_union=False, log_dir="hdk_log", **kwargs):
  global g_enable_debug_timer
  if enable_debug_timer is not None:
    g_enable_debug_timer = enable_debug_timer

  # Remove legacy params to provide better compatibility with PyOmniSciDbe
  kwargs.pop("enable_union", None)
  kwargs.pop("enable_thrift_logs", None)

  cmd_str = "".join(' --%s %r' % arg for arg in kwargs.iteritems())
  cmd_str = cmd_str.replace("_", "-")
  cdef string app = "PyHDK".encode('UTF-8')
  cdef CConfigBuilder builder
  builder.parseCommandLineArgs(app, cmd_str, False)
  cdef Config config = Config()
  config.c_config = builder.config()
  config.c_config.get().debug.log_dir = log_dir
  return config

def initLogger(*, debug_logs=False, log_dir="hdk_log", **kwargs):
  argv0 = "PyHDK".encode('UTF-8')
  cdef char *cargv0 = argv0
  cdef string default_log_dir = log_dir
  cdef unique_ptr[CLogOptions] opts = make_unique[CLogOptions](cargv0, default_log_dir)
  cmd_str = "".join(' --%s %r' % arg for arg in kwargs.iteritems())
  cmd_str = cmd_str.replace("_", "-")
  opts.get().parse_command_line(argv0, cmd_str)
  if debug_logs:
    opts.get().severity_ = CSeverity.DEBUG3
  CInitLogger(dereference(opts))
