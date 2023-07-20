#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t
from libcpp.memory cimport make_shared, make_unique
from libcpp.utility cimport move
from cython.operator cimport dereference, preincrement, address

from pyarrow.lib cimport pyarrow_wrap_table
from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CConfig, Config, boost_get, CType, CArrayBaseType
from pyhdk._storage cimport SchemaProvider, CDataMgr, DataMgr
from pyhdk._execute cimport Executor, CExecutorDeviceType, CArrowResultSetConverter, CResultSet
from pyhdk._execute cimport CNullableString, CScalarTargetValue, CArrayTargetValue, CTargetValue, isNull
from pyhdk._execute cimport isNull, isInt, getInt, isFloat, getFloat, isDouble, getDouble, isString, getString

cdef class Calcite:
  cdef CalciteMgr* calcite
  cdef CSchemaProviderPtr schema_provider
  cdef shared_ptr[CConfig] config

  def __cinit__(self, SchemaProvider schema_provider, Config config, **kwargs):
    cdef string udf_filename = kwargs.get("udf_filename", "")
    cdef size_t calcite_max_mem_mb = kwargs.get("calcite_max_mem_mb", 1024)

    self.calcite = CalciteMgr.get(udf_filename, config.c_config.get().debug.log_dir, calcite_max_mem_mb)
    self.schema_provider = schema_provider.c_schema_provider
    self.config = config.c_config

    CExtensionFunctionsWhitelist.add(self.calcite.getExtensionFunctionWhitelist())
    if not udf_filename.empty():
      CExtensionFunctionsWhitelist.addUdfs(self.calcite.getUserDefinedFunctionWhitelist())

    cdef vector[CExtensionFunction] udfs = move(vector[CExtensionFunction]())
    self.calcite.setRuntimeExtensionFunctions(udfs, False)

  def process(self, string sql, **kwargs):
    cdef string db_name = kwargs.get("db_name", "test-db")
    cdef vector[FilterPushDownInfo] filter_push_down_info = vector[FilterPushDownInfo]()
    cdef bool legacy_syntax = kwargs.get("legacy_syntax", True)
    cdef bool is_explain = kwargs.get("is_explain", False)
    cdef bool is_view_optimize = kwargs.get("is_view_optimize", False)
    return self.calcite.process(db_name, sql, self.schema_provider.get(), self.config.get(), filter_push_down_info, legacy_syntax, is_explain, is_view_optimize)

cdef extract_scalar_value(const CScalarTargetValue &scalar, const CType *c_type):
  if isNull(scalar, c_type):
    return None
  if isInt(scalar):
    return getInt(scalar)
  if isFloat(scalar):
    return getFloat(scalar)
  if isDouble(scalar):
    return getDouble(scalar)
  if isString(scalar):
    return getString(scalar)
  return None

cdef extract_array_value(const CArrayTargetValue *array, const CType *c_type):
  cdef vector[CScalarTargetValue].const_iterator it
  cdef const CType* elem_type = c_type.asType[CArrayBaseType]().elemType()

  if dereference(array):
    res = []
    it = dereference(dereference(array)).const_begin()
    while it != dereference(dereference(array)).const_end():
      res.append(extract_scalar_value(dereference(it), elem_type))
      preincrement(it)
    return res

  return None

cdef class ExecutionResult:
  def row_count(self):
    cdef CResultSetTableTokenPtr c_token = self.c_result.getToken()
    return int(c_token.get().rowCount())

  def to_arrow(self):
    cdef CResultSetTableTokenPtr c_token = self.c_result.getToken()
    cdef shared_ptr[CArrowTable] at = c_token.get().toArrow()
    return pyarrow_wrap_table(at)

  def to_explain_str(self):
    return self.c_result.getExplanation()

  @property
  def desc(self):
    cdef CResultSetTableTokenPtr c_token = self.c_result.getToken()
    return c_token.get().description()

  @property
  def memory_desc(self):
    cdef CResultSetTableTokenPtr c_token = self.c_result.getToken()
    return c_token.get().memoryDescription()

  @property
  def table_name(self):
    return self.c_result.tableName()

  @property
  def scan(self):
    return self._scan

  @scan.setter
  def scan(self, val):
    self._scan = val

  def row(self, row_id):
    res = []
    cdef vector[CTargetValue] vals = self.c_result.getToken().get().row(row_id, True, True)
    cdef vector[CTargetValue].const_iterator it = vals.const_begin()
    cdef const CScalarTargetValue *scalar
    cdef const CArrayTargetValue *array
    cdef size_t col_idx = 0
    cdef const CType *col_type

    while col_idx < self.c_result.getTargetsMeta().size():
      col_type = self.c_result.getTargetsMeta().at(col_idx).type()

      scalar = boost_get[CScalarTargetValue](address(vals.at(col_idx)))
      if scalar != NULL:
        res.append(extract_scalar_value(dereference(scalar), col_type))
      else:
        array = boost_get[CArrayTargetValue](address(vals.at(col_idx)))
        if array != NULL:
          res.append(extract_array_value(array, col_type))
        else:
          res.append(None);

      col_idx += 1

    return res

  def head(self, n):
    res = ExecutionResult()
    res.c_result = self.c_result.head(n)
    res.c_data_mgr = self.c_data_mgr
    if self._scan is not None:
      res._scan = self._scan.hdk.scan(res.table_name)
    return res

  def tail(self, n):
    res = ExecutionResult()
    res.c_result = self.c_result.tail(n)
    res.c_data_mgr = self.c_data_mgr
    if self._scan is not None:
      res._scan = self._scan.hdk.scan(res.table_name)
    return res

  def __str__(self):
    res = "Schema:\n"
    for key, type_str in self.schema.items():
      res += f"  {key}: {type_str}\n"
    res += "Data:\n"
    res += self.c_result.getToken().get().contentToString(False)
    return res

  def __repr__(self):
    return self.__str__()

  def __getattr__(self, attr):
    return self._scan.__getattribute__(attr)

  def __getitem__(self, col):
    return self._scan.__getitem__(col)

cdef class RelAlgExecutor:
  def __cinit__(self, Executor executor, SchemaProvider schema_provider, DataMgr data_mgr, ra_json=None, QueryDag dag=None):
    cdef CExecutor* c_executor = executor.c_executor.get()
    cdef CSchemaProviderPtr c_schema_provider = schema_provider.c_schema_provider
    cdef unique_ptr[CQueryDag] c_dag
    cdef int db_id = 0

    # Choose the default database ID. Ignore ResultSetRegistry.
    db_ids = schema_provider.listDatabases()
    assert len(db_ids) <= 2
    if len(db_ids) == 1:
      db_id = db_ids[0]
    elif len(db_ids) == 2:
      db_id = db_ids[1] if db_ids[0] == ((100 << 24) + 1) else db_ids[0]

    if ra_json is not None:
      c_dag.reset(new CRelAlgDagBuilder(ra_json, db_id, c_schema_provider, c_executor.getConfigPtr()))
    else:
      assert dag is not None
      c_dag = move(dag.c_dag)

    self.c_rel_alg_executor = make_shared[CRelAlgExecutor](c_executor, c_schema_provider, move(c_dag))
    self.c_data_mgr = data_mgr.c_data_mgr

  def execute(self, **kwargs):
    cdef const CConfig *config = self.c_rel_alg_executor.get().getExecutor().getConfigPtr().get()
    cdef CCompilationOptions c_co
    if kwargs.get("device_type", "auto") == "GPU" and not config.exec.cpu_only:
      c_co = CCompilationOptions.defaults(CExecutorDeviceType.GPU, False)
    else:
      c_co = CCompilationOptions.defaults(CExecutorDeviceType.CPU, False)
    c_co.allow_lazy_fetch = kwargs.get("enable_lazy_fetch", config.rs.enable_lazy_fetch)
    c_co.with_dynamic_watchdog = kwargs.get("enable_dynamic_watchdog", config.exec.watchdog.enable_dynamic)
    cdef unique_ptr[CExecutionOptions] c_eo = make_unique[CExecutionOptions](CExecutionOptions.fromConfig(dereference(config)))
    c_eo.get().output_columnar_hint = kwargs.get("enable_columnar_output", config.rs.enable_columnar_output)
    c_eo.get().with_watchdog = kwargs.get("enable_watchdog", config.exec.watchdog.enable)
    c_eo.get().with_dynamic_watchdog = kwargs.get("enable_dynamic_watchdog", config.exec.watchdog.enable_dynamic)
    c_eo.get().just_explain = kwargs.get("just_explain", False)
    c_eo.get().forced_gpu_proportion = kwargs.get("forced_gpu_proportion", config.exec.heterogeneous.forced_gpu_proportion)
    c_eo.get().forced_cpu_proportion = 100 - c_eo.get().forced_gpu_proportion
    cdef CExecutionResult c_res = self.c_rel_alg_executor.get().executeRelAlgQuery(c_co, dereference(c_eo.get()), False)
    cdef ExecutionResult res = ExecutionResult()
    res.c_result = move(c_res)
    res.c_data_mgr = self.c_data_mgr
    return res
