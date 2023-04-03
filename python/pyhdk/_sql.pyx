#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, make_unique
from libcpp.utility cimport move
from cython.operator cimport dereference, preincrement

from pyarrow.lib cimport pyarrow_wrap_table
from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CConfig, Config
from pyhdk._storage cimport SchemaProvider, CDataMgr, DataMgr
from pyhdk._execute cimport Executor, CExecutorDeviceType, CArrowResultSetConverter, CResultSet

cdef class Calcite:
  cdef CalciteMgr* calcite
  cdef CSchemaProviderPtr schema_provider
  cdef shared_ptr[CConfig] config

  def __cinit__(self, SchemaProvider schema_provider, Config config, **kwargs):
    cdef string udf_filename = kwargs.get("udf_filename", "")
    cdef size_t calcite_max_mem_mb = kwargs.get("calcite_max_mem_mb", 1024)

    self.calcite = CalciteMgr.get(udf_filename, calcite_max_mem_mb)
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
    cdef bool legacy_syntax = kwargs.get("legacy_syntax", False)
    cdef bool is_explain = kwargs.get("is_explain", False)
    cdef bool is_view_optimize = kwargs.get("is_view_optimize", False)
    return self.calcite.process(db_name, sql, self.schema_provider.get(), self.config.get(), filter_push_down_info, legacy_syntax, is_explain, is_view_optimize)

cdef class ExecutionResult:
  def row_count(self):
    cdef shared_ptr[CResultSet] c_res
    c_res = self.c_result.getRows()
    return int(c_res.get().rowCount())

  def to_arrow(self):
    cdef vector[string] col_names
    cdef vector[CTargetMetaInfo].const_iterator it = self.c_result.getTargetsMeta().const_begin()

    while it != self.c_result.getTargetsMeta().const_end():
      col_names.push_back(dereference(it).get_resname())
      preincrement(it)

    cdef unique_ptr[CArrowResultSetConverter] converter = make_unique[CArrowResultSetConverter](self.c_result.getRows(), col_names, -1)
    cdef shared_ptr[CArrowTable] at = converter.get().convertToArrowTable()
    return pyarrow_wrap_table(at)

  def to_explain_str(self):
    return self.c_result.getExplanation()

  @property
  def desc(self):
    return self.c_result.getRows().get().summaryToString()

  @property
  def memory_desc(self):
    return self.c_result.getRows().get().toString()

  @property
  def table_name(self):
    return self.c_result.tableName()

  @property
  def scan(self):
    return self._scan

  @scan.setter
  def scan(self, val):
    self._scan = val

  def __str__(self):
    res = "Schema:\n"
    for key, type_str in self.schema.items():
      res += f"  {key}: {type_str}\n"
    res += "Data:\n"
    res += self.c_result.getRows().get().contentToString(False)
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
    cdef CExecutionResult c_res = self.c_rel_alg_executor.get().executeRelAlgQuery(c_co, dereference(c_eo.get()), False)
    cdef ExecutionResult res = ExecutionResult()
    res.c_result = move(c_res)
    res.c_data_mgr = self.c_data_mgr
    return res
