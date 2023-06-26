#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CConfig, CType
from pyhdk._storage cimport CSchemaProvider, CSchemaProviderPtr, CDataProvider, CDataMgr, CBufferProvider
from pyhdk._execute cimport CExecutor, CResultSetPtr, CCompilationOptions, CExecutionOptions, CTargetMetaInfo

cdef extern from "omniscidb/QueryEngine/ExtensionFunctionsWhitelist.h":
  cdef cppclass CExtensionFunction "ExtensionFunction":
    pass

  cdef cppclass CExtensionFunctionsWhitelist "ExtensionFunctionsWhitelist":
    @staticmethod
    void add(const string&)

    @staticmethod
    void addUdfs(const string)

cdef extern from "omniscidb/Calcite/CalciteJNI.h":
  cdef cppclass FilterPushDownInfo:
    int input_prev;
    int input_start;
    int input_next;

  cdef cppclass CalciteMgr:    
    @staticmethod
    CalciteMgr* get(const string&, const string&, size_t);
    
    string process(const string&, const string&, CSchemaProvider*, CConfig*, const vector[FilterPushDownInfo]&, bool, bool, bool) except +

    string getExtensionFunctionWhitelist()
    string getUserDefinedFunctionWhitelist()
    void setRuntimeExtensionFunctions(const vector[CExtensionFunction]&, bool)

cdef extern from "omniscidb/IR/Node.h":
  cdef cppclass CQueryDag "hdk::ir::QueryDag":
    pass

cdef class QueryDag:
  cdef unique_ptr[CQueryDag] c_dag

cdef extern from "omniscidb/QueryEngine/RelAlgDagBuilder.h":
  cdef cppclass CRelAlgDagBuilder "RelAlgDagBuilder"(CQueryDag):
    CRelAlgDagBuilder(const string&, int, CSchemaProviderPtr, shared_ptr[CConfig]) except +

cdef extern from "omniscidb/ResultSetRegistry/ResultSetTableToken.h":
  cdef cppclass CResultSetTableToken "hdk::ResultSetTableToken":
    size_t rowCount()
    shared_ptr[CArrowTable] toArrow() except +
    string description()
    string memoryDescription()
    string contentToString(bool)

ctypedef shared_ptr[const CResultSetTableToken] CResultSetTableTokenPtr

cdef extern from "omniscidb/QueryEngine/Descriptors/RelAlgExecutionDescriptor.h":
  cdef cppclass CExecutionResult "ExecutionResult":
    CExecutionResult()
    CExecutionResult(const CExecutionResult&)
    CExecutionResult(CExecutionResult&&)

    const CResultSetPtr& getRows()
    const vector[CTargetMetaInfo]& getTargetsMeta()
    string getExplanation()
    const string& tableName()
    CResultSetTableTokenPtr getToken()

    CExecutionResult head(size_t) except +
    CExecutionResult tail(size_t) except +

cdef class ExecutionResult:
  cdef CExecutionResult c_result
  # DataMgr has to outlive ResultSet objects to avoid use-after-free errors.
  # Currently, C++ library doesn't enforce this and user is responsible for
  # obects lifetime control. In Python we achieve it by holding DataMgr in
  # each ExecutionResult object.
  cdef shared_ptr[CDataMgr] c_data_mgr
  # Scan object is used to forward query builder related calls. It is used
  # to provide an ability to work with an execution result as with a regular
  # table
  cdef object _scan

cdef extern from "omniscidb/QueryEngine/RelAlgExecutor.h":
  cdef cppclass CRelAlgExecutor "RelAlgExecutor":
    CRelAlgExecutor(CExecutor*, CSchemaProviderPtr, unique_ptr[CQueryDag])

    CExecutionResult executeRelAlgQuery(const CCompilationOptions&, const CExecutionOptions&, const bool) except +
    CExecutor *getExecutor()

cdef class RelAlgExecutor:
  cdef shared_ptr[CRelAlgExecutor] c_rel_alg_executor
  # DataMgr is used only to pass it to each produced ExecutionResult
  cdef shared_ptr[CDataMgr] c_data_mgr
