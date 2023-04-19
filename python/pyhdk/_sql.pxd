#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyhdk._common cimport CConfig
from pyhdk._storage cimport CSchemaProviderPtr, CDataProvider, CDataMgr, CBufferProvider
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
    CalciteMgr* get(CSchemaProviderPtr, shared_ptr[CConfig], const string&, size_t);
    
    string process(const string&, const string&, const vector[FilterPushDownInfo]&, bool, bool, bool) except +

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

cdef extern from "omniscidb/QueryEngine/Descriptors/RelAlgExecutionDescriptor.h":
  cdef cppclass CExecutionResult "ExecutionResult":
    CExecutionResult()
    CExecutionResult(const CExecutionResult&)
    CExecutionResult(CExecutionResult&&)

    const CResultSetPtr& getRows()
    const vector[CTargetMetaInfo]& getTargetsMeta()
    string getExplanation()

cdef class ExecutionResult:
  cdef CExecutionResult c_result
  # DataMgr has to outlive ResultSet objects to avoid use-after-free errors.
  # Currently, C++ library doesn't enforce this and user is responsible for
  # obects lifetime control. In Python we achieve it by holding DataMgr in
  # each ExecutionResult object.
  cdef shared_ptr[CDataMgr] c_data_mgr

cdef extern from "omniscidb/QueryEngine/RelAlgExecutor.h":
  cdef cppclass CRelAlgExecutor "RelAlgExecutor":
    CRelAlgExecutor(CExecutor*, CSchemaProviderPtr, unique_ptr[CQueryDag])

    CExecutionResult executeRelAlgQuery(const CCompilationOptions&, const CExecutionOptions&, const bool) except +
    CExecutor *getExecutor()

cdef class RelAlgExecutor:
  cdef shared_ptr[CRelAlgExecutor] c_rel_alg_executor
  # DataMgr is used only to pass it to each produced ExecutionResult
  cdef shared_ptr[CDataMgr] c_data_mgr
