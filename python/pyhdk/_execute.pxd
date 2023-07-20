#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.memory cimport shared_ptr, make_shared, unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CType, CConfig
from pyhdk._storage cimport MemoryLevel, DataMgr, CDataMgr, CBufferProvider, CSchemaProvider, CAbstractDataProvider

cdef extern from "omniscidb/QueryEngine/Compiler/CodegenTraitsDescriptor.h" namespace "compiler":
  enum CCallingConvDesc "CallingConvDesc":
    C "CallingConvDesc::C", 
    SPIR "CallingConvDesc::SPIR_FUNC",

  cdef cppclass CCodegenTraitsDescriptor "CodegenTraitsDescriptor":
    unsigned local_addr_space_
    unsigned global_addr_space_
    CCallingConvDesc conv_
    string triple_

cdef extern from "omniscidb/QueryEngine/CompilationOptions.h":
  enum CExecutorDeviceType "ExecutorDeviceType":
    CPU "ExecutorDeviceType::CPU",
    GPU "ExecutorDeviceType::GPU",

  enum CExecutorOptLevel "ExecutorOptLevel":
    OptLevel_Default "ExecutorOptLevel::Default",
    ReductionJIT "ExecutorOptLevel::ReductionJIT",

  enum CExecutorExplainType "ExecutorExplainType":
    ExplainType_Default "ExecutorExplainType::Default",
    Optimized "ExecutorExplainType::Optimized",

  enum CExecutorDispatchMode "ExecutorDispatchMode":
    KernelPerFragment "ExecutorDispatchMode::KernelPerFragment",
    MultifragmentKernel "ExecutorDispatchMode::MultifragmentKernel",

  cdef cppclass CCompilationOptions "CompilationOptions":
    CExecutorDeviceType device_type
    bool hoist_literals
    CExecutorOptLevel opt_level
    bool with_dynamic_watchdog
    bool allow_lazy_fetch
    bool filter_on_deleted_column
    CExecutorExplainType explain_type
    bool register_intel_jit_listener
    bool use_groupby_buffer_desc
    CCodegenTraitsDescriptor codegen_traits_desc

    @staticmethod
    CCompilationOptions makeCpuOnly(const CCompilationOptions&)

    @staticmethod
    CCodegenTraitsDescriptor getCgenTraitsDesc(const CExecutorDeviceType, const bool)

    @staticmethod
    CCompilationOptions defaults(const CExecutorDeviceType, const bool)

  enum CExecutorType "ExecutorType":
    Native "ExecutorType::Native",
    Extern "ExecutorType::Extern",

  cdef cppclass CExecutionOptions "ExecutionOptions":
    bool output_columnar_hint
    bool allow_multifrag
    bool just_explain
    bool allow_loop_joins
    bool with_watchdog
    bool jit_debug
    bool just_validate
    bool with_dynamic_watchdog
    unsigned dynamic_watchdog_time_limit
    bool find_push_down_candidates
    bool just_calcite_explain
    double gpu_input_mem_limit_percent
    bool allow_runtime_query_interrupt
    double running_query_interrupt_freq
    unsigned forced_gpu_proportion
    unsigned forced_cpu_proportion
    unsigned pending_query_interrupt_freq
    CExecutorType executor_type
    vector[size_t] outer_fragment_indices
    bool multifrag_result
    bool preserve_order

    @staticmethod
    CExecutionOptions fromConfig(const CConfig)

cdef extern from "omniscidb/ResultSet/TargetValue.h":
  cdef cppclass CNullableString "NullableString":
    pass

  cdef cppclass CScalarTargetValue "ScalarTargetValue":
    pass

  cdef cppclass CArrayTargetValue "ArrayTargetValue":
    bool operator bool()
    vector[CScalarTargetValue] &operator *()

  cdef cppclass CTargetValue "TargetValue":
    pass

  bool isNull(const CScalarTargetValue&, const CType*)
  bool isFloat(const CScalarTargetValue&)
  float getFloat(const CScalarTargetValue&)
  bool isDouble(const CScalarTargetValue&)
  double getDouble(const CScalarTargetValue&)
  bool isInt(const CScalarTargetValue&)
  int64_t getInt(const CScalarTargetValue&)
  bool isString(const CScalarTargetValue&)
  string getString(const CScalarTargetValue&)

cdef extern from "omniscidb/ResultSet/ResultSet.h":
  cdef cppclass CResultSet "ResultSet":
    size_t rowCount()

    string toString() const
    string contentToString(bool) const
    string summaryToString() const

    vector[CTargetValue] getRowAt(size_t, bool, bool) except +

ctypedef shared_ptr[CResultSet] CResultSetPtr

cdef extern from "omniscidb/ResultSetRegistry/ResultSetRegistry.h":
  cdef cppclass CResultSetRegistry "hdk::ResultSetRegistry"(CSchemaProvider, CAbstractDataProvider):
    CResultSetRegistry(shared_ptr[CConfig]) except +;

cdef extern from "omniscidb/IR/TargetMetaInfo.h":
  cdef cppclass CTargetMetaInfo "hdk::ir::TargetMetaInfo":
    const string& get_resname()
    const CType* type()

cdef extern from "omniscidb/ResultSet/ArrowResultSet.h":
  cdef cppclass CArrowResultSetConverter "ArrowResultSetConverter":
    CArrowResultSetConverter(const CResultSetPtr&, const vector[string]&, int)

    shared_ptr[CArrowTable] convertToArrowTable()

cdef extern from "omniscidb/QueryEngine/Execute.h":
  cdef cppclass CExecutor "Executor":
    @staticmethod
    shared_ptr[CExecutor] getExecutor(CDataMgr*, shared_ptr[CConfig], const string&, const string&)
    @staticmethod
    void clearMemory(MemoryLevel, CDataMgr*)
    const CConfig &getConfig()
    shared_ptr[CConfig] getConfigPtr()

cdef class Executor:
  cdef shared_ptr[CExecutor] c_executor
