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
from libcpp.memory cimport shared_ptr, make_shared, unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CSystemParameters, CSQLTypeInfo, CConfig
from pyhdk._storage cimport CDataMgr, CBufferProvider

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

    @staticmethod
    CCompilationOptions makeCpuOnly(const CCompilationOptions&)

    @staticmethod
    CCompilationOptions defaults(const CExecutorDeviceType)

  enum CExecutorType "ExecutorType":
    Native "ExecutorType::Native",
    Extern "ExecutorType::Extern",
    TableFunctions "ExecutorType::TableFunctions",

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
    unsigned pending_query_interrupt_freq
    CExecutorType executor_type
    vector[size_t] outer_fragment_indices
    bool multifrag_result
    bool preserve_order

    @staticmethod
    CExecutionOptions defaults()

cdef extern from "omniscidb/QueryEngine/ResultSet.h":
  cdef cppclass CResultSet "ResultSet":
    pass

ctypedef shared_ptr[CResultSet] CResultSetPtr

cdef extern from "omniscidb/QueryEngine/TargetMetaInfo.h":
  cdef cppclass CTargetMetaInfo "TargetMetaInfo":
    const string& get_resname()
    const CSQLTypeInfo& get_type_info()
    const CSQLTypeInfo& get_physical_type_info()

cdef extern from "omniscidb/QueryEngine/ArrowResultSet.h":
  cdef cppclass CArrowResultSetConverter "ArrowResultSetConverter":
    CArrowResultSetConverter(const CResultSetPtr&, const vector[string]&, int)

    shared_ptr[CArrowTable] convertToArrowTable()

cdef extern from "omniscidb/QueryEngine/Execute.h":
  cdef cppclass CExecutor "Executor":
    @staticmethod
    shared_ptr[CExecutor] getExecutor(size_t, CDataMgr*, CBufferProvider*, shared_ptr[CConfig], const string&, const string&, const CSystemParameters&)

    const CConfig &getConfig()
    shared_ptr[CConfig] getConfigPtr()

cdef class Executor:
  cdef shared_ptr[CExecutor] c_executor
