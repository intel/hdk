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
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

cdef extern from "omniscidb/Shared/sqltypes.h":
  enum CSQLTypes "SQLTypes":
    kNULLT = 0,
    kBOOLEAN = 1,
    kCHAR = 2,
    kVARCHAR = 3,
    kNUMERIC = 4,
    kDECIMAL = 5,
    kINT = 6,
    kSMALLINT = 7,
    kFLOAT = 8,
    kDOUBLE = 9,
    kTIME = 10,
    kTIMESTAMP = 11,
    kBIGINT = 12,
    kTEXT = 13,
    kDATE = 14,
    kARRAY = 15,
    kINTERVAL_DAY_TIME = 16,
    kINTERVAL_YEAR_MONTH = 17,
    kTINYINT = 18,
    kEVAL_CONTEXT_TYPE = 19,
    kVOID = 20,
    kCURSOR = 21,
    kCOLUMN = 22,
    kCOLUMN_LIST = 23,
    kSQLTYPE_LAST = 24,

  enum CEncodingType "EncodingType":
    kENCODING_NONE = 0,
    kENCODING_FIXED = 1,
    kENCODING_RL = 2,
    kENCODING_DIFF = 3,
    kENCODING_DICT = 4,
    kENCODING_SPARSE = 5,
    kENCODING_DATE_IN_DAYS = 7,
    kENCODING_LAST = 8,

  cdef cppclass CSQLTypeInfo "SQLTypeInfo":
    CSQLTypeInfo(CSQLTypes t, int d, int s, bool n, CEncodingType c, int p, CSQLTypes st)
    CSQLTypeInfo(CSQLTypes t, int d, int s, bool n)
    CSQLTypeInfo(CSQLTypes t, CEncodingType c, int p, CSQLTypes st)
    CSQLTypeInfo(CSQLTypes t, int d, int s)
    CSQLTypeInfo(CSQLTypes t, bool n)
    CSQLTypeInfo(CSQLTypes t)
    CSQLTypeInfo(CSQLTypes t, bool n, CEncodingType c)
    CSQLTypeInfo()

    CSQLTypes get_type()
    CSQLTypes get_subtype()
    int get_dimension()
    int get_precision()
    int get_input_srid()
    int get_scale()
    int get_output_srid()
    bool get_notnull()
    CEncodingType get_compression()
    int get_comp_param()
    int get_size()
    int get_logical_size()

    string toString()

cdef class TypeInfo:
  cdef CSQLTypeInfo c_type_info

cdef extern from "omniscidb/Shared/SystemParameters.h":
  cdef cppclass CSystemParameters "SystemParameters":
    CSystemParameters()

cdef extern from "omniscidb/ThriftHandler/CommandLineOptions.h":
  cdef bool g_enable_debug_timer

cdef extern from "omniscidb/Logger/Logger.h" namespace "logger":
  cdef cppclass CLogOptions "logger::LogOptions":
    CLogOptions(const char*)

  cdef void CInitLogger "logger::init"(const CLogOptions &)

cdef extern from "omniscidb/Shared/Config.h":
  cdef cppclass CWatchdogConfig "WatchdogConfig":
    bool enable
    bool enable_dynamic
    size_t time_limit
    size_t baseline_max_groups
    size_t parallel_top_max

  cdef cppclass CCpuSubTasksConfig "CpuSubTasksConfig":
    bool enable
    size_t sub_task_size

  cdef cppclass CJoinConfig "JoinConfig":
    bool allow_loop_joins
    unsigned trivial_loop_join_threshold
    bool inner_join_fragment_skipping
    size_t huge_join_hash_threshold
    size_t huge_join_hash_min_load

  cdef cppclass CGroupByConfig "GroupByConfig":
    bool bigint_count
    size_t default_max_groups_buffer_entry_guess
    size_t big_group_threshold
    bool use_groupby_buffer_desc
    bool enable_gpu_smem_group_by
    bool enable_gpu_smem_non_grouped_agg
    bool enable_gpu_smem_grouped_non_count_agg
    size_t gpu_smem_threshold
    unsigned hll_precision_bits
    size_t baseline_threshold

  cdef cppclass CWindowFunctionsConfig "WindowFunctionsConfig":
    bool enable
    bool parallel_window_partition_compute
    size_t parallel_window_partition_compute_threshold
    bool parallel_window_partition_sort
    size_t parallel_window_partition_sort_threshold

  cdef cppclass CHeterogenousConfig "HeterogenousConfig":
    bool enable_heterogeneous_execution
    bool enable_multifrag_heterogeneous_execution
    bool forced_heterogeneous_distribution
    unsigned forced_cpu_proportion
    unsigned forced_gpu_proportion
    bool allow_cpu_retry
    bool allow_query_step_cpu_retry

  cdef cppclass CInterruptConfig "InterruptConfig":
    bool enable_runtime_query_interrupt
    bool enable_non_kernel_time_query_interrupt
    double running_query_interrupt_freq

  cdef cppclass CCodegenConfig "CodegenConfig":
    bool inf_div_by_zero
    bool null_div_by_zero
    bool hoist_literals
    bool enable_filter_function

  cdef cppclass CExecutionConfig "ExecutionConfig":
    CWatchdogConfig watchdog
    CCpuSubTasksConfig sub_tasks
    CJoinConfig join
    CGroupByConfig group_by
    CWindowFunctionsConfig window_func
    CHeterogenousConfig heterogeneous
    CInterruptConfig interrupt
    CCodegenConfig codegen
    size_t streaming_topn_max
    size_t parallel_top_min
    bool enable_experimental_string_functions
    bool enable_interop
    size_t parallel_linearization_threshold
    bool enable_multifrag_rs

  cdef cppclass CFilterPushdownConfig "FilterPushdownConfig":
    bool enable
    float low_frac
    float high_frac
    size_t passing_row_ubound

  cdef cppclass COptimizationsConfig "OptimizationsConfig":
    CFilterPushdownConfig filter_pushdown
    bool from_table_reordering
    bool strip_join_covered_quals
    size_t constrained_by_in_threshold
    bool skip_intermediate_count
    bool enable_left_join_filter_hoisting

  cdef cppclass CResultSetConfig "ResultSetConfig":
    bool enable_columnar_output
    bool optimize_row_initialization
    bool enable_direct_columnarization
    bool enable_lazy_fetch

  cdef cppclass CGpuMemoryConfig "GpuMemoryConfig":
    bool enable_bump_allocator
    size_t min_memory_allocation_size
    size_t max_memory_allocation_size
    double bump_allocator_step_reduction
    double input_mem_limit_percent

  cdef cppclass CCpuMemoryConfig "CpuMemoryConfig":
    bool enable_tiered_cpu_mem
    size_t pmem_size

  cdef cppclass CMemoryConfig "MemoryConfig":
    CCpuMemoryConfig cpu
    CGpuMemoryConfig gpu

  cdef cppclass CCacheConfig "CacheConfig":
    bool use_estimator_result_cache
    bool enable_data_recycler
    bool use_hashtable_cache
    size_t hashtable_cache_total_bytes
    size_t max_cacheable_hashtable_size_bytes
    double gpu_fraction_code_cache_to_evict
    size_t dag_cache_size
    size_t code_cache_size

  cdef cppclass CDebugConfig "DebugConfig":
    string build_ra_cache
    string use_ra_cache
    bool enable_automatic_ir_metadata

  cdef cppclass CConfig "Config":
    CExecutionConfig exec
    COptimizationsConfig opts
    CResultSetConfig rs
    CMemoryConfig mem
    CCacheConfig cache
    CDebugConfig debug

cdef class Config:
  cdef shared_ptr[CConfig] c_config

cdef extern from "omniscidb/ConfigBuilder/ConfigBuilder.h":
  cdef cppclass CConfigBuilder "ConfigBuilder":
    CConfigBuilder()

    bool parseCommandLineArgs(const string&, const string&, bool) except +
    shared_ptr[CConfig] config()
