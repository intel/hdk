#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

cdef extern from "omniscidb/IR/Type.h":
  enum CTypeId "hdk::ir::Type::Id":
      kNull "hdk::ir::Type::Id::kNull" = 0,
      kBoolean "hdk::ir::Type::Id::kBoolean" = 1,
      kInteger "hdk::ir::Type::Id::kInteger" = 2,
      kFloatingPoint "hdk::ir::Type::Id::kFloatingPoint" = 3,
      kDecimal "hdk::ir::Type::Id::kDecimal" = 4,
      kVarChar "hdk::ir::Type::Id::kVarChar" = 5,
      kText "hdk::ir::Type::kText" = 6,
      kDate "hdk::ir::Type::kDate" = 7,
      kTime "hdk::ir::Type::Id::kTime" = 8,
      kTimestamp "hdk::ir::Type::Id::kTimestamp" = 9,
      kInterval "hdk::ir::Type::Id::kInterval" = 10,
      kFixedLenArray "hdk::ir::Type::Id::kFixedLenArray" = 11,
      kVarLenArray "hdk::ir::Type::Id::kVarLenArray" = 12,
      kExtDictionary "hdk::ir::Type::Id::kExtDictionary" = 13,
      kColumn "hdk::ir::Type::Id::kColumn" = 14,
      kColumnList "hdk::ir::Type::Id::kColumnList" = 15,

cdef extern from "omniscidb/IR/Type.h":
  cdef cppclass CType "hdk::ir::Type":
    
    CTypeId id()

    int size()
  
    bool nullable()

    bool isNull()
    bool isBoolean()
    bool isInteger()
    bool isFloatingPoint()
    bool isDecimal()
    bool isVarChar()
    bool isText()
    bool isDate()
    bool isTime()
    bool isTimestamp()
    bool isInterval()
    bool isFixedLenArray()
    bool isVarLenArray()
    bool isExtDictionary()
    bool isColumn()
    bool isColumnList()

    bool isInt8()
    bool isInt16()
    bool isInt32()
    bool isInt64()
    bool isFp32()
    bool isFp64()

    bool isNumber()
    bool isString()
    bool isDateTime()
    bool isArray()
    bool isVarLen()
    bool isBuffer()

    string toString()

    void print()

    const T* asType "as"[T]() const

  cdef cppclass CArrayBaseType "hdk::ir::ArrayBaseType"(CType):
    const CType* elemType() const

  cdef cppclass CVarLenArrayType "hdk::ir::VarLenArrayType"(CType):
    pass

  cdef cppclass CExtDictionaryType "hdk::ir::ExtDictionaryType"(CType):
    const CType* elemType() const

cdef extern from "omniscidb/IR/Context.h":
  cdef cppclass CContext "hdk::ir::Context":
    CContext()

    const CVarLenArrayType* arrayVarLen(const CType*, int, bool) except +

    @staticmethod
    CContext& defaultCtx()

    const CType* typeFromString(const string&) except +

cdef class TypeInfo:
  cdef const CType* c_type_info

cdef extern from "omniscidb/Utils/CommandLineOptions.h":
  cdef bool g_enable_debug_timer

cdef extern from "omniscidb/Logger/Logger.h" namespace "logger":
  enum CSeverity "Severity":
    DEBUG4 = 0,
    DEBUG3 = 1,
    DEBUG2 = 2,
    DEBUG1 = 3,
    INFO = 4,
    WARNING = 5,
    ERROR = 6,
    FATAL = 7,
    _NSEVERITIES = 8

  cdef cppclass CLogOptions "logger::LogOptions":
    CLogOptions(const char*)
    void parse_command_line(const string&, const string&)
    CSeverity severity_

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
    size_t override_gpu_block_size
    size_t override_gpu_grid_size
    bool cpu_only
    string initialize_with_gpu_vendor;

  cdef cppclass CFilterPushdownConfig "FilterPushdownConfig":
    bool enable
    float low_frac
    float high_frac
    size_t passing_row_ubound

  cdef cppclass COptimizationsConfig "OptimizationsConfig":
    CFilterPushdownConfig filter_pushdown
    bool from_table_reordering
    size_t constrained_by_in_threshold
    bool enable_left_join_filter_hoisting

  cdef cppclass CResultSetConfig "ResultSetConfig":
    bool enable_columnar_output
    bool optimize_row_initialization
    bool enable_direct_columnarization
    bool enable_lazy_fetch

  cdef cppclass CGpuMemoryConfig "GpuMemoryConfig":
    size_t min_memory_allocation_size
    size_t max_memory_allocation_size
    double input_mem_limit_percent
    size_t reserved_mem_bytes

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

  cdef cppclass CStorageConfig "StorageConfig":
    bool enable_lazy_dict_materialization

  cdef cppclass CConfig "Config":
    CExecutionConfig exec
    COptimizationsConfig opts
    CResultSetConfig rs
    CMemoryConfig mem
    CCacheConfig cache
    CDebugConfig debug
    CStorageConfig storage

ctypedef shared_ptr[CConfig] CConfigPtr

cdef class Config:
  cdef CConfigPtr c_config

cdef extern from "omniscidb/ConfigBuilder/ConfigBuilder.h":
  cdef cppclass CConfigBuilder "ConfigBuilder":
    CConfigBuilder()

    bool parseCommandLineArgs(const string&, const string&, bool) except +
    CConfigPtr config()

cdef extern from "boost/variant.hpp":
  T *boost_get "boost::get"[T](void *)
