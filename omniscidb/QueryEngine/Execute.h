/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef QUERYENGINE_EXECUTE_H
#define QUERYENGINE_EXECUTE_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <rapidjson/document.h>

#include "BufferProvider/BufferProvider.h"
#include "QueryEngine/AggregatedColRange.h"
#include "QueryEngine/CartesianProduct.h"
#include "QueryEngine/CgenState.h"
#include "QueryEngine/CodeCache.h"
#include "QueryEngine/CodeCacheAccessor.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Compiler/Backend.h"
#include "QueryEngine/Compiler/Exceptions.h"
#include "QueryEngine/DateTimeUtils.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/ExecutionKernel.h"
#include "QueryEngine/ExtensionModules.h"
#include "QueryEngine/GpuSharedMemoryContext.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/LoopControlFlow/JoinLoop.h"
#include "QueryEngine/PlanState.h"
#include "QueryEngine/QueryPlanDagCache.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "QueryEngine/RowFuncBuilder.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "QueryEngine/TableGenerations.h"
#include "QueryEngine/WindowContext.h"

#include "DataMgr/Chunk/Chunk.h"
#include "IR/Expr.h"
#include "Logger/Logger.h"
#include "ResultSetRegistry/ResultSetTable.h"
#include "SchemaMgr/SchemaProvider.h"
#include "Shared/Config.h"
#include "Shared/funcannotations.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/thread_count.h"
#include "Shared/toString.h"
#include "StringDictionary/LruCache.hpp"
#include "StringDictionary/StringDictionary.h"

#include "CostModel/CostModel.h"

using QueryCompilationDescriptorOwned = std::unique_ptr<QueryCompilationDescriptor>;
class QueryMemoryDescriptor;
using QueryMemoryDescriptorOwned = std::unique_ptr<QueryMemoryDescriptor>;

class ColumnFetcher;

class WatchdogException : public std::runtime_error {
 public:
  WatchdogException(const std::string& cause) : std::runtime_error(cause) {}
};

enum FragmentSkipStatus { SKIPPABLE, NOT_SKIPPABLE, INVALID };

class Executor;

inline llvm::Value* get_arg_by_name(llvm::Function* func, const std::string& name) {
  for (auto& arg : func->args()) {
    if (arg.getName() == name) {
      return &arg;
    }
  }
  CHECK(false);
  return nullptr;
}

inline uint32_t log2_bytes(const uint32_t bytes) {
  switch (bytes) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    default:
      abort();
  }
}

inline const hdk::ir::Expr* extract_cast_arg(const hdk::ir::Expr* expr) {
  const auto cast_expr = dynamic_cast<const hdk::ir::UOper*>(expr);
  if (!cast_expr || !cast_expr->isCast()) {
    return expr;
  }
  return cast_expr->operand();
}

inline std::string numeric_type_name(const hdk::ir::Type* type) {
  if (type->isInteger() || type->isDecimal() || type->isBoolean()) {
    return "int" + std::to_string(type->size() * 8) + "_t";
  }
  if (type->isFloatingPoint()) {
    return type->isFp64() ? "double" : "float";
  }
  if (type->isExtDictionary()) {
    return "int32_t";
  }
  CHECK(type->isDateTime() || type->isInterval())
      << "Unexpected type: " << type->toString();
  return "int64_t";
}

inline hdk::ResultSetTableTokenPtr get_temporary_table(
    const TemporaryTables* temporary_tables,
    const int table_id) {
  CHECK_LT(table_id, 0);
  const auto it = temporary_tables->find(table_id);
  CHECK(it != temporary_tables->end());
  return it->second;
}

// TODO(alex): Adjust interfaces downstream and make this not needed.
inline std::vector<const hdk::ir::Expr*> get_exprs_not_owned(
    const std::vector<hdk::ir::ExprPtr>& exprs) {
  std::vector<const hdk::ir::Expr*> exprs_not_owned;
  for (const auto& expr : exprs) {
    exprs_not_owned.push_back(expr.get());
  }
  return exprs_not_owned;
}

// Throwing QueryMustRunOnCpu allows us retry a query step on CPU if
// allow_query_step_cpu_retry is true (on by default) by catching
// the exception at the query step execution level in RelAlgExecutor,
// or if allow_query_step_cpu_retry is false but allow_cpu_retry is true,
// by retrying the entire query on CPU (if both flags are false, we return an
// error). This flag is thrown for the following broad categories of conditions:
// 1) we have not implemented an operator on GPU and so cannot codegen for GPU
// 2) we catch an unexpected GPU compilation/linking error (perhaps due
//    to an outdated driver/CUDA installation not allowing a modern operator)
// 3) when we detect up front that we will not have enough GPU memory to execute
//    a query.
// There is a fourth scenerio where our pre-flight GPU memory check passed but for
// whatever reason we still run out of memory. In those cases we go down the
// handleOutOfMemoryRetry path, which will first try per-fragment execution on GPU,
// and if that fails, CPU execution.

class QueryMustRunOnCpu : public std::runtime_error {
 public:
  QueryMustRunOnCpu() : std::runtime_error("Query must run in cpu mode.") {}

  QueryMustRunOnCpu(const std::string& err) : std::runtime_error(err) {}
};

class StringConstInResultSet : public std::runtime_error {
 public:
  StringConstInResultSet()
      : std::runtime_error(
            "NONE ENCODED String types are not supported as input result set.") {}
};

class ExtensionFunction;
struct ColumnDescriptor;

using ColumnToFragmentsMap = std::map<const ColumnDescriptor*, std::set<int32_t>>;
using TableToFragmentIds = std::map<int32_t, std::set<int32_t>>;

struct TableUpdateMetadata {
  ColumnToFragmentsMap columns_for_metadata_update;
  TableToFragmentIds fragments_with_deleted_rows;
};

using LLVMValueVector = std::vector<llvm::Value*>;

class QueryCompilationDescriptor;

std::ostream& operator<<(std::ostream&, FetchResult const&);

struct StreamExecutionContext {
  std::unique_ptr<QueryCompilationDescriptor> query_comp_desc;
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;
  std::unique_ptr<ColumnCacheMap> column_cache;
  std::unique_ptr<ColumnFetcher> column_fetcher;
  RelAlgExecutionUnit ra_exe_unit;
  CompilationOptions co;
  ExecutionOptions eo;
  std::unique_ptr<SharedKernelContext> shared_context;
  bool is_agg;

  StreamExecutionContext(RelAlgExecutionUnit ra_exe_unit,
                         const CompilationOptions& co,
                         const ExecutionOptions& eo)
      : ra_exe_unit(ra_exe_unit), co(co), eo(eo) {}
};

class Executor : public StringDictionaryProvider {
  static_assert(sizeof(float) == 4 && sizeof(double) == 8,
                "Host hardware not supported, unexpected size of float / double.");
  static_assert(sizeof(time_t) == 8,
                "Host hardware not supported, 64-bit time support is required.");

 public:
  using ExecutorId = size_t;
  static const ExecutorId INVALID_EXECUTOR_ID = SIZE_MAX;

  // NOTE: Executor should always be initialized through getExecutor to ensure the
  // executors map is populated
  Executor(const ExecutorId id,
           Data_Namespace::DataMgr* data_mgr,
           ConfigPtr config,
           const std::string& debug_dir,
           const std::string& debug_file);

  void clearCaches(bool runtime_only = false);

  void reset(const bool discard_runtime_modules_only = false);

  static std::shared_ptr<Executor> getExecutor(Data_Namespace::DataMgr* data_mgr,
                                               ConfigPtr config = nullptr,
                                               const std::string& debug_dir = "",
                                               const std::string& debug_file = "");

  // runs clear memory routines under the executor lock to prevent flushing pages in use
  static void clearMemory(const Data_Namespace::MemoryLevel memory_level,
                          Data_Namespace::DataMgr* data_mgr);

  static size_t getArenaBlockSize();

  static void addUdfIrToModule(const std::string& udf_ir_filename, const bool is_cuda_ir);

  // Globally available mapping of extension module sources. Not thread-safe.
  static std::map<ExtModuleKinds, std::string> extension_module_sources;
  static void initialize_extension_module_sources();

  bool has_udf_module(bool is_gpu = false) const {
    return has_extension_module(
        (is_gpu ? ExtModuleKinds::udf_gpu_module : ExtModuleKinds::udf_cpu_module));
  }
  bool has_rt_udf_module(bool is_gpu = false) const {
    return has_extension_module(
        (is_gpu ? ExtModuleKinds::rt_udf_gpu_module : ExtModuleKinds::rt_udf_cpu_module));
  }

  ExtensionModuleContext* getExtensionModuleContext() const {
    CHECK(extension_module_context_);
    return extension_module_context_.get();
  }

  std::shared_ptr<costmodel::CostModel> getCostModel();

  /**
   * Returns pointer to the intermediate tables vector currently stored by this
   * executor.
   */
  const TemporaryTables* getTemporaryTables() const;
  hdk::ResultSetTableTokenPtr getTemporaryTable(int table_id) const;

  /**
   * Returns a string dictionary proxy using the currently active row set memory owner.
   */
  virtual StringDictionary* getStringDictionary(const int dict_id,
                                                const bool with_generation) const {
    CHECK(row_set_mem_owner_);
    return getStringDictionaryProxy(dict_id, row_set_mem_owner_, with_generation);
  }

  StringDictionary* getStringDictionaryProxy(
      const int dictId,
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const bool with_generation) const;

  const std::vector<int32_t>* getStringProxyTranslationMap(
      const int source_dict_id,
      const int dest_dict_id,
      const RowSetMemoryOwner::StringTranslationType translation_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const bool with_generation) const;

  const std::vector<int32_t>* getIntersectionStringProxyTranslationMap(
      const StringDictionary* source_proxy,
      const StringDictionary* dest_proxy,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const;

  bool isCPUOnly() const;

  bool needsUnnestDoublePatch(llvm::Value const* val_ptr,
                              const std::string& agg_base_name,
                              const bool threads_share_memory,
                              const CompilationOptions& co) const;

  bool isArchMaxwell(const ExecutorDeviceType dt) const;

  void prependForceSync();

  bool containsLeftDeepOuterJoin() const {
    return cgen_state_->contains_left_deep_outer_join_;
  }

  SchemaProviderPtr getSchemaProvider() const { return schema_provider_; }
  void setSchemaProvider(SchemaProviderPtr provider) { schema_provider_ = provider; }

  Data_Namespace::DataMgr* getDataMgr() const {
    CHECK(data_mgr_);
    return data_mgr_;
  }

  BufferProvider* getBufferProvider() const {
    CHECK(data_mgr_);
    return data_mgr_->getBufferProvider();
  }

  const Config& getConfig() const { return *config_; }

  ConfigPtr getConfigPtr() const { return config_; }

  const std::shared_ptr<RowSetMemoryOwner> getRowSetMemoryOwner() const;

  std::shared_ptr<const TableFragmentsInfo> getTableInfo(const int db_id,
                                                         const int table_id) const;

  const TableGeneration& getTableGeneration(int db_id, int table_id) const;

  ExpressionRange getColRange(const PhysicalInput&) const;

  size_t getNumBytesForFetchedRow(const std::set<int>& table_ids_to_fetch) const;

  bool hasLazyFetchColumns(const std::vector<const hdk::ir::Expr*>& target_exprs) const;
  std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
      const std::vector<const hdk::ir::Expr*>& target_exprs) const;

  void registerActiveModule(void* module, const int device_id) const;
  void unregisterActiveModule(void* module, const int device_id) const;
  void interrupt();
  void resetInterrupt();

  static const size_t high_scan_limit{32000000};

  int8_t warpSize() const;
  unsigned gridSize() const;
  unsigned numBlocksPerMP() const;
  unsigned blockSize() const;
  size_t maxGpuSlabSize() const;

  hdk::ResultSetTable executeWorkUnit(size_t& max_groups_buffer_entry_guess,
                                      const bool is_agg,
                                      const std::vector<InputTableInfo>&,
                                      const RelAlgExecutionUnit&,
                                      const CompilationOptions&,
                                      const ExecutionOptions& options,
                                      const bool has_cardinality_estimation,
                                      DataProvider* data_provider,
                                      ColumnCacheMap& column_cache);

  void addTransientStringLiterals(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::shared_ptr<RowSetMemoryOwner>& row_set_mem_owner);

 private:
  void clearMetaInfoCache();

  int deviceCount(const ExecutorDeviceType) const;
  int deviceCountForMemoryLevel(const Data_Namespace::MemoryLevel memory_level) const;

  // Generate code for a window function target.
  llvm::Value* codegenWindowFunction(const size_t target_index,
                                     const CompilationOptions& co);

  // Generate code for an aggregate window function target.
  llvm::Value* codegenWindowFunctionAggregate(const CompilationOptions& co);

  // The aggregate state requires a state reset when starting a new partition. Generate
  // the new partition check and return the continuation basic block.
  llvm::BasicBlock* codegenWindowResetStateControlFlow(const CompilationOptions& co);

  // Generate code for initializing the state of a window aggregate.
  void codegenWindowFunctionStateInit(llvm::Value* aggregate_state,
                                      const CompilationOptions& co);

  // Generates the required calls for an aggregate window function and returns the final
  // result.
  llvm::Value* codegenWindowFunctionAggregateCalls(llvm::Value* aggregate_state,
                                                   const CompilationOptions& co);

  // The AVG window function requires some post-processing: the sum is divided by count
  // and the result is stored back for the current row.
  void codegenWindowAvgEpilogue(llvm::Value* crt_val,
                                llvm::Value* window_func_null_val,
                                llvm::Value* multiplicity_lv,
                                const CompilationOptions& co);

  // Generates code which loads the current aggregate value for the window context.
  llvm::Value* codegenAggregateWindowState(const CompilationOptions& co);

  llvm::Value* aggregateWindowStatePtr(const CompilationOptions& co);

  CudaMgr_Namespace::CudaMgr* cudaMgr() const;

  GpuMgr* gpuMgr() const;

  bool deviceSupportsFP64(const ExecutorDeviceType dt) const;

  bool needFetchAllFragments(const InputColDescriptor& col_desc,
                             const RelAlgExecutionUnit& ra_exe_unit,
                             const FragmentsList& selected_fragments) const;

  bool needLinearizeAllFragments(const InputColDescriptor& inner_col_desc,
                                 const RelAlgExecutionUnit& ra_exe_unit,
                                 const FragmentsList& selected_fragments,
                                 const Data_Namespace::MemoryLevel memory_level) const;

  using PerFragmentCallBack = std::function<void(ResultSetPtr, const FragmentInfo&)>;

  /**
   * @brief Compiles and dispatches a work unit per fragment processing results with the
   * per fragment callback.
   * Currently used for computing metrics over fragments (metadata).
   */
  void executeWorkUnitPerFragment(const RelAlgExecutionUnit& ra_exe_unit,
                                  const InputTableInfo& table_info,
                                  const CompilationOptions& co,
                                  const ExecutionOptions& eo,
                                  DataProvider* data_provider,
                                  PerFragmentCallBack& cb,
                                  const std::set<size_t>& fragment_indexes_param);

  ResultSetPtr executeExplain(const QueryCompilationDescriptor&);

  ExecutorDeviceType getDeviceTypeForTargets(
      const RelAlgExecutionUnit& ra_exe_unit,
      const ExecutorDeviceType requested_device_type);

  bool needFallbackOnCPU(const RelAlgExecutionUnit& ra_exe_unit,
                         const ExecutorDeviceType requested_device_type);

  std::pair<std::unique_ptr<policy::ExecutionPolicy>, ExecutorDeviceType>
  getExecutionPolicyForTargets(const RelAlgExecutionUnit& ra_exe_unit,
                               const ExecutorDeviceType requested_device_type,
                               const std::vector<InputTableInfo>& query_infos,
                               size_t& max_groups_buffer_entry_guess,
                               const ExecutionOptions& eo);

  std::set<ExecutorDeviceType> getAvailableDeviceTypes() const;

  std::set<ExecutorDeviceType> getDeviceTypesForQuery(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& table_infos,
      const ExecutorDeviceType requested_dt,
      size_t& max_groups_buffer_entry_guess,
      const ExecutionOptions& eo);

  std::unique_ptr<policy::ExecutionPolicy> getExecutionPolicy(
      const bool is_agg,
      const std::map<ExecutorDeviceType, std::unique_ptr<QueryMemoryDescriptor>>&
          query_mem_descs,
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& table_infos,
      const ExecutionOptions& eo);

  hdk::ResultSetTable collectAllDeviceResults(
      SharedKernelContext& shared_context,
      const RelAlgExecutionUnit& ra_exe_unit,
      const QueryMemoryDescriptor& query_mem_desc,
      const ExecutorDeviceType device_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const CompilationOptions& co,
      const ExecutionOptions& eo);

  std::unordered_map<int, const hdk::ir::BinOper*> getInnerTabIdToJoinCond() const;

  /**
   * @brief Exprimental execution dispatch mode for heterogeneous kernel submission.
   *
   */
  std::vector<std::unique_ptr<ExecutionKernel>> createKernels(
      SharedKernelContext& shared_context,
      const RelAlgExecutionUnit& ra_exe_unit,
      ColumnFetcher& column_fetcher,
      const std::vector<InputTableInfo>& table_infos,
      const ExecutionOptions& eo,
      const CompilationOptions& co,
      const bool allow_single_frag_table_opt,
      const std::map<ExecutorDeviceType, std::unique_ptr<QueryCompilationDescriptor>>&
          query_comp_descs,
      const std::map<ExecutorDeviceType, std::unique_ptr<QueryMemoryDescriptor>>&
          query_mem_descs,
      const policy::ExecutionPolicy* policy,
      const size_t device_count);

  /**
   * Launches execution kernels created by `createKernels` asynchronously using a thread
   * pool.
   */
  void launchKernels(SharedKernelContext& shared_context,
                     std::vector<std::unique_ptr<ExecutionKernel>>&& kernels,
                     const ExecutorDeviceType device_type,
                     const CompilationOptions& co);

  std::vector<size_t> getTableFragmentIndices(
      const RelAlgExecutionUnit& ra_exe_unit,
      const ExecutorDeviceType device_type,
      const size_t table_idx,
      const size_t outer_frag_idx,
      std::map<TableRef, const TableFragments*>& selected_tables_fragments,
      const std::unordered_map<int, const hdk::ir::BinOper*>&
          inner_table_id_to_join_condition);

  bool skipFragmentPair(const FragmentInfo& outer_fragment_info,
                        const FragmentInfo& inner_fragment_info,
                        const int inner_table_id,
                        const std::unordered_map<int, const hdk::ir::BinOper*>&
                            inner_table_id_to_join_condition,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        const ExecutorDeviceType device_type);

  FetchResult fetchChunks(const ColumnFetcher&,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          const int device_id,
                          const Data_Namespace::MemoryLevel,
                          const std::map<TableRef, const TableFragments*>&,
                          const FragmentsList& selected_fragments,
                          std::list<ChunkIter>&,
                          std::list<std::shared_ptr<Chunk_NS::Chunk>>&,
                          DeviceAllocator* device_allocator,
                          const size_t thread_idx,
                          const bool allow_runtime_interrupt);

  FetchResult fetchUnionChunks(const ColumnFetcher&,
                               const RelAlgExecutionUnit& ra_exe_unit,
                               const int device_id,
                               const Data_Namespace::MemoryLevel,
                               const std::map<TableRef, const TableFragments*>&,
                               const FragmentsList& selected_fragments,
                               std::list<ChunkIter>&,
                               std::list<std::shared_ptr<Chunk_NS::Chunk>>&,
                               DeviceAllocator* device_allocator,
                               const size_t thread_idx,
                               const bool allow_runtime_interrupt);

  std::pair<std::vector<std::vector<int64_t>>, std::vector<std::vector<uint64_t>>>
  getRowCountAndOffsetForAllFrags(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::vector<size_t>>& frag_ids_crossjoin,
      const std::vector<InputDescriptor>& input_descs,
      const std::map<TableRef, const TableFragments*>& all_tables_fragments);

  void buildSelectedFragsMapping(
      std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
      std::vector<size_t>& local_col_to_frag_pos,
      const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
      const FragmentsList& selected_fragments,
      const RelAlgExecutionUnit& ra_exe_unit);

  void buildSelectedFragsMappingForUnion(
      std::vector<std::vector<size_t>>& selected_fragments_crossjoin,
      std::vector<size_t>& local_col_to_frag_pos,
      const std::list<std::shared_ptr<const InputColDescriptor>>& col_global_ids,
      const FragmentsList& selected_fragments,
      const RelAlgExecutionUnit& ra_exe_unit);

  std::vector<size_t> getFragmentCount(const FragmentsList& selected_fragments,
                                       const size_t scan_idx,
                                       const RelAlgExecutionUnit& ra_exe_unit);

  // pass nullptr to results if it shouldn't be extracted from the execution context
  int32_t executePlan(const RelAlgExecutionUnit& ra_exe_unit,
                      const CompilationResult&,
                      const bool hoist_literals,
                      ResultSetPtr* results,
                      const ExecutorDeviceType device_type,
                      const CompilationOptions& co,
                      std::vector<std::vector<const int8_t*>>& col_buffers,
                      const std::vector<size_t> outer_tab_frag_ids,
                      QueryExecutionContext*,
                      const std::vector<std::vector<int64_t>>& num_rows,
                      const std::vector<std::vector<uint64_t>>& frag_offsets,
                      Data_Namespace::DataMgr*,
                      const int device_id,
                      const int outer_table_id,
                      const int64_t limit,
                      const uint32_t start_rowid,
                      const uint32_t num_tables,
                      const bool allow_runtime_interrupt,
                      const int64_t rows_to_process = -1);

 public:  // Temporary, ask saman about this
  static std::pair<int64_t, int32_t> reduceResults(hdk::ir::AggType agg,
                                                   const hdk::ir::Type* type,
                                                   const int64_t agg_init_val,
                                                   const int8_t out_byte_width,
                                                   const int64_t* out_vec,
                                                   const size_t out_vec_sz,
                                                   const bool is_group_by,
                                                   const bool float_argument_input);

 private:
  hdk::ResultSetTable resultsUnion(SharedKernelContext& shared_context,
                                   const RelAlgExecutionUnit& ra_exe_unit,
                                   bool merge,
                                   bool sort_by_table_id = false,
                                   const std::map<int, size_t>& order_map = {});

  std::vector<int64_t> getJoinHashTablePtrs(const ExecutorDeviceType device_type,
                                            const int device_id);
  ResultSetPtr reduceMultiDeviceResults(
      const RelAlgExecutionUnit&,
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&,
      const CompilationOptions& co);
  ResultSetPtr reduceMultiDeviceResultSets(
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&,
      const CompilationOptions&);
  ResultSetPtr reduceSpeculativeTopN(
      const RelAlgExecutionUnit&,
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& all_fragment_results,
      std::shared_ptr<RowSetMemoryOwner>,
      const QueryMemoryDescriptor&) const;
  hdk::ResultSetTable reducePartitionHistogram(
      std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device,
      const QueryMemoryDescriptor& query_mem_desc,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) const;

  void allocateShuffleBuffers(const std::vector<InputTableInfo>& query_infos,
                              const RelAlgExecutionUnit& ra_exe_unit,
                              std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                              SharedKernelContext& shared_context);
  hdk::ResultSetTable executeWorkUnitImpl(size_t& max_groups_buffer_entry_guess,
                                          const bool is_agg,
                                          const bool allow_single_frag_table_opt,
                                          const std::vector<InputTableInfo>&,
                                          const RelAlgExecutionUnit&,
                                          const CompilationOptions&,
                                          const ExecutionOptions& options,
                                          std::shared_ptr<RowSetMemoryOwner>,
                                          const bool has_cardinality_estimation,
                                          DataProvider* data_provider,
                                          ColumnCacheMap& column_cache);
  hdk::ResultSetTable executeHeterogeneousWorkUnitImpl(
      size_t& max_groups_buffer_entry_guess,
      const bool is_agg,
      const bool allow_single_frag_table_opt,
      const std::vector<InputTableInfo>&,
      const RelAlgExecutionUnit&,
      const CompilationOptions&,
      const ExecutionOptions& options,
      std::shared_ptr<RowSetMemoryOwner>,
      const bool has_cardinality_estimation,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache);

  std::shared_ptr<StreamExecutionContext> prepareStreamingExecution(
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      const std::vector<InputTableInfo>& table_infos,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache);

  ResultSetPtr runOnBatch(std::shared_ptr<StreamExecutionContext> ctx,
                          const FragmentsList& fragments);

  hdk::ResultSetTable finishStreamExecution(std::shared_ptr<StreamExecutionContext> ctx);

  std::vector<llvm::Value*> inlineHoistedLiterals();

  std::tuple<CompilationResult, std::unique_ptr<QueryMemoryDescriptor>> compileWorkUnit(
      const std::vector<InputTableInfo>& query_infos,
      const RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      const GpuMgr* gpu_mgr,
      const bool allow_lazy_fetch,
      std::shared_ptr<RowSetMemoryOwner>,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      const bool has_cardinality_estimation,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache);

  std::vector<JoinLoop> buildJoinLoops(RelAlgExecutionUnit& ra_exe_unit,
                                       const CompilationOptions& co,
                                       const ExecutionOptions& eo,
                                       const std::vector<InputTableInfo>& query_infos,
                                       DataProvider* data_provider,
                                       ColumnCacheMap& column_cache);

  // Create a callback which hoists left hand side filters above the join for left
  // joins, eliminating extra computation of the probe and matches if the row does not
  // pass the filters
  JoinLoop::HoistedFiltersCallback buildHoistLeftHandSideFiltersCb(
      const RelAlgExecutionUnit& ra_exe_unit,
      const size_t level_idx,
      const int inner_table_id,
      const CompilationOptions& co);
  // Builds a join hash table for the provided conditions on the current level.
  // Returns null iff on failure and provides the reasons in `fail_reasons`.
  std::shared_ptr<HashJoin> buildCurrentLevelHashTable(
      const JoinCondition& current_level_join_conditions,
      size_t level_idx,
      RelAlgExecutionUnit& ra_exe_unit,
      const CompilationOptions& co,
      const std::vector<InputTableInfo>& query_infos,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache,
      std::vector<std::string>& fail_reasons);
  void redeclareFilterFunction();
  llvm::Value* addJoinLoopIterator(const std::vector<llvm::Value*>& prev_iters,
                                   const size_t level_idx);
  void codegenJoinLoops(const std::vector<JoinLoop>& join_loops,
                        const RelAlgExecutionUnit& ra_exe_unit,
                        RowFuncBuilder& row_func_builder,
                        llvm::Function* query_func,
                        llvm::BasicBlock* entry_bb,
                        QueryMemoryDescriptor& query_mem_desc,
                        const CompilationOptions& co,
                        const ExecutionOptions& eo);
  bool compileBody(const RelAlgExecutionUnit& ra_exe_unit,
                   RowFuncBuilder& row_func_builder,
                   QueryMemoryDescriptor& query_mem_desc,
                   const CompilationOptions& co,
                   const GpuSharedMemoryContext& gpu_smem_context = {});

  void createErrorCheckControlFlow(llvm::Function* query_func,
                                   bool run_with_dynamic_watchdog,
                                   bool run_with_allowing_runtime_interrupt,
                                   ExecutorDeviceType device_type,
                                   const std::vector<InputTableInfo>& input_table_infos);

  void insertErrorCodeChecker(llvm::Function* query_func,
                              bool hoist_literals,
                              bool allow_runtime_query_interrupt);

  void preloadFragOffsets(const std::vector<InputDescriptor>& input_descs,
                          const std::vector<InputTableInfo>& query_infos);

  struct JoinHashTableOrError {
    std::shared_ptr<HashJoin> hash_table;
    std::string fail_reason;
  };

  JoinHashTableOrError buildHashTableForQualifier(
      const std::shared_ptr<const hdk::ir::BinOper>& qual_bin_oper,
      const std::vector<InputTableInfo>& query_infos,
      const MemoryLevel memory_level,
      const JoinType join_type,
      const HashType preferred_hash_type,
      DataProvider* data_provider,
      ColumnCacheMap& column_cache,
      const HashTableBuildDagMap& hashtable_build_dag_map,
      const TableIdToNodeMap& table_id_to_node_map);
  void nukeOldState(const bool allow_lazy_fetch,
                    const std::vector<InputTableInfo>& query_infos,
                    const RelAlgExecutionUnit* ra_exe_unit);

  std::shared_ptr<CompilationContext> optimizeAndCodegenCPU(
      llvm::Function*,
      llvm::Function*,
      std::shared_ptr<compiler::Backend>,
      const std::unordered_set<llvm::Function*>&,
      const CompilationOptions&);
  std::shared_ptr<CompilationContext> optimizeAndCodegenGPU(
      llvm::Function*,
      llvm::Function*,
      std::shared_ptr<compiler::Backend>,
      std::unordered_set<llvm::Function*>&,
      const CompilationOptions&);

  int64_t deviceCycles(int milliseconds) const;

  struct GroupColLLVMValue {
    llvm::Value* translated_value;
    llvm::Value* original_value;
  };

  GroupColLLVMValue groupByColumnCodegen(const hdk::ir::Expr* group_by_col,
                                         const size_t col_width,
                                         const CompilationOptions&,
                                         const bool translate_null_val,
                                         const int64_t translated_null_val,
                                         DiamondCodegen&,
                                         std::stack<llvm::BasicBlock*>&,
                                         const bool thread_mem_shared);
  llvm::Value* arrayLoopCodegen(const hdk::ir::Expr* array_expr,
                                std::stack<llvm::BasicBlock*>& array_loops,
                                DiamondCodegen& diamond_codegen,
                                const CompilationOptions& co,
                                llvm::Value* array_size = nullptr);

  llvm::Value* castToFP(llvm::Value*,
                        const hdk::ir::Type* from_type,
                        const hdk::ir::Type* to_type);

  FragmentSkipStatus canSkipFragmentForFpQual(const hdk::ir::BinOper* comp_expr,
                                              const hdk::ir::ColumnVar* lhs_col,
                                              const FragmentInfo& fragment,
                                              const hdk::ir::Constant* rhs_const) const;

  std::pair<bool, int64_t> skipFragment(
      const InputDescriptor& table_desc,
      const FragmentInfo& frag_info,
      const std::list<hdk::ir::ExprPtr>& simple_quals,
      const std::vector<uint64_t>& frag_offsets,
      const size_t frag_idx,
      compiler::CodegenTraitsDescriptor codegen_traits_desc);

  std::pair<bool, int64_t> skipFragmentInnerJoins(
      const InputDescriptor& table_desc,
      const RelAlgExecutionUnit& ra_exe_unit,
      const FragmentInfo& fragment,
      const std::vector<uint64_t>& frag_offsets,
      const size_t frag_idx,
      compiler::CodegenTraitsDescriptor codegen_traits_desc);

  AggregatedColRange computeColRangesCache(
      const std::unordered_set<InputColDescriptor>& col_descs);
  StringDictionaryGenerations computeStringDictionaryGenerations(
      const std::unordered_set<InputColDescriptor>& col_descs);
  TableGenerations computeTableGenerations(
      std::unordered_set<std::pair<int, int>> phys_table_ids);

 public:
  void setupCaching(DataProvider* data_provider,
                    const std::unordered_set<InputColDescriptor>& col_descs,
                    const std::unordered_set<std::pair<int, int>>& phys_table_ids);

  void setColRangeCache(const AggregatedColRange& aggregated_col_range) {
    agg_col_range_cache_ = aggregated_col_range;
  }

  ExecutorId getExecutorId() const { return executor_id_; };

  // check whether the current session that this executor manages is interrupted
  // while performing non-kernel time task
  bool checkNonKernelTimeInterrupted() const;

  // true when we have matched cardinality, and false otherwise
  using CachedCardinality = std::pair<bool, size_t>;
  void addToCardinalityCache(const std::string& cache_key, const size_t cache_value);
  CachedCardinality getCachedCardinality(const std::string& cache_key);

  mapd_shared_mutex& getDataRecyclerLock();
  QueryPlanDagCache& getQueryPlanDagCache();
  JoinColumnsInfo getJoinColumnsInfo(const hdk::ir::Expr* join_expr,
                                     JoinColumnSide target_side,
                                     bool extract_only_col_id);

  CgenState* getCgenStatePtr() const { return cgen_state_.get(); }

  llvm::LLVMContext& getContext() { return *context_.get(); }
  void update_extension_modules(bool update_runtime_modules_only = false);

  bool isLazyFetchAllowed() const { return plan_state_->allow_lazy_fetch_; }

 private:
  std::vector<int8_t> serializeLiterals(
      const std::unordered_map<int, CgenState::LiteralValues>& literals,
      const int device_id);

  static size_t align(const size_t off_in, const size_t alignment) {
    size_t off = off_in;
    if (off % alignment != 0) {
      off += (alignment - off % alignment);
    }
    return off;
  }

  const ExecutorId executor_id_;
  std::unique_ptr<llvm::LLVMContext> context_;

 public:
  // CgenStateManager uses RAII pattern to ensure that recursive code
  // generation (e.g. as in multi-step multi-subqueries) uses a new
  // CgenState instance for each recursion depth while restoring the
  // old CgenState instances when returning from recursion.
  class CgenStateManager {
   public:
    CgenStateManager(Executor& executor);
    CgenStateManager(Executor& executor,
                     const bool allow_lazy_fetch,
                     const std::vector<InputTableInfo>& query_infos,
                     const RelAlgExecutionUnit* ra_exe_unit);
    ~CgenStateManager();

   private:
    Executor& executor_;
    std::chrono::steady_clock::time_point lock_queue_clock_;
    std::lock_guard<std::mutex> lock_;
    std::unique_ptr<CgenState> cgen_state_;
  };

 private:
  ConfigPtr config_;
  std::unique_ptr<CgenState> cgen_state_;

  ExtensionModuleContext* getExtModuleContext() const {
    CHECK(extension_module_context_);
    return extension_module_context_.get();
  }

  bool has_extension_module(ExtModuleKinds kind) const {
    auto& extension_modules = getExtModuleContext()->getExtensionModules();
    return extension_modules.find(kind) != extension_modules.end();
  }

  std::unique_ptr<ExtensionModuleContext> extension_module_context_;

  class FetchCacheAnchor {
   public:
    FetchCacheAnchor(CgenState* cgen_state)
        : cgen_state_(cgen_state), saved_fetch_cache(cgen_state_->fetch_cache_) {}
    ~FetchCacheAnchor() { cgen_state_->fetch_cache_.swap(saved_fetch_cache); }

   private:
    CgenState* cgen_state_;
    std::unordered_map<int, std::vector<llvm::Value*>> saved_fetch_cache;
  };

  llvm::Value* spillDoubleElement(llvm::Value* elem_val, llvm::Type* elem_ty);

  std::unique_ptr<PlanState> plan_state_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  StringDictionaryGenerations string_dictionary_generations_;

  static const int max_gpu_count{16};
  std::mutex gpu_exec_mutex_[max_gpu_count];

  static std::mutex gpu_active_modules_mutex_;
  static uint32_t gpu_active_modules_device_mask_;
  static void* gpu_active_modules_[max_gpu_count];
  // indicates whether this executor has been interrupted
  std::atomic<bool> interrupted_{false};

  mutable std::mutex str_dict_mutex_;

 public:
  static std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> s_stubs_accessor;
  static std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> s_code_accessor;
  static std::unique_ptr<CodeCacheAccessor<CpuCompilationContext>> cpu_code_accessor;
  static std::unique_ptr<CodeCacheAccessor<CompilationContext>> gpu_code_accessor;
  static size_t code_cache_size;  // for re-initializing code caches

  static void
  resetCodeCache();  // ensure code cache is destroyed before tearing down data mgr

 private:
  const uint32_t block_size_x_;
  const uint32_t grid_size_x_;
  const std::string debug_dir_;
  const std::string debug_file_;

  SchemaProviderPtr schema_provider_;
  Data_Namespace::DataMgr* data_mgr_;
  const TemporaryTables* temporary_tables_;
  TableIdToNodeMap table_id_to_node_map_;

  std::vector<std::vector<int8_t*>> shuffle_out_bufs_;
  std::vector<int8_t**> shuffle_out_buf_ptrs_;

  int64_t kernel_queue_time_ms_ = 0;
  int64_t compilation_queue_time_ms_ = 0;

  std::shared_ptr<costmodel::CostModel> cost_model;

  // Singleton instance used for an execution unit which is a project with window
  // functions.
  std::unique_ptr<WindowProjectNodeContext> window_project_node_context_owned_;
  // The active window function.
  WindowFunctionContext* active_window_function_{nullptr};

  mutable InputTableInfoCache input_table_info_cache_;
  AggregatedColRange agg_col_range_cache_;
  TableGenerations table_generations_;

  // for blocking executors for clear memory, etc
  static mapd_shared_mutex execute_mutex_;

  static std::unique_ptr<QueryPlanDagCache> query_plan_dag_cache_;
  static std::once_flag first_init_flag_;
  const QueryPlanHash INVALID_QUERY_PLAN_HASH{std::hash<std::string>{}(EMPTY_QUERY_PLAN)};
  static mapd_shared_mutex recycler_mutex_;
  static std::unordered_map<std::string, size_t> cardinality_cache_;

 public:
  static const int32_t ERR_DIV_BY_ZERO{1};
  static const int32_t ERR_OUT_OF_GPU_MEM{2};
  static const int32_t ERR_OUT_OF_SLOTS{3};
  static const int32_t ERR_UNSUPPORTED_SELF_JOIN{4};
  static const int32_t ERR_OUT_OF_CPU_MEM{6};
  static const int32_t ERR_OVERFLOW_OR_UNDERFLOW{7};
  static const int32_t ERR_OUT_OF_TIME{9};
  static const int32_t ERR_INTERRUPTED{10};
  static const int32_t ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED{11};
  static const int32_t ERR_TOO_MANY_LITERALS{12};
  static const int32_t ERR_STRING_CONST_IN_RESULTSET{13};
  static const int32_t ERR_SINGLE_VALUE_FOUND_MULTIPLE_VALUES{15};
  static const int32_t ERR_WIDTH_BUCKET_INVALID_ARGUMENT{16};

  // Although compilation is Executor-local, an executor may trigger
  // threaded compilations (see executeWorkUnitPerFragment) that share
  // executor cgen_state and LLVM context, for instance.
  //
  // Rule of thumb: when `executor->thread_id_ != logger::thread_id()`
  // and executor LLVM Context is being modified (modules are cloned,
  // etc), one should protect such a code with
  //
  //  std::lock_guard<std::mutex> compilation_lock(executor->compilation_mutex_);
  //
  // to ensure thread safety.
  std::mutex compilation_mutex_;
  const logger::ThreadId thread_id_;

  // Runtime extension function registration updates
  // extension_modules_ that needs to be kept blocked from codegen
  // until the update is complete.
  // TODO(adb): move to ExtensionModuleContext?
  static std::shared_mutex register_runtime_extension_functions_mutex_;

  static std::mutex kernel_mutex_;  // TODO: should this be executor-local mutex?

  static std::atomic<size_t> executor_id_ctr_;

  friend class BaselineJoinHashTable;
  friend class CodeGenerator;
  friend class ColumnFetcher;
  friend struct DiamondCodegen;  // cgen_state_
  friend class ExecutionKernel;
  friend class KernelSubtask;
  friend class HashJoin;  // cgen_state_
  friend class RowFuncBuilder;
  friend class QueryCompilationDescriptor;
  friend class QueryMemoryInitializer;
  friend class QueryFragmentDescriptor;
  friend class QueryExecutionContext;
  friend class ResultSet;
  friend class InValuesBitmap;
  friend class StringDictionaryTranslationMgr;
  friend class LeafAggregator;
  friend class PerfectJoinHashTable;
  friend class QueryRewriter;
  friend class PendingExecutionClosure;
  friend class RelAlgExecutor;
  friend class GpuReductionHelperJIT;  // ExtensionModuleContext
  friend struct TargetExprCodegenBuilder;
  friend struct TargetExprCodegen;
  friend class WindowProjectNodeContext;
};

inline std::string get_null_check_suffix(const hdk::ir::Type* lhs_type,
                                         const hdk::ir::Type* rhs_type) {
  if (!lhs_type->nullable() && !rhs_type->nullable()) {
    return "";
  }
  std::string null_check_suffix{"_nullable"};
  if (!lhs_type->nullable()) {
    CHECK(rhs_type->nullable());
    null_check_suffix += "_rhs";
  } else if (!rhs_type->nullable()) {
    null_check_suffix += "_lhs";
  }
  return null_check_suffix;
}

inline bool is_unnest(const hdk::ir::Expr* expr) {
  return dynamic_cast<const hdk::ir::UOper*>(expr) &&
         static_cast<const hdk::ir::UOper*>(expr)->isUnnest();
}

bool is_trivial_loop_join(const std::vector<InputTableInfo>& query_infos,
                          const RelAlgExecutionUnit& ra_exe_unit,
                          unsigned trivial_loop_join_threshold);

extern "C" RUNTIME_EXPORT void register_buffer_with_executor_rsm(int64_t exec,
                                                                 int8_t* buffer);

const hdk::ir::Expr* remove_cast_to_int(const hdk::ir::Expr* expr);

inline std::string toString(const ExtModuleKinds& kind) {
  switch (kind) {
    case ExtModuleKinds::template_module:
      return "template_module";
    case ExtModuleKinds::l0_template_module:
      return "l0_template_module";
    case ExtModuleKinds::spirv_helper_funcs_module:
      return "spirv_helper_funcs_module";
    case ExtModuleKinds::rt_libdevice_module:
      return "rt_libdevice_module";
    case ExtModuleKinds::udf_cpu_module:
      return "udf_cpu_module";
    case ExtModuleKinds::udf_gpu_module:
      return "udf_gpu_module";
    case ExtModuleKinds::rt_udf_cpu_module:
      return "rt_udf_cpu_module";
    case ExtModuleKinds::rt_udf_gpu_module:
      return "rt_udf_gpu_module";
  }
  LOG(FATAL) << "Invalid LLVM module kind.";
  return "";
}

#endif  // QUERYENGINE_EXECUTE_H
