/*
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

#include "ConfigBuilder.h"

#include "Logger/Logger.h"

#include <boost/crc.hpp>
#include <boost/program_options.hpp>

#include <iostream>

namespace po = boost::program_options;

namespace {

template <typename T>
auto get_range_checker(T min, T max, const char* opt) {
  return [min, max, opt](T val) {
    if (val < min || val > max) {
      throw po::validation_error(
          po::validation_error::invalid_option_value, opt, std::to_string(val));
    }
  };
}

}  // namespace

ConfigBuilder::ConfigBuilder() {
  config_ = std::make_shared<Config>();
}

ConfigBuilder::ConfigBuilder(ConfigPtr config) : config_(config) {}

bool ConfigBuilder::parseCommandLineArgs(int argc,
                                         char const* const* argv,
                                         bool allow_gtest_flags) {
  po::options_description opt_desc;

  opt_desc.add_options()("help,h", "Show available options.");

  // exec.watchdog
  opt_desc.add_options()("enable-watchdog",
                         po::value<bool>(&config_->exec.watchdog.enable)
                             ->default_value(config_->exec.watchdog.enable)
                             ->implicit_value(true),
                         "Enable watchdog.");
  opt_desc.add_options()("enable-dynamic-watchdog",
                         po::value<bool>(&config_->exec.watchdog.enable_dynamic)
                             ->default_value(config_->exec.watchdog.enable_dynamic)
                             ->implicit_value(true),
                         "Enable dynamic watchdog.");
  opt_desc.add_options()("dynamic-watchdog-time-limit",
                         po::value<size_t>(&config_->exec.watchdog.time_limit)
                             ->default_value(config_->exec.watchdog.time_limit),
                         "Dynamic watchdog time limit, in milliseconds.");
  opt_desc.add_options()("watchdog-baseline-max-groups",
                         po::value<size_t>(&config_->exec.watchdog.baseline_max_groups)
                             ->default_value(config_->exec.watchdog.baseline_max_groups),
                         "Watchdog baseline aggregation groups limit.");
  opt_desc.add_options()(
      "parallel-top-max",
      po::value<size_t>(&config_->exec.watchdog.parallel_top_max)
          ->default_value(config_->exec.watchdog.parallel_top_max),
      "For ResultSets requiring a heap sort, the maximum number of rows allowed by "
      "watchdog.");

  // exec.sub_tasks
  opt_desc.add_options()(
      "enable-cpu-sub-tasks",
      po::value<bool>(&config_->exec.sub_tasks.enable)
          ->default_value(config_->exec.sub_tasks.enable)
          ->implicit_value(true),
      "Enable parallel processing of a single data fragment on CPU. This can improve CPU "
      "load balance and decrease reduction overhead.");
  opt_desc.add_options()("cpu-sub-task-size",
                         po::value<size_t>(&config_->exec.sub_tasks.sub_task_size)
                             ->default_value(config_->exec.sub_tasks.sub_task_size),
                         "Set CPU sub-task size in rows.");

  // exec.join
  opt_desc.add_options()("enable-loop-join",
                         po::value<bool>(&config_->exec.join.allow_loop_joins)
                             ->default_value(config_->exec.join.allow_loop_joins)
                             ->implicit_value(true),
                         "Enable/disable loop-based join execution.");
  opt_desc.add_options()(
      "loop-join-limit",
      po::value<unsigned>(&config_->exec.join.trivial_loop_join_threshold)
          ->default_value(config_->exec.join.trivial_loop_join_threshold),
      "Maximum number of rows in an inner table allowed for loop join.");
  opt_desc.add_options()("huge-join-hash-threshold",
                         po::value<size_t>(&config_->exec.join.huge_join_hash_threshold)
                             ->default_value(config_->exec.join.huge_join_hash_threshold),
                         "Number of etries in a pefect join hash table to make it "
                         "considered as a huge one.");
  opt_desc.add_options()(
      "huge-join-hash-min-load",
      po::value<size_t>(&config_->exec.join.huge_join_hash_min_load)
          ->default_value(config_->exec.join.huge_join_hash_min_load),
      "A minimal predicted load level for huge perfect hash tables in percent.");

  // exec.group_by
  opt_desc.add_options()("bigint-count",
                         po::value<bool>(&config_->exec.group_by.bigint_count)
                             ->default_value(config_->exec.group_by.bigint_count)
                             ->implicit_value(true),
                         "Use 64-bit count.");
  opt_desc.add_options()(
      "default-max-groups-buffer-entry-guess",
      po::value<size_t>(&config_->exec.group_by.default_max_groups_buffer_entry_guess)
          ->default_value(config_->exec.group_by.default_max_groups_buffer_entry_guess),
      "Default guess for group-by buffer size.");
  opt_desc.add_options()(
      "big-group-threshold",
      po::value<size_t>(&config_->exec.group_by.big_group_threshold)
          ->default_value(config_->exec.group_by.big_group_threshold),
      "Threshold at which guessed group-by buffer size causes NDV estimator to be used.");
  opt_desc.add_options()(
      "use-groupby-buffer-desc",
      po::value<bool>(&config_->exec.group_by.use_groupby_buffer_desc)
          ->default_value(config_->exec.group_by.use_groupby_buffer_desc)
          ->implicit_value(true),
      "Use GroupBy Buffer Descriptor for hash tables.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-group-by",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_group_by)
          ->default_value(config_->exec.group_by.enable_gpu_smem_group_by)
          ->implicit_value(true),
      "Enable using GPU shared memory for some GROUP BY queries.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-non-grouped-agg",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->default_value(config_->exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for non-grouped aggregate queries.");
  opt_desc.add_options()(
      "enable-gpu-shared-mem-grouped-non-count-agg",
      po::value<bool>(&config_->exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->default_value(config_->exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for grouped non-count aggregate queries.");
  opt_desc.add_options()(
      "enable-cpu-groupby-multifrag-kernels",
      po::value<bool>(&config_->exec.group_by.enable_cpu_multifrag_kernels)
          ->default_value(config_->exec.group_by.enable_cpu_multifrag_kernels)
          ->implicit_value(true),
      "Enable multifragment kernels for groupby queries on CPU.");
  opt_desc.add_options()(
      "gpu-shared-mem-threshold",
      po::value<size_t>(&config_->exec.group_by.gpu_smem_threshold)
          ->default_value(config_->exec.group_by.gpu_smem_threshold),
      "GPU shared memory threshold (in bytes). If query requires larger buffers than "
      "this threshold, we disable those optimizations. 0 means no static cap.");
  opt_desc.add_options()("hll-precision-bits",
                         po::value<unsigned>(&config_->exec.group_by.hll_precision_bits)
                             ->default_value(config_->exec.group_by.hll_precision_bits)
                             ->notifier(get_range_checker(1U, 16U, "hll-precision-bits")),
                         "Number of bits in range [1, 16] used from the hash value used "
                         "to specify the bucket number.");
  opt_desc.add_options()(
      "groupby-baseline-threshold",
      po::value<size_t>(&config_->exec.group_by.baseline_threshold)
          ->default_value(config_->exec.group_by.baseline_threshold),
      "Prefer baseline hash if number of entries exceeds this threshold.");
  opt_desc.add_options()("large-ndv-threshold",
                         po::value<int64_t>(&config_->exec.group_by.large_ndv_threshold)
                             ->default_value(config_->exec.group_by.large_ndv_threshold),
                         "Value range threshold at which large NDV estimator is used.");
  opt_desc.add_options()(
      "large-ndv-multiplier",
      po::value<size_t>(&config_->exec.group_by.large_ndv_multiplier)
          ->default_value(config_->exec.group_by.large_ndv_multiplier),
      "A multiplier applied to NDV estimator buffer size for large ranges.");

  // exec.window
  opt_desc.add_options()("enable-window-functions",
                         po::value<bool>(&config_->exec.window_func.enable)
                             ->default_value(config_->exec.window_func.enable)
                             ->implicit_value(true),
                         "Enable experimental window function support.");
  opt_desc.add_options()(
      "enable-parallel-window-partition-compute",
      po::value<bool>(&config_->exec.window_func.parallel_window_partition_compute)
          ->default_value(config_->exec.window_func.parallel_window_partition_compute)
          ->implicit_value(true),
      "Enable parallel window function partition computation.");
  opt_desc.add_options()(
      "parallel-window-partition-compute-threshold",
      po::value<size_t>(
          &config_->exec.window_func.parallel_window_partition_compute_threshold)
          ->default_value(
              config_->exec.window_func.parallel_window_partition_compute_threshold),
      "Parallel window function partition computation threshold (in rows).");
  opt_desc.add_options()(
      "enable-parallel-window-partition-sort",
      po::value<bool>(&config_->exec.window_func.parallel_window_partition_sort)
          ->default_value(config_->exec.window_func.parallel_window_partition_sort)
          ->implicit_value(true),
      "Enable parallel window function partition sorting.");
  opt_desc.add_options()(
      "parallel-window-partition-sort-threshold",
      po::value<size_t>(
          &config_->exec.window_func.parallel_window_partition_sort_threshold)
          ->default_value(
              config_->exec.window_func.parallel_window_partition_sort_threshold),
      "Parallel window function partition sorting threshold (in rows).");

  // exec.heterogeneous
  opt_desc.add_options()(
      "enable-heterogeneous",
      po::value<bool>(&config_->exec.heterogeneous.enable_heterogeneous_execution)
          ->default_value(config_->exec.heterogeneous.enable_heterogeneous_execution)
          ->implicit_value(true),
      "Allow the engine to schedule kernels heterogeneously.");
  opt_desc.add_options()(
      "enable-multifrag-heterogeneous",
      po::value<bool>(
          &config_->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->default_value(
              config_->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->implicit_value(true),
      "Allow mutifragment heterogeneous kernels.");
  opt_desc.add_options()(
      "force-heterogeneous-distribution",
      po::value<bool>(&config_->exec.heterogeneous.forced_heterogeneous_distribution)
          ->default_value(config_->exec.heterogeneous.forced_heterogeneous_distribution)
          ->implicit_value(true),
      "Keep user-defined load distribution in heterogeneous execution.");
  opt_desc.add_options()(
      "force-cpu-proportion",
      po::value<unsigned>(&config_->exec.heterogeneous.forced_cpu_proportion)
          ->default_value(config_->exec.heterogeneous.forced_cpu_proportion),
      "Set CPU proportion for forced heterogeneous distribution.");
  opt_desc.add_options()(
      "force-gpu-proportion",
      po::value<unsigned>(&config_->exec.heterogeneous.forced_gpu_proportion)
          ->default_value(config_->exec.heterogeneous.forced_gpu_proportion),
      "Set GPU proportion for forced heterogeneous distribution.");
  opt_desc.add_options()("allow-cpu-retry",
                         po::value<bool>(&config_->exec.heterogeneous.allow_cpu_retry)
                             ->default_value(config_->exec.heterogeneous.allow_cpu_retry)
                             ->implicit_value(true),
                         "Allow the queries which failed on GPU to retry on CPU, even "
                         "when watchdog is enabled.");
  opt_desc.add_options()(
      "allow-query-step-cpu-retry",
      po::value<bool>(&config_->exec.heterogeneous.allow_query_step_cpu_retry)
          ->default_value(config_->exec.heterogeneous.allow_query_step_cpu_retry)
          ->implicit_value(true),
      "Allow certain query steps to retry on CPU, even when allow-cpu-retry is disabled");

  // exec.interrupt
  opt_desc.add_options()(
      "enable-runtime-query-interrupt",
      po::value<bool>(&config_->exec.interrupt.enable_runtime_query_interrupt)
          ->default_value(config_->exec.interrupt.enable_runtime_query_interrupt)
          ->implicit_value(true),
      "Enable runtime query interrupt.");
  opt_desc.add_options()(
      "enable-non-kernel-time-query-interrupt",
      po::value<bool>(&config_->exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->default_value(config_->exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->implicit_value(true),
      "Enable non-kernel time query interrupt.");
  opt_desc.add_options()(
      "running-query-interrupt-freq",
      po::value<double>(&config_->exec.interrupt.running_query_interrupt_freq)
          ->default_value(config_->exec.interrupt.running_query_interrupt_freq),
      "A frequency of checking the request of running query "
      "interrupt from user (0.0 (less frequent) ~ (more frequent) 1.0).");

  // exec.codegen
  opt_desc.add_options()(
      "null-div-by-zero",
      po::value<bool>(&config_->exec.codegen.null_div_by_zero)
          ->default_value(config_->exec.codegen.null_div_by_zero)
          ->implicit_value(true),
      "Return NULL on division by zero instead of throwing an exception.");
  opt_desc.add_options()(
      "inf-div-by-zero",
      po::value<bool>(&config_->exec.codegen.inf_div_by_zero)
          ->default_value(config_->exec.codegen.inf_div_by_zero)
          ->implicit_value(true),
      "Return INF on fp division by zero instead of throwing an exception.");
  opt_desc.add_options()(
      "null-mod-by-zero",
      po::value<bool>(&config_->exec.codegen.null_mod_by_zero)
          ->default_value(config_->exec.codegen.null_mod_by_zero)
          ->implicit_value(true),
      "Return NULL on modulo by zero instead of throwing an exception.");
  opt_desc.add_options()("enable-hoist-literals",
                         po::value<bool>(&config_->exec.codegen.hoist_literals)
                             ->default_value(config_->exec.codegen.hoist_literals)
                             ->implicit_value(true),
                         "Enable literals hoisting during codegen to increase generated "
                         "code cache hit rate.");
  opt_desc.add_options()(
      "enable-filter-function",
      po::value<bool>(&config_->exec.codegen.enable_filter_function)
          ->default_value(config_->exec.codegen.enable_filter_function)
          ->implicit_value(true),
      "Enable the filter function protection feature for the SQL JIT compiler. "
      "Normally should be on but techs might want to disable for troubleshooting.");

  // exec
  opt_desc.add_options()("streaming-top-n-max",
                         po::value<size_t>(&config_->exec.streaming_topn_max)
                             ->default_value(config_->exec.streaming_topn_max),
                         "The maximum number of rows allowing streaming top-N sorting.");
  opt_desc.add_options()(
      "parallel-top-min",
      po::value<size_t>(&config_->exec.parallel_top_min)
          ->default_value(config_->exec.parallel_top_min),
      "For ResultSets requiring a heap sort, the number of rows necessary to trigger "
      "parallelTop() to sort.");
  opt_desc.add_options()(
      "enable-experimental-string-functions",
      po::value<bool>(&config_->exec.enable_experimental_string_functions)
          ->default_value(config_->exec.enable_experimental_string_functions)
          ->implicit_value(true),
      "Enable experimental string functions.");
  opt_desc.add_options()(
      "enable-interoperability",
      po::value<bool>(&config_->exec.enable_interop)
          ->default_value(config_->exec.enable_interop)
          ->implicit_value(true),
      "Enable offloading of query portions to an external execution engine.");
  opt_desc.add_options()(
      "parallel-linearization-threshold",
      po::value<size_t>(&config_->exec.parallel_linearization_threshold)
          ->default_value(config_->exec.parallel_linearization_threshold),
      "Threshold for parallel varlen col linearization");
  opt_desc.add_options()(
      "enable-multifrag-results",
      po::value<bool>(&config_->exec.enable_multifrag_rs)
          ->default_value(config_->exec.enable_multifrag_rs)
          ->implicit_value(true),
      "Enable multi-fragment intermediate results to improve execution parallelism for "
      "queries with multiple execution steps");
  opt_desc.add_options()("gpu-block-size",
                         po::value<size_t>(&config_->exec.override_gpu_block_size)
                             ->default_value(config_->exec.override_gpu_block_size),
                         "Force the size of block to use on GPU.");
  opt_desc.add_options()("gpu-grid-size",
                         po::value<size_t>(&config_->exec.override_gpu_grid_size)
                             ->default_value(config_->exec.override_gpu_grid_size),
                         "Size of grid to use on GPU.");
  opt_desc.add_options()("cpu-only",
                         po::value<bool>(&config_->exec.cpu_only)
                             ->default_value(config_->exec.cpu_only)
                             ->implicit_value(true),
                         "Run on CPU only, even if GPUs are available.");
  opt_desc.add_options()("initialize-with-gpu-vendor",
                         po::value<std::string>(&config_->exec.initialize_with_gpu_vendor)
                             ->default_value(config_->exec.initialize_with_gpu_vendor),
                         "GPU vendor to use for Data Manager initialization. Valid "
                         "values are \"intel\" and \"nvidia\".");

  opt_desc.add_options()(
      "use-cost-model",
      po::value<bool>(&config_->exec.enable_cost_model)->default_value(false),
      "Use Cost Model for query execution when it is possible.");

  // opts.filter_pushdown
  opt_desc.add_options()("enable-filter-push-down",
                         po::value<bool>(&config_->opts.filter_pushdown.enable)
                             ->default_value(config_->opts.filter_pushdown.enable)
                             ->implicit_value(true),
                         "Enable filter push down through joins.");
  opt_desc.add_options()(
      "filter-push-down-low-frac",
      po::value<float>(&config_->opts.filter_pushdown.low_frac)
          ->default_value(config_->opts.filter_pushdown.low_frac)
          ->implicit_value(config_->opts.filter_pushdown.low_frac),
      "Lower threshold for selectivity of filters that are pushed down.");
  opt_desc.add_options()(
      "filter-push-down-high-frac",
      po::value<float>(&config_->opts.filter_pushdown.high_frac)
          ->default_value(config_->opts.filter_pushdown.high_frac)
          ->implicit_value(config_->opts.filter_pushdown.high_frac),
      "Higher threshold for selectivity of filters that are pushed down.");
  opt_desc.add_options()(
      "filter-push-down-passing-row-ubound",
      po::value<size_t>(&config_->opts.filter_pushdown.passing_row_ubound)
          ->default_value(config_->opts.filter_pushdown.passing_row_ubound)
          ->implicit_value(config_->opts.filter_pushdown.passing_row_ubound),
      "Upperbound on the number of rows that should pass the filter "
      "if the selectivity is less than "
      "the high fraction threshold.");

  // opts
  opt_desc.add_options()("from-table-reordering",
                         po::value<bool>(&config_->opts.from_table_reordering)
                             ->default_value(config_->opts.from_table_reordering)
                             ->implicit_value(true),
                         "Enable automatic table reordering in FROM clause.");
  opt_desc.add_options()("constrained-by-in-threshold",
                         po::value<size_t>(&config_->opts.constrained_by_in_threshold)
                             ->default_value(config_->opts.constrained_by_in_threshold),
                         "Threshold for constrained-by-in reqrite optimiation.");
  opt_desc.add_options()(
      "enable-left-join-filter-hoisting",
      po::value<bool>(&config_->opts.enable_left_join_filter_hoisting)
          ->default_value(config_->opts.enable_left_join_filter_hoisting)
          ->implicit_value(true),
      "Enable hoisting left hand side filters through left joins.");

  // rs
  opt_desc.add_options()("enable-columnar-output",
                         po::value<bool>(&config_->rs.enable_columnar_output)
                             ->default_value(config_->rs.enable_columnar_output)
                             ->implicit_value(true),
                         "Enable columnar output for intermediate/final query steps.");
  opt_desc.add_options()("optimize-row-init",
                         po::value<bool>(&config_->rs.optimize_row_initialization)
                             ->default_value(config_->rs.optimize_row_initialization)
                             ->implicit_value(true),
                         "Optimize row initialization.");
  opt_desc.add_options()("enable-direct-columnarization",
                         po::value<bool>(&config_->rs.enable_direct_columnarization)
                             ->default_value(config_->rs.enable_direct_columnarization)
                             ->implicit_value(true),
                         "Enables/disables a more optimized columnarization method "
                         "for intermediate steps in multi-step queries.");
  opt_desc.add_options()("enable-lazy-fetch",
                         po::value<bool>(&config_->rs.enable_lazy_fetch)
                             ->default_value(config_->rs.enable_lazy_fetch)
                             ->implicit_value(true),
                         "Enable lazy fetch columns in query results.");

  // mem.cpu
  opt_desc.add_options()("enable-tiered-cpu-mem",
                         po::value<bool>(&config_->mem.cpu.enable_tiered_cpu_mem)
                             ->default_value(config_->mem.cpu.enable_tiered_cpu_mem)
                             ->implicit_value(true),
                         "Enable additional tiers of CPU memory (PMEM, etc...)");
  opt_desc.add_options()("pmem-size",
                         po::value<size_t>(&config_->mem.cpu.pmem_size)
                             ->default_value(config_->mem.cpu.pmem_size),
                         "An amount of PMEM memory to use.");
  opt_desc.add_options()("cpu-buffer-mem-bytes",
                         po::value<size_t>(&config_->mem.cpu.max_size)
                             ->default_value(config_->mem.cpu.max_size),
                         "Size of memory reserved for CPU buffers, in bytes. Use all "
                         "available memory if 0.");
  opt_desc.add_options()(
      "min-cpu-slab-size",
      po::value<size_t>(&config_->mem.cpu.min_slab_size)
          ->default_value(config_->mem.cpu.min_slab_size),
      "Min slab size (size of memory allocations) for CPU buffer pool.");
  opt_desc.add_options()(
      "max-cpu-slab-size",
      po::value<size_t>(&config_->mem.cpu.max_slab_size)
          ->default_value(config_->mem.cpu.max_slab_size),
      "Max CPU buffer pool slab size (size of memory allocations). Note if "
      "there is not enough free memory to accomodate the target slab size, smaller "
      "slabs will be allocated, down to the minimum size specified by "
      "min-cpu-slab-size.");

  // mem.gpu
  opt_desc.add_options()(
      "min-output-projection-allocation-bytes",
      po::value<size_t>(&config_->mem.gpu.min_memory_allocation_size)
          ->default_value(config_->mem.gpu.min_memory_allocation_size),
      "Minimum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. If an allocation of this size cannot be "
      "obtained, the query will be retried with different execution parameters and/or "
      "on CPU (if allow-cpu-retry is enabled). Requires bump allocator.");
  opt_desc.add_options()(
      "max-output-projection-allocation-bytes",
      po::value<size_t>(&config_->mem.gpu.max_memory_allocation_size)
          ->default_value(config_->mem.gpu.max_memory_allocation_size),
      "Maximum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. Default is the maximum slab size (sizes "
      "greater than the maximum slab size have no affect). Requires bump allocator.");
  opt_desc.add_options()(
      "gpu-input-mem-limit",
      po::value<double>(&config_->mem.gpu.input_mem_limit_percent)
          ->default_value(config_->mem.gpu.input_mem_limit_percent)
          ->notifier(get_range_checker(0.01, 0.99, "gpu-input-mem-limit")),
      "Max part of GPU memory that can be used for input data. Must be in range [0.01, "
      "0.99].");
  opt_desc.add_options()(
      "res-gpu-mem",
      po::value<size_t>(&config_->mem.gpu.reserved_mem_bytes)
          ->default_value(config_->mem.gpu.reserved_mem_bytes),
      "Reduces GPU memory available to the OmniSci allocator by this amount. Used for "
      "compiled code cache and ancillary GPU functions and other processes that may also "
      "be using the GPU concurrent with OmniSciDB.");
  opt_desc.add_options()("gpu-buffer-mem-bytes",
                         po::value<size_t>(&config_->mem.gpu.max_size)
                             ->default_value(config_->mem.gpu.max_size),
                         "Size of memory reserved for GPU buffers, in bytes, per GPU. "
                         "Use all available memory if 0.");
  opt_desc.add_options()(
      "min-gpu-slab-size",
      po::value<size_t>(&config_->mem.gpu.min_slab_size)
          ->default_value(config_->mem.gpu.min_slab_size),
      "Min slab size (size of memory allocations) for GPU buffer pools.");
  opt_desc.add_options()(
      "max-gpu-slab-size",
      po::value<size_t>(&config_->mem.gpu.max_slab_size)
          ->default_value(config_->mem.gpu.max_slab_size),
      "Max GPU buffer pool slab size (size of memory allocations). Note if "
      "there is not enough free memory to accomodate the target slab size, smaller "
      "slabs will be allocated, down to the minimum size speified by "
      "min-gpu-slab-size.");

  // cache
  opt_desc.add_options()("use-estimator-result-cache",
                         po::value<bool>(&config_->cache.use_estimator_result_cache)
                             ->default_value(config_->cache.use_estimator_result_cache)
                             ->implicit_value(true),
                         "Use estimator result cache.");
  opt_desc.add_options()("enable-data-recycler",
                         po::value<bool>(&config_->cache.enable_data_recycler)
                             ->default_value(config_->cache.enable_data_recycler)
                             ->implicit_value(true),
                         "Use data recycler.");
  opt_desc.add_options()("use-hashtable-cache",
                         po::value<bool>(&config_->cache.use_hashtable_cache)
                             ->default_value(config_->cache.use_hashtable_cache)
                             ->implicit_value(true),
                         "Use hashtable cache.");
  opt_desc.add_options()("hashtable-cache-total-bytes",
                         po::value<size_t>(&config_->cache.hashtable_cache_total_bytes)
                             ->default_value(config_->cache.hashtable_cache_total_bytes),
                         "Size of total memory space for hashtable cache, in bytes.");
  opt_desc.add_options()(
      "max-cacheable-hashtable-size-bytes",
      po::value<size_t>(&config_->cache.max_cacheable_hashtable_size_bytes)
          ->default_value(config_->cache.max_cacheable_hashtable_size_bytes),
      "The maximum size of hashtable that is available to cache, in bytes");
  opt_desc.add_options()(
      "gpu-code-cache-eviction-percent",
      po::value<double>(&config_->cache.gpu_fraction_code_cache_to_evict)
          ->default_value(config_->cache.gpu_fraction_code_cache_to_evict),
      "Percentage of the GPU code cache to evict if an out of memory error is "
      "encountered while attempting to place generated code on the GPU.");
  opt_desc.add_options()("dag-cache-size",
                         po::value<size_t>(&config_->cache.dag_cache_size)
                             ->default_value(config_->cache.dag_cache_size),
                         "Maximum number of nodes in DAG cache");
  opt_desc.add_options()("code-cache-size",
                         po::value<size_t>(&config_->cache.code_cache_size)
                             ->default_value(config_->cache.code_cache_size),
                         "Maximum number of entries in a code cache");

  // debug
  opt_desc.add_options()("build-rel-alg-cache",
                         po::value<std::string>(&config_->debug.build_ra_cache)
                             ->default_value(config_->debug.build_ra_cache),
                         "Used in tests to store all parsed SQL queries in a cache and "
                         "write them to the specified file when program finishes.");
  opt_desc.add_options()("use-rel-alg-cache",
                         po::value<std::string>(&config_->debug.use_ra_cache)
                             ->default_value(config_->debug.use_ra_cache),
                         "Used in tests to load pre-generated cache of parsed SQL "
                         "queries from the specified file to avoid Calcite usage.");
  opt_desc.add_options()("enable-automatic-ir-metadata",
                         po::value<bool>(&config_->debug.enable_automatic_ir_metadata)
                             ->default_value(config_->debug.enable_automatic_ir_metadata)
                             ->implicit_value(true),
                         "Enable automatic IR metadata (debug builds only).");
  opt_desc.add_options()(
      "enable-gpu-code-compilation-cache",
      po::value<bool>(&config_->debug.enable_gpu_code_compilation_cache)
          ->default_value(config_->debug.enable_gpu_code_compilation_cache)
          ->implicit_value(true),
      "Enable GPU compilation code caching.");

  // storage
  opt_desc.add_options()(
      "enable-lazy-dict-materialization",
      po::value<bool>(&config_->storage.enable_lazy_dict_materialization)
          ->default_value(config_->storage.enable_lazy_dict_materialization)
          ->implicit_value(true),
      "Enable lazy materialization of string dictionary columns from Arrow Storage.");
  opt_desc.add_options()(
      "enable-non-lazy-data-import",
      po::value<bool>(&config_->storage.enable_non_lazy_data_import)
          ->default_value(config_->storage.enable_non_lazy_data_import)
          ->implicit_value(true),
      "Enable non-lazy data import in Arrow Storage. When enabled, we do as much data "
      "processing on import as we might require. This might increase overall execution "
      "time. This option can be used to split data import and execution for performance "
      "measurements.");

  if (allow_gtest_flags) {
    opt_desc.add_options()("gtest_list_tests", "list all test");
    opt_desc.add_options()("gtest_filter", "filters tests, use --help for details");
  }

  // Right now we setup logging independently. Until it's fixed, simply ignore logger
  // options here.
  logger::LogOptions log_opts("dummy_opts");
  log_opts.set_options();

  po::options_description all_opts;
  all_opts.add(opt_desc).add(log_opts.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(all_opts).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << all_opts << std::endl;
    return true;
  }

  return false;
}

bool ConfigBuilder::parseCommandLineArgs(const std::string& app_name,
                                         const std::string& cmd_args,
                                         bool allow_gtest_flags) {
  std::vector<std::string> args;
  if (!cmd_args.empty()) {
    args = po::split_unix(cmd_args);
  }

  // Generate command line to  CommandLineOptions for DBHandler
  std::vector<const char*> argv;
  argv.push_back(app_name.c_str());
  for (auto& arg : args) {
    argv.push_back(arg.c_str());
  }
  return parseCommandLineArgs(
      static_cast<int>(argv.size()), argv.data(), allow_gtest_flags);
}

ConfigPtr ConfigBuilder::config() {
  return config_;
}
