#include "Config.h"

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

void Config::setOptions() {
  options_ = std::make_unique<boost::program_options::options_description>("");

  // exec.watchdog
  options_->add_options()("enable-watchdog",
                          po::value<bool>(&exec.watchdog.enable)
                              ->default_value(exec.watchdog.enable)
                              ->implicit_value(true),
                          "Enable watchdog.");
  options_->add_options()("enable-dynamic-watchdog",
                          po::value<bool>(&exec.watchdog.enable_dynamic)
                              ->default_value(exec.watchdog.enable_dynamic)
                              ->implicit_value(true),
                          "Enable dynamic watchdog.");
  options_->add_options()("dynamic-watchdog-time-limit",
                          po::value<size_t>(&exec.watchdog.time_limit)
                              ->default_value(exec.watchdog.time_limit),
                          "Dynamic watchdog time limit, in milliseconds.");
  options_->add_options()("watchdog-baseline-max-groups",
                          po::value<size_t>(&exec.watchdog.baseline_max_groups)
                              ->default_value(exec.watchdog.baseline_max_groups),
                          "Watchdog baseline aggregation groups limit.");
  options_->add_options()(
      "parallel-top-max",
      po::value<size_t>(&exec.watchdog.parallel_top_max)
          ->default_value(exec.watchdog.parallel_top_max),
      "For ResultSets requiring a heap sort, the maximum number of rows allowed by "
      "watchdog.");

  // exec.sub_tasks
  options_->add_options()(
      "enable-cpu-sub-tasks",
      po::value<bool>(&exec.sub_tasks.enable)
          ->default_value(exec.sub_tasks.enable)
          ->implicit_value(true),
      "Enable parallel processing of a single data fragment on CPU. This can improve CPU "
      "load balance and decrease reduction overhead.");
  options_->add_options()("cpu-sub-task-size",
                          po::value<size_t>(&exec.sub_tasks.sub_task_size)
                              ->default_value(exec.sub_tasks.sub_task_size),
                          "Set CPU sub-task size in rows.");

  // exec.join
  options_->add_options()("enable-loop-join",
                          po::value<bool>(&exec.join.allow_loop_joins)
                              ->default_value(exec.join.allow_loop_joins)
                              ->implicit_value(true),
                          "Enable/disable loop-based join execution.");
  options_->add_options()(
      "loop-join-limit",
      po::value<unsigned>(&exec.join.trivial_loop_join_threshold)
          ->default_value(exec.join.trivial_loop_join_threshold),
      "Maximum number of rows in an inner table allowed for loop join.");
  options_->add_options()("inner-join-fragment-skipping",
                          po::value<bool>(&exec.join.inner_join_fragment_skipping)
                              ->default_value(exec.join.inner_join_fragment_skipping)
                              ->implicit_value(true),
                          "Enable/disable inner join fragment skipping. This feature is "
                          "considered stable and is enabled by default. This "
                          "parameter will be removed in a future release.");
  options_->add_options()("huge-join-hash-threshold",
                          po::value<size_t>(&exec.join.huge_join_hash_threshold)
                              ->default_value(exec.join.huge_join_hash_threshold),
                          "Number of etries in a pefect join hash table to make it "
                          "considered as a huge one.");
  options_->add_options()(
      "huge-join-hash-min-load",
      po::value<size_t>(&exec.join.huge_join_hash_min_load)
          ->default_value(exec.join.huge_join_hash_min_load),
      "A minimal predicted load level for huge perfect hash tables in percent.");

  // exec.group_by
  options_->add_options()("bigint-count",
                          po::value<bool>(&exec.group_by.bigint_count)
                              ->default_value(exec.group_by.bigint_count)
                              ->implicit_value(true),
                          "Use 64-bit count.");
  options_->add_options()(
      "default-max-groups-buffer-entry-guess",
      po::value<size_t>(&exec.group_by.default_max_groups_buffer_entry_guess)
          ->default_value(exec.group_by.default_max_groups_buffer_entry_guess),
      "Default guess for group-by buffer size.");
  options_->add_options()(
      "big-group-threshold",
      po::value<size_t>(&exec.group_by.big_group_threshold)
          ->default_value(exec.group_by.big_group_threshold),
      "Threshold at which guessed group-by buffer size causes NDV estimator to be used.");
  options_->add_options()("use-groupby-buffer-desc",
                          po::value<bool>(&exec.group_by.use_groupby_buffer_desc)
                              ->default_value(exec.group_by.use_groupby_buffer_desc)
                              ->implicit_value(true),
                          "Use GroupBy Buffer Descriptor for hash tables.");
  options_->add_options()("enable-gpu-shared-mem-group-by",
                          po::value<bool>(&exec.group_by.enable_gpu_smem_group_by)
                              ->default_value(exec.group_by.enable_gpu_smem_group_by)
                              ->implicit_value(true),
                          "Enable using GPU shared memory for some GROUP BY queries.");
  options_->add_options()(
      "enable-gpu-shared-mem-non-grouped-agg",
      po::value<bool>(&exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->default_value(exec.group_by.enable_gpu_smem_non_grouped_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for non-grouped aggregate queries.");
  options_->add_options()(
      "enable-gpu-shared-mem-grouped-non-count-agg",
      po::value<bool>(&exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->default_value(exec.group_by.enable_gpu_smem_grouped_non_count_agg)
          ->implicit_value(true),
      "Enable using GPU shared memory for grouped non-count aggregate queries.");
  options_->add_options()(
      "gpu-shared-mem-threshold",
      po::value<size_t>(&exec.group_by.gpu_smem_threshold)
          ->default_value(exec.group_by.gpu_smem_threshold),
      "GPU shared memory threshold (in bytes). If query requires larger buffers than "
      "this threshold, we disable those optimizations. 0 means no static cap.");
  options_->add_options()(
      "hll-precision-bits",
      po::value<unsigned>(&exec.group_by.hll_precision_bits)
          ->default_value(exec.group_by.hll_precision_bits)
          ->notifier(get_range_checker(1U, 16U, "hll-precision-bits")),
      "Number of bits in range [1, 16] used from the hash value used "
      "to specify the bucket number.");
  options_->add_options()(
      "groupby-baseline-threshold",
      po::value<size_t>(&exec.group_by.baseline_threshold)
          ->default_value(exec.group_by.baseline_threshold),
      "Prefer baseline hash if number of entries exceeds this threshold.");

  // exec.window
  options_->add_options()("enable-window-functions",
                          po::value<bool>(&exec.window_func.enable)
                              ->default_value(exec.window_func.enable)
                              ->implicit_value(true),
                          "Enable experimental window function support.");
  options_->add_options()(
      "enable-parallel-window-partition-compute",
      po::value<bool>(&exec.window_func.parallel_window_partition_compute)
          ->default_value(exec.window_func.parallel_window_partition_compute)
          ->implicit_value(true),
      "Enable parallel window function partition computation.");
  options_->add_options()(
      "parallel-window-partition-compute-threshold",
      po::value<size_t>(&exec.window_func.parallel_window_partition_compute_threshold)
          ->default_value(exec.window_func.parallel_window_partition_compute_threshold),
      "Parallel window function partition computation threshold (in rows).");
  options_->add_options()(
      "enable-parallel-window-partition-sort",
      po::value<bool>(&exec.window_func.parallel_window_partition_sort)
          ->default_value(exec.window_func.parallel_window_partition_sort)
          ->implicit_value(true),
      "Enable parallel window function partition sorting.");
  options_->add_options()(
      "parallel-window-partition-sort-threshold",
      po::value<size_t>(&exec.window_func.parallel_window_partition_sort_threshold)
          ->default_value(exec.window_func.parallel_window_partition_sort_threshold),
      "Parallel window function partition sorting threshold (in rows).");

  // exec.heterogeneous
  options_->add_options()(
      "enable-heterogeneous",
      po::value<bool>(&exec.heterogeneous.enable_heterogeneous_execution)
          ->default_value(exec.heterogeneous.enable_heterogeneous_execution)
          ->implicit_value(true),
      "Allow the engine to schedule kernels heterogeneously.");
  options_->add_options()(
      "enable-multifrag-heterogeneous",
      po::value<bool>(&exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->default_value(exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->implicit_value(true),
      "Allow mutifragment heterogeneous kernels.");
  options_->add_options()(
      "force-heterogeneous-distribution",
      po::value<bool>(&exec.heterogeneous.forced_heterogeneous_distribution)
          ->default_value(exec.heterogeneous.forced_heterogeneous_distribution)
          ->implicit_value(true),
      "Keep user-defined load distribution in heterogeneous execution.");
  options_->add_options()("force-cpu-proportion",
                          po::value<unsigned>(&exec.heterogeneous.forced_cpu_proportion)
                              ->default_value(exec.heterogeneous.forced_cpu_proportion),
                          "Set CPU proportion for forced heterogeneous distribution.");
  options_->add_options()("force-gpu-proportion",
                          po::value<unsigned>(&exec.heterogeneous.forced_gpu_proportion)
                              ->default_value(exec.heterogeneous.forced_gpu_proportion),
                          "Set GPU proportion for forced heterogeneous distribution.");
  options_->add_options()("allow-cpu-retry",
                          po::value<bool>(&exec.heterogeneous.allow_cpu_retry)
                              ->default_value(exec.heterogeneous.allow_cpu_retry)
                              ->implicit_value(true),
                          "Allow the queries which failed on GPU to retry on CPU, even "
                          "when watchdog is enabled.");
  options_->add_options()(
      "allow-query-step-cpu-retry",
      po::value<bool>(&exec.heterogeneous.allow_query_step_cpu_retry)
          ->default_value(exec.heterogeneous.allow_query_step_cpu_retry)
          ->implicit_value(true),
      "Allow certain query steps to retry on CPU, even when allow-cpu-retry is disabled");

  // exec.interrupt
  options_->add_options()(
      "enable-runtime-query-interrupt",
      po::value<bool>(&exec.interrupt.enable_runtime_query_interrupt)
          ->default_value(exec.interrupt.enable_runtime_query_interrupt)
          ->implicit_value(true),
      "Enable runtime query interrupt.");
  options_->add_options()(
      "enable-non-kernel-time-query-interrupt",
      po::value<bool>(&exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->default_value(exec.interrupt.enable_non_kernel_time_query_interrupt)
          ->implicit_value(true),
      "Enable non-kernel time query interrupt.");
  options_->add_options()(
      "running-query-interrupt-freq",
      po::value<double>(&exec.interrupt.running_query_interrupt_freq)
          ->default_value(exec.interrupt.running_query_interrupt_freq),
      "A frequency of checking the request of running query "
      "interrupt from user (0.0 (less frequent) ~ (more frequent) 1.0).");

  // exec.codegen
  options_->add_options()(
      "null-div-by-zero",
      po::value<bool>(&exec.codegen.null_div_by_zero)
          ->default_value(exec.codegen.null_div_by_zero)
          ->implicit_value(true),
      "Return NULL on division by zero instead of throwing an exception.");
  options_->add_options()(
      "inf-div-by-zero",
      po::value<bool>(&exec.codegen.inf_div_by_zero)
          ->default_value(exec.codegen.inf_div_by_zero)
          ->implicit_value(true),
      "Return INF on fp division by zero instead of throwing an exception.");
  options_->add_options()("enable-hoist-literals",
                          po::value<bool>(&exec.codegen.hoist_literals)
                              ->default_value(exec.codegen.hoist_literals)
                              ->implicit_value(true),
                          "Enable literals hoisting during codegen to increase generated "
                          "code cache hit rate.");
  options_->add_options()(
      "enable-filter-function",
      po::value<bool>(&exec.codegen.enable_filter_function)
          ->default_value(exec.codegen.enable_filter_function)
          ->implicit_value(true),
      "Enable the filter function protection feature for the SQL JIT compiler. "
      "Normally should be on but techs might want to disable for troubleshooting.");

  // exec
  options_->add_options()(
      "streaming-top-n-max",
      po::value<size_t>(&exec.streaming_topn_max)->default_value(exec.streaming_topn_max),
      "The maximum number of rows allowing streaming top-N sorting.");
  options_->add_options()(
      "parallel-top-min",
      po::value<size_t>(&exec.parallel_top_min)->default_value(exec.parallel_top_min),
      "For ResultSets requiring a heap sort, the number of rows necessary to trigger "
      "parallelTop() to sort.");
  options_->add_options()("enable-experimental-string-functions",
                          po::value<bool>(&exec.enable_experimental_string_functions)
                              ->default_value(exec.enable_experimental_string_functions)
                              ->implicit_value(true),
                          "Enable experimental string functions.");
  options_->add_options()(
      "enable-interoperability",
      po::value<bool>(&exec.enable_interop)
          ->default_value(exec.enable_interop)
          ->implicit_value(true),
      "Enable offloading of query portions to an external execution engine.");
  options_->add_options()("parallel-linearization-threshold",
                          po::value<size_t>(&exec.parallel_linearization_threshold)
                              ->default_value(exec.parallel_linearization_threshold),
                          "Threshold for parallel varlen col linearization");
  options_->add_options()(
      "enable-multifrag-results",
      po::value<bool>(&exec.enable_multifrag_rs)
          ->default_value(exec.enable_multifrag_rs)
          ->implicit_value(true),
      "Enable multi-fragment intermediate results to improve execution parallelism for "
      "queries with multiple execution steps");
  options_->add_options()("gpu-block-size",
                          po::value<size_t>(&exec.override_gpu_block_size)
                              ->default_value(exec.override_gpu_block_size),
                          "Force the size of block to use on GPU.");
  options_->add_options()("gpu-grid-size",
                          po::value<size_t>(&exec.override_gpu_grid_size)
                              ->default_value(exec.override_gpu_grid_size),
                          "Size of grid to use on GPU.");
  options_->add_options()(
      "cpu-only",
      po::value<bool>(&exec.cpu_only)->default_value(exec.cpu_only)->implicit_value(true),
      "Run on CPU only, even if GPUs are available.");

  // opts.filter_pushdown
  options_->add_options()("enable-filter-push-down",
                          po::value<bool>(&opts.filter_pushdown.enable)
                              ->default_value(opts.filter_pushdown.enable)
                              ->implicit_value(true),
                          "Enable filter push down through joins.");
  options_->add_options()(
      "filter-push-down-low-frac",
      po::value<float>(&opts.filter_pushdown.low_frac)
          ->default_value(opts.filter_pushdown.low_frac)
          ->implicit_value(opts.filter_pushdown.low_frac),
      "Lower threshold for selectivity of filters that are pushed down.");
  options_->add_options()(
      "filter-push-down-high-frac",
      po::value<float>(&opts.filter_pushdown.high_frac)
          ->default_value(opts.filter_pushdown.high_frac)
          ->implicit_value(opts.filter_pushdown.high_frac),
      "Higher threshold for selectivity of filters that are pushed down.");
  options_->add_options()("filter-push-down-passing-row-ubound",
                          po::value<size_t>(&opts.filter_pushdown.passing_row_ubound)
                              ->default_value(opts.filter_pushdown.passing_row_ubound)
                              ->implicit_value(opts.filter_pushdown.passing_row_ubound),
                          "Upperbound on the number of rows that should pass the filter "
                          "if the selectivity is less than "
                          "the high fraction threshold.");

  // opts
  options_->add_options()("from-table-reordering",
                          po::value<bool>(&opts.from_table_reordering)
                              ->default_value(opts.from_table_reordering)
                              ->implicit_value(true),
                          "Enable automatic table reordering in FROM clause.");
  options_->add_options()("strip-join-covered-quals",
                          po::value<bool>(&opts.strip_join_covered_quals)
                              ->default_value(opts.strip_join_covered_quals)
                              ->implicit_value(true),
                          "Remove quals from the filtered count if they are covered by a "
                          "join condition (currently only ST_Contains).");
  options_->add_options()("constrained-by-in-threshold",
                          po::value<size_t>(&opts.constrained_by_in_threshold)
                              ->default_value(opts.constrained_by_in_threshold),
                          "Threshold for constrained-by-in reqrite optimiation.");
  options_->add_options()(
      "skip-intermediate-count",
      po::value<bool>(&opts.skip_intermediate_count)
          ->default_value(opts.skip_intermediate_count)
          ->implicit_value(true),
      "Skip pre-flight counts for intermediate projections with no filters.");
  options_->add_options()("enable-left-join-filter-hoisting",
                          po::value<bool>(&opts.enable_left_join_filter_hoisting)
                              ->default_value(opts.enable_left_join_filter_hoisting)
                              ->implicit_value(true),
                          "Enable hoisting left hand side filters through left joins.");

  // rs
  options_->add_options()("enable-columnar-output",
                          po::value<bool>(&rs.enable_columnar_output)
                              ->default_value(rs.enable_columnar_output)
                              ->implicit_value(true),
                          "Enable columnar output for intermediate/final query steps.");
  options_->add_options()("optimize-row-init",
                          po::value<bool>(&rs.optimize_row_initialization)
                              ->default_value(rs.optimize_row_initialization)
                              ->implicit_value(true),
                          "Optimize row initialization.");
  options_->add_options()("enable-direct-columnarization",
                          po::value<bool>(&rs.enable_direct_columnarization)
                              ->default_value(rs.enable_direct_columnarization)
                              ->implicit_value(true),
                          "Enables/disables a more optimized columnarization method "
                          "for intermediate steps in multi-step queries.");
  options_->add_options()("enable-lazy-fetch",
                          po::value<bool>(&rs.enable_lazy_fetch)
                              ->default_value(rs.enable_lazy_fetch)
                              ->implicit_value(true),
                          "Enable lazy fetch columns in query results.");

  // mem.cpu
  options_->add_options()("enable-tiered-cpu-mem",
                          po::value<bool>(&mem.cpu.enable_tiered_cpu_mem)
                              ->default_value(mem.cpu.enable_tiered_cpu_mem)
                              ->implicit_value(true),
                          "Enable additional tiers of CPU memory (PMEM, etc...)");
  options_->add_options()(
      "pmem-size",
      po::value<size_t>(&mem.cpu.pmem_size)->default_value(mem.cpu.pmem_size),
      "An amount of PMEM memory to use.");

  // mem.gpu
  options_->add_options()(
      "enable-bump-allocator",
      po::value<bool>(&mem.gpu.enable_bump_allocator)
          ->default_value(mem.gpu.enable_bump_allocator)
          ->implicit_value(true),
      "Enable the bump allocator for projection queries on GPU. The bump allocator will "
      "allocate a fixed size buffer for each query, track the number of rows passing the "
      "kernel during query execution, and copy back only the rows that passed the kernel "
      "to CPU after execution. When disabled, pre-flight count queries are used to size "
      "the output buffer for projection queries.");
  options_->add_options()(
      "min-output-projection-allocation-bytes",
      po::value<size_t>(&mem.gpu.min_memory_allocation_size)
          ->default_value(mem.gpu.min_memory_allocation_size),
      "Minimum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. If an allocation of this size cannot be "
      "obtained, the query will be retried with different execution parameters and/or "
      "on CPU (if allow-cpu-retry is enabled). Requires bump allocator.");
  options_->add_options()(
      "max-output-projection-allocation-bytes",
      po::value<size_t>(&mem.gpu.max_memory_allocation_size)
          ->default_value(mem.gpu.max_memory_allocation_size),
      "Maximum allocation size for a fixed output buffer allocation for projection "
      "queries with no pre-flight count. Default is the maximum slab size (sizes "
      "greater than the maximum slab size have no affect). Requires bump allocator.");
  options_->add_options()(
      "max-output-projection-allocation-bytes",
      po::value<double>(&mem.gpu.bump_allocator_step_reduction)
          ->default_value(mem.gpu.bump_allocator_step_reduction)
          ->notifier(
              get_range_checker(0.01, 0.99, "max-output-projection-allocation-bytes")),
      "Step for re-trying memory allocation of a fixed output buffer allocation for "
      "projection queries with no pre-flight count. Must be in range [0.01, 0.99].");
  options_->add_options()(
      "gpu-input-mem-limit",
      po::value<double>(&mem.gpu.input_mem_limit_percent)
          ->default_value(mem.gpu.input_mem_limit_percent)
          ->notifier(get_range_checker(0.01, 0.99, "gpu-input-mem-limit")),
      "Max part of GPU memory that can be used for input data. Must be in range [0.01, "
      "0.99].");
  options_->add_options()(
      "res-gpu-mem",
      po::value<size_t>(&mem.gpu.reserved_mem_bytes)
          ->default_value(mem.gpu.reserved_mem_bytes),
      "Reduces GPU memory available to the OmniSci allocator by this amount. Used for "
      "compiled code cache and ancillary GPU functions and other processes that may also "
      "be using the GPU concurrent with OmniSciDB.");

  // cache
  options_->add_options()("use-estimator-result-cache",
                          po::value<bool>(&cache.use_estimator_result_cache)
                              ->default_value(cache.use_estimator_result_cache)
                              ->implicit_value(true),
                          "Use estimator result cache.");
  options_->add_options()("enable-data-recycler",
                          po::value<bool>(&cache.enable_data_recycler)
                              ->default_value(cache.enable_data_recycler)
                              ->implicit_value(true),
                          "Use data recycler.");
  options_->add_options()("use-hashtable-cache",
                          po::value<bool>(&cache.use_hashtable_cache)
                              ->default_value(cache.use_hashtable_cache)
                              ->implicit_value(true),
                          "Use hashtable cache.");
  options_->add_options()("hashtable-cache-total-bytes",
                          po::value<size_t>(&cache.hashtable_cache_total_bytes)
                              ->default_value(cache.hashtable_cache_total_bytes),
                          "Size of total memory space for hashtable cache, in bytes.");
  options_->add_options()(
      "max-cacheable-hashtable-size-bytes",
      po::value<size_t>(&cache.max_cacheable_hashtable_size_bytes)
          ->default_value(cache.max_cacheable_hashtable_size_bytes),
      "The maximum size of hashtable that is available to cache, in bytes");
  options_->add_options()(
      "gpu-code-cache-eviction-percent",
      po::value<double>(&cache.gpu_fraction_code_cache_to_evict)
          ->default_value(cache.gpu_fraction_code_cache_to_evict),
      "Percentage of the GPU code cache to evict if an out of memory error is "
      "encountered while attempting to place generated code on the GPU.");
  options_->add_options()(
      "dag-cache-size",
      po::value<size_t>(&cache.dag_cache_size)->default_value(cache.dag_cache_size),
      "Maximum number of nodes in DAG cache");
  options_->add_options()(
      "code-cache-size",
      po::value<size_t>(&cache.code_cache_size)->default_value(cache.code_cache_size),
      "Maximum number of entries in a code cache");

  // debug
  options_->add_options()(
      "build-rel-alg-cache",
      po::value<std::string>(&debug.build_ra_cache)->default_value(debug.build_ra_cache),
      "Used in tests to store all parsed SQL queries in a cache and "
      "write them to the specified file when program finishes.");
  options_->add_options()(
      "use-rel-alg-cache",
      po::value<std::string>(&debug.use_ra_cache)->default_value(debug.use_ra_cache),
      "Used in tests to load pre-generated cache of parsed SQL "
      "queries from the specified file to avoid Calcite usage.");
  options_->add_options()("enable-automatic-ir-metadata",
                          po::value<bool>(&debug.enable_automatic_ir_metadata)
                              ->default_value(debug.enable_automatic_ir_metadata)
                              ->implicit_value(true),
                          "Enable automatic IR metadata (debug builds only).");
}

po::options_description const& Config::getOptions() const {
  return *options_;
}
