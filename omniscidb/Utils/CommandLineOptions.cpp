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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <optional>

#include "CommandLineOptions.h"
#include "MapDRelease.h"
#include "OSDependent/omnisci_fs.h"
#include "StringDictionary/StringDictionary.h"

#ifdef _WIN32
#include <io.h>
#endif

bool g_use_table_device_offset;  // TODO(adb): where did this go?
extern bool g_cache_string_hash;
extern int64_t g_bitmap_memory_limit;
extern size_t g_approx_quantile_buffer;
extern size_t g_approx_quantile_centroids;
extern size_t g_estimator_failure_max_groupby_size;
extern bool g_columnar_large_projections;
extern size_t g_columnar_large_projections_threshold;

unsigned connect_timeout{20000};
unsigned recv_timeout{300000};
unsigned send_timeout{300000};
bool with_keepalive{false};

void CommandLineOptions::init_logging() {
  if (verbose_logging && logger::Severity::DEBUG1 < log_options_.severity_) {
    log_options_.severity_ = logger::Severity::DEBUG1;
  }
  validate_base_path();
  log_options_.set_base_path(base_path);
  logger::init(log_options_);
}

void CommandLineOptions::fillOptions() {
  help_desc.add_options()("help,h", "Show available options.");
  help_desc.add_options()("allow-loop-joins",
                          po::value<bool>(&allow_loop_joins)
                              ->default_value(allow_loop_joins)
                              ->implicit_value(true),
                          "Enable loop joins.");

  if (!dist_v5_) {
    help_desc.add_options()(
        "data",
        po::value<std::string>(&base_path)->required()->default_value("data"),
        "Directory path to OmniSci data storage (catalogs, raw data, log files, etc).");
    positional_options.add("data", 1);
  }
  help_desc.add_options()("db-query-list",
                          po::value<std::string>(&db_query_file),
                          "Path to file containing OmniSci warmup queries.");
  help_desc.add_options()(
      "exit-after-warmup",
      po::value<bool>(&exit_after_warmup)->default_value(false)->implicit_value(true),
      "Exit after OmniSci warmup queries.");
  help_desc.add_options()("enable-debug-timer",
                          po::value<bool>(&g_enable_debug_timer)
                              ->default_value(g_enable_debug_timer)
                              ->implicit_value(true),
                          "Enable debug timer logging.");
  help_desc.add_options()("enable-dynamic-watchdog",
                          po::value<bool>(&enable_dynamic_watchdog)
                              ->default_value(enable_dynamic_watchdog)
                              ->implicit_value(true),
                          "Enable dynamic watchdog.");
  help_desc.add_options()("pending-query-interrupt-freq",
                          po::value<unsigned>(&pending_query_interrupt_freq)
                              ->default_value(pending_query_interrupt_freq)
                              ->implicit_value(1000),
                          "A frequency of checking the request of pending query "
                          "interrupt from user (in millisecond).");

  if (!dist_v5_) {
    help_desc.add_options()(
        "enable-string-dict-hash-cache",
        po::value<bool>(&g_cache_string_hash)
            ->default_value(g_cache_string_hash)
            ->implicit_value(true),
        "Cache string hash values in the string dictionary server during import.");
  }
  help_desc.add_options()("enable-watchdog",
                          po::value<bool>(&enable_watchdog)
                              ->default_value(enable_watchdog)
                              ->implicit_value(true),
                          "Enable watchdog.");

  if (!dist_v5_) {
    help_desc.add_options()("http-port",
                            po::value<int>(&http_port)->default_value(http_port),
                            "HTTP port number.");
  }
  help_desc.add_options()(
      "idle-session-duration",
      po::value<int>(&idle_session_duration)->default_value(idle_session_duration),
      "Maximum duration of idle session.");
  help_desc.add_options()(
      "max-session-duration",
      po::value<int>(&max_session_duration)->default_value(max_session_duration),
      "Maximum duration of active session.");
  help_desc.add_options()(
      "num-reader-threads",
      po::value<size_t>(&num_reader_threads)->default_value(num_reader_threads),
      "Number of reader threads to use.");
  help_desc.add_options()(
      "read-only",
      po::value<bool>(&read_only)->default_value(read_only)->implicit_value(true),
      "Enable read-only mode.");

  help_desc.add_options()(
      "res-gpu-mem",
      po::value<size_t>(&reserved_gpu_mem)->default_value(reserved_gpu_mem),
      "Reduces GPU memory available to the OmniSci allocator by this amount. Used for "
      "compiled code cache and ancillary GPU functions and other processes that may also "
      "be using the GPU concurrent with OmniSciDB.");

  help_desc.add_options()("verbose",
                          po::value<bool>(&verbose_logging)
                              ->default_value(verbose_logging)
                              ->implicit_value(true),
                          "Write additional debug log messages to server logs.");
  help_desc.add_options()(
      "enable-runtime-udf",
      po::value<bool>(&enable_runtime_udf)
          ->default_value(enable_runtime_udf)
          ->implicit_value(true),
      "Enable runtime UDF registration by passing signatures and corresponding LLVM IR "
      "to the `register_runtime_udf` endpoint. For use with the Python Remote Backend "
      "Compiler server, packaged separately.");
  help_desc.add_options()("version,v", "Print Version Number.");

  help_desc.add_options()(
      "stringdict-parallelizm",
      po::value<bool>(&g_enable_stringdict_parallel)
          ->default_value(g_enable_stringdict_parallel)
          ->implicit_value(true),
      "Allow StringDictionary to parallelize loads using multiple threads");
  help_desc.add_options()("log-user-origin",
                          po::value<bool>(&log_user_origin)
                              ->default_value(log_user_origin)
                              ->implicit_value(true),
                          "Lookup the origin of inbound connections by IP address/DNS "
                          "name, and print this information as part of stdlog.");
  help_desc.add_options()(
      "allowed-import-paths",
      po::value<std::string>(&allowed_import_paths),
      "List of allowed root paths that can be used in import operations.");
  help_desc.add_options()(
      "allowed-export-paths",
      po::value<std::string>(&allowed_export_paths),
      "List of allowed root paths that can be used in export operations.");

  help_desc.add(log_options_.get_options());
}

void CommandLineOptions::fillAdvancedOptions() {
  developer_desc.add_options()("dev-options", "Print internal developer options.");
  developer_desc.add_options()("enable-legacy-syntax",
                               po::value<bool>(&enable_legacy_syntax)
                                   ->default_value(enable_legacy_syntax)
                                   ->implicit_value(true),
                               "Enable legacy syntax.");
  developer_desc.add_options()(
      "enable-multifrag",
      po::value<bool>(&allow_multifrag)
          ->default_value(allow_multifrag)
          ->implicit_value(true),
      "Enable execution over multiple fragments in a single round-trip to GPU.");
  developer_desc.add_options()(
      "offset-device-by-table-id",
      po::value<bool>(&g_use_table_device_offset)
          ->default_value(g_use_table_device_offset)
          ->implicit_value(true),
      "Enables/disables offseting the chosen device ID by the table ID for a given "
      "fragment. This improves balance of fragments across GPUs.");
  developer_desc.add_options()("enable-table-functions",
                               po::value<bool>(&g_enable_table_functions)
                                   ->default_value(g_enable_table_functions)
                                   ->implicit_value(true),
                               "Enable experimental table functions support.");
  developer_desc.add_options()(
      "jit-debug-ir",
      po::value<bool>(&jit_debug)->default_value(jit_debug)->implicit_value(true),
      "Enable runtime debugger support for the JIT. Note that this flag is "
      "incompatible "
      "with the `ENABLE_JIT_DEBUG` build flag. The generated code can be found at "
      "`/tmp/mapdquery`.");
  developer_desc.add_options()(
      "intel-jit-profile",
      po::value<bool>(&intel_jit_profile)
          ->default_value(intel_jit_profile)
          ->implicit_value(true),
      "Enable runtime support for the JIT code profiling using Intel VTune.");

  developer_desc.add_options()(
      "ssl-trust-ca-server",
      po::value<std::string>(&authMetadata.ca_file_name)->default_value(std::string("")),
      "SSL public CA certificates to validate TLS connection(as a server).");

  developer_desc.add_options()(
      "udf",
      po::value<std::string>(&udf_file_name),
      "Load user defined extension functions from this file at startup. The file is "
      "expected to be a C/C++ file with extension .cpp.");

  developer_desc.add_options()(
      "udf-compiler-path",
      po::value<std::string>(&udf_compiler_path),
      "Provide absolute path to clang++ used in udf compilation.");

  developer_desc.add_options()("udf-compiler-options",
                               po::value<std::vector<std::string>>(&udf_compiler_options),
                               "Specify compiler options to tailor udf compilation.");

  developer_desc.add_options()("approx_quantile_buffer",
                               po::value<size_t>(&g_approx_quantile_buffer)
                                   ->default_value(g_approx_quantile_buffer));
  developer_desc.add_options()("approx_quantile_centroids",
                               po::value<size_t>(&g_approx_quantile_centroids)
                                   ->default_value(g_approx_quantile_centroids));
  developer_desc.add_options()(
      "bitmap-memory-limit",
      po::value<int64_t>(&g_bitmap_memory_limit)->default_value(g_bitmap_memory_limit),
      "Limit for count distinct bitmap memory use. The limit is computed by taking the "
      "size of the group by buffer (entry count in Query Memory Descriptor) and "
      "multiplying it by the number of count distinct expression and the size of bitmap "
      "required for each. For approx_count_distinct this is typically 8192 bytes.");
  developer_desc.add_options()(
      "max-log-length",
      po::value<size_t>(&g_max_log_length)->default_value(g_max_log_length),
      "The maximum number of characters that a log message can has. If the log message "
      "is longer than this, we only record \'g_max_log_message_length\' characters.");
  developer_desc.add_options()(
      "estimator-failure-max-groupby-size",
      po::value<size_t>(&g_estimator_failure_max_groupby_size)
          ->default_value(g_estimator_failure_max_groupby_size),
      "Maximum size of the groupby buffer if the estimator fails. By default we use the "
      "number of tuples in the table up to this value.");
  developer_desc.add_options()("columnar-large-projections",
                               po::value<bool>(&g_columnar_large_projections)
                                   ->default_value(g_columnar_large_projections)
                                   ->implicit_value(true),
                               "Prefer columnar output if projection size is >= "
                               "threshold set by --columnar-large-projections-threshold "
                               "(default 1,000,000 rows).");
  developer_desc.add_options()(
      "columnar-large-projections-threshold",
      po::value<size_t>(&g_columnar_large_projections_threshold)
          ->default_value(g_columnar_large_projections_threshold),
      "Threshold (in minimum number of rows) to prefer columnar output for projections. "
      "Requires --columnar-large-projections to be set.");
}

namespace {

std::stringstream sanitize_config_file(std::ifstream& in) {
  // Strip the web section out of the config file so boost can validate program options
  std::stringstream ss;
  std::string line;
  while (std::getline(in, line)) {
    ss << line << "\n";
    if (line == "[web]") {
      break;
    }
  }
  return ss;
}

bool trim_and_check_file_exists(std::string& filename, const std::string desc) {
  if (!filename.empty()) {
    boost::algorithm::trim_if(filename, boost::is_any_of("\"'"));
    if (!boost::filesystem::exists(filename)) {
      std::cerr << desc << " " << filename << " does not exist." << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace

void CommandLineOptions::validate_base_path() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  if (!boost::filesystem::exists(base_path)) {
    throw std::runtime_error("OmniSci base directory does not exist at " + base_path);
  }
}

void CommandLineOptions::validate() {
  boost::algorithm::trim_if(base_path, boost::is_any_of("\"'"));
  const auto data_path = boost::filesystem::path(base_path) / "mapd_data";
  if (!boost::filesystem::exists(data_path)) {
    throw std::runtime_error("OmniSci data directory does not exist at '" + base_path +
                             "'");
  }

  boost::algorithm::trim_if(db_query_file, boost::is_any_of("\"'"));
  if (db_query_file.length() > 0 && !boost::filesystem::exists(db_query_file)) {
    throw std::runtime_error("File containing DB queries " + db_query_file +
                             " does not exist.");
  }
  if (license_path.length() == 0) {
    license_path = base_path + "/omnisci.license";
  }

  // add all parameters to be displayed on startup
  LOG(INFO) << "OmniSci started with data directory at '" << base_path << "'";
  if (vm.count("license-path")) {
    LOG(INFO) << "License key path set to '" << license_path << "'";
  }
  LOG(INFO) << " Server read-only mode is " << read_only;
  LOG(INFO) << " Watchdog is set to " << enable_watchdog;
  LOG(INFO) << " Dynamic Watchdog is set to " << enable_dynamic_watchdog;
  if (enable_dynamic_watchdog) {
    LOG(INFO) << " Dynamic Watchdog timeout is set to " << dynamic_watchdog_time_limit;
  }
  LOG(INFO) << " Runtime query interrupt is set to " << enable_runtime_query_interrupt;
  if (enable_runtime_query_interrupt) {
    LOG(INFO) << " A frequency of checking pending query interrupt request is set to "
              << pending_query_interrupt_freq << " (in ms.)";
    LOG(INFO) << " A frequency of checking running query interrupt request is set to "
              << running_query_interrupt_freq << " (0.0 ~ 1.0)";
  }
  LOG(INFO) << " Non-kernel time query interrupt is set to "
            << enable_non_kernel_time_query_interrupt;

  LOG(INFO) << " Debug Timer is set to " << g_enable_debug_timer;
  LOG(INFO) << " Maximum idle session duration " << idle_session_duration;
  LOG(INFO) << " Maximum active session duration " << max_session_duration;

  LOG(INFO) << "Allowed import paths is set to " << allowed_import_paths;
  LOG(INFO) << "Allowed export paths is set to " << allowed_export_paths;
}

std::optional<int> CommandLineOptions::parse_command_line(
    int argc,
    char const* const* argv,
    const bool should_init_logging) {
  po::options_description all_desc("All options");
  all_desc.add(help_desc).add(developer_desc);

  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(all_desc)
                  .positional(positional_options)
                  .run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cerr << "Usage: omnisci_server <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << help_desc << std::endl;
      return 0;
    }
    if (vm.count("dev-options")) {
      std::cout << "Usage: omnisci_server <data directory path> [-p <port number>] "
                   "[--http-port <http port number>] [--flush-log] [--version|-v]"
                << std::endl
                << std::endl;
      std::cout << developer_desc << std::endl;
      return 0;
    }
    if (vm.count("version")) {
      std::cout << "OmniSci Version: " << MAPD_RELEASE << std::endl;
      return 0;
    }

    if (should_init_logging) {
      init_logging();
    }

    if (!trim_and_check_file_exists(authMetadata.ca_file_name, "ca file name")) {
      return 1;
    }

    g_pending_query_interrupt_freq = pending_query_interrupt_freq;

  } catch (po::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    return 1;
  }

  if (vm.count("udf")) {
    boost::algorithm::trim_if(udf_file_name, boost::is_any_of("\"'"));

    if (!boost::filesystem::exists(udf_file_name)) {
      LOG(ERROR) << " User defined function file " << udf_file_name << " does not exist.";
      return 1;
    }

    LOG(INFO) << " User provided extension functions loaded from " << udf_file_name;
  }

  if (vm.count("udf-compiler-path")) {
    boost::algorithm::trim_if(udf_compiler_path, boost::is_any_of("\"'"));
  }

  auto trim_string = [](std::string& s) {
    boost::algorithm::trim_if(s, boost::is_any_of("\"'"));
  };

  if (vm.count("udf-compiler-options")) {
    std::for_each(udf_compiler_options.begin(), udf_compiler_options.end(), trim_string);
  }

  if (enable_runtime_udf) {
    LOG(INFO) << " Runtime user defined extension functions enabled globally.";
  }

  LOG(INFO) << " Allow Local Auth Fallback: "
            << (authMetadata.allowLocalAuthFallback ? "enabled" : "disabled");

  boost::algorithm::trim_if(authMetadata.distinguishedName, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.uri, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapQueryUrl, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapRoleRegex, boost::is_any_of("\"'"));
  boost::algorithm::trim_if(authMetadata.ldapSuperUserRole, boost::is_any_of("\"'"));

  return std::nullopt;
}
