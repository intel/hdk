#include <benchmark/benchmark.h>

#include "Tests/ArrowSQLRunner/ArrowSQLRunner.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <tbb/parallel_for.h>

#include "Shared/measure.h"

boost::filesystem::path g_data_path;
size_t num_threads = 15;
size_t g_fragment_size = 140'000'000 / num_threads;
bool g_use_parquet{false};
ExecutorDeviceType g_device_type{ExecutorDeviceType::GPU};

EXTERN extern bool g_lazy_materialize_dictionaries;

using namespace TestHelpers::ArrowSQLRunner;

// #define USE_HOT_DATA
#define PARALLEL_IMPORT_ENABLED

// when we want to measure storage latencies, read the csv files before starting the
// benchmark
#ifndef USE_HOT_DATA
std::vector<std::shared_ptr<arrow::Table>> g_taxi_data_files;
#endif

std::istream& operator>>(std::istream& in, ExecutorDeviceType& device_type) {
  std::string token;
  in >> token;
  if (token == "CPU") {
    device_type = ExecutorDeviceType::CPU;
  } else if (token == "GPU") {
    device_type = ExecutorDeviceType::GPU;
  } else {
    throw std::runtime_error("Invalid device type: " + token);
  }
  return in;
}

static void createTaxiTableParquet() {
  getStorage()->dropTable("trips");
  ArrowStorage::TableOptions to{g_fragment_size};
  createTable("trips",
              {{"trip_id", ctx().fp64()},
               {"vendor_id", ctx().text()},
               {"pickup_datetime", ctx().timestamp(hdk::ir::TimeUnit::kMilli)},
               {"dropoff_datetime", ctx().timestamp(hdk::ir::TimeUnit::kMilli)},
               {"store_and_fwd_flag", ctx().text()},
               {"rate_code_id", ctx().int16()},
               {"pickup_longitude", ctx().fp64()},
               {"pickup_latitude", ctx().fp64()},
               {"dropoff_longitude", ctx().fp64()},
               {"dropoff_latitude", ctx().fp64()},
               {"passenger_count", ctx().int16()},
               {"trip_distance", ctx().fp64()},
               {"fare_amount", ctx().fp64()},
               {"extra", ctx().fp64()},
               {"mta_tax", ctx().fp64()},
               {"tip_amount", ctx().fp64()},
               {"tolls_amount", ctx().fp64()},
               {"ehail_fee", ctx().fp64()},
               {"improvement_surcharge", ctx().fp64()},
               {"total_amount", ctx().fp64()},
               {"payment_type", ctx().text()},
               {"trip_type", ctx().int16()},  // note: converted to int16 due to lack of
                                              // tinyint support in velox
               {"pickup", ctx().text()},
               {"dropoff", ctx().text()},
               {"cab_type", ctx().extDict(ctx().text(), 0)},  // grouped
               {"precipitation", ctx().fp64()},
               {"snow_depth", ctx().int16()},
               {"snowfall", ctx().fp64()},
               {"max_temperature", ctx().int16()},
               {"min_temperature", ctx().int16()},
               {"average_wind_speed", ctx().fp64()},
               {"pickup_ctlabel", ctx().fp64()},
               {"pickup_borocode", ctx().int32()},
               {"pickup_boroname", ctx().text()},  // TODO: share dict
               {"pickup_ct2010", ctx().int32()},
               {"pickup_boroct2010", ctx().int32()},
               {"pickup_cdeligibil", ctx().text()},
               {"pickup_ntacode", ctx().text()},
               {"pickup_ntaname", ctx().text()},
               {"pickup_puma", ctx().int32()},
               {"dropoff_ctlabel", ctx().fp64()},
               {"dropoff_borocode", ctx().int64()},
               {"dropoff_boroname", ctx().text()},
               {"dropoff_ct2010", ctx().int32()},
               {"dropoff_boroct2010", ctx().int32()},
               {"dropoff_cdeligibil", ctx().text()},
               {"dropoff_ntacode", ctx().text()},
               {"dropoff_ntaname", ctx().text()},
               {"dropoff_puma", ctx().int32()}},
              to);
}

static void createTaxiTableCsv() {
  getStorage()->dropTable("trips");
  ArrowStorage::TableOptions to{g_fragment_size};
  createTable("trips",
              {{"trip_id", ctx().fp64()},
               {"vendor_id", ctx().extDict(ctx().text(), 0)},
               {"pickup_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
               {"dropoff_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
               {"store_and_fwd_flag", ctx().extDict(ctx().text(), 0)},
               {"rate_code_id", ctx().int16()},
               {"pickup_longitude", ctx().fp64()},
               {"pickup_latitude", ctx().fp64()},
               {"dropoff_longitude", ctx().fp64()},
               {"dropoff_latitude", ctx().fp64()},
               {"passenger_count", ctx().int16()},
               {"trip_distance", ctx().fp64()},
               {"fare_amount", ctx().fp64()},
               {"extra", ctx().fp64()},
               {"mta_tax", ctx().fp64()},
               {"tip_amount", ctx().fp64()},
               {"tolls_amount", ctx().fp64()},
               {"ehail_fee", ctx().fp64()},
               {"improvement_surcharge", ctx().fp64()},
               {"total_amount", ctx().fp64()},
               {"payment_type", ctx().extDict(ctx().text(), 0)},
               {"trip_type", ctx().int8()},
               {"pickup", ctx().extDict(ctx().text(), 0)},
               {"dropoff", ctx().extDict(ctx().text(), 0)},
               {"cab_type", ctx().extDict(ctx().text(), 0, 4)},  // grouped
               {"precipitation", ctx().fp64()},
               {"snow_depth", ctx().int16()},
               {"snowfall", ctx().fp64()},
               {"max_temperature", ctx().int16()},
               {"min_temperature", ctx().int16()},
               {"average_wind_speed", ctx().fp64()},
               {"pickup_nyct2010_gid", ctx().int64()},
               {"pickup_ctlabel", ctx().fp64()},
               {"pickup_borocode", ctx().int32()},
               {"pickup_boroname", ctx().extDict(ctx().text(), 0)},  // TODO: share dict
               {"pickup_ct2010", ctx().int32()},
               {"pickup_boroct2010", ctx().int32()},
               {"pickup_cdeligibil", ctx().extDict(ctx().text(), 0)},
               {"pickup_ntacode", ctx().extDict(ctx().text(), 0)},
               {"pickup_ntaname", ctx().extDict(ctx().text(), 0)},
               {"pickup_puma", ctx().int32()},
               {"dropoff_nyct2010_gid", ctx().int64()},
               {"dropoff_ctlabel", ctx().fp64()},
               {"dropoff_borocode", ctx().int64()},
               {"dropoff_boroname", ctx().extDict(ctx().text(), 0)},
               {"dropoff_ct2010", ctx().int32()},
               {"dropoff_boroct2010", ctx().int32()},
               {"dropoff_cdeligibil", ctx().extDict(ctx().text(), 0)},
               {"dropoff_ntacode", ctx().extDict(ctx().text(), 0)},
               {"dropoff_ntaname", ctx().extDict(ctx().text(), 0)},
               {"dropoff_puma", ctx().int32()}},
              to);
}

static void createTaxiTable() {
  if (g_use_parquet) {
    createTaxiTableParquet();
  } else {
    createTaxiTableCsv();
  }
}

static std::vector<std::shared_ptr<arrow::Table>> readTaxiFilesCsv(
    const ColumnInfoList& col_infos) {
  std::vector<std::shared_ptr<arrow::Table>> taxi_arrow_data;
  auto time = measure<>::execution([&]() {
    namespace fs = boost::filesystem;
    ArrowStorage::CsvParseOptions po;
    po.header = false;
    if (fs::is_directory(g_data_path)) {
      std::vector<std::string> csv_files;
      for (auto it = fs::directory_iterator{g_data_path}; it != fs::directory_iterator{};
           it++) {
        if (fs::is_directory(it->path())) {
          continue;
        }
        csv_files.push_back(it->path().string());
      }
      tbb::task_arena arena(num_threads);
      taxi_arrow_data.resize(csv_files.size());
      arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, csv_files.size()),
                          [&](const tbb::blocked_range<size_t>& r) {
                            for (size_t i = r.begin(); i != r.end(); ++i) {
                              taxi_arrow_data[i] =
                                  getStorage()->parseCsvFile(csv_files[i], po, col_infos);
                            }
                          });
      });
    } else {
      taxi_arrow_data.push_back(
          getStorage()->parseCsvFile(g_data_path.string(), po, col_infos));
    }
  });

  std::cout << "Read taxi csv files in " << time << " ms" << std::endl;
  return taxi_arrow_data;
}

static void loadTaxiArrowData() {
  for (auto& data_file : g_taxi_data_files) {
    getStorage()->appendArrowTable(data_file, "trips");
  }
}

static void populateTaxiTableCsv() {
  namespace fs = boost::filesystem;
  ArrowStorage::CsvParseOptions po;
  po.header = false;
  if (fs::is_directory(g_data_path)) {
    std::vector<std::string> csv_files;
    for (auto it = fs::directory_iterator{g_data_path}; it != fs::directory_iterator{};
         it++) {
      if (fs::is_directory(it->path())) {
        continue;
      }
      csv_files.push_back(it->path().string());
    }
    tbb::task_arena arena(num_threads);
    arena.execute([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, csv_files.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            getStorage()->appendCsvFile(csv_files[i], "trips", po);
                          }
                        });
    });
  } else {
    getStorage()->appendCsvFile(g_data_path.string(), "trips", po);
  }
}

static void populateTaxiTableParquet() {
  namespace fs = boost::filesystem;
  ArrowStorage::CsvParseOptions po;
  po.header = false;
  if (fs::is_directory(g_data_path)) {
    std::vector<std::string> files;
    for (auto it = fs::directory_iterator{g_data_path}; it != fs::directory_iterator{};
         it++) {
      if (fs::is_directory(it->path())) {
        continue;
      }
      files.push_back(it->path().string());
    }
#ifndef PARALLEL_IMPORT_ENABLED
    for (auto& file : files) {
      getStorage()->appendParquetFile(file, "trips");
    }
#else
    tbb::task_arena arena(num_threads);
    arena.execute([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, files.size()),
                        [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            getStorage()->appendParquetFile(files[i], "trips");
                          }
                        });
    });
#endif
  } else {
    getStorage()->appendParquetFile(g_data_path.string(), "trips");
  }
}

static void populateTaxiTable() {
  if (g_use_parquet) {
    populateTaxiTableParquet();
  } else {
    if (!g_taxi_data_files.empty()) {
      loadTaxiArrowData();
    } else {
      populateTaxiTableCsv();
    }
  }
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  auto p = boost::get<T>(scalar_r);
  return *p;
}

static void table_count(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    auto res = v<int64_t>(run_simple_agg("select count(*) from trips", g_device_type));
    std::cout << "Number of loaded tuples: " << res << std::endl;
  }
}

static void taxi_q1(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    run_multiple_agg("select cab_type, count(*) from trips group by cab_type",
                     g_device_type);
  }
}

static void taxi_q2(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    run_multiple_agg(
        "SELECT passenger_count, avg(total_amount) FROM trips GROUP BY passenger_count",
        g_device_type);
  }
}

static void taxi_q3(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    run_multiple_agg(
        "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
        "count(*) FROM trips GROUP BY passenger_count, pickup_year",
        g_device_type);
  }
}

static void taxi_q4(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    run_multiple_agg(
        "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
        "cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP "
        "BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count "
        "desc",
        g_device_type);
  }
}

// functions as warmup for hot data
BENCHMARK(table_count)->Unit(benchmark::kMillisecond)->MinTime(10.0);
BENCHMARK(taxi_q1)->Unit(benchmark::kMillisecond)->MinTime(10.0);
BENCHMARK(taxi_q2)->Unit(benchmark::kMillisecond)->MinTime(10.0);
BENCHMARK(taxi_q3)->Unit(benchmark::kMillisecond)->MinTime(10.0);
BENCHMARK(taxi_q4)->Unit(benchmark::kMillisecond)->MinTime(10.0);

int main(int argc, char* argv[]) {
  ::benchmark::Initialize(&argc, argv);

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  auto config = std::make_shared<Config>();

  po::options_description desc("Options");
  desc.add_options()(
      "enable-heterogeneous",
      po::value<bool>(&config->exec.heterogeneous.enable_heterogeneous_execution)
          ->default_value(config->exec.heterogeneous.enable_heterogeneous_execution)
          ->implicit_value(true),
      "Allow heterogeneous execution.");
  desc.add_options()(
      "enable-multifrag",
      po::value<bool>(
          &config->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->default_value(
              config->exec.heterogeneous.enable_multifrag_heterogeneous_execution)
          ->implicit_value(true),
      "Allow multifrag heterogeneous execution.");
  desc.add_options()("data", po::value<fs::path>(&g_data_path), "Path to taxi dataset.");
  desc.add_options()("fragment-size",
                     po::value<size_t>(&g_fragment_size)->default_value(g_fragment_size),
                     "Table fragment size.");
  desc.add_options()(
      "use-parquet",
      po::value<bool>(&g_use_parquet)->default_value(g_use_parquet)->implicit_value(true),
      "Use parquet for input file format.");
  desc.add_options()(
      "allow-query-step-cpu-retry",
      po::value<bool>(&config->exec.heterogeneous.allow_query_step_cpu_retry)
          ->default_value(config->exec.heterogeneous.allow_query_step_cpu_retry)
          ->implicit_value(false),
      "Allow certain query steps to retry on CPU, even when allow-cpu-retry is disabled");
  desc.add_options()("allow-cpu-retry",
                     po::value<bool>(&config->exec.heterogeneous.allow_cpu_retry)
                         ->default_value(config->exec.heterogeneous.allow_cpu_retry)
                         ->implicit_value(false),
                     "Allow the queries which failed on GPU to retry on CPU, even "
                     "when watchdog is enabled.");
  desc.add_options()("device",
                     po::value<ExecutorDeviceType>(&g_device_type)
                         ->implicit_value(ExecutorDeviceType::GPU)
                         ->default_value(ExecutorDeviceType::CPU),
                     "Device type to use.");

  desc.add_options()("use-lazy-materialization",
                     po::value<bool>(&g_lazy_materialize_dictionaries)
                         ->implicit_value(true)
                         ->default_value(g_lazy_materialize_dictionaries));

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage:" << std::endl << desc << std::endl;
  }

  logger::init(log_options);
  init(config);

  if (g_lazy_materialize_dictionaries) {
    std::cout << "Using lazy materialization!" << std::endl;
  }

  try {
#ifdef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#else
    if (g_use_parquet) {
      throw std::runtime_error("Cannot use parquet files in cold data mode yet.");
    }
    createTaxiTable();
    auto table_info = getStorage()->getTableInfo(getStorage()->dbId(), "trips");
    if (!table_info) {
      throw std::runtime_error("Cannot find table \"trips\", creation failed?");
    }

    auto col_infos = getStorage()->listColumns(table_info->db_id, table_info->table_id);
    g_taxi_data_files = readTaxiFilesCsv(col_infos);
#endif
    // warmup();
    ::benchmark::RunSpecifiedBenchmarks();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  reset();
}
