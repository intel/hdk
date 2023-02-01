#include <benchmark/benchmark.h>

#include "Tests/ArrowSQLRunner/ArrowSQLRunner.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <tbb/parallel_for.h>

#include <ittnotify.h>
__itt_domain* domain = __itt_domain_create("HDK.Taxi.Bench");
__itt_string_handle* handle_import_csv = __itt_string_handle_create("Import CSV");
__itt_string_handle* handle_import = __itt_string_handle_create("Import Parquet");
__itt_string_handle* handle_count = __itt_string_handle_create("Count");
__itt_string_handle* handle_q1 = __itt_string_handle_create("Q1");
__itt_string_handle* handle_q2 = __itt_string_handle_create("Q2");
__itt_string_handle* handle_q3 = __itt_string_handle_create("Q3");
__itt_string_handle* handle_q4 = __itt_string_handle_create("Q4");

boost::filesystem::path g_data_path;
size_t num_threads = 64;
size_t g_fragment_size = 160000000 / num_threads;
bool g_use_parquet{false};

using namespace TestHelpers::ArrowSQLRunner;

#define USE_HOT_DATA
#define PARALLEL_IMPORT_ENABLED

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
               {"payment_type", ctx().extDict(ctx().text(), 0)},
               {"trip_type", ctx().int8()},
               {"pickup", ctx().text()},
               {"dropoff", ctx().text()},
               {"cab_type", ctx().extDict(ctx().text(), 0)},  // grouped
               {"precipitation", ctx().fp64()},
               {"snow_depth", ctx().int16()},
               {"snowfall", ctx().fp64()},
               {"max_temperature", ctx().int16()},
               {"min_temperature", ctx().int16()},
               {"average_wind_speed", ctx().fp64()},
               {"pickup_nyct2010_gid", ctx().int64()},
               {"pickup_ctlabel", ctx().fp64()},
               {"pickup_borocode", ctx().int32()},
               {"pickup_boroname", ctx().text()},  // TODO: share dict
               {"pickup_ct2010", ctx().int32()},
               {"pickup_boroct2010", ctx().int32()},
               {"pickup_cdeligibil", ctx().text()},
               {"pickup_ntacode", ctx().text()},
               {"pickup_ntaname", ctx().text()},
               {"pickup_puma", ctx().int32()},
               {"dropoff_nyct2010_gid", ctx().int64()},
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

static void createTaxiTable() {
  if (g_use_parquet) {
    createTaxiTableParquet();
  } else {
    createTaxiTableCsv();
  }
}

static void populateTaxiTableCsv() {
  __itt_task_begin(domain, __itt_null, __itt_null, handle_import_csv);

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

  __itt_task_end(domain);
}

static void populateTaxiTableParquet() {
  __itt_task_begin(domain, __itt_null, __itt_null, handle_import);

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

  __itt_task_end(domain);
}

static void populateTaxiTable() {
  if (g_use_parquet) {
    populateTaxiTableParquet();
  } else {
    populateTaxiTableCsv();
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
    __itt_task_begin(domain, __itt_null, __itt_null, handle_count);
    auto res =
        v<int64_t>(run_simple_agg("select count(*) from trips", ExecutorDeviceType::CPU));
    std::cout << "Number of loaded tuples: " << res << std::endl;
    __itt_task_end(domain);
  }
}

static void taxi_q1(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    __itt_task_begin(domain, __itt_null, __itt_null, handle_q1);
    run_multiple_agg("select cab_type, count(*) from trips group by cab_type",
                     ExecutorDeviceType::CPU);
    __itt_task_end(domain);
  }
}

static void taxi_q2(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    __itt_task_begin(domain, __itt_null, __itt_null, handle_q2);
    run_multiple_agg(
        "SELECT passenger_count, avg(total_amount) FROM trips GROUP BY passenger_count",
        ExecutorDeviceType::CPU);
    __itt_task_end(domain);
  }
}

static void taxi_q3(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    __itt_task_begin(domain, __itt_null, __itt_null, handle_q3);
    run_multiple_agg(
        "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
        "count(*) FROM trips GROUP BY passenger_count, pickup_year",
        ExecutorDeviceType::CPU);
    __itt_task_end(domain);
  }
}

static void taxi_q4(benchmark::State& state) {
  for (auto _ : state) {
#ifndef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif

    __itt_task_begin(domain, __itt_null, __itt_null, handle_q4);
    run_multiple_agg(
        "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
        "cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP "
        "BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count "
        "desc",
        ExecutorDeviceType::CPU);
    __itt_task_end(domain);
  }
}

// functions as warmup for hot data
BENCHMARK(table_count)->Unit(benchmark::kMillisecond);
BENCHMARK(taxi_q1)->Unit(benchmark::kMillisecond);
BENCHMARK(taxi_q2)->Unit(benchmark::kMillisecond);
BENCHMARK(taxi_q3)->Unit(benchmark::kMillisecond);
BENCHMARK(taxi_q4)->Unit(benchmark::kMillisecond);

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

  try {
#ifdef USE_HOT_DATA
    createTaxiTable();
    populateTaxiTable();
#endif
    // warmup();
    ::benchmark::RunSpecifiedBenchmarks();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  reset();
}
