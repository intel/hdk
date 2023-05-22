/*
 * Copyright 2023 Intel Corporation.
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

#include "TestHelpers.h"

#include <benchmark/benchmark.h>

#include <random>
#include <string>
#include <vector>

#include "ArrowStorage/ArrowStorage.h"
#include "Shared/Config.h"

std::shared_ptr<Config> g_config;

std::string generate_random_str(std::mt19937& generator, const int64_t str_len) {
  constexpr char alphanum_lookup_table[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  constexpr size_t char_mod = sizeof(alphanum_lookup_table) - 1;
  std::uniform_int_distribution<int32_t> rand_distribution(0, char_mod);

  std::string tmp_s;
  tmp_s.reserve(str_len);
  for (int i = 0; i < str_len; ++i) {
    tmp_s += alphanum_lookup_table[rand_distribution(generator)];
  }
  return tmp_s;
}

std::vector<std::string> generate_random_strs(const size_t num_unique_strings,
                                              const size_t str_len,
                                              const uint64_t seed) {
  std::mt19937 rand_generator(seed);
  std::vector<std::string> unique_strings(num_unique_strings);
  for (size_t string_idx = 0; string_idx < num_unique_strings; ++string_idx) {
    unique_strings[string_idx] = generate_random_str(rand_generator, str_len);
  }
  return unique_strings;
}

std::string genCsv(size_t rows, const uint64_t seed = 42) {
  auto strings =
      generate_random_strs(/*num_unique_strs=*/rows * 0.05, /*str_len=*/15, seed);
  CHECK_GE(strings.size(), size_t(0));

  std::mt19937 gen(seed);
  std::uniform_int_distribution<> distrib(0, strings.size() - 1);
  std::string csv{""};
  for (size_t i = 0; i < rows; i++) {
    csv += std::to_string(i) + "," + strings[distrib(gen)] + "\n";
  }

  return csv;
}

static void import_csv(benchmark::State& state) {
  ArrowStorage storage(/*schema_id=*/123, /*schema_name=*/"test", /*db_id=*/1, g_config);

  // generate some data
  auto csv = genCsv(10'000'000);

  ArrowStorage::CsvParseOptions parse_options;

  for (auto _ : state) {
    auto csv_table = storage.parseCsvData(csv, parse_options);
    storage.importArrowTable(csv_table, "test_table");
    storage.dropTable("test_table");
  }
}

static void dictionary_materialize(benchmark::State& state) {
  ArrowStorage storage(/*schema_id=*/123, /*schema_name=*/"test", /*db_id=*/1, g_config);

  // generate some data
  auto csv = genCsv(10'000'000);
  ArrowStorage::CsvParseOptions parse_options;
  auto csv_table = storage.parseCsvData(csv, parse_options);

  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are resumed.
    storage.dropTable("test_table", /*throw_if_not_exist=*/false);
    const auto table_info = storage.importArrowTable(csv_table, "test_table");
    CHECK(table_info);
    const auto db_id = table_info->db_id;
    const auto table_id = table_info->table_id;
    const auto col_infos = storage.listColumns(db_id, table_id);
    CHECK_EQ(col_infos.size(), size_t(3));  // int col, str col, row_id
    const auto& dict_col_info_ptr = col_infos[1];
    CHECK(dict_col_info_ptr);
    const auto dict_id =
        dict_col_info_ptr->type->as<hdk::ir::ExtDictionaryType>()->dictId();
    state.ResumeTiming();  // And resume timers. They are now counting again.
    storage.getDictMetadata(dict_id, /*load_dict=*/true);
  }
}

BENCHMARK(import_csv)->Unit(benchmark::kMillisecond)->Iterations(10);
BENCHMARK(dictionary_materialize)->Unit(benchmark::kMillisecond)->Iterations(25);

int main(int argc, char* argv[]) {
  ::benchmark::Initialize(&argc, argv);

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  g_config = std::make_shared<Config>();

  po::options_description desc("Options");

  desc.add_options()(
      "use-lazy-materialization",
      po::value<bool>(&g_config->storage.enable_lazy_dict_materialization)
          ->implicit_value(true)
          ->default_value(g_config->storage.enable_lazy_dict_materialization));

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.severity_clog_ = logger::Severity::WARNING;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage:" << std::endl << desc << std::endl;
  }

  logger::init(log_options);
  if (g_config->storage.enable_lazy_dict_materialization) {
    LOG(WARNING) << "Using lazy materialization.";
  }

  try {
    ::benchmark::RunSpecifiedBenchmarks();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  return 0;
}