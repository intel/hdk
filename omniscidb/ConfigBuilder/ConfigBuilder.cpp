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
#include "Shared/ConfigOptions.h"

#include "Logger/Logger.h"

#include <boost/crc.hpp>
#include <boost/program_options.hpp>

#include <iostream>

namespace po = boost::program_options;

ConfigBuilder::ConfigBuilder() {
  config_ = std::make_shared<Config>();
}

ConfigBuilder::ConfigBuilder(ConfigPtr config) : config_(config) {}

bool ConfigBuilder::parseCommandLineArgs(int argc,
                                         char const* const* argv,
                                         bool allow_gtest_flags) {
  po::options_description opt_desc =
      get_config_builder_options(allow_gtest_flags, config_);
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(opt_desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opt_desc << std::endl;
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
