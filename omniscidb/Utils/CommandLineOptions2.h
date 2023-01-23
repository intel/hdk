#pragma once

#include <boost/program_options.hpp>
#include "Logger/Logger.h"
#include "Shared/Config.h"

class CommandLineOptions2 {
 public:
  CommandLineOptions2(char const* argv0);
  CommandLineOptions2(const CommandLineOptions2& other) = delete;
  CommandLineOptions2(CommandLineOptions2&& other) = delete;

  void parseCommandLineArgs(int argc, char const* const* argv);

  ConfigPtr config();

 private:
  boost::program_options::options_description const& fillConfigOptions();
  boost::program_options::options_description const& fillLogOptions(char const* argv0);
  boost::program_options::options_description const& fillGenericOptions();

  boost::program_options::options_description all_options_;
  ConfigPtr config_;
};