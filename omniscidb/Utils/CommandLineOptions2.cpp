#include "CommandLineOptions2.h"

CommandLineOptions2::CommandLineOptions2(char const* argv0) {
  auto generic_desc = fillGenericOptions();
  auto config_desc = fillConfigOptions();
  auto logger_desc = fillLogOptions(argv0);

  all_options_.add(generic_desc).add(config_desc).add(logger_desc);
}

namespace po = boost::program_options;

po::options_description const& CommandLineOptions2::fillGenericOptions() {
  po::options_description generic_options("Generic options");
  generic_options.add_options()("help,h", "Print help messages");
  return generic_options;
}

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

po::options_description const& CommandLineOptions2::fillConfigOptions() {
  config_ = std::make_shared<Config>();
  config_->setOptions();
  return config_->getOptions();
}

po::options_description const& CommandLineOptions2::fillLogOptions(char const* argv0) {
  logger::LogOptions log_options(argv0);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();

  return log_options.get_options();
}
