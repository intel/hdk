/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include "Shared/Config.h"

namespace po = boost::program_options;

template <typename T>
auto get_option_range_checker(T min, T max, const char* opt);

po::options_description get_config_builder_options(bool allow_gtest_flags,
                                                   ConfigPtr config_);