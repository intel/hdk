/**
 * Copyright (C) 2022 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>

constexpr std::array<char const*, dtINVALID> datetrunc_fname_lookup{
    {"datetrunc_year",
     "datetrunc_quarter",
     "datetrunc_month",
     "datetrunc_day",
     "datetrunc_hour",
     "datetrunc_minute",
     "datetrunc_second",       // not used
     "datetrunc_millisecond",  // not used
     "datetrunc_microsecond",  // not used
     "datetrunc_nanosecond",   // not used
     "datetrunc_millennium",
     "datetrunc_century",
     "datetrunc_decade",
     "datetrunc_week_monday",
     "datetrunc_week_sunday",
     "datetrunc_week_saturday",
     "datetrunc_quarterday"}};

// Arithmetic which relies on these enums being consecutive is used elsewhere.
static_assert(dtSECOND + 1 == dtMILLISECOND, "Please keep these consecutive.");
static_assert(dtMILLISECOND + 1 == dtMICROSECOND, "Please keep these consecutive.");
static_assert(dtMICROSECOND + 1 == dtNANOSECOND, "Please keep these consecutive.");
