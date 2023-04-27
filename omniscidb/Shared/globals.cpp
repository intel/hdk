/*
 * Copyright 2021 OmniSci, Inc.
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

bool g_enable_table_functions{false};
unsigned g_pending_query_interrupt_freq{1000};
bool g_is_test_env{false};  // operating under a unit test environment. Currently only
                            // limits the allocation for the output buffer arena
                            // and data recycler test

size_t g_approx_quantile_buffer{1000};
size_t g_approx_quantile_centroids{300};

size_t g_max_log_length{500};
