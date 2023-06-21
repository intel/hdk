/**
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "Shared/funcannotations.h"

template <class T>
struct remove_addr_space {
  typedef T type;
};

#ifdef L0_RUNTIME_ENABLED
template <class T>
struct remove_addr_space<GENERIC_ADDR_SPACE T> {
  typedef T type;
};
#endif
