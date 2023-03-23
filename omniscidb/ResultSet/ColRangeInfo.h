/*
 * Copyright (C) 2023 Intel Corporation
 * Copyright 2017 MapD Technologies, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultType.h"

#include <cstdint>

struct ColRangeInfo {
  QueryDescriptionType hash_type_;
  int64_t min;
  int64_t max;
  int64_t bucket;
  bool has_nulls;
  bool isEmpty() { return min == 0 && max == -1; }

  int64_t getBucketedCardinality() const;
};
