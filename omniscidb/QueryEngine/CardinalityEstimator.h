/*
 * Copyright 2017 MapD Technologies, Inc.
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

/**
 * @file    CardinalityEstimator.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Estimators to be used when precise cardinality isn't useful.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_CARDINALITYESTIMATOR_H
#define QUERYENGINE_CARDINALITYESTIMATOR_H

#include "RelAlgExecutionUnit.h"

#include "Analyzer/Analyzer.h"
#include "IR/CardinalityEstimator.h"
#include "IR/Expr.h"
#include "Logger/Logger.h"

class CardinalityEstimationRequired : public std::runtime_error {
 public:
  CardinalityEstimationRequired(const int64_t range)
      : std::runtime_error("CardinalityEstimationRequired"), range_(range) {}

  int64_t range() const { return range_; }

 private:
  const int64_t range_;
};

class RequestPartitionedAggregation : public std::runtime_error {
 public:
  RequestPartitionedAggregation(size_t entry_size, size_t estimated_buffer_entries)
      : std::runtime_error("RequestPartitionedAggregation")
      , entry_size_(entry_size)
      , estimated_buffer_entries_(estimated_buffer_entries) {}

  size_t entrySize() const { return entry_size_; }
  size_t estimatedBufferEntries() const { return estimated_buffer_entries_; }
  size_t estimatedBufferSize() const { return entry_size_ * estimated_buffer_entries_; }

 private:
  size_t entry_size_;
  size_t estimated_buffer_entries_;
};

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit,
                                              SchemaProvider* schema_provider,
                                              const Config& config,
                                              const int64_t range);

RelAlgExecutionUnit create_count_all_execution_unit(
    const RelAlgExecutionUnit& ra_exe_unit,
    hdk::ir::ExprPtr replacement_target);

ResultSetPtr reduce_estimator_results(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device);

#endif  // QUERYENGINE_CARDINALITYESTIMATOR_H
