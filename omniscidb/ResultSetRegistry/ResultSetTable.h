/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ResultSet/ResultSet.h"

#include <vector>

namespace hdk {

class ResultSetTable {
 public:
  ResultSetTable() {}
  ResultSetTable(ResultSetPtr result) { results_.emplace_back(std::move(result)); }
  ResultSetTable(std::vector<ResultSetPtr> results) : results_(std::move(results)) {}

  ResultSetTable(const ResultSetTable& other) = default;
  ResultSetTable(ResultSetTable&& other) = default;

  ResultSetTable& operator=(const ResultSetTable& other) = default;
  ResultSetTable& operator=(ResultSetTable&& other) = default;

  bool empty() const { return results_.empty(); }
  size_t size() const { return results_.size(); }

  const std::vector<ResultSetPtr>& results() const { return results_; }
  const ResultSetPtr& result(size_t idx) const {
    CHECK_LT(idx, results_.size());
    return results_[idx];
  }

  ResultSetPtr& operator[](size_t idx) { return results_[idx]; }
  const ResultSetPtr& operator[](size_t idx) const { return results_[idx]; }

  void setQueueTime(const int64_t queue_time_ms) {
    if (!empty()) {
      results_.front()->setQueueTime(queue_time_ms);
    }
  }

  void setKernelQueueTime(const int64_t kernel_queue_time) {
    if (!empty()) {
      results_.front()->setKernelQueueTime(kernel_queue_time);
    }
  }

  void addCompilationQueueTime(const int64_t compilation_queue_time) {
    if (!empty()) {
      results_.front()->addCompilationQueueTime(compilation_queue_time);
    }
  }

  void setValidationOnlyRes() {
    if (!empty()) {
      results_.front()->setValidationOnlyRes();
    }
  }

 private:
  std::vector<ResultSetPtr> results_;
};

}  // namespace hdk
