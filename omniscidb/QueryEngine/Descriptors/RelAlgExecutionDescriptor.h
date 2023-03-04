/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include "QueryEngine/JoinFilterPushDown.h"
#include "ResultSet/QueryMemoryDescriptor.h"
#include "ResultSet/ResultSet.h"
#include "Shared/TargetInfo.h"
#include "Shared/toString.h"

class ResultSet;

class ExecutionResult {
 public:
  ExecutionResult();

  ExecutionResult(const ResultSetPtr& rows,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(ResultSetPtr&& result, const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const TemporaryTable& results,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(TemporaryTable&& results,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const ExecutionResult& that);

  ExecutionResult(ExecutionResult&& that);

  ExecutionResult(const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
                  bool filter_push_down_enabled);

  ExecutionResult& operator=(const ExecutionResult& that);

  const ResultSetPtr& getRows() const {
    CHECK_EQ(results_.getFragCount(), 1);
    return results_[0];
  }

  bool empty() const { return results_.empty(); }

  const ResultSetPtr& getDataPtr() const {
    CHECK_EQ(results_.getFragCount(), 1);
    return results_[0];
  }

  const TemporaryTable& getTable() const { return results_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const;

  const bool isFilterPushDownEnabled() const { return filter_push_down_enabled_; }

  void setQueueTime(const int64_t queue_time_ms) {
    CHECK(!results_.empty());
    results_[0]->setQueueTime(queue_time_ms);
  }

  std::string toString() const {
    return ::typeName(this) + "(" + ::toString(results_) + ", " +
           ::toString(targets_meta_) + ")";
  }

  enum RType { QueryResult, SimpleResult, Explaination, CalciteDdl };

  std::string getExplanation();
  void updateResultSet(const std::string& query_ra, RType type, bool success = true);
  RType getResultType() const { return type_; }
  void setResultType(RType type) { type_ = type; }
  int64_t getExecutionTime() const { return execution_time_ms_; }
  void setExecutionTime(int64_t execution_time_ms) {
    execution_time_ms_ = execution_time_ms;
  }
  void addExecutionTime(int64_t execution_time_ms) {
    execution_time_ms_ += execution_time_ms;
  }

 private:
  TemporaryTable results_;
  std::vector<TargetMetaInfo> targets_meta_;
  // filters chosen to be pushed down
  std::vector<PushedDownFilterInfo> pushed_down_filter_info_;
  // whether or not it was allowed to look for filters to push down
  bool filter_push_down_enabled_;

  bool success_;
  uint64_t execution_time_ms_;
  RType type_;
};

namespace hdk::ir {
class Node;
}

class RaExecutionDesc {
 public:
  RaExecutionDesc(const hdk::ir::Node* body)
      : body_(body)
      , result_(std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                            ExecutorDeviceType::CPU,
                                            QueryMemoryDescriptor(),
                                            nullptr,
                                            nullptr,
                                            0,
                                            0),
                {}) {}

  const ExecutionResult& getResult() const { return result_; }

  void setResult(const ExecutionResult& result);

  const hdk::ir::Node* getBody() const;

 private:
  const hdk::ir::Node* body_;
  ExecutionResult result_;
};
