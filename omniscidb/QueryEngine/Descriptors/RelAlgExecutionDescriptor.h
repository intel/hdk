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
#include "ResultSetRegistry/ResultSetRegistry.h"
#include "Shared/TargetInfo.h"
#include "Shared/toString.h"

class ResultSet;

class ExecutionResult {
 public:
  ExecutionResult();

  ExecutionResult(hdk::ResultSetTableTokenPtr token,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const ExecutionResult& that);

  ExecutionResult(ExecutionResult&& that);

  ExecutionResult(const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
                  bool filter_push_down_enabled);

  ExecutionResult& operator=(const ExecutionResult& that);

  hdk::ResultSetTableTokenPtr getToken() const { return result_token_; }

  ResultSetPtr getRows() const {
    CHECK(result_token_);
    CHECK_EQ(result_token_->resultSetCount(), (size_t)1);
    return result_token_->resultSet(0);
  }

  bool empty() const { return !result_token_; }

  const std::string& tableName() const {
    CHECK(!empty());
    return result_token_->tableName();
  }

  ExecutionResult head(size_t n) {
    CHECK(result_token_);
    return {result_token_->head(n), targets_meta_};
  }

  ExecutionResult tail(size_t n) {
    CHECK(result_token_);
    return {result_token_->tail(n), targets_meta_};
  }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const;

  const bool isFilterPushDownEnabled() const { return filter_push_down_enabled_; }

  std::string toString() const {
    std::string res = ::typeName(this) + "(";
    if (result_token_) {
      res += ::toString(result_token_);
    } else {
      res += "empty";
    }
    res += ", " + ::toString(targets_meta_) + ")";
    return res;
  }

  enum RType { QueryResult, SimpleResult, Explaination, CalciteDdl };

  std::string getExplanation();
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
  hdk::ResultSetTableTokenPtr result_token_;
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
  RaExecutionDesc(const hdk::ir::Node* body) : body_(body) {}

  const ExecutionResult& getResult() const { return result_; }

  void setResult(const ExecutionResult& result);

  const hdk::ir::Node* getBody() const;

 private:
  const hdk::ir::Node* body_;
  ExecutionResult result_;
};
