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

#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/RelAlgDagBuilder.h"

ExecutionResult::ExecutionResult()
    : filter_push_down_enabled_(false)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult::ExecutionResult(hdk::ResultSetTableTokenPtr token,
                                 const std::vector<hdk::ir::TargetMetaInfo>& targets_meta)
    : result_token_(std::move(token))
    , targets_meta_(targets_meta)
    , filter_push_down_enabled_(false)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult::ExecutionResult(const ExecutionResult& that)
    : targets_meta_(that.targets_meta_)
    , pushed_down_filter_info_(that.pushed_down_filter_info_)
    , filter_push_down_enabled_(that.filter_push_down_enabled_)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {
  if (!pushed_down_filter_info_.empty() ||
      (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
    return;
  }
  result_token_ = that.result_token_;
}

ExecutionResult::ExecutionResult(ExecutionResult&& that)
    : targets_meta_(std::move(that.targets_meta_))
    , pushed_down_filter_info_(std::move(that.pushed_down_filter_info_))
    , filter_push_down_enabled_(std::move(that.filter_push_down_enabled_))
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {
  if (!pushed_down_filter_info_.empty() ||
      (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
    return;
  }
  result_token_ = std::move(that.result_token_);
}

ExecutionResult::ExecutionResult(
    const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
    bool filter_push_down_enabled)
    : pushed_down_filter_info_(pushed_down_filter_info)
    , filter_push_down_enabled_(filter_push_down_enabled)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult& ExecutionResult::operator=(const ExecutionResult& that) {
  if (!that.pushed_down_filter_info_.empty() ||
      (that.filter_push_down_enabled_ && that.pushed_down_filter_info_.empty())) {
    pushed_down_filter_info_ = that.pushed_down_filter_info_;
    filter_push_down_enabled_ = that.filter_push_down_enabled_;
    return *this;
  }
  result_token_ = that.result_token_;
  targets_meta_ = that.targets_meta_;
  success_ = that.success_;
  execution_time_ms_ = that.execution_time_ms_;
  type_ = that.type_;
  return *this;
}

const std::vector<PushedDownFilterInfo>& ExecutionResult::getPushedDownFilterInfo()
    const {
  return pushed_down_filter_info_;
}

std::string ExecutionResult::getExplanation() {
  if (!empty()) {
    return getRows()->getExplanation();
  }
  return {};
}

void RaExecutionDesc::setResult(const ExecutionResult& result) {
  result_ = result;
  body_->setContextData(this);
}

const hdk::ir::Node* RaExecutionDesc::getBody() const {
  return body_;
}
