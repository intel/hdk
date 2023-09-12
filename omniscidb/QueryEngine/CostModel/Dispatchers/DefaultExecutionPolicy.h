/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include "ExecutionPolicy.h"

namespace policy {
class FragmentIDAssignmentExecutionPolicy : public ExecutionPolicy {
 public:
  FragmentIDAssignmentExecutionPolicy(
      ExecutorDeviceType dt,
      const std::map<ExecutorDeviceType, ExecutorDispatchMode>& devices_dispatch_modes)
      : ExecutionPolicy(devices_dispatch_modes), dt_(dt){};
  SchedulingAssignment scheduleSingleFragment(const FragmentInfo&,
                                              size_t frag_id,
                                              size_t frag_num) const override;
  std::set<ExecutorDeviceType> devices() const override;
  std::string name() const override { return "ExecutionPolicy::FragmentIDAssignment"; };

 private:
  const ExecutorDeviceType dt_;
};
}  // namespace policy
