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

#include "DataProvider/TableFragmentsInfo.h"
#include "QueryEngine/CompilationOptions.h"

#include <ostream>

namespace policy {
using TableFragments = std::vector<FragmentInfo>;

struct SchedulingAssignment {
  ExecutorDeviceType dt;
  int device_id;
};

class ExecutionPolicy {
  std::map<ExecutorDeviceType, ExecutorDispatchMode> devices_dispatch_modes_;

 public:
  ExecutionPolicy(
      const std::map<ExecutorDeviceType, ExecutorDispatchMode>& devices_dispatch_modes)
      : devices_dispatch_modes_(devices_dispatch_modes){};
  virtual SchedulingAssignment scheduleSingleFragment(const FragmentInfo&,
                                                      size_t frag_id,
                                                      size_t frag_num) const = 0;

  virtual std::set<ExecutorDeviceType> devices() const {
    std::set<ExecutorDeviceType> res;
    for (const auto& dt_mode : devices_dispatch_modes_) {
      res.insert(dt_mode.first);
    }
    return res;
  }

  virtual bool hasDevice(const ExecutorDeviceType dt) const {
    return (devices_dispatch_modes_.count(dt) != 0);
  }

  virtual ExecutorDispatchMode getExecutionMode(const ExecutorDeviceType dt) const {
    CHECK(hasDevice(dt));
    return devices_dispatch_modes_.at(dt);
  }

  virtual std::map<ExecutorDeviceType, ExecutorDispatchMode> getExecutionModes() const {
    return devices_dispatch_modes_;
  }
  virtual std::string name() const = 0;

  virtual ~ExecutionPolicy() = default;
};

inline std::ostream& operator<<(std::ostream& os, const ExecutionPolicy& policy) {
  os << policy.name() << "\n";
  os << "Dispatching modes: \n";
  for (const auto& device_disp_mode : policy.getExecutionModes()) {
    os << device_disp_mode.first << " - " << device_disp_mode.second << "\n";
  }
  return os;
}

}  // namespace policy
