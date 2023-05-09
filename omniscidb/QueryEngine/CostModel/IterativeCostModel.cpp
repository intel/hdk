/*
    Copyright (c) 2023 Intel Corporation
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

#include "IterativeCostModel.h"
#include "Dispatchers/ProportionBasedExecutionPolicy.h"

#ifdef HAVE_DWARF_BENCH
#include "DataSources/DwarfBench.h"
#endif

#include <cmath>

namespace costmodel {

#ifdef HAVE_DWARF_BENCH
IterativeCostModel::IterativeCostModel()
    : CostModel({std::make_unique<DwarfBenchDataSource>()}) {}
#else
IterativeCostModel::IterativeCostModel()
    : CostModel({std::make_unique<EmptyDataSource>()}) {}
#endif

std::unique_ptr<policy::ExecutionPolicy> IterativeCostModel::predict(
    QueryInfo query_info) const {
  std::shared_lock<std::shared_mutex> l(latch_);

  unsigned cpu_prop = 1, gpu_prop = 0;
  size_t opt_step =
      std::ceil(static_cast<float>(query_info.bytes_size) / optimization_iterations_);
  size_t runtime_prediction = std::numeric_limits<size_t>::max();

  std::vector<DeviceExtrapolations> devices_extrapolations = getExtrapolations(
      {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}, query_info.templs);

  for (size_t cur_size = 0; cur_size < query_info.bytes_size; cur_size += opt_step) {
    size_t cpu_size = cur_size;
    size_t gpu_size = query_info.bytes_size - cur_size;

    size_t cpu_prediction = 0;
    size_t gpu_prediction = 0;

    for (DeviceExtrapolations dev_extrapolations : devices_extrapolations) {
      for (auto extrapolation : dev_extrapolations.extrapolations) {
        if (dev_extrapolations.device == ExecutorDeviceType::CPU) {
          cpu_prediction += extrapolation->getExtrapolatedData(cpu_size);
        } else if (dev_extrapolations.device == ExecutorDeviceType::GPU) {
          gpu_prediction += extrapolation->getExtrapolatedData(gpu_size);
        }
      }
    }

    size_t cur_prediction = std::max(gpu_prediction, cpu_prediction);

    if (cur_prediction < runtime_prediction) {
      runtime_prediction = cur_prediction;

      cpu_prop = cpu_size;
      gpu_prop = gpu_size;
    }
  }

  std::map<ExecutorDeviceType, unsigned> proportion;

  proportion[ExecutorDeviceType::GPU] = gpu_prop;
  proportion[ExecutorDeviceType::CPU] = cpu_prop;

  return std::make_unique<policy::ProportionBasedExecutionPolicy>(std::move(proportion));
}
}  // namespace costmodel
