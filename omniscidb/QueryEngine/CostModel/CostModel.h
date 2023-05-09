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

#include <memory>
#include <shared_mutex>

#include "DataSources/DataSource.h"
#include "ExtrapolationModels/ExtrapolationModelProvider.h"
#include "Measurements.h"

#include "Dispatchers/ExecutionPolicy.h"
#include "QueryEngine/CompilationOptions.h"

namespace costmodel {

struct CaibrationConfig {
  std::vector<ExecutorDeviceType> devices;
};

struct QueryInfo {
  std::vector<AnalyticalTemplate> templs;
  size_t bytes_size;
};

struct CostModelConfig {
  std::unique_ptr<DataSource> data_source;
};

using TemplatePredictions =
    std::unordered_map<AnalyticalTemplate, std::shared_ptr<ExtrapolationModel>>;
using DevicePredictions = std::unordered_map<ExecutorDeviceType, TemplatePredictions>;

class CostModel {
 public:
  CostModel(CostModelConfig config);
  virtual ~CostModel() = default;

  virtual void calibrate(const CaibrationConfig& conf);
  virtual std::unique_ptr<policy::ExecutionPolicy> predict(
      QueryInfo query_info) const = 0;

 protected:
  struct DeviceExtrapolations {
    ExecutorDeviceType device;
    std::vector<std::shared_ptr<ExtrapolationModel>> extrapolations;
  };

  std::vector<DeviceExtrapolations> getExtrapolations(
      const std::vector<ExecutorDeviceType>& devices,
      const std::vector<AnalyticalTemplate>& templs) const;

  CostModelConfig config_;

  ExtrapolationModelProvider extrapolation_provider_;

  DevicePredictions dp_;

  static const std::vector<AnalyticalTemplate> templates_;

  std::vector<ExecutorDeviceType> devices_ = {ExecutorDeviceType::CPU,
                                              ExecutorDeviceType::GPU};

  mutable std::shared_mutex latch_;
};

class CostModelException : public std::runtime_error {
 public:
  CostModelException(const std::string& msg)
      : std::runtime_error("CostModel exception: " + msg){};
};

}  // namespace costmodel
