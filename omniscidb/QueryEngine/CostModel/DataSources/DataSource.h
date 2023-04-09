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

#include "QueryEngine/CostModel/Measurements.h"

#include <unordered_map>
#include <vector>
#include <set>

namespace costmodel {

struct DataSourceConfig {
  std::string dataSourceName;
  std::set<ExecutorDeviceType> supportedDevices;
  std::set<AnalyticalTemplate> supportedTemplates;
};

class DataSource {
 public:
  virtual ~DataSource() = default;

  virtual Detail::DeviceMeasurements getMeasurements(
      const std::vector<ExecutorDeviceType>& devices,
      const std::vector<AnalyticalTemplate>& templates) = 0;

  bool isDeviceSupported(ExecutorDeviceType device);
  bool isTemplateSupported(AnalyticalTemplate templ);
  const std::string& getName();

 protected:
  DataSource(const DataSourceConfig& config);

 private:
  DataSourceConfig config_;
  std::string dataSourceName_;
};

class DataSourceException : public std::runtime_error {
 public:
  DataSourceException(const std::string& msg)
      : std::runtime_error("Data Source exception: " + msg){};
};

class UnsupportedAnalyticalTemplate : public DataSourceException {
 public:
  UnsupportedAnalyticalTemplate(AnalyticalTemplate templ)
      : DataSourceException("unsupported template: " + templateToString(templ)){};
};

class UnsupportedDevice : public DataSourceException {
 public:
  UnsupportedDevice(ExecutorDeviceType device)
      : DataSourceException("unsupported device: " + deviceToString(device)){};
};

}  // namespace costmodel
