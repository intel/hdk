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

#include "DwarfBench.h"

#include "Logger/Logger.h"

#include <bench.hpp>

#include <fstream>
#include <iostream>

namespace costmodel {

struct DwarfBenchDataSource::PrivateImpl {
  DwarfBench::Dwarf convertToDwarf(AnalyticalTemplate templ);
  DwarfBench::DeviceType convertDeviceType(ExecutorDeviceType device);
  std::vector<Detail::Measurement> convertMeasurement(
      const std::vector<DwarfBench::Measurement> &measurements);

  DwarfBench::DwarfBench db;
};

DwarfBench::Dwarf DwarfBenchDataSource::PrivateImpl::convertToDwarf(
    AnalyticalTemplate templ) {
  switch (templ) {
    case AnalyticalTemplate::GroupBy:
      return DwarfBench::Dwarf::GroupBy;
    case AnalyticalTemplate::Scan:
      return DwarfBench::Dwarf::Scan;
    case AnalyticalTemplate::Join:
      return DwarfBench::Dwarf::Join;
    case AnalyticalTemplate::Reduce:
      throw UnsupportedAnalyticalTemplate(templ);
    case AnalyticalTemplate::Sort:
      return DwarfBench::Dwarf::Sort;
  }
}

DwarfBench::DeviceType DwarfBenchDataSource::PrivateImpl::convertDeviceType(
    ExecutorDeviceType device) {
  switch (device) {
    case ExecutorDeviceType::CPU:
      return DwarfBench::DeviceType::CPU;
    case ExecutorDeviceType::GPU:
      return DwarfBench::DeviceType::GPU;
  }
}

std::vector<Detail::Measurement> DwarfBenchDataSource::PrivateImpl::convertMeasurement(
    const std::vector<DwarfBench::Measurement> &measurements) {
  std::vector<Detail::Measurement> ms;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(ms),
                 [](DwarfBench::Measurement m) {
                   return Detail::Measurement{.bytes = m.dataSize,
                                              .milliseconds = m.microseconds / 1000};
                 });
  return ms;
}

DwarfBenchDataSource::DwarfBenchDataSource()
    : DataSource(DataSourceConfig{
          .dataSourceName = "DwarfBench",
          .supportedDevices = {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU},
          .supportedTemplates = {AnalyticalTemplate::GroupBy,
                                 AnalyticalTemplate::Join,
                                 AnalyticalTemplate::Scan,
                                 AnalyticalTemplate::Sort}})
    , pimpl_(new PrivateImpl()) {}

DwarfBenchDataSource::~DwarfBenchDataSource() = default;

Detail::DeviceMeasurements DwarfBenchDataSource::getMeasurements(
    const std::vector<ExecutorDeviceType>& devices,
    const std::vector<AnalyticalTemplate>& templates) {
  Detail::DeviceMeasurements dm;
  for (AnalyticalTemplate templ : templates) {
    CHECK(isTemplateSupported(templ));
    for (ExecutorDeviceType device : devices) {
      CHECK(isDeviceSupported(device));

      dm[device][templ] = measureTemplateOnDevice(device, templ);
    }
  }

  return dm;
}

std::vector<Detail::Measurement> DwarfBenchDataSource::measureTemplateOnDevice(
    ExecutorDeviceType device,
    AnalyticalTemplate templ) {
  std::vector<Detail::Measurement> ms;
  for (size_t inputSize : dwarfBenchInputSizes_) {
    DwarfBench::RunConfig rc = {
        .device = pimpl_->convertDeviceType(device),
        .inputSize = inputSize,
        .iterations = dwarfBenchIterations_,
        .dwarf = pimpl_->convertToDwarf(templ),
    };

    std::vector<Detail::Measurement> inputSizeMeasurements =
        pimpl_->convertMeasurement(pimpl_->db.makeMeasurements(rc));

    ms.insert(ms.end(), inputSizeMeasurements.begin(), inputSizeMeasurements.end());
  }

  return ms;
}

}  // namespace costmodel
