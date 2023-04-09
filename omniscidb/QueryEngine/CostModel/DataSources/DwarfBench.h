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

#include <set>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "DataSource.h"

namespace costmodel {

class DwarfBenchCoverter;

// This is a temporary implementation while there is no
// library for interaction in dwarf bench
class DwarfBenchDataSource : public DataSource {
 public:
  DwarfBenchDataSource();

  virtual ~DwarfBenchDataSource();

  Detail::DeviceMeasurements getMeasurements(
      const std::vector<ExecutorDeviceType>& devices,
      const std::vector<AnalyticalTemplate>& templates) override;

 private:
  const size_t dwarfBenchIterations_ = 10;
  const std::vector<size_t> dwarfBenchInputSizes_ = {256, 512, 1024, 2048};

  std::vector<Detail::Measurement> measureTemplateOnDevice(ExecutorDeviceType device,
                                                           AnalyticalTemplate templ);

  struct PrivateImpl;
  PrivateImpl* pimpl_;
};

}  // namespace costmodel
