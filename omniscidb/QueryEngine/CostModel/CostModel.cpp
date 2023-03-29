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

#include "CostModel.h"
#include "ExtrapolationModels/LinearExtrapolation.h"

namespace costmodel {

CostModel::CostModel(std::unique_ptr<DataSource> _dataSource)
    : dataSource_(std::move(_dataSource)) {
  for (AnalyticalTemplate templ : templates_) {
    if (!dataSource_->isTemplateSupported(templ))
      throw CostModelException("template " + templateToString(templ) +
                               " not supported in " + dataSource_->getName() +
                               " data source");
  }

  for (ExecutorDeviceType device : devices_) {
    if (!dataSource_->isDeviceSupported(device))
      throw CostModelException("device " + deviceToString(device) + " not supported in " +
                               dataSource_->getName() + " data source");
  }
}

void CostModel::calibrate(const CaibrationConfig& conf) {
  std::lock_guard<std::mutex> g{latch_};

  Detail::DeviceMeasurements dm;

  try {
    dm = dataSource_->getMeasurements(conf.devices, templates_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Cost model calibration failure: " << e.what();
    return;
  }

  for (const auto& dmEntry : dm) {
    ExecutorDeviceType device = dmEntry.first;

    for (auto& templateMeasurement : dmEntry.second) {
      AnalyticalTemplate templ = templateMeasurement.first;
      dp_[device][templ] =
          std::make_unique<LinearExtrapolation>(std::move(templateMeasurement.second));
    }
  }
}

const std::vector<AnalyticalTemplate> CostModel::templates_ = {GroupBy,
                                                              Join,
                                                              Scan,
                                                              Reduce};

}  // namespace costmodel
