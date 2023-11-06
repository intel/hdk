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

#include "AnalyticalTemplatesExtractor.h"
#include <limits>
#include <stdexcept>
#include <vector>
#include "QueryEngine/CostModel/Measurements.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

std::vector<costmodel::AnalyticalTemplate> AnalyticalTemplatesExtractor::extractTemplates(
    const RelAlgExecutionUnit& ra_exe_unit) const {
  Priority current_priority = std::numeric_limits<Priority>::max();
  std::vector<costmodel::AnalyticalTemplate> templates;

  for (auto [priority, templ] : priotities_) {
    if (priority > current_priority) {
      return templates;
    }

    if (templSuits(ra_exe_unit, templ)) {
      current_priority = priority;
      templates.push_back(templ);
    }
  }

  return templates;
}

bool AnalyticalTemplatesExtractor::templSuits(const RelAlgExecutionUnit& ra_exe_unit,
                                              costmodel::AnalyticalTemplate templ) const {
  using costmodel::AnalyticalTemplate;

  switch (templ) {
    case AnalyticalTemplate::GroupBy:
      return !ra_exe_unit.groupby_exprs.empty();
    case AnalyticalTemplate::Join:
      return !ra_exe_unit.join_quals.empty();
    case AnalyticalTemplate::Sort:
      return !ra_exe_unit.sort_info.order_entries.empty();
    case AnalyticalTemplate::Reduce:
      // TODO(bagrorg): currently we don't even
      // use Reduce for cost model
      return false;
    case AnalyticalTemplate::Scan:
      return !ra_exe_unit.quals.empty() || !ra_exe_unit.simple_quals.empty();
    default:
      throw std::runtime_error("Unknown template for check was given");
  }
}
