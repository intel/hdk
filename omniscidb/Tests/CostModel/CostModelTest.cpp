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

#include <gtest/gtest.h>

#include "QueryEngine/CostModel/DataSources/DataSource.h"
#include "QueryEngine/CostModel/ExtrapolationModels/LinearExtrapolation.h"

using namespace costmodel;

class DataSourceTest : public DataSource {
 public:
  DataSourceTest()
      : DataSource(DataSourceConfig{"DataSourceTest",
                                    {ExecutorDeviceType::CPU},
                                    {AnalyticalTemplate::GroupBy}}) {}

  Detail::DeviceMeasurements getMeasurements(
      const std::vector<ExecutorDeviceType>& devices,
      const std::vector<AnalyticalTemplate>& templates) override {
    return {};
  }
};

TEST(DataSourceTests, SupportCheckTest) {
  DataSourceTest ds;
  ASSERT_EQ(ds.getName(), "DataSourceTest");
  ASSERT_TRUE(ds.isDeviceSupported(ExecutorDeviceType::CPU));
  ASSERT_TRUE(ds.isTemplateSupported(AnalyticalTemplate::GroupBy));
  ASSERT_FALSE(ds.isDeviceSupported(ExecutorDeviceType::GPU));
  ASSERT_FALSE(ds.isTemplateSupported(AnalyticalTemplate::Join));
}

TEST(ExtrapolationModelsTests, LinearExtrapolationTest1) {
  LinearExtrapolation le{{
      {10, 100},
      {20, 200},
      {30, 300},
  }};

  ASSERT_EQ(le.getExtrapolatedData(15), (size_t)150);
  ASSERT_EQ(le.getExtrapolatedData(25), (size_t)250);
  ASSERT_EQ(le.getExtrapolatedData(35), (size_t)350);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
