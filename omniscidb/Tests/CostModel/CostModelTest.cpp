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
#include "QueryEngine/CostModel/ExtrapolationModels/LinearRegression.h"
#include "QueryEngine/CostModel/Measurements.h"

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

bool doubleEquals(double x, double y) {
  const double EPS = 0.0001;
  return std::abs(x - y) < EPS;
}

class LinearRegressionTest : public LinearRegression {
 public:
  LinearRegressionTest(std::vector<Detail::Measurement>& measurement)
      : LinearRegression(measurement) {}

  void equalsWs(arma::vec wExpected) {
    ASSERT_EQ(w.size(), wExpected.size());

    for (size_t i = 0; i < w.size(); i++) {
      ASSERT_TRUE(doubleEquals(w(i), wExpected(i)));
    }
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

TEST(ExtrapolationModelsTests, LinearRegressionTest1) {
  std::vector<Detail::Measurement> ms = {
      {.bytes = 10, .milliseconds = 10},
      {.bytes = 20, .milliseconds = 20},
      {.bytes = 30, .milliseconds = 30},
  };

  LinearRegressionTest lrt(ms);
  lrt.equalsWs({0.0, 1.0});

  ASSERT_EQ(lrt.getExtrapolatedData(40), 40);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
