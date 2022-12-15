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

#include "QueryEngine/CostModel/DataSources/DwarfBench.h"
#include "QueryEngine/CostModel/Measurements.h"

using namespace costmodel;

#define GENERATE_DWARF_BENCH_INTEGRATION_TEST(DEVICE, TEMPLATE)                                                                 \
    TEST(DwarfBenchIntegrationTests, GetMeasurements_##DEVICE##TEMPLATE) {                                                      \
        DwarfBenchDataSource dbds;                                                                                              \
        if (!dbds.isDeviceSupported(ExecutorDeviceType::DEVICE) || !dbds.isTemplateSupported(AnalyticalTemplate::TEMPLATE)) {   \
            GTEST_SKIP();                                                                                                       \
        }                                                                                                                       \
                                                                                                                                \
        std::vector<ExecutorDeviceType> devices = { ExecutorDeviceType::DEVICE };                                               \
        std::vector<AnalyticalTemplate> templates = { AnalyticalTemplate::TEMPLATE };                                           \  
                                                                                                                                \  
        Detail::DeviceMeasurements ms;                                                                                          \
        ASSERT_NO_FATAL_FAILURE(ms = dbds.getMeasurements(devices, templates));                                                 \
                                                                                                                                \
        ASSERT_GT(ms[ExecutorDeviceType::DEVICE][AnalyticalTemplate::TEMPLATE].size(), 0);                                      \
    }                                                                                                                           \


TEST(DwarfBenchIntegrationTests, GetMeasurements) {
    DwarfBenchDataSource dbds;
    std::vector<ExecutorDeviceType> devices = { ExecutorDeviceType::CPU, ExecutorDeviceType::GPU };
    std::vector<AnalyticalTemplate> templates = { AnalyticalTemplate::GroupBy, AnalyticalTemplate::Join, AnalyticalTemplate::Reduce, AnalyticalTemplate::Scan };

    Detail::DeviceMeasurements ms;
    ASSERT_NO_FATAL_FAILURE(ms = dbds.getMeasurements(devices, templates));
}

GENERATE_DWARF_BENCH_INTEGRATION_TEST(CPU, GroupBy);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(CPU, Join);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(CPU, Scan);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(CPU, Reduce);

GENERATE_DWARF_BENCH_INTEGRATION_TEST(GPU, GroupBy);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(GPU, Join);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(GPU, Scan);
GENERATE_DWARF_BENCH_INTEGRATION_TEST(GPU, Reduce);

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

