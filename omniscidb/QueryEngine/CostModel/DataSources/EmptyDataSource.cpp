#include "EmptyDataSource.h"

namespace costmodel {

EmptyDataSource::EmptyDataSource()
    : DataSource(DataSourceConfig{"EmptyDataSource",
                                  {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU},
                                  {AnalyticalTemplate::GroupBy,
                                   AnalyticalTemplate::Join,
                                   AnalyticalTemplate::Reduce,
                                   AnalyticalTemplate::Scan}}) {}

Detail::DeviceMeasurements EmptyDataSource::getMeasurements(
    const std::vector<ExecutorDeviceType>& devices,
    const std::vector<AnalyticalTemplate>& templates) {
  return {};
}

}  // namespace costmodel
