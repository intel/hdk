#pragma once

#include "QueryEngine/CostModel/Measurements.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

class AnalyticalTemplatesExtractor {
public:
    std::vector<costmodel::AnalyticalTemplate> extractTemplates(const RelAlgExecutionUnit &ra_exe_unit) const;

private:
    using Priority = uint8_t;

    template <costmodel::AnalyticalTemplate templ>
    bool templSuits(const RelAlgExecutionUnit &ra_exe_unit) const;

    bool templSuits(const RelAlgExecutionUnit &ra_exe_unit, costmodel::AnalyticalTemplate templ) const;
    
    // We want to greedy extract templates
    std::map<Priority, costmodel::AnalyticalTemplate> priotities_ = {
        {0, costmodel::AnalyticalTemplate::GroupBy},
        {0, costmodel::AnalyticalTemplate::Join},
        {0, costmodel::AnalyticalTemplate::Sort},
        {1, costmodel::AnalyticalTemplate::Scan},
        {1, costmodel::AnalyticalTemplate::Reduce},
    };
};
