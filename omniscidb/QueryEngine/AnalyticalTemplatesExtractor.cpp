#include "AnalyticalTemplatesExtractor.h"
#include <limits>
#include <stdexcept>
#include <vector>
#include "QueryEngine/CostModel/Measurements.h"
#include "QueryEngine/RelAlgExecutionUnit.h"


std::vector<costmodel::AnalyticalTemplate> AnalyticalTemplatesExtractor::extractTemplates(const RelAlgExecutionUnit &ra_exe_unit) const {
    Priority current_priority = std::numeric_limits<Priority>::max();
    std::vector<costmodel::AnalyticalTemplate> templates;

    for (auto [priority, templ]: priotities_) {
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

template <>
bool AnalyticalTemplatesExtractor::templSuits<costmodel::AnalyticalTemplate::GroupBy>(const RelAlgExecutionUnit &ra_exe_unit) const {
    return !ra_exe_unit.groupby_exprs.empty();
}

template <>
bool AnalyticalTemplatesExtractor::templSuits<costmodel::AnalyticalTemplate::Join>(const RelAlgExecutionUnit &ra_exe_unit) const {
    return !ra_exe_unit.join_quals.empty();
}

template <>
bool AnalyticalTemplatesExtractor::templSuits<costmodel::AnalyticalTemplate::Scan>(const RelAlgExecutionUnit &ra_exe_unit) const {
    return !ra_exe_unit.quals.empty() || !ra_exe_unit.simple_quals.empty();
}

template <>
bool AnalyticalTemplatesExtractor::templSuits<costmodel::AnalyticalTemplate::Sort>(const RelAlgExecutionUnit &ra_exe_unit) const {
    return !ra_exe_unit.sort_info.order_entries.empty();
}

template <>
bool AnalyticalTemplatesExtractor::templSuits<costmodel::AnalyticalTemplate::Reduce>(const RelAlgExecutionUnit &ra_exe_unit) const {
    // TODO(bagrorg): currently we don't even
    // use Reduce for cost model
    return false;
}

bool AnalyticalTemplatesExtractor::templSuits(const RelAlgExecutionUnit &ra_exe_unit, costmodel::AnalyticalTemplate templ) const {
    using costmodel::AnalyticalTemplate;

    switch (templ) {
    case AnalyticalTemplate::GroupBy:
        return templSuits<AnalyticalTemplate::GroupBy>(ra_exe_unit);
    case AnalyticalTemplate::Join:
        return templSuits<AnalyticalTemplate::Join>(ra_exe_unit);
    case AnalyticalTemplate::Sort:
        return templSuits<AnalyticalTemplate::Sort>(ra_exe_unit);
    case AnalyticalTemplate::Reduce:
        return templSuits<AnalyticalTemplate::Reduce>(ra_exe_unit);
    case AnalyticalTemplate::Scan:
        return templSuits<AnalyticalTemplate::Scan>(ra_exe_unit);
    default:
        throw std::runtime_error("Unknown template for check was given");
    }
}
