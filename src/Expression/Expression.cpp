#include "Expression/Expression.h"

namespace hdk {

std::string Column::toString() const {
  return "Column: " + type_->toString();
}

std::string Aggregate::toString() const {
  return "Aggregate: " + aggTypeToString(agg_type_) + " (" + type_->toString() +
         ") over " + target_expression_->toString();
}

std::string Aggregate::aggTypeToString(const AggType type) {
  switch (type) {
    case AggType::kAVG:
      return "AVG";
    case AggType::kMIN:
      return "MIN";
    case AggType::kMAX:
      return "MAX";
    case AggType::kSUM:
      return "SUM";
    case AggType::kCOUNT:
      return "COUNT";
  }
  return "";
}

}  // namespace hdk
