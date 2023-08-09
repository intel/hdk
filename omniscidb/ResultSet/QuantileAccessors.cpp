/*
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QuantileAccessors.h"

#include "Shared/TypePunning.h"

namespace {

template <typename ResultType>
void finalizeQuantile(hdk::quantile::Quantile* quantile,
                      const hdk::ir::Type* arg_type,
                      double q,
                      hdk::ir::Interpolation interpolation) {
  if (arg_type->isFloatingPoint()) {
    switch (arg_type->size()) {
      case 4:
        return quantile->finalize<float, ResultType>(q, interpolation);
      case 8:
        return quantile->finalize<double, ResultType>(q, interpolation);
    }
  } else {
    switch (arg_type->canonicalSize()) {
      case 1:
        return quantile->finalize<int8_t, ResultType>(q, interpolation);
      case 2:
        return quantile->finalize<int16_t, ResultType>(q, interpolation);
      case 4:
        return quantile->finalize<int32_t, ResultType>(q, interpolation);
      case 8:
        return quantile->finalize<int64_t, ResultType>(q, interpolation);
    }
  }
  CHECK(false);
}

template <typename ResultType>
ResultType getQuantile(hdk::quantile::Quantile* quantile,
                       const hdk::ir::Type* arg_type,
                       double q,
                       hdk::ir::Interpolation interpolation) {
  if (arg_type->isFloatingPoint()) {
    switch (arg_type->size()) {
      case 4:
        return quantile->quantile<float, ResultType>(q, interpolation);
      case 8:
        return quantile->quantile<double, ResultType>(q, interpolation);
    }
  } else {
    switch (arg_type->canonicalSize()) {
      case 1:
        return quantile->quantile<int8_t, ResultType>(q, interpolation);
      case 2:
        return quantile->quantile<int16_t, ResultType>(q, interpolation);
      case 4:
        return quantile->quantile<int32_t, ResultType>(q, interpolation);
      case 8:
        return quantile->quantile<int64_t, ResultType>(q, interpolation);
    }
  }
  CHECK(false);
  return 0;
}

}  // namespace

void finalizeQuantile(hdk::quantile::Quantile* quantile, const TargetInfo& target_info) {
  if (target_info.type->isFloatingPoint()) {
    switch (target_info.type->size()) {
      case 4:
        return finalizeQuantile<float>(quantile,
                                       target_info.agg_arg_type,
                                       target_info.quantile_param,
                                       target_info.interpolation);
      case 8:
        return finalizeQuantile<double>(quantile,
                                        target_info.agg_arg_type,
                                        target_info.quantile_param,
                                        target_info.interpolation);
    }
  } else {
    switch (target_info.type->canonicalSize()) {
      case 1:
        return finalizeQuantile<int8_t>(quantile,
                                        target_info.agg_arg_type,
                                        target_info.quantile_param,
                                        target_info.interpolation);
      case 2:
        return finalizeQuantile<int16_t>(quantile,
                                         target_info.agg_arg_type,
                                         target_info.quantile_param,
                                         target_info.interpolation);
      case 4:
        return finalizeQuantile<int32_t>(quantile,
                                         target_info.agg_arg_type,
                                         target_info.quantile_param,
                                         target_info.interpolation);
      case 8:
        return finalizeQuantile<int64_t>(quantile,
                                         target_info.agg_arg_type,
                                         target_info.quantile_param,
                                         target_info.interpolation);
    }
  }
  CHECK(false);
}

TargetValue getQuantile(hdk::quantile::Quantile* quantile,
                        const TargetInfo& target_info) {
  if (target_info.type->isFloatingPoint()) {
    switch (target_info.type->size()) {
      case 4:
        return getQuantile<float>(quantile,
                                  target_info.agg_arg_type,
                                  target_info.quantile_param,
                                  target_info.interpolation);
      case 8:
        return getQuantile<double>(quantile,
                                   target_info.agg_arg_type,
                                   target_info.quantile_param,
                                   target_info.interpolation);
    }
  } else {
    switch (target_info.type->canonicalSize()) {
      case 1:
        return static_cast<int64_t>(getQuantile<int8_t>(quantile,
                                                        target_info.agg_arg_type,
                                                        target_info.quantile_param,
                                                        target_info.interpolation));
      case 2:
        return static_cast<int64_t>(getQuantile<int16_t>(quantile,
                                                         target_info.agg_arg_type,
                                                         target_info.quantile_param,
                                                         target_info.interpolation));
      case 4:
        return static_cast<int64_t>(getQuantile<int32_t>(quantile,
                                                         target_info.agg_arg_type,
                                                         target_info.quantile_param,
                                                         target_info.interpolation));
      case 8:
        return getQuantile<int64_t>(quantile,
                                    target_info.agg_arg_type,
                                    target_info.quantile_param,
                                    target_info.interpolation);
    }
  }
  CHECK(false) << "Unexpected quantile type: " << target_info.type->toString();
  return (int64_t)0;
}

InternalTargetValue getQuantileInternal(hdk::quantile::Quantile* quantile,
                                        const TargetInfo& target_info) {
  if (target_info.type->isFloatingPoint()) {
    switch (target_info.type->size()) {
      case 4: {
        float fval = getQuantile<float>(quantile,
                                        target_info.agg_arg_type,
                                        target_info.quantile_param,
                                        target_info.interpolation);
        return InternalTargetValue(
            static_cast<int64_t>(*reinterpret_cast<int32_t*>(may_alias_ptr(&fval))));
      }
      case 8: {
        double dval = getQuantile<double>(quantile,
                                          target_info.agg_arg_type,
                                          target_info.quantile_param,
                                          target_info.interpolation);
        return InternalTargetValue(*reinterpret_cast<int64_t*>(may_alias_ptr(&dval)));
      }
    }
  } else {
    switch (target_info.type->canonicalSize()) {
      case 1:
        return InternalTargetValue(
            static_cast<int64_t>(getQuantile<int8_t>(quantile,
                                                     target_info.agg_arg_type,
                                                     target_info.quantile_param,
                                                     target_info.interpolation)));
      case 2:
        return InternalTargetValue(
            static_cast<int64_t>(getQuantile<int16_t>(quantile,
                                                      target_info.agg_arg_type,
                                                      target_info.quantile_param,
                                                      target_info.interpolation)));
      case 4:
        return InternalTargetValue(
            static_cast<int64_t>(getQuantile<int32_t>(quantile,
                                                      target_info.agg_arg_type,
                                                      target_info.quantile_param,
                                                      target_info.interpolation)));
      case 8:
        return InternalTargetValue(getQuantile<int64_t>(quantile,
                                                        target_info.agg_arg_type,
                                                        target_info.quantile_param,
                                                        target_info.interpolation));
    }
  }
  CHECK(false) << "Unexpected quantile type: " << target_info.type->toString();
  return InternalTargetValue();
}
